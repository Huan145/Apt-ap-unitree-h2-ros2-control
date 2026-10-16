#!/usr/bin/env python3
"""
h2_fullbody_controller.py

H2 humanoid locomotion policy controller, adapted from NVIDIA's reference
h1_fullbody_controller.py (IsaacSim-ros_workspaces repo) to H2's specific
joint configuration, confirmed against env.yaml from the h2_flat training run.

Key architectural differences from the earlier h2_rl_policy_node.py:
  - Message-driven control (TimeSynchronizer on joint_states + imu), not a
    wall-clock ROS2 timer. Control cadence is tied to the sim's own physics
    step rate via a decimation counter, matching Isaac Lab's decimation=4
    exactly, regardless of wall-clock/rendering load.
  - use_sim_time=True -- all timing uses Isaac Sim's /clock, not wall time.
  - Uses an IMU sensor on the pelvis for angular velocity + orientation
    (gravity projection), and integrates linear acceleration for linear
    velocity -- NOT an odometry node.

HEAD PITCH OVERRIDE (new):
  - Subscribes to std_msgs/Float64 on `head_pitch_cmd` (radians, joint angle).
  - While commands arrive, head_pitch_joint is overridden: the head slews
    toward the target at `head_rate` rad/s, clamped to
    [head_pitch_min, head_pitch_max]. All other joints stay policy-driven.
  - The override is written back INTO the policy's action vector, so the
    policy's last_action observation stays consistent with what was actually
    commanded.
  - Release: send NaN, or stop publishing for `head_cmd_timeout` s. The head
    then slews back to its default angle and control returns to the policy.
  - Override starts from the MEASURED head angle, so there is no jump.

Also new: on a backwards time jump (Isaac Stop -> Play while this node
keeps running), integrator / last action / head override are reset.

Confirmed against env.yaml (h2_flat, 2026-05-25_09-21-25 run):
  - sim.dt = 0.005, decimation = 4  -> policy runs at 50Hz, physics at 200Hz
  - actions.joint_pos: scale=0.5, offset=0.0, use_default_offset=True, joints=".*"
  - observations.policy: base_lin_vel, base_ang_vel, projected_gravity,
    velocity_commands, joint_pos_rel, joint_vel_rel, last_action
    -- ALL have scale=null, clip=null (no hidden scaling anywhere)
  - init_state.joint_pos defaults (see DEFAULT_JOINT_POS)

Run (start BEFORE pressing Play in Isaac Sim):
  python3 h2_fullbody_controller.py [--policy policy.pt]
  # optional ROS params:
  #   --ros-args -p head_pitch_min:=-0.5 -p head_pitch_max:=0.5 \
  #              -p head_rate:=0.5 -p head_cmd_timeout:=1.0

Quick head test (robot standing):
  ros2 topic pub -r 20 /head_pitch_cmd std_msgs/msg/Float64 "{data: 0.3}"
  -> check which way the head moves; Ctrl-C -> head returns after timeout.
"""

import argparse
import io
import math
import os
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64
from message_filters import Subscriber, TimeSynchronizer


# ============================================================================
# CONFIG
# ============================================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_REPO_ROOT, "models/h2_walking")

_parser = argparse.ArgumentParser()
_parser.add_argument("--policy", default="policy.pt",
                     help="policy filename inside models/ (default: policy.pt)")
_args, _ = _parser.parse_known_args()
POLICY_PATH = os.path.join(_MODELS_DIR, _args.policy)
ACTION_SCALE = 0.5   # confirmed: actions.joint_pos.scale in env.yaml
DECIMATION = 4       # confirmed: env.yaml decimation (policy runs every 4th tick)
SIM_DT = 0.005       # confirmed: env.yaml sim.dt
POLICY_DT = SIM_DT * DECIMATION   # 0.02 s
NUM_JOINTS = 31

JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "head_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2, "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0, "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.42, "right_knee_joint": 0.42, "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.28, "right_shoulder_pitch_joint": 0.28,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0, "head_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16, "right_shoulder_roll_joint": -0.16,
    "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.52, "right_elbow_joint": 0.52,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}

HEAD_PITCH_JOINT = "head_pitch_joint"

OBS_DIM = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS  # 105


# ============================================================================
# Node
# ============================================================================

class H2FullbodyController(Node):
    """Fullbody controller for the H2 humanoid, with external head-pitch override."""

    def __init__(self):
        super().__init__('h2_fullbody_controller')

        self.declare_parameter('policy_path', POLICY_PATH)
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )

        # head override params
        self.declare_parameter('head_pitch_topic', 'head_pitch_cmd')
        self.declare_parameter('head_pitch_min', -0.5)   # rad — CHECK H2 URDF limits
        self.declare_parameter('head_pitch_max', 0.5)    # rad — CHECK H2 URDF limits
        self.declare_parameter('head_rate', 0.5)         # rad/s slew limit
        self.declare_parameter('head_cmd_timeout', 1.0)  # s sim time -> release

        self._logger = self.get_logger()

        self.joint_names = JOINT_NAMES
        self.default_pos = np.array([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float64)

        # --- state (initialized before any subscription is created) ---
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._action_scale = ACTION_SCALE
        self._previous_action = np.zeros(NUM_JOINTS)
        self._policy_counter = 0
        self._decimation = DECIMATION
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)
        self._dt = 0.0
        self.action = np.zeros(NUM_JOINTS)

        # --- head override state ---
        g = lambda n: self.get_parameter(n).value
        self._head_idx = JOINT_NAMES.index(HEAD_PITCH_JOINT)
        self._head_min = float(g('head_pitch_min'))
        self._head_max = float(g('head_pitch_max'))
        self._head_rate = float(g('head_rate'))
        self._head_timeout = float(g('head_cmd_timeout'))
        self._head_override = False     # True while we own head_pitch_joint
        self._head_target = None        # latest requested angle (None = release)
        self._head_cur = 0.0            # angle currently commanded by override
        self._head_last_msg = None      # rclpy Time of last valid command
        self._head_meas = float(self.default_pos[self._head_idx])  # measured
        self._head_releasing_logged = False

        # --- ROS interfaces ---
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        self._cmd_vel_subscription = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_callback, qos_profile=10)

        self._head_subscription = self.create_subscription(
            Float64, g('head_pitch_topic'), self._head_callback, qos_profile=10)

        self._joint_publisher = self.create_publisher(
            JointState, 'joint_command', qos_profile=sim_qos_profile)

        self._imu_sub_filter = Subscriber(self, Imu, 'imu', qos_profile=sim_qos_profile)
        self._joint_states_sub_filter = Subscriber(self, JointState, 'joint_states', qos_profile=sim_qos_profile)

        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        self._logger.info(
            f"Initializing H2FullbodyController: obs_dim={OBS_DIM}, "
            f"action_dim={NUM_JOINTS}, decimation={self._decimation} | "
            f"head override on '{g('head_pitch_topic')}' "
            f"range [{self._head_min:+.2f},{self._head_max:+.2f}] rad, "
            f"rate {self._head_rate} rad/s, timeout {self._head_timeout}s"
        )

    # ------------------------------------------------------------------ inputs

    def _cmd_vel_callback(self, msg):
        self._cmd_vel = msg

    def _head_callback(self, msg: Float64):
        v = float(msg.data)
        if not math.isfinite(v):
            # explicit release request
            if self._head_override and self._head_target is not None:
                self._logger.info("[HEAD] release requested -> returning to default")
            self._head_target = None
            return
        self._head_target = float(np.clip(v, self._head_min, self._head_max))
        self._head_last_msg = self.get_clock().now()
        self._head_releasing_logged = False
        if not self._head_override:
            self._head_override = True
            self._head_cur = self._head_meas   # start from where the head IS
            self._logger.info(
                f"[HEAD] override engaged: from {self._head_cur:+.3f} "
                f"toward {self._head_target:+.3f} rad")

    # ------------------------------------------------------------------ head

    def _update_head(self):
        """Advance the head override by one policy step.
        Returns the head angle to command, or None if the policy owns the head."""
        if not self._head_override:
            return None

        default = float(self.default_pos[self._head_idx])
        stale = self._head_target is None or self._head_last_msg is None
        if not stale:
            age = (self.get_clock().now() - self._head_last_msg).nanoseconds * 1e-9
            stale = age > self._head_timeout

        if stale and not self._head_releasing_logged:
            self._logger.info("[HEAD] no command (timeout/release) -> slewing to default")
            self._head_releasing_logged = True

        goal = default if stale else self._head_target
        step = self._head_rate * POLICY_DT
        self._head_cur += float(np.clip(goal - self._head_cur, -step, step))

        if stale and abs(self._head_cur - default) < 1e-3:
            self._head_override = False
            self._head_target = None
            self._logger.info("[HEAD] override released -> policy controls head")
            return None
        return self._head_cur

    # ------------------------------------------------------------------ main loop

    def _reset_state(self):
        self._lin_vel_b = np.zeros(3)
        self._previous_action = np.zeros(NUM_JOINTS)
        self.action = np.zeros(NUM_JOINTS)
        self._policy_counter = 0
        self._head_override = False
        self._head_target = None
        self._head_last_msg = None

    def _tick(self, joint_state: JointState, imu: Imu):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.')
            self._reset_state()
            self._last_tick_time = now

        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        self.forward(joint_state, imu)

        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        R_BI = self.quat_to_rot_matrix(quat_array).T

        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ])
        # Naive integration -- drifts over long runs, matches the reference
        # H1 implementation.
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b

        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
        ])

        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(OBS_DIM)
        obs[0:3] = self._lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b

        cmd_vel = [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z]
        obs[9:12] = np.array(cmd_vel)

        current_joint_pos = np.zeros(NUM_JOINTS)
        current_joint_vel = np.zeros(NUM_JOINTS)
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        self._head_meas = float(current_joint_pos[self._head_idx])

        obs[12:12 + NUM_JOINTS] = current_joint_pos - self.default_pos
        obs[12 + NUM_JOINTS:12 + 2 * NUM_JOINTS] = current_joint_vel
        obs[12 + 2 * NUM_JOINTS:12 + 3 * NUM_JOINTS] = self._previous_action

        return obs

    def _compute_action(self, obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_t).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        obs = self._compute_observation(joint_state, imu)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            head = self._update_head()
            if head is not None:
                # Express the override in the policy's action space so that
                # default_pos + action*scale == head, and last_action obs
                # reflects what was really commanded.
                self.action[self._head_idx] = (
                    (head - self.default_pos[self._head_idx]) / self._action_scale)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    # ------------------------------------------------------------------ utils

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.policy = torch.jit.load(buffer)

    def _get_stamp_prefix(self) -> str:
        import time
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'


def main(args=None):
    rclpy.init(args=args)
    node = H2FullbodyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# ============================================================================
# SETUP NOTES
#
# 1. IMU sensor on the pelvis + ActionGraph "ROS_Imu" (On Physics Step ->
#    Isaac Read IMU (Read Gravity unchecked) -> ROS2 Publish IMU "imu",
#    with Isaac Read Simulation Time, Reset on Stop checked).
# 2. Existing /joint_states publisher and /joint_command -> Articulation
#    Controller graphs unchanged.
# 3. Start this node BEFORE pressing Play.
# 4. Head override: the camera TF (pelvis <- Camera) must come from a TF
#    tree that includes head_pitch_joint (Isaac TF publisher or
#    robot_state_publisher on /joint_states), NOT a static transform —
#    otherwise the navigator's box estimate will be wrong once the head tilts.
# ============================================================================