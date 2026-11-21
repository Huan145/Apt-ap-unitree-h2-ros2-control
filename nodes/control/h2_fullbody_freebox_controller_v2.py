#!/usr/bin/env python3
"""
h2_fullbody_freebox_controller.py

Deployment controller for the FREE-BOX policy (Isaac-Velocity-Flat-H2-FreeBox-v0,
h2_locomotion_freebox). The policy drives ALL 31 joints, arms included (it grips the box
itself). Based on h2_fullbody_box_controller.py (v1); differences, marked "FREEBOX:":
  1. POLICY_PATH -> models/h2_walking_freebox/
  2. DEFAULT_JOINT_POS = HOLD_JOINT_POS (captured hold, head pitch 0) -> joint_pos_rel obs.
  3. ACTION_OFFSET = HOLD_ACTION_OFFSET: legs/waist/head = hold pose, ARMS = the sequencer's
     hold targets (squeeze + load compensation). Command = ACTION_OFFSET + action * scale.
     (Not default + action * scale: in this task offset != default for the arms.)
  4. No /arm_targets override (v2/v3): the policy owns the arms.
  5. Head: policy-driven (scale 0.5); /head_pitch_cmd overrides head pitch by setting its joint
     target directly (last_action keeps the policy's own output, as in training).
Action scale 0.25 arms / 0.5 rest, obs layout (105), decimation 4, dt 0.005: unchanged.
Verify 2./3. against params/env.yaml of the run (scene.robot.init_state.joint_pos and
actions.joint_pos.offset).

HANDOVER: action 0 == the sequencer's hold command, and _previous_action starts at 0, so the
first policy steps continue the hold. Start this node BEFORE the handover; the sequencer
must stop publishing joint commands when this node takes over. Straighten the head first
(/head_pitch_cmd 0.0): the policy was trained with head pitch 0.

HEAD PITCH OVERRIDE (unchanged from v1): std_msgs/Float64 on `head_pitch_cmd`, slews at
head_rate, release with NaN or by stopping for head_cmd_timeout s.

SETTLE PHASE (optional, --settle S): for the first S seconds (sim time) after the node gets its
first /joint_states, it does NOT run the policy. It ramps the joint targets from the MEASURED
angles to the trained start command over --settle_ramp seconds and then holds it (with 'offset',
the arms go straight to their hold targets - the sequencer already commanded exactly those):
  --settle_targets offset  (default) = HOLD_ACTION_OFFSET = exactly what the policy commands at
                            action 0 (arms: squeeze + load compensation) -> the policy takes over
                            with no jump in any target
  --settle_targets default = HOLD_JOINT_POS (measured hold angles; arms WITHOUT squeeze/load
                            compensation -> the box may sag)
Then the policy starts with a fresh state (body-speed estimate 0, last action 0), as in training.

STANDBY / SWITCH (--standby): the node runs from the start but does not command anything until
h2_control_mux.py reports "freebox" on /control_mode_active (latched). At that moment it starts fresh (body-speed
estimate 0, last action 0 = the hold command) and commands from its next physics step. "balance"
(or anything else) puts it back into standby. Use with h2_control_mux.py, which forwards only the
active controller's commands to /joint_command, so the two never command at the same time:
  python3 H2_balance_controller.py --cmd-topic /joint_command_balance
  python3 h2_fullbody_freebox_controller.py --standby --ros-args -r joint_command:=/joint_command_freebox
  python3 h2_control_mux.py
  ros2 topic pub --once /control_mode std_msgs/msg/String "{data: freebox}"     # switch

Run (start BEFORE pressing Play in Isaac Sim):
  python3 h2_fullbody_freebox_controller.py [--policy policy.pt] [--settle 2.0 --settle_ramp 1.0]
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
from std_msgs.msg import Float64, String
from message_filters import Subscriber, TimeSynchronizer


# ============================================================================
# CONFIG
# ============================================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# FREEBOX: own model folder so the welded-box / plain policies are never loaded by mistake
_MODELS_DIR = os.path.join(_REPO_ROOT, "models/h2_walking_freebox")

_parser = argparse.ArgumentParser()
_parser.add_argument("--policy", default="policy.pt",
                     help="policy filename inside models/ (default: policy.pt)")
_parser.add_argument("--settle", type=float, default=0.0,
                     help="hold the trained start command for S s (sim time) before the policy starts (0 = off)")
_parser.add_argument("--settle_ramp", type=float, default=1.0,
                     help="s to ramp the targets from the measured angles to the start command")
_parser.add_argument("--settle_targets", choices=("offset", "default"), default="offset",
                     help="offset = what the policy commands at action 0 (recommended); default = hold angles")
_parser.add_argument("--standby", action="store_true",
                     help="wait for /control_mode 'freebox' before commanding (use with h2_control_mux.py)")
_parser.add_argument("--mode_topic", default="/control_mode_active",
                     help="std_msgs/String from h2_control_mux.py (latched): 'freebox' = take control")
_args, _ = _parser.parse_known_args()
POLICY_PATH = os.path.join(_MODELS_DIR, _args.policy)
# BOX: per-joint scale, built below from ARM_ACTION_SCALE / OTHER_ACTION_SCALE.
#      Must match actions.joint_pos.scale in the box run's env.yaml.
ARM_ACTION_SCALE = 0.25
OTHER_ACTION_SCALE = 0.5
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

# FREEBOX: = HOLD_JOINT_POS (h2_locomotion_freebox/assets/unitree_h2.py) = init_state.joint_pos.
# Used ONLY for the joint_pos_rel observation.
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.22659, "right_hip_pitch_joint": -0.18589,
    "left_hip_roll_joint": 0.00768, "right_hip_roll_joint": -0.00850,
    "left_hip_yaw_joint": 0.01183, "right_hip_yaw_joint": 0.01534,
    "left_knee_joint": 0.42029, "right_knee_joint": 0.41341,
    "left_ankle_pitch_joint": -0.20664, "right_ankle_pitch_joint": -0.23429,
    "left_ankle_roll_joint": 0.01045, "right_ankle_roll_joint": 0.00404,
    "waist_yaw_joint": 0.00290, "waist_roll_joint": -0.00067, "waist_pitch_joint": 0.01061,
    "head_pitch_joint": 0.0, "head_yaw_joint": -0.00057,
    "left_shoulder_pitch_joint": 0.02407, "right_shoulder_pitch_joint": 0.00099,
    "left_shoulder_roll_joint": 0.25733, "right_shoulder_roll_joint": -0.24917,
    "left_shoulder_yaw_joint": 0.00173, "right_shoulder_yaw_joint": -0.01659,
    "left_elbow_joint": -0.03338, "right_elbow_joint": -0.05281,
    "left_wrist_roll_joint": -0.16820, "right_wrist_roll_joint": 0.18680,
    "left_wrist_pitch_joint": 0.06132, "right_wrist_pitch_joint": 0.03923,
    "left_wrist_yaw_joint": 0.17171, "right_wrist_yaw_joint": -0.18862,
}

# FREEBOX: = HOLD_ARM_TARGETS (the sequencer's commanded hold). Arms' action offset.
HOLD_ARM_TARGETS = {
    "left_shoulder_pitch_joint": -0.19508, "right_shoulder_pitch_joint": -0.19298,
    "left_shoulder_roll_joint": 0.24868, "right_shoulder_roll_joint": -0.24487,
    "left_shoulder_yaw_joint": -0.04848, "right_shoulder_yaw_joint": 0.04891,
    "left_elbow_joint": -0.24409, "right_elbow_joint": -0.24371,
    "left_wrist_roll_joint": -0.15776, "right_wrist_roll_joint": 0.17496,
    "left_wrist_pitch_joint": -0.03035, "right_wrist_pitch_joint": -0.03356,
    "left_wrist_yaw_joint": 0.15381, "right_wrist_yaw_joint": -0.15897,
}

# FREEBOX: = HOLD_ACTION_OFFSET = actions.joint_pos.offset. Command = offset + action * scale.
ACTION_OFFSET = {**DEFAULT_JOINT_POS, **HOLD_ARM_TARGETS}

# BOX: per-joint action scale in JOINT_NAMES order
_ARM_KEYS = ("_shoulder_", "_elbow_", "_wrist_")
ACTION_SCALE = np.array(
    [ARM_ACTION_SCALE if any(k in j for k in _ARM_KEYS) else OTHER_ACTION_SCALE for j in JOINT_NAMES],
    dtype=np.float64,
)
# FREEBOX: head action scale as in training (actions.joint_pos.scale head_.*_joint).
HEAD_ACTION_SCALE = 0.5
ACTION_SCALE[[i for i, j in enumerate(JOINT_NAMES) if j.startswith("head_")]] = HEAD_ACTION_SCALE

HEAD_PITCH_JOINT = "head_pitch_joint"

OBS_DIM = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS  # 105


# ============================================================================
# Node
# ============================================================================

class H2FullbodyFreeBoxController(Node):
    """Fullbody controller for the H2 carrying a FREE box (policy grips it), with head-pitch override."""

    def __init__(self):
        super().__init__('h2_fullbody_freebox_controller')

        self.declare_parameter('policy_path', POLICY_PATH)
        self.set_parameters(
            [rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )

        # head override params
        self.declare_parameter('head_pitch_topic', 'head_pitch_cmd')
        self.declare_parameter('head_pitch_min', -0.5)   # rad — CHECK H2 URDF limits
        self.declare_parameter('head_pitch_max', 0.7)    # rad — BOX: raised for the ~0.59 pickup camera tilt; CHECK H2 URDF limits
        self.declare_parameter('head_rate', 0.5)         # rad/s slew limit
        self.declare_parameter('head_cmd_timeout', 1.0)  # s sim time -> release

        self._logger = self.get_logger()

        self.joint_names = JOINT_NAMES
        self.default_pos = np.array([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], dtype=np.float64)
        self.action_offset = np.array([ACTION_OFFSET[j] for j in JOINT_NAMES], dtype=np.float64)  # FREEBOX

        # --- state (initialized before any subscription is created) ---
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._action_scale = ACTION_SCALE.copy()   # BOX: per-joint array
        self._previous_action = np.zeros(NUM_JOINTS)
        self._policy_counter = 0
        self._decimation = DECIMATION
        # BOX: None until the first tick. If the node starts while the sim is already playing,
        # the first dt would otherwise be the whole elapsed sim time -> huge integrated base
        # velocity in the observation -> violent policy output.
        self._last_tick_time = None
        self._max_dt = 0.05   # s; a larger gap (pause, lag) is not integrated
        self._lin_vel_b = np.zeros(3)
        self._dt = 0.0
        self.action = np.zeros(NUM_JOINTS)
        self._head_cmd = None   # FREEBOX: head target from the override (None = offset)
        # SETTLE: hold the start command before the policy takes over
        self._settle_goal = self.action_offset if _args.settle_targets == "offset" else self.default_pos
        self._settling = _args.settle > 0.0
        self._settle_t0 = None
        self._settle_from = None
        self._q_meas = None      # last measured joint angles (JOINT_NAMES order)
        self._active = not _args.standby   # STANDBY: commanding only after /control_mode 'freebox'

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
        self._head_meas_valid = False   # BOX: True once /joint_states has given a real head angle
        self._head_need_init = False    # BOX: override engaged, start angle still to be taken

        # --- ROS interfaces ---
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        self._cmd_vel_subscription = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_callback, qos_profile=10)
        latched = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                                       durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                                       history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self._mode_subscription = self.create_subscription(
            String, _args.mode_topic, self._mode_callback, qos_profile=latched)

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
            f"Initializing H2FullbodyFreeBoxController: policy={self.policy_path} | obs_dim={OBS_DIM}, "
            f"action_dim={NUM_JOINTS}, decimation={self._decimation}, "
            f"action scale arms={ARM_ACTION_SCALE} other={OTHER_ACTION_SCALE} | "
            f"head override on '{g('head_pitch_topic')}' "
            f"range [{self._head_min:+.2f},{self._head_max:+.2f}] rad, "
            f"rate {self._head_rate} rad/s, timeout {self._head_timeout}s"
        )

    # ------------------------------------------------------------------ inputs

    def _cmd_vel_callback(self, msg):
        self._cmd_vel = msg

    def _mode_callback(self, msg: String):
        """STANDBY: 'freebox' -> take control with a fresh state; anything else -> standby."""
        want = msg.data.strip().lower() == "freebox"
        if want and not self._active:
            self._lin_vel_b = np.zeros(3)                  # fresh start, as in training
            self._previous_action = np.zeros(NUM_JOINTS)   # last action 0 = the hold command
            self.action = np.zeros(NUM_JOINTS)
            self._policy_counter = 0
            self._settling = _args.settle > 0.0
            self._settle_t0 = None
            self._active = True
            err = np.abs(self._q_meas - self.default_pos) if self._q_meas is not None else np.zeros(1)
            self._logger.info(f"[MODE] freebox policy TAKES CONTROL. Largest joint offset from the trained "
                              f"start pose: {np.degrees(err.max()):.2f} deg"
                              + (f" ({JOINT_NAMES[int(err.argmax())]})" if self._q_meas is not None else ""))
        elif not want and self._active:
            self._active = False
            self._logger.info(f"[MODE] '{msg.data}' -> freebox policy in STANDBY (not commanding)")

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
            self._head_need_init = True        # BOX: start angle taken from the next measurement
            self._logger.info(f"[HEAD] override engaged toward {self._head_target:+.3f} rad")

    # ------------------------------------------------------------------ head

    def _update_head(self):
        """Advance the head override by one policy step.
        Returns the head angle to command, or None if the policy owns the head."""
        if not self._head_override:
            return None
        if self._head_need_init:                  # BOX: start from where the head IS (measured)
            if not self._head_meas_valid:
                return None
            self._head_cur = self._head_meas
            self._head_need_init = False
            self._logger.info(f"[HEAD] starting from measured {self._head_cur:+.3f} rad")

        default = float(self.action_offset[self._head_idx])   # FREEBOX: = hold head pitch (0.0)
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
        self._last_tick_time = None
        self._lin_vel_b = np.zeros(3)
        self._previous_action = np.zeros(NUM_JOINTS)
        self.action = np.zeros(NUM_JOINTS)
        self._policy_counter = 0
        self._head_override = False
        self._head_target = None
        self._head_last_msg = None
        self._head_cmd = None
        self._settling = _args.settle > 0.0      # SETTLE again after a sim restart
        self._settle_t0 = None

    def _settle_tick(self, joint_state: JointState, imu: Imu, now: float):
        """SETTLE: ramp measured -> start command, hold it, then hand over to the policy."""
        self._compute_observation(joint_state, imu)            # keeps head measurement / q up to date
        if self._settle_t0 is None:
            self._settle_t0, self._settle_from = now, self._q_meas.copy()
            if _args.settle_targets == "offset":
                # arms: the sequencer was already commanding exactly these hold targets (squeeze +
                # load compensation). Ramping them from the MEASURED (sagged) angles would loosen the
                # grip for a moment, so the arms start at the target right away.
                arms = [i for i, j in enumerate(JOINT_NAMES) if any(k in j for k in _ARM_KEYS)]
                self._settle_from[arms] = self._settle_goal[arms]
            self._logger.info(f"[SETTLE] ramping to the '{_args.settle_targets}' start command over "
                              f"{_args.settle_ramp:.1f} s, policy starts after {_args.settle:.1f} s")
        t = now - self._settle_t0
        a = min(1.0, t / max(_args.settle_ramp, 1e-6))
        target = self._settle_from + a * (self._settle_goal - self._settle_from)
        if self._policy_counter % self._decimation == 0:
            self._head_cmd = self._update_head()
        self._policy_counter += 1
        if self._head_cmd is not None:
            target[self._head_idx] = self._head_cmd
        self._publish(target)
        if t >= _args.settle:
            self._settling = False
            self._lin_vel_b = np.zeros(3)                       # fresh start, as in training
            self._previous_action = np.zeros(NUM_JOINTS)
            self.action = np.zeros(NUM_JOINTS)
            self._policy_counter = 0
            err = np.abs(self._q_meas - self.default_pos)
            self._logger.info(f"[SETTLE] done -> policy in control. Largest joint offset from the trained "
                              f"start pose: {np.degrees(err.max()):.2f} deg ({JOINT_NAMES[int(err.argmax())]})")

    def _publish(self, target: np.ndarray):
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names
        self._joint_command.position = target.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _tick(self, joint_state: JointState, imu: Imu):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_tick_time is None:          # BOX: first tick -> start integrating from here
            self._last_tick_time = now
        if now < self._last_tick_time:
            self._logger.error(f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.')
            self._reset_state()
            self._last_tick_time = now

        self._dt = (now - self._last_tick_time)
        if self._dt > self._max_dt:               # BOX: never integrate across a long gap
            self._dt = 0.0
        self._last_tick_time = now

        if not self._active:                           # STANDBY: keep measuring, command nothing
            self._compute_observation(joint_state, imu)
            return

        if self._settling:                             # SETTLE phase: no policy yet
            self._settle_tick(joint_state, imu, now)
            return

        self.forward(joint_state, imu)

        action_pos = self.action_offset + self.action * self._action_scale   # FREEBOX: offset, not default
        if self._head_cmd is not None:                 # FREEBOX: head override written as a target directly
            action_pos[self._head_idx] = self._head_cmd
        self._publish(action_pos)

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

        self._q_meas = current_joint_pos
        self._head_meas = float(current_joint_pos[self._head_idx])
        self._head_meas_valid = True

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
            # FREEBOX: head scale is 0, so the override can't go through the action vector; it is
            # applied to the joint target in _tick. last_action keeps the policy's own output,
            # exactly as in training.
            self._head_cmd = self._update_head()
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
    node = H2FullbodyFreeBoxController()
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