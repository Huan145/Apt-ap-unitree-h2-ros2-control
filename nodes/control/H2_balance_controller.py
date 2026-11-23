#!/usr/bin/env python3
"""
H2 Pickup Balance Controller - the single owner of /joint_command.

Every control step (50 Hz):
  obs    = built exactly like the h2_balance training env (layout read from
           policy_io.json, so a changed obs config fails loudly instead of silently)
  action = trained policy (TorchScript from Isaac Lab's play.py export)
  cmd    = policy joints: offset + scale * action     (legs, waist, head - 17)
         + arm joints:    latest /arm_targets          (from H2_pickup_sequencer - 14)
  -> ONE JointState with all 31 joints on /joint_command

Timing (--clock stamp, default): /joint_states and /imu are paired by identical
stamp (TimeSynchronizer, one pair per physics step, as in h2_fullbody_controller)
and the policy steps every step_dt (20 ms) of STAMP time - sim time in Isaac
Sim - so it runs at its trained rate at any real-time factor. Start this node
BEFORE pressing Play, so the policy balances from the first physics step.

Startup: the robot should already stand in the TRAINING start pose
(policy_io.json default_joint_pos - check with check_start_pose.py); the
policy starts on the first control step and logs a warning for any joint more
than START_TOL rad off. Arms hold the training default pose until the
sequencer's first /arm_targets message. Optional: --ramp/--hold ease all
joints into the default pose first (off by default); arm targets that arrive
before "Policy active" are ignored.

HEAD PITCH OVERRIDE (same behaviour as h2_fullbody_controller.py):
  - std_msgs/Float64 on /head_pitch_cmd (radians, joint angle). While commands
    arrive, head_pitch_joint follows them at --head-rate rad/s, clamped to
    [--head-min, --head-max], starting from the MEASURED head angle (no jump).
  - Written back into the action vector, so the policy's last-action observation
    matches what was really commanded.
  - Release: send NaN or stop publishing for --head-timeout s -> the head slews
    back to its training default and the policy takes the head again.
  Use: lower the head to the box-locomotion default (0.0) BEFORE handing over to
  h2_fullbody_box_controller.py, e.g.
    ros2 topic pub -r 20 /head_pitch_cmd std_msgs/msg/Float64 "{data: 0.0}"

ARM HOLD (switching with h2_control_mux.py): arm targets come from /arm_targets while it is
fresh (a message in the last ARM_FRESH_S). Without it, the arms hold the arm values last APPLIED
on --applied-topic (/joint_command = the mux output = what the previous controller commanded),
so switching to this controller never moves the arms (box held by the box controller stays
held). While this controller is the active one, that is its own last command -> the arms stay put.

Subscribes: /joint_states (JointState), /imu (sensor_msgs/Imu, pelvis-mounted),
            /arm_targets (JointState, 14 arm joints),
            /load_expected (std_msgs/Bool, from the sequencer; False until received),
            /head_pitch_cmd (std_msgs/Float64, optional head override)
            --applied-topic (/joint_command, JointState): arm values to hold without /arm_targets
Publishes:  /joint_command (JointState, 31 joints, radians)

Usage:
  conda activate ik_env && source /opt/ros/jazzy/setup.bash
  python3 H2_balance_controller.py      # loads <repo>/models/h2_balance_box_pickup/{policy.pt, policy_io.json}
  python3 H2_balance_controller.py --policy <other>/policy.pt --io <other>/policy_io.json   # override
"""

import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
import torch
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64

# Same as h2_fullbody_controller.py (proven with this sim): reliable, keep every message
SIM_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_ALL)

STALE_S = 0.1        # --clock wall: stop commanding if joint_states/imu are older than this (wall time)
IMU_MAX_LAG_S = 0.02  # --clock stamp: skip a step if IMU and joint_states stamps are further apart (sim time)
START_TOL = 0.1       # rad; warn if a joint starts further than this from the training start pose
ARM_FRESH_S = 0.5     # wall s; /arm_targets older than this -> hold the last applied arm values
HEAD_PITCH_JOINT = "head_pitch_joint"

# <repo>/nodes/control/this_file.py -> <repo>/models/h2_balance_box_pickup/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(_REPO_ROOT, "models", "h2_balance_box_pickup")


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class BalanceController(Node):
    def __init__(self, policy_path, io_path, cmd_topic, arm_topic, imu_topic, applied_topic, ramp_s=0.0, hold_s=0.0,
                 clock="stamp", head_topic="/head_pitch_cmd", head_min=-0.5, head_max=0.7,
                 head_rate=0.5, head_timeout=1.0):
        super().__init__("h2_balance_controller")
        self.ramp_s, self.hold_s, self.clock = ramp_s, hold_s, clock
        self.startup_q = self.startup_t0 = None
        self.policy_active = False
        with open(io_path) as f:
            io = json.load(f)

        self.joint_names = io["joint_names"]                      # Isaac Lab order (obs order)
        self.default_pos = np.array(io["default_joint_pos"])
        self.act_names = io["action_joint_names"]
        self.act_scale = np.array(io["action_scale"])
        self.act_offset = np.array(io["action_offset"])
        self.arm_names = io["arm_joint_names"]
        self.step_dt = float(io["step_dt"])
        self.hz = 1.0 / self.step_dt
        self.obs_terms = [(n, int(np.prod(d))) for n, d in io["policy_obs_terms"]]

        builders = {
            "base_ang_vel": lambda: self.ang_vel,
            "projected_gravity": lambda: self.gravity_b,
            "joint_pos": lambda: self.q - self.default_pos,
            "joint_vel": lambda: self.dq,
            "arm_target": lambda: self.arm_target,
            "actions": lambda: self.last_action,
            "load_expected": lambda: [self.load_expected],
        }
        unknown = [n for n, _ in self.obs_terms if n not in builders]
        if unknown:
            raise RuntimeError(f"policy obs terms {unknown} have no deployment builder - update the controller")
        self.builders = [(n, dim, builders[n]) for n, dim in self.obs_terms]

        self.policy = torch.jit.load(policy_path, map_location="cpu").eval()
        self.obs_dim = sum(d for _, d in self.obs_terms)

        n = len(self.joint_names)
        self.q, self.dq = np.zeros(n), np.zeros(n)
        self.ang_vel, self.gravity_b = np.zeros(3), np.array([0.0, 0.0, -1.0])
        self.last_action = np.zeros(len(self.act_names))
        self.arm_target = None
        self.t_arm = None                            # wall time of the last /arm_targets message
        self.applied_arm = None                      # arm values last seen on applied_topic
        self.arm_holding = False                     # True while holding applied_arm (no fresh /arm_targets)
        self.load_expected = 0.0
        self.t_joints = self.t_imu = None            # wall time of last message
        self.stamp_imu = None                        # sim/robot time (header stamp) of last IMU message
        self.next_step = None                        # --clock stamp: stamp time of the next policy step
        self.last_stamp, self.same_stamp_count = None, 0
        self.arm_idx = [self.joint_names.index(j) for j in self.arm_names]

        # --- head override ---
        self.head_min, self.head_max = float(head_min), float(head_max)
        self.head_rate, self.head_timeout = float(head_rate), float(head_timeout)
        self.head_act_i = self.act_names.index(HEAD_PITCH_JOINT) if HEAD_PITCH_JOINT in self.act_names else None
        self.head_q_i = self.joint_names.index(HEAD_PITCH_JOINT)
        self.head_override = False      # True while we own head_pitch_joint
        self.head_need_init = False     # engaged, start angle not taken yet
        self.head_target = None         # latest requested angle (None = release)
        self.head_cur = 0.0             # angle currently commanded by the override
        self.head_last_t = None         # control-clock time of the last valid command
        self.head_released_logged = False
        self.now = None                 # control-clock time of the latest control step

        if clock == "stamp":
            # joint_states + IMU paired by identical stamp (one pair per physics step), like h2_fullbody_controller
            self._sync_subs = [Subscriber(self, JointState, "/joint_states", qos_profile=SIM_QOS),
                               Subscriber(self, Imu, imu_topic, qos_profile=SIM_QOS)]
            self._sync = TimeSynchronizer(self._sync_subs, 10)
            self._sync.registerCallback(self.cb_synced)
        else:
            self.create_subscription(JointState, "/joint_states", self.cb_joints, SIM_QOS)
            self.create_subscription(Imu, imu_topic, self.cb_imu, SIM_QOS)
            self.create_timer(1.0 / self.hz, self.tick)
        self.create_subscription(JointState, arm_topic, self.cb_arm, 10)
        self.create_subscription(JointState, applied_topic, self.cb_applied, SIM_QOS)
        self.create_subscription(Bool, "/load_expected", self.cb_flag, 10)
        if self.head_act_i is None:
            self.get_logger().warn(f"{HEAD_PITCH_JOINT} is not a policy joint - head override disabled")
        else:
            self.create_subscription(Float64, head_topic, self.cb_head, 10)
        self.pub = self.create_publisher(JointState, cmd_topic, SIM_QOS)

        self.get_logger().info(
            f"Policy {policy_path} | obs {self.obs_dim} = " + " + ".join(f"{n}({d})" for n, d in self.obs_terms)
            + f" | {len(self.act_names)} policy joints + {len(self.arm_names)} arm joints @ {self.hz:.0f} Hz "
            + ("of /joint_states stamp time (sim time in Isaac Sim)" if clock == "stamp" else "wall clock")
            + f" -> {cmd_topic}"
            + (f" | head override on {head_topic} [{self.head_min:+.2f},{self.head_max:+.2f}] rad, "
               f"{self.head_rate} rad/s, timeout {self.head_timeout}s" if self.head_act_i is not None else ""))

    # ── Inputs ──────────────────────────────────────────────────────────────

    def cb_joints(self, msg):
        idx = {name: i for i, name in enumerate(msg.name)}
        missing = [j for j in self.joint_names if j not in idx]
        if missing:
            self.get_logger().error(f"/joint_states missing {missing}", throttle_duration_sec=2.0)
            return False
        order = [idx[j] for j in self.joint_names]
        self.q = np.asarray(msg.position)[order]
        self.dq = np.asarray(msg.velocity)[order] if len(msg.velocity) == len(msg.name) else np.zeros(len(order))
        self.t_joints = time.monotonic()
        return True

    def cb_synced(self, js, imu):
        """--clock stamp: one joint_states + IMU pair with the same stamp (one physics step)."""
        self.cb_imu(imu)
        if self.cb_joints(js):
            self.on_joint_stamp(stamp_s(js))

    def cb_imu(self, msg):
        o, w = msg.orientation, msg.angular_velocity
        self.gravity_b = Rotation.from_quat([o.x, o.y, o.z, o.w]).inv().apply([0.0, 0.0, -1.0])
        self.ang_vel = np.array([w.x, w.y, w.z])
        self.t_imu = time.monotonic()
        self.stamp_imu = stamp_s(msg)

    def cb_flag(self, msg):
        if msg.data and not self.load_expected:
            self.get_logger().info("load_expected -> True")
        self.load_expected = float(msg.data)

    def cb_head(self, msg):
        v = float(msg.data)
        if not math.isfinite(v):                       # explicit release
            if self.head_override and self.head_target is not None:
                self.get_logger().info("[HEAD] release requested -> returning to default")
            self.head_target = None
            return
        self.head_target = float(np.clip(v, self.head_min, self.head_max))
        self.head_last_t = self._clock_now()
        self.head_released_logged = False
        if not self.head_override:
            self.head_override, self.head_need_init = True, True
            self.get_logger().info(f"[HEAD] override engaged toward {self.head_target:+.3f} rad")

    def _clock_now(self):
        """Time on the control clock: latest /joint_states stamp (--clock stamp) or wall time."""
        if self.clock == "stamp":
            return self.last_stamp if self.last_stamp is not None else 0.0
        return time.monotonic()

    def update_head(self, now):
        """Advance the override by one control step; head angle to command, or None (policy owns it)."""
        if not self.head_override or not self.policy_active:
            return None
        if self.head_need_init:                        # start from where the head IS
            self.head_cur = float(self.q[self.head_q_i])
            self.head_need_init = False
            self.get_logger().info(f"[HEAD] starting from measured {self.head_cur:+.3f} rad")
        default = float(self.default_pos[self.head_q_i])
        stale = self.head_target is None or self.head_last_t is None or now - self.head_last_t > self.head_timeout
        if stale and not self.head_released_logged:
            self.get_logger().info("[HEAD] no command (timeout/release) -> slewing to default")
            self.head_released_logged = True
        goal = default if stale else self.head_target
        step = self.head_rate * self.step_dt
        self.head_cur += float(np.clip(goal - self.head_cur, -step, step))
        if stale and abs(self.head_cur - default) < 1e-3:
            self.head_override, self.head_target = False, None
            self.get_logger().info("[HEAD] override released -> policy controls head")
            return None
        return self.head_cur

    def cb_arm(self, msg):
        if not self.policy_active:
            self.get_logger().warn("Ignoring /arm_targets until the policy is active - start the sequencer "
                                   "after 'Policy active'", throttle_duration_sec=2.0)
            return
        idx = {name: i for i, name in enumerate(msg.name)}
        if all(j in idx for j in self.arm_names):
            self.arm_target = np.array([msg.position[idx[j]] for j in self.arm_names])
            self.t_arm = time.monotonic()

    def cb_applied(self, msg):
        """Arm values the robot is actually being commanded (mux output / own command)."""
        idx = {name: i for i, name in enumerate(msg.name)}
        if all(j in idx for j in self.arm_names):
            self.applied_arm = np.array([msg.position[idx[j]] for j in self.arm_names])

    def update_arm_source(self):
        """Fresh /arm_targets -> follow them; otherwise hold the last applied arm values."""
        fresh = self.t_arm is not None and time.monotonic() - self.t_arm <= ARM_FRESH_S
        if fresh or self.applied_arm is None:
            if self.arm_holding and fresh:
                self.get_logger().info("[ARMS] following /arm_targets")
            self.arm_holding = False
            return
        if not self.arm_holding:
            self.get_logger().info("[ARMS] no /arm_targets - holding the last applied arm command")
            self.arm_holding = True
        self.arm_target = self.applied_arm.copy()

    # ── Control step ────────────────────────────────────────────────────────

    def on_joint_stamp(self, t):
        """--clock stamp (default): step the policy every step_dt of /joint_states STAMP time.
        In Isaac Sim that is sim time, so the policy runs at its trained rate however fast or
        slow the sim runs relative to the wall clock; on the real robot stamps are real time."""
        if self.last_stamp is not None and t < self.last_stamp - 1e-6:
            self.get_logger().warn("Sim time jumped back (Stop/Play?) - restarting from the start pose")
            self.reset_state()
        self.same_stamp_count = self.same_stamp_count + 1 if t == self.last_stamp else 0
        self.last_stamp = t
        if self.same_stamp_count > 20:
            self.get_logger().error("/joint_states stamps are not advancing - wire Isaac Read Simulation Time into "
                                    "the publisher, or run with --clock wall", throttle_duration_sec=2.0)
            return
        if abs(t - self.stamp_imu) > IMU_MAX_LAG_S:
            self.get_logger().warn(f"IMU and /joint_states stamps {abs(t - self.stamp_imu) * 1000:.0f} ms apart "
                                   "- skipping step", throttle_duration_sec=1.0)
            return
        if self.next_step is None:
            self.next_step = t
        if t + 1e-6 < self.next_step:
            return
        self.control_step(t)
        self.next_step += self.step_dt
        if self.next_step <= t:                                    # missed messages: resync to the grid
            self.next_step = t + self.step_dt

    def tick(self):
        """--clock wall: fixed wall-clock timer (only correct if the sim runs at real time)."""
        now = time.monotonic()
        if self.t_joints is None or self.t_imu is None:
            self.get_logger().info("Waiting for /joint_states and IMU ...", throttle_duration_sec=2.0)
            return
        if now - self.t_joints > STALE_S or now - self.t_imu > STALE_S:
            self.get_logger().warn("Sensor data stale - not commanding", throttle_duration_sec=1.0)
            return
        self.control_step(now)

    def control_step(self, now):
        """One policy step at time `now` (stamp or wall time, per --clock)."""
        if self.startup_q is None:
            self.check_start_pose()
        if not self.policy_active:
            self.startup_step(now)
            if not self.policy_active:
                return
        self.update_arm_source()

        obs = np.concatenate([np.asarray(f(), dtype=np.float32).ravel() for _, _, f in self.builders])
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs)[None])[0].numpy().astype(np.float64)
        head = self.update_head(now)
        if head is not None:
            # express the override in action space: offset + scale * action == head,
            # so last_action (an observation) matches what is really commanded
            i = self.head_act_i
            action[i] = (head - self.act_offset[i]) / self.act_scale[i]
        self.last_action = action
        self.publish(self.act_names + self.arm_names,
                     np.concatenate([self.act_offset + self.act_scale * action, self.arm_target]))

    def reset_state(self):
        """Back to 'just started' (e.g. after Stop/Play in Isaac Sim)."""
        self.next_step = None
        self.startup_q = self.startup_t0 = None
        self.policy_active = False
        self.last_action[:] = 0.0
        self.arm_target = None
        self.t_arm = self.applied_arm = None
        self.arm_holding = False
        self.load_expected = 0.0
        self.head_override, self.head_need_init, self.head_target, self.head_last_t = False, False, None, None

    def check_start_pose(self):
        """One-time warning if the robot doesn't start where every training episode did."""
        diff = self.q - self.default_pos
        off = [f"{self.joint_names[i]} {diff[i]:+.2f}" for i in np.flatnonzero(np.abs(diff) > START_TOL)]
        if off:
            self.get_logger().warn(f"{len(off)} joints start > {START_TOL} rad from the training start pose "
                                   f"(sim - train): {', '.join(off)}")
        else:
            self.get_logger().info(f"Start pose OK: all joints within {START_TOL} rad of training")

    def startup_step(self, now):
        """Optional (--ramp/--hold): ease every joint to the training default pose, hold, then hand
        over to the policy. With both 0 (default) the policy starts on this first step."""
        if self.startup_q is None:
            self.startup_q, self.startup_t0 = self.q.copy(), now
            if self.ramp_s > 0 or self.hold_s > 0:
                self.get_logger().info(f"Startup: easing to training default pose over {self.ramp_s:.1f}s, "
                                       f"holding {self.hold_s:.1f}s")
        t = now - self.startup_t0
        if self.ramp_s > 0 or self.hold_s > 0:
            a = min(t / self.ramp_s, 1.0) if self.ramp_s > 0 else 1.0
            a = a * a * (3 - 2 * a)                     # smoothstep
            self.publish(self.joint_names, self.startup_q + a * (self.default_pos - self.startup_q))
        if t >= self.ramp_s + self.hold_s:
            self.policy_active = True
            self.arm_target = self.default_pos[self.arm_idx].copy()   # as at the start of every training episode
            self.get_logger().info("Policy active - start the sequencer when the robot is steady")

    def publish(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in positions]
        self.pub.publish(msg)


def main():
    p = argparse.ArgumentParser(description="H2 pickup balance controller")
    p.add_argument("--policy", default=os.path.join(MODEL_DIR, "policy.pt"),
                   help="TorchScript policy from Isaac Lab play.py (default: %(default)s)")
    p.add_argument("--io", default=os.path.join(MODEL_DIR, "policy_io.json"),
                   help="policy_io.json from export_policy_io.py, same run as the policy (default: %(default)s)")
    p.add_argument("--cmd-topic", default="/joint_command", help="whole-body command topic the sim/robot listens on")
    p.add_argument("--arm-topic", default="/arm_targets")
    p.add_argument("--imu-topic", default="/imu", help="pelvis IMU (sensor_msgs/Imu)")
    p.add_argument("--applied-topic", default="/joint_command",
                   help="what the robot is actually commanded (mux output); arms hold these values without /arm_targets")
    p.add_argument("--ramp", type=float, default=0.0,
                   help="optional startup: seconds to ease into the training default pose (0 = off)")
    p.add_argument("--hold", type=float, default=0.0,
                   help="optional startup: seconds to hold the default pose before the policy starts (0 = off)")
    p.add_argument("--clock", choices=["stamp", "wall"], default="stamp",
                   help="stamp (default): step on /joint_states header time = sim time in Isaac Sim, correct at any "
                        "real-time factor; wall: fixed wall-clock timer")
    p.add_argument("--head-topic", default="/head_pitch_cmd", help="std_msgs/Float64 head pitch override (rad)")
    p.add_argument("--head-min", type=float, default=-0.5, help="head override lower clamp (rad) - check URDF limits")
    p.add_argument("--head-max", type=float, default=0.7, help="head override upper clamp (rad) - check URDF limits")
    p.add_argument("--head-rate", type=float, default=0.5, help="head override slew rate (rad/s)")
    p.add_argument("--head-timeout", type=float, default=1.0,
                   help="release the head override after this long without a command (s, control clock)")
    args = p.parse_args()
    for path in (args.policy, args.io):
        if not os.path.isfile(path):
            raise SystemExit(f"Not found: {path}\nCopy policy.pt and policy_io.json from the same run into {MODEL_DIR}")

    rclpy.init()
    node = BalanceController(args.policy, args.io, args.cmd_topic, args.arm_topic, args.imu_topic, args.applied_topic,
                             ramp_s=args.ramp, hold_s=args.hold, clock=args.clock,
                             head_topic=args.head_topic, head_min=args.head_min, head_max=args.head_max,
                             head_rate=args.head_rate, head_timeout=args.head_timeout)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()