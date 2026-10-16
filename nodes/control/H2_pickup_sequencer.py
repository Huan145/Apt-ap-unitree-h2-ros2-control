#!/usr/bin/env python3
"""
H2 Pickup Sequencer - arm side of the box pickup, built on H2_ik_node.

Per arm, the same IK as H2_ik_node: full pose (position + orientation of the
/target_pose), 1 mm / 0.01 rad, 20 mm Cartesian warm-start waypoints, targets
more than 50 mm out of reach rejected, smoothstep joint moves limited to
--max-vel deg/s. On top of that, a phase machine. A phase ends when the
MEASURED hands (FK of /joint_states) are within --tol of the commanded pose,
not on a timer:

  1 wide      one move straight to a FIXED pose, --wide X Y Z for the left hand (right: -Y),
              pelvis frame, independent of the targets; position only, orientation free.
              A via point for clearance only: ends as soon as the hands settle, however far off.
  2 standoff  exactly the target position, IDENTITY hand orientation (palms facing each
              other, fingers forward); live: replanned when a /target_pose moved (--replan-mm/deg)
  3 pinch     GRASP_REACH_M + --squeeze (5 + 3 cm) inward from the targets at the START of the
              pinch, identity orientation, close-in >= --pinch-time s, then --settle s hold, then on.
              NOT replanned (the hands touch/occlude the box -> targets jitter -> endless replans).
              The squeeze depth sets the grip force (PD spring: 2 cm ~ 10-15 N per hand), but too
              deep and the box twists the soft wrists (hand slants): 5 cm was too much
  4 tilt      joint space, one move: *_wrist_pitch_joint += --tilt-wrist (-9 deg)
              (+ *_elbow_joint += --tilt-elbow, default 0)
  5 carry     both hands shifted together so the grip midpoint goes to CARRY_CENTER (grip width
              kept); hands held --carry-pitch (9 deg, same direction as the tilt) from the GROUND,
              using the pelvis IMU (pelvis lean is compensated)
  6 hold      the carry pose; re-levels the hands when the pelvis tilts > LEVEL_KEEP_DEG

"Outside / inward" = along the horizontal line between the two targets, so the
targets' orientation convention does not matter. The target is the grasp
STANDOFF published by pose_overlay_visualizer_tracking_grasp.py: GRASP_REACH
(5 cm) outside the box face + GRASP_PELVIS_OFFSET, the same rule as the
training library; the pinch then moves 5 cm + --squeeze in (--squeeze past the face).
Targets are in the LIVE pelvis frame (TF pelvis <- Camera through the moving
waist/head), so the IK model uses the measured waist (--waist measured).
Live replanning follows the
box in the pelvis frame while the torso sways, but never interrupts a move:
a replan starts only once the current move has finished, so every arm move
starts and ends at rest (interrupting moves jerks the arms -> the torso sways
-> the pelvis-frame targets move -> more replans). After the pinch the grip is
frozen (the box then moves with the hands). Pinch, tilt and carry end after
CONTACT_TIMEOUT_S even if not reached (the box blocks the squeeze); wide and
standoff wait until reached, or until the hands have settled (moved < SETTLE_MOVE_MM
for SETTLE_S) within SETTLED_MAX_MM (wide: any distance / standoff 30 mm), and
log which joints are off.

Gravity compensation (--gravity-comp, default on): the arm drives are PD
springs, so under the arms' own weight they settle a few degrees low (5-7 cm
at the hand with the arms raised). Each published arm target is raised by
tau_gravity(q) / kp (Pinocchio gravity torque, kp = joint stiffness from
policy_io.json), so the hands settle where planned. Plans, logs and the
"hands vs command" check use the uncompensated pose.
Box weight (--box-mass, default 7 kg): from the start of carry, each hand also
holds half the box. That load is added the same way (J^T * upward force at the
hand / kp), faded in over LOAD_RAMP_S, so the hands keep their carry height and
their pitch to the ground instead of sagging under the box.

Clock (--clock stamp, default): moves and phases run on /joint_states stamp
time (sim time in Isaac Sim), using only /joint_states messages with an /imu
message of the same stamp, like the balance controller.

Subscribes: /target_pose_left, /target_pose_right (PoseStamped, pelvis frame, hand link pose)
            /joint_states + /imu
Publishes:  /arm_targets (JointState, 14 arm joints, rad) at 50 Hz -> H2_balance_controller
            /load_expected (Bool): True from the start of carry (the lift), as in training

Usage:
  conda activate ik_env && source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=99
  python3 H2_pickup_sequencer.py                       # with H2_balance_controller running
  python3 H2_pickup_sequencer.py --squeeze 0.04 --settle 1.0   # firmer grip, longer hold before tilt + lift
  python3 H2_pickup_sequencer.py --topic /arm_joints --clock wall   # drive the sim directly (pinned torso, no policy)
"""

import argparse
import json
import os
import threading
import time

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(_REPO_ROOT, "assets", "h2_description", "H2.urdf")
POLICY_IO = os.path.join(_REPO_ROOT, "models", "h2_balance_box_pickup", "policy_io.json")   # joint stiffness (kp)

LEFT_ARM_NAMES = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
RIGHT_ARM_NAMES = [n.replace("left_", "right_") for n in LEFT_ARM_NAMES]
ARM_JOINT_NAMES = LEFT_ARM_NAMES + RIGHT_ARM_NAMES

CONTROL_HZ = 50
# IK - as H2_ik_node
IK_MAX_ITER, IK_EPS_POS, IK_EPS_ROT, IK_DT, IK_DAMP = 1000, 1e-3, 1e-2, 0.5, 1e-6
REJECT_M = 0.05           # final IK error above this -> target rejected, arm holds
IK_WARN_MM, IK_WARN_DEG = 5.0, 5.0   # accepted but logged as a warning
WAYPOINT_M = 0.02
MIN_MOVE_S = 0.5
# Geometry (m, pelvis frame). "out" = away from the box along the line between the two targets
# wide: --wide-out / --wide-up (default 15 / 15 cm) outside / above the standoff, at the target's x
#       (hands pulled back beside the body put the arm mass behind the robot -> it fell backwards)
GRASP_REACH_M = 0.05                   # the target is this far outside the box face (visualizer GRASP_REACH)
LEVEL_KEEP_DEG = 3.0                   # hold: re-level the hands when the pelvis tilted more than this
ELBOW_IDX, WRIST_PITCH_IDX = 3, 5      # index in each arm's 7 joints (see LEFT_ARM_NAMES)
CARRY_CENTER = np.array([0.20, 0.0, 0.275])
# Phase machine
PHASES = ["wide", "standoff", "pinch", "tilt", "carry", "hold"]
LIVE = {"standoff"}                       # replanned (after the current move) when the target moved
RETRY = {"wide", "standoff", "pinch"}     # an IK-rejected arm retries on the next target update
CONTACT = {"pinch", "tilt", "carry"}      # may be blocked by the box -> end after CONTACT_TIMEOUT_S
CONTACT_TIMEOUT_S = 1.0
WARN_EVERY_S = 2.0
WAIT_TIMEOUT_S = 10.0
LEVEL_WARN_M = 0.03       # warn when the two targets differ in height by more than this
SETTLE_S, SETTLE_MOVE_MM = 1.0, 2.0      # hands moved < 2 mm in 1 s = settled
SETTLED_MAX_MM = {"wide": np.inf, "standoff": 30.0}   # settled within this -> phase ends anyway (warning);
                                                      # wide is only a clearance via point; standoff next to the box
                                                      # far off = blocked -> keep waiting
GRAV_MAX_RAD = 0.3        # cap on the gravity-compensation offset per joint
LOAD_RAMP_S = 1.0         # box-weight compensation fades in over this time from the start of carry

SIM_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_ALL)


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def pose_of(msg):
    """(pos, R) or None if the message is invalid (NaN, zero-norm quaternion)."""
    p, o = msg.pose.position, msg.pose.orientation
    pos, q = np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6):
        return None
    return pos, Rotation.from_quat(q).as_matrix()


def moved(a, b, dist_m, ang_deg):
    return (a is None or np.linalg.norm(a[0] - b[0]) > dist_m
            or np.degrees(np.linalg.norm(pin.log3(a[1].T @ b[1]))) > ang_deg)


class Arm:
    def __init__(self, side, names, ee, sign, q_idx, v_idx):
        self.side, self.names, self.ee, self.sign, self.q_idx, self.v_idx = side, names, ee, sign, q_idx, v_idx
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.q_cmd = None                  # (7,) commanded now
        self.q_from = self.q_to = None     # current joint-space move
        self.t0 = None                     # move start time (None = start on the next tick)
        self.dur = 0.0
        self.goal = None                   # (pos, rot) of the current move
        self.planned_for = None            # target (pos, rot) the current move was planned from
        self.busy = False                  # IK thread running
        self.failed = False                # last IK rejected
        self.rejected_for = None           # target that was rejected (retry only once it moved)


class PickupSequencer(Node):
    def __init__(self, args):
        super().__init__("h2_pickup_sequencer")
        self.a = args
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        jnt = lambda n: self.model.joints[self.model.getJointId(n)]
        self.arms = [Arm(s, names, self.model.getFrameId(ee), sign, [jnt(n).idx_q for n in names], [jnt(n).idx_v for n in names])
                     for s, names, ee, sign in [("L", LEFT_ARM_NAMES, "left_hand_link", 1.0),
                                                ("R", RIGHT_ARM_NAMES, "right_hand_link", -1.0)]]
        self.joint_state = None
        self.imu = None                    # latest /imu paired with /joint_states (--clock stamp)
        self.targets = {"L": None, "R": None}
        self.phase = 0
        self.started = self.done = False
        self.now = self.t_start = None
        self.last_frame = self.last_stamp = None
        self.done_t = self.warn_t = None   # time both arms finished the current move / last warning

        if args.clock == "stamp":   # same filter as the controller: only sim-stamped joint_states (paired with /imu)
            self._sync_subs = [Subscriber(self, JointState, "/joint_states", qos_profile=SIM_QOS),
                               Subscriber(self, Imu, args.imu_topic, qos_profile=SIM_QOS)]
            self._sync = TimeSynchronizer(self._sync_subs, 10)
            self._sync.registerCallback(lambda js, imu: self.cb_joint_states(js, imu))
        else:
            self.create_subscription(JointState, "/joint_states", self.cb_joint_states, SIM_QOS)
        self.create_subscription(PoseStamped, "/target_pose_left", lambda m: self.cb_target("L", m), 10)
        self.create_subscription(PoseStamped, "/target_pose_right", lambda m: self.cb_target("R", m), 10)
        self.pub = self.create_publisher(JointState, args.topic, 10)
        self.kp = None
        if args.gravity_comp:
            try:
                io = json.load(open(args.io))
                kp = dict(zip(io["joint_names"], io["stiffness"]))
                self.kp = np.array([kp[n] for n in ARM_JOINT_NAMES])
                self.get_logger().info(f"Gravity compensation on, arm kp {self.kp.min():.0f}-{self.kp.max():.0f} Nm/rad ({args.io})")
            except (OSError, KeyError, ValueError) as e:
                self.get_logger().warning(f"Gravity compensation OFF - cannot read kp from {args.io}: {e}")
        self.settle_ref = None             # (time, measured hand positions) for the settled check
        self.load_t0 = None                # time carry started (box-weight compensation ramp)
        self.flag_pub = self.create_publisher(Bool, "/load_expected", 10)

    def log(self, s, warn=False):
        t = f"[{self.now - self.t_start:6.2f}s] " if self.started else ""
        if warn:        # rclpy: one severity per call site -> separate lines
            self.get_logger().warning(t + s)
        else:
            self.get_logger().info(t + s)

    @property
    def name(self):
        return PHASES[self.phase]

    # ── Inputs ──────────────────────────────────────────────────────────────

    def wait_for_inputs(self):
        js = "/joint_states" + (f"+{self.a.imu_topic} (same stamp)" if self.a.clock == "stamp" else "")
        self.get_logger().info(f"Waiting for {js} and /target_pose_left/right ...")
        t0 = time.time()
        while rclpy.ok() and (self.joint_state is None or None in self.targets.values()):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > WAIT_TIMEOUT_S:
                missing = [n for n, v in [(js, self.joint_state), ("/target_pose_left", self.targets["L"]),
                                          ("/target_pose_right", self.targets["R"])] if v is None]
                raise RuntimeError(f"Timed out waiting for {missing} (ROS_DOMAIN_ID? sim playing?)")

    def cb_joint_states(self, msg, imu=None):
        self.joint_state = msg
        if imu is not None:
            self.imu = imu
        if not self.started or self.a.clock != "stamp" or self.done:
            return
        t = stamp_s(msg)
        if self.last_stamp is not None and t < self.last_stamp - 1e-6:
            self.get_logger().error("Sim time jumped back (Stop/Play?) - stopping; rerun the sequencer")
            self.done = True
            return
        self.last_stamp = t
        frame = int(t * CONTROL_HZ + 1e-6)
        if frame != self.last_frame:
            self.last_frame = frame
            self.tick(t)

    def cb_target(self, side, msg):
        pose = pose_of(msg)
        if pose is None:            # keep the last valid target
            o = msg.pose.orientation
            self.get_logger().warning(f"ignoring invalid /target_pose_{'left' if side == 'L' else 'right'} "
                                      f"(quat {o.x:.2f} {o.y:.2f} {o.z:.2f} {o.w:.2f})", throttle_duration_sec=2.0)
            return
        self.targets[side] = pose
        self.target_frame = msg.header.frame_id
        if None not in self.targets.values():
            dz = self.targets["L"][0][2] - self.targets["R"][0][2]
            if abs(dz) > LEVEL_WARN_M:
                self.get_logger().warning(
                    f"targets differ in height by {dz * 100:+.0f} cm (L - R): box pose tilted (FoundationPose roll?)"
                    + (" - using their mean height" if self.a.level_targets else ""), throttle_duration_sec=5.0)
        arm = self.arms[0 if side == "L" else 1]
        if (self.started and self.name in RETRY and arm.failed and not arm.busy     # rejected arm (holds still):
                and moved(arm.rejected_for, pose, self.a.replan_mm / 1000, self.a.replan_deg)):   # retry once it moved
            self.start_plan(arm)

    def replan_moved_targets(self, now):
        """Live phases: replan an arm whose target moved, but only once its current move has finished."""
        if self.name not in LIVE:
            return
        for arm in self.arms:
            idle = not arm.busy and arm.q_to is not None and arm.t0 is not None and now - arm.t0 >= arm.dur
            if idle and arm.planned_for is not None and moved(arm.planned_for, self.targets[arm.side],
                                                                self.a.replan_mm / 1000, self.a.replan_deg):
                self.log(f"{arm.side} target moved {np.linalg.norm(self.targets[arm.side][0] - arm.planned_for[0]) * 1000:.0f} mm "
                         f"-> replan {self.name}")
                self.start_plan(arm)

    # ── Model helpers ───────────────────────────────────────────────────────

    def base_q(self, arm_source="cmd"):
        """Full Pinocchio q. Waist/legs: measured (default) or 0 (--waist zero, as H2_ik_node).
        Arms: commanded (cmd) or measured (meas)."""
        q = pin.neutral(self.model)
        meas = dict(zip(self.joint_state.name, self.joint_state.position))
        if self.a.waist == "measured":
            for n, v in meas.items():
                if self.model.existJointName(n) and self.model.joints[self.model.getJointId(n)].nq == 1:
                    q[self.model.joints[self.model.getJointId(n)].idx_q] = v
        for arm in self.arms:
            q[arm.q_idx] = arm.q_cmd if arm_source == "cmd" else [meas.get(n, 0.0) for n in arm.names]
        return q

    def fk(self, data, q, ee):
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        return data.oMf[ee].translation.copy(), data.oMf[ee].rotation.copy()

    def target_pos(self, side):
        """Target position; with --level-targets both hands use the mean target height (a box on a
        table has both grasp faces at the same height - a height difference = tilted box pose)."""
        p = self.targets[side][0].copy()
        if self.a.level_targets:
            p[2] = (self.targets["L"][0][2] + self.targets["R"][0][2]) / 2
        return p

    def pelvis_tilt(self):
        """Pelvis roll/pitch as a rotation: pelvis frame relative to its yaw-only (level) frame, from the
        IMU orientation. Identity without an IMU (--clock wall)."""
        if self.imu is None:
            return np.eye(3)
        o = self.imu.orientation
        R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        return Rotation.from_euler("z", -np.arctan2(R[1, 0], R[0, 0])).as_matrix() @ R

    def carry_rot(self, arm, R_ref):
        """Hand orientation (pelvis frame) pitched --carry-pitch from the ground, fingers forward, in the
        pitch direction of R_ref (the post-tilt hand). Remembers the pelvis tilt it was computed for."""
        s = np.sign(Rotation.from_matrix(R_ref).as_euler("xyz")[1]) or -1.0
        arm.level_used = self.pelvis_tilt()
        return arm.level_used.T @ Rotation.from_euler("y", s * self.a.carry_pitch, degrees=True).as_matrix()

    def relevel_hold(self, now):
        """Hold: re-plan an idle arm when the pelvis tilted > LEVEL_KEEP_DEG since its last plan."""
        if self.name != "hold" or self.imu is None:
            return
        for arm in self.arms:
            idle = not arm.busy and arm.q_to is not None and arm.t0 is not None and now - arm.t0 >= arm.dur
            if idle and getattr(arm, "level_used", None) is not None:
                d = np.degrees(np.linalg.norm(pin.log3(arm.level_used.T @ self.pelvis_tilt())))
                if d > LEVEL_KEEP_DEG:
                    self.log(f"{arm.side} pelvis tilted {d:.1f} deg -> re-level hands")
                    self.start_plan(arm)

    def out_dir(self):
        """Unit vector right target -> left target, horizontal."""
        u = (self.targets["L"][0] - self.targets["R"][0]) * [1, 1, 0]
        return u / np.linalg.norm(u)

    # ── IK (as H2_ik_node, arm-only Jacobian in the hand frame) ─────────────

    def ik(self, data, arm, q, pos, rot):
        """Returns (q_arm (7,), pos_err_mm, rot_err_deg), 'reject' if > REJECT_M, None if cancelled."""
        p0, R0 = self.fk(data, q, arm.ee)
        n = max(int(np.ceil(np.linalg.norm(pos - p0) / WAYPOINT_M)), 1)
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([R0, rot]))) if rot is not None else None
        for i in range(1, n + 1):
            wp_p, wp_R = p0 + (pos - p0) * i / n, (slerp(i / n).as_matrix() if slerp else None)
            for _ in range(IK_MAX_ITER):
                if arm.cancel.is_set():
                    return None
                pin.forwardKinematics(self.model, data, q)
                pin.updateFramePlacements(self.model, data)
                oMf = data.oMf[arm.ee]
                if wp_R is not None:
                    err = pin.log6(oMf.actInv(pin.SE3(wp_R, wp_p))).vector
                    if np.linalg.norm(err[:3]) < IK_EPS_POS and np.linalg.norm(err[3:]) < IK_EPS_ROT:
                        break
                    J = pin.computeFrameJacobian(self.model, data, q, arm.ee, pin.ReferenceFrame.LOCAL)
                else:     # position only
                    err = np.r_[wp_p - oMf.translation, 0.0, 0.0, 0.0]
                    if np.linalg.norm(err[:3]) < IK_EPS_POS:
                        break
                    J = pin.computeFrameJacobian(self.model, data, q, arm.ee, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
                    J[3:] = 0.0
                J = J[:, arm.v_idx]
                v = np.zeros(self.model.nv)
                v[arm.v_idx] = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(6), err)
                q = np.clip(pin.integrate(self.model, q, IK_DT * v), self.model.lowerPositionLimit, self.model.upperPositionLimit)
        p, R = self.fk(data, q, arm.ee)
        e_pos = np.linalg.norm(p - pos)
        e_rot = np.degrees(np.linalg.norm(pin.log3(R.T @ rot))) if rot is not None else 0.0
        return "reject" if e_pos > REJECT_M else (q[arm.q_idx].copy(), e_pos * 1000, e_rot)

    # ── Planning ────────────────────────────────────────────────────────────

    def phase_goal(self, arm):
        p, R = self.target_pos(arm.side), self.targets[arm.side][1]
        out = arm.sign * self.out_dir()
        if self.name == "wide":      # fixed pose (right = mirrored y), orientation free
            return np.array(self.a.wide) * [1.0, arm.sign, 1.0], None
        if self.name == "standoff":  # target position, identity orientation (target orientation not used)
            return p, np.eye(3)
        if self.name == "pinch":
            return p - (GRASP_REACH_M + self.a.squeeze) * out, np.eye(3)
        return None     # carry: from the post-tilt pose, in the thread

    def start_plan(self, arm):
        arm.cancel.set()
        if arm.thread and arm.thread.is_alive():
            arm.thread.join(timeout=0.2)
        arm.cancel.clear()
        if self.name == "tilt":     # joint space, no IK
            q = arm.q_to.copy()
            for i, deg in ((WRIST_PITCH_IDX, self.a.tilt_wrist), (ELBOW_IDX, self.a.tilt_elbow)):
                j = arm.q_idx[i]
                q[i] = np.clip(q[i] + np.radians(deg), self.model.lowerPositionLimit[j], self.model.upperPositionLimit[j])
            self.commit(arm, q, None)
            self.log(f"{arm.side} tilt: wrist pitch {self.a.tilt_wrist:+.0f} deg, elbow {self.a.tilt_elbow:+.0f} deg, "
                     f"move {arm.dur:.2f}s")
            return
        arm.planned_for = self.targets[arm.side]
        arm.busy = True
        arm.thread = threading.Thread(target=self.plan, args=(arm, self.name, self.base_q(), self.phase_goal(arm)),
                                      daemon=True)
        arm.thread.start()

    def plan(self, arm, name, q, goal):
        """IK thread (like H2_ik_node): solve, then hand the result to tick() via commit()."""
        data = self.model.createData()
        if name == "carry":     # shift both hands by the same offset (grip width kept), hands pitched from the ground
            mid = np.mean([self.fk(data, q, a.ee)[0] for a in self.arms], axis=0)
            p_now, R_now = self.fk(data, q, arm.ee)
            pos, rot = p_now + CARRY_CENTER - mid, self.carry_rot(arm, R_now)
        elif name == "hold":    # re-level: same position, orientation re-expressed for the current pelvis tilt
            pos, rot = arm.goal[0], self.carry_rot(arm, arm.goal[1])
        else:
            pos, rot = goal
        res = self.ik(data, arm, q, pos, rot)
        if res is None or arm.cancel.is_set():
            return
        if res == "reject":
            self.log(f"{arm.side} {name}: target REJECTED (> {REJECT_M * 1000:.0f} mm out of reach) "
                     f"- arm holds; pos {pos.round(3)}", warn=True)
            arm.failed, arm.busy, arm.rejected_for, arm.planned_for = True, False, arm.planned_for, None
            return
        q_arm, e_pos, e_rot = res
        self.commit(arm, q_arm, (pos, rot), self.a.pinch_time if name == "pinch" else MIN_MOVE_S)
        bad = e_pos > IK_WARN_MM or e_rot > IK_WARN_DEG
        self.log(f"{arm.side} {name}: IK {e_pos:.1f} mm" + (f" / {e_rot:.1f} deg" if rot is not None else " (orientation free)")
                 + f", move {arm.dur:.2f}s -> {pos.round(3)}"
                 + (" - IK could not fully reach this pose (limits?)" if bad else ""), warn=bad)

    def commit(self, arm, q_to, goal, min_s=MIN_MOVE_S):
        with arm.lock:
            arm.q_from, arm.q_to, arm.goal = arm.q_cmd.copy(), q_to, goal
            arm.dur = max(np.degrees(np.max(np.abs(q_to - arm.q_from))) / self.a.max_vel, min_s)
            arm.t0, arm.busy, arm.failed = None, False, False

    # ── 50 Hz tick ──────────────────────────────────────────────────────────

    def start(self):
        meas = dict(zip(self.joint_state.name, self.joint_state.position))
        for arm in self.arms:
            arm.q_cmd = np.array([meas[n] for n in arm.names])
        self.now = self.t_start = stamp_s(self.joint_state) if self.a.clock == "stamp" else time.monotonic()
        self.started = True
        self.log(f"targets in frame '{self.target_frame}'")
        for side, (p, R) in self.targets.items():
            self.log(f"{side} target pos {p.round(3)} | rpy {Rotation.from_matrix(R).as_euler('xyz', degrees=True).round(1)} deg")
        self.check_targets()
        self.log(f"-> phase 1: {self.name}")
        for arm in self.arms:
            self.start_plan(arm)
        if self.a.clock == "wall":
            self.create_timer(1.0 / CONTROL_HZ, lambda: self.tick(time.monotonic()))

    def check_targets(self):
        """Warn about targets that cannot be a box grasp (the sequencer still runs; IK rejects them)."""
        (pl, Rl), (pr, Rr) = self.targets["L"], self.targets["R"]
        for side, p in (("L", pl), ("R", pr)):
            if not (-0.4 < p[2] < 0.6 and 0.05 < p[0] < 0.9):
                self.log(f"{side} target {p.round(2)} is outside the plausible box area (x 0.05..0.9, z -0.4..0.6 m in "
                         "the pelvis frame) - robot fallen / sim reset / FoundationPose lost the box / wrong TF?", warn=True)
        if pl[1] <= pr[1]:
            self.log("left target is not left of the right target - left/right swapped?", warn=True)
        if np.degrees(np.linalg.norm(pin.log3(Rl.T @ Rr))) > 90:
            self.log("left and right target orientations differ by > 90 deg - an unpatched "
                     "pose_overlay_visualizer_tracking_grasp.py (left hand turned ~180 deg)?", warn=True)

    def tick(self, now):
        self.now = now
        for arm in self.arms:
            with arm.lock:
                if arm.q_to is not None:
                    if arm.t0 is None:
                        arm.t0 = now
                    s = min((now - arm.t0) / arm.dur, 1.0)
                    arm.q_cmd = arm.q_from + s * s * (3 - 2 * s) * (arm.q_to - arm.q_from)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES
        msg.position = (np.concatenate([a.q_cmd for a in self.arms]) + self.gravity_offset()).tolist()
        self.pub.publish(msg)
        self.flag_pub.publish(Bool(data=self.phase >= PHASES.index("carry")))
        self.step_phase(now)
        self.replan_moved_targets(now)
        self.relevel_hold(now)

    def load_factor(self):
        """0 before carry, then ramps 0 -> 1 over LOAD_RAMP_S: share of the box weight compensated."""
        if self.phase < PHASES.index("carry") or self.load_t0 is None:
            return 0.0
        return float(np.clip((self.now - self.load_t0) / LOAD_RAMP_S, 0.0, 1.0))

    def gravity_offset(self):
        """(arm gravity + half the box weight per hand) torque / kp for the 14 arm joints at the commanded
        pose (0 if off). The load torque is J^T * (upward force the hand must give to hold half the box)."""
        if self.kp is None:
            return 0.0
        q = self.base_q("cmd")
        tau = pin.computeGeneralizedGravity(self.model, self.data, q).copy()
        f_up = np.array([0.0, 0.0, self.load_factor() * self.a.box_mass * 9.81 / 2])
        if f_up[2] > 0:
            for arm in self.arms:
                J = pin.computeFrameJacobian(self.model, self.data, q, arm.ee, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                tau[arm.v_idx] += J[:3, arm.v_idx].T @ f_up
        return np.clip(np.concatenate([tau[a.v_idx] for a in self.arms]) / self.kp, -GRAV_MAX_RAD, GRAV_MAX_RAD)

    def hand_errors_mm(self):
        """Measured hand position vs the commanded end of the current move, per arm; and the measured positions."""
        q_meas = self.base_q("meas")
        err, pos = [], []
        for arm in self.arms:
            q_goal = q_meas.copy()
            q_goal[arm.q_idx] = arm.q_to
            p = self.fk(self.data, q_meas, arm.ee)[0]
            err.append(np.linalg.norm(p - self.fk(self.data, q_goal, arm.ee)[0]) * 1000)
            pos.append(p)
        return err, np.array(pos)

    def settled(self, now, pos):
        """True once the measured hands moved < SETTLE_MOVE_MM for SETTLE_S."""
        if self.settle_ref is None or np.abs(pos - self.settle_ref[1]).max() * 1000 > SETTLE_MOVE_MM:
            self.settle_ref = (now, pos)
        return now - self.settle_ref[0] >= SETTLE_S

    def joints_off(self):
        """Joints > 0.05 rad from the command: planned cmd, value sent (with gravity comp), measured,
        gravity torque the pose needs (model) and the torque the sim reports (/joint_states effort)."""
        js = self.joint_state
        meas = dict(zip(js.name, js.position))
        eff = dict(zip(js.name, js.effort)) if len(js.effort) == len(js.name) else {}
        q_goal = self.base_q()
        for arm in self.arms:
            q_goal[arm.q_idx] = arm.q_to
        g = pin.computeGeneralizedGravity(self.model, self.data, q_goal).copy()
        off = self.gravity_offset() if self.kp is not None else np.zeros(14)
        out = []
        for i, (arm, n, c) in enumerate((a, n, c) for a in self.arms for n, c in zip(a.names, a.q_to)):
            if abs(meas[n] - c) > 0.05:
                j = arm.v_idx[arm.names.index(n)]
                out.append(f"{n}: cmd {c:+.2f} sent {c + off[i]:+.2f} meas {meas[n]:+.2f} | "
                           f"gravity needs {g[j]:+.1f} Nm, sim effort " + (f"{eff[n]:+.1f} Nm" if n in eff else "n/a"))
        return "; ".join(out) or "none > 0.05 rad"

    def step_phase(self, now):
        if self.name == "hold":
            return
        if not all(not a.busy and not a.failed and a.q_to is not None and a.t0 is not None and now - a.t0 >= a.dur
                   for a in self.arms):     # a rejected arm still holds its previous move -> not done
            self.done_t = None
            if any(a.failed for a in self.arms) and (self.warn_t is None or now - self.warn_t > WARN_EVERY_S):
                self.warn_t = now
                self.log(f"{self.name}: waiting for a reachable target ({', '.join(a.side for a in self.arms if a.failed)})", warn=True)
            return
        if self.done_t is None:
            self.done_t = self.warn_t = now
            self.settle_ref = None
        waited = now - self.done_t
        if self.name == "pinch" and waited < self.a.settle:
            return
        (el, er), pos = self.hand_errors_mm()
        reached = max(el, er) <= self.a.tol * 1000
        still = self.settled(now, pos) and max(el, er) <= SETTLED_MAX_MM.get(self.name, 0.0)
        timeout = self.name in CONTACT and waited >= (self.a.settle if self.name == "pinch" else CONTACT_TIMEOUT_S)
        if reached or still or timeout:
            why = "" if reached else (" - settled short of the command, moving on" if still else " - not reached, moving on (contact)")
            self.log(f"{self.name} done: hands vs command L {el:.0f} mm, R {er:.0f} mm"
                     + (f"{why}. Joints off: {self.joints_off()}" if why else ""), warn=bool(why))
            self.phase += 1
            self.log(f"-> phase {self.phase + 1}: {self.name}" + ("  | load_expected -> True" if self.name == "carry" else ""))
            if self.name == "carry":
                self.load_t0 = now
                if self.kp is not None and self.a.box_mass > 0:
                    self.log(f"box weight compensation: {self.a.box_mass:.1f} kg ({self.a.box_mass / 2:.1f} kg per hand), "
                             f"fading in over {LOAD_RAMP_S:.1f}s")
            if self.name != "hold":
                self.done_t = None
                for arm in self.arms:
                    self.start_plan(arm)
        elif now - self.warn_t > WARN_EVERY_S:
            self.warn_t = now
            self.log(f"{self.name}: waiting for hands (L {el:.0f} mm, R {er:.0f} mm > {self.a.tol * 1000:.0f} mm). "
                     f"Joints off: {self.joints_off()}", warn=True)


def main():
    p = argparse.ArgumentParser(description="H2 pickup arm sequencer (H2_ik_node IK + phases)")
    p.add_argument("--topic", default="/arm_targets", help="/arm_targets (balance controller) or /arm_joints (sim directly)")
    p.add_argument("--imu-topic", default="/imu", help="IMU paired with /joint_states (--clock stamp)")
    p.add_argument("--clock", choices=["stamp", "wall"], default="stamp",
                   help="stamp: /joint_states time (sim time); wall: wall-clock timer")
    p.add_argument("--max-vel", type=float, default=30.0, help="joint speed limit, deg/s (H2_ik_node: 30)")
    p.add_argument("--tol", type=float, default=0.015, help="phase ends when both measured hands are this close (m)")
    p.add_argument("--settle", type=float, default=0.5, help="pinch: hold the squeeze this long before tilt + lift (s)")
    p.add_argument("--squeeze", type=float, default=0.03, help="pinch: command this far past the box face (m) = grip force")
    p.add_argument("--pinch-time", type=float, default=1.0, help="pinch: minimum duration of the close-in move (s)")
    p.add_argument("--carry-pitch", type=float, default=9.0, help="carry/hold: hand pitch from the ground (deg)")
    p.add_argument("--replan-mm", type=float, default=10.0, help="replan when a target moves more than this")
    p.add_argument("--replan-deg", type=float, default=3.0, help="... or rotates more than this")
    p.add_argument("--wide", type=float, nargs=3, default=[0.1, 0.40, 0.25], metavar=("X", "Y", "Z"),
                   help="wide pose of the LEFT hand in the pelvis frame (m); right = (X, -Y, Z)")
    p.add_argument("--tilt-wrist", type=float, default=-9.0, help="tilt phase: wrist pitch change (deg)")
    p.add_argument("--tilt-elbow", type=float, default=0.0, help="tilt phase: elbow change (deg, + = more bend)")
    p.add_argument("--level-targets", action=argparse.BooleanOptionalAction, default=True,
                   help="use the mean height of both targets for both hands (--no-level-targets: as published)")
    p.add_argument("--gravity-comp", action=argparse.BooleanOptionalAction, default=True,
                   help="raise arm targets by tau_gravity/kp so the hands do not sag (--no-gravity-comp to disable)")
    p.add_argument("--box-mass", type=float, default=7.0,
                   help="box weight compensated from the start of carry (kg, split over both hands; 0 = off)")
    p.add_argument("--io", default=POLICY_IO, help="policy_io.json with the joint stiffness (kp) used in the sim")
    p.add_argument("--waist", choices=["measured", "zero"], default="measured",
                   help="waist in the IK model: measured (targets are in the live pelvis frame) or zero (H2_ik_node, pinned torso)")
    args = p.parse_args()

    rclpy.init()
    node = PickupSequencer(args)
    try:
        node.wait_for_inputs()
        node.start()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (RuntimeError, KeyboardInterrupt) as e:
        if isinstance(e, RuntimeError):
            node.get_logger().error(str(e))
    finally:
        for arm in node.arms:
            arm.cancel.set()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()