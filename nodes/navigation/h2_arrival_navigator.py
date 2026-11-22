#!/usr/bin/env python3
"""
H2 Arrival Navigator: walk to the workstation (packing table) while carrying the KLT box.

Same interface as h2_local_navigator.py (Trigger services ~/start and ~/cancel, latched ~/status and
~/result, cmd_vel, head_pitch_cmd, same velocity shaping and debug logs), built for a table that the
camera loses near the end of the walk (box in the hands, table filling the view):

  SCAN             no table yet: stand still scan_wait_s, then turn slowly until poses arrive
  ROTATE, APPROACH walk to the standoff point on the table estimate
  LATCH            table face <= latch_far (2.0 m) ahead: STOP, wait latch_settle_s, take the poses stamped
                   after the stop until they agree, freeze their median. The window latch_near..latch_far
                   (1.7..2.0 m) is where the whole 2.47 m table still fits the 82 deg field of view
  ROTATE, APPROACH, FINAL_ALIGN
                   on the frozen table and the pelvis pose from TF (odometry): to place_standoff from the
                   table face, square to it. Vision is ignored from the latch on

The table is kept in a FIXED frame (odom_frame): every /workstation_pose is mapped with the pelvis pose at
its image time, so a late or missing detection cannot bend the walk (the table does not move), and walking
on odometry after the latch is the same code path as before it, with the estimate frozen.

Table: mesh X = short side (table_depth), mesh Y = long side, as in targets.yaml. The robot faces a long side.

Needs: /active_target = workstation (the visualizer then publishes /workstation_pose, PoseStamped, pelvis
frame, image time) and TF odom_frame <- pelvis (any fixed frame in which the pelvis moves when it walks).

Start / cancel:
  ros2 service call /h2_arrival_navigator/start  std_srvs/srv/Trigger
  ros2 service call /h2_arrival_navigator/cancel std_srvs/srv/Trigger
Parameters: ros2 run style, e.g.  python3 h2_arrival_navigator.py --ros-args -p odom_frame:=World -p place_standoff:=0.6
"""

import json
import math
import traceback
from collections import deque
from enum import Enum
from types import SimpleNamespace as NS

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

PARAMS = dict(
    # topics and frames
    pose_topic='/workstation_pose', cmd_vel_topic='cmd_vel', joint_states_topic='joint_states',
    head_pitch_topic='head_pitch_cmd', odom_frame='odom', pelvis_frame='pelvis', camera_frame='Camera',
    control_rate_hz=20.0,
    # table (size from targets.yaml) and where to stand
    table_depth=0.762,          # m, short side (mesh X)
    place_standoff=0.50,        # m, pelvis -> table near face at the end. Placeholder: the carry pose of H2_pickup_sequencer
                                #    holds the box front 0.35 m ahead of the pelvis (grip x 0.20 + half the 0.30 m box), + 0.15 m
    lateral_offset=0.0,         # m, stand this far to the robot's LEFT of the table centre
    # latch
    latch_far=2.0, latch_near=1.7,   # m, pelvis -> table near face
    latch_tol=0.15,             # m, stationary range beyond latch_far + tol: walk on and latch again
    latch_max_tries=3,
    latch_settle_s=1.5,         # s standing still before the poses count (image stamp, not arrival time)
    window_s=1.0, latch_samples=5,   # poses taken in the last window_s: their median is the estimate; >= latch_samples
    latch_pos_tol=0.05, latch_yaw_tol=math.radians(3),   # m, rad: the two halves of the window must agree this well
    latch_timeout_s=20.0,       # s (tracker gaps of 2-9 s were seen)
    # scan
    scan_wait_s=3.0, scan_w=0.25, scan_dir=1.0, scan_max_deg=400.0, min_samples=3,
    # pose sanity
    max_range=6.0, min_horizontal_y=0.20,
    # motion: same names and defaults as h2_local_navigator (tuned there for the no-box policy)
    v_min=0.17, v_max=0.50, w_min=0.25, w_max=0.50, kp_lin=1.0, kp_ang=1.5, kp_perp=0.6,
    sidestep_max_s=2.0, sidestep_pause_s=2.0,
    pos_tol=0.05, yaw_tol=math.radians(4), capture_radius=0.15, bearing_gate=math.radians(30),
    settle_cycles=20, settle_hyst=1.4, nav_timeout_s=120.0, lost_grace_s=5.0,
    # head (center mode of h2_local_navigator: keep the table centre in the vertical middle of the image)
    head_enable=True, head_joint='head_pitch_joint', head_pitch_sign=1.0, head_pitch_min=-0.5,
    head_pitch_max=0.5, head_gain=0.7, head_deadband=math.radians(1.0), head_hold_after_done=True,
    # debug
    log_period_s=0.5, log_poses=False,
)


class Phase(Enum):
    IDLE = 'idle'
    SCAN = 'scan'
    ROTATE = 'rotate_to_goal'
    APPROACH = 'approach'
    LATCH = 'latch'
    FINAL = 'final_align'
    DONE = 'succeeded'
    FAILED = 'failed'


RUNNING = (Phase.SCAN, Phase.ROTATE, Phase.APPROACH, Phase.LATCH, Phase.FINAL)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def mat(t, q):
    """4x4 transform from a translation and a quaternion (messages with .x .y .z [.w])."""
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = t.x, t.y, t.z
    return T


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


def shape(err, kp, vmin, vmax, tol):
    """Proportional command with a dead band (tol) and a minimum speed the policy responds to."""
    return 0.0 if abs(err) < tol else math.copysign(min(max(abs(kp * err), vmin), vmax), err)


def fuse(s):
    """Samples (t, x, y, axis angle) -> (median x, median y, mean axis angle mod pi). The table's long axis
    has no direction, so the angle is averaged as 2*angle."""
    _, x, y, a = np.array(s).T
    return float(np.median(x)), float(np.median(y)), 0.5 * math.atan2(np.sin(2 * a).mean(), np.cos(2 * a).mean())


class H2ArrivalNavigator(Node):

    def __init__(self):
        super().__init__('h2_arrival_navigator')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        for k, v in PARAMS.items():
            self.declare_parameter(k, v)
        self.p = NS(**{k: self.get_parameter(k).value for k in PARAMS})
        self.phase = Phase.IDLE
        self.head_hist = deque(maxlen=1000)        # (sim time, measured head angle)
        self.reset_run()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseStamped, self.p.pose_topic, self.on_pose, 10)
        self.create_subscription(JointState, self.p.joint_states_topic, self.on_joint_states, 50)
        self.pub_cmd = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.pub_head = self.create_publisher(Float64, self.p.head_pitch_topic, 10)
        self.pub_point = self.create_publisher(PointStamped, '~/target_point', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_status = self.create_publisher(String, '~/status', latched)
        self.pub_result = self.create_publisher(String, '~/result', latched)
        self.pub_table = self.create_publisher(PoseStamped, '~/table_pose', latched)
        self.create_service(Trigger, '~/start', self.on_start)
        self.create_service(Trigger, '~/cancel', self.on_cancel)
        self.create_timer(1.0 / self.p.control_rate_hz, self.control_step)
        self.publish_status()
        p = self.p
        self.get_logger().info(
            f'H2 arrival navigator | latch window {p.latch_near}..{p.latch_far} m from the table face | '
            f'end {p.place_standoff} m from the face (lateral {p.lateral_offset:+.2f}) | table {p.table_depth} m deep | '
            f'odom frame "{p.odom_frame}" -> "{p.pelvis_frame}" | head {"ON" if p.head_enable else "off"}')

    def reset_run(self):
        self.samples = deque(maxlen=100)           # (image time, x, y, long-axis angle) in the odom frame
        self.est, self.latched, self.tries, self.latch_info = None, False, 0, {}
        self.t_start = self.t_latch = self.latch_t0 = self.lost_since = self.side_since = self.yaw_prev = None
        self.side_pause_until, self.settle, self.scanned = 0.0, 0, 0.0
        self.head_target = self.head_cmd = self.head_err = None
        self.last_cmd, self.last_hb = (0.0, 0.0, 0.0), None
        self.n_rx = self.n_acc = self.n_rej = self.n_tf_fail = 0

    # ---------------- time, TF ----------------

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tf(self, target, source, stamp=None):
        """4x4 transform source -> target at `stamp` (None: latest), or None."""
        t = rclpy.time.Time.from_msg(stamp) if stamp is not None else rclpy.time.Time()
        try:
            ts = self.tf_buffer.lookup_transform(
                target, source, t, timeout=rclpy.duration.Duration(seconds=0.05 if stamp is not None else 0.0))
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'[TF] {target}<-{source} failed: {e}', throttle_duration_sec=2.0)
            return None
        return mat(ts.transform.translation, ts.transform.rotation)

    def robot_pose(self):
        """Pelvis (x, y, yaw) in the odom frame, latest."""
        T = self.tf(self.p.odom_frame, self.p.pelvis_frame)
        return None if T is None else (T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))

    # ---------------- perception ----------------

    def on_pose(self, msg):
        self.n_rx += 1
        if self.latched or self.phase not in (Phase.SCAN, Phase.ROTATE, Phase.APPROACH, Phase.LATCH):
            return
        t = stamp_s(msg.header.stamp)
        if self.phase is Phase.LATCH and t < self.latch_t0:
            return                                                   # image taken before the robot stood still
        p, q = msg.pose.position, msg.pose.orientation
        v = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(map(math.isfinite, v)) or np.linalg.norm(v[:3]) > self.p.max_range:
            return self.reject('pose is NaN or further than max_range')
        T_odom = self.tf(self.p.odom_frame, msg.header.frame_id, msg.header.stamp)
        if T_odom is None:
            self.n_tf_fail += 1
            return
        T = T_odom @ mat(p, q)                                       # table in the fixed frame
        u = T[:2, 1]                                                 # long axis (mesh Y), horizontal part
        if np.hypot(*u) < self.p.min_horizontal_y:
            return self.reject('table long axis is vertical')
        self.samples.append((t, T[0, 3], T[1, 3], math.atan2(u[1], u[0])))
        self.n_acc += 1
        self.aim_head(msg, t)
        if self.p.log_poses:
            self.get_logger().info(f'[POSE] odom ({T[0, 3]:.3f},{T[1, 3]:.3f}) axis {math.degrees(self.samples[-1][3]):+.1f} deg')

    def reject(self, why):
        self.n_rej += 1
        self.get_logger().warn(f'[POSE] rejected: {why}', throttle_duration_sec=2.0)

    def window(self):
        """Poses taken within window_s before the newest one."""
        t0 = self.samples[-1][0] - self.p.window_s
        return [s for s in self.samples if s[0] >= t0]

    def estimate(self):
        return fuse(self.window()) if self.samples else None

    def stable(self):
        """(estimate, drift m, drift rad, n) once the poses since the stop cover window_s, are enough and the first
        and second half of the window agree. A jump (new estimation) or drift pulls one median away from the other;
        plain jitter does not, whatever its size."""
        if not self.samples or self.samples[-1][0] - self.samples[0][0] < self.p.window_s:
            return None
        w = self.window()
        if len(w) < self.p.latch_samples:
            return None
        a, b = fuse(w[:len(w) // 2]), fuse(w[len(w) // 2:])
        dp, dy = math.hypot(a[0] - b[0], a[1] - b[1]), abs(0.5 * wrap(2 * (a[2] - b[2])))
        return (fuse(w), dp, dy, len(w)) if dp <= self.p.latch_pos_tol and dy <= self.p.latch_yaw_tol else None

    # ---------------- geometry ----------------

    def geometry(self, est, r):
        """Table estimate (odom) and pelvis pose r -> (goal x, goal y in the pelvis frame, heading error to face
        the table, pelvis -> table near face distance)."""
        c, h = np.array(est[:2]), self.p.table_depth / 2
        n = np.array([-math.sin(est[2]), math.cos(est[2])])        # table normal
        d = np.array(r[:2]) - c
        if n @ d < 0:
            n = -n                                                   # the robot's side
        goal = c + (h + self.p.place_standoff) * n + self.p.lateral_offset * np.array([n[1], -n[0]])
        dx, dy = goal - r[:2]
        co, si = math.cos(r[2]), math.sin(r[2])
        return co * dx + si * dy, -si * dx + co * dy, wrap(math.atan2(-n[1], -n[0]) - r[2]), float(n @ d) - h

    # ---------------- head ----------------

    def on_joint_states(self, msg):
        if self.p.head_joint not in msg.name:
            return
        q, t = msg.position[msg.name.index(self.p.head_joint)], stamp_s(msg.header.stamp)
        if self.head_hist and t < self.head_hist[-1][0]:
            self.head_hist.clear()                                   # sim time went back (Stop -> Play)
        if math.isfinite(q):
            self.head_hist.append((t, q))

    def aim_head(self, msg, t):
        """Head angle at image time + the table centre's vertical angle off the optical axis: mount tilt and
        camera TF errors cancel, tracker latency is compensated."""
        T_cam = self.tf(self.p.camera_frame, msg.header.frame_id, msg.header.stamp)
        if not self.p.head_enable or T_cam is None or not self.head_hist:
            return
        c = msg.pose.position
        pc = T_cam @ (c.x, c.y, c.z, 1.0)                            # camera frame: x right, y down, z forward
        if pc[2] < 0.05:
            return
        self.head_err = math.atan2(pc[1], pc[2])                     # + = table below the image centre
        ts, qs = zip(*self.head_hist)
        self.head_target = float(np.interp(t, ts, qs)) + self.p.head_pitch_sign * self.p.head_gain * self.head_err

    def publish_head(self, freeze=False):
        if not self.p.head_enable:
            return
        if not freeze and self.head_target is not None:
            tgt = min(max(self.head_target, self.p.head_pitch_min), self.p.head_pitch_max)
            if self.head_cmd is None or abs(tgt - self.head_cmd) >= self.p.head_deadband:
                if self.head_cmd is None:
                    self.get_logger().info(f'[HEAD] aiming at the table: {math.degrees(tgt):+.1f} deg')
                self.head_cmd = tgt
        if self.head_cmd is not None:
            self.pub_head.publish(Float64(data=float(self.head_cmd)))

    def release_head(self, why):
        if self.p.head_enable:
            self.pub_head.publish(Float64(data=float('nan')))        # NaN: the controller slews the head back
            if self.head_cmd is not None:
                self.get_logger().info(f'[HEAD] released ({why})')
        self.head_cmd = None

    # ---------------- services ----------------

    def on_start(self, req, resp):
        self.get_logger().info(f'[SRV] start requested | phase={self.phase.value}')
        resp.success = False
        if self.phase in RUNNING:
            resp.message = f'already running ({self.phase.value}); cancel first'
        elif self.robot_pose() is None:
            try:
                frames = self.tf_buffer.all_frames_as_string()
            except Exception:
                frames = '?'
            resp.message = (f'no TF {self.p.odom_frame} <- {self.p.pelvis_frame}: set -p odom_frame:=<fixed frame>. '
                            f'TF knows: {frames.replace(chr(10), " ")}')
        else:
            self.reset_run()
            self.t_start = self.now()
            self.set_phase(Phase.SCAN)
            resp.success, resp.message = True, f'started: waiting {self.p.min_samples} poses of the table'
        (self.get_logger().info if resp.success else self.get_logger().warn)(f'[SRV] {resp.message}')
        return resp

    def on_cancel(self, req, resp):
        self.get_logger().info(f'[SRV] cancel requested | phase={self.phase.value}')
        self.publish_cmd(0.0, 0.0, 0.0, 'cancel')
        self.release_head('cancel')
        self.set_phase(Phase.IDLE)
        resp.success, resp.message = True, 'Cancelled.'
        return resp

    # ---------------- control ----------------

    def gate_sidestep(self, vy, now):
        """Lateral duty cycle (the policy trips on long sidesteps): v_min for at most sidestep_max_s, then
        sidestep_pause_s of none."""
        if abs(vy) < 1e-6:
            self.side_since = None
            return 0.0
        if now < self.side_pause_until:
            return 0.0
        self.side_since = now if self.side_since is None else self.side_since
        if now - self.side_since >= self.p.sidestep_max_s:
            self.side_pause_until, self.side_since = now + self.p.sidestep_pause_s, None
            self.get_logger().info(f'[SIDESTEP] {self.p.sidestep_max_s}s reached -> forced {self.p.sidestep_pause_s}s pause')
            return 0.0
        return math.copysign(self.p.v_min, vy)

    def control_step(self):
        try:
            self.step()
        except Exception as e:                   # a crashed loop must not leave the last cmd_vel running
            self.get_logger().error(f'[CRASH] {traceback.format_exc()}')
            self.fail(f'internal error: {e!r}')

    def step(self):
        if self.phase is Phase.DONE and self.p.head_hold_after_done:
            self.publish_head(freeze=True)       # keep the head where it ended; publishing stops the controller's timeout
        if self.phase not in RUNNING:
            return
        now, p = self.now(), self.p
        if now - self.t_start > p.nav_timeout_s:
            return self.fail(f'timeout: {now - self.t_start:.0f}s > nav_timeout_s={p.nav_timeout_s}s (phase {self.phase.value})')
        r = self.robot_pose()
        if r is None:                            # no odometry: stand still
            self.n_tf_fail += 1
            self.lost_since = now if self.lost_since is None else self.lost_since
            if now - self.lost_since > p.lost_grace_s:
                return self.fail(f'no TF {p.odom_frame} <- {p.pelvis_frame} for {p.lost_grace_s}s')
            return self.publish_cmd(0.0, 0.0, 0.0, 'no odometry')
        self.lost_since = None
        if self.phase is Phase.SCAN:
            return self.scan_step(now, r)
        if self.phase is Phase.LATCH:
            return self.latch_step(now, r)

        est = self.est if self.latched else self.estimate()
        tx, ty, eyaw, rng = g = self.geometry(est, r)
        self.publish_point(tx, ty)
        self.publish_head(freeze=self.latched)
        if not self.latched and rng <= p.latch_far and self.phase in (Phase.ROTATE, Phase.APPROACH):
            self.tries += 1
            self.t_latch, self.latch_t0 = now, now + p.latch_settle_s
            self.samples.clear()                 # only poses taken after the robot stands still count
            self.get_logger().info(f'[LATCH] try {self.tries}: table face {rng:.2f} m ahead <= {p.latch_far} m '
                                   f'-> standing still, poses count from +{p.latch_settle_s}s')
            self.set_phase(Phase.LATCH)
            return self.publish_cmd(0.0, 0.0, 0.0, 'latch: stop')

        dist, bearing = math.hypot(tx, ty), math.atan2(ty, tx)
        vx = vy = wz = 0.0
        why = self.phase.value
        if self.phase is Phase.ROTATE:
            if dist < p.capture_radius:
                self.set_phase(Phase.FINAL)
            elif abs(bearing) < p.bearing_gate * 0.5:
                self.get_logger().info(f'[PHASE] bearing {math.degrees(bearing):.1f} deg aimed -> approach')
                self.set_phase(Phase.APPROACH)
            else:
                wz = shape(bearing, p.kp_ang, p.w_min, p.w_max, p.bearing_gate * 0.4)
                why = f'rotate_to_goal: bearing {math.degrees(bearing):.1f} deg'
        elif self.phase is Phase.APPROACH:
            if dist < p.capture_radius:
                self.get_logger().info(f'[PHASE] dist {dist:.3f} < capture {p.capture_radius} -> final_align')
                self.set_phase(Phase.FINAL)
            elif abs(bearing) > p.bearing_gate:
                self.get_logger().info(f'[PHASE] bearing {math.degrees(bearing):.1f} deg blew the gate -> re-aim')
                self.set_phase(Phase.ROTATE)
            else:
                vx = shape(tx, p.kp_lin, p.v_min, p.v_max, p.pos_tol)
                vy = self.gate_sidestep(shape(ty, p.kp_lin, p.v_min, p.v_max, p.pos_tol * 2), now)
                w_raw = p.kp_ang * bearing + p.kp_perp * eyaw        # aim at the goal and square to the table
                if abs(bearing) >= p.yaw_tol * 2 or abs(eyaw) >= p.yaw_tol:
                    wz = math.copysign(min(max(abs(w_raw), p.w_min), p.w_max), w_raw)
                why = f'approach: dist {dist:.3f} m bearing={math.degrees(bearing):+.1f} dyaw={math.degrees(eyaw):+.1f}'
        else:                                                        # FINAL: yaw first, then position
            hyst = p.settle_hyst if self.settle > 0 else 1.0
            ok = (abs(tx) < p.pos_tol * hyst, abs(ty) < p.pos_tol * hyst, abs(eyaw) < p.yaw_tol * hyst)
            if all(ok):
                self.settle += 1
                self.publish_cmd(0.0, 0.0, 0.0, f'settle {self.settle}/{p.settle_cycles}')
                if self.settle >= p.settle_cycles:
                    self.succeed(r, g)
                return self.heartbeat(r, g)
            if self.settle:
                self.get_logger().info(f'[ALIGN] settle broken at {self.settle}: dx={tx:.3f} dy={ty:.3f} dyaw={math.degrees(eyaw):.1f} deg')
            self.settle = 0
            if not ok[2]:
                wz = shape(eyaw, p.kp_ang, p.w_min, p.w_max, p.yaw_tol)
                why = f'final_align yaw: {math.degrees(eyaw):.1f} deg'
            else:
                vx = shape(tx, p.kp_lin, p.v_min, p.v_max, p.pos_tol)
                vy = self.gate_sidestep(shape(ty, p.kp_lin, p.v_min, p.v_max, p.pos_tol), now)
                why = f'final_align pos: dx={tx:.3f} dy={ty:.3f}'
        self.publish_cmd(vx, vy, wz, why)
        self.heartbeat(r, g)

    def scan_step(self, now, r):
        if len(self.samples) >= self.p.min_samples:
            self.get_logger().info(f'[SCAN] table found after {now - self.t_start:.1f}s, turned {math.degrees(self.scanned):.0f} deg')
            self.set_phase(Phase.ROTATE)
            return self.publish_cmd(0.0, 0.0, 0.0, 'table found')
        self.scanned += abs(wrap(r[2] - self.yaw_prev)) if self.yaw_prev is not None else 0.0
        self.yaw_prev = r[2]
        if math.degrees(self.scanned) > self.p.scan_max_deg:
            return self.fail(f'table not found after turning {math.degrees(self.scanned):.0f} deg ({self.n_rx} poses '
                             f'received, {self.n_acc} accepted): is /active_target "workstation" and {self.p.pose_topic} published?')
        self.publish_head()
        wz = self.p.scan_dir * self.p.scan_w if now - self.t_start >= self.p.scan_wait_s else 0.0
        self.publish_cmd(0.0, 0.0, wz, f'scan: {len(self.samples)}/{self.p.min_samples} poses')
        self.heartbeat(r, None)

    def latch_step(self, now, r):
        self.publish_cmd(0.0, 0.0, 0.0, 'latch: standing still')
        self.publish_head()
        s = self.stable()
        if s is None:
            if now - self.t_latch > self.p.latch_timeout_s:
                self.fail(f'no stable table pose within {self.p.latch_timeout_s}s of standing still '
                          f'({len(self.samples)} poses since the stop)')
            return self.heartbeat(r, None)
        est, dp, dy, n = s
        rng, p = self.geometry(est, r)[3], self.p
        if rng > p.latch_far + p.latch_tol and self.tries < p.latch_max_tries:
            self.get_logger().info(f'[LATCH] standing still the table face is {rng:.2f} m ahead (> {p.latch_far + p.latch_tol:.2f}): '
                                   f'the live estimate stopped us early -> walking on')
            return self.set_phase(Phase.ROTATE)
        if not p.latch_near - p.latch_tol <= rng <= p.latch_far + p.latch_tol:
            self.get_logger().warn(f'[LATCH] range {rng:.2f} m is outside {p.latch_near}..{p.latch_far} m: latching anyway '
                                   f'(a close table may be partly out of view)')
        self.est, self.latched = est, True
        self.latch_info = dict(latched_range_m=round(rng, 3), latch_tries=self.tries, latch_samples=n,
                               latch_drift_mm=round(dp * 1000, 1), latch_drift_deg=round(math.degrees(dy), 2))
        self.publish_table(est)
        self.get_logger().info(f'[LATCH] table at odom ({est[0]:.3f},{est[1]:.3f}) axis {math.degrees(est[2]):+.1f} deg, face {rng:.2f} m '
                               f'ahead, {n} poses, halves agree within {dp * 1000:.0f} mm / {math.degrees(dy):.1f} deg -> vision ignored, walking on odometry')
        self.set_phase(Phase.ROTATE)

    def succeed(self, r, g):
        self.publish_cmd(0.0, 0.0, 0.0, 'succeeded')
        c = np.array(self.est[:2]) - r[:2]
        co, si = math.cos(r[2]), math.sin(r[2])
        result = dict(outcome='succeeded', face_dist_m=round(g[3], 4), dyaw_deg=round(math.degrees(g[2]), 2),
                      table_in_pelvis_m=[round(co * c[0] + si * c[1], 4), round(-si * c[0] + co * c[1], 4)],
                      goal_error_m=[round(g[0], 4), round(g[1], 4)], **self.latch_info,
                      head_pitch_deg=None if self.head_cmd is None else round(math.degrees(self.head_cmd), 2))
        self.set_phase(Phase.DONE)
        self.publish_result(result)
        if not self.p.head_hold_after_done:
            self.release_head('succeeded')
        self.get_logger().info(f'[DONE] {result}')
        self.log_counters()

    def fail(self, reason):
        last = self.phase.value
        self.publish_cmd(0.0, 0.0, 0.0, f'failed: {reason}')
        self.release_head('failed')
        self.set_phase(Phase.FAILED)
        self.publish_result(dict(outcome='failed', reason=reason, phase=last))
        self.get_logger().error(f'[FAIL] {reason}')
        self.log_counters()

    # ---------------- debug and plumbing ----------------

    def publish_cmd(self, vx, vy, wz, why):
        """The only place that publishes cmd_vel: logs every change and blocks non-finite values."""
        if not all(map(math.isfinite, (vx, vy, wz))):
            self.get_logger().error(f'[CMD] BLOCKED non-finite ({vx},{vy},{wz}) | {self.phase.value} | {why}')
            vx = vy = wz = 0.0
        new = (round(vx, 3), round(vy, 3), round(wz, 3))
        if new != self.last_cmd:
            self.get_logger().info(f'[CMD] vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f} | {self.phase.value} | {why}')
            self.last_cmd = new
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        self.pub_cmd.publish(msg)

    def heartbeat(self, r, g):
        now = self.now()
        if self.last_hb is not None and now - self.last_hb < self.p.log_period_s:
            return
        self.last_hb = now
        est = self.est if self.latched else (self.estimate() if self.samples else None)
        tab = 'table=none' if est is None else f'table@odom=({est[0]:+.2f},{est[1]:+.2f}) axis={math.degrees(est[2]):+.0f}'
        geo = 'goal=n/a' if g is None else (f'face={g[3]:.2f} goal=({g[0]:+.2f},{g[1]:+.2f}) '
                                            f'bearing={math.degrees(math.atan2(g[1], g[0])):+.1f} dyaw={math.degrees(g[2]):+.1f}')
        head = 'head=policy' if self.head_cmd is None else f'head={math.degrees(self.head_cmd):+.1f}'
        age = f'{now - self.samples[-1][0]:.1f}s' if self.samples else 'n/a'
        c = self.last_cmd
        self.get_logger().info(
            f'[HB] {self.phase.value}{" LATCHED" if self.latched else ""} | {tab} | pelvis@odom=({r[0]:+.2f},{r[1]:+.2f},{math.degrees(r[2]):+.0f}) | '
            f'{geo} | {head} | pose age={age} n={len(self.samples)} | cmd=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) | '
            f'rx={self.n_rx} acc={self.n_acc} rej={self.n_rej} tf_fail={self.n_tf_fail}')

    def log_counters(self):
        self.get_logger().info(f'[STATS] poses rx={self.n_rx} accepted={self.n_acc} rejected={self.n_rej} tf_failed={self.n_tf_fail} '
                               f'latch_tries={self.tries}')

    def set_phase(self, phase):
        if phase != self.phase:
            self.get_logger().info(f'Phase -> {phase.value}')
            self.phase = phase
            self.settle = 0
            self.publish_status()

    def publish_status(self):
        self.pub_status.publish(String(data='running' if self.phase in RUNNING else self.phase.value))

    def publish_result(self, d):
        self.pub_result.publish(String(data=json.dumps(d)))

    def publish_point(self, tx, ty):
        m = PointStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), self.p.pelvis_frame
        m.point.x, m.point.y, m.point.z = float(tx), float(ty), 0.0
        self.pub_point.publish(m)

    def publish_table(self, est):
        m = PoseStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), self.p.odom_frame
        m.pose.position.x, m.pose.position.y = float(est[0]), float(est[1])
        m.pose.orientation.z, m.pose.orientation.w = math.sin(est[2] / 2), math.cos(est[2] / 2)   # long axis = +x of the pose
        self.pub_table.publish(m)


def main():
    rclpy.init()
    node = H2ArrivalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd(0.0, 0.0, 0.0, 'shutdown')
            node.pub_head.publish(Float64(data=float('nan')))
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
