#!/usr/bin/env python3
"""
H2 Arrival Navigator: walk to the workstation (packing table) while carrying the KLT box.

Same interface as h2_local_navigator.py (Trigger services ~/start and ~/cancel, latched ~/status and
~/result, cmd_vel, head_pitch_cmd, same velocity shaping and debug logs), built for a table that the
camera loses near the end of the walk (box in the hands, table filling the view). The robot comes in along the
table's BISECTOR, the line through the table centre normal to its long side:

  SCAN             no table yet: stand still scan_wait_s, then turn until the table is WELL IN VIEW (poses whose table
                   centre is too far off the camera axis are ignored: the table is only partly in the image and its pose
                   is wrong), then stand still until those poses agree
  ROTATE, APPROACH leg "meet": walk to the meet point on the bisector, meet_dist (3.0 m) from the table face,
                   steering by bearing only. Within line_tol (0.4 m) of the bisector: leg "line": turn to the table
                   and walk in along the bisector (the aim point is the end point)
  LATCH            on the bisector, face <= latch_far (2.0 m) away (real distance to the face, not to its plane):
                   STOP, turn on the spot to face the table if needed, wait latch_settle_s, take the poses stamped
                   after that until they agree, freeze their median. The window latch_near..latch_far (1.7..2.0 m)
                   is where the whole 2.47 m table still fits the 82 deg field of view
  ROTATE, APPROACH, FINAL_ALIGN
                   on the frozen table and the pelvis pose from TF (odometry): straight in along the bisector to
                   place_standoff from the table face, square to it. Vision is ignored from the latch on (the live
                   pose is only logged, next to the latched one). If it pushes for block_s without getting closer
                   (the table stops the body), within block_accept of the end point it succeeds with a warning
                   (result "blocked": true), anywhere else it fails at once with the reason

Only poses with the table centre close enough to the camera axis that the whole table fits the image count
(view_limit); while it is further off the estimate stays frozen and the robot walks on odometry. If the table
should be in view but no pose arrived for stale_pose_s, the robot stands still (and turns to it) until poses return.

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
import signal
import time
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
    table_length=2.474,         # m, long side (mesh Y)
    place_standoff=0.10,        # m, pelvis -> table near face at the end. The carry pose of H2_pickup_sequencer holds the box
                                #    front 0.35 m ahead of the pelvis (grip x 0.20 + half the 0.30 m box): 0.25 m over the table at 0.10.
                                #    The body is stopped by the table at about 0.15 m (Nov 23 run, see block_*)
    lateral_offset=0.0,         # m, stand this far to the robot's LEFT of the table centre (the bisector moves with it)
    # approach: meet the bisector, then walk in along it
    meet_dist=3.0,              # m, meet point on the bisector, from the near face (keep it above latch_far)
    line_tol=0.4,               # m, this close to the bisector counts as on it: turn to the table and walk in
    lookahead=1.0,              # m, on the bisector the aim point is this far up the line (pure pursuit): the path
                                #    converges onto the line instead of meeting it only at the end point
    aim_blend=0.6,              # m, closer than this to the aim point the bearing fades out (it is noise there)
    # latch
    latch_far=2.0, latch_near=1.7,   # m, pelvis -> table near face
    latch_tol=0.15,             # m, stationary range beyond latch_far + tol: walk on and latch again
    latch_max_tries=3,
    latch_settle_s=1.5,         # s standing still before the poses count (image stamp, not arrival time)
    window_s=1.0, latch_samples=5,   # poses taken in the last window_s: their median is the estimate; >= latch_samples
    latch_pos_tol=0.05, latch_yaw_tol=math.radians(3),   # m, rad: the two halves of the window must agree this well
    latch_timeout_s=20.0,       # s (tracker gaps of 2-9 s were seen)
    # scan
    scan_wait_s=3.0, scan_w=0.70, scan_dir=1.0, scan_max_deg=400.0, min_samples=3,
    scan_center_deg=8.0,        # the scan turns until the table centre is this close to the camera axis, then stands still
    scan_settle_s=8.0,          # s standing still with the table in view; after that go on with what there is
    # which poses count: the table centre must be within view_limit() of the camera axis (a table at the image edge is
    # only partly visible and its pose comes out wrong: axis +76 deg instead of 0, centre 0.8 m off, Nov 23)
    fov_half_deg=41.0, view_margin_deg=4.0, view_min_deg=10.0,
    stale_pose_s=1.5,           # s without a pose although the table should be in view: stand still ...
    lost_timeout_s=12.0,        # s ... and fail after this long
    unmerge_tol=0.8,            # m, off the bisector by more than this (a corrected estimate): meet leg again
    # blocked: after the latch the robot pushes (cmd_vx/vy) for block_s without getting block_progress closer to the goal
    # (the table is in the way): FINAL within block_accept of the goal -> succeeded with a warning, anywhere else -> fail
    block_s=4.0, block_progress=0.03, block_accept=0.10,
    # pose sanity
    max_range=9.0, min_horizontal_y=0.20,
    # motion: same names and defaults as h2_local_navigator (tuned there for the no-box policy)
    v_min=0.17, v_max=0.40, w_min=0.65, w_max=0.90, kp_lin=1.0, kp_ang=1.5, kp_perp=0.6,
    sidestep_max_s=2.0, sidestep_pause_s=2.0,
    pos_tol=0.05, yaw_tol=math.radians(4), capture_radius=0.15, bearing_gate=math.radians(30),
    settle_cycles=20, settle_hyst=1.4, nav_timeout_s=180.0, phase_pause_s=1.0, lost_grace_s=5.0,
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
        self.merged, self.n_ref = False, None      # reached the bisector / the table side (unit normal) chosen first
        self.pause_until = 0.0                     # standing still until then (a pause after every phase change)
        self.live, self.face_yaw = None, 0.0       # newest pose after the latch (logged only) / heading (odom) that faces the table
        self.t_start = self.t_latch = self.latch_t0 = self.lost_since = self.side_since = self.yaw_prev = None
        self.side_pause_until, self.settle, self.scanned = 0.0, 0, 0.0
        self.push_prev, self.push_best, self.push_s = None, math.inf, 0.0   # blocked check: last push time / closest to the goal / s pushing since
        self.head_target = self.head_cmd = self.head_err = None
        self.last_cmd, self.last_hb = (0.0, 0.0, 0.0), None
        self.n_rx = self.n_acc = self.n_rej = self.n_tf_fail = self.n_edge = 0
        self.lost_t0 = self.lost_n0 = self.scan_t0 = None   # table lost since / poses then / table in view since (scan)
        self.edge = None                           # (image time, x, y in odom) of the newest pose ignored for being too far off the axis

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
        if self.phase not in RUNNING:
            return
        t = stamp_s(msg.header.stamp)
        if self.phase is Phase.LATCH and t < self.latch_t0:
            return                                                   # image taken before the robot stood still
        p, q = msg.pose.position, msg.pose.orientation
        v = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(map(math.isfinite, v)) or np.linalg.norm(v[:3]) > self.p.max_range:
            return None if self.latched else self.reject('pose is NaN or further than max_range')
        if not self.latched and not self.in_view(p.x, p.y):
            self.n_edge += 1
            self.aim_head(msg, t)                                    # the head only needs the elevation, which survives
            T_odom = self.tf(self.p.odom_frame, msg.header.frame_id, msg.header.stamp)
            if T_odom is not None:                                   # where the table is seen, to turn toward it (never to steer by)
                c = T_odom @ np.array([p.x, p.y, p.z, 1.0])
                self.edge = (t, float(c[0]), float(c[1]))
            self.get_logger().info(f'[VIEW] pose ignored: table centre {math.degrees(math.atan2(p.y, p.x)):+.0f} deg off the camera axis '
                                   f'at {math.hypot(p.x, p.y):.1f} m, more than {self.view_limit(math.hypot(p.x, p.y)):.0f} deg: the '
                                   f'table is only partly in the image', throttle_duration_sec=5.0)
            return
        T_odom = self.tf(self.p.odom_frame, msg.header.frame_id, msg.header.stamp)
        if T_odom is None:
            self.n_tf_fail += 1
            return
        T = T_odom @ mat(p, q)                                       # table in the fixed frame
        u = T[:2, 1]                                                 # long axis (mesh Y), horizontal part
        if np.hypot(*u) < self.p.min_horizontal_y:
            return None if self.latched else self.reject('table long axis is vertical')
        s = (t, float(T[0, 3]), float(T[1, 3]), math.atan2(u[1], u[0]))
        if self.latched:
            self.live = s                                            # logged next to the latched pose, never used
            return
        self.samples.append(s)
        self.n_acc += 1
        self.aim_head(msg, t)
        if self.p.log_poses:
            self.get_logger().info(f'[POSE] odom ({s[1]:.3f},{s[2]:.3f}) axis {math.degrees(s[3]):+.1f} deg')

    def view_limit(self, dist):
        """Largest angle (deg) between the camera axis and the table centre, `dist` m away, at which the whole table
        (worst case: seen broadside) is still inside the image; at least view_min_deg."""
        p = self.p
        return max(p.fov_half_deg - p.view_margin_deg - math.degrees(math.atan2(p.table_length / 2, dist)), p.view_min_deg)

    def last_seen(self, r, now):
        """(bearing rad off the camera axis, age s) of where the table was last seen: the newest of the accepted poses and
        the ones ignored for being too far off the axis (those are only good enough to turn toward)."""
        t_acc = self.samples[-1][0] if self.samples else -1e9
        t_edge = self.edge[0] if self.edge else -1e9
        if t_edge > t_acc:
            cx, cy = self.edge[1:]
        else:
            cx, cy = self.estimate()[:2]
        return wrap(math.atan2(cy - r[1], cx - r[0]) - r[2]), now - max(t_acc, t_edge)

    def in_view(self, x, y):
        """Table centre (x forward, y left, pelvis frame) close enough to the camera axis for a good pose."""
        return abs(math.degrees(math.atan2(y, x))) <= self.view_limit(math.hypot(x, y))

    def reject(self, why):
        self.n_rej += 1
        self.get_logger().warn(f'[POSE] rejected: {why}', throttle_duration_sec=2.0)

    def window(self):
        """Poses taken within window_s before the newest one."""
        t0 = self.samples[-1][0] - self.p.window_s
        return [s for s in self.samples if s[0] >= t0]

    def estimate(self):
        return fuse(self.window()) if self.samples else None

    @staticmethod
    def drift(w):
        """(m, rad) by which the median of the first half of the poses w differs from the second half's."""
        a, b = fuse(w[:len(w) // 2]), fuse(w[len(w) // 2:])
        return math.hypot(a[0] - b[0], a[1] - b[1]), abs(0.5 * wrap(2 * (a[2] - b[2])))

    def stable(self):
        """(estimate, drift m, drift rad, n) once the poses since the stop cover window_s, are enough and the first
        and second half of the window agree. A jump (new estimation) or drift pulls one median away from the other;
        plain jitter does not, whatever its size."""
        if not self.samples or self.samples[-1][0] - self.samples[0][0] < self.p.window_s:
            return None
        w = self.window()
        if len(w) < self.p.latch_samples:
            return None
        dp, dy = self.drift(w)
        return (fuse(w), dp, dy, len(w)) if dp <= self.p.latch_pos_tol and dy <= self.p.latch_yaw_tol else None

    def latch_status(self):
        """Why the latch is not done yet, with the numbers stable() looks at."""
        p = self.p
        if not self.samples:
            return 'no poses since the stop'
        w = self.window()
        dp, dy = self.drift(w) if len(w) > 1 else (0.0, 0.0)
        return (f'{len(w)}/{p.latch_samples} poses, {self.samples[-1][0] - self.samples[0][0]:.1f}/{p.window_s}s since the '
                f'first, halves differ {dp * 1000:.0f}/{p.latch_pos_tol * 1000:.0f} mm {math.degrees(dy):.1f}/'
                f'{math.degrees(p.latch_yaw_tol):.0f} deg')

    # ---------------- geometry ----------------

    @property
    def aim(self):
        """(stand, ahead) for geometry(): the meet point first, then the end point with a pure-pursuit lookahead."""
        p = self.p
        return (p.place_standoff, p.lookahead) if self.merged else (p.meet_dist, 0.0)

    def geometry(self, est, r, stand, ahead=0.0):
        """Table estimate (odom), pelvis pose r and where to stand (`stand` m from the near face along the bisector,
        the line through the table centre normal to the long side; with `ahead` > 0 the aim point is that far up the
        line from the robot, but never closer to the table than `stand`) -> NS(
          tx, ty  the aim point in the pelvis frame,   eyaw  heading error to face the table,
          rng     distance from the pelvis to the near FACE (not to its plane: a tilted estimate moves the plane
                  next to a table that is metres away),   off  how far the robot stands to the left of the bisector).
        The table side is the robot's; within 0.5 m of the table's long-axis plane it keeps the side chosen before."""
        p, c, h = self.p, np.array(est[:2]), self.p.table_depth / 2
        u = np.array([math.cos(est[2]), math.sin(est[2])])           # long axis
        n = np.array([-u[1], u[0]])                                  # normal
        d = np.array(r[:2]) - c
        ref = self.n_ref if self.n_ref is not None and abs(n @ d) <= 0.5 else d
        n = -n if n @ ref < 0 else n
        self.n_ref = n
        t = np.array([n[1], -n[0]])                                  # the robot's left when it faces the table
        s = max(stand, float(n @ d) - h - ahead) if ahead else stand
        goal = c + p.lateral_offset * t + (h + s) * n
        dx, dy = goal - r[:2]
        co, si = math.cos(r[2]), math.sin(r[2])
        return NS(tx=float(co * dx + si * dy), ty=float(-si * dx + co * dy),
                  eyaw=wrap(math.atan2(-n[1], -n[0]) - r[2]), off=float(t @ d) - p.lateral_offset,
                  rng=math.hypot(max(float(n @ d) - h, 0.0), max(abs(float(u @ d)) - p.table_length / 2, 0.0)))

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
        """Head angle at image time + the table centre's elevation off the optical-axis plane (what is left to fix
        once the robot faces the table): mount tilt and camera TF errors cancel, tracker latency is compensated.
        atan2(y, z) would blow up for a table far to the side (z -> 0) and pin the head at its limit."""
        T_cam = self.tf(self.p.camera_frame, msg.header.frame_id, msg.header.stamp)
        if not self.p.head_enable or T_cam is None or not self.head_hist:
            return
        c = msg.pose.position
        x, y, z = (T_cam @ (c.x, c.y, c.z, 1.0))[:3]                 # camera frame: x right, y down, z forward
        self.head_err = math.atan2(y, math.hypot(x, z))              # + = table below the image centre
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

    def stop_robot(self, repeats=5, pause=0.05):
        """Zero cmd_vel and release the head, repeated: a velocity controller keeps the last command, so one lost
        sample would leave the robot turning."""
        for _ in range(repeats):
            try:
                self.publish_cmd(0.0, 0.0, 0.0, 'shutdown')
                if self.p.head_enable:
                    self.pub_head.publish(Float64(data=float('nan')))
            except Exception as e:
                self.get_logger().error(f'[SHUTDOWN] stop command NOT sent: {e!r}')
                return
            time.sleep(pause)

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
        if now < self.pause_until:               # between phases: stand still so the robot settles
            return self.publish_cmd(0.0, 0.0, 0.0, f'pause {self.pause_until - now:.1f}s after the phase change')
        if self.phase is Phase.SCAN:
            return self.scan_step(now, r)
        if self.phase is Phase.LATCH:
            return self.latch_step(now, r)

        est = self.est if self.latched else self.estimate()
        if not self.latched and self.lost_guard(now, r, est):
            return
        g = self.geometry(est, r, *self.aim)
        if not self.merged and abs(g.off) <= p.line_tol:             # (a latched robot has merged)
            self.merged = True
            self.get_logger().info(f'[LEG] on the bisector ({g.off:+.2f} m off, face {g.rng:.2f} m away) '
                                   f'-> turn to the table, walk in along it')
            g = self.geometry(est, r, *self.aim)                     # the aim point is now up the line
        elif self.merged and not self.latched and abs(g.off) > p.unmerge_tol:
            self.merged = False                                      # the estimate changed: the line is somewhere else
            self.get_logger().info(f'[LEG] {g.off:+.2f} m off the bisector (more than {p.unmerge_tol} m): meet leg again')
            g = self.geometry(est, r, *self.aim)
        tx, ty, eyaw = g.tx, g.ty, g.eyaw
        self.publish_point(tx, ty)
        self.publish_head(freeze=self.latched)
        if not self.latched and self.merged and g.rng <= p.latch_far and self.phase in (Phase.ROTATE, Phase.APPROACH):
            self.tries += 1
            self.t_latch, self.latch_t0 = now, now + p.latch_settle_s
            self.face_yaw = r[2] + eyaw          # the heading that faces the table (odom): LATCH turns to it if needed
            self.samples.clear()                 # only poses taken after the robot stands still count
            self.get_logger().info(f'[LATCH] try {self.tries}: table face {g.rng:.2f} m away <= {p.latch_far} m, {g.off:+.2f} m off '
                                   f'the bisector -> standing still, poses count from +{p.latch_settle_s}s')
            self.set_phase(Phase.LATCH)
            return self.publish_cmd(0.0, 0.0, 0.0, 'latch: stop')

        dist = math.hypot(tx, ty)
        bearing = math.atan2(ty, tx) * (1.0 if dist >= p.aim_blend else dist / p.aim_blend)   # fades out close to the aim point
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
                sq = eyaw if self.merged else 0.0                    # square to the table only on the bisector, not on the way to it
                w_raw = p.kp_ang * bearing + p.kp_perp * sq          # aim at the goal
                if (abs(bearing) >= p.yaw_tol * 2 or abs(sq) >= p.yaw_tol) and abs(w_raw) >= p.w_min / 2:
                    wz = math.copysign(min(max(abs(w_raw), p.w_min), p.w_max), w_raw)   # (the two terms cancel: no +-w_min kicks)
                why = f'approach: dist {dist:.3f} m bearing={math.degrees(bearing):+.1f} dyaw={math.degrees(eyaw):+.1f}'
        else:                                                        # FINAL: yaw first, then position
            if not self.latched:
                return self.fail(f'at the end point without a latched table pose: never on the bisector within {p.latch_far} m '
                                 f'of the face ({self.n_acc} poses accepted)')
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
        if self.latched and (vx or vy):                              # pushing on odometry: is the robot getting anywhere?
            if self.pushing(now, dist):
                return self.blocked(r, g, dist)
        else:
            self.push_prev = None
        self.publish_cmd(vx, vy, wz, why)
        self.heartbeat(r, g)

    def pushing(self, now, dist):
        """True when the robot has pushed for block_s without getting block_progress closer to the goal."""
        p = self.p
        dt = min(now - self.push_prev, 0.5) if self.push_prev is not None else 0.0
        self.push_prev = now
        if dist < self.push_best - p.block_progress:
            self.push_best, self.push_s = dist, 0.0
        else:
            self.push_s += dt
        return self.push_s >= p.block_s

    def blocked(self, r, g, dist):
        p = self.p
        why = (f'pushing for {p.block_s:.0f}s without getting closer: {dist:.2f} m from the goal (dx={g.tx:.3f} dy={g.ty:.3f} '
               f'dyaw={math.degrees(g.eyaw):.1f} deg), pelvis {g.rng:.2f} m from the table face')
        if abs(g.tx) <= p.block_accept and abs(g.ty) <= 2 * p.pos_tol and abs(g.eyaw) <= 2 * p.yaw_tol:
            self.get_logger().warn(f'[BLOCKED] {why} -> something stops the robot (the table?): accepting this pose')
            return self.succeed(r, g, blocked=True)
        self.fail(f'blocked: {why} (more than {p.block_accept} m from the goal, or not squared up: the table or something else '
                  f'is in the way, or the robot does not follow vx)')

    def scan_step(self, now, r):
        p = self.p
        self.scanned += abs(wrap(r[2] - self.yaw_prev)) if self.yaw_prev is not None else 0.0
        self.yaw_prev = r[2]
        if math.degrees(self.scanned) > p.scan_max_deg:
            return self.fail(f'table not found after turning {math.degrees(self.scanned):.0f} deg ({self.n_rx} poses '
                             f'received, {self.n_acc} accepted, {self.n_edge} ignored as too far off the camera axis): is '
                             f'/active_target "workstation" and {p.pose_topic} published?')
        self.publish_head()
        wz, status = 0.0, f'scan: {len(self.samples)} poses'
        if self.samples and now - self.samples[-1][0] <= p.stale_pose_s + p.window_s:   # the table has been well in view
            self.scan_t0 = now if self.scan_t0 is None else self.scan_t0
            est = self.estimate()
            b = wrap(math.atan2(est[1] - r[1], est[0] - r[0]) - r[2])      # where the table centre is now (odometry)
            settled = self.stable() is not None
            if abs(math.degrees(b)) > p.scan_center_deg:                  # the turn overshot (or the table is off to a side): centre it
                wz, status = shape(b, p.kp_ang, p.w_min, p.w_max, math.radians(p.scan_center_deg)), f'scan: centring the table, {math.degrees(b):+.0f} deg'
                self.scan_t0 = now
            elif settled or (now - self.scan_t0 > p.scan_settle_s and len(self.samples) >= p.min_samples):
                self.get_logger().info(
                    f'[SCAN] table found after {now - self.t_start:.1f}s, turned {math.degrees(self.scanned):.0f} deg, centre '
                    f'{math.degrees(b):+.0f} deg off the camera axis'
                    + ('' if settled else f' (poses never settled in {p.scan_settle_s}s: {self.latch_status()})'))
                self.set_phase(Phase.ROTATE)
                return self.publish_cmd(0.0, 0.0, 0.0, 'table found')
            else:
                status = 'scan: table in view, waiting for its pose to settle: ' + self.latch_status()
        else:
            self.scan_t0 = None
            if self.edge and now - self.edge[0] <= p.stale_pose_s + p.window_s:     # seen, but too far off the axis: turn to it
                b = self.last_seen(r, now)[0]
                wz, status = math.copysign(p.scan_w, b), f'scan: table at the image edge ({math.degrees(b):+.0f} deg), turning to it'
            else:
                wz = p.scan_dir * p.scan_w if now - self.t_start >= p.scan_wait_s else 0.0
        self.publish_cmd(0.0, 0.0, wz, status)
        self.heartbeat(r, None, status)

    def lost_guard(self, now, r, est):
        """True while standing still because the table should be in the image (by the estimate and the odometry) but no
        pose came for stale_pose_s: the tracker lost it, or the estimate is wrong. The robot stops (it does not walk on
        a frozen estimate) and turns to where the table should be, until poses return; fails after lost_timeout_s.
        A table that is out of view on purpose (the meet leg walks with it off to the side) is not lost."""
        p = self.p
        dx, dy = est[0] - r[0], est[1] - r[1]
        b, age = wrap(math.atan2(dy, dx) - r[2]), now - self.samples[-1][0]      # by the estimate
        if self.lost_t0 is not None and self.n_acc - self.lost_n0 >= p.min_samples:
            self.get_logger().info(f'[LOST] table seen again after {now - self.lost_t0:.1f}s')
            self.lost_t0 = None
        if self.lost_t0 is None:
            if age <= p.stale_pose_s or abs(math.degrees(b)) > self.view_limit(math.hypot(dx, dy)):
                return False
            self.lost_t0, self.lost_n0 = now, self.n_acc
            self.get_logger().warn(f'[LOST] no pose for {age:.1f}s although the table should be in view '
                                   f'({math.degrees(b):+.0f} deg off the camera axis): standing still, turning to it')
        if now - self.lost_t0 > p.lost_timeout_s:
            self.fail(f'table lost: no pose for {age:.0f}s ({math.degrees(b):+.0f} deg off the camera axis by the estimate, '
                      f'{self.n_acc} poses accepted, {self.n_edge} ignored as too far off the axis)')
            return True
        b = self.last_seen(r, now)[0]                                # ... but turn to where it was last SEEN (ignored poses too)
        self.publish_head()
        self.publish_cmd(0.0, 0.0, shape(b, p.kp_ang, p.w_min, p.w_max, 2 * p.yaw_tol), f'lost: no pose for {age:.1f}s')
        self.heartbeat(r, self.geometry(est, r, *self.aim), f'lost the table {now - self.lost_t0:.0f}/{p.lost_timeout_s:.0f}s')
        return True

    def latch_step(self, now, r):
        p = self.p
        self.publish_head()
        if (not self.samples and self.edge and now - self.edge[0] <= p.stale_pose_s + p.window_s
                and now - self.t_latch > p.latch_settle_s + p.stale_pose_s):
            self.face_yaw = r[2] + self.last_seen(r, now)[0]   # only ignored (off-axis) poses: the table is not where the estimate
            #                                                    put it. Turn to what the camera sees, so that its poses count
        turn, s = wrap(self.face_yaw - r[2]), None
        if abs(turn) > 2 * p.yaw_tol:            # stopped on the bisector but not facing the table: turn on the spot first
            self.latch_t0 = now + p.latch_settle_s    # the poses count from when it stands still and faces the table
            self.samples.clear()
            self.publish_cmd(0.0, 0.0, shape(turn, p.kp_ang, p.w_min, p.w_max, p.yaw_tol), f'latch: turn to the table {math.degrees(turn):+.1f} deg')
            status = f'turning to the table, {math.degrees(turn):+.0f} deg to go'
        else:
            self.publish_cmd(0.0, 0.0, 0.0, 'latch: standing still')
            s, status = self.stable(), self.latch_status()
        if s is None:
            if now - self.t_latch > p.latch_timeout_s:
                self.fail(f'no stable table pose within {p.latch_timeout_s}s of standing still ({status})')
            live = self.estimate() if self.samples else None
            return self.heartbeat(r, live and self.geometry(live, r, *self.aim), 'latch: ' + status)
        est, dp, dy, n = s
        g = self.geometry(est, r, *self.aim)
        if g.rng > p.latch_far + p.latch_tol and self.tries < p.latch_max_tries:
            self.get_logger().info(f'[LATCH] standing still the table face is {g.rng:.2f} m away (> {p.latch_far + p.latch_tol:.2f}): '
                                   f'the live estimate stopped us early -> walking on')
            return self.set_phase(Phase.ROTATE)
        if not (p.latch_near - p.latch_tol <= g.rng <= p.latch_far + p.latch_tol and abs(g.off) <= p.line_tol):
            self.get_logger().warn(f'[LATCH] face {g.rng:.2f} m away and {g.off:+.2f} m off the bisector is outside {p.latch_near}..'
                                   f'{p.latch_far} m and +-{p.line_tol} m: latching anyway (a close table may be partly out of view)')
        self.est, self.latched = est, True
        self.latch_info = dict(latched_range_m=round(g.rng, 3), latched_off_m=round(g.off, 3), latch_tries=self.tries,
                               latch_samples=n, latch_drift_mm=round(dp * 1000, 1), latch_drift_deg=round(math.degrees(dy), 2))
        self.publish_table(est)
        self.get_logger().info(f'[LATCH] table at odom ({est[0]:.3f},{est[1]:.3f}) axis {math.degrees(est[2]):+.1f} deg, face {g.rng:.2f} m '
                               f'away, {g.off:+.2f} m off the bisector, {n} poses, halves agree within {dp * 1000:.0f} mm / '
                               f'{math.degrees(dy):.1f} deg -> vision ignored, walking on odometry')
        self.set_phase(Phase.ROTATE)

    def succeed(self, r, g, blocked=False):
        self.publish_cmd(0.0, 0.0, 0.0, 'succeeded')
        c = np.array(self.est[:2]) - r[:2]
        co, si = math.cos(r[2]), math.sin(r[2])
        result = dict(outcome='succeeded', face_dist_m=round(g.rng, 4), dyaw_deg=round(math.degrees(g.eyaw), 2),
                      table_in_pelvis_m=[round(float(co * c[0] + si * c[1]), 4), round(float(-si * c[0] + co * c[1]), 4)],
                      goal_error_m=[round(g.tx, 4), round(g.ty, 4)], **self.latch_info,
                      head_pitch_deg=None if self.head_cmd is None else round(math.degrees(self.head_cmd), 2))
        if blocked:
            result['blocked'] = True                         # stopped by the table, not by reaching the goal: goal_error_m says how far short
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

    def heartbeat(self, r, g, extra=''):
        """One status line per log_period_s. g: geometry() of the table estimate (None: no estimate yet)."""
        now = self.now()
        if self.last_hb is not None and now - self.last_hb < self.p.log_period_s:
            return
        self.last_hb = now
        est = self.est if self.latched else (self.estimate() if self.samples else None)
        tab = 'table=none' if est is None else f'table@odom=({est[0]:+.2f},{est[1]:+.2f}) axis={math.degrees(est[2]):+.0f}'
        geo = 'goal=n/a' if g is None else (f'face={g.rng:.2f} off={g.off:+.2f} goal=({g.tx:+.2f},{g.ty:+.2f}) '
                                            f'bearing={math.degrees(math.atan2(g.ty, g.tx)):+.1f} dyaw={math.degrees(g.eyaw):+.1f}')
        if self.latched and self.live:                                # the camera's opinion of the frozen table
            dp = math.hypot(self.live[1] - est[0], self.live[2] - est[1])
            extra = (f'live-latched {dp:.2f} m {math.degrees(abs(0.5 * wrap(2 * (self.live[3] - est[2])))):.1f} deg '
                     f'(pose age {now - self.live[0]:.1f}s)')
        leg = ' LATCHED' if self.latched else ('' if self.phase is Phase.SCAN else f' leg={"line" if self.merged else "meet"}')
        head = 'head=policy' if self.head_cmd is None else f'head={math.degrees(self.head_cmd):+.1f}'
        age = f'{now - self.samples[-1][0]:.1f}s' if self.samples else 'n/a'
        c = self.last_cmd
        self.get_logger().info(
            f'[HB] {self.phase.value}{leg} | {tab} | pelvis@odom=({r[0]:+.2f},{r[1]:+.2f},{math.degrees(r[2]):+.0f}) | '
            f'{geo} | {head} | pose age={age} n={len(self.samples)} | cmd=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) | '
            f'rx={self.n_rx} acc={self.n_acc} rej={self.n_rej} edge={self.n_edge} tf_fail={self.n_tf_fail}' + (f' | {extra}' if extra else ''))

    def log_counters(self):
        self.get_logger().info(f'[STATS] poses rx={self.n_rx} accepted={self.n_acc} rejected={self.n_rej} off_axis_ignored={self.n_edge} tf_failed={self.n_tf_fail} '
                               f'latch_tries={self.tries}')

    def set_phase(self, phase):
        if phase != self.phase:
            self.get_logger().info(f'Phase -> {phase.value}')
            self.phase = phase
            self.settle = 0
            self.push_prev, self.push_best, self.push_s = None, math.inf, 0.0
            if phase in RUNNING and phase is not Phase.SCAN and self.p.phase_pause_s > 0:
                self.pause_until = self.now() + self.p.phase_pause_s
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
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)     # rclpy's own handlers kill the context before the stop
    signal.signal(signal.SIGTERM, signal.default_int_handler)      # kill / closing the terminal: same path as Ctrl-C
    node = H2ArrivalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, signal.SIG_IGN)                       # a second Ctrl-C must not cut the stop short
        node.stop_robot()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
