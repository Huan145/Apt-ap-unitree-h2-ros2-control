#!/usr/bin/env python3
"""
H2 Local Navigator Node — verbose debug build.

Same logic as before (stamped TF, EMA filter, rotate/approach/align state
machine, hold-then-fail, global timeout) plus a debug layer:

  - EVERY cmd_vel change is logged immediately:  [CMD] vx vy wz | phase | why
  - Periodic heartbeat (default 2 Hz) with full state: phase, box estimate,
    errors, distance/bearing, pose age, current cmd
  - Perception accounting: received / accepted / TF-failed / rejected
    counters, logged with each heartbeat; every accepted pose can be dumped
    raw-vs-filtered with log_poses:=true
  - Service calls, phase transitions, terminal states all logged (as before)

Debug params:
  log_period_s   heartbeat interval, seconds sim time (default 0.5)
  log_poses      log every accepted pose raw vs filtered (default false —
                 very noisy, enable when debugging perception jitter)

Everything logs at INFO so it shows without changing ROS log levels.
"""

import json
import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

import tf2_ros
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_to_matrix(ts):
    tr, ro = ts.transform.translation, ts.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(ro)
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def wrap_half(a):
    a = wrap(a)
    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi
    return a


class Phase(Enum):
    IDLE = "idle"
    ROTATE_TO_GOAL = "rotate_to_goal"
    APPROACH = "approach"
    FINAL_ALIGN = "final_align"
    HOLD = "hold"
    DONE = "succeeded"
    FAILED = "failed"


class H2LocalNavigator(Node):

    def __init__(self):
        super().__init__("h2_local_navigator")
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time",
                                       rclpy.Parameter.Type.BOOL, True)]
        )

        p = self.declare_parameter
        p("tracking_topic", "/tracking/output")
        p("cmd_vel_topic", "cmd_vel")
        p("pelvis_frame", "pelvis")
        p("camera_frame", "Camera")
        p("control_rate_hz", 20.0)

        p("standoff_x", 0.35)
        p("v_min", 0.17)
        p("v_max", 0.30)
        p("w_min", 0.25)
        p("w_max", 0.50)

        p("kp_lin", 1.0)
        p("sidestep_max_s", 2.0)       # max continuous lateral walking before
                                       # a forced pause (policy trips on long
                                       # sidesteps)
        p("sidestep_pause_s", 2.0)     # forced pause after hitting the max,
                                       # before lateral motion is allowed again
        p("kp_ang", 1.5)
        p("kp_perp", 0.6)              # approach-phase weight on squaring to
                                       # the box face (eyaw) alongside bearing,
                                       # so the robot arrives perpendicular and
                                       # does less rotating up close

        p("pos_tol", 0.05)
        p("yaw_tol", math.radians(4))
        p("capture_radius", 0.15)
        p("bearing_gate", math.radians(30))
        p("settle_cycles", 20)
        p("settle_hyst", 1.4)          # once settling, break only if error
                                       # exceeds tol * this factor (kills
                                       # livelock at the tolerance boundary)

        p("nav_timeout_s", 60.0)
        p("pose_timeout", 0.8)
        p("lost_grace_s", 8.0)
        p("pose_lpf_alpha", 0.35)
        p("min_horizontal_y", 0.20)

        # debug layer
        p("log_period_s", 0.5)
        p("log_poses", False)

        g = lambda n: self.get_parameter(n).value
        self.standoff_x = g("standoff_x")
        self.v_min, self.v_max = g("v_min"), g("v_max")
        self.w_min, self.w_max = g("w_min"), g("w_max")
        self.kp_lin, self.kp_ang = g("kp_lin"), g("kp_ang")
        self.sidestep_max_s = g("sidestep_max_s")
        self.sidestep_pause_s = g("sidestep_pause_s")
        self.kp_perp = g("kp_perp")
        self.pos_tol, self.yaw_tol = g("pos_tol"), g("yaw_tol")
        self.capture_radius = g("capture_radius")
        self.bearing_gate = g("bearing_gate")
        self.settle_cycles = g("settle_cycles")
        self.settle_hyst = g("settle_hyst")
        self.nav_timeout_s = g("nav_timeout_s")
        self.pose_timeout = g("pose_timeout")
        self.lost_grace_s = g("lost_grace_s")
        self.alpha = g("pose_lpf_alpha")
        self.min_hy = g("min_horizontal_y")
        self.pelvis_frame = g("pelvis_frame")
        self.camera_frame = g("camera_frame")
        self.log_period_s = g("log_period_s")
        self.log_poses = g("log_poses")

        # state
        self.phase = Phase.IDLE
        self.phase_before_hold = None
        self.hold_since = None
        self.start_time = None
        self.settle_count = 0

        # sidestep duty-cycle: track continuous lateral motion and enforce
        # a cooldown so the policy can't be commanded into a long sidestep
        self.sidestep_active_since = None   # when current lateral run began
        self.sidestep_pause_until = None    # cooldown end (clock time)

        self.bx = self.by = None
        self.gy = None
        self.last_stamp = None

        # debug state
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_heartbeat = None
        self.n_rx = 0          # detections received
        self.n_acc = 0         # accepted into the filter
        self.n_tf_fail = 0     # dropped: TF lookup failed
        self.n_rej_vert = 0    # dropped: box Y near-vertical
        self.n_rej_nan = 0     # dropped: non-finite values from tracker
        self.n_stamp_fallback = 0  # TF served with latest instead of stamp
        self.n_cmd_pub = 0     # cmd_vel messages published

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Detection3DArray, g("tracking_topic"),
                                 self.on_tracking, 10)
        self.pub_cmd = self.create_publisher(Twist, g("cmd_vel_topic"), 10)
        self.pub_point = self.create_publisher(PointStamped, "~/target_point", 10)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_status = self.create_publisher(String, "~/status", latched)
        self.pub_result = self.create_publisher(String, "~/result", latched)

        self.create_service(Trigger, "~/start", self.on_start)
        self.create_service(Trigger, "~/cancel", self.on_cancel)

        self.dt = 1.0 / g("control_rate_hz")
        self.create_timer(self.dt, self.control_step)

        self.publish_status()
        self.get_logger().info(
            f"H2 local navigator (DEBUG build) | standoff_x={self.standoff_x:.2f} m | "
            f"v in [{self.v_min},{self.v_max}] m/s | "
            f"w in [{self.w_min},{self.w_max}] rad/s | "
            f"tol pos={self.pos_tol} m yaw={math.degrees(self.yaw_tol):.1f} deg | "
            f"heartbeat every {self.log_period_s}s")

    # ---------------- perception ----------------

    def on_tracking(self, msg: Detection3DArray):
        self.n_rx += 1
        if not msg.detections:
            return
        det = msg.detections[0]
        pose = det.results[0].pose.pose

        # Gate 0: FoundationPose tracking can emit NaN when it diverges
        # (typically right at track-loss boundaries). NaN poisons the EMA
        # permanently and sails through shape() because every NaN comparison
        # is False — reject before it touches anything.
        vals = (pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w)
        if not all(math.isfinite(v) for v in vals):
            self.n_rej_nan += 1
            self.get_logger().warn(
                "[POSE] rejected: non-finite values from tracker "
                "(diverged track?)", throttle_duration_sec=1.0)
            return

        T_pc = self.lookup_pelvis_cam(msg.header.stamp)
        if T_pc is None:
            self.n_tf_fail += 1
            return

        p_cam = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])
        p_pel = (T_pc @ p_cam)[:3]

        R_box_cam = quat_to_rot(pose.orientation)
        y_pel = T_pc[:3, :3] @ R_box_cam[:, 1]
        y_h = y_pel[:2]
        n = np.linalg.norm(y_h)
        if n < self.min_hy:
            self.n_rej_vert += 1
            self.get_logger().warn(
                f"[POSE] rejected: box Y near-vertical (|y_h|={n:.3f} < "
                f"{self.min_hy}) — box on its side?",
                throttle_duration_sec=2.0)
            return
        y_h = y_h / n

        flipped = False
        if self.gy is not None and float(np.dot(y_h, self.gy)) < 0.0:
            y_h = -y_h
            flipped = True

        raw = (float(p_pel[0]), float(p_pel[1]), float(p_pel[2]))
        # Self-heal: if the filter state is somehow non-finite (poisoned by a
        # pre-guard NaN), discard it and reinitialize from this good pose.
        if self.bx is not None and not (
                math.isfinite(self.bx) and math.isfinite(self.by)
                and np.all(np.isfinite(self.gy))):
            self.get_logger().error(
                "[POSE] filter state was non-finite — resetting filter")
            self.bx = self.by = None
            self.gy = None
        if self.bx is None:
            self.bx, self.by = raw[0], raw[1]
            self.gy = y_h
            self.get_logger().info(
                f"[POSE] first accepted pose: box at pelvis "
                f"({raw[0]:.3f},{raw[1]:.3f},{raw[2]:.3f}) m, "
                f"grasp axis ({y_h[0]:.2f},{y_h[1]:.2f})")
        else:
            a = self.alpha
            self.bx += a * (raw[0] - self.bx)
            self.by += a * (raw[1] - self.by)
            v = (1 - a) * self.gy + a * y_h
            self.gy = v / np.linalg.norm(v)

        self.n_acc += 1
        if self.log_poses:
            self.get_logger().info(
                f"[POSE] raw=({raw[0]:.3f},{raw[1]:.3f}) "
                f"filt=({self.bx:.3f},{self.by:.3f}) "
                f"axis=({self.gy[0]:.2f},{self.gy[1]:.2f})"
                f"{' [sign-flipped]' if flipped else ''}")

        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.last_stamp = stamp if stamp.nanoseconds > 0 else self.get_clock().now()

    def lookup_pelvis_cam(self, stamp=None):
        query_time = rclpy.time.Time()
        if stamp is not None:
            t = rclpy.time.Time.from_msg(stamp)
            if t.nanoseconds > 0:
                query_time = t
        try:
            ts = self.tf_buffer.lookup_transform(
                self.pelvis_frame, self.camera_frame, query_time,
                timeout=rclpy.duration.Duration(seconds=0.05))
            return transform_to_matrix(ts)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass
        try:
            ts = self.tf_buffer.lookup_transform(
                self.pelvis_frame, self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            self.n_stamp_fallback += 1
            self.get_logger().warn(
                "[TF] stamped lookup unavailable — using latest transform "
                "(bias possible while moving)",
                throttle_duration_sec=5.0)
            return transform_to_matrix(ts)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"[TF] {self.pelvis_frame}<-{self.camera_frame} failed: {e}",
                throttle_duration_sec=2.0)
            return None

    def pose_age(self):
        if self.last_stamp is None:
            return float("inf")
        return (self.get_clock().now() - self.last_stamp).nanoseconds * 1e-9

    # ---------------- goal geometry ----------------

    def compute_errors(self):
        n1 = np.array([self.gy[1], -self.gy[0]])
        b = np.array([self.bx, self.by])
        n = n1 if float(np.dot(n1, -b)) > 0.0 else -n1
        t = b + self.standoff_x * n
        phi = math.atan2(self.gy[1], self.gy[0])
        eyaw = wrap_half(phi - math.pi / 2)
        return float(t[0]), float(t[1]), eyaw

    # ---------------- services ----------------

    def on_start(self, req, resp):
        age = self.pose_age()
        self.get_logger().info(
            f"[SRV] start requested | pose age={age:.2f}s | phase={self.phase.value}")
        if age > self.pose_timeout:
            resp.success = False
            resp.message = "No fresh box track — refusing to start."
            self.get_logger().warn(f"[SRV] start REFUSED: pose age {age:.2f}s "
                                   f"> {self.pose_timeout}s")
            return resp
        self.settle_count = 0
        self.hold_since = None
        self.start_time = self.get_clock().now()
        self.set_phase(Phase.ROTATE_TO_GOAL)
        tx, ty, eyaw = self.compute_errors()
        resp.success = True
        resp.message = (f"Approach started: standoff at ({tx:.2f},{ty:.2f}) m, "
                        f"grasp-axis error {math.degrees(eyaw):.1f} deg")
        self.get_logger().info(f"[SRV] {resp.message}")
        return resp

    def on_cancel(self, req, resp):
        self.get_logger().info(f"[SRV] cancel requested | phase={self.phase.value}")
        self.publish_cmd(Twist(), "cancel")
        self.set_phase(Phase.IDLE)
        resp.success = True
        resp.message = "Cancelled."
        return resp

    # ---------------- velocity shaping ----------------

    def shape(self, err, kp, vmin, vmax, tol):
        if abs(err) < tol:
            return 0.0
        v = kp * err
        return math.copysign(min(max(abs(v), vmin), vmax), v)

    def gate_sidestep(self, vy_desired):
        """Duty-cycle limiter for lateral velocity. Caps magnitude to v_min
        and enforces sidestep_max_s of motion followed by sidestep_pause_s
        of no lateral motion — the policy trips on long continuous sidesteps.
        Fore/aft and yaw are unaffected; only vy is gated."""
        now = self.get_clock().now()

        if abs(vy_desired) < 1e-6:
            # No sidestep wanted — end any active run, clear pause.
            self.sidestep_active_since = None
            return 0.0

        # In cooldown? Suppress lateral motion until it expires.
        if self.sidestep_pause_until is not None:
            if now < self.sidestep_pause_until:
                remaining = (self.sidestep_pause_until - now).nanoseconds * 1e-9
                self.get_logger().info(
                    f"[SIDESTEP] paused {remaining:.1f}s more (cooldown)",
                    throttle_duration_sec=0.5)
                return 0.0
            self.sidestep_pause_until = None
            self.sidestep_active_since = None

        # Start a run if none active.
        if self.sidestep_active_since is None:
            self.sidestep_active_since = now

        run = (now - self.sidestep_active_since).nanoseconds * 1e-9
        if run >= self.sidestep_max_s:
            # Hit the limit — force a pause.
            self.sidestep_pause_until = now + rclpy.duration.Duration(
                seconds=self.sidestep_pause_s)
            self.sidestep_active_since = None
            self.get_logger().info(
                f"[SIDESTEP] {self.sidestep_max_s}s reached -> forced "
                f"{self.sidestep_pause_s}s pause")
            return 0.0

        # Allowed: minimal-magnitude sidestep.
        return math.copysign(self.v_min, vy_desired)

    # ---------------- control loop ----------------

    def control_step(self):
        if self.phase in (Phase.IDLE, Phase.DONE, Phase.FAILED):
            return

        if self.start_time is not None:
            elapsed = (self.get_clock().now() -
                       self.start_time).nanoseconds * 1e-9
            if elapsed > self.nav_timeout_s:
                self.fail(f"timeout: {elapsed:.1f}s > "
                          f"nav_timeout_s={self.nav_timeout_s}s "
                          f"(last phase: {self.phase.value})")
                return

        age = self.pose_age()

        if age > self.pose_timeout:
            if self.phase != Phase.HOLD:
                self.phase_before_hold = self.phase
                self.hold_since = self.get_clock().now()
                # A stationary robot's position cannot change during a
                # perception gap — settle evidence from final_align stays
                # valid across the hold.
                self.set_phase(Phase.HOLD, preserve_settle=(
                    self.phase_before_hold == Phase.FINAL_ALIGN))
                self.get_logger().warn(
                    f"[HOLD] track stale (age={age:.2f}s > {self.pose_timeout}s) "
                    f"— zero cmd, waiting for re-estimation (grace "
                    f"{self.lost_grace_s}s)")
            else:
                held = (self.get_clock().now() -
                        self.hold_since).nanoseconds * 1e-9
                if held > self.lost_grace_s:
                    self.fail(f"Track lost for {held:.1f}s "
                              f"(> {self.lost_grace_s}s grace)")
                    return
            self.publish_cmd(Twist(), "hold: stale pose")
            self.heartbeat(None, None, None, age)
            return

        if self.phase == Phase.HOLD:
            held = (self.get_clock().now() -
                    self.hold_since).nanoseconds * 1e-9
            self.get_logger().info(
                f"[HOLD] track recovered after {held:.1f}s — resuming "
                f"{(self.phase_before_hold or Phase.ROTATE_TO_GOAL).value}")
            self.set_phase(self.phase_before_hold or Phase.ROTATE_TO_GOAL,
                           preserve_settle=True)

        tx, ty, eyaw = self.compute_errors()
        self.publish_point(tx, ty)

        dist = math.hypot(tx, ty)
        bearing = math.atan2(ty, tx)
        cmd = Twist()
        why = self.phase.value

        if self.phase == Phase.ROTATE_TO_GOAL:
            if dist < self.capture_radius:
                self.get_logger().info(
                    f"[PHASE] dist {dist:.3f} < capture {self.capture_radius} "
                    f"-> final_align")
                self.set_phase(Phase.FINAL_ALIGN)
            elif abs(bearing) < self.bearing_gate * 0.5:
                self.get_logger().info(
                    f"[PHASE] bearing {math.degrees(bearing):.1f} deg aimed "
                    f"-> approach")
                self.set_phase(Phase.APPROACH)
            else:
                cmd.angular.z = self.shape(
                    bearing, self.kp_ang, self.w_min, self.w_max,
                    self.bearing_gate * 0.4)
                why = f"rotate_to_goal: bearing {math.degrees(bearing):.1f} deg"

        elif self.phase == Phase.APPROACH:
            if dist < self.capture_radius:
                self.get_logger().info(
                    f"[PHASE] dist {dist:.3f} < capture {self.capture_radius} "
                    f"-> final_align")
                self.set_phase(Phase.FINAL_ALIGN)
            elif abs(bearing) > self.bearing_gate:
                self.get_logger().info(
                    f"[PHASE] bearing {math.degrees(bearing):.1f} deg blew "
                    f"gate {math.degrees(self.bearing_gate):.0f} -> re-aim")
                self.set_phase(Phase.ROTATE_TO_GOAL)
            else:
                cmd.linear.x = self.shape(tx, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol)
                cmd.linear.y = self.gate_sidestep(
                    self.shape(ty, self.kp_lin, self.v_min,
                               self.v_max, self.pos_tol * 2))
                # Steer with two blended angular terms:
                #   bearing -> keep the nose on the standoff point
                #   eyaw    -> square the torso to the box face NOW, while
                #              far and cheap to rotate, so final_align does
                #              little turning up close (where 1/distance makes
                #              rotation the main tracking-killer).
                w_raw = self.kp_ang * bearing + self.kp_perp * eyaw
                if abs(bearing) < self.yaw_tol * 2 and abs(eyaw) < self.yaw_tol:
                    cmd.angular.z = 0.0
                else:
                    cmd.angular.z = math.copysign(
                        min(max(abs(w_raw), self.w_min), self.w_max), w_raw)
                why = (f"approach: dist {dist:.3f} m "
                       f"bearing={math.degrees(bearing):+.1f} "
                       f"perp_err={math.degrees(eyaw):+.1f}")

        elif self.phase == Phase.FINAL_ALIGN:
            # Hysteresis: while settling (cmd already zero), only break if
            # the error exceeds tol * settle_hyst — otherwise estimator
            # jitter exactly at the boundary livelocks settle/trim forever.
            hyst = self.settle_hyst if self.settle_count > 0 else 1.0
            ex_ok = abs(tx) < self.pos_tol * hyst
            ey_ok = abs(ty) < self.pos_tol * hyst
            eyaw_ok = abs(eyaw) < self.yaw_tol * hyst

            if ex_ok and ey_ok and eyaw_ok:
                self.publish_cmd(Twist(),
                                 f"settle {self.settle_count + 1}/"
                                 f"{self.settle_cycles}")
                self.settle_count += 1
                if self.settle_count >= self.settle_cycles:
                    self.succeed(tx, ty, eyaw)
                self.heartbeat(tx, ty, eyaw, age)
                return
            if self.settle_count > 0:
                self.get_logger().info(
                    f"[ALIGN] settle broken at {self.settle_count}: "
                    f"dx={tx:.3f}({'ok' if ex_ok else 'X'}) "
                    f"dy={ty:.3f}({'ok' if ey_ok else 'X'}) "
                    f"dyaw={math.degrees(eyaw):.1f}deg"
                    f"({'ok' if eyaw_ok else 'X'})")
            self.settle_count = 0
            if not eyaw_ok:
                cmd.angular.z = self.shape(eyaw, self.kp_ang, self.w_min,
                                           self.w_max, self.yaw_tol)
                why = f"final_align yaw: {math.degrees(eyaw):.1f} deg"
            else:
                cmd.linear.x = self.shape(tx, self.kp_lin, self.v_min,
                                          self.v_max, self.pos_tol)
                cmd.linear.y = self.gate_sidestep(
                    self.shape(ty, self.kp_lin, self.v_min,
                               self.v_max, self.pos_tol))
                why = f"final_align pos: dx={tx:.3f} dy={ty:.3f}"

        self.publish_cmd(cmd, why)
        self.heartbeat(tx, ty, eyaw, age)

    # ---------------- terminal states ----------------

    def succeed(self, tx, ty, eyaw):
        self.publish_cmd(Twist(), "succeeded")
        self.set_phase(Phase.DONE)
        result = {
            "outcome": "succeeded",
            "dx_m": round(tx, 4),
            "dy_m": round(ty, 4),
            "dyaw_deg": round(math.degrees(eyaw), 2),
            "box_x_m": round(self.bx, 4),
            "box_y_m": round(self.by, 4),
        }
        self.publish_result(result)
        self.get_logger().info(f"[DONE] standoff reached: {result}")
        self.log_counters()

    def fail(self, reason):
        self.publish_cmd(Twist(), f"failed: {reason}")
        self.set_phase(Phase.FAILED)
        self.publish_result({"outcome": "failed", "reason": reason})
        self.get_logger().error(f"[FAIL] {reason}")
        self.log_counters()

    # ---------------- debug helpers ----------------

    def publish_cmd(self, cmd: Twist, why: str):
        """Single funnel for every cmd_vel publish — logs every CHANGE.
        Hard-blocks non-finite commands: NaN reaching the policy's
        observation vector NaNs the whole action output and drops the
        robot instantly."""
        vx, vy, wz = cmd.linear.x, cmd.linear.y, cmd.angular.z
        if not (math.isfinite(vx) and math.isfinite(vy)
                and math.isfinite(wz)):
            self.get_logger().error(
                f"[CMD] BLOCKED non-finite command "
                f"({vx},{vy},{wz}) | phase={self.phase.value} | {why} "
                f"— publishing zero instead")
            cmd = Twist()
            vx = vy = wz = 0.0
        new = (round(vx, 3), round(vy, 3), round(wz, 3))
        if new != self.last_cmd:
            self.get_logger().info(
                f"[CMD] vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f} | "
                f"phase={self.phase.value} | {why}")
            self.last_cmd = new
        self.n_cmd_pub += 1
        self.pub_cmd.publish(cmd)

    def heartbeat(self, tx, ty, eyaw, age):
        now = self.get_clock().now()
        if (self.last_heartbeat is not None and
                (now - self.last_heartbeat).nanoseconds * 1e-9 < self.log_period_s):
            return
        self.last_heartbeat = now
        if tx is None:
            geo = "errors=n/a (stale pose)"
        else:
            geo = (f"target=({tx:+.3f},{ty:+.3f})m dist={math.hypot(tx, ty):.3f} "
                   f"bearing={math.degrees(math.atan2(ty, tx)):+.1f}deg "
                   f"dyaw={math.degrees(eyaw):+.1f}deg")
        box = (f"box=({self.bx:+.3f},{self.by:+.3f}) "
               f"axis=({self.gy[0]:+.2f},{self.gy[1]:+.2f})"
               if self.bx is not None else "box=none")
        c = self.last_cmd
        self.get_logger().info(
            f"[HB] {self.phase.value} | {box} | {geo} | age={age:.2f}s | "
            f"cmd=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) | "
            f"rx={self.n_rx} acc={self.n_acc} tf_fail={self.n_tf_fail} "
            f"rej={self.n_rej_vert} nan={self.n_rej_nan} "
            f"fb={self.n_stamp_fallback}")

    def log_counters(self):
        self.get_logger().info(
            f"[STATS] detections rx={self.n_rx} accepted={self.n_acc} "
            f"tf_failed={self.n_tf_fail} rejected_vertical={self.n_rej_vert} "
            f"rejected_nan={self.n_rej_nan} "
            f"stamp_fallbacks={self.n_stamp_fallback} "
            f"cmds_published={self.n_cmd_pub}")

    # ---------------- plumbing ----------------

    def set_phase(self, phase: Phase, preserve_settle: bool = False):
        if phase != self.phase:
            self.get_logger().info(
                f"Phase -> {phase.value}"
                f"{' (settle preserved: %d)' % self.settle_count if preserve_settle and self.settle_count else ''}")
            self.phase = phase
            if not preserve_settle:
                self.settle_count = 0
            self.publish_status()

    def publish_status(self):
        running = self.phase in (Phase.ROTATE_TO_GOAL, Phase.APPROACH,
                                 Phase.FINAL_ALIGN, Phase.HOLD)
        msg = String()
        msg.data = "running" if running else self.phase.value
        self.pub_status.publish(msg)

    def publish_result(self, d):
        msg = String()
        msg.data = json.dumps(d)
        self.pub_result.publish(msg)

    def publish_point(self, tx, ty):
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.pelvis_frame
        m.point.x, m.point.y = tx, ty
        self.pub_point.publish(m)


def main():
    rclpy.init()
    node = H2LocalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub_cmd.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
