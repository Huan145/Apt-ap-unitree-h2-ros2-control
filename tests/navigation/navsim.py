"""Closed-loop test world for the arrival navigator (kinematics only, no ROS).

A robot with first-order velocity lag drives on the node's cmd_vel; a table stands in the odom frame; a
delayed, noisy, flaky vision stream publishes /workstation_pose in the pelvis frame at image time; the TF
buffer is fed from the robot's pose history, so stamped lookups behave like tf2.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import helpers
import ros_stubs
from ros_stubs import Msg

T_PC = np.eye(4)
T_PC[:3, :3], T_PC[:3, 3] = helpers.R_PELVIS_CAM, helpers.T_PELVIS_CAM
T_CP = np.linalg.inv(T_PC)                                    # Camera <- pelvis
PELVIS_Z, TABLE_Z = 0.9, 0.525


def tf_msg(T):
    m = Msg()
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, T[:3, 3])
    (m.transform.rotation.x, m.transform.rotation.y,
     m.transform.rotation.z, m.transform.rotation.w) = Rot.from_matrix(T[:3, :3]).as_quat()
    return m


def stamp_msg(t):
    return Msg(sec=int(t), nanosec=int(round((t % 1) * 1e9)))


class FakeTF:
    """odom <- pelvis from a pose history (linear interpolation, ExtrapolationException outside), Camera <- pelvis fixed."""

    def __init__(self):
        self.hist, self.broken = [], False                    # (t, x, y, yaw)

    def lookup_transform(self, target, source, time, timeout=None):
        if self.broken or not self.hist:
            raise ros_stubs.LookupException('no transform')
        if (target, source) == ('Camera', 'pelvis'):
            return tf_msg(T_CP)
        if (target, source) != ('odom', 'pelvis'):
            raise ros_stubs.LookupException(f'{target} <- {source}')
        ts = [h[0] for h in self.hist]
        t = time.sec + time.nanosec * 1e-9 if isinstance(time, Msg) else ts[-1]
        if not ts[0] - 1e-9 <= t <= ts[-1] + 1e-9:
            raise ros_stubs.ExtrapolationException(f'{t} outside {ts[0]}..{ts[-1]}')
        x, y, yaw = (float(np.interp(t, ts, [h[i] for h in self.hist])) for i in (1, 2, 3))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rot.from_euler('z', yaw).as_matrix(), (x, y, PELVIS_Z)
        return tf_msg(T)

    def all_frames_as_string(self):
        return 'World: parent none\nmap: parent World'


class Sim:
    def __init__(self, node, robot=(0.0, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), *, latency=0.3, vision_hz=10.0,
                 sigma=(0.02, math.radians(1.0)), tau=0.25, blocked_below=1.2, fov=math.radians(41), gaps=None,
                 flip_prob=0.0, seed=0, dt=0.05):
        self.node, self.dt, self.tau, self.latency = node, dt, tau, latency
        self.table, self.sigma, self.blocked_below, self.fov = table, sigma, blocked_below, fov
        self.gaps, self.flip_prob, self.rng = gaps, flip_prob, np.random.default_rng(seed)
        self.vision_period = 1.0 / vision_hz
        self.x, self.y, self.yaw = robot
        self.v, self.t, self.next_vision = np.zeros(3), 0.0, 0.0
        self.tf = node.tf_buffer = FakeTF()
        self.hist = []                                        # (t, x, y, yaw, speed)
        self.hooks, self.moving_bias, self.bias_windows, self.vision_on = [], 0.0, [], True
        self.rec = []                                         # (t, x, y, yaw, speed, phase value)
        self._record()

    # ---- world ----
    def _record(self):
        speed = float(np.hypot(*self.v[:2]))
        self.tf.hist.append((self.t, self.x, self.y, self.yaw))
        self.hist.append((self.t, self.x, self.y, self.yaw, speed))
        self.rec.append((self.t, self.x, self.y, self.yaw, speed, self.node.phase.value))

    def pose_at(self, t):
        ts = [h[0] for h in self.hist]
        return tuple(float(np.interp(t, ts, [h[i] for h in self.hist])) for i in (1, 2, 3, 4))

    def true_errors(self, pose=None):
        """(dx, dy, dyaw, face distance): goal error in the robot frame, heading error to face the table, and distance
        to the table's near face, for the true pose. Own formulas (the node's parameters, not its code)."""
        standoff, lateral, depth = (self.node.p.place_standoff, self.node.p.lateral_offset, self.node.p.table_depth)
        x, y, th = pose or (self.x, self.y, self.yaw)
        cx, cy, phi = self.table
        n = np.array([math.sin(phi), -math.cos(phi)])         # unit normal, either sign
        if n @ (np.array([x, y]) - (cx, cy)) < 0:
            n = -n
        goal = np.array([cx, cy]) + (depth / 2 + standoff) * n + lateral * np.array([n[1], -n[0]])
        e = Rot.from_euler('z', -th).as_matrix()[:2, :2] @ (goal - (x, y))
        face_dir = math.atan2(-n[1], -n[0])
        return e[0], e[1], (face_dir - th + math.pi) % (2 * math.pi) - math.pi, float(n @ (np.array([x, y]) - (cx, cy))) - depth / 2

    # ---- vision ----
    def pose_msg(self, t_img):
        x, y, th, speed = self.pose_at(t_img)
        cx, cy, phi = self.table
        face = self.true_errors((x, y, th))[3]
        d = np.array([cx - x, cy - y])
        rel = np.array([math.cos(th) * d[0] + math.sin(th) * d[1], -math.sin(th) * d[0] + math.cos(th) * d[1]])
        if (not self.vision_on or abs(math.atan2(rel[1], rel[0])) > self.fov or np.hypot(*d) > 5.0
                or face < self.blocked_below or (self.gaps and self.gaps(t_img))):
            return None
        n = np.array([-math.sin(phi), math.cos(phi)])
        if n @ (np.array([x, y]) - (cx, cy)) < 0:
            n = -n
        off = np.zeros(2)
        if speed > 0.05:
            off += self.moving_bias * n                       # while walking the table looks closer (n points to the robot)
        for t0, t1, b in self.bias_windows:                   # table looks displaced (odom frame) while t0 <= t_img < t1
            if t0 <= t_img < t1:
                off += b
        rel = rel + np.array([math.cos(th) * off[0] + math.sin(th) * off[1], -math.sin(th) * off[0] + math.cos(th) * off[1]])
        rel = rel + self.rng.normal(0, self.sigma[0], 2)
        yaw = phi - math.pi / 2 - th + self.rng.normal(0, self.sigma[1])    # mesh Y = long axis, so the mesh frame is the axis rotated by -90 deg
        if self.rng.random() < self.flip_prob:
            yaw += math.pi                                    # symmetric 180 deg flip about z
        m = helpers.MSG['PoseStamped']()
        m.header.stamp, m.header.frame_id = stamp_msg(t_img), 'pelvis'
        m.pose.position.x, m.pose.position.y, m.pose.position.z = rel[0], rel[1], TABLE_Z - PELVIS_Z
        m.pose.orientation.x = m.pose.orientation.y = 0.0
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m

    # ---- loop ----
    def step(self):
        node = self.node
        node.get_clock().t = self.t
        node.control_step()
        cmd = node.pubs['cmd_vel'].msgs[-1] if node.pubs['cmd_vel'].msgs else None
        u = np.zeros(3) if cmd is None else np.array([cmd.linear.x, cmd.linear.y, cmd.angular.z])
        self.v += (u - self.v) * (1 - math.exp(-self.dt / self.tau))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * self.v[0] - s * self.v[1]) * self.dt
        self.y += (s * self.v[0] + c * self.v[1]) * self.dt
        self.yaw += self.v[2] * self.dt
        self.t += self.dt
        self._record()
        node.get_clock().t = self.t
        js = helpers.MSG['JointState']()
        js.header.stamp, js.name, js.position = stamp_msg(self.t), ['head_pitch_joint'], [0.0]
        node.subs['joint_states'](js)
        while self.t >= self.next_vision - 1e-9:
            self.next_vision += self.vision_period
            if self.t - self.latency >= 0:
                m = self.pose_msg(self.t - self.latency)
                if m is not None:
                    node.subs['/workstation_pose'](m)
        for h in self.hooks:
            h(self)

    def run(self, until=None, limit=200.0):
        """Step until `until(sim)` is true or the node stops running (or `limit` s). Returns the node's phase."""
        while self.t < limit:
            self.step()
            if (until and until(self)) or self.node.phase.value in ('succeeded', 'failed', 'idle'):
                break
        return self.node.phase.value

    def speed_during(self, phase, after=0.0):
        """Largest true speed in `phase`, `after` s after it began."""
        t0 = next((r[0] for r in self.rec if r[5] == phase), None)
        return max([r[4] for r in self.rec if r[5] == phase and r[0] >= (t0 or 0) + after] or [0.0])
