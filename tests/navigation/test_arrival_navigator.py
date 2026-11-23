import json
import math
import signal
import sys
import types

import pytest
from scipy.spatial.transform import Rotation as Rot

import helpers
import ros_stubs
from navsim import PELVIS_Z, TABLE_Z, FakeTF, Sim
from ros_stubs import Msg

import h2_arrival_navigator as nav

HALF = 0.762 / 2
STANDOFF = nav.PARAMS['place_standoff']
GOAL = HALF + STANDOFF                   # pelvis -> table centre at the end


def make(**params):
    ros_stubs.Node.overrides = params
    return nav.H2ArrivalNavigator()


def start(n):
    return n.srvs['~/start'](Msg(), Msg())


def cmd(n):
    m = n.pubs['cmd_vel'].msgs[-1]
    return m.linear.x, m.linear.y, m.angular.z


def result(n):
    return json.loads(n.pubs['~/result'].msgs[-1].data)


def no_crash(n):
    assert not n.log.has('error', 'CRASH'), [m for l, m in n.log.records if 'CRASH' in m]


# ---------------- geometry and estimator ----------------

def geo(n, est, r, stand=STANDOFF):
    g = n.geometry(est, r, stand)
    return g.tx, g.ty, g.eyaw, g.rng, g.off


def test_geometry_table_ahead():
    assert geo(make(), (3.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0)) == pytest.approx((3 - GOAL, 0.0, 0.0, 3 - HALF, 0.0), abs=1e-9)


def test_geometry_robot_yawed_and_on_the_other_side():
    n = make()
    tx, ty, eyaw, *_ = geo(n, (3.0, 0.0, math.pi / 2), (0.0, 0.0, math.radians(30)))
    g = 3 - GOAL
    assert (tx, ty, eyaw) == pytest.approx((g * math.cos(math.radians(30)), -g * math.sin(math.radians(30)),
                                            math.radians(-30)), abs=1e-9)
    assert geo(n, (3.0, 0.0, math.pi / 2), (6.0, 0.0, math.pi)) == pytest.approx((3 - GOAL, 0.0, 0.0, 3 - HALF, 0.0), abs=1e-9)  # from the +x side
    assert geo(n, (3.0, 0.0, -math.pi / 2), (0.0, 0.0, 0.0))[:3] == pytest.approx((3 - GOAL, 0, 0), abs=1e-9)                   # axis has no sign


def test_lateral_offset_is_to_the_robots_left_whichever_side_it_stands():
    n = make(lateral_offset=0.5)
    for r in ((0.0, 0.0, 0.0), (6.0, 0.0, math.pi)):
        tx, ty, _, _, off = geo(n, (3.0, 0.0, math.pi / 2), r)
        assert (tx, ty, off) == pytest.approx((3 - GOAL, 0.5, -0.5), abs=1e-9)            # the line is 0.5 m to its left


def test_off_is_how_far_the_robot_stands_to_the_left_of_the_bisector_and_the_goal_is_on_the_line():
    for stand in (0.2, 3.0):
        tx, ty, _, rng, off = geo(make(), (4.0, 0.0, math.pi / 2), (1.0, 0.7, 0.0), stand)
        assert off == pytest.approx(0.7) and ty == pytest.approx(-0.7) and rng == pytest.approx(3 - HALF)
        assert tx == pytest.approx(3 - HALF - stand)                                      # on the line, `stand` from the face


def test_range_is_the_distance_to_the_face_not_to_its_plane():
    """Seen live: the table centre 5.1 m away with the axis 12 deg off the line of sight put the (infinite) plane of the
    near face 0.65 m from the robot, so the latch fired 5 m from the table. The face itself is 3.8 m away."""
    d = math.radians
    g = make().geometry((5.11, 0.0, d(11.6)), (0.0, 0.0, 0.0), STANDOFF)
    plane = 5.11 * math.sin(d(11.6)) - HALF
    assert plane == pytest.approx(0.65, abs=0.01) and g.rng == pytest.approx(3.82, abs=0.02)
    assert make().geometry((3.0, 0.0, math.pi / 2), (1.0, 3.0, 0.0), STANDOFF).rng == pytest.approx(math.hypot(2 - HALF, 3 - 1.237))


def test_the_table_side_is_kept_beside_the_axis_plane_and_changes_beyond_it():
    n = make()
    est = (3.0, 0.0, math.pi / 2)                                                          # long faces look to -x and +x
    n.geometry(est, (3.3, -4.0, 0.0), STANDOFF)                                            # 0.3 m on the +x side: chooses +x
    assert n.n_ref == pytest.approx((1.0, 0.0), abs=1e-9)
    n.geometry(est, (2.9, -4.0, 0.0), STANDOFF)                                            # now 0.1 m on the -x side: keeps +x
    assert n.n_ref == pytest.approx((1.0, 0.0), abs=1e-9)
    n.geometry(est, (2.0, -4.0, 0.0), STANDOFF)                                            # 1.0 m on the -x side: changes
    assert n.n_ref == pytest.approx((-1.0, 0.0), abs=1e-9)


def test_fuse_takes_the_median_and_averages_the_axis_mod_pi():
    d = math.radians
    s = [(0.0, 1.00, 2.00, d(89.0)), (0.1, 1.01, 2.00, d(-91.0)), (0.2, 1.00, 2.01, d(90.5)),
         (0.3, 1.02, 1.99, d(90.0)), (0.4, 5.00, 5.00, d(90.0))]             # one wild pose
    x, y, a = nav.fuse(s)
    assert (x, y) == pytest.approx((1.01, 2.00), abs=0.011) and abs(math.degrees(a) - 90.0) < 1.0
    _, _, a2 = nav.fuse([(0, 0, 0, d(179.0)), (0, 0, 0, d(-179.0))])           # the same axis from both sides of +-180
    assert abs(nav.wrap(2 * a2)) < 1e-6                                        # = 0 or 180 deg, not 90


def test_pose_is_mapped_with_the_pelvis_pose_at_image_time_not_at_arrival():
    n = make()
    sim = Sim(n, robot=(0.0, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), sigma=(0.0, 0.0))
    msg = sim.pose_msg(0.0)                                                    # image taken at t=0 ...
    sim.x, sim.y, sim.yaw, sim.t = 1.0, 0.2, 0.3, 1.0                          # ... robot has walked on by arrival
    sim._record()
    n.phase = nav.Phase.SCAN
    n.on_pose(msg)
    t, x, y, phi = n.samples[-1]
    assert (x, y, phi) == pytest.approx((3.8, 0.0, math.pi / 2), abs=1e-6)


def test_bad_poses_are_rejected_and_counted():
    n = make()
    sim = Sim(n, sigma=(0.0, 0.0))
    n.phase = nav.Phase.SCAN
    good = sim.pose_msg(0.0)
    nan = sim.pose_msg(0.0)
    nan.pose.position.x = float('nan')
    far = sim.pose_msg(0.0)
    far.pose.position.x = 9.0
    vertical = sim.pose_msg(0.0)
    (vertical.pose.orientation.x, vertical.pose.orientation.y,
     vertical.pose.orientation.z, vertical.pose.orientation.w) = Rot.from_euler('x', 90, degrees=True).as_quat()
    for m in (nan, far, vertical):
        n.on_pose(m)
    assert len(n.samples) == 0 and n.n_rej == 3
    n.on_pose(good)
    assert len(n.samples) == 1 and n.n_acc == 1


def test_non_finite_command_is_blocked():
    n = make()
    n.publish_cmd(float('nan'), 0.0, 0.0, 'test')
    assert cmd(n) == (0.0, 0.0, 0.0) and n.log.has('error', 'BLOCKED')


# ---------------- services ----------------

def test_start_without_odometry_says_which_frames_exist():
    n = make()
    n.tf_buffer = FakeTF()                                                      # nothing recorded: no odom <- pelvis
    r = start(n)
    assert not r.success and 'odom_frame' in r.message and 'World' in r.message
    assert n.phase is nav.Phase.IDLE and n.log.has('warn', 'no TF odom')


def test_start_twice_is_refused_and_cancel_stops_and_releases_the_head():
    n = make()
    sim = Sim(n)
    assert start(n).success and not start(n).success
    sim.run(until=lambda s: s.t > 6.0)
    assert n.head_cmd is not None
    r = n.srvs['~/cancel'](Msg(), Msg())
    assert r.success and n.phase is nav.Phase.IDLE and cmd(n) == (0.0, 0.0, 0.0)
    assert math.isnan(n.pubs['head_pitch_cmd'].msgs[-1].data) and n.pubs['~/status'].msgs[-1].data == 'idle'
    no_crash(n)


# ---------------- head and shutdown ----------------

def test_head_aims_at_the_tables_elevation_not_its_bearing():
    """Same camera -> table-centre range straight ahead and 77 deg to the right: the same head angle. (atan2(y, z)
    pinned the head at its limit for the table off to the side: z -> 0.)"""
    dh = PELVIS_Z + helpers.T_PELVIS_CAM[2] - TABLE_Z                          # camera above the table centre
    rng = math.hypot(0.8, 3.87)                                                # horizontal, camera -> table centre
    for table in ((0.1 + rng, 0.0, math.pi / 2), (0.9, -3.87, 0.0)):
        n = make()
        sim = Sim(n, table=table, sigma=(0.0, 0.0), fov=math.radians(179), blocked_below=0.0)
        n.phase = nav.Phase.SCAN
        n.head_hist.append((0.0, 0.0))
        n.on_pose(sim.pose_msg(0.0))
        assert n.head_target == pytest.approx(n.p.head_gain * math.atan2(dh, rng), abs=2e-3)
        assert n.head_target < n.p.head_pitch_max


def test_stop_robot_repeats_the_zero_command_and_releases_the_head():
    n = make()
    n.publish_cmd(0.0, 0.0, -0.5, 'turning')
    n.stop_robot(pause=0.0)
    assert len(n.pubs['cmd_vel'].msgs) == 6 and all(m.angular.z == 0.0 for m in n.pubs['cmd_vel'].msgs[1:])
    assert [math.isnan(m.data) for m in n.pubs['head_pitch_cmd'].msgs] == [True] * 5


def test_stop_robot_reports_instead_of_raising_when_the_context_is_gone(monkeypatch):
    n = make()

    def invalid(_):
        raise RuntimeError("publisher's context is invalid")
    monkeypatch.setattr(n.pub_cmd, 'publish', invalid)
    n.stop_robot(pause=0.0)                                                    # must not raise
    assert n.log.has('error', 'stop command NOT sent')


def test_main_sends_the_stop_on_ctrl_c_before_rclpy_shuts_down(monkeypatch):
    """Seen live: rclpy's own SIGINT handler shut the context down before the stop could go out, and the controller
    kept the last cmd_vel. main() must switch that handler off, ignore a second Ctrl-C while stopping, shut rclpy
    down only after the stop, and treat SIGTERM like Ctrl-C."""
    n = make()
    n.publish_cmd(0.0, 0.0, -0.5, 'turning')
    NO = object()
    ctx = types.SimpleNamespace(valid=True, opt=None, sig=[])

    def spin(_):
        ctx.valid = ctx.opt is NO                                              # rclpy's default handler ends the context
        raise KeyboardInterrupt

    def guard(pub):
        real = pub.publish

        def publish(m):
            if not ctx.valid:
                raise RuntimeError("publisher's context is invalid")
            real(m)
        monkeypatch.setattr(pub, 'publish', publish)
    guard(n.pub_cmd)
    guard(n.pub_head)
    signals = types.ModuleType('rclpy.signals')
    signals.SignalHandlerOptions = types.SimpleNamespace(NO=NO)
    monkeypatch.setitem(sys.modules, 'rclpy.signals', signals)
    monkeypatch.setattr(nav.rclpy, 'init', lambda **kw: setattr(ctx, 'opt', kw.get('signal_handler_options')))
    monkeypatch.setattr(nav.rclpy, 'spin', spin)
    monkeypatch.setattr(nav.rclpy, 'try_shutdown', lambda: setattr(ctx, 'valid', False), raising=False)
    monkeypatch.setattr(signal, 'signal', lambda s, h: ctx.sig.append((s, h)))     # not the test runner's handlers
    monkeypatch.setattr(nav.time, 'sleep', lambda s: None)
    monkeypatch.setattr(nav, 'H2ArrivalNavigator', lambda: n)
    nav.main()
    assert not n.log.has('error', 'NOT sent')
    assert len(n.pubs['cmd_vel'].msgs) == 6 and all(m.angular.z == 0.0 for m in n.pubs['cmd_vel'].msgs[1:])
    assert [math.isnan(m.data) for m in n.pubs['head_pitch_cmd'].msgs] == [True] * 5
    assert (signal.SIGTERM, signal.default_int_handler) in ctx.sig
    assert (signal.SIGINT, signal.SIG_IGN) in ctx.sig and (signal.SIGTERM, signal.SIG_IGN) in ctx.sig


# ---------------- scan ----------------

def test_scan_stands_still_then_turns_then_gives_up():
    n = make(scan_wait_s=2.0, scan_max_deg=100.0)
    sim = Sim(n, table=(-3.0, 0.0, math.pi / 2))                                # behind the robot: nothing in view
    start(n)
    sim.run(until=lambda s: s.t >= 1.8)
    assert cmd(n) == (0.0, 0.0, 0.0)
    sim.run(until=lambda s: s.t >= 3.0)
    assert cmd(n)[2] == pytest.approx(0.25)
    assert sim.run() == 'failed'
    r = result(n)
    assert 'table not found' in r['reason'] and r['phase'] == 'scan' and cmd(n) == (0.0, 0.0, 0.0)
    assert 100 < math.degrees(sim.yaw) < 125


def test_scan_stops_turning_when_the_table_comes_into_view_and_carries_on():
    n = make(scan_wait_s=1.0)
    sim = Sim(n, robot=(0.0, 0.0, math.radians(150)), table=(3.8, 0.0, math.pi / 2))
    start(n)
    assert sim.run() == 'succeeded'
    assert n.log.has('info', '[SCAN] table found')
    dx, dy, dyaw, _ = sim.true_errors()
    assert math.hypot(dx, dy) < 0.08 and abs(dyaw) < math.radians(6)
    no_crash(n)


# ---------------- latch ----------------

def test_poses_taken_before_the_robot_stood_still_do_not_count():
    n = make()
    sim = Sim(n, robot=(1.5, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), sigma=(0.01, 0.0))   # face 1.9 m: inside the window
    sim.bias_windows.append((0.0, 1.9, (0.0, 0.3)))                                         # early images show a wrong table
    start(n)
    sim.run(until=lambda s: n.latched)
    assert n.latch_t0 > 1.9 and n.est[:2] == pytest.approx((3.8, 0.0), abs=0.03)
    no_crash(n)


def test_a_jump_inside_the_window_delays_the_latch_and_the_new_pose_wins():
    n = make()
    sim = Sim(n, robot=(1.5, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), sigma=(0.01, 0.0))
    start(n)
    sim.run(until=lambda s: n.phase is nav.Phase.LATCH)
    jump_at = n.latch_t0 + 0.5
    sim.bias_windows.append((jump_at, 1e9, (0.0, 0.25)))                                    # re-estimation lands 25 cm away
    sim.run(until=lambda s: n.latched)
    assert sim.t > n.latch_t0 + n.p.window_s + sim.latency + 0.1                            # not on the first full window
    assert n.est[:2] == pytest.approx((3.8, 0.25), abs=0.03)


def test_no_poses_after_the_stop_means_a_clean_failure_not_a_blind_walk():
    n = make(latch_timeout_s=4.0)
    sim = Sim(n, robot=(0.0, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), blocked_below=2.5)   # camera lost at 2.5 m
    start(n)
    assert sim.run() == 'failed'
    assert 'no stable table pose' in result(n)['reason'] and 'no poses since the stop' in result(n)['reason']
    assert cmd(n) == (0.0, 0.0, 0.0)
    face = sim.true_errors()[3]
    assert 1.6 < face < 2.1                                                                   # stopped at the window, not past it
    assert n.pubs['~/status'].msgs[-1].data == 'failed' and math.isnan(n.pubs['head_pitch_cmd'].msgs[-1].data)


# ---------------- closed loop ----------------

def run_nominal(hooks=(), **kw):
    n = make(**kw.pop('params', {}))
    sim = Sim(n, **{'robot': (0.0, -0.4, math.radians(8)), 'table': (3.8, 0.0, math.pi / 2), **kw})
    marks = {}
    sim.hooks.append(lambda s: marks.setdefault('acc', n.n_acc) if n.latched else None)
    sim.hooks.extend(hooks)
    start(n)
    return n, sim, marks, sim.run()


def clearance(sim, table=(3.8, 0.0, math.pi / 2), pts=None):
    """Smallest distance from the robot's true path (or `pts`) to the table's footprint (own formula)."""
    cx, cy, phi = table
    u, n = (math.cos(phi), math.sin(phi)), (-math.sin(phi), math.cos(phi))
    out = []
    for r in (pts or sim.rec):
        dx, dy = r[1] - cx, r[2] - cy
        out.append(math.hypot(max(abs(n[0] * dx + n[1] * dy) - HALF, 0.0), max(abs(u[0] * dx + u[1] * dy) - 2.474 / 2, 0.0)))
    return min(out)


def test_walks_in_latches_in_the_window_and_arrives_on_odometry_with_the_camera_blind():
    n, sim, marks, phase = run_nominal()
    assert phase == 'succeeded'
    dx, dy, dyaw, face = sim.true_errors()
    assert STANDOFF == 0.20                                                                  # the end: 0.2 m pelvis -> table face
    assert math.hypot(dx, dy) < 0.08 and abs(dyaw) < math.radians(6) and face == pytest.approx(STANDOFF, abs=0.08)
    info = result(n)
    assert 1.7 - 0.15 <= info['latched_range_m'] <= 2.0 + 0.15 and info['latch_tries'] == 1
    assert sim.speed_during('latch', after=1.0) < 0.02                                       # stood still while latching
    assert n.n_acc == marks['acc']                                                           # nothing accepted after the latch
    assert STANDOFF - 0.1 < info['face_dist_m'] < STANDOFF + 0.1 and cmd(n) == (0.0, 0.0, 0.0)
    assert n.pubs['~/status'].msgs[-1].data == 'succeeded' and n.pubs['~/table_pose'].msgs
    assert max(r[4] for r in sim.rec) <= n.p.v_max + 0.05
    no_crash(n)


def test_head_follows_the_table_then_holds_after_the_latch():
    n, sim, _, _ = run_nominal()
    heads = [m.data for m in n.pubs['head_pitch_cmd'].msgs]
    assert heads and all(0.0 < h <= n.p.head_pitch_max for h in heads)                       # looking down at the table
    assert len(set(heads[-20:])) == 1 and heads[-1] == pytest.approx(n.head_cmd)             # frozen, still published


@pytest.mark.parametrize('kw', [
    dict(gaps=lambda t: t % 8 >= 5),                                                         # tracker gaps: 5 s valid, 3 s none
    dict(flip_prob=0.4),                                                                     # symmetric 180 deg flips
    dict(sigma=(0.04, math.radians(3))),                                                     # noisier poses
    dict(robot=(0.0, 1.2, math.radians(-20)), latency=0.6),                                  # off to the side, slow vision
    dict(robot=(1.0, 0.0, 0.0)),                                                             # starts inside the window region
    dict(robot=(2.5, 3.5, math.radians(-60)), table=(4.5, 0.0, math.pi / 2)),                # beside the table's end, 3.5 m off the line
    dict(robot=(7.2, 0.3, math.pi)),                                                         # on the table's other long side
    dict(table=(4.0, 1.0, math.radians(70))),                                                # long axis 20 deg off the robot's y
])
def test_arrives_despite(kw):
    n, sim, _, phase = run_nominal(**kw)
    assert phase == 'succeeded', result(n)
    dx, dy, dyaw, _ = sim.true_errors()
    assert math.hypot(dx, dy) < 0.10 and abs(dyaw) < math.radians(7)
    no_crash(n)


def test_stops_early_estimate_then_walks_on_and_latches_again_in_the_window():
    n = make()
    sim = Sim(n, robot=(0.0, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), sigma=(0.01, 0.0))
    sim.moving_bias = 0.35                                                                   # walking: table looks 35 cm closer
    sim.hooks.append(lambda s: setattr(s, 'moving_bias', 0.0 if n.tries >= 1 else 0.35))
    start(n)
    assert sim.run() == 'succeeded'
    info = result(n)
    assert info['latch_tries'] == 2 and 1.7 - 0.15 <= info['latched_range_m'] <= 2.0 + 0.15
    assert n.log.has('info', 'walking on')


def test_gives_up_retrying_and_latches_anyway_with_a_warning():
    n = make(latch_max_tries=1)
    sim = Sim(n, table=(3.8, 0.0, math.pi / 2), sigma=(0.01, 0.0))
    sim.moving_bias = 0.35
    start(n)
    assert sim.run() == 'succeeded'
    assert result(n)['latch_tries'] == 1 and n.log.has('warn', 'outside')


def test_lateral_offset_param_moves_the_stopping_point_along_the_table():
    n, sim, _, phase = run_nominal(params=dict(lateral_offset=0.4))
    assert phase == 'succeeded'
    dx, dy, dyaw, _ = sim.true_errors()
    assert math.hypot(dx, dy) < 0.08
    assert sim.y - 0.0 == pytest.approx(0.4, abs=0.1)                                        # table centre y=0, robot faces +x: left = +y


def test_odometry_lost_mid_walk_stops_the_robot_then_fails():
    n = make(lost_grace_s=1.0)
    sim = Sim(n)
    start(n)
    sim.run(until=lambda s: s.t > 5.0)
    sim.tf.broken = True
    assert sim.run(limit=sim.t + 5) == 'failed'
    assert 'no TF' in result(n)['reason'] and cmd(n) == (0.0, 0.0, 0.0)


def test_a_crash_in_the_loop_stops_the_robot(monkeypatch):
    n = make()
    sim = Sim(n)
    start(n)
    sim.run(until=lambda s: n.phase is nav.Phase.APPROACH)
    monkeypatch.setattr(n, 'geometry', lambda *a: 1 / 0)
    sim.step()
    assert n.phase is nav.Phase.FAILED and cmd(n) == (0.0, 0.0, 0.0) and 'internal error' in result(n)['reason']


def test_nav_timeout_fails_and_stops():
    n = make(nav_timeout_s=3.0)
    sim = Sim(n)
    start(n)
    assert sim.run() == 'failed' and 'timeout' in result(n)['reason'] and cmd(n) == (0.0, 0.0, 0.0)


# ---------------- approach along the bisector ----------------

def record(m, key, cond):
    """Sim hook: keep (t, x, y, yaw, heading error to face the table, plane distance) the first time cond(node) holds."""
    return lambda s: m.setdefault(key, (s.t, s.x, s.y, s.yaw) + tuple(s.true_errors()[2:])) if cond(s.node) else None


def test_from_beside_the_table_it_meets_the_bisector_beyond_2m_faces_the_table_and_walks_straight_in():
    table, m = (4.5, 0.0, math.pi / 2), {}
    n, sim, _, phase = run_nominal(robot=(2.5, 3.5, math.radians(-60)), table=table, hooks=[
        record(m, 'merge', lambda n: n.merged), record(m, 'latch', lambda n: n.phase is nav.Phase.LATCH),
        record(m, 'done', lambda n: n.latched)])
    assert phase == 'succeeded', result(n)
    assert m['merge'][5] > 2.4 and abs(m['merge'][2]) <= n.p.line_tol + 0.1                  # met the line more than 2 m out
    assert abs(m['latch'][4]) < math.radians(10) and abs(result(n)['latched_off_m']) < 0.2   # latched facing it, on the line
    after = [r for r in sim.rec if r[0] >= m['done'][0]]
    assert max(abs(r[2]) for r in after) < 0.2 and max(abs(math.degrees(r[3])) for r in after) < 12   # then straight in
    dx, dy, dyaw, face = sim.true_errors()
    assert math.hypot(dx, dy) < 0.10 and abs(dyaw) < math.radians(7) and face == pytest.approx(0.20, abs=0.08)
    assert clearance(sim, table) > 0.1
    no_crash(n)


def test_seen_end_on_it_goes_round_the_table_to_a_long_face_and_never_across_it():
    table = (3.8, 0.0, math.pi / 2)                                  # long axis along y, the robot on its extension, 0.1 m off
    n, sim, _, phase = run_nominal(robot=(3.9, -4.5, math.pi / 2), table=table, blocked_below=-9.0)
    assert phase == 'succeeded', result(n)
    dx, dy, dyaw, face = sim.true_errors()
    assert math.hypot(dx, dy) < 0.10 and face == pytest.approx(STANDOFF, abs=0.08)
    assert sim.x == pytest.approx(3.8 + HALF + STANDOFF, abs=0.1) and abs(sim.y) < 0.1       # on the +x long face, not a short end
    assert clearance(sim, table) > 0.1
    no_crash(n)


def test_on_the_bisector_it_does_not_detour_to_the_meet_point():
    n, sim, _, phase = run_nominal(robot=(0.0, 0.1, 0.0))
    assert phase == 'succeeded' and max(abs(r[2]) for r in sim.rec) < 0.25


def test_closer_than_the_window_on_the_bisector_it_latches_where_it_stands_and_never_backs_up():
    n, sim, _, phase = run_nominal(robot=(2.1, 0.0, 0.0))                                    # face 1.3 m away
    assert phase == 'succeeded' and n.log.has('warn', 'outside')
    assert result(n)['latch_tries'] == 1 and result(n)['latched_range_m'] == pytest.approx(1.32, abs=0.05)
    assert min(r[1] for r in sim.rec) > 2.1 - 0.05


def test_stopped_in_the_window_but_turned_away_it_turns_on_the_spot_to_face_the_table_before_it_latches():
    m = {}
    n, sim, _, phase = run_nominal(robot=(1.5, 0.0, math.radians(35)),                      # inside the window, turned 35 deg away
                                   hooks=[record(m, 'stop', lambda n: n.phase is nav.Phase.LATCH),
                                          record(m, 'latched', lambda n: n.latched)])
    assert phase == 'succeeded'
    assert abs(m['stop'][4]) > math.radians(30) and abs(m['latched'][4]) < math.radians(9)
    assert abs(m['latched'][1] - m['stop'][1]) < 0.1 and abs(m['latched'][2] - m['stop'][2]) < 0.1   # turned in place, no step closer
    assert any('latch: turn to the table' in msg for _, msg in n.log.records)


def test_inside_the_window_but_off_the_bisector_it_does_not_latch_there_but_joins_the_line_first():
    m = {}
    n, sim, _, phase = run_nominal(robot=(1.9, 0.8, 0.0), hooks=[                            # face 1.5 m away, 0.8 m off the line
        record(m, 'merge', lambda n: n.merged), record(m, 'latch', lambda n: n.phase is nav.Phase.LATCH)])
    assert phase == 'succeeded', result(n)
    assert m['merge'][0] < m['latch'][0] and abs(result(n)['latched_off_m']) < 0.25


def test_on_the_way_to_the_bisector_it_aims_at_the_meet_point_without_squaring_to_the_table():
    seq = []
    n, sim, _, phase = run_nominal(robot=(2.5, 3.5, math.radians(-60)), table=(4.5, 0.0, math.pi / 2),
                                   hooks=[lambda s: seq.append(s.node.phase.value) if not s.node.merged else None])
    changes = sum(a != b for a, b in zip(seq, seq[1:]))                                      # scan, rotate, approach: 2
    assert phase == 'succeeded' and changes <= 4, changes


def test_a_sideways_slip_next_to_the_table_does_not_swing_the_heading():
    """0.55 m from the table the bearing to a goal 0.4 m ahead is mostly the lateral error: steering on it swung the
    heading 14 deg for a 15 cm slip, next to the table with the box in the hands."""
    st = {}

    def slip(s):
        if s.node.latched and not st and s.true_errors()[3] < 0.55:
            st['yaw'] = s.yaw
            s.y += 0.15
    n, sim, _, phase = run_nominal(robot=(0.0, -0.1, 0.0), hooks=[slip])
    assert phase == 'succeeded' and st
    assert max(abs(r[3] - st['yaw']) for r in sim.rec if r[1] > 3.0) < math.radians(6)


def test_a_robot_that_cannot_turn_fails_the_latch_instead_of_waiting_for_ever():
    n = make(latch_timeout_s=6.0)
    sim = Sim(n, robot=(1.5, 0.0, math.radians(35)), table=(3.8, 0.0, math.pi / 2), tau=1e9)    # velocity never follows the command
    start(n)
    assert sim.run() == 'failed'
    assert 'no stable table pose' in result(n)['reason'] and 'turning to the table' in result(n)['reason']
    assert cmd(n) == (0.0, 0.0, 0.0) and sim.t < 12.0


def test_the_latch_looks_at_the_distance_to_the_face_not_to_its_plane():
    """The Nov 22 run latched 5 m from the table because the (infinite) plane of the near face passed 1.6 m from the
    robot. Here the other two conditions (on the bisector, aimed at the table) hold, so only the range can refuse."""
    phi = math.radians(24)                                           # table 4.9 m away, long axis 24 deg off the line of sight
    n = make()
    sim = Sim(n, robot=(0.0, 0.0, math.atan2(-math.cos(phi), math.sin(phi))), table=(4.9, 0.0, phi), sigma=(0.0, 0.0),
              fov=math.radians(179), blocked_below=-9.0)
    n.phase, n.t_start, n.merged = nav.Phase.SCAN, 0.0, True
    for _ in range(3):
        n.on_pose(sim.pose_msg(0.0))
    g = n.geometry(n.estimate(), (sim.x, sim.y, sim.yaw), STANDOFF)
    assert 4.9 * math.sin(phi) - HALF <= n.p.latch_far and abs(g.eyaw) < math.radians(8) and g.rng > 3.0
    n.phase = nav.Phase.ROTATE
    sim.step()
    assert n.phase is nav.Phase.ROTATE and n.tries == 0


def test_arriving_at_the_end_point_without_a_latch_fails_instead_of_succeeding():
    n = make()
    sim = Sim(n, robot=(3.0, 0.0, 0.0), table=(3.8, 0.0, math.pi / 2), sigma=(0.0, 0.0), blocked_below=-9.0)
    n.phase, n.t_start = nav.Phase.SCAN, 0.0
    for _ in range(3):
        n.on_pose(sim.pose_msg(0.0))
    n.phase = nav.Phase.FINAL
    sim.step()
    assert n.phase is nav.Phase.FAILED and 'without a latched table pose' in result(n)['reason'] and cmd(n) == (0.0, 0.0, 0.0)


def test_the_live_pose_after_the_latch_is_logged_next_to_the_latched_one_and_never_used():
    shifted = []

    def hook(s):                                                     # after the latch the camera sees the table 0.3 m to the side
        if s.node.latched and not shifted:
            shifted.append(s.bias_windows.append((s.t, 1e9, (0.0, 0.3))))
    n, sim, marks, phase = run_nominal(hooks=[hook])
    assert phase == 'succeeded'
    assert n.est[:2] == pytest.approx((3.8, 0.0), abs=0.03) and n.n_acc == marks['acc']      # frozen, nothing accepted
    assert n.live is not None and n.live[2] == pytest.approx(0.3, abs=0.06)
    assert n.log.has('info', 'live-latched 0.3')


def test_latch_status_says_how_far_apart_the_halves_are():
    n = make()
    n.samples.extend([(i * 0.1, 1.0 if i < 5 else 1.3, 0.0, 0.0) for i in range(10)])
    s = n.latch_status()
    assert '10/5 poses' in s and 'halves differ 300/50 mm' in s
    n.samples.clear()
    assert n.latch_status() == 'no poses since the stop'


def test_no_back_and_forth_turning_kicks_while_walking_in():
    """The two heading terms (aim at the goal, square to the table) cancel near the line; forcing the minimum turn rate
    on their tiny sum flipped wz between +-0.25 every few commands (36 flips in 66 commands in the Nov 22 log)."""
    n, sim, _, phase = run_nominal()
    wz = [m.angular.z for m in n.pubs['cmd_vel'].msgs if m.angular.z]
    flips = sum(a * b < 0 for a, b in zip(wz, wz[1:]))
    assert phase == 'succeeded' and flips <= 3, flips
