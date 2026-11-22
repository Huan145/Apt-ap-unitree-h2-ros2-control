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

def test_geometry_table_ahead():
    n = make()
    tx, ty, eyaw, face = n.geometry((3.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0))
    assert (tx, ty, eyaw, face) == pytest.approx((3 - GOAL, 0.0, 0.0, 3 - HALF), abs=1e-9)


def test_geometry_robot_yawed_and_on_the_other_side():
    n = make()
    tx, ty, eyaw, _ = n.geometry((3.0, 0.0, math.pi / 2), (0.0, 0.0, math.radians(30)))
    g = 3 - GOAL
    assert (tx, ty, eyaw) == pytest.approx((g * math.cos(math.radians(30)), -g * math.sin(math.radians(30)),
                                            math.radians(-30)), abs=1e-9)
    tx, ty, eyaw, face = n.geometry((3.0, 0.0, math.pi / 2), (6.0, 0.0, math.pi))     # facing -x from the +x side
    assert (tx, ty, eyaw, face) == pytest.approx((3 - GOAL, 0.0, 0.0, 3 - HALF), abs=1e-9)
    assert n.geometry((3.0, 0.0, -math.pi / 2), (0.0, 0.0, 0.0))[:3] == pytest.approx((3 - GOAL, 0, 0), abs=1e-9)  # axis has no sign


def test_lateral_offset_is_to_the_robots_left_whichever_side_it_stands():
    n = make(lateral_offset=0.5)
    assert n.geometry((3.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0))[:2] == pytest.approx((3 - GOAL, 0.5), abs=1e-9)
    assert n.geometry((3.0, 0.0, math.pi / 2), (6.0, 0.0, math.pi))[:2] == pytest.approx((3 - GOAL, 0.5), abs=1e-9)


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
    assert 'no stable table pose' in result(n)['reason'] and cmd(n) == (0.0, 0.0, 0.0)
    face = sim.true_errors()[3]
    assert 1.6 < face < 2.1                                                                   # stopped at the window, not past it
    assert n.pubs['~/status'].msgs[-1].data == 'failed' and math.isnan(n.pubs['head_pitch_cmd'].msgs[-1].data)


# ---------------- closed loop ----------------

def run_nominal(**kw):
    n = make(**kw.pop('params', {}))
    sim = Sim(n, **{'robot': (0.0, -0.4, math.radians(8)), 'table': (3.8, 0.0, math.pi / 2), **kw})
    marks = {}
    sim.hooks.append(lambda s: marks.setdefault('acc', n.n_acc) if n.latched else None)
    start(n)
    return n, sim, marks, sim.run()


def test_walks_in_latches_in_the_window_and_arrives_on_odometry_with_the_camera_blind():
    n, sim, marks, phase = run_nominal()
    assert phase == 'succeeded'
    dx, dy, dyaw, face = sim.true_errors()
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
