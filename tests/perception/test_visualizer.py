import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as Rot

import ros_stubs
from helpers import (R_PELVIS_CAM, T_PELVIS_CAM, cam_info, det_msg, obj_in_cam, rgb_msg, stamp, string,
                     tf_stamped, xyz)
from ros_stubs import Buffer

KLT = (0.30, 0.48, 0.15)
WS = (0.80, 0.60, 0.90)

# left/right grasp poses (x y z qx qy qz qw, pelvis frame) of the visualizer BEFORE target switching was
# added, for a 0.30x0.48x0.15 box centred at (0.75, 0.05, 0.25) with the given yaw in the pelvis frame
GOLDEN = {
    'square': {
        '/target_pose_left': [0.66, 0.34, 0.27, 0.0, 0.0, 0.0, 1.0],
        '/target_pose_right': [0.66, -0.24, 0.27, 0.0, 0.0, 0.0, 1.0],
    },
    'yaw20': {
        '/target_pose_left': [0.560814, 0.322511, 0.27, 0.0, -0.0, 0.173648, 0.984808],
        '/target_pose_right': [0.759186, -0.222511, 0.27, 0.0, -0.0, 0.173648, 0.984808],
    },
    'yaw-35': {
        '/target_pose_left': [0.826337, 0.287554, 0.27, 0.0, -0.0, -0.300706, 0.953717],
        '/target_pose_right': [0.493663, -0.187554, 0.27, 0.0, -0.0, -0.300706, 0.953717],
    },
    'yaw20_flipped': {            # FoundationPose may flip the box Y sign between frames
        '/target_pose_left': [0.560814, 0.322511, 0.27, -0.0, 0.0, 0.173648, 0.984808],
        '/target_pose_right': [0.759186, -0.222511, 0.27, -0.0, 0.0, 0.173648, 0.984808],
    },
    'yaw80': {
        '/target_pose_left': [0.374406, 0.100358, 0.27, 0.0, 0.0, 0.642788, 0.766044],
        '/target_pose_right': [0.945594, -0.000358, 0.27, 0.0, 0.0, 0.642788, 0.766044],
    },
}
YAW = {'square': (0, 0), 'yaw20': (20, 0), 'yaw-35': (-35, 0), 'yaw20_flipped': (20, 180), 'yaw80': (80, 0)}


@pytest.fixture
def targets_file(tmp_path):
    f = tmp_path / 'targets.yaml'
    f.write_text(yaml.safe_dump({
        'klt_box': {'label': 'klt_box', 'mode': 'grasp', 'size': list(KLT)},
        'workstation': {'label': 'workstation', 'mode': 'pose', 'size': list(WS)},
        'unsized': {'label': 'u', 'mode': 'pose', 'size': None},
    }))
    return str(f)


@pytest.fixture
def node(targets_file):
    import pose_overlay_visualizer_tracking_grasp as viz
    ros_stubs.Node.overrides = {'targets_file': targets_file}
    Buffer.T = tf_stamped()
    n = viz.PoseOverlayVisualizerTracking()
    n.subs['/camera_info'](cam_info())
    return n


def feed(n, size, yaw=0, p_pelvis=(0.75, 0.05, 0.25), st=None):
    R, t = obj_in_cam(Rot.from_euler('z', yaw, degrees=True).as_matrix(), p_pelvis)
    n.subs['/tracking/output'](det_msg(R, t, size, st))


def frame(n):
    n.last_publish_time = 0.0       # bypass the 20 Hz cap
    n.subs['/rgb'](rgb_msg())


def msgs(n, topic):
    return n.pubs[topic].msgs


# ── grasp mode: unchanged behaviour ───────────────────────────────────────────

@pytest.mark.parametrize('case', GOLDEN)
def test_grasp_mode_matches_pre_refactor_poses(node, case):
    yaw, extra = YAW[case]
    feed(node, KLT, yaw + extra)
    frame(node)
    for topic, want in GOLDEN[case].items():
        m = msgs(node, topic)[0]
        got = [*xyz(m.pose.position), m.pose.orientation.x, m.pose.orientation.y,
               m.pose.orientation.z, m.pose.orientation.w]
        np.testing.assert_allclose(got, want, atol=2e-6)
        assert m.header.frame_id == 'pelvis' and m.header.stamp.nanosec == 700   # image stamp, as before
    assert msgs(node, '/workstation_pose') == []
    assert len(msgs(node, '/tracking_visualization')) == 1


# ── pose mode ─────────────────────────────────────────────────────────────────

def test_pose_mode_publishes_object_pose_in_pelvis_frame(node):
    node.subs['/active_target'](string('workstation'))
    feed(node, WS, yaw=30, p_pelvis=(1.5, -0.2, 0.45), st=stamp(40, 123))
    m, = msgs(node, '/workstation_pose')
    np.testing.assert_allclose(xyz(m.pose.position), [1.5, -0.2, 0.45], atol=1e-9)
    q = m.pose.orientation
    np.testing.assert_allclose(Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True),
                               [0, 0, 30], atol=1e-6)
    assert m.header.frame_id == 'pelvis' and (m.header.stamp.sec, m.header.stamp.nanosec) == (40, 123)


def test_pose_mode_never_publishes_grasp_targets_and_never_republishes_stale_poses(node):
    node.subs['/active_target'](string('workstation'))
    feed(node, WS)
    for _ in range(5):
        frame(node)
    assert len(msgs(node, '/workstation_pose')) == 1          # only when a new detection arrives
    assert msgs(node, '/target_pose_left') == [] and msgs(node, '/target_pose_right') == []
    assert len(msgs(node, '/tracking_visualization')) == 5    # overlay keeps running


def test_pose_mode_without_tf_publishes_nothing(node):
    node.subs['/active_target'](string('workstation'))
    Buffer.T = None
    feed(node, WS)
    assert msgs(node, '/workstation_pose') == []


# ── switching ────────────────────────────────────────────────────────────────

def test_size_gate_drops_stale_detections_after_a_switch(node):
    feed(node, KLT)
    node.subs['/active_target'](string('workstation'))
    assert node.latest_pose is None                           # old target's pose dropped
    feed(node, KLT)                                           # tracking still emits the box for a while
    assert node.latest_pose is None and msgs(node, '/workstation_pose') == []
    assert node.log.has('warn', 'Ignoring detection')
    feed(node, WS)
    assert node.latest_pose is not None and len(msgs(node, '/workstation_pose')) == 1


def nan_det(size, position=True, orientation=True):
    """What FoundationPose publishes while it re-estimates: NaN pose (both parts), normal bbox."""
    m = det_msg(np.eye(3), (np.nan if position else 1.0,) * 3, size)
    o = m.detections[0].results[0].pose.pose.orientation
    if orientation:
        o.x = o.y = o.z = o.w = float('nan')
    return m


@pytest.mark.parametrize('position, orientation', [(True, True), (True, False), (False, True)])
def test_nan_pose_is_ignored_and_the_last_good_pose_kept(node, position, orientation):
    node.subs['/active_target'](string('workstation'))
    feed(node, WS, p_pelvis=(1.5, -0.2, 0.45))
    good = node.latest_pose
    node.subs['/tracking/output'](nan_det(WS, position, orientation))
    assert node.latest_pose is good and len(msgs(node, '/workstation_pose')) == 1     # nothing new published
    assert node.log.has('info', 'NaN pose')
    frame(node)
    assert len(msgs(node, '/tracking_visualization')) == 1                            # overlay keeps running


def test_nan_pose_never_reaches_the_grasp_topics(node):
    feed(node, KLT)
    frame(node)
    node.subs['/tracking/output'](nan_det(KLT))
    frame(node)
    for topic in ('/target_pose_left', '/target_pose_right'):
        got = msgs(node, topic)
        assert len(got) == 2 and all(np.isfinite(xyz(m.pose.position)).all() for m in got)


def test_nan_before_the_first_pose_keeps_waiting(node):
    node.subs['/tracking/output'](nan_det(KLT))
    assert node.latest_pose is None


def test_switch_back_resumes_grasp_output(node):
    node.subs['/active_target'](string('workstation'))
    feed(node, WS)
    node.subs['/active_target'](string('klt_box'))
    feed(node, KLT)
    frame(node)
    assert len(msgs(node, '/target_pose_left')) == 1 and len(msgs(node, '/workstation_pose')) == 1


def test_target_without_size_accepts_anything_but_warns(node):
    node.subs['/active_target'](string('unsized'))
    assert node.log.has('warn', 'no size')
    feed(node, KLT)
    assert node.latest_pose is not None


def test_unknown_target_keeps_current_one(node):
    node.subs['/active_target'](string('toaster'))
    assert node.target == 'klt_box' and node.log.has('error', 'toaster')


def test_republished_target_does_not_reset_pose(node):
    feed(node, KLT)
    node.subs['/active_target'](string('klt_box'))            # the orchestrator's 1 Hz republish
    assert node.latest_pose is not None


def test_initial_target_parameter(targets_file):
    import pose_overlay_visualizer_tracking_grasp as viz
    ros_stubs.Node.overrides = {'targets_file': targets_file, 'target': 'workstation'}
    assert viz.PoseOverlayVisualizerTracking().cfg['mode'] == 'pose'
    ros_stubs.Node.overrides = {'targets_file': targets_file, 'target': 'nope'}
    with pytest.raises(SystemExit):
        viz.PoseOverlayVisualizerTracking()
