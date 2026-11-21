import json

import numpy as np

import ros_stubs
from helpers import MSG, string, stamp


def make_node():
    import semantic_mask_node
    return semantic_mask_node.SemanticMaskNode()


def seg_msg(ids, w=4, h=3):
    m = MSG['Image']()
    m.header.stamp, m.height, m.width = stamp(), h, w
    m.data = np.array(ids, np.int32).reshape(h, w).tobytes()
    return m


LABELS = json.dumps({'0': {'class': 'BACKGROUND'}, '2': {'class': 'klt_box'}, '5': {'class': 'workstation'}})
IDS = [0, 2, 2, 5,  0, 2, 5, 5,  0, 0, 0, 5]


def mask(n):
    m = n.pubs['/segmentation_mask'].msgs[-1]
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width), m


def test_default_target_masks_klt_box():
    n = make_node()
    n.subs['/semantic_labels'](string(LABELS))
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    arr, m = mask(n)
    assert n.target == 'klt_box' and n.target_id == 2
    assert (arr == 255).sum() == 3 and arr[0, 1] == 255 and arr[0, 3] == 0
    assert m.encoding == 'mono8' and m.step == 4 and m.header.stamp.sec == 12


def test_switch_follows_new_label_immediately():
    n = make_node()
    n.subs['/semantic_labels'](string(LABELS))
    n.subs['/active_target'](string('workstation'))             # label map is cached -> no wait
    assert n.target == 'workstation' and n.target_id == 5
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    assert (mask(n)[0] == 255).sum() == 4


def test_no_mask_until_label_known_and_never_the_old_one():
    n = make_node()
    n.subs['/active_target'](string('workstation'))             # no label map yet
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    assert n.pubs['/segmentation_mask'].msgs == []
    n.subs['/semantic_labels'](string(LABELS))
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    assert len(n.pubs['/segmentation_mask'].msgs) == 1 and n.target_id == 5


def test_switch_to_label_missing_from_map_never_publishes_the_old_mask():
    n = make_node()
    n.subs['/semantic_labels'](string(json.dumps({'2': {'class': 'klt_box'}})))
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    assert len(n.pubs['/segmentation_mask'].msgs) == 1
    n.subs['/active_target'](string('workstation'))             # not in the map (yet)
    n.subs['/semantic_segmentation'](seg_msg(IDS))
    assert len(n.pubs['/segmentation_mask'].msgs) == 1 and n.target_id is None


def test_unknown_target_ignored_and_repeats_are_cheap():
    n = make_node()
    n.subs['/semantic_labels'](string(LABELS))
    n.subs['/active_target'](string('toaster'))
    assert n.target == 'klt_box' and n.log.has('error', 'toaster')
    n.subs['/active_target'](string('klt_box'))                 # same target: no reset, no log
    assert n.target_id == 2


def test_new_id_after_isaac_sim_restart():
    n = make_node()
    n.subs['/semantic_labels'](string(LABELS))
    n.subs['/semantic_labels'](string(json.dumps({'1': {'class': 'klt_box'}})))
    assert n.target_id == 1


def test_initial_target_parameter_and_bad_value():
    ros_stubs.Node.overrides = {'target': 'workstation'}
    assert make_node().target == 'workstation'
    ros_stubs.Node.overrides = {'target': 'nope'}
    try:
        make_node()
        assert False, 'expected SystemExit'
    except SystemExit:
        pass
