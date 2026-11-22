import pytest
import yaml
from types import SimpleNamespace as NS

from targets import find_label_id, load_targets, size_matches


def test_default_yaml_loads():
    t = load_targets()
    assert t['klt_box']['mode'] == 'grasp' and t['workstation']['mode'] == 'pose'
    assert t['klt_box']['size'] == [0.30, 0.48, 0.15]
    assert all(len(v.get('size') or []) == 3 for v in t.values())   # shipped targets keep the stale-detection gate on


@pytest.mark.parametrize('bad', [
    {'a': {'label': 'x'}},                                                   # no mode
    {'a': {'label': 'x', 'mode': 'fly'}},                                    # unknown mode
    {'a': {'label': 'x', 'mode': 'pose', 'foundationpose': {'symmetry_axes': []}}},   # untyped empty list
])
def test_load_rejects_bad_entries(tmp_path, bad):
    f = tmp_path / 't.yaml'
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_targets(f)


LABELS = {'0': {'class': 'BACKGROUND'}, '2': {'class': 'klt_box'}, '5': {'class': 'Workstation,table'},
          '7': {'class': 'workstation_lamp'}}


def test_find_label_id():
    assert find_label_id(LABELS, 'klt_box') == 2
    assert find_label_id(LABELS, 'workstation') == 5            # exact token beats 'workstation_lamp' (id 7)
    assert find_label_id(LABELS, 'lamp') == 7                   # substring fallback (old behaviour)
    assert find_label_id(LABELS, 'conveyor') is None
    assert find_label_id({'3': 'klt_box'}, 'klt_box') == 3      # plain string values
    assert find_label_id({'3': {'a': 'x', 'class': 'klt_box'}}, 'klt_box') == 3
    assert find_label_id({}, 'klt_box') is None


def test_find_label_id_reads_isaac_sim_5_1_string_entries():
    # real /semantic_labels from the lab: non-"class" instances arrive as "instance:label ..." strings
    real = {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}, '2': 'table:workstation wikidata_qcode:',
            '3': {'class': 'floor'}, '4': 'klt_box_1:KLT_box'}
    assert find_label_id(real, 'workstation') == 2 and find_label_id(real, 'klt_box') == 4
    # an older prim labelled simple_workstation with a LOWER id must not win over the exact label
    clash = {'2': 'simple_workstation:simple_workstation', '5': 'table:workstation wikidata_qcode:'}
    assert find_label_id(clash, 'workstation') == 5
    assert find_label_id(clash, 'station') == 2                   # substring fallback still works


def test_size_matches():
    s = NS(x=0.48, y=0.15, z=0.30)
    assert size_matches(s, [0.30, 0.48, 0.15])                  # axis order does not matter
    assert size_matches(s, [0.32, 0.46, 0.17])                  # within 3 cm
    assert not size_matches(s, [0.36, 0.48, 0.15])              # 6 cm off
    assert not size_matches(s, [0.8, 0.6, 0.9])
    assert size_matches(s, None)
