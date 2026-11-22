"""Target registry shared by the mask node, the visualizer and the apt2 switcher (pure Python, no ROS).

targets.yaml maps a target name (the string published on /active_target) to its semantic label,
mode, expected bbox size and the FoundationPose parameters to apply at runtime.
"""
import re
from pathlib import Path

import yaml

DEFAULT_FILE = Path(__file__).with_name('targets.yaml')
MODES = ('grasp', 'pose')


def load_targets(path=None):
    with open(path or DEFAULT_FILE) as f:
        targets = yaml.safe_load(f)
    for name, t in targets.items():
        if not {'label', 'mode'} <= t.keys() or t['mode'] not in MODES:
            raise ValueError(f"{name}: needs 'label' and 'mode' ({'|'.join(MODES)})")
        for k, v in t.get('foundationpose', {}).items():
            if v == []:   # ROS cannot type an empty list
                raise ValueError(f"{name}.foundationpose.{k}: empty list - use [''] for none")
    return targets


def find_label_id(label_map, label):
    """Semantic ID of `label` in Isaac Sim's label map, or None. A value is {"class": "a,b"} or, for another
    semantic instance, a plain string like "table:workstation wikidata_qcode:" (Isaac Sim 5.1).
    Exact match on a comma/colon/space-separated label first, then substring (the old behaviour)."""
    want = label.lower()
    exact = lambda v: want in [s.strip() for s in v.split(',')] + re.split(r'[\s,:]+', v)
    for match in (exact, lambda v: want in v):
        for id_str, d in label_map.items():
            if any(match(str(v).lower()) for v in (d.values() if isinstance(d, dict) else [d])):
                return int(id_str)


def size_matches(size, expected, tol=0.03):
    """True if a detection's bbox size (x, y, z attrs) equals `expected` [x, y, z] in any axis order.
    expected=None accepts everything."""
    if expected is None:
        return True
    return all(abs(a - b) <= tol for a, b in zip(sorted((size.x, size.y, size.z)), sorted(expected)))
