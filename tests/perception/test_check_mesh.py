import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'nodes' / 'perception' / 'check_mesh.py'

# KLT-like box in centimetres, shifted off-centre, with texture coords, normals and a vertex colour column
CUBE = '\n'.join([
    'mtllib box.mtl', 'o box',
    *[f'v {x} {y} {z}' + (' 0.5 0.5 0.5' if i == 0 else '')
      for i, (x, y, z) in enumerate((x, y, z) for x in (10, 40) for y in (-5, 43) for z in (0, 15))],
    'vt 0 0', 'vn 0 0 1', 'f 1/1/1 2/1/1 4/1/1 3/1/1', 'f 5/1/1 6/1/1 8/1/1 7/1/1', ''])


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_report_flags_cm_and_offcentre(tmp_path):
    f = tmp_path / 'box.obj'
    f.write_text(CUBE)
    r = run(str(f))
    assert r.returncode == 0
    assert 'size   (x y z): 30.000 48.000 15.000' in r.stdout
    assert 'cm or mm' in r.stdout and '--recentre' in r.stdout      # 48 "m" long, bbox centre far from the origin


def test_scale_and_recentre_write_a_metre_mesh(tmp_path):
    f, out = tmp_path / 'box.obj', tmp_path / 'box_m.obj'
    f.write_text(CUBE)
    r = run(str(f), '--scale', '0.01', '--recentre', '-o', str(out))
    assert r.returncode == 0 and 'size: [0.300, 0.480, 0.150]' in r.stdout and '!' not in r.stdout
    lines = out.read_text().splitlines()
    v = np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')])
    np.testing.assert_allclose(v.max(0) - v.min(0), [0.30, 0.48, 0.15], atol=1e-6)
    np.testing.assert_allclose((v.max(0) + v.min(0)) / 2, 0, atol=1e-6)
    assert [l for l in lines if l.startswith('v ')][0].endswith('0.5 0.5 0.5')     # extra columns kept
    assert [l for l in lines if not l.startswith('v ')] == [l for l in CUBE.splitlines() if not l.startswith('v ')]


def test_per_axis_scale_applies_a_prim_scale_the_export_left_out(tmp_path):
    f, out = tmp_path / 'box.obj', tmp_path / 'box_m.obj'
    f.write_text(CUBE)
    r = run(str(f), '--scale', '0.01', '0.01', '0.012', '--recentre', '-o', str(out))
    assert r.returncode == 0 and 'size: [0.300, 0.480, 0.180]' in r.stdout      # z: 15 cm * 0.012
    v = np.array([[float(x) for x in l.split()[1:4]] for l in out.read_text().splitlines() if l.startswith('v ')])
    np.testing.assert_allclose(v.max(0) - v.min(0), [0.30, 0.48, 0.18], atol=1e-6)
    assert run(str(f), '--scale', '1', '2').returncode != 0                     # one value or three


def test_no_vertices_is_an_error(tmp_path):
    f = tmp_path / 'empty.obj'
    f.write_text('o nothing\n')
    assert run(str(f)).returncode != 0
