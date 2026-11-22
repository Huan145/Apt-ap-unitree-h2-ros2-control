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


# Isaac Sim style export of a multi-mesh asset: one g part per mesh, one material each, unwelded corners
SQUARE = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
PENTAGON = [(np.cos(a), np.sin(a), 0.0) for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
TRIANGLE = [(2, 0, 0), (3, 0, 0), (2, 1, 0)]
PARTS = [('top', 'wood', [SQUARE]), ('body', 'paint', [PENTAGON, TRIANGLE]), ('bolts', 'metal', [TRIANGLE])]


def multi_part_obj():
    lines, n = ['mtllib t.mtl'], 0
    for name, mtl, polys in PARTS:
        lines += [f'g {name}', *[f'v {x} {y} {z}' for p in polys for x, y, z in p], 'vt 0 0', 'vn 0 0 1', f'usemtl {mtl}']
        for p in polys:
            lines.append('f ' + ' '.join(f'{n + i + 1}/1/1' for i in range(len(p))))
            n += len(p)
    return '\n'.join(lines) + '\n'


def vector_area(loop):
    """Half the summed cross products of a closed loop: the same for any triangulation of it."""
    loop = np.asarray(loop, float)
    return 0.5 * np.cross(loop, np.roll(loop, -1, axis=0)).sum(0)


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


def test_report_flags_a_multi_part_mesh(tmp_path):
    f = tmp_path / 'ws.obj'
    f.write_text(multi_part_obj())
    r = run(str(f))
    assert '3 parts' in r.stdout and '--flatten' in r.stdout           # FoundationPose 4.4 would load only 'top'
    f.write_text(CUBE)
    r = run(str(f))
    assert '1 part,' in r.stdout and '--flatten' not in r.stdout


def test_flatten_makes_one_part_one_material_and_triangles(tmp_path):
    f, out, again = tmp_path / 'ws.obj', tmp_path / 'one.obj', tmp_path / 'again.obj'
    f.write_text(multi_part_obj())
    r = run(str(f), '--flatten', '-o', str(out))
    assert r.returncode == 0 and '(one part, 7 triangles)' in r.stdout       # quad 2 + pentagon 3 + 2 triangles
    lines = out.read_text().splitlines()
    assert not [l for l in lines if l.split()[0] in ('g', 'o')]
    assert [l for l in lines if l.startswith('usemtl')] == ['usemtl wood']     # the first material
    faces = [l.split() for l in lines if l.startswith('f ')]
    assert len(faces) == 7 and all(len(t) == 4 for t in faces)
    keep = ('mtllib', 'vt', 'vn')                                              # UVs and normals untouched
    assert [l for l in lines if l.split()[0] in keep] == [l for l in multi_part_obj().splitlines() if l.split()[0] in keep]
    verts = np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')])
    np.testing.assert_allclose(verts, [c for _, _, polys in PARTS for p in polys for c in p], atol=1e-6)   # same order
    got = sum(vector_area([verts[int(c.split('/')[0]) - 1] for c in t[1:]]) for t in faces)
    np.testing.assert_allclose(got, sum(vector_area(p) for _, _, polys in PARTS for p in polys), atol=1e-5)
    assert run(str(out), '--flatten', '-o', str(again)).returncode == 0 and again.read_text() == out.read_text()
    assert '--flatten' not in run(str(out)).stdout                             # one part now: no warning


def test_flatten_needs_an_output_file(tmp_path):
    f = tmp_path / 'ws.obj'
    f.write_text(multi_part_obj())
    assert run(str(f), '--flatten').returncode != 0
