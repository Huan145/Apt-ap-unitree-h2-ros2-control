#!/usr/bin/env python3
"""
Check (and optionally fix) an OBJ mesh before giving it to FoundationPose.

  python3 check_mesh.py workstation.obj
  python3 check_mesh.py workstation.obj --scale 0.01 --recentre -o workstation_metres.obj
  python3 check_mesh.py workstation.obj --scale 1 1 1.05626 --recentre -o workstation_metres.obj

Prints size, bbox centre and unit/origin sanity checks, plus the `size:` line for targets.yaml.
  --scale S [S S]  multiply all vertices by S (0.01: cm -> m, 0.001: mm -> m) or per axis by "sx sy sz"
                   (file name first). The printed size must equal the object's size in the scene: the
                   Isaac Sim 5.1 OBJ export left out a prim's Transform > Scale, so pass it here.
  --recentre       move the bbox centre to the origin (poses and the wireframe refer to the mesh origin)
  -o OUT           write the fixed mesh (faces, normals, UVs and materials untouched; normals are not
                   rescaled, which is fine for a small per-axis scale)
Make sure Z is up (e.g. in MeshLab) - the mesh frame is the frame FoundationPose reports.
"""

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('obj')
    ap.add_argument('--scale', type=float, nargs='+', default=[1.0], metavar='S')
    ap.add_argument('--recentre', action='store_true')
    ap.add_argument('-o', '--out')
    a = ap.parse_args()
    if len(a.scale) not in (1, 3):
        ap.error('--scale takes one value or three (x y z)')
    scale = a.scale * 3 if len(a.scale) == 1 else a.scale

    lines = open(a.obj).read().splitlines()
    idx = [i for i, l in enumerate(lines) if l.lstrip().startswith('v ')]
    if not idx:
        raise SystemExit(f'{a.obj}: no vertices found')
    v = np.array([[float(x) for x in lines[i].split()[1:4]] for i in idx]) * scale
    if a.recentre:
        v -= (v.min(0) + v.max(0)) / 2
    lo, hi = v.min(0), v.max(0)
    size, ctr = hi - lo, (hi + lo) / 2
    n_faces = sum(l.lstrip().startswith('f ') for l in lines)

    print(f'{a.obj}: {len(v)} vertices, {n_faces} faces, '
          f'{"textured" if any(l.startswith("mtllib") for l in lines) else "no material"}')
    print(f'  size   (x y z): {size[0]:.3f} {size[1]:.3f} {size[2]:.3f} m')
    print(f'  centre (x y z): {ctr[0]:.3f} {ctr[1]:.3f} {ctr[2]:.3f} m')
    if size.max() > 5:
        print('  ! larger than 5 m: probably cm or mm -> --scale 0.01 (cm) / 0.001 (mm)')
    if size.max() < 0.02:
        print('  ! smaller than 2 cm: probably scaled twice')
    if np.abs(ctr).max() > 0.01:
        print(f'  ! bbox centre is {np.linalg.norm(ctr):.3f} m from the origin: the pose and the wireframe '
              'refer to the origin -> --recentre')
    if n_faces > 100_000:
        print('  ! more than 100k faces: simplify (MeshLab: Quadric Edge Collapse) to keep FoundationPose fast')
    print(f'  targets.yaml:  size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]')

    if a.out:
        for i, p in zip(idx, v):
            lines[i] = 'v ' + ' '.join([f'{x:.6f}' for x in p] + lines[i].split()[4:])
        with open(a.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'  wrote {a.out}')


if __name__ == '__main__':
    main()
