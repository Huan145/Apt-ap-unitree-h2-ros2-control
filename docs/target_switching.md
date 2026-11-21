# Target switching: KLT box ⇄ workstation

One FoundationPose pipeline, one target at a time. The orchestrator publishes the target name on
`/active_target` (`std_msgs/String`, **republish at ~1 Hz**); everything else follows.

```
orchestrator ──/active_target──┬─► semantic_mask_node        label → mask            (apt-ipc, domain 99)
                               ├─► pose_overlay_visualizer   grasp | pose mode       (apt-ipc, domain 99)
                               └─► domain_bridge 99→0 ─► target_switcher ─► ros2 param set /foundationpose_node
                                                                              mesh_file_path, symmetry_axes   (apt2, domain 0)
```

| target        | mode    | publishes                                                         |
|---------------|---------|-------------------------------------------------------------------|
| `klt_box`     | `grasp` | `/target_pose_left`, `/target_pose_right` (unchanged)             |
| `workstation` | `pose`  | `/workstation_pose` – mesh-origin pose in the **pelvis** frame, stamped with the detection's image time, sent once per new detection (a lost track stops the stream instead of repeating an old pose) |

## Files (`nodes/perception/`)

| file | what |
|---|---|
| `targets.yaml`, `targets.py` | the registry: label, mode, expected bbox `size`, FoundationPose params per target |
| `semantic_mask_node.py` | **replaces yours**: label from the active target, exact label match (substring fallback), cached label map so a switch takes effect immediately |
| `pose_overlay_visualizer_tracking_grasp.py` | **replaces yours**: grasp mode publishes the same poses (checked on 5 box yaws); adds pose mode, `/active_target`, size gate (~60 lines changed) |
| `target_switcher.py` | new, **apt2**: applies the target's mesh/symmetry to the running FoundationPose node |
| `check_mesh.py` | new: checks an OBJ (units, origin, size), prints the `size:` line, fixes scale/origin |

Put `targets.py`, `targets.yaml` and `target_switcher.py` on apt2 too. From `nodes/perception/`:
`rsync -av targets.py targets.yaml target_switcher.py apt2@192.168.0.110:~/workspaces/isaac_ros-dev/`
(that is `/workspaces/isaac_ros-dev/` inside the container).

## Two small edits to files I could not see in their current state

**`bridge_config.yaml`** – next to `/rgb` (99 → 0), then restart the bridge:
```yaml
  /active_target:
    type: std_msgs/msg/String
    from_domain: 99
    to_domain: 0
```

**`foundationpose_klt_tracking.launch.py`** – the node default `max_depth` is 2.0 m, so a workstation seen
from further away is cut off. Add to the constants and to the parameters of **both** `foundationpose_node` and
`foundationpose_tracking_node`:
```python
MIN_DEPTH, MAX_DEPTH = 0.1, 4.0          # top of the file (floats, with the decimal point)
...
            'min_depth': MIN_DEPTH,
            'max_depth': MAX_DEPTH,
```

## Workstation checklist

1. Isaac Sim: give the workstation the semantic label `workstation` (Semantics Schema Editor, as for the box).
2. Hide everything but the workstation, then File ▸ Export ▸ OBJ with only *Export Materials and Textures* and
   *Export Visible Only* ticked. Run `python3 check_mesh.py ws.obj` and, if it complains,
   `python3 check_mesh.py ws.obj --scale 0.01 --recentre -o workstation_metres.obj` (cm → m, origin to the bbox centre).
   Z must be up. **The printed size must equal the object's size in the scene.** The Isaac Sim 5.1 export left out
   the prim's Transform ▸ Scale (z 1.05626: 0.994 m in the file, 1.050 m in the scene), so pass it on:
   `python3 check_mesh.py ws.obj --scale 1 1 1.05626 --recentre -o workstation_metres.obj` (`--scale` takes one value
   or `sx sy sz`; file name first). Read the prim's Scale in Property ▸ Transform before exporting.
3. Copy the new .obj and the .mtl it names (`grep mtllib workstation_metres.obj`) to apt2, **keeping both file names**
   (the .obj finds the .mtl by name):
   `rsync -av workstation_metres.obj packing_workstation.mtl apt2@192.168.0.110:~/workspaces/isaac_ros-dev/workstation/`
4. `targets.yaml`: paste the printed `size: [...]` and set `symmetry_axes`: `['']` (none) for a detailed asset,
   `[x_180, y_180, z_180]` only for a plain cuboid. Packing workstation: 0.762 × 2.474 × 1.050 m (long side =
   mesh Y, Z up), 93k faces, no symmetry (add `z_180` only if its pose flips).

## Mesh-switch test (do this before using the switcher)

KLT pipeline running as usual, the workstation filled in in `targets.yaml`, **without** `target_switcher.py`:
```bash
# apt2 container (domain 0)
ros2 param set /foundationpose_node mesh_file_path /workspaces/isaac_ros-dev/workstation/workstation_metres.obj
ros2 param set /foundationpose_tracking_node mesh_file_path /workspaces/isaac_ros-dev/workstation/workstation_metres.obj
# apt-ipc (domain 99)
ros2 topic pub -r 1 /active_target std_msgs/msg/String "{data: workstation}"
```
(`symmetry_axes` is `['']` for both targets, so there is nothing to set there.)
Good: the wireframe in `/tracking_visualization` sits on the workstation and stays there for 30 s.
Bad: it drifts or resets, or the visualizer logs `Ignoring detection with bbox 0.30x0.48x0.15` (the tracker still
has the box mesh) -> use the relaunch fallback.

## Run and test

```bash
# apt-ipc (domain 99)         conda activate ik_env && source /opt/ros/jazzy/setup.bash
python3 -c "import yaml" || pip install pyyaml          # once, in ik_env (and in the apt2 container if it complains)
python3 semantic_mask_node.py
python3 pose_overlay_visualizer_tracking_grasp.py
# apt2 container (domain 0)   source /opt/ros/jazzy/setup.bash
python3 target_switcher.py

# apt-ipc: flip the target by hand (what the orchestrator will do)
ros2 topic pub -r 1 /active_target std_msgs/msg/String "{data: workstation}"
ros2 topic echo /workstation_pose --once
```
Expected: mask node `Active target: workstation` + `Found "workstation" → semantic ID=…`; switcher
`Target "workstation" applied`; `/workstation_pose` appears once FoundationPose has re-estimated. Confirm on apt2:
`ros2 param get /foundationpose_node mesh_file_path`.

## Behaviour to know

- **Switch latency**: after the mesh changes, the Selector re-estimates at its next `reset_period` (5 s in your
  launch). Until then the tracker still emits the old target; the visualizer drops those via `size`
  (±3 cm, any axis order; `size: null` = no filter, with a warning). If it logs `Ignoring detection with
  bbox …` for the box you actually want, copy that bbox into `size:`. The orchestrator should wait for the
  first `/workstation_pose` and for a few consistent ones before latching it.
- **Symmetric meshes** can flip 180° about an axis between frames. For navigation use the workstation's
  position and its *axis* nearest the robot, not the sign of the axes.
- **Mesh weight.** The packing workstation is 93k faces / 364k vertices. If FoundationPose is slow with it or
  fails to load it, decimate it in MeshLab (Filters ▸ Remeshing, Simplification and Reconstruction ▸ Quadric Edge
  Collapse Decimation, then File ▸ Export Mesh As .obj). NVIDIA's tutorial suggests this to cut the rendering
  cost but names no target face count. Then run `check_mesh.py --recentre` on the result again.
- The switcher retries every message until FoundationPose accepts the parameters (errors are logged with the
  node's reason). Empty lists have no ROS type: "no symmetry" is `['']`, the node default.
- **Isaac ROS 4.4 caveat.** NVIDIA's 4.4 tutorial (*switch the mesh during runtime*) says the live switch "is only
  supported for the pose estimation node" and "may lead to unexpected behavior if used with the tracking node";
  the latest (5.x) docs say both nodes follow. `target_switcher.py` sets only `/foundationpose_node`, so on 4.4
  the tracker may keep the old mesh. Run the test below before relying on it. Fallback if it fails: the switcher
  restarts the FoundationPose launch with the other mesh (tens of seconds, hidden while the robot walks).
- Not verified on the real system (no GPU/ROS in the sandbox): how long the first workstation detection takes,
  and whether `MAX_DEPTH` 4.0 m is enough for a 2.5 m object seen from a distance (raise it if the far end is cut off).

## Tests

`pytest tests/perception` (37 tests, no ROS needed: small stand-ins for rclpy/messages/TF). They pin the
grasp poses of the unmodified visualizer for 5 box yaws, the pose-mode transform and timestamp, the size
gate, the mask node switching, the switcher's retry/lock behaviour and the mesh checker.
