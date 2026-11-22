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
| `pose_overlay_visualizer_tracking_grasp.py` | **replaces yours**: grasp mode publishes the same poses (checked on 5 box yaws); adds pose mode, `/active_target`, size gate, skips NaN poses (~65 lines changed) |
| `target_switcher.py` | new, **apt2**: applies the target's mesh/symmetry to the running FoundationPose node |
| `check_mesh.py` | new: checks an OBJ (units, origin, size, parts), prints the `size:` line, fixes scale/origin, `--flatten` merges it into one part |

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

1. Isaac Sim: give the workstation the semantic label `workstation` (Semantics Schema Editor, as for the box; select
   the table prim in the Stage first, not the floor or the box). Check it with the table in the camera's view (the
   list seems to cover only what the camera sees) and the sim playing:
   `ros2 topic echo /semantic_labels --once --full-length` (`--full-length`: echo cuts long strings at 128 characters).
   A label that is not under `class` shows up as a plain string, e.g. `"2":"table:workstation wikidata_qcode:"`;
   the mask node reads both forms. A label added while the sim was open may only appear after restarting Isaac Sim.
2. Hide everything but the workstation, then File ▸ Export ▸ OBJ with only *Export Materials and Textures* and
   *Export Visible Only* ticked. Run `python3 check_mesh.py ws.obj` and, if it complains, fix it in one go:
   `python3 check_mesh.py ws.obj --scale 0.01 --recentre --flatten -o workstation_one.obj` (cm → m, origin to the bbox
   centre, one part). Z must be up. **The printed size must equal the object's size in the scene.** The Isaac Sim 5.1
   export left out the prim's Transform ▸ Scale (z 1.05626: 0.994 m in the file, 1.050 m in the scene), so pass it on:
   `python3 check_mesh.py ws.obj --scale 1 1 1.05626 --recentre --flatten -o workstation_one.obj` (`--scale` takes one
   value or `sx sy sz`; file name first). Read the prim's Scale in Property ▸ Transform before exporting.
   **One part only.** FoundationPose (Isaac ROS 4.4) reads only the first `o`/`g` part of an OBJ. The packing table
   exports as 4 parts (table top, bolts, body; the top first) and FoundationPose saw just the 4.8 cm top:
   `bbox.size` 0.761 × 2.439 × 0.047 m in `/tracking/output`, with a NaN and later a drifting pose. `check_mesh.py`
   prints the part count; `--flatten` writes one part with the first material and triangles only (vertices, UVs and
   normals unchanged; 93k faces become 177k triangles).
3. Copy the new .obj and the .mtl it names (`grep mtllib workstation_one.obj`) to apt2, **keeping both file names**
   (the .obj finds the .mtl by name):
   `rsync -av workstation_one.obj packing_workstation.mtl apt2@192.168.0.110:~/workspaces/isaac_ros-dev/workstation/`
4. `targets.yaml`: paste the printed `size: [...]` and set `symmetry_axes`: `['']` (none) for a detailed asset,
   `[x_180, y_180, z_180]` only for a plain cuboid. Packing workstation: 0.762 × 2.474 × 1.050 m (long side =
   mesh Y, Z up), 93k faces, no symmetry (add `z_180` only if its pose flips).
5. Check what FoundationPose really loaded, whenever the mesh changes:
   `ros2 topic echo /tracking/output --once | grep -E -A3 'position:|size:'`. `size` is the loaded mesh's bbox and
   must equal the `size:` line of step 4 (the position is NaN for a few seconds after a start).

## Mesh-switch test (do this before using the switcher)

KLT pipeline running as usual, the workstation filled in in `targets.yaml`, **without** `target_switcher.py`:
```bash
# apt2 container (domain 0)
ros2 param set /foundationpose_node mesh_file_path /workspaces/isaac_ros-dev/workstation/workstation_one.obj
ros2 param set /foundationpose_tracking_node mesh_file_path /workspaces/isaac_ros-dev/workstation/workstation_one.obj
# apt-ipc (domain 99)
ros2 topic pub -r 1 /active_target std_msgs/msg/String "{data: workstation}"
```
(`symmetry_axes` is `['']` for both targets, so there is nothing to set there.)
Good: `/tracking/output` `bbox.size` becomes the table's (checklist step 5), the visualizer logs `[TRACKING]` with
`box=(0.762×2.474×1.050)`, and the wireframe in `/tracking_visualization` sits on the workstation for 30 s.
Bad: it drifts or resets, or the visualizer logs `Ignoring detection with bbox 0.30x0.48x0.15` (the tracker still
has the box mesh) -> use the relaunch fallback.
Status (Nov 22): a **relaunch** with `mesh_file_path:=…/workstation_one.obj` detects the table (valid poses ~3.75 m
away). The **live switch is still untested**: the first attempt used the 4-part mesh, which FoundationPose cut down to
the table top, so it says nothing about the live switch.

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

## How FoundationPose works here (Isaac ROS 4.4)

```
RGB + depth + mask ─► Selector ─┬─► ESTIMATION  from scratch: several hundred pose guesses ─► refine ─► score ─► best
                                │               (GPU-heavy, needs only the mask; seconds with the table mesh) ──┐ pose
                                └─► TRACKING    refine network only, starts from last frame's pose, every frame ◄┘
                                                └─► /tracking/output   (what the visualizer reads)
```

- **Estimation** finds the object without any prior. **Tracking** nudges last frame's pose to fit the new frame: fast
  ("refine network only, no batching"), but it only fixes small motion per frame and can drift.
- The **Selector** runs an estimation first, then tracking. `reset_period` (Selector, ms; Isaac ROS default 20000, our
  launch 5000) forces a fresh estimation that often, to cancel drift. The tracker can also ask for an early reset when
  too few depth points support its pose (`enable_auto_reset`, `min_pointcloud_support` 50).
- `pose_estimation_timeout_ms` (estimation node, default 5000) is a watchdog on one estimation.
- Seen with the 177k-triangle table and `reset_period` 5000: `/tracking/output` valid for ~5 s, then NaN for 2–9 s,
  repeating. Reading: NaN while an estimation runs (or hits the 5 s watchdog); the light box mesh hides it. The
  visualizer skips NaN and keeps the last good pose.
- **Walking.** Tracking must follow large frame-to-frame motion with one refine step per frame, an estimation takes
  seconds and its result is already old when it arrives, and the pose is stamped in the pelvis frame of the image
  time. Live pose can lag during a walk: latch the pose while standing and walk on odometry
  (done in `arrival_navigator.md`).

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

`pytest tests/perception` (46 tests; the navigation tests are in `arrival_navigator.md`, no ROS needed: small stand-ins for rclpy/messages/TF). They pin the
grasp poses of the unmodified visualizer for 5 box yaws, the pose-mode transform and timestamp, the size
gate, NaN poses being skipped, the mask node switching, the label lookup (including the lab's real Isaac Sim 5.1
label map), the switcher's retry/lock behaviour and the mesh checker (including `--flatten`).
