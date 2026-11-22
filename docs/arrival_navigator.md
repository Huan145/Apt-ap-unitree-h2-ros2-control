# Arrival navigator: walk to the workstation with the box

`nodes/navigation/h2_arrival_navigator.py`, a sibling of `h2_local_navigator.py` (same services, `cmd_vel`,
`head_pitch_cmd`, velocity shaping, `[CMD]`/`[HB]` logs). The table is tracked in a **fixed frame**, the robot **stops
to latch** the table pose at 1.7 to 2.0 m, and the last stretch is walked on **odometry** because the camera is blocked
near the table (box in the hands, table filling the view).

```
/workstation_pose ─► map to odom with the pelvis pose AT IMAGE TIME ─► median of the last 1 s ─► table estimate
(pelvis frame,                                                                                       │
 image time)     SCAN ─► ROTATE ─► APPROACH ──(face ≤ 2.0 m)──► LATCH: stop, 1.5 s, poses taken after the stop,
                                                  halves agree ─► freeze ─► ROTATE ─► APPROACH ─► FINAL_ALIGN ─► done
                                                                  vision ignored from here, pelvis pose from TF (odometry)
```

## Which pose is "the last pose"

1. Walk on the live estimate until the table's **near face is ≤ `latch_far` (2.0 m)** ahead, then **stop**.
2. Wait `latch_settle_s` (1.5 s). Only poses whose **image stamp is after that** count (arrival time would let a
   late pose from the walk in).
3. Take the poses of the last `window_s` (1 s). Latch when there are ≥ `latch_samples` (5), they cover the full
   window, and the **first and second half agree** within 5 cm / 3° (a new estimation or a drift pulls the halves
   apart; plain jitter does not, whatever its size). The latched pose is the **median** of that window.
4. From then on vision is ignored. `~/table_pose` (odom frame) shows what was latched.

| stationary range | what happens |
|---|---|
| 1.55 to 2.15 m (window ± `latch_tol`) | latch |
| > 2.15 m (the live estimate stopped us early) | walk on, latch again (`latch_max_tries` 3, then latch with a warning) |
| < 1.55 m | latch with a warning (a close table may be partly out of view; it never backs up) |
| no stable poses within `latch_timeout_s` (20 s; tracker gaps of 2 to 9 s were seen) | **fail**: it never walks blind past the window |

**Why 1.7 to 2.0 m** (geometry from the camera intrinsics, fx 412.31, cx 360, so 82.3° horizontal; not measured with
FoundationPose): the 2.47 m long table is wholly in the image from 1.42 m if centred, from 1.76 m with 0.3 m sideways
error. At 2.0 m it spans 510 of 720 px. Closer than that the ends leave the image and the along-table position gets
unreliable.

**Why a fixed frame.** The table does not move, so every pose is mapped to `odom_frame` with the pelvis pose at its
image time. A late, dropped or NaN detection then cannot bend the walk, filtering over a second is free (no pelvis-frame
smearing while walking), and "walk on odometry" is the same code as before the latch with the estimate frozen.
**Watch `table@odom` in the `[HB]` line while it walks: it should hold within a few cm. If it wanders by 20 cm or more,
the tracker is lagging during the walk.**

## Run

Needs `/active_target` = `workstation` (the visualizer then publishes `/workstation_pose`) and a TF **fixed frame to
pelvis in which the pelvis moves when the robot walks**. `h2_local_navigator` needs none (it works in the pelvis frame
only: the box stays in view); this one must remember the table after the camera loses it. In Isaac Sim `pelvis` and
`Camera` hang off **`world`**, so use `-p odom_frame:=world` (on the real robot: the frame its odometry publishes).
Check what you have:

```bash
ros2 run tf2_ros tf2_echo world pelvis       # x, y must change when the robot walks
ros2 run tf2_tools view_frames               # lists the frames
```
```bash
# apt-ipc, every terminal (domain 99 = Isaac Sim)
conda activate h2_rl && source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=99

# terminal 1: keep the target on the workstation until the orchestrator does it
ros2 topic pub -r 1 /active_target std_msgs/msg/String "{data: workstation}"

# terminal 2: the navigator
cd ~/Github/h2-ros2-control/nodes/navigation
python3 h2_arrival_navigator.py --ros-args -p odom_frame:=world

# terminal 3: start / cancel (start is refused with the frames TF knows if odom_frame is wrong)
ros2 service call /h2_arrival_navigator/start std_srvs/srv/Trigger
ros2 service call /h2_arrival_navigator/cancel std_srvs/srv/Trigger
```
Floats need a decimal point (`-p latch_far:=2.0`). `-p log_poses:=true` prints every accepted pose in odom coordinates.
Log landmarks: `[SCAN] table found`, `[LATCH] try 1 … standing still`, `[LATCH] table at odom …`, `[PHASE]`, `[DONE]`.

**Stopping.** `cancel` (or Ctrl-C / SIGTERM: the node sends zero velocity 5 times and releases the head before it
exits; a second Ctrl-C meanwhile is ignored). The
free-box controller keeps the **last `cmd_vel` forever** (no timeout; the head override times out after 1 s), so a stop
that never leaves the node keeps the robot turning. If that happens (log line `[SHUTDOWN] stop command NOT sent`, or
a robot that turns on after the node is gone):
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

## Parameters that matter

| parameter | default | note |
|---|---|---|
| `odom_frame` | `odom` | the fixed frame in TF: **`world` in Isaac Sim** (pass `-p odom_frame:=world`) |
| `place_standoff` | 0.50 | pelvis → table near face at the end. **Placeholder**: the carry pose of `H2_pickup_sequencer` (grip x 0.20 m, box 0.30 m deep) puts the box front 0.35 m ahead of the pelvis, + 0.15 m. Set it from the walking-with-box pose and what the place step needs |
| `lateral_offset` | 0.0 | stand this far to the robot's **left** of the table centre |
| `table_depth` | 0.762 | `targets.yaml` size[0]; the robot faces a long side (mesh Y) |
| `latch_far` / `latch_near` | 2.0 / 1.7 | window, pelvis → table face |
| `latch_settle_s`, `window_s`, `latch_samples` | 1.5, 1.0, 5 | |
| `latch_pos_tol`, `latch_yaw_tol` | 0.05 m, 3° | how well the two halves must agree |
| `scan_wait_s`, `scan_w`, `scan_dir` | 3.0, 0.25, +1 | stand still, then turn left at 0.25 rad/s; fails after `scan_max_deg` 400 |
| `v_min v_max w_min w_max kp_* *_tol` | as `h2_local_navigator` | tuned there for the **no-box** policy; check them with the box |
| `head_*` | as `h2_local_navigator` | aims at the table centre's **elevation** (independent of where the robot faces, so a table far off to the side no longer pins the head at the limit); held after the latch. The free-box policy was trained with head pitch 0: if it walks badly with the head down, cap it with `-p head_pitch_max:=0.3` |

## Interfaces

`~/start`, `~/cancel` (Trigger). `~/status` (latched: `idle`/`running`/`succeeded`/`failed`). `~/result` (latched JSON):
success carries `face_dist_m` (pelvis → table face, what the place step needs), `dyaw_deg`, `table_in_pelvis_m`,
`goal_error_m`, `latched_range_m`, `latch_tries`, `latch_samples`, `latch_drift_mm/deg`; failure carries `reason` and
`phase`. `~/table_pose` (latched PoseStamped in odom, x along the table's long axis), `~/target_point` (goal in the
pelvis frame). A crash in the loop stops the robot and fails the run.

## Tests

`pytest tests` (77 in all; 31 in `tests/navigation`, no ROS needed). They drive the node in a small kinematic world
(`navsim.py`): velocity lag 0.25 s, vision 0.3 s late and noisy, camera blind below 1.2 m, tracker gaps, 180° flips,
a re-estimation jump inside the latch window, odometry loss. They check arrival within 8 cm / 6° of the goal (10 cm / 7° for the noisy and sideways-start cases), the latch
in the window with the robot standing still, no vision used after the latch, retry, scan, the head angle for a table
off to the side, the stop at shutdown (`main()` on a stubbed rclpy: the stop goes out before the context ends), and
every failure path. 20 deliberate breakages of the node (image-time TF, the settle rule, the retry, the axis average,
the head formula, the stop, the signal handling, …) are all caught.

**First real run (Nov 22, Isaac Sim, free-box controller).** Two bugs came out of it, both fixed: the head went to its
28.6° limit (`atan2(y, z)` of a table 77° off to the side), and Ctrl-C never sent the stop (rclpy had already shut the
context down; the error was swallowed), so the controller kept `wz` −0.5 and the robot started turning once the node
was gone. During the run the yaw barely moved (−15°, then flat for 8 s at `wz` −0.5). The head at 28.6° while the
policy was trained with head 0 is a candidate reason, not confirmed: check that the robot turns with the new, smaller
head angle.

**Not verified** (no ROS, GPU or robot here): real tracker noise and lag, how the free-box policy follows `cmd_vel`
with the head tilted, the real `place_standoff`, and `main()` on a real rclpy (its signal handling follows the rclpy
source and is tested on a stub). The
table's pose along its long axis can be off by ~0.3 m between FoundationPose estimations; this node cannot fix that,
it only keeps the robot from chasing it.
