# Arrival navigator: walk to the workstation with the box

`nodes/navigation/h2_arrival_navigator.py`, a sibling of `h2_local_navigator.py` (same services, `cmd_vel`,
`head_pitch_cmd`, velocity shaping, `[CMD]`/`[HB]` logs). The robot comes in along the table's **bisector** (the line
through the table centre, normal to its long side), **stops to latch** the table pose 1.7 to 2.0 m from the face, and
walks the last stretch **straight in on odometry** to 0.2 m from the face, because the camera is blocked near the table
(box in the hands, table filling the view). The table is tracked in a **fixed frame**.

```
SCAN ─► ROTATE/APPROACH leg "meet"  ──(within 0.4 m of the bisector)──►  leg "line": turn to the table, walk in
        to the meet point on the bisector,                                along the bisector (aim point 1 m up the line)
        3.0 m from the face, bearing only                                              │
                                                              real distance to the face ≤ 2.0 m
                                                                                       ▼
   done ◄─ FINAL_ALIGN ◄─ ROTATE/APPROACH, straight in ◄─ freeze ◄─ LATCH: stop, turn to face the table if needed,
          (0.2 m from the face,    on odometry, vision ignored       wait 1.5 s, poses taken after that agree
           square to it)                                              (camera blind from ~1.2 m)
```

## The approach (what the robot does, in order)

1. **SCAN.** No table yet: stand 3 s, then turn left at 0.7 rad/s until 3 poses arrive (gives up after 400°).
2. **Meet leg.** If the robot is more than `line_tol` (0.4 m) off the bisector it walks to the **meet point**: on the
   bisector, `meet_dist` (3.0 m) from the table face, aiming at it only (no squaring to the table on the way). The
   robot is always on its own side of the table; beside the table's long-axis plane (±0.5 m) it keeps the side it chose
   first, so a table seen end-on does not flip it between the two long faces. The meet leg is a straight line: there is
   no path planning around the table.
3. **Leg "line".** As soon as the robot is within `line_tol` of the bisector (`[LEG] on the bisector (...)` in the log)
   it turns to the table and walks in. The aim point is `lookahead` (1.0 m) up the bisector from the robot (pure
   pursuit), so the path melts into the line and arrives on it, instead of meeting it only at the end point. A robot that
   starts on the bisector skips the meet leg.
4. **Latch** at a real distance to the face ≤ `latch_far` (2.0 m), see below.
5. **Straight in** on the frozen table and the pelvis pose from TF to `place_standoff` (0.20 m, pelvis to face), then
   FINAL_ALIGN: yaw first, then x/y creep (±5 cm, ±4°), 20 settled cycles, `succeeded`.

It never walks backwards: if the meet point is behind the robot (it starts close to the table and off the line), it
turns around and walks there, and the table leaves the image until it turns back.

## Which pose is "the last pose"

1. The trigger is: the robot has reached the bisector **and** the **real distance to the table's near face** (to the face
   itself, not to its plane) is ≤ `latch_far` (2.0 m). Then it **stops**. (The first version measured the distance to
   the infinite plane of the face. A tilted estimate puts that plane next to a table that is metres away: on Nov 22 it
   latched 5 m from the table.)
2. If it is not facing the table (more than 2·`yaw_tol` = 8° off), it **turns on the spot** first (`latch: turn to the
   table`).
3. It waits `latch_settle_s` (1.5 s). Only poses whose **image stamp is after that** count (arrival time would let a
   late pose from the walk in).
4. It takes the poses of the last `window_s` (1 s). Latch when there are ≥ `latch_samples` (5), they cover the full
   window, and the **first and second half agree** within 5 cm / 3° (a new estimation or a drift pulls the halves
   apart; plain jitter does not, whatever its size). The latched pose is the **median** of that window.
5. From then on vision is ignored. `~/table_pose` (odom frame) shows what was latched. The live pose is still read and
   **logged next to the latched one** (`live-latched 0.05 m 1.2 deg` in `[HB]`), never used.

| stationary range to the face | what happens |
|---|---|
| 1.55 to 2.15 m (window ± `latch_tol`) | latch |
| > 2.15 m (the live estimate stopped us early) | walk on, latch again (`latch_max_tries` 3, then latch with a warning) |
| < 1.55 m, or more than `line_tol` off the bisector | latch with a warning (a close table may be partly out of view; it never backs up) |
| no stable poses within `latch_timeout_s` (20 s from the stop, the turn included; tracker gaps of 2 to 9 s were seen) | **fail**, with the numbers: how many poses, how long, how far apart the halves are, or the angle still to turn |
| the end point reached without a latch | **fail**: it never declares success on a table it did not latch |

**Why 1.7 to 2.0 m** (geometry from the camera intrinsics, fx 412.31, cx 360, so 82.3° horizontal; not measured with
FoundationPose): the 2.47 m long table is wholly in the image from 1.42 m if centred, from 1.76 m with 0.3 m sideways
error. At 2.0 m it spans 510 of 720 px. Closer than that the ends leave the image and the along-table position gets
unreliable. A heading error of more than ~8° pushes one end out of the image, hence the turn before the latch.

**Why a fixed frame.** The table does not move, so every pose is mapped to `odom_frame` with the pelvis pose at its
image time. A late, dropped or NaN detection then cannot bend the walk, filtering over a second is free (no pelvis-frame
smearing while walking), and "walk on odometry" is the same code as before the latch with the estimate frozen. The meet
leg relies on it: the robot can turn away from the table and still know where it is.
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
Log landmarks: `[SCAN] table found`, `[LEG] on the bisector`, `[LATCH] try 1 … standing still`, `[LATCH] table at odom …`,
`[PHASE]`, `[DONE]`.

**Start the robot with the table in view and at least ~2.5 m from its face** (the meet point is 3.0 m out). A start inside
the window and on the bisector latches where it stands, with a warning. A start inside the window and off the bisector
walks out to the meet point first, which turns the robot away from the table.

**Stopping.** `cancel` (or Ctrl-C / SIGTERM: the node sends zero velocity 5 times and releases the head before it
exits; a second Ctrl-C meanwhile is ignored). The free-box controller keeps the **last `cmd_vel` forever** (no timeout;
the head override times out after 1 s), so a stop that never leaves the node keeps the robot turning. If that happens
(log line `[SHUTDOWN] stop command NOT sent`, or a robot that turns on after the node is gone):
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

## Parameters that matter

| parameter | default | note |
|---|---|---|
| `odom_frame` | `odom` | the fixed frame in TF: **`world` in Isaac Sim** (pass `-p odom_frame:=world`) |
| `place_standoff` | 0.20 | pelvis → table near face at the end. The carry pose of `H2_pickup_sequencer` (grip x 0.20 m, box 0.30 m deep) puts the box front 0.35 m ahead of the pelvis, so **the box ends 0.15 m over the table edge** and the box front passes the edge 0.15 m before the end: the carry height must clear the table top (1.05 m) by then. Box front to face 0.20 m would be 0.55 |
| `lateral_offset` | 0.0 | stand this far to the robot's **left** of the table centre (the bisector moves with it) |
| `meet_dist` | 3.0 | m, meet point on the bisector, from the face. Keep it above `latch_far` |
| `line_tol` | 0.4 | m, this close to the bisector counts as on it (also the warning limit at the latch) |
| `lookahead` | 1.0 | m, aim point up the bisector. Shorter: converges faster, turns harder |
| `aim_blend` | 0.6 | m, closer than this to the aim point the bearing fades out (a 15 cm slip 0.55 m from the table swung the heading 14°, now 3°) |
| `table_depth` / `table_length` | 0.762 / 2.474 | `targets.yaml` size; the robot faces a long side (mesh Y) |
| `latch_far` / `latch_near` | 2.0 / 1.7 | window, pelvis → table face |
| `latch_settle_s`, `window_s`, `latch_samples` | 1.5, 1.0, 5 | |
| `latch_pos_tol`, `latch_yaw_tol` | 0.05 m, 3° | how well the two halves must agree |
| `max_range` | 9.0 | m, poses further than this are rejected. Needs `max_depth` 10.0 in the FoundationPose launch (see `target_switching.md`) |
| `scan_wait_s`, `scan_w`, `scan_dir` | 3.0, 0.70, +1 | stand still, then turn left at 0.7 rad/s; fails after `scan_max_deg` 400 |
| `v_min v_max w_min w_max kp_* *_tol` | as `h2_local_navigator` | tuned there for the **no-box** policy; check them with the box |
| `head_*` | as `h2_local_navigator` | aims at the table centre's **elevation** (independent of where the robot faces), held after the latch. The free-box policy was trained with head pitch 0: if it walks badly with the head down, cap it with `-p head_pitch_max:=0.3` |

## Reading the `[HB]` line

```
[HB] approach leg=meet | table@odom=(+6.25,+3.70) axis=+20 | pelvis@odom=(+1.2,+0.3,-30) | face=2.85 off=+0.40 goal=(+2.1,+0.4) bearing=+10 dyaw=+5 | head=+9.0 | pose age=0.3s n=11 | cmd=(+0.5,+0.0,+0.0) | rx=… acc=… rej=… tf_fail=…
```
`leg=meet|line` (`LATCHED` after the latch), `face` = real distance to the near face, `off` = how far the robot stands to
the left of the bisector, `goal` = the aim point in the pelvis frame, `dyaw` = heading error to face the table. During
LATCH the end shows `latch: 4/5 poses, 0.8/1.0s since the first, halves differ 12/50 mm 0.4/3 deg`. After the latch it
shows `live-latched 0.05 m 1.2 deg (pose age 0.3s)`: a large number means the camera disagrees with the frozen table.

## Interfaces

`~/start`, `~/cancel` (Trigger). `~/status` (latched: `idle`/`running`/`succeeded`/`failed`). `~/result` (latched JSON):
success carries `face_dist_m` (pelvis → table face, what the place step needs), `dyaw_deg`, `table_in_pelvis_m`,
`goal_error_m`, `latched_range_m`, `latched_off_m`, `latch_tries`, `latch_samples`, `latch_drift_mm/deg`; failure carries
`reason` and `phase`. `~/table_pose` (latched PoseStamped in odom, x along the table's long axis), `~/target_point`
(aim point in the pelvis frame). A crash in the loop stops the robot and fails the run.

## Tests

`pytest tests` (97 in all; 51 in `tests/navigation`, no ROS needed). They drive the node in a small kinematic world
(`navsim.py`): velocity lag 0.25 s, vision 0.3 s late and noisy, camera blind below 1.2 m, tracker gaps, 180° flips,
a re-estimation jump inside the latch window, odometry loss. They check arrival within 8 cm / 6° of the goal (10 cm / 7°
for the noisy and sideways cases), and the new approach: from beside the table's end the robot meets the bisector more
than 2 m out, faces the table, latches on the line (within 0.2 m) and walks straight in; a table seen end-on is passed on
the outside and approached at a long face; a start on the bisector does not detour; a start in the window latches where it
stands, turned away it turns on the spot first, off the line it joins the line first; the range is the distance to the
face, not to its plane (the Nov 22 trigger bug); the end point without a latch fails; the live pose is logged but not
used; no ±0.25 turn-rate kicks while walking in; the heading holds after a sideways slip next to the table.
Deliberate breakages of the node were all caught by a test: 20 on the first version (image-time TF, the settle rule, the
retry, the axis average, the head formula, the stop, the signal handling, …) and 17 on the bisector approach (plane
distance as range, no on-the-line condition in the trigger, no turn before the latch, no meet leg, meet point inside the
window, no side memory, no lookahead, no bearing fade, no guard against success without a latch, a latch timeout that
ignores the turn, …).

## Runs so far

- **First real run (Nov 22, Isaac Sim).** The head went to its 28.6° limit (`atan2(y, z)` of a table 77° off to the side)
  and Ctrl-C never sent the stop. Both fixed.
- **Second run (Nov 22).** The latch fired about 5 m from the table (plane-distance trigger, fixed above), and the
  robot ended at the table's **short face**: with `max_depth` 4.0 only the nearest 0.5 to 0.8 m of the table was inside
  the depth cut, the long axis came out about 90° off (latched +19.6°, robot heading 108° at the end). "Succeeded"
  there only meant "reached the pose it latched". Best explanation, not proven: **re-test with `max_depth` 10.0** and
  compare the latched `axis` with the table's Rotate Z in Isaac Sim.

**Not verified** (no ROS, GPU or robot here): real tracker noise and lag, whether 10 m of `max_depth` fixes the axis,
how the free-box policy follows `cmd_vel` with the head tilted, the carry height at the 0.2 m end, and `main()` on a real
rclpy (its signal handling follows the rclpy source and is tested on a stub). The table's pose along its long axis can
be off by ~0.3 m between FoundationPose estimations; this node cannot fix that, it only keeps the robot from chasing it.
Known limit: no path planning around the table; the meet leg is a straight line, safe for a robot in front of a long
face or beside an end, not for one that starts right next to the table's side.
