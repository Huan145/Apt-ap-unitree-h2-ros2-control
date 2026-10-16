# H2 pickup: balance policy + IK arms

## Install / update
```bash
unzip -o h2_balance_ik_bundle.zip -d ~/h2_bundle && bash ~/h2_bundle/install_update.sh
# custom repo paths: bash ~/h2_bundle/install_update.sh <TRAIN_REPO> <ROS_REPO>
```
Copies everything below into place, backs up replaced files to `~/h2_update_backups/<time>/`, prints `git status` for both repos.

## Layout
| Folder | Goes to | Contents |
|---|---|---|
| `h2_balance/` | `UnitreeH2-isaaclab-training/.../tasks/manager_based/` (next to `h2_pickup_dynamic`) | training task (`Isaac-Balance-H2-v0`) |
| `scripts/` | `UnitreeH2-isaaclab-training/scripts/` | `export_policy_io.py` (joint order / gains / obs layout), `pad_checkpoint.py` (resume after new obs), `set_drive_gains.py` (Isaac Sim Script Editor) |
| `ik_library/` | `h2-ros2-control/nodes/control/` | IK library, playback, sequencer, balance controller |

## Train
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Balance-H2-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Balance-H2-Play-v0   # also exports policy.pt
./isaaclab.sh -p scripts/export_policy_io.py --task Isaac-Balance-H2-v0 \
    --out logs/rsl_rl/h2_balance/<run>/exported/policy_io.json
```

## Resume after adding observation terms
New obs terms go LAST in their group; pad the checkpoint so it loads (the added inputs start at zero weight = no behavior change):
```bash
python3 scripts/pad_checkpoint.py --src logs/rsl_rl/h2_balance/<run>/model_800.pt \
    --dst logs/rsl_rl/h2_balance/<run>/model_800_flag.pt --actor-add 1 --critic-add 2
# set iter_offset=800 in H2BalanceCurriculumCfg, then:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Balance-H2-v0 --headless \
    --resume --load_run <run> --checkpoint model_800_flag.pt --max_iterations 1000
```

## Deploy (Isaac Sim or real robot)
```
FoundationPose ─/target_pose_left,right─► H2_pickup_sequencer ─/arm_targets (14)─┐
                                                              └─/load_expected────┤
                                                                                 ▼
sim/robot ─/joint_states, /imu──────────────────────────────────► H2_balance_controller
sim/robot ◄──────────────────────────/joint_command (all 31)────────────────────┘
```
```bash
conda activate ik_env && source /opt/ros/jazzy/setup.bash
# 1. balance first - it holds the arms where they are until the sequencer starts
python3 H2_balance_controller.py --policy <run>/exported/policy.pt --io <run>/exported/policy_io.json
# 2. once standing steadily, start the pickup
python3 H2_pickup_sequencer.py                  # --settle 1.0 for a pause after pinch
```

## Must match training
- **One writer**: only the balance controller publishes `/joint_command`. Don't run `H2_ik_node.py` or `H2_lift_node.py` at the same time.
- **Sim side**: one articulation controller on `/joint_command`, joint drive gains = the kp/kd printed by `export_policy_io.py`, and an IMU on the pelvis published as `sensor_msgs/Imu` on `/imu` (orientation + angular velocity, pelvis frame).
- **Load flag**: the sequencer publishes `/load_expected` = True from the start of the lift (end of the tilt). Training: in 80% of episodes the load starts 0-0.5 s after the flag, in 20% anywhere from 1 s before to 2 s after; ramp 0.1-1.0 s; mass 0-100% of the sawtooth maximum (0-7 kg every 700 iterations); 5% of episodes drop the box mid-carry. The policy needs the flag - without it, the controller keeps it at 0.
- **Motion**: the sequencer uses `generate_ik_library.py`'s code (speeds, tilt-then-carry order, L/R sync). If you change speeds or phases there, regenerate the library and retrain.
