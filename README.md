# h2-ros2-control

ROS 2 stack for controlling a Unitree H2 humanoid in Isaac Sim: RL locomotion,
arm IK, and FoundationPose-based 6-DoF box detection for grasping.

Trained policies come from the companion training repo
(`UnitreeH2-isaaclab-training`); this repo consumes the exported policy.

## Architecture

Perception chain (KLT box pose):

    Isaac Sim  ->  /rgb, /depth, /camera_info, /semantic_segmentation
                       |
    semantic_mask_node ->  /segmentation_mask   (binary mask of the box)
                       |
    FoundationPose (container) ->  /tracking/output   (Detection3DArray, ~20Hz)
                       |
    pose_overlay_visualizer_tracking_grasp
                       |->  /tracking_visualization  (RGB + wireframe overlay)
                       |->  /target_pose_left        (grasp pose, pelvis frame)
                       |->  /target_pose_right

Control chain (box pickup, robot balancing):

    /target_pose_left|right  ->  H2_pickup_sequencer  ->  /arm_targets (14 arm joints)
                                                      ->  /load_expected (lift flag)
    /joint_states + /imu + /arm_targets + /load_expected
                             ->  H2_balance_controller  ->  /joint_command (all 31)  ->  Isaac Sim

Other control nodes:

    h2_fullbody_controller   ->  RL locomotion policy (joint_states + IMU, /cmd_vel, /head_pitch_cmd)
    H2_ik_node               ->  arm IK only, torso pinned (/arm_joints)

## Machines

| Machine | ROS_DOMAIN_ID | Runs |
|---|---|---|
| Sim / control | 99 | Isaac Sim, control nodes, semantic mask, visualizer |
| Perception | 0 | FoundationPose container |

    export ROS_DOMAIN_ID=99

The domains are isolated by design; topics are forwarded with the standard
`domain_bridge` package.

    sudo apt-get install -y ros-jazzy-domain-bridge
    ros2 run domain_bridge domain_bridge config/bridge_config.yaml

Run one instance (on either machine — it bridges both directions). `config/bridge_config.yaml` forwards camera topics
(/rgb, /depth, /camera_info, /segmentation_mask) from domain 99 to domain 0, and
results (/tracking/output, /tracking_visualization) back from domain 0 to 99.


## Nodes and environments

| Node | Env | Purpose |
|---|---|---|
| `nodes/control/h2_fullbody_controller.py` | `h2_rl` | RL locomotion policy |
| `nodes/control/H2_balance_controller.py` | `ik_env` | RL balance policy (legs/waist/head) + arm targets -> `/joint_command` |
| `nodes/control/H2_pickup_sequencer.py` | `ik_env` | live-IK arm phases: wide, standoff, pinch, tilt, carry |
| `nodes/control/H2_ik_node.py` | `ik_env` | arm IK to grasp poses (torso pinned, testing) |
| `nodes/perception/semantic_mask_node.py` | `yolo-env` | Isaac Sim semantics -> binary mask |
| `nodes/perception/pose_overlay_visualizer_tracking_grasp.py` | `yolo-env` | overlay + grasp poses |
| `launch/foundationpose_klt_tracking.launch.py` | container | FoundationPose pipeline |

Create the environments:

    conda env create -f envs/h2_rl.yml
    conda env create -f envs/ik_env.yml

Every terminal needs the env activated, ROS 2 sourced, and the domain set:

    conda activate <env>
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=<domain>      # sim/control machine: 99, perception machine: 0

## Running

Nodes are currently run individually for testing.

    # Semantic mask; target label set by TARGET_LABEL in the file (ik_env)
    python3 nodes/perception/semantic_mask_node.py
    
    # Object pose visualizer (yolo-env)
    python3 nodes/perception/pose_overlay_visualizer_tracking_grasp.py

    # RL locomotion (h2_rl env)
    python3 nodes/control/h2_fullbody_controller.py [--policy <file>]

    # Box pickup: balance policy + arm sequencer (ik_env). Start the controller
    # BEFORE pressing Play; start the sequencer once the robot stands steady.
    python3 nodes/control/H2_balance_controller.py
    python3 nodes/control/H2_pickup_sequencer.py      # flags + troubleshooting: nodes/control/DEPLOY_h2_pickup.md

    # Arm IK only, torso pinned (ik_env)
    python3 nodes/control/H2_ik_node.py

FoundationPose pipeline: see `docker/README.md` (runs inside the container).

## Assets and models

- `assets/h2_description/` - H2 URDF + STL meshes (used by the IK nodes)
- `assets/KLT_box/` - KLT box assets. `KLT_box_metres.obj` is used by
  FoundationPose for pose matching; copy it into the container's mounted
  workspace so it lands at `/workspaces/isaac_ros-dev/KLT_box/`. `klt_box.usd`
  is a self-contained USD for adding the box to new Isaac Sim scenes.
- `assets/scenes/` - Isaac Sim scene USDs (flattened, self-contained)
- `models/h2_balance_box_pickup/` - balance policy (`policy.pt`) + `policy_io.json`
  (joint order, gains, obs layout) loaded by `H2_balance_controller.py` and
  `H2_pickup_sequencer.py`
- `models/h2_walking/` - locomotion policy for `h2_fullbody_controller.py`
- FoundationPose weights are not committed; download from NVIDIA NGC into the
  container workspace (see `docker/README.md`)

## Notes

- USD scenes are flattened snapshots; re-flatten and re-commit if you modify one.
- Binary assets (STL, USD, PNG) are tracked with Git LFS.

## Visualization

Image streams are viewed with RViz2 on the sim/control machine:

    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=99
    rviz2

Add Image displays for:

| Topic | Shows |
|---|---|
| `/rgb` | camera feed from Isaac Sim |
| `/depth` | depth stream |
| `/segmentation_mask` | binary mask of the target object |
| `/tracking_visualization` | RGB + box wireframe + grasp point markers |

## Network prerequisites (both machines)

Two settings are required for ROS 2 data to flow between the machines. Without
them, discovery succeeds but no data is delivered: topics appear in
`ros2 topic list` while `ros2 topic hz` hangs indefinitely, with no error.

### 1. UDP receive buffers

Raise the kernel limits (the Ubuntu default is too small for image topics):

    echo "net.core.rmem_max=8388608" | sudo tee -a /etc/sysctl.conf
    echo "net.core.rmem_default=8388608" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p

`sysctl -w` alone does not survive a reboot.

### 2. Restrict DDS to the wired LAN

Both machines have several network interfaces (wired LAN, WiFi, docker0).
Fast DDS advertises all of them and may attempt data transfer over an
unreachable one. Restrict it to the wired interface with a profile per machine:

    mkdir -p ~/.ros
    cat > ~/.ros/fastdds_lan_only.xml << 'XML'
    <?xml version="1.0" encoding="UTF-8" ?>
    <dds xmlns="http://www.eprosima.com">
      <profiles>
        <transport_descriptors>
          <transport_descriptor>
            <transport_id>lan_udp</transport_id>
            <type>UDPv4</type>
            <interfaceWhiteList>
              <address>THIS_MACHINE_WIRED_IP</address>
              <address>127.0.0.1</address>
            </interfaceWhiteList>
          </transport_descriptor>
        </transport_descriptors>
        <participant profile_name="lan_only" is_default_profile="true">
          <rtps>
            <userTransports>
              <transport_id>lan_udp</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
          </rtps>
        </participant>
      </profiles>
    </dds>
    XML

Replace `THIS_MACHINE_WIRED_IP` with the machine's own wired address, then:

    echo 'export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_lan_only.xml' >> ~/.bashrc

The container needs the same profile — see `docker/README.md`.

Note: the profile hardcodes IPs, so static addresses (or DHCP reservations)
are recommended.

Loopback (`127.0.0.1`) is included so single-machine work still functions when
the wired interface is down. Without it, Fast DDS fails to start any transport:

    [TRANSPORT_UDPV4 Error] All whitelist interfaces were filtered out
    [RTPS_PARTICIPANT Error] No unicast locators to create unique flows

If you hit that error with loopback already present, the wired IP in the
profile no longer matches any active interface — check `ip addr` and update it.
