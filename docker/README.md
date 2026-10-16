# Perception container (FoundationPose)

The FoundationPose perception pipeline runs inside a container on the
perception machine, not on the host.

## Image

`isaac_ros_foundationpose_custom:v1` — a customised NVIDIA Isaac ROS image
with FoundationPose. It was built on the sim/control machine from NVIDIA's
Isaac ROS FoundationPose base image, then transferred to the perception
machine. The customisations were made with `docker commit` rather than a
Dockerfile. `docker/Dockerfile` now reproduces the same environment from
the same NVIDIA base image.

## Run

Container flags are fixed at creation time. `docker start` on an existing
container reuses the flags it was created with — you cannot add mounts or
environment variables to it. If any of the settings below change, create a
new container rather than restarting the old one:

    docker ps -a --filter name=isaac_ros_fp   # does one exist?
    docker rm -f isaac_ros_fp                 # image and mounted workspace are kept

First copy the DDS profile into the mounted workspace (see the network
prerequisites in the main README):

    cp ~/.ros/fastdds_lan_only.xml ~/workspaces/isaac_ros-dev/

Then, on the perception machine (runs on GPU 0 only, leaving the other GPU
free for other processes):

    docker run -it \
      --name isaac_ros_fp \
      --gpus '"device=0"' \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e CUDA_VISIBLE_DEVICES=0 \
      --network host \
      --ipc host \
      --privileged \
      -v $HOME/workspaces/isaac_ros-dev:/workspaces/isaac_ros-dev \
      -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \
      -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_lan_only.xml \
      isaac_ros_foundationpose_custom:v1 \
      bash

Key points:
- `--gpus '"device=0"'` gives the container GPU 0 only. `--privileged` can
  still expose every GPU's device files, so `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0` pin CUDA to the same GPU 0 that host `nvidia-smi`
  shows. To use all GPUs instead, replace those three lines with `--gpus all`.
  Check with `nvidia-smi` on the host while the pipeline runs.
- The host workspace is mounted at `/workspaces/isaac_ros-dev` inside the
  container. All paths in the launch file are relative to this.
- `ISAAC_ROS_WS` is passed in so the launch file picks it up.
- `--network host` and `--ipc host` are required so ROS 2 reaches the other
  machine.
- `FASTRTPS_DEFAULT_PROFILES_FILE` restricts DDS to the wired LAN; without it
  the container discovers topics but receives no data.

To reattach to a running container:

    docker exec -it isaac_ros_fp bash

## Launching the pipeline

Inside the container:

    ros2 launch /workspaces/isaac_ros-dev/foundationpose_klt_tracking.launch.py

Verify data is arriving before launching:

    ros2 topic hz /rgb

If this hangs, check the network prerequisites in the main README.

(The launch file also lives in `launch/` in this repo — the copy in the
mounted workspace is what the container runs. Keep them in sync.)

## Required assets in the mounted workspace

These are not committed to this repo (large, downloadable, or NVIDIA-provided):

- FoundationPose models (`refine_model.onnx`, `score_model.onnx`, and the
  corresponding TensorRT `.plan` engines) under
  `isaac_ros_assets/models/foundationpose/` — download from NVIDIA NGC.
- The KLT box mesh — committed here in `assets/KLT_box/`; copy the whole
  folder into the mounted workspace so it appears at
  `/workspaces/isaac_ros-dev/KLT_box/`.
- `fastdds_lan_only.xml` — the DDS profile (see Run, above).

## Known warnings

`[MeshStorage] No texture path found for mesh` — the KLT box `.mtl` has no
`map_Kd` entry and the mesh has no UV coordinates, so FoundationPose falls
back to a flat colour. Pose estimation works; adding a proper UV-mapped
texture could improve matching accuracy.

## Building from the Dockerfile

`docker/Dockerfile` reproduces this environment from NVIDIA's base image:

    docker build -t isaac_ros_foundationpose_custom:v2 -f docker/Dockerfile .

Verified to build and produce the FoundationPose node libraries. Note that
apt will install current package versions, which may be newer than those in
the existing v1 image.
