#!/usr/bin/env python3
"""
Playback IK reference trajectories in Isaac Sim.

Loads the library and streams joint commands to verify the arms:
  1. Wide stance hold (X=0.15m, free orientation)
  2. Reach to grasp point
  3. Approach (pinch inward)
  4. Lift to carry pose
  5. Apply wrist tilt

Usage:
  source /opt/ros/jazzy/setup.bash
  python3 playback_ik_trajectories.py --index 0_0_0_0                 # both arms (default)
  python3 playback_ik_trajectories.py --index 0_0_0_0 --arm left      # left arm only
  python3 playback_ik_trajectories.py --index 0_0_0_0 --arm right     # right arm only
  python3 playback_ik_trajectories.py --index 0_0_0_0 --speed 0.5     # half speed
  python3 playback_ik_trajectories.py --index 2_2_2_1 --loop          # loop continuously
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import torch
import numpy as np
import argparse
import os

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_NAMES = ARM_JOINT_NAMES[7:]

# Default standing/relaxed postures extracted from Isaac Sim
RELAXED_LEFT_ARM = [0.2802, 0.1350, -0.0088, 0.5610, 0.0004, 0.0107, -0.0006]
RELAXED_RIGHT_ARM = [0.2802, -0.1355, 0.0089, 0.5611, -0.0004, 0.0107, 0.0006]

PUBLISH_HZ = 50


class PlaybackNode(Node):
    def __init__(self, warmup_sec=2.0):
        super().__init__("playback_ik_trajectories")
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)
        
        # Will be set by main()
        self.trajectory = None
        self.arm_name = None
        
        # Warm-up / Holding settings
        self.warmup_steps = int(warmup_sec * PUBLISH_HZ)
        self.current_tick = 0
        self.step = 0
        self.done = False
    
    def timer_callback(self):
        if self.done or self.trajectory is None:
            return
        
        # Phase A: Hold initial pose (trajectory[0]) so controller/sim settles
        if self.current_tick < self.warmup_steps:
            step_data = self.trajectory[0, :]
            if self.current_tick == 0:
                self.get_logger().info("Holding Phase 1 start pose for controller initialization...")
            self.current_tick += 1
        # Phase B: Playback full trajectory
        else:
            if self.step >= len(self.trajectory):
                self.get_logger().info("Playback complete")
                self.done = True
                return
            
            step_data = self.trajectory[self.step, :]
            self.step += 1
        
        # Build joint message using default standing pose for unselected arms
        if self.arm_name == "left":
            left_q = step_data.tolist()
            right_q = RELAXED_RIGHT_ARM
        elif self.arm_name == "right":
            left_q = RELAXED_LEFT_ARM
            right_q = step_data.tolist()
        else:  # both
            left_q = step_data[:7].tolist()
            right_q = step_data[7:14].tolist()
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES
        msg.position = left_q + right_q
        
        self.pub.publish(msg)


def parse_index(index_str):
    """Parse index string like '0_0_0_0' into (i_x, i_y, i_z, i_yaw)."""
    parts = index_str.split('_')
    if len(parts) != 4:
        raise ValueError(f"Index must be 'i_x_i_y_i_z_i_yaw', got '{index_str}'")
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="Playback IK trajectory in Isaac Sim")
    parser.add_argument(
        "--index", type=str, default="0_0_0_0",
        help="Grid index: i_x_i_y_i_z_i_yaw (e.g., '0_0_0_0' for first box pose)"
    )
    parser.add_argument(
        "--arm", type=str, choices=["left", "right", "both"], default="both",
        help="Which arm to playback (default: both)"
    )
    parser.add_argument(
        "--library", type=str, default="ik_trajectory_library_5phase.pt",
        help="Path to the IK trajectory library file"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop playback continuously"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--warmup", type=float, default=2.0,
        help="Initial hold time in seconds at starting pose (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Check library file exists
    if not os.path.exists(args.library):
        print(f"ERROR: Library file not found: {args.library}")
        return 1
    
    # Load library
    print(f"Loading library: {args.library}")
    lib = torch.load(args.library, weights_only=False)
    
    i_x, i_y, i_z, i_yaw = parse_index(args.index)
    
    # Validate indices
    if i_x >= lib["left"].shape[0]:
        print(f"ERROR: i_x={i_x} out of range [0, {lib['left'].shape[0]-1}]")
        return 1
    if i_y >= lib["left"].shape[1]:
        print(f"ERROR: i_y={i_y} out of range [0, {lib['left'].shape[1]-1}]")
        return 1
    if i_z >= lib["left"].shape[2]:
        print(f"ERROR: i_z={i_z} out of range [0, {lib['left'].shape[2]-1}]")
        return 1
    if i_yaw >= lib["left"].shape[3]:
        print(f"ERROR: i_yaw={i_yaw} out of range [0, {lib['left'].shape[3]-1}]")
        return 1
    
    # Extract trajectory(ies)
    if args.arm == "left":
        trajectory = lib["left"][i_x, i_y, i_z, i_yaw, :, :].numpy()  # shape (50, 7)
        arm_name = "left"
    elif args.arm == "right":
        trajectory = lib["right"][i_x, i_y, i_z, i_yaw, :, :].numpy()  # shape (50, 7)
        arm_name = "right"
    else:  # both
        left_traj = lib["left"][i_x, i_y, i_z, i_yaw, :, :].numpy()   # shape (50, 7)
        right_traj = lib["right"][i_x, i_y, i_z, i_yaw, :, :].numpy() # shape (50, 7)
        trajectory = np.concatenate([left_traj, right_traj], axis=1)   # shape (50, 14)
        arm_name = "both"
    
    # Check for NaN (failed trajectory)
    if np.isnan(trajectory).any():
        print(f"ERROR: Trajectory at index {args.index} is NaN (failed to solve)")
        return 1
    
    # Get box pose from library metadata
    x = lib["x_range"][i_x]
    y = lib["y_range"][i_y]
    z = lib["z_range"][i_z]
    yaw = lib["yaw_range"][i_yaw]
    
    print(f"\n{'='*70}")
    if arm_name == "both":
        print(f"Playback trajectory for BOTH arms")
        print(f"Trajectory: {trajectory.shape[0]} timesteps × {trajectory.shape[1]} joints (7 per arm)")
    else:
        print(f"Playback trajectory for {arm_name} arm")
        print(f"Trajectory: {trajectory.shape[0]} timesteps × {trajectory.shape[1]} joints")
    print(f"Box pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={np.degrees(yaw):.0f}°")
    print(f"Speed: {args.speed}x")
    print(f"Warmup hold time: {args.warmup}s")
    if args.loop:
        print(f"Looping: enabled")
    print(f"{'='*70}\n")
    
    # Initialize ROS
    rclpy.init()
    node = PlaybackNode(warmup_sec=args.warmup)
    node.trajectory = trajectory
    node.arm_name = arm_name
    
    # Adjust speed if needed
    if args.speed != 1.0:
        n_steps = int(len(trajectory) / args.speed)
        n_steps = max(n_steps, 10)
        t_old = np.linspace(0, 1, len(trajectory))
        t_new = np.linspace(0, 1, n_steps)
        
        trajectory_resampled = np.zeros((n_steps, trajectory.shape[1]))
        for j in range(trajectory.shape[1]):
            trajectory_resampled[:, j] = np.interp(t_new, t_old, trajectory[:, j])
            
        node.trajectory = trajectory_resampled
    
    try:
        print("Starting playback. Press Ctrl+C to stop.\n")
        
        loop_count = 0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            if node.done:
                loop_count += 1
                if args.loop:
                    print(f"\nCompleted loop {loop_count}. Restarting...")
                    node.step = 0
                    node.current_tick = 0
                    node.done = False
                else:
                    print(f"\nPlayback complete.")
                    break
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    
    finally:
        node.destroy_node()
        rclpy.shutdown()
        return 0


if __name__ == "__main__":
    exit(main())