#!/usr/bin/env python3
"""
Converts Isaac Sim semantic segmentation (32SC1) to a binary mask (mono8)
for use with FoundationPose.

Looks up the active target's semantic ID dynamically by label name (targets.yaml),
so it works even when Isaac Sim reassigns IDs across sessions. The target starts as
the `target` parameter and is switched at runtime by /active_target.

Subscribes: /semantic_segmentation         (sensor_msgs/Image, 32SC1)
            /semantic_labels               (std_msgs/String, JSON label map)
            /active_target                 (std_msgs/String, key of targets.yaml)
Publishes:  /segmentation_mask             (sensor_msgs/Image, mono8)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import json

from targets import load_targets, find_label_id


class SemanticMaskNode(Node):

    def __init__(self):
        super().__init__('semantic_mask_node')

        self.declare_parameter('targets_file', '')
        self.declare_parameter('target', 'klt_box')   # initial target, /active_target overrides
        self.targets = load_targets(self.get_parameter('targets_file').value or None)

        self.target    = None
        self.label     = ''     # semantic label of the active target
        self.label_map = {}     # last label map from Isaac Sim
        # The current semantic ID for our target — discovered at runtime
        self.target_id = None

        # Subscribe to the segmentation image
        self.sub_seg = self.create_subscription(
            Image,
            '/semantic_segmentation',
            self.seg_callback,
            10
        )

        # Subscribe to the label map that Isaac Sim publishes alongside the image
        # This topic publishes a JSON string mapping ID → label
        self.sub_labels = self.create_subscription(
            String,
            '/semantic_labels',
            self.labels_callback,
            10
        )

        self.sub_target = self.create_subscription(
            String,
            '/active_target',
            self.target_callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/segmentation_mask',
            10
        )

        if not self.set_target(self.get_parameter('target').value):
            raise SystemExit(f'unknown initial target; known: {list(self.targets)}')

    def target_callback(self, msg: String):
        if msg.data != self.target:      # the orchestrator republishes at ~1 Hz
            self.set_target(msg.data)

    def set_target(self, name):
        """Switch the active target; the mask follows as soon as its label is in the label map."""
        if name not in self.targets:
            self.get_logger().error(f'Unknown target "{name}" (known: {list(self.targets)})',
                                    throttle_duration_sec=5.0)
            return False
        self.target, self.label = name, self.targets[name]['label']
        self.target_id = None   # never publish the previous target's mask under the new one
        self.get_logger().info(f'Active target: {name} (label="{self.label}")')
        self.resolve_id()
        return True

    def resolve_id(self):
        new_id = find_label_id(self.label_map, self.label)
        if new_id is None:
            if self.label_map:
                self.get_logger().warn(
                    f'Label "{self.label}" not found in label map. '
                    f'Available labels: {self.label_map}',
                    throttle_duration_sec=5.0
                )
            return
        if new_id != self.target_id:
            self.get_logger().info(
                f'Found "{self.label}" → semantic ID={new_id} '
                f'(was {self.target_id})'
            )
            self.target_id = new_id

    def labels_callback(self, msg: String):
        """
        Isaac Sim publishes a JSON label map alongside the segmentation image.
        Example: {"0": {"class": "BACKGROUND"}, "2": {"class": "klt_box"}, ...}
        We parse this to find which ID corresponds to our target label.
        """
        try:
            self.label_map = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse label map JSON')
            return
        self.resolve_id()

    def seg_callback(self, msg: Image):
        """Convert semantic segmentation image to binary mask."""

        if self.target_id is None:
            self.get_logger().warn(
                f'Waiting for label map to find ID for "{self.label}". '
                f'Checking if /semantic_labels is published...',
                throttle_duration_sec=5.0
            )
            # Fallback: print unique IDs to help debugging
            raw = np.frombuffer(msg.data, dtype=np.int32)
            unique_ids = np.unique(raw)
            self.get_logger().info(
                f'Unique semantic IDs in current frame: {unique_ids}',
                throttle_duration_sec=5.0
            )
            return

        # Convert raw bytes → numpy int32 array
        raw = np.frombuffer(msg.data, dtype=np.int32)
        seg = raw.reshape((msg.height, msg.width))

        # Binary mask: target object = 255, everything else = 0
        mask = np.where(seg == self.target_id, 255, 0).astype(np.uint8)

        num_pixels = int(np.sum(mask > 0))
        if num_pixels == 0:
            self.get_logger().warn(
                f'No pixels found for "{self.label}" (ID={self.target_id}). '
                f'Is it in the camera frame?',
                throttle_duration_sec=5.0
            )

        # Build output message
        out = Image()
        out.header       = msg.header
        out.height       = msg.height
        out.width        = msg.width
        out.encoding     = 'mono8'
        out.is_bigendian = 0
        out.step         = msg.width
        out.data         = mask.tobytes()

        self.pub.publish(out)

        self.get_logger().info(
            f'Mask published: {num_pixels}/{msg.width * msg.height} pixels '
            f'are "{self.label}" (ID={self.target_id})',
            throttle_duration_sec=2.0
        )


def main():
    rclpy.init()
    node = SemanticMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
