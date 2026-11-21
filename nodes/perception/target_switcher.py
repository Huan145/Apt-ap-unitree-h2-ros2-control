#!/usr/bin/env python3
"""
FoundationPose target switcher - run on apt2 (ROS_DOMAIN_ID=0, Isaac ROS container).

ROS parameters don't cross the domain bridge, so this node follows /active_target (bridged 99 -> 0)
and applies the target's `foundationpose:` parameters from targets.yaml (mesh_file_path,
symmetry_axes, ...) to the running FoundationPose node. That is NVIDIA's "switch mesh during
runtime" mechanism (`ros2 param set /foundationpose mesh_file_path ...`). Isaac ROS 5.x passes the
mesh on to the tracking node; on 4.4 NVIDIA documents the live switch for the estimation node only
(see docs/target_switching.md before relying on it).

Needs targets.py + targets.yaml next to it.
  python3 target_switcher.py [--ros-args -p fp_node:=/foundationpose_node -p targets_file:=<yaml>]
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String

from targets import load_targets


class TargetSwitcher(Node):

    def __init__(self):
        super().__init__('target_switcher')
        self.declare_parameter('targets_file', '')
        self.declare_parameter('fp_node', '/foundationpose_node')
        self.targets = load_targets(self.get_parameter('targets_file').value or None)
        fp_node = self.get_parameter('fp_node').value

        self.cli     = self.create_client(SetParameters, f'{fp_node}/set_parameters')
        self.applied = None     # target whose parameters FoundationPose has now (None: the launch values)
        self.busy    = False
        self.create_subscription(String, '/active_target', self.target_callback, 10)
        self.get_logger().info(f'Following /active_target, setting parameters on {fp_node}')

    def target_callback(self, msg: String):
        name = msg.data
        if name == self.applied or self.busy:       # the orchestrator republishes at ~1 Hz
            return
        if name not in self.targets:
            self.get_logger().error(f'Unknown target "{name}"', throttle_duration_sec=5.0)
            return
        if not self.cli.service_is_ready():
            self.get_logger().warn('FoundationPose set_parameters service not available',
                                   throttle_duration_sec=5.0)
            return

        params = self.targets[name].get('foundationpose', {})
        req = SetParameters.Request(
            parameters=[Parameter(k, value=v).to_parameter_msg() for k, v in params.items()])
        self.busy = True
        self.cli.call_async(req).add_done_callback(lambda fut: self.done(name, params, fut))

    def done(self, name, params, fut):
        self.busy = False
        try:
            results = fut.result().results
        except Exception as e:
            self.get_logger().error(f'"{name}": set_parameters call failed: {e}', throttle_duration_sec=5.0)
            return
        rejected = {k: r.reason for k, r in zip(params, results) if not r.successful}
        if rejected:     # self.applied stays, so the next /active_target message retries
            self.get_logger().error(f'"{name}": FoundationPose rejected {rejected}', throttle_duration_sec=5.0)
            return
        self.applied = name
        self.get_logger().info(f'Target "{name}" applied: {params}')


def main():
    rclpy.init()
    node = TargetSwitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
