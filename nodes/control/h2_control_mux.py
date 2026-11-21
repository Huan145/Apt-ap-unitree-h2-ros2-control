#!/usr/bin/env python3
"""
h2_control_mux.py - exactly ONE controller commands the robot at any time.

Forwards /joint_command from the ACTIVE controller only and drops the others, so a handover
can never have two controllers fighting over the joints. Switch with std_msgs/String on
/control_mode. The requested mode is re-published LATCHED on /control_mode_active, which the
controllers listen to (so a controller that connects late still gets it). The OLD controller
keeps commanding until the NEW one sends its first command - the robot is never uncommanded.

  python3 H2_balance_controller.py --cmd-topic /joint_command_balance
  python3 h2_fullbody_freebox_controller.py --standby --ros-args -r joint_command:=/joint_command_freebox
  python3 h2_control_mux.py                                    # starts with 'balance' active
  ros2 topic pub --once /control_mode std_msgs/msg/String "{data: freebox}"

Options: --initial balance|freebox|none   --out /joint_command
         --source NAME=TOPIC (repeatable; default balance=/joint_command_balance freebox=/joint_command_freebox)
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

SIM_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_ALL)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class ControlMux(Node):
    def __init__(self, sources: dict, out: str, initial: str, mode_topic: str):
        super().__init__("h2_control_mux")
        self.active = initial          # whose commands are forwarded now
        self.requested = initial       # mode asked for; becomes active on that source's first message
        self.count = {k: 0 for k in sources}
        self.pub = self.create_publisher(JointState, out, SIM_QOS)
        self.state_pub = self.create_publisher(String, mode_topic + "_active", LATCHED)
        for name, topic in sources.items():
            self.create_subscription(JointState, topic, lambda m, n=name: self.on_cmd(n, m), SIM_QOS)
        self.create_subscription(String, mode_topic, self.on_mode, 10)
        self.state_pub.publish(String(data=self.active))
        self.get_logger().info(f"sources {sources} -> {out} | active: '{self.active}' | switch: {mode_topic}")

    def on_mode(self, msg: String):
        new = msg.data.strip().lower()
        if new != self.requested:
            self.requested = new
            self.state_pub.publish(String(data=new))          # latched: tells the controllers
            self.get_logger().info(f"[SWITCH] requested '{new}' - '{self.active}' keeps commanding until "
                                   f"'{new}' sends its first command")

    def on_cmd(self, name: str, msg: JointState):
        if name == self.requested and name != self.active:
            self.get_logger().info(f"[SWITCH] '{self.active}' -> '{name}' now (forwarded so far: {self.count})")
            self.active = name
        if name == self.active:
            self.count[name] += 1
            self.pub.publish(msg)


def main():
    p = argparse.ArgumentParser(description="joint_command mux: one active controller at a time")
    p.add_argument("--source", action="append", default=None, help="NAME=TOPIC, repeatable")
    p.add_argument("--out", default="/joint_command")
    p.add_argument("--initial", default="balance")
    p.add_argument("--mode-topic", default="/control_mode")
    a, _ = p.parse_known_args()
    sources = dict(s.split("=", 1) for s in (a.source or
                   ["balance=/joint_command_balance", "freebox=/joint_command_freebox"]))
    rclpy.init()
    node = ControlMux(sources, a.out, a.initial.lower(), a.mode_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()