#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion
from dynus_interfaces.msg import State


class FixedStatePublisher(Node):
    def __init__(self):
        super().__init__("fixed_state_publisher")

        # Publish on the same topic/type used by the existing state pipeline.
        self.publisher_ = self.create_publisher(State, "state", 10)
        self.timer_ = self.create_timer(1.0 / 20.0, self._publish_fixed_state)

        self.get_logger().info("Publishing fixed State at 20 Hz on topic 'state'")

    def _publish_fixed_state(self):
        msg = State()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pos = Vector3(x=0.0, y=0.0, z=0.0)
        msg.vel = Vector3(x=0.0, y=0.0, z=0.0)
        msg.quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixedStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
