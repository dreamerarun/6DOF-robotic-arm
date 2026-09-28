#!/usr/bin/env python3
"""
serial_bridge_node.py

Bridges MoveIt2 (ROS2 Humble) to the Arduino Uno + PCA9685 arm.

It implements a FollowJointTrajectory action server. Configure MoveIt's
moveit_controllers.yaml so the "arm_controller" action name matches this
node's action_name param, and MoveIt will send planned trajectories
straight here - no ros2_control hardware interface required.

For each trajectory point, this node:
  1. Converts each joint angle from radians -> degrees
  2. Clamps to 0-180 (matching the Arduino sketch's expected range)
  3. Sends "a0,a1,a2,a3,a4,a5\n" over serial
  4. Sleeps until the point's time_from_start, then moves to the next point
  5. Publishes /joint_states so RViz / MoveIt see the arm's reported pose

Joint order (must match your URDF and moveit_controllers.yaml):
  joint_1 .. joint_6  ->  Arduino channels 0..5
"""

import time
import serial
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import math

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class SerialBridgeNode(Node):
    def __init__(self):
        super().__init__("arm_serial_bridge")

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("action_name", "/arm_controller/follow_joint_trajectory")

        port = self.get_parameter("serial_port").value
        baud = self.get_parameter("baud_rate").value
        action_name = self.get_parameter("action_name").value

        self.get_logger().info(f"Opening serial port {port} @ {baud}")
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # allow Arduino to reset after port open

        self.current_deg = [90.0] * 6
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(0.1, self.publish_joint_states)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            action_name,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f"FollowJointTrajectory action server ready on {action_name}")

    def send_angles(self, angles_deg):
        angles_deg = [max(0, min(180, a)) for a in angles_deg]
        line = ",".join(str(round(a)) for a in angles_deg) + "\n"
        self.ser.write(line.encode())
        self.current_deg = angles_deg

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [math.radians(a) for a in self.current_deg]
        self.joint_pub.publish(msg)

    def execute_callback(self, goal_handle):
        traj = goal_handle.request.trajectory
        names = traj.joint_names
        self.get_logger().info(f"Executing trajectory with {len(traj.points)} points")

        t0 = time.time()
        for point in traj.points:
            angles_deg = [0.0] * 6
            for i, name in enumerate(names):
                if name in JOINT_NAMES:
                    idx = JOINT_NAMES.index(name)
                    angles_deg[idx] = math.degrees(point.positions[i])

            # wait until this point's scheduled time before sending
            target_t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            while (time.time() - t0) < target_t:
                time.sleep(0.005)

            self.send_angles(angles_deg)

            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = JOINT_NAMES
            feedback.actual.positions = [math.radians(a) for a in self.current_deg]
            goal_handle.publish_feedback(feedback)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
