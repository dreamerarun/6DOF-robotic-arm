#!/usr/bin/env python3
"""
arduino_arm_bridge.py
----------------------------------------------------------------
ROS2 Humble node that lets MoveIt2 drive the real Arduino arm.

MoveIt talks to "controllers" over the FollowJointTrajectory action
interface. This node *is* that controller: it exposes a
control_msgs/action/FollowJointTrajectory action server, and for every
goal it receives, walks through the trajectory points and streams
"P,a0,a1,a2,a3,a4,a5" pose commands to the Arduino sketch over serial,
converting from radians (ROS/URDF convention) to degrees (0-180, what
the sketch expects). It also publishes /joint_states so RViz/MoveIt
shows the arm where it actually is.

Install deps:
    pip install pyserial
    sudo apt install ros-humble-control-msgs

Run:
    ros2 run <your_package> arduino_arm_bridge.py --ros-args \
        -p serial_port:=/dev/ttyUSB0 -p baud:=115200

moveit_controllers.yaml should reference this as a controller, e.g.:

    moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager
    moveit_simple_controller_manager:
      controller_names: [arm_controller]
      arm_controller:
        action_ns: follow_joint_trajectory
        type: FollowJointTrajectory
        default: true
        joints:
          - joint_base
          - joint_shoulder
          - joint_elbow
          - joint_wrist_pitch
          - joint_wrist_roll
          - joint_gripper

The action server name below ("arm_controller") must match that.
----------------------------------------------------------------
"""

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

import serial

# Must match the URDF joint order == Arduino channel order (0-5)
JOINT_NAMES = [
    "joint_base",
    "joint_shoulder",
    "joint_elbow",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_gripper",
]

# If a joint's URDF limits aren't simply 0..pi mapped to the servo's
# 0..180 degrees, adjust the conversion here per-joint.
def rad_to_servo_deg(rad):
    deg = math.degrees(rad)
    # Shift/clamp assuming URDF limit is roughly [-pi/2, pi/2] -> [0,180].
    # Change this mapping to match how you defined joint limits in your URDF.
    deg = deg + 90.0
    return max(0, min(180, int(round(deg))))


def servo_deg_to_rad(deg):
    return math.radians(deg - 90.0)


class ArduinoArmBridge(Node):
    def __init__(self):
        super().__init__("arduino_arm_bridge")

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        port = self.get_parameter("serial_port").value
        baud = self.get_parameter("baud").value

        self.get_logger().info(f"Opening serial port {port} @ {baud}")
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)  # let the Uno reset after the port opens
        self.ser.reset_input_buffer()

        self.current_deg = [90, 90, 90, 90, 90, 90]
        self.lock = threading.Lock()

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.state_timer = self.create_timer(0.1, self.publish_state)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "arm_controller/follow_joint_trajectory",
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info("arduino_arm_bridge ready")

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        with self.lock:
            msg.position = [servo_deg_to_rad(d) for d in self.current_deg]
        self.joint_pub.publish(msg)

    def send_pose(self, degs):
        cmd = "P," + ",".join(str(d) for d in degs) + "\n"
        self.ser.write(cmd.encode())
        # Wait for the Arduino's "OK"/"ERR" reply so we don't flood it
        line = self.ser.readline().decode(errors="ignore").strip()
        return line

    def execute_callback(self, goal_handle):
        traj = goal_handle.request.trajectory
        names = traj.joint_names
        # map incoming joint order to our fixed JOINT_NAMES/channel order
        order = [names.index(n) for n in JOINT_NAMES]

        result = FollowJointTrajectory.Result()

        prev_t = 0.0
        for point in traj.points:
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            wait = max(0.0, t - prev_t)
            prev_t = t

            degs = []
            for i in order:
                degs.append(rad_to_servo_deg(point.positions[i]))

            reply = self.send_pose(degs)
            with self.lock:
                self.current_deg = degs

            if reply != "OK":
                self.get_logger().warn(f"Arduino replied '{reply}', expected OK")

            # pace waypoints roughly to the trajectory's own timing
            time.sleep(min(wait, 2.0))

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
