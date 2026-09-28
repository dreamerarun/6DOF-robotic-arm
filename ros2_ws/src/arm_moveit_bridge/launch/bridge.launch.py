from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baud_rate', default_value='115200'),

        Node(
            package='arm_moveit_bridge',
            executable='serial_bridge_node',
            name='arm_serial_bridge',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
                'action_name': '/arm_controller/follow_joint_trajectory',
            }],
        ),
    ])
