from setuptools import setup

package_name = 'arm_moveit_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bridge.launch.py']),
        ('share/' + package_name + '/config', ['config/moveit_controllers.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arun',
    maintainer_email='you@example.com',
    description='MoveIt2 <-> Arduino serial bridge for a 6-axis SG90/MG995 arm',
    license='MIT',
    entry_points={
        'console_scripts': [
            'serial_bridge_node = arm_moveit_bridge.serial_bridge_node:main',
        ],
    },
)
