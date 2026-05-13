#!/usr/bin/env python3

import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    namespace = 'ST01'
    return LaunchDescription([
        Node(
            package='mighty',
            executable='convert_vicon_to_obstacles', 
            name='convert_vicon_to_obstacles',
            namespace=namespace,
            remappings=[
                ('world', 'world'),  # Remap incoming PoseStamped topic
            ],
            output='screen'
        )
    ])
