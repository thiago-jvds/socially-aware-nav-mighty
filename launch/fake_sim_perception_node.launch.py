#!/usr/bin/env python3

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    truth_source = LaunchConfiguration('truth_source')
    frame_id = LaunchConfiguration('frame_id')
    num_humans = LaunchConfiguration('num_humans')
    human_name_prefix = LaunchConfiguration('human_name_prefix')
    detection_rate_hz = LaunchConfiguration('detection_rate_hz')


    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='ST01',
            description='ROS namespace for fake_sim_perception_node'
        ),
        DeclareLaunchArgument(
            'truth_source',
            default_value='odom',
            description='Truth source: odom, gazebo_model_states, or auto'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='Frame id for published detections'
        ),
        DeclareLaunchArgument(
            'num_humans',
            default_value='10',
            description='Number of humans to include'
        ),
        DeclareLaunchArgument(
            'human_name_prefix',
            default_value='human_',
            description='Gazebo model name prefix used to identify humans'
        ),
        DeclareLaunchArgument(
            'detection_rate_hz',
            default_value='10.0',
            description='Detection publishing rate'
        ),
        Node(
            package='mighty',
            executable='fake_sim_perception_node', 
            name='fake_sim_perception_node',
            namespace=namespace,
            remappings=[
            ],
            parameters=[{
                'truth_source': truth_source,
                'frame_id': frame_id,
                'num_humans': num_humans,
                'human_name_prefix': human_name_prefix,
                'detection_rate_hz': detection_rate_hz,
            }],
            output='screen'
        )
    ])
