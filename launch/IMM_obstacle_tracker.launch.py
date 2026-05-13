#!/usr/bin/env python3

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    frame_id = LaunchConfiguration('frame_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='ST01',
            description='ROS namespace for imm_obstacle_tracker'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value=['', namespace, '/map'],
            description='Frame ID for obstacle tracker (defaults to namespace/map)'
        ),
        Node(
            package='mighty',
            executable='IMM_obstacle_tracker_node', 
            name='imm_obstacle_tracker',
            namespace=namespace,
            remappings=[
            ],
            parameters=[{
                "frame_id": frame_id,
                "prob_transition_stay": 0.90,
                "imm_adapt_tpm": False,
                "imm_adapt_tpm_gain": 0.2,
                "imm_model_noise_cv": 0.03,
                "imm_model_noise_ca": 0.25,
                "imm_model_noise_sta": 0.05,
                "imm_p_init_pos_var": 0.1,
                "imm_p_init_vel_var": 5.0,
                "imm_p_init_acc_var": 25.0,
                "imm_r_meas_pos_var": 0.02
            }],
            output='screen'
        )
    ])
