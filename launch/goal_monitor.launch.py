# /* ----------------------------------------------------------------------------
#  * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Kota Kondo, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # tolerance arg
    goal_tol_arg = DeclareLaunchArgument(
        'goal_tolerance',
        default_value='0.5',
        description='Distance tolerance to consider a goal reached'
    )

    use_hardware_arg = DeclareLaunchArgument(
        'use_hardware',
        default_value='false',
        description='Use hardware mode (affects goal frame_id)'
    )
    goal_points_arg = DeclareLaunchArgument(
        'goal_points',
        default_value="[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]",
        description='List of goal points for the agents in the format: [[x1, y1, z1], [x2, y2, z2], ...]'
    )
    use_ground_robot_arg = DeclareLaunchArgument(
        'use_ground_robot',
        default_value='false',
        description='Ground robot mode (sets goal z to 0.2 instead of 1.0)'
    )

    def launch_setup(context):
        use_hardware = LaunchConfiguration('use_hardware').perform(context)
        goal_tolerance = LaunchConfiguration('goal_tolerance').perform(context)
        goal_points = LaunchConfiguration('goal_points').perform(context)
        # num_agents = int(LaunchConfiguration('num_agents').perform(context))
        # radius = float(LaunchConfiguration('radius').perform(context))
        # use_ground_robot = LaunchConfiguration('use_ground_robot').perform(context)

        ns = 'NX01'

        node = Node(
            package='mighty',
            executable='goal_monitor_node.py',
            namespace=ns,
            name='goal_monitor_node',
            output='screen',
            parameters=[{
                'goal_tolerance': float(goal_tolerance),
                'use_hardware': use_hardware.lower() in ('true', '1'),
                'goal_points': goal_points 
            }]
        )
            
        return [node]

    return LaunchDescription([
        goal_tol_arg,
        use_hardware_arg,
        goal_points_arg,
        use_ground_robot_arg,
        # num_agents_arg,
        # radius_arg,
        OpaqueFunction(function=launch_setup),
    ])
