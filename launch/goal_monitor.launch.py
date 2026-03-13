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

    def launch_setup(context):
        use_hardware = LaunchConfiguration('use_hardware').perform(context)
        goal_tolerance = LaunchConfiguration('goal_tolerance').perform(context)
        goal_points = LaunchConfiguration('goal_points').perform(context)

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
        OpaqueFunction(function=launch_setup),
    ])
