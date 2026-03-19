import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def convert_str_to_bool(str):
    return True if (str == 'true' or str == 'True' or str == 1 or str == '1') else False

def generate_launch_description():
    
    # Declare a new argument "env" to choose the world file.
    env_arg = DeclareLaunchArgument(
        'env',
        default_value='easy_forest',
        description='Environment name to determine the Gazebo world file'
    )

    # Remove the previous world_path argument and use env instead.
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true', description='Flag to enable or disable RViz'
    )
    use_gazebo_gui_arg = DeclareLaunchArgument(
        'use_gazebo_gui', default_value='false', description='Flag to enable or disable gazebo gui'
    )
    use_dyn_obs_arg = DeclareLaunchArgument(
        'use_dyn_obs', default_value='true', description='Flag to enable or disable dynamic obstacles'
    )
    use_ground_robot_arg = DeclareLaunchArgument(
        'use_ground_robot', default_value='false', description='Use ground robot (affects RViz config)'
    )
    use_hunav_sim_arg = DeclareLaunchArgument(
        'use_hunav_sim', default_value='false', description='Flag to enable or disable hunav sim components (loader, world gen, manager)'
    )
    use_ped_obstacles_arg = DeclareLaunchArgument(
        'use_ped_obstacles', default_value='false', description='Flag to enable or disable pedestrian obstacles'
    )

    # benchmark name
    benchmark_name_arg = DeclareLaunchArgument('benchmark_name', default_value='benchmark_name', description='Benchmark name')

    # Opaque function to launch nodes
    def launch_setup(context, *args, **kwargs):
        
        # Get the environment value from the 'env' launch argument.
        env_value = LaunchConfiguration('env').perform(context)
        # Map environment names to corresponding Gazebo world file names.
        world_mapping = {
            'high_res_forest': 'big_forest_high_res.world',
            'static_uncertainty_test2': 'static_uncertainty_test2.world',
            'static_uncertainty_test3': 'static_uncertainty_test3.world',
            'static_uncertainty_test4': 'static_uncertainty_test4.world',
            'office_faster': 'office.world',
            'office': 'office.world',
            'cave_start': 'simple_tunnel.world',
            'cave_vertical': 'simple_tunnel.world',
            'cave_person': 'simple_tunnel.world',
            'forest3': 'forest3.world',
            'yaw_benchmark': 'forest3.world',
            'global_planner': 'forest3.world',
            'multiagent_performance': 'forest3.world',
            'path_push': 'forest3.world',
            'ACL_office': 'ACL_office.world',
            'ground_robot': 'ACL_office.world',
            'multiagent_testing': 'empty.world',
            'empty_wo_ground': 'empty_wo_ground.world',
            'empty': 'empty.world',
            'hospital': 'hospital.world',
            'easy_forest': 'easy_forest.world',
            'easy_high_forest': 'easy_high_forest.world',
            'medium_forest': 'medium_forest.world',
            'hard_forest': 'hard_forest.world',
            'dynamic_forest': 'dynamic_forest.world',
            'empty_corridor': 'empty_corridor.world',
            'T_junction': 'T_junction.world',
            'corridor_overtake': 'corridor_overtake.world'
        }

        # Choose the world file based on the provided environment.
        world_file = world_mapping.get(env_value, 'easy_forest.world')
        world_path = PathJoinSubstitution([FindPackageShare('mighty'), 'worlds', world_file])

        use_rviz = convert_str_to_bool(LaunchConfiguration('use_rviz').perform(context))
        use_dyn_obs = convert_str_to_bool(LaunchConfiguration('use_dyn_obs').perform(context))
        use_gazebo_gui = LaunchConfiguration('use_gazebo_gui').perform(context)
        use_ground_robot = convert_str_to_bool(LaunchConfiguration('use_ground_robot').perform(context))
        use_hunav_sim = convert_str_to_bool(LaunchConfiguration('use_hunav_sim').perform(context))
        use_ped_obstacles = convert_str_to_bool(LaunchConfiguration('use_ped_obstacles').perform(context))

        # Create a rviz node - use ground robot config if available, otherwise use default
        rviz_config_filename = 'mighty_sim_ground_robot.rviz' if use_ground_robot else 'mighty.rviz'
        rviz_config_file = os.path.join(
            get_package_share_directory('mighty'),
            'rviz',
            rviz_config_filename
        )
        # Fallback to default if ground robot config doesn't exist
        if use_ground_robot and not os.path.exists(rviz_config_file):
            rviz_config_file = os.path.join(
                get_package_share_directory('mighty'),
                'rviz',
                'mighty.rviz'
            )

        rviz_node = Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    output='log',
                    emulate_tty=True,
                    arguments=['-d', rviz_config_file, '--ros-args', '--log-level', 'error'],
                    parameters=[{'use_sim_time': False}]
                )

        # Include Gazebo launch file
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
            ),
            launch_arguments={'world': world_path, 'use_sim_time': 'false', 'gui': use_gazebo_gui, 'enable_gpu': 'true'}.items()
        )

        # Number of dynamic obstacles
        num_dyn_obstacles = 10 if env_value in ["empty_wo_ground"] else 10

        # Dynamic obstacles
        if use_dyn_obs:
            dynamic_obstacles_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare('mighty'), 'launch', 'dyn_obstacles.launch.py'])
                ),
                launch_arguments={"num_obstacles": f"{num_dyn_obstacles}",
                                    "publish_rate_hz": "50.0",
                                    "seed": "0",
                                    "launch_forest_node":"true",
                                    "forest_start_delay":"2.0",
                                    "spawn_interval": "1.0",
                                    }.items()
            )

        peds_obstacles_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare('mighty'), 'launch', 'dyn_obstacles_pedestrians.launch.py'])
                ),
                launch_arguments={
                    "num_obstacles": "25",
                    # "urdf_xacro": "human_box.urdf.xacro",
                    # "spawn_interval": "1.0",
                    # "use_sim_time": 'true',
                    "env_value": env_value,
                    "seed": '25',
                }.items()
            )

        to_launch = []
        if use_hunav_sim:
            from launch.actions import TimerAction
            robot_name = 'NX01'

            assert env_value in ['empty_corridor'], "Hunav sim only works with a few environments."
            
            # 1. Load the agents YAML
            agents_yaml_path = os.path.join(get_package_share_directory('hunav_gazebo_wrapper'), 'scenarios', 'simple_agents.yaml')
            hunav_loader_node = Node(
                package='hunav_agent_manager',
                executable='hunav_loader',
                name='hunav_loader',
                output='screen',
                parameters=[agents_yaml_path]
            )

            generated_world_path = PathJoinSubstitution([
                FindPackageShare('mighty'), 'worlds', 'generatedWorld.world'
            ])

            # 2. Generate the dynamic world
            hunav_gazebo_worldgen_node = Node(
                        package='hunav_gazebo_wrapper',
                        executable='hunav_gazebo_world_generator',
                        name='hunav_gazebo_world_generator',
                        output='screen',
                        parameters=[{
                            'base_world': world_path,
                            'agents_file': agents_yaml_path,         
                            'generated_world_path': generated_world_path, 
                            'use_gazebo_obs': True,
                            'update_rate': 1000.0,
                            'robot_name': robot_name, 
                            'global_frame_to_publish': 'map',
                            'use_navgoal_to_start': False,
                            'ignore_models': 'ground_plane',
                            'use_collision': True,
                        }]
                    )
            # 3. The manager that calculates the social forces
            hunav_manager_node = Node(
                package='hunav_agent_manager',
                executable='hunav_agent_manager',
                name='hunav_agent_manager',
                output='screen',
                parameters=[{
                    'robot_name': robot_name,
                }],
            )


            gazebo_launch = TimerAction(
                period=3.0, # Wait 3 seconds for world generator to finish
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            PathJoinSubstitution([FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'])
                        ),
                        launch_arguments={'world': generated_world_path, 'use_sim_time': 'false', 'gui': use_gazebo_gui, 'enable_gpu': 'true'}.items()
                    )
                ]
            )

            static_tf_bridge = Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_bridge',
                arguments=['0', '0', '0', '0', '0', '0', 'NX01', 'NX01/base_link']
            )

            to_launch.extend([static_tf_bridge, hunav_loader_node, hunav_gazebo_worldgen_node, hunav_manager_node])


        # Return launch description
        nodes_to_start = [gazebo_launch]
        nodes_to_start.append(rviz_node) if use_rviz else None
        # nodes_to_start.append(dynamic_obstacles_launch) if use_dyn_obs else None
        nodes_to_start.append(peds_obstacles_launch) if use_ped_obstacles else None
        nodes_to_start.extend(to_launch) if use_hunav_sim else None

        return nodes_to_start

    return LaunchDescription([
        env_arg,
        use_rviz_arg,
        use_gazebo_gui_arg,
        use_dyn_obs_arg,
        use_ground_robot_arg,
        use_hunav_sim_arg,
        use_ped_obstacles_arg,
        benchmark_name_arg,
        OpaqueFunction(function=launch_setup)
    ])
