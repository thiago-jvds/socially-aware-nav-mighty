import os
import random
import json
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import OpaqueFunction, DeclareLaunchArgument, TimerAction


def _as(context, name, cast, default):
    """
    Safe retrieval & casting of a LaunchConfiguration.
    If the argument is absent or empty, returns default; otherwise casts.
    """
    raw = LaunchConfiguration(name).perform(context)
    if raw is None:
        print(f"[dyn_obstacles] {name}=None -> default {default}")
        return default
    s = raw.strip()
    if s == '':
        print(f"[dyn_obstacles] {name}=<empty> -> default {default}")
        return default
    try:
        return cast(s)
    except ValueError:
        raise RuntimeError(f"[dyn_obstacles] Argument '{name}' expected {cast.__name__}, got '{s}'")


def get_human_trajectory(
    traj_mode, x, y, z, speed, x_min=0, x_max=10, y_min=0, y_max=10
):
    """
    Generates trajectory strings for the URDF/Xacro to evaluate.
    """
    traj_x = str(x)
    traj_y = str(y)
    traj_z = str(z)

    if traj_mode == "FORWARD_X":
        traj_x = f"{x}+({speed}*t)"

    elif traj_mode == "PATROL_Y":
        amplitude = (y_max - y_min) / 2.0
        omega = speed / (amplitude if amplitude > 0 else 1.0)
        phase = round(random.uniform(0, 6.28), 2)
        traj_y = f"{0}+{amplitude}*sin({omega}*t+{phase})"

    elif traj_mode == "CORRIDOR":
        amplitude = (x_max - x_min) / 2.0
        center = (x_max + x_min) / 2.0
        phase = round(random.uniform(0, 6.28), 2)
        omega = speed / (amplitude if amplitude > 0 else 1.0)
        traj_x = f"{center}+{amplitude}*sin({omega}*t+{phase})"

    elif traj_mode == "ONE_WAY_Y":
        span = y_max - y_min
        initial_offset = y - y_min
        traj_y = f"{y_min}+(({initial_offset}+{speed}*t)%{span})"

    elif traj_mode == "ONE_WAY_X" or traj_mode == "MIDDLE":
        initial_offset = x_max - x
        span_x = x_max - x_min
        traj_x = f"{x_max}-(({initial_offset}+{speed}*t)%{span_x})"

    elif traj_mode == "OVERTAKE":
        initial_offset = x - x_min
        span_x = x_max - x_min
        traj_x = f"{x_min}+(({initial_offset}+{speed}*t)%{span_x})"

    return traj_x, traj_y, traj_z


FIXED_Z_HEIGHT = 0.0
SPAWN_AREA = {"x_min": 2.5, "x_max": 150.0, "y_min": -3.0, "y_max": 3.0}
HUMAN_SPEED_RANGE = [0.5, 1.5]

_PEDESTRIANS_JSON_STORAGE = []


def generate_human_entities(context):
    actions = []

    urdf_path = os.path.join(
        get_package_share_directory("mighty"), "urdf", "human_box.urdf.xacro"
    )

    num_obstacles = _as(context, "num_obstacles", int, 10)
    env_value = _as(context, "env_value", str, "unknown")
    seed = _as(context, "seed", int, 42)
    random.seed(seed)

    if env_value == "empty_corridor":
        SPAWN_AREA = {"x_min": -95.0, "x_max": 95.0, "y_min": -5.0, "y_max": 5.0}
    elif env_value == "T_junction":
        SPAWN_AREA = {"x_min": -25.0, "x_max": 70.0, "y_min": -2.5, "y_max": 2.5}
    else:
        SPAWN_AREA = {"x_min": -10.0, "x_max": 10.0, "y_min": -2.0, "y_max": 2.0}

    lane_width = (SPAWN_AREA["y_max"] - SPAWN_AREA["y_min"]) / (
        num_obstacles if num_obstacles > 0 else 1
    )
    current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)

    for i in range(num_obstacles):
        human_name = f"human_{i}"
        traj_x_min = SPAWN_AREA["x_min"]
        traj_x_max = SPAWN_AREA["x_max"]
        traj_y_min = SPAWN_AREA["y_min"]
        traj_y_max = SPAWN_AREA["y_max"]

        traj_mode = "STATIC"

        z = FIXED_Z_HEIGHT
        speed = round(random.uniform(HUMAN_SPEED_RANGE[0], HUMAN_SPEED_RANGE[1]), 2)

        if env_value == "empty_corridor":
            group = i % 3
            if group == 0:
                traj_x_min = 2.5
                traj_y_min = -5.0
                traj_y_max = 5.0

                x = round(random.uniform(traj_x_min, traj_x_min + 30.0), 2)
                y = round(
                    random.uniform(
                        current_y - lane_width / 2.0, current_y + lane_width / 2.0
                    ),
                    2,
                )
                y = min(traj_y_max, y)
                traj_mode = "OVERTAKE"

                current_y += lane_width
                if current_y > traj_y_max:
                    current_y = traj_y_min + (lane_width / 2.0)

            elif group == 1:
                traj_x_max = 95.0
                traj_y_min = -5.0
                traj_y_max = 5.0

                x = round(random.uniform(traj_x_max - 30.0, traj_x_max), 2)
                y = round(
                    random.uniform(
                        current_y - lane_width / 2.0, current_y + lane_width / 2.0
                    ),
                    2,
                )
                y = min(traj_y_max, y)
                traj_mode = "ONE_WAY_X"

                current_y += lane_width
                if current_y > traj_y_max:
                    current_y = traj_y_min + (lane_width / 2.0)

            else:
                traj_x_max = 150.0
                traj_x_min = 5.0
                traj_y_min = -7.0
                traj_y_max = 7.0

                x = round(random.uniform(traj_x_min, traj_x_max), 2)
                y = round(random.uniform(traj_y_min, traj_y_max), 2)
                traj_mode = "ONE_WAY_Y"

        elif env_value == "T_junction":
            group = i % 3

            if group == 0:
                traj_x_min = -25.0
                traj_x_max = 70.0
                traj_y_min = -3.0
                traj_y_max = 3.0

                x = round(random.uniform(traj_x_min, 0.0), 2)
                y = round(random.uniform(traj_y_min, traj_y_max), 2)
                traj_mode = "OVERTAKE"

            elif group == 1:
                traj_x_min = -25.0
                traj_x_max = 70.0
                traj_y_min = -3.0
                traj_y_max = 3.0
                x = round(random.uniform(40.0, traj_x_max), 2)
                y = round(random.uniform(traj_y_min, traj_y_max), 2)
                traj_mode = "ONE_WAY_X"

            else:
                traj_x_min = 72.0
                traj_x_max = 78.0
                traj_y_min = -45.0
                traj_y_max = 45.0

                x = round(random.uniform(traj_x_min, traj_x_max), 2)
                y = round(random.uniform(-40.0, 0.0), 2)
                traj_mode = "ONE_WAY_Y"

        elif env_value == "corridor_overtake":
            group = i % 2
            if group == 0:
                traj_x_max = 200.0
                traj_x_min = -5.0

                x = round(random.uniform(traj_x_min, traj_x_max), 2)
                y = round(0.0, 2)

                traj_mode = "OVERTAKE"
                speed = 1.0

            elif group == 1:
                traj_x_max = 200.0
                traj_x_min = -5.0

                x = round(random.uniform(traj_x_min, traj_x_max), 2)
                y = round(-2.0, 2)
                traj_mode = "ONE_WAY_X"

                speed = 1.8

        elif env_value == "social_crossing":
            traj_x_max = 200.0
            traj_x_min = 5.0
            traj_y_min = -7.0
            traj_y_max = 7.0

            cur_loc = i * 10.0 + traj_x_min
            if cur_loc > traj_x_max:
                continue

            x = round(cur_loc, 2)
            y = round(random.uniform(traj_y_min, traj_y_max), 2)
            traj_mode = "ONE_WAY_Y"

        elif env_value == "social_passing":
            traj_x_max = 95.0
            traj_x_min = 5.0
            traj_y_min = -5.0
            traj_y_max = 5.0

            x = round(random.uniform(traj_x_max - 50.0, traj_x_max), 2)
            y = round(random.uniform(0.0, 2.0), 2)
            y = min(traj_y_max, y)
            traj_mode = "ONE_WAY_X"

        elif env_value == "social_overtaking":
            traj_x_max = 200.0
            traj_x_min = 5.0

            x = round(random.uniform(traj_x_min, traj_x_max), 2)
            y = round(0.0, 2)

            traj_mode = "OVERTAKE"
            speed = 1.0

        else:
            x = round(random.uniform(SPAWN_AREA["x_min"], SPAWN_AREA["x_max"]), 2)
            y = round(random.uniform(SPAWN_AREA["y_min"], SPAWN_AREA["y_max"]), 2)
            traj_mode = "STATIC"

        traj_x_str, traj_y_str, traj_z_str = get_human_trajectory(
            traj_mode,
            x,
            y,
            z,
            speed,
            traj_x_min,
            traj_x_max,
            traj_y_min,
            traj_y_max,
        )

        _PEDESTRIANS_JSON_STORAGE.append(
            {
                "name": human_name,
                "traj_x": traj_x_str,
                "traj_y": traj_y_str,
                "traj_z": traj_z_str,
                "x0": x,
                "y0": y,
            }
        )

        robot_description = ParameterValue(
            Command([
                'xacro ', urdf_path,
                ' traj_x:=', traj_x_str,
                ' traj_y:=', traj_y_str,
                ' traj_z:=', traj_z_str,
                ' robot_name:=', human_name,
            ]),
            value_type=str
        )

        rsp_node = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=human_name,
            name="robot_state_publisher",
            parameters=[
                {
                    "robot_description": robot_description,
                    'frame_prefix': human_name + '/'
                }
            ],
            remappings=[('/robot_description', f'/{human_name}/robot_description')]
        )

        spawn_node = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name=f'{human_name}_spawn',
            arguments=[
                "-entity", human_name,
                "-topic", f"/{human_name}/robot_description",
                "-x", "0.0",
                "-y", "0.0",
                "-z", "0.0",
            ],
            output="screen",
        )
        actions.append(
            TimerAction(
                period=i * 1.0,
                actions=[rsp_node, TimerAction(period=0.3, actions=[spawn_node])]
            )
        )

    formatted_json = json.dumps(_PEDESTRIANS_JSON_STORAGE, indent=4)
    print(f"\n[LAUNCH DEBUG] Generated Pedestrians JSON:\n{formatted_json}\n")

    central_node = Node(
        package="mighty",
        executable="dynamic_pedestrian_node",
        name="dynamic_pedestrian_node",
        output="screen",
        parameters=[
            {
                "pedestrians_json": json.dumps(_PEDESTRIANS_JSON_STORAGE),
                "publish_rate_hz": 50.0,
                "global_frame": "map",
                "base_link_name": "base_link",
            }
        ],
    )
    actions.append(
        TimerAction(period=1.0, actions=[central_node])
    )

    return actions


def generate_launch_description():
    ld = LaunchDescription(
        [
            DeclareLaunchArgument("num_obstacles", default_value="10"),
            DeclareLaunchArgument("env_value", default_value="unknown"),
        ]
    )

    ld.add_action(OpaqueFunction(function=generate_human_entities))

    return ld
