import os
import random
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import TimerAction, OpaqueFunction, DeclareLaunchArgument

# --- CONFIGURATION ---
FIXED_Z_HEIGHT = 0.0
SPAWN_AREA = {"x_min": 2.5, "x_max": 150.0, "y_min": -3.0, "y_max": 3.0}
HUMAN_SPEED_RANGE = [0.5, 1.5]

# ---------- Helpers ----------
def _as_bool(context, name, default=False):
    raw = LaunchConfiguration(name).perform(context)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s == '':
        return default
    return s in ['true', '1', 'yes', 'on', 't']

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

def get_human_trajectory(mode, x, y, z, speed, x_min=0, x_max=10, y_min=0, y_max=10):
    """
    Generates trajectory strings for the URDF/Xacro to evaluate.
    """
    traj_x = str(x)
    traj_y = str(y)
    traj_z = str(z)

    if mode == "FORWARD_X":
        # Simple linear motion: x(t) = x0 + v*t
        traj_x = f"{x}+({speed}*t)"

    elif mode == "PATROL_Y":
        # Oscillates in Y
        amplitude = (y_max - y_min) / 2.0
        omega = speed / (amplitude if amplitude > 0 else 1.0)
        phase = round(random.uniform(0, 6.28), 2)
        traj_y = f"{0}+{amplitude}*sin({omega}*t+{phase})"

    elif mode == "CORRIDOR":
        # Oscillates in X between x_min and x_max
        # Formula: center + amplitude * sin(speed * t + phase)
        # where amplitude = (max - min) / 2
        # center = (max + min) / 2

        amplitude = (x_max - x_min) / 2.0
        center = (x_max + x_min) / 2.0
        phase = round(random.uniform(0, 6.28), 2)

        # We adjust speed slightly so the "period" isn't identical for everyone
        # angular_freq (omega) ~ speed / amplitude
        omega = speed / (amplitude if amplitude > 0 else 1.0)

        traj_x = f"{center}+{amplitude}*sin({omega}*t+{phase})"
    elif mode == "ONE_WAY_Y":
        # 1. Calculate the total span of the area
        span = y_max - y_min

        # 2. Calculate where the human starts relative to y_min
        # This ensures they don't all teleport at the exact same time
        initial_offset = y - y_min

        # 3. Sawtooth Wave formula:
        # y(t) = y_min + ( (initial_offset + speed*t) % span )
        # We use fmod() which is the standard C++ math function for floating point modulo
        traj_y = f"{y_min}+(({initial_offset}+{speed}*t)%{span})"
    elif mode == "ONE_WAY_X" or mode == "MIDDLE":
        initial_offset = x_max - x
        span_x = x_max - x_min
        traj_x = f"{x_max}-(({initial_offset}+{speed}*t)%{span_x})"
    elif mode == "OVERTAKE":
        initial_offset = x - x_min
        span_x = x_max - x_min
        traj_x = f"{x_min}+(({initial_offset}+{speed}*t)%{span_x})"

    # 'STATIC' is just default (no change)

    return traj_x, traj_y, traj_z

def generate_human_entities(context):
    """
    Main logic to spawn human entities. Returns a list of Nodes.
    """
    actions = []

    num_pedestrians     = _as(context, 'num_obstacles', int, 10)
    mode                = _as(context, 'mode', str, 'ONE_WAY_X')
    urdf_xacro          = _as(context, 'urdf_xacro', str, 'human_box.urdf.xacro')
    spawn_interval      = _as(context, 'spawn_interval', float, 1.0)
    use_sim_time        = _as_bool(context, 'use_sim_time', True) 
    seed                = _as(context, 'seed', int, 42)

    random.seed(seed)

    urdf_path = os.path.join(get_package_share_directory('mighty'), 'urdf', urdf_xacro)

    # --- LANE CALCULATIONS FOR CORRIDOR ---
    # We want to distribute Y positions evenly to avoid collisions
    lane_width = (SPAWN_AREA["y_max"] - SPAWN_AREA["y_min"]) / num_pedestrians
    current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)
    traj_mode = mode

    for i in range(num_pedestrians):
        # 1. Generate Parameters
        if mode == "CORRIDOR" or mode == "ONE_WAY_X":
            x = round(SPAWN_AREA["x_max"], 2)
            y = round(random.uniform(current_y - lane_width/2.0, current_y + lane_width/2.0), 2)
            y = min(SPAWN_AREA["y_max"], y)
            current_y += lane_width
            if current_y > SPAWN_AREA["y_max"]:
                current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)
        elif mode == 'MIDDLE':
            assert num_pedestrians == 1
            x = round(150.0, 2)
            y = 0.00

        elif mode == 'OVERTAKE':
            x = round(random.uniform(SPAWN_AREA["x_min"], 10.0), 2)
            y = round(random.uniform(-2.0, 2.0), 2)  # Start in the upper half to encourage overtaking

        elif mode == "TWO_WAY_X": # Alternate directions to get an even 50/50 split of pedestrians
            is_forward = (i % 2 == 0)
            
            # Spread them evenly across the Y-axis lanes to prevent immediate clumping
            y = round(random.uniform(current_y - lane_width/2.0, current_y + lane_width/2.0), 2)
            y = min(SPAWN_AREA["y_max"], y)
            current_y += lane_width
            if current_y > SPAWN_AREA["y_max"]:
                current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)
            
            if is_forward:
                # Move +X: Start near x_min, use your existing OVERTAKE logic
                x = round(random.uniform(SPAWN_AREA["x_min"], SPAWN_AREA["x_min"] + 15.0), 2)
                traj_mode = "OVERTAKE" 
            else:
                # Move -X: Start near x_max, use your existing ONE_WAY_X logic
                x = round(random.uniform(SPAWN_AREA["x_max"] - 15.0, SPAWN_AREA["x_max"]), 2)
                traj_mode = "ONE_WAY_X"
        
        elif mode == "FULL":
            # Split pedestrians into 3 groups: +X, -X, and +Y (cross-traffic)
            group = i % 3

            if group == 0:
                # Move +X
                x = round(random.uniform(SPAWN_AREA["x_min"], SPAWN_AREA["x_min"] + 15.0), 2)
                y = round(random.uniform(current_y - lane_width/2.0, current_y + lane_width/2.0), 2)
                y = min(SPAWN_AREA["y_max"], y)
                traj_mode = "OVERTAKE"
                
                current_y += lane_width
                if current_y > SPAWN_AREA["y_max"]:
                    current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)

            elif group == 1:
                # Move -X
                x = round(random.uniform(SPAWN_AREA["x_max"] - 15.0, SPAWN_AREA["x_max"]), 2)
                y = round(random.uniform(current_y - lane_width/2.0, current_y + lane_width/2.0), 2)
                y = min(SPAWN_AREA["y_max"], y)
                traj_mode = "ONE_WAY_X"

                current_y += lane_width
                if current_y > SPAWN_AREA["y_max"]:
                    current_y = SPAWN_AREA["y_min"] + (lane_width / 2.0)

            else:
                # Move +Y (Cross-Traffic)
                # Spread them randomly across the X corridor span
                x = round(random.uniform(SPAWN_AREA["x_min"], SPAWN_AREA["x_max"]), 2)
                # Give them a random starting Y so they don't all cross at once
                y = round(random.uniform(-5.0, 5.0), 2)
                traj_mode = "ONE_WAY_Y"
        else:
            # Random X and Y for other modes
            x = round(random.uniform(SPAWN_AREA["x_min"], SPAWN_AREA["x_max"]), 2)
            y = round(random.uniform(SPAWN_AREA["y_min"], SPAWN_AREA["y_max"]), 2)

        z = FIXED_Z_HEIGHT
        speed = round(random.uniform(HUMAN_SPEED_RANGE[0], HUMAN_SPEED_RANGE[1]), 2)

        # 2. Get Trajectory Strings
        # Note: We pass Spawn Area limits specifically for Corridor calculation
        traj_x_str, traj_y_str, traj_z_str = get_human_trajectory(
            traj_mode,
            x,
            y,
            z,
            speed,
            SPAWN_AREA["x_min"],
            SPAWN_AREA["x_max"],
            SPAWN_AREA["y_min"],
            SPAWN_AREA["y_max"],
        )

        namespace = f"human_{i}"

        # 3. Create Robot Description (Xacro)
        # We compute the Command now and pass it as a string to avoid complex substitutions
        robot_desc = Command(
            [
                "xacro ",
                urdf_path,
                ' traj_x:="',
                traj_x_str,
                '"',
                ' traj_y:="',
                traj_y_str,
                '"',
                ' traj_z:="',
                traj_z_str,
                '"',
                " namespace:=",
                namespace,
            ]
        )

        # 4. Robot State Publisher Node
        rsp_node = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name=f"rsp_{namespace}",
            output="screen",
            namespace=namespace,
            parameters=[
                {
                    "robot_description": ParameterValue(robot_desc, value_type=str),
                    "use_sim_time": use_sim_time,  
                    "frame_prefix": namespace + "/",
                }
            ],
        )

        # 5. Spawn Entity Node (Gazebo)
        spawn_node = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name=f"spawn_{namespace}",
            output="screen",
            arguments=[
                "-topic",
                f"{namespace}/robot_description",  # Use the specific topic
                "-entity",
                namespace,
                "-x",
                "0.0",
                "-y",
                "0.0",
                "-z",
                "0.0",  # URDF handles the internal position logic
            ],
        )

        # Stagger spawning to prevent Gazebo service overload
        # We group the RSP and Spawn together for clarity
        actions.append(
            TimerAction(
                period=i * spawn_interval,
                actions=[rsp_node,
                         TimerAction(period=0.3, actions=[spawn_node])]
            )
        )

    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('num_obstacles', default_value='10'),
        DeclareLaunchArgument('mode', default_value='ONE_WAY_X'),
        DeclareLaunchArgument('urdf_xacro', default_value='human_box.urdf.xacro'),
        DeclareLaunchArgument('spawn_interval', default_value='1.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('seed', default_value='42')
    ]

    ld = LaunchDescription(args)

    # 1. Human Spawner (OpaqueFunction executes the loop above)
    ld.add_action(OpaqueFunction(function=lambda context: generate_human_entities(context)))

    # 2. Fake Simulation Perception Node
    # Collects ground truth from all humans and publishes 'detected_objects'
    perception_node = Node(
        package="mighty",
        executable="fake_sim_perception_node",
        name="fake_sim_perception",
        output="screen",
        parameters=[{
            "num_humans": LaunchConfiguration('num_obstacles'), 
            "frame_id": "map"
        }],
    )
    ld.add_action(perception_node)

    # 3. Tracker Node
    # Subscribes to 'detected_objects' -> Publishes 'predicted_trajs'
    tracker_node = Node(
        package="mighty",
        executable="IMM_obstacle_tracker_prediction_node",
        name="IMM_obstacle_tracker_prediction",
        output="screen",
        namespace="NX01",
        remappings=[
            ("detected_objects", "/detected_objects"),
        ],
        parameters=[
            {
                "visual_level": 2,
                "prediction_horizon": 3.0,  # seconds
            }
        ],
    )
    ld.add_action(tracker_node)

    return ld

