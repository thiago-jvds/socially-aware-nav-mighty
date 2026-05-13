#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Namespace for the fixed state publisher",
    )

    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="state",
        description="Output topic for dynus_interfaces/State",
    )

    node = Node(
        package="mighty",
        executable="fixed_state_publisher.py",
        name="fixed_state_publisher",
        namespace=LaunchConfiguration("namespace"),
        remappings=[("state", LaunchConfiguration("state_topic"))],
        output="screen",
    )

    return LaunchDescription([
        namespace_arg,
        state_topic_arg,
        node,
    ])
