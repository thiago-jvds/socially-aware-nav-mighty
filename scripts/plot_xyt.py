#!/usr/bin/env python3
"""
Plot XY trajectories from ROS2 bags for two PoseStamped topics

Scans bags in ~/livox_dataset/bags/hw_* and looks for topics
  /Lucas6/world and /STAR/world

Saves PNG plots to ~/livox_dataset/plots_hw_mighty/

This script tries to use rosbag2_py and rclpy for reading ROS2 bags.
If those imports are unavailable, it will print an error explaining
what's missing.
"""
import os
import glob
import argparse
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def find_bags(bags_root, pattern="hw_*"):
    p = os.path.expanduser(bags_root)
    glob_pattern = os.path.join(p, pattern)
    return sorted(glob.glob(glob_pattern))


def ensure_outdir(outdir):
    out = os.path.expanduser(outdir)
    os.makedirs(out, exist_ok=True)
    return out


def plot_trajectories(data_dict, outpath, title=None):
    # data_dict: {label: {'x':[], 'y':[], 't':[]}}
    fig, ax = plt.subplots(figsize=(10, 8))

    # compute global time zero (seconds from beginning) and global norm for coloring
    all_times = []
    for d in data_dict.values():
        all_times.extend(d['t'])
    if len(all_times) == 0:
        return
    t0 = min(all_times)
    times_all = np.asarray(all_times, dtype=float) - float(t0)
    cmap_name = 'jet'  
    global_norm = plt.Normalize(times_all.min(), times_all.max())

    # marker color mapping (start/end)
    start_color = {'/STAR/world': 'black', '/Lucas6/world': 'red'}

    for label, d in data_dict.items():
        x = np.asarray(d['x'])
        y = np.asarray(d['y'])
        t = np.asarray(d['t'])
        if len(x) < 2:
            continue

        # normalize times relative to beginning
        t = t - t0

        # create segments for LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # color by time (use midpoints of segment times), use global normalization and vivid colormap
        times = (t[:-1] + t[1:]) / 2.0
        lc = LineCollection(segments, cmap=cmap_name, norm=global_norm)
        lc.set_array(times)
        lc.set_linewidth(7.0)
        lc.set_zorder(1)
        lc.set_capstyle('round')
        lc.set_joinstyle('round')
        ax.add_collection(lc)

        # Mark start/end (start dot colored per mapping)
        sc = start_color.get(label, 'black')
        # increase marker sizes for visibility
        # initial marker and final marker styling:
        if label == '/STAR/world':
            # STAR: filled black initial dot
            ax.scatter(x[0], y[0], color='black', marker='o', s=250, zorder=10)
        elif label == '/Lucas6/world':
            # Lucas6: empty blue initial circle
            ax.scatter(x[0], y[0], facecolors='none', edgecolors='black', marker='o', s=250, zorder=10)
        else:
            ax.scatter(x[0], y[0], color=sc, marker='o', s=250, zorder=10)
    ax.set_ylabel('y', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.autoscale()
    ax.axis('equal')
    ax.grid(True)

    # Draw corridor lines around STAR location: +/- 4.0 feet in y
    feet_to_m = 0.3048
    offset = 4.0 * feet_to_m
    star_key = '/STAR/world'
    if star_key in data_dict and len(data_dict[star_key]['y']) > 0:
        # use STAR's start location (first recorded y) as corridor center
        star_y = float(data_dict[star_key]['y'][0])
        # ensure axes limits are up to date
        ax.relim()
        ax.autoscale_view()
        xmin, xmax = ax.get_xlim()
        y_plus = star_y + offset
        y_minus = star_y - offset
        ax.plot([xmin, xmax], [y_plus, y_plus], color='black', linewidth=3.5, zorder=0)
        ax.plot([xmin, xmax], [y_minus, y_minus], color='black', linewidth=3.5, zorder=0)

    # add colorbar using a ScalarMappable with the same colormap and global norm
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=global_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    # Set colorbar to show only 4 ticks
    cbar_ticks = np.linspace(global_norm.vmin, global_norm.vmax, 4)
    cbar.set_ticks(cbar_ticks)

    # Round colorbar ticks to nearest integer
    cbar.set_ticklabels([str(int(round(t))) for t in cbar_ticks])

    # Make colorbar tick labels and label larger
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('time (s)', fontsize=20)

    # Legend: STAR is Robot, Lucas6 is Person
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='black', label='Robot', markerfacecolor='black', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='black', label='Person', markerfacecolor='none', markeredgecolor='black', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='_', color='black', label='Corridor', linewidth=2.0)
    ]
    ax.legend(handles=legend_elems, loc='best', fontsize=14, markerscale=1.8)

    # Add a 1-meter scale bar at (x=-4, y=-4)
    bar_x = -4.0
    bar_y = -3.0
    ax.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], color='black', linewidth=2, solid_capstyle='butt', zorder=20)
    ax.plot([bar_x, bar_x], [bar_y - 0.1, bar_y + 0.1], color='black', linewidth=2, zorder=20)
    ax.plot([bar_x + 1.0, bar_x + 1.0], [bar_y - 0.1, bar_y + 0.1], color='black', linewidth=2, zorder=20)
    ax.text(bar_x + 0.5, bar_y - 0.3, '1 m', ha='center', va='top', fontsize=14, color='black')

    # Set grid to 1m x 1m
    ax.set_xticks(np.arange(-4, 6, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.grid(True, which='major', linewidth=1, alpha=0.3)

    # Remove axis numbers (tick labels)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def read_ros2_bag_topics(bag_path, topics_of_interest):
    """Attempt to read PoseStamped messages from a ROS2 bag.

    Returns a dict {topic: {'x':[], 'y':[], 't':[]}}
    This is best-effort and will raise informative errors if ROS2
    Python packages are not available.
    """
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except Exception as e:
        raise RuntimeError("rosbag2_py import failed: {}\nMake sure you run this script in a ROS2-enabled Python environment.".format(e))

    try:
        import rclpy
        from rclpy.serialization import deserialize_message
        from geometry_msgs.msg import PoseStamped
    except Exception as e:
        raise RuntimeError("rclpy/geometry_msgs import failed: {}\nRun in ROS2 Python environment (source /opt/ros/<distro>/setup.bash).".format(e))

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # get topic types
    try:
        topics_and_types = reader.get_all_topics_and_types()
    except Exception:
        topics_and_types = []

    topic_type_map = {t.name: t.type for t in topics_and_types}

    selected_types = {}
    for top in topics_of_interest:
        if top in topic_type_map:
            selected_types[top] = topic_type_map[top]

    # prepare output structure
    out = {top: {'x': [], 'y': [], 't': []} for top in topics_of_interest}

    # read messages sequentially
    # reader.read_next() may return different shapes depending on rosbag2_py version
    while reader.has_next():
        try:
            element = reader.read_next()
        except Exception:
            break

        # element may be a tuple or object
        topic = None
        serialized = None
        if isinstance(element, tuple) or isinstance(element, list):
            # common: (topic, serialized_data, timestamp)
            if len(element) >= 2:
                topic = element[0]
                serialized = element[1]
        else:
            # object with attributes
            topic = getattr(element, 'topic_name', None) or getattr(element, 'topic', None)
            serialized = getattr(element, 'serialized_data', None) or getattr(element, 'serialized_message', None) or getattr(element, 'message', None)

        if topic not in topics_of_interest:
            continue

        # obtain raw bytes
        raw = None
        if isinstance(serialized, (bytes, bytearray)):
            raw = bytes(serialized)
        else:
            # object may have 'data' attribute
            raw = getattr(serialized, 'data', None)
            if raw is None:
                # maybe the element itself has data
                raw = getattr(element, 'data', None)

        if raw is None:
            # cannot deserialize; skip
            continue

        try:
            msg = deserialize_message(raw, PoseStamped)
        except Exception:
            # if deserialize_message fails, skip
            continue

        # extract position and time
        try:
            hx = msg.pose.position.x
            hy = msg.pose.position.y
            hs = msg.header.stamp
            t = float(hs.sec) + float(hs.nanosec) * 1e-9
        except Exception:
            # fallback names
            try:
                hx = msg.pose.position.x
                hy = msg.pose.position.y
                hs = msg.header.stamp
                t = float(hs.sec) + float(hs.nanosec) * 1e-9
            except Exception:
                continue

        out[topic]['x'].append(hx)
        out[topic]['y'].append(hy)
        out[topic]['t'].append(t)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags-root', default='~/livox_dataset/bags', help='Root folder containing hw_* bag directories')
    parser.add_argument('--pattern', default='hw_*', help='Glob pattern for bag folders')
    parser.add_argument('--outdir', default='~/livox_dataset/plots_hw_mighty/', help='Where to save plots')
    parser.add_argument('--topics', nargs='+', default=['/Lucas6/world', '/STAR/world'], help='Topics to plot')
    args = parser.parse_args()

    bag_dirs = find_bags(args.bags_root, args.pattern)
    if not bag_dirs:
        print('No bag directories found in', args.bags_root)
        sys.exit(1)

    outdir = ensure_outdir(args.outdir)

    for bag in bag_dirs:
        print('Processing', bag)
        try:
            data = read_ros2_bag_topics(bag, args.topics)
        except RuntimeError as e:
            print('Failed to read bag {}: {}'.format(bag, e))
            continue

        # If no data at all, skip
        if all(len(v['x']) == 0 for v in data.values()):
            print('No matching messages found in', bag)
            continue

        # merge into single figure
        bn = os.path.basename(bag.rstrip('/'))
        outpath = os.path.join(outdir, f"{bn}_xy.png")
        plot_trajectories(data, outpath, title=bn)
        print('Saved', outpath)


if __name__ == '__main__':
    main()
