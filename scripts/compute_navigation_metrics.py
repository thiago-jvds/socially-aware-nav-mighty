#!/usr/bin/env python3
"""
Compute navigation metrics from ROS2 bag files and generate per-metric comparison plots.

This script scans bags named like:
  sim_{method}_{scenario}_0{x}
where method in ["socially_aware", "standard"]
      scenario in ["crossing", "overtaking", "passing", "mixed"]
      x in [1..5]

Outputs three PNGs in the output directory:
  - closest_distance_comparison.png
  - success_rate_comparison.png
  - psi_comparison.png

Plots show values grouped by scenario, with blue = socially_aware and red = standard.

Run (headless):
  python3 src/mighty/scripts/compute_navigation_metrics.py --bags-root ~/livox_dataset/bags --outdir ~/livox_dataset/plots_sim_mighty/
"""

import argparse
import csv
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SCENARIOS = ['crossing', 'overtaking', 'passing', 'mixed']
METHODS = ['socially_aware', 'standard']


########################
# Bag reading helpers
########################

def _canonical_topic_name(topic_name: str) -> str:
    return '/' + topic_name.lstrip('/')


def _coerce_serialized_bytes(serialized) -> Optional[bytes]:
    if serialized is None:
        return None
    if isinstance(serialized, (bytes, bytearray, memoryview)):
        return bytes(serialized)
    # try common attributes
    for attr in ('serialized_data', 'serialized_message', 'data', 'buffer'):
        val = getattr(serialized, attr, None)
        if val is None:
            continue
        try:
            return bytes(val)
        except Exception:
            try:
                return bytes(memoryview(val))
            except Exception:
                continue
    try:
        return bytes(serialized)
    except Exception:
        return None


def _extract_message_timestamp(msg, fallback_timestamp=None) -> float:
    try:
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        pass
    if fallback_timestamp is None:
        return 0.0
    try:
        ts = float(fallback_timestamp)
    except Exception:
        return 0.0
    # if it's very large assume nanoseconds
    if ts > 1e12:
        ts *= 1e-9
    return ts


def _resolve_message_type(msg_type_str: str):
    # Prefer direct generated module import for local interfaces
    try:
        pkg, _, typ = msg_type_str.partition('/msg/')
        if pkg and typ:
            mod = __import__(f"{pkg}.msg", fromlist=[typ])
            return getattr(mod, typ)
    except Exception:
        pass
    # fallback to rosidl utility if available at runtime
    try:
        from rosidl_runtime_py.utilities import get_message
        return get_message(msg_type_str)
    except Exception as e:
        raise RuntimeError(f"Unable to resolve message type {msg_type_str}: {e}")


def read_ros2_bag_topics(bag_path: str, topics_of_interest: List[str]) -> Dict[str, List[Tuple[float, object]]]:
    """Read selected topics from a rosbag2 sqlite folder.

    Returns dict topic -> list of (timestamp_seconds, message)
    """
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except Exception as e:
        raise RuntimeError(f"rosbag2_py import failed: {e}")
    try:
        from rclpy.serialization import deserialize_message
    except Exception as e:
        raise RuntimeError(f"rclpy import failed: {e}")

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    try:
        topics_and_types = reader.get_all_topics_and_types()
    except Exception as e:
        logger.error('Failed to list topics in bag %s: %s', bag_path, e)
        return {t: [] for t in topics_of_interest}

    topic_type_map = {t.name: t.type for t in topics_and_types}
    logger.debug('Topics in bag %s: %s', bag_path, ','.join(topic_type_map.keys()))

    out = {t: [] for t in topics_of_interest}

    while reader.has_next():
        try:
            elem = reader.read_next()
        except Exception:
            break
        topic = None
        raw_obj = None
        ts = None
        if isinstance(elem, (tuple, list)):
            if len(elem) >= 2:
                topic = elem[0]
                raw_obj = elem[1]
            if len(elem) >= 3:
                ts = elem[2]
        else:
            topic = getattr(elem, 'topic_name', None) or getattr(elem, 'topic', None)
            raw_obj = (getattr(elem, 'serialized_data', None)
                       or getattr(elem, 'serialized_message', None)
                       or getattr(elem, 'message', None)
                       or elem)
            ts = getattr(elem, 'timestamp', None)
        if not topic:
            continue
        topic = _canonical_topic_name(topic)
        if topic not in out:
            continue
        raw = _coerce_serialized_bytes(raw_obj)
        if raw is None:
            logger.debug('No raw bytes for topic %s', topic)
            continue
        msg_type_str = topic_type_map.get(topic)
        if not msg_type_str:
            logger.debug('Unknown type for topic %s', topic)
            continue
        try:
            msg_type = _resolve_message_type(msg_type_str)
            msg = deserialize_message(raw, msg_type)
            timestamp = _extract_message_timestamp(msg, fallback_timestamp=ts)
            out[topic].append((timestamp, msg))
        except Exception as e:
            logger.debug('Failed to deserialize %s: %s', topic, e)
            continue

    return out


########################
# Metrics computation
########################

def extract_robot_position_and_time(state_messages: List[Tuple[float, object]]) -> Tuple[np.ndarray, np.ndarray]:
    ts = []
    pos = []
    for t, m in state_messages:
        try:
            x = float(m.pos.x)
            y = float(m.pos.y)
            ts.append(float(t))
            pos.append([x, y])
        except Exception:
            continue
    return np.array(ts), np.array(pos)


def extract_pedestrian_positions(detection_messages: List[Tuple[float, object]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    ts = []
    dets = []
    for t, m in detection_messages:
        try:
            arr = []
            if hasattr(m, 'detections'):
                for d in m.detections:
                    try:
                        c = d.bbox.center.position
                        arr.append([float(c.x), float(c.y)])
                    except Exception:
                        continue
            if arr:
                ts.append(float(t))
                dets.append(np.array(arr))
        except Exception:
            continue
    return np.array(ts), dets


def synchronize_data(state_ts: np.ndarray, state_pos: np.ndarray, det_ts: np.ndarray, det_pos: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    if len(state_ts) == 0 or len(det_ts) == 0:
        return np.array([]), np.array([]).reshape(0,2), []
    synced_times = []
    synced_robot = []
    synced_peds = []
    for i, st in enumerate(state_ts):
        idx = np.argmin(np.abs(det_ts - st))
        if abs(det_ts[idx] - st) < 0.05:
            synced_times.append(st)
            synced_robot.append(state_pos[i])
            synced_peds.append(det_pos[idx])
    if not synced_times:
        return np.array([]), np.array([]).reshape(0,2), []
    return np.array(synced_times), np.array(synced_robot), synced_peds


def compute_closest_pedestrian_distance(robot_positions: np.ndarray, pedestrian_positions: List[np.ndarray]) -> float:
    if robot_positions.size == 0 or len(pedestrian_positions) == 0:
        return float('inf')
    min_d = float('inf')
    for i, r in enumerate(robot_positions):
        if i >= len(pedestrian_positions):
            break
        p = pedestrian_positions[i]
        if p.size == 0:
            continue
        dists = np.linalg.norm(p - r[np.newaxis,:], axis=1)
        min_d = min(min_d, float(np.min(dists)))
    return min_d


def compute_collision(flag_robot: np.ndarray, pedestrian_positions: List[np.ndarray], threshold: float=0.2) -> int:
    if flag_robot.size == 0 or len(pedestrian_positions) == 0:
        return 0
    collisions = 0
    for i, r in enumerate(flag_robot):
        if i >= len(pedestrian_positions):
            break
        p = pedestrian_positions[i]
        if p.size == 0:
            continue
        if np.any(np.linalg.norm(p - r[np.newaxis,:], axis=1) < threshold):
            collisions += 1
    return collisions


def compute_psi(timestamps: np.ndarray, robot_positions: np.ndarray, pedestrian_positions: List[np.ndarray], dt: float=0.1) -> float:
    # Discretize time at dt from trial start and take max weight across pedestrians
    if timestamps.size == 0 or robot_positions.size == 0 or len(pedestrian_positions) == 0:
        return 0.0

    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    if t1 <= t0:
        return 0.0

    grid = np.arange(t0, t1 + 1e-9, dt)
    psi = 0.0
    half_dt = dt * 0.5

    for tg in grid:
        idx = int(np.argmin(np.abs(timestamps - tg)))
        if abs(timestamps[idx] - tg) > half_dt:
            continue
        if idx >= len(pedestrian_positions):
            continue
        r = robot_positions[idx]
        p = pedestrian_positions[idx]
        if p is None or len(p) == 0:
            continue
        dists = np.linalg.norm(p - r[np.newaxis, :], axis=1)
        
        # Distance-based weights: constant * (distance - bracket_min)
        weights = np.zeros_like(dists, dtype=float)
        mask_0 = dists < 0.45
        mask_1 = (dists >= 0.45) & (dists < 0.8)
        mask_2 = (dists >= 0.8) & (dists < 1.2)
        mask_3 = (dists >= 1.2) & (dists < 2.0)
        
        # Intimate Zone: Lethal/Max Penalty
        weights[mask_0] = 64.0
        
        # Inner Personal Zone: Decays from 10.0 down to 5.0
        # Formula: Max_Weight - Weight_Drop * (Current_Dist - Min_Dist) / (Max_Dist - Min_Dist)
        weights[mask_1] = 64.0 - 16.0 * ((dists[mask_1] - 0.45) / (0.8 - 0.45))
        
        # Outer Personal Zone: Decays from 5.0 down to 2.5
        weights[mask_2] = 16.0 - 4.0 * ((dists[mask_2] - 0.8) / (1.2 - 0.8))
        
        # Social Zone: Decays from 2.5 down to 0.0
        weights[mask_3] = 4.0 - 4.0 * ((dists[mask_3] - 1.2) / (2.0 - 1.2))
        w = float(np.max(weights)) if len(weights) > 0 else 0.0
        psi += float(w) * float(dt)

    return float(psi)


########################
# High-level processing
########################

def process_bag(bag_dir: str) -> Optional[Dict]:
    logger.info('Processing bag %s', bag_dir)
    topics = ['/NX01/state', '/NX01/detected_objects']
    try:
        msgs = read_ros2_bag_topics(bag_dir, topics)
    except Exception as e:
        logger.error('Failed reading bag %s: %s', bag_dir, e)
        return None
    state_msgs = msgs.get('/NX01/state', [])
    det_msgs = msgs.get('/NX01/detected_objects', [])
    logger.info('  counts: state=%d, detections=%d', len(state_msgs), len(det_msgs))
    if len(state_msgs) == 0 or len(det_msgs) == 0:
        logger.warning('Missing required topics in %s', bag_dir)
        return None
    state_ts, state_pos = extract_robot_position_and_time(state_msgs)
    det_ts, det_pos = extract_pedestrian_positions(det_msgs)
    sync_ts, sync_robot_pos, sync_ped_pos = synchronize_data(state_ts, state_pos, det_ts, det_pos)
    if sync_ts.size == 0:
        logger.warning('No synchronized points in %s', bag_dir)
        return None
    closest = compute_closest_pedestrian_distance(sync_robot_pos, sync_ped_pos)
    collisions = compute_collision(sync_robot_pos, sync_ped_pos, threshold=0.6)
    has_collision = collisions > 0
    psi = compute_psi(sync_ts, sync_robot_pos, sync_ped_pos)
    return {'closest_distance': closest, 'has_collision': has_collision, 'psi_score': psi}


def compute_method_metrics(bag_dirs: List[str]) -> Optional[Dict]:
    trials = []
    for b in bag_dirs:
        m = process_bag(b)
        if m:
            trials.append(m)
    if not trials:
        return None
    closest = [t['closest_distance'] for t in trials]
    psi = [t['psi_score'] for t in trials]
    collisions = [1.0 if t['has_collision'] else 0.0 for t in trials]
    # filter inf
    valid_closest = [c for c in closest if np.isfinite(c)]
    return {
        'closest_distance_mean': float(np.mean(valid_closest)) if valid_closest else 0.0,
        'closest_distance_std': float(np.std(valid_closest)) if valid_closest else 0.0,
        'success_rate': float(1.0 - np.mean(collisions)),
        'success_rate_std': float(np.std(1.0 - np.array(collisions))),
        'psi_mean': float(np.mean(psi)),
        'psi_std': float(np.std(psi)),
        'num_trials': len(trials),
        'per_trial': trials
    }


def find_bags_root(bags_root: str) -> Dict[str, Dict[str, List[str]]]:
    root = Path(bags_root).expanduser()
    if not root.exists():
        logger.error('Bags root %s does not exist', root)
        return {}
    results = {}
    for scen in SCENARIOS:
        results[scen] = {}
        for m in METHODS:
            pattern = str(root / f'sim_{m}_{scen}_0[1-5]')
            found = sorted(glob.glob(pattern))
            results[scen][m] = found
            logger.info('Found %d bags for %s/%s', len(found), scen, m)
    return results


def create_metric_plots(results_by_scenario: Dict[str, Dict[str, Dict]], outdir: Path) -> None:
    scenarios = SCENARIOS
    methods = METHODS
    # Softer, colorblind-friendly palette (muted blue / muted red)
    colors = {'socially_aware': '#4E79A7', 'standard': '#E15759'}

    closest_means = {m: [] for m in methods}
    closest_stds = {m: [] for m in methods}
    success_means = {m: [] for m in methods}
    success_stds = {m: [] for m in methods}
    psi_means = {m: [] for m in methods}
    psi_stds = {m: [] for m in methods}

    for scen in scenarios:
        for m in methods:
            metrics = results_by_scenario.get(scen, {}).get(m)
            if metrics:
                closest_means[m].append(metrics['closest_distance_mean'])
                closest_stds[m].append(metrics['closest_distance_std'])
                success_means[m].append(metrics['success_rate'] * 100.0)
                success_stds[m].append(metrics['success_rate_std'] * 100.0)
                psi_means[m].append(metrics['psi_mean'])
                psi_stds[m].append(metrics['psi_std'])
            else:
                closest_means[m].append(0.0)
                closest_stds[m].append(0.0)
                success_means[m].append(0.0)
                success_stds[m].append(0.0)
                psi_means[m].append(0.0)
                psi_stds[m].append(0.0)

    x = np.arange(len(scenarios))
    width = 0.35

    def save_bar_plot(filename: str, title: str, ylabel: str, means: Dict, stds: Dict, percent: bool=False, add_arrow: bool=False):
        # Use a clean seaborn-like whitegrid but we'll tone down the grid lines
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10,5))
        # Draw bars (no per-axis legend) and slightly bolder visual weight
        ax.bar(x - width/2, means['socially_aware'], width, yerr=stds['socially_aware'],
            color=colors['socially_aware'], capsize=6)
        ax.bar(x + width/2, means['standard'], width, yerr=stds['standard'],
            color=colors['standard'], capsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        # For PSI we optionally remove the y-label and add an upward arrow showing
        # increased discomfort.
        if not add_arrow:
            ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Make grid subtle and behind the data
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E0E0E0', linestyle=':', linewidth=1, alpha=0.6)

        # De-spine for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if percent:
            ax.set_ylim(0, 105)

        # If requested, draw a vertical arrow (axes-fraction coords) with label
        if add_arrow:
            ax.annotate('', xy=(0.95, 0.9), xytext=(0.95, 0.35), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color='black', lw=2), annotation_clip=False)
            ax.text(0.955, 0.92, 'Increased discomfort', transform=ax.transAxes, va='bottom', ha='left', fontsize=11)

        # Create a single, horizontal legend for the figure (top-center)
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors['socially_aware']),
                          plt.Rectangle((0, 0), 1, 1, color=colors['standard'])]
        fig.legend(legend_handles, ['Socially aware', 'Standard'],
                   loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03))

        outpath = outdir / filename
        fig.tight_layout()
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('Saved plot: %s', outpath)

    save_bar_plot('closest_distance_comparison.png', 'Closest pedestrian distance by scenario', 'Distance (m)', closest_means, closest_stds)
    save_bar_plot('success_rate_comparison.png', 'Success rate (no collision) by scenario', 'Success rate (%)', success_means, success_stds, percent=True)
    save_bar_plot('psi_comparison.png', 'Personal Space Intrusion (PSI) by scenario', 'PSI score', psi_means, psi_stds, add_arrow=True)


def save_metrics_csv(results_by_scenario: Dict[str, Dict[str, Dict]], outdir: Path) -> Path:
    outpath = outdir / 'navigation_metrics_statistics.csv'
    fieldnames = [
        'scenario',
        'method',
        'num_trials',
        'closest_distance_mean_m',
        'closest_distance_std_m',
        'success_rate',
        'success_rate_percent',
        'success_rate_std',
        'success_rate_std_percent',
        'psi_mean',
        'psi_std',
    ]

    with outpath.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scen in SCENARIOS:
            for method in METHODS:
                metrics = results_by_scenario.get(scen, {}).get(method)
                if not metrics:
                    continue
                writer.writerow({
                    'scenario': scen,
                    'method': method,
                    'num_trials': metrics.get('num_trials', 0),
                    'closest_distance_mean_m': metrics.get('closest_distance_mean', 0.0),
                    'closest_distance_std_m': metrics.get('closest_distance_std', 0.0),
                    'success_rate': metrics.get('success_rate', 0.0),
                    'success_rate_percent': metrics.get('success_rate', 0.0) * 100.0,
                    'success_rate_std': metrics.get('success_rate_std', 0.0),
                    'success_rate_std_percent': metrics.get('success_rate_std', 0.0) * 100.0,
                    'psi_mean': metrics.get('psi_mean', 0.0),
                    'psi_std': metrics.get('psi_std', 0.0),
                })

    logger.info('Saved CSV: %s', outpath)
    return outpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags-root', default='~/livox_dataset/bags/', help='Root folder with bags')
    parser.add_argument('--outdir', default='~/livox_dataset/plots_sim_mighty/', help='Output directory')
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # find bags
    bag_index = find_bags_root(args.bags_root)

    # process
    results_by_scenario = {}
    for scen, methods_map in bag_index.items():
        results_by_scenario[scen] = {}
        for m, bags in methods_map.items():
            if not bags:
                logger.warning('No bags for %s/%s', scen, m)
                continue
            logger.info('Processing method=%s scenario=%s: %d bags', m, scen, len(bags))
            metrics = compute_method_metrics(bags)
            if metrics:
                results_by_scenario[scen][m] = metrics
                logger.info(' %s/%s: closest=%.3f psi=%.2f success=%.1f%%', scen, m, metrics['closest_distance_mean'], metrics['psi_mean'], metrics['success_rate']*100.0)
            else:
                logger.warning('Failed to compute metrics for %s/%s', scen, m)

    save_metrics_csv(results_by_scenario, outdir)

    # create per-metric plots across scenarios
    create_metric_plots(results_by_scenario, outdir)
    logger.info('All done. Plots in %s', outdir)


if __name__ == '__main__':
    main()
