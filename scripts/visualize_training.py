"""
Training Visualization Module for MIGHTY Evasion Policy

Provides real-time and saved visualizations of ego/adversary trajectories
during neural network training.

Usage:
    viz = TrainingVisualizer(save_dir='training_viz', live=False, plot_every=10)
    viz.log_episode(episode, loss, ego_traj, adv_traj, ego_start, ego_goal, adv_start)
    viz.plot_training_curves()
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
from typing import Optional, Tuple
import json

# Import Axes3D (registers 3D projection)
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    # Fallback if mpl_toolkits not available
    pass


class TrainingVisualizer:
    """Visualize ego and adversary trajectories during training."""

    def __init__(self, save_dir='training_viz', live=False, plot_every=10):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
            live: If True, show live matplotlib window (requires display)
            plot_every: Save/show visualization every N episodes
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.live = live
        self.plot_every = plot_every

        # Training history
        self.history = {
            'episodes': [],
            'losses': [],
            'min_distances': [],
            'goal_distances': [],
            'converged': [],
            'theta_values': [],  # List of dicts
        }

        # Live plot setup
        if self.live:
            # Switch to interactive backend for live mode
            matplotlib.use('TkAgg')
            plt.ion()
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig = None
            self.ax = None

    def log_episode(self, episode: int, loss: float,
                   ego_trajectory: np.ndarray,
                   adversary_trajectory: np.ndarray,
                   ego_start: np.ndarray,
                   ego_goal: np.ndarray,
                   adv_start: np.ndarray,
                   converged: bool = True,
                   theta: Optional[dict] = None,
                   obstacles: Optional[list] = None):
        """
        Log an episode and optionally visualize.

        Args:
            episode: Episode number
            loss: Task loss value
            ego_trajectory: (T, 3) ego positions
            adversary_trajectory: (T, 3) adversary positions
            ego_start: (3,) start position
            ego_goal: (3,) goal position
            adv_start: (3,) adversary start
            converged: Whether optimizer converged
            theta: Dict of cost weights from policy network
            obstacles: List of obstacle boxes (not used yet)
        """
        # Compute metrics
        min_dist = self._compute_min_distance(ego_trajectory, adversary_trajectory)
        goal_dist = np.linalg.norm(ego_trajectory[-1] - ego_goal)

        # Store in history
        self.history['episodes'].append(episode)
        self.history['losses'].append(float(loss))
        self.history['min_distances'].append(float(min_dist))
        self.history['goal_distances'].append(float(goal_dist))
        self.history['converged'].append(converged)
        if theta is not None:
            self.history['theta_values'].append({k: float(v) for k, v in theta.items()})

        # Visualize if it's time
        if episode % self.plot_every == 0:
            if self.live:
                self._plot_live(episode, loss, ego_trajectory, adversary_trajectory,
                              ego_start, ego_goal, adv_start, min_dist, goal_dist)
            else:
                self._save_plot(episode, loss, ego_trajectory, adversary_trajectory,
                              ego_start, ego_goal, adv_start, min_dist, goal_dist)

        # Save training curves periodically
        if episode % 50 == 0 and episode > 0:
            self.plot_training_curves()
            self._save_history()

    def _compute_min_distance(self, ego_traj: np.ndarray, adv_traj: np.ndarray) -> float:
        """Compute minimum distance between trajectories."""
        # Align trajectories if different lengths
        T_ego = ego_traj.shape[0]
        T_adv = adv_traj.shape[0]

        if T_ego != T_adv:
            # Resample to match
            t_ego = np.linspace(0, 1, T_ego)
            t_adv = np.linspace(0, 1, T_adv)
            adv_aligned = np.array([
                np.interp(t_ego, t_adv, adv_traj[:, dim])
                for dim in range(3)
            ]).T
        else:
            adv_aligned = adv_traj

        # Compute distances at each time step
        dists = np.linalg.norm(ego_traj - adv_aligned, axis=1)
        return float(np.min(dists))

    def _plot_live(self, episode: int, loss: float,
                  ego_traj: np.ndarray, adv_traj: np.ndarray,
                  ego_start: np.ndarray, ego_goal: np.ndarray,
                  adv_start: np.ndarray,
                  min_dist: float, goal_dist: float):
        """Update live matplotlib window."""
        self.ax.clear()
        self._plot_trajectories(self.ax, episode, loss, ego_traj, adv_traj,
                               ego_start, ego_goal, adv_start, min_dist, goal_dist)
        plt.draw()
        plt.pause(0.5)  # Pause briefly to see the plot

    def _save_plot(self, episode: int, loss: float,
                  ego_traj: np.ndarray, adv_traj: np.ndarray,
                  ego_start: np.ndarray, ego_goal: np.ndarray,
                  adv_start: np.ndarray,
                  min_dist: float, goal_dist: float):
        """Save plot to PNG file."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        self._plot_trajectories(ax, episode, loss, ego_traj, adv_traj,
                               ego_start, ego_goal, adv_start, min_dist, goal_dist)

        # Save
        filepath = self.save_dir / f"episode_{episode:04d}.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  💾 Saved visualization: {filepath}")

    def _plot_trajectories(self, ax, episode: int, loss: float,
                          ego_traj: np.ndarray, adv_traj: np.ndarray,
                          ego_start: np.ndarray, ego_goal: np.ndarray,
                          adv_start: np.ndarray,
                          min_dist: float, goal_dist: float):
        """Core plotting function."""
        # Ego trajectory (blue)
        ax.plot(ego_traj[:, 0], ego_traj[:, 1], ego_traj[:, 2],
               'b-', linewidth=2, label='Ego trajectory')

        # Add arrow to show ego direction
        if len(ego_traj) > 1:
            mid_idx = len(ego_traj) // 2
            ego_dir = ego_traj[mid_idx] - ego_traj[mid_idx-1]
            ax.quiver(ego_traj[mid_idx, 0], ego_traj[mid_idx, 1], ego_traj[mid_idx, 2],
                     ego_dir[0], ego_dir[1], ego_dir[2],
                     color='blue', arrow_length_ratio=0.3, linewidth=2)

        # Adversary trajectory (red)
        ax.plot(adv_traj[:, 0], adv_traj[:, 1], adv_traj[:, 2],
               'r-', linewidth=2, label='Adversary trajectory')

        # Add arrow to show adversary direction
        if len(adv_traj) > 1:
            mid_idx = len(adv_traj) // 2
            adv_dir = adv_traj[mid_idx] - adv_traj[mid_idx-1]
            ax.quiver(adv_traj[mid_idx, 0], adv_traj[mid_idx, 1], adv_traj[mid_idx, 2],
                     adv_dir[0], adv_dir[1], adv_dir[2],
                     color='red', arrow_length_ratio=0.3, linewidth=2)

        # Start positions
        ax.scatter(*ego_start, c='blue', marker='o', s=100,
                  label='Ego start', edgecolors='black', linewidths=2)
        ax.scatter(*adv_start, c='red', marker='o', s=100,
                  label='Adversary start', edgecolors='black', linewidths=2)

        # Goal position (green star)
        ax.scatter(*ego_goal, c='green', marker='*', s=300,
                  label='Goal', edgecolors='black', linewidths=2)

        # Find closest points and draw line
        dists = np.linalg.norm(ego_traj[:, None, :] - adv_traj[None, :, :], axis=2)
        min_idx_ego, min_idx_adv = np.unravel_index(np.argmin(dists), dists.shape)
        closest_ego = ego_traj[min_idx_ego]
        closest_adv = adv_traj[min_idx_adv]

        ax.plot([closest_ego[0], closest_adv[0]],
               [closest_ego[1], closest_adv[1]],
               [closest_ego[2], closest_adv[2]],
               'k--', linewidth=1, alpha=0.5, label=f'Min dist: {min_dist:.2f}m')

        # Labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)

        ax.set_title(f'Episode {episode} | Loss: {loss:.3f} | '
                    f'Min dist: {min_dist:.2f}m | Goal dist: {goal_dist:.2f}m',
                    fontsize=12, fontweight='bold')

        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        all_points = np.vstack([ego_traj, adv_traj, ego_start, ego_goal, adv_start])
        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

    def plot_training_curves(self):
        """Plot training curves (loss, distances) over episodes."""
        if len(self.history['episodes']) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        episodes = self.history['episodes']

        # Loss curve
        axes[0, 0].plot(episodes, self.history['losses'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Min distance curve
        axes[0, 1].plot(episodes, self.history['min_distances'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Safety margin (1m)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Min Distance (m)')
        axes[0, 1].set_title('Minimum Ego-Adversary Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Goal distance curve
        axes[1, 0].plot(episodes, self.history['goal_distances'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Goal Distance (m)')
        axes[1, 0].set_title('Final Distance to Goal')
        axes[1, 0].grid(True, alpha=0.3)

        # Convergence rate
        converged_rate = []
        window = 10
        for i in range(len(self.history['converged'])):
            start = max(0, i - window + 1)
            rate = np.mean(self.history['converged'][start:i+1]) * 100
            converged_rate.append(rate)

        axes[1, 1].plot(episodes, converged_rate, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Convergence Rate (%)')
        axes[1, 1].set_title(f'Optimizer Convergence Rate (window={window})')
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        filepath = self.save_dir / "training_curves.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Saved training curves: {filepath}")

    def _save_history(self):
        """Save training history to JSON."""
        filepath = self.save_dir / "training_history.json"
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def print_summary(self, episode: int, window: int = 10):
        """Print training summary for recent episodes."""
        if len(self.history['episodes']) < 1:
            return

        idx = self.history['episodes'].index(episode)
        start_idx = max(0, idx - window + 1)

        recent_losses = self.history['losses'][start_idx:idx+1]
        recent_min_dists = self.history['min_distances'][start_idx:idx+1]
        recent_goal_dists = self.history['goal_distances'][start_idx:idx+1]
        recent_converged = self.history['converged'][start_idx:idx+1]

        loss = self.history['losses'][idx]
        min_dist = self.history['min_distances'][idx]
        goal_dist = self.history['goal_distances'][idx]
        converged = self.history['converged'][idx]

        print(f"Episode {episode:4d} | "
              f"Loss: {loss:7.3f} (avg: {np.mean(recent_losses):7.3f}) | "
              f"Min dist: {min_dist:5.2f}m (avg: {np.mean(recent_min_dists):5.2f}m) | "
              f"Goal dist: {goal_dist:5.2f}m (avg: {np.mean(recent_goal_dists):5.2f}m) | "
              f"Conv: {converged}")

    def close(self):
        """Close live plot if open."""
        if self.live and self.fig is not None:
            plt.close(self.fig)
            plt.ioff()


# Example usage
if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing TrainingVisualizer...")

    viz = TrainingVisualizer(save_dir='test_viz', live=False, plot_every=1)

    # Generate dummy trajectories
    for episode in range(5):
        t = np.linspace(0, 1, 50)

        # Ego: curved path
        ego_traj = np.column_stack([
            t * 5,
            np.sin(t * np.pi) * 2,
            t * 3
        ])

        # Adversary: straight line
        adv_traj = np.column_stack([
            t * 3 + 1,
            t * 4,
            t * 2.5
        ])

        ego_start = ego_traj[0]
        ego_goal = np.array([5.0, 0.0, 3.0])
        adv_start = adv_traj[0]

        loss = -2.0 + episode * 0.5

        viz.log_episode(episode, loss, ego_traj, adv_traj,
                       ego_start, ego_goal, adv_start, converged=True)
        viz.print_summary(episode)

    viz.plot_training_curves()
    print("\n✓ Test complete! Check test_viz/ directory for outputs.")
