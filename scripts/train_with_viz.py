"""
Training Loop with Visualization for MIGHTY Evasion Policy

Integrated training pipeline with:
- Neural network policy training
- Simple adversary simulation (Level 1: straight line, Level 2: proportional nav)
- Real-time or saved trajectory visualization
- Comprehensive metrics logging
- Checkpoint saving
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import optax
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple
import pickle

from policy_network import EvasionPolicy, initialize_policy
from mighty_jax import sample_trajectory_jax, reconstruct_jax
from mighty_solver_jax import solve_mighty_jax, solve_and_extract_trajectory
from visualize_training import TrainingVisualizer
from environments import (
    sample_random_scenario,
    create_initial_guess_from_env,
    get_observation_from_scenario
)


# ============================================================================
# Adversary Simulation
# ============================================================================

def simulate_adversary_level1(adv_start: jnp.ndarray,
                               adv_vel: jnp.ndarray,
                               ego_times: jnp.ndarray) -> jnp.ndarray:
    """
    Level 1 adversary: Constant velocity (straight line).

    Args:
        adv_start: (3,) adversary start position
        adv_vel: (3,) adversary velocity vector
        ego_times: (T,) time points for adversary trajectory

    Returns:
        adv_trajectory: (T, 3) adversary positions
    """
    # Simple constant velocity motion
    adv_trajectory = adv_start[None, :] + adv_vel[None, :] * ego_times[:, None]
    return adv_trajectory


def simulate_adversary_level2(adv_start: jnp.ndarray,
                               adv_vel_init: jnp.ndarray,
                               ego_trajectory: jnp.ndarray,
                               ego_times: jnp.ndarray,
                               pursuit_gain: float = 2.0) -> jnp.ndarray:
    """
    Level 2 adversary: Proportional navigation toward ego.

    Args:
        adv_start: (3,) adversary start position
        adv_vel_init: (3,) initial adversary velocity
        ego_trajectory: (T, 3) ego positions (for pursuit)
        ego_times: (T,) time points
        pursuit_gain: Pursuit gain (higher = more aggressive)

    Returns:
        adv_trajectory: (T, 3) adversary positions
    """
    T = len(ego_times)
    adv_trajectory = []
    adv_pos = adv_start
    adv_vel = adv_vel_init

    speed = jnp.linalg.norm(adv_vel_init)  # Maintain constant speed

    for i in range(T):
        adv_trajectory.append(adv_pos)

        if i < T - 1:
            # Proportional navigation: steer toward ego's current position
            dt = ego_times[i+1] - ego_times[i]
            to_ego = ego_trajectory[i] - adv_pos
            distance = jnp.linalg.norm(to_ego)

            # Desired velocity direction (toward ego)
            if distance > 1e-6:
                desired_dir = to_ego / distance
            else:
                desired_dir = adv_vel / (jnp.linalg.norm(adv_vel) + 1e-6)

            # Proportional navigation (blend current velocity with desired)
            adv_vel = adv_vel + pursuit_gain * dt * (desired_dir * speed - adv_vel)
            adv_vel = adv_vel / (jnp.linalg.norm(adv_vel) + 1e-6) * speed  # Normalize to constant speed

            # Update position
            adv_pos = adv_pos + adv_vel * dt

    return jnp.array(adv_trajectory)


# ============================================================================
# Scenario Generation
# ============================================================================

def sample_simple_scenario(rng: jax.random.PRNGKey,
                          bounds: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[Dict, jax.random.PRNGKey]:
    """
    Sample a simple 2-segment scenario (start -> midpoint -> goal).

    Args:
        rng: JAX random key
        bounds: (min, max) position bounds

    Returns:
        env_params: Environment parameters dict
        rng: Updated random key
    """
    rng, *subkeys = jax.random.split(rng, 6)  # Need 5 subkeys + 1 for return

    # Sample start and goal (keep apart by at least 5m)
    x0 = jax.random.uniform(subkeys[0], (3,), minval=bounds[0], maxval=bounds[1])
    xf = jax.random.uniform(subkeys[1], (3,), minval=bounds[0], maxval=bounds[1])

    # Ensure minimum separation
    while jnp.linalg.norm(xf - x0) < 5.0:
        xf = jax.random.uniform(subkeys[1], (3,), minval=bounds[0], maxval=bounds[1])

    # Midpoint waypoint (slightly off the straight line)
    mid = (x0 + xf) / 2.0
    perturbation = jax.random.uniform(subkeys[2], (3,), minval=-1.0, maxval=1.0)
    mid = mid + perturbation

    # Initial velocities (zero)
    v0 = jnp.zeros(3)
    vf = jnp.zeros(3)

    # Adversary: sample position and velocity
    adv_pos = jax.random.uniform(subkeys[3], (3,), minval=bounds[0], maxval=bounds[1])

    # Adversary velocity: toward goal with some speed
    to_goal = xf - adv_pos
    distance_to_goal = jnp.linalg.norm(to_goal)
    if distance_to_goal > 1e-6:
        adv_direction = to_goal / distance_to_goal
    else:
        adv_direction = jnp.array([1.0, 0.0, 0.0])

    adv_speed = jax.random.uniform(subkeys[4], minval=1.0, maxval=3.0)
    adv_vel = adv_direction * adv_speed

    # Build env_params
    env_params = {
        'x0': x0,
        'v0': v0,
        'xf': xf,
        'vf': vf,
        'waypoints': jnp.array([mid]),  # Single waypoint
        'N_seg': 2,
        'adversary_pos': adv_pos,
        'adversary_vel': adv_vel,
    }

    return env_params, rng


def create_simple_initial_guess(env_params: Dict) -> jnp.ndarray:
    """
    Create initial guess for 2-segment trajectory.

    Decision vector z = [CP_free (2 x 3 = 6), T (2 segments = 2)] = 8 elements

    Args:
        env_params: Environment parameters

    Returns:
        z_init: (8,) initial decision vector
    """
    x0 = env_params['x0']
    xf = env_params['xf']
    waypoints = env_params['waypoints']

    # Control points: [x0, waypoint, xf]
    # Free CPs: [waypoint] (1 waypoint = 3 elements, but we have 2 segments)
    # Actually for 2 segments: CP = [x0, CP1, xf]
    # Free CP is just CP1 (the midpoint)

    # Use the waypoint as initial free CP
    CP1 = waypoints[0]

    # Estimate segment durations
    dist1 = jnp.linalg.norm(CP1 - x0)
    dist2 = jnp.linalg.norm(xf - CP1)
    T1 = dist1 / 2.0  # Assume speed ~2 m/s
    T2 = dist2 / 2.0

    # But wait, we have 2 segments, so we need 2 free CPs
    # Let me check the structure...
    # For N_seg=2, we have 3 CPs total: [x0, CP1, xf]
    # x0 and xf are fixed
    # CP1 is free (3 params)
    # But we also need the midpoint between segments...

    # Actually for N_seg=2:
    # Segment 0: x0 -> waypoint
    # Segment 1: waypoint -> xf
    # CPs per segment: each segment has 4 CPs (cubic Bezier)
    # Total CPs: 2*4 - 1 (shared middle) = 7 CPs
    # Fixed: x0, xf (2 CPs)
    # Free: 5 CPs

    # Let me simplify: just interpolate between start and goal
    # For 2 segments, we need 5 free CPs (7 total - 2 fixed)

    # Free CPs: linearly interpolate
    # Positions for 7 CPs: x0, cp1, cp2, cp3, cp4, cp5, xf
    # Free: cp1, cp2, cp3, cp4, cp5

    N_seg = env_params['N_seg']
    N_cp_per_seg = 4
    N_cp_total = N_seg * N_cp_per_seg - (N_seg - 1)  # 2*4 - 1 = 7

    # Free CPs are all except x0 and xf
    N_free = N_cp_total - 2  # 5 free CPs

    # Interpolate
    free_CPs = []
    for i in range(1, N_cp_total - 1):
        alpha = i / (N_cp_total - 1)
        cp = (1 - alpha) * x0 + alpha * xf
        free_CPs.append(cp)

    CP_free = jnp.concatenate(free_CPs)  # (15,) for 5 CPs

    # Segment durations
    total_dist = jnp.linalg.norm(xf - x0)
    T_per_seg = total_dist / (N_seg * 2.0)  # Assume speed ~2 m/s
    T = jnp.ones(N_seg) * T_per_seg

    # Concatenate
    z_init = jnp.concatenate([CP_free, T])  # (15 + 2,) = (17,)

    return z_init


def create_observation(env_params: Dict, ego_pos: jnp.ndarray, ego_vel: jnp.ndarray) -> jnp.ndarray:
    """
    Create 12D observation for policy network.

    Args:
        env_params: Environment parameters
        ego_pos: (3,) ego position
        ego_vel: (3,) ego velocity

    Returns:
        obs: (12,) observation [ego_pos, ego_vel, adv_pos, adv_vel]
    """
    adv_pos = env_params['adversary_pos']
    adv_vel = env_params['adversary_vel']

    obs = jnp.concatenate([ego_pos, ego_vel, adv_pos, adv_vel])
    return obs


# ============================================================================
# Task Loss
# ============================================================================

def compute_task_loss(ego_trajectory: jnp.ndarray,
                     adversary_trajectory: jnp.ndarray,
                     goal_pos: jnp.ndarray,
                     goal_weight: float = 1.0,
                     safety_margin: float = 1.0) -> Tuple[float, Dict]:
    """
    Compute task loss for evasion.

    Objective:
    - MAXIMIZE minimum distance from adversary (negative loss)
    - MINIMIZE distance to goal
    - PENALIZE getting too close to adversary (< safety_margin)

    Args:
        ego_trajectory: (T, 3) ego positions
        adversary_trajectory: (T, 3) adversary positions
        goal_pos: (3,) goal position
        goal_weight: Weight for goal reaching term
        safety_margin: Safety margin for proximity penalty (meters)

    Returns:
        loss: Scalar task loss (minimize this)
        metrics: Dict with min_dist, goal_dist, proximity_violations
    """
    # Align trajectories
    T_ego = ego_trajectory.shape[0]
    T_adv = adversary_trajectory.shape[0]

    if T_ego != T_adv:
        t_ego = jnp.linspace(0, 1, T_ego)
        t_adv = jnp.linspace(0, 1, T_adv)
        adv_aligned = jnp.array([
            jnp.interp(t_ego, t_adv, adversary_trajectory[:, dim])
            for dim in range(3)
        ]).T
    else:
        adv_aligned = adversary_trajectory

    # Compute distances
    dists = jnp.linalg.norm(ego_trajectory - adv_aligned, axis=1)
    min_dist = jnp.min(dists)

    # Proximity penalty (penalize distances < safety_margin)
    proximity_violations = jnp.clip(safety_margin - dists, 0.0, None)
    proximity_penalty = jnp.sum(proximity_violations**2)

    # Goal reaching cost
    goal_dist = jnp.linalg.norm(ego_trajectory[-1] - goal_pos)
    goal_cost = goal_dist**2

    # Total loss: maximize min_dist (so -min_dist), minimize goal_dist, penalize proximity
    loss = -min_dist + goal_weight * goal_cost + 10.0 * proximity_penalty

    metrics = {
        'min_dist': float(min_dist),
        'goal_dist': float(goal_dist),
        'proximity_penalty': float(proximity_penalty),
    }

    return float(loss), metrics


# ============================================================================
# Forward Pass
# ============================================================================

def forward_pass_simple(policy_params: Dict,
                       policy: EvasionPolicy,
                       obs: jnp.ndarray,
                       env_params: Dict,
                       z_init: jnp.ndarray,
                       solver_iters: int = 20) -> Tuple[jnp.ndarray, Dict]:
    """
    Simple forward pass using MIGHTY solver.

    Args:
        policy_params: Policy network parameters
        policy: Policy module
        obs: (12,) observation
        env_params: Environment parameters
        z_init: Initial decision vector
        solver_iters: Number of solver iterations

    Returns:
        ego_trajectory: (T, 3) ego trajectory positions
        info: Dict with theta, z_star, convergence info
    """
    from mighty_solver_jax import solve_and_extract_trajectory

    # 1. Policy forward: obs -> theta
    theta = policy.apply(policy_params, obs)

    # 2. Solve MIGHTY optimization
    z_star, traj_info = solve_and_extract_trajectory(
        theta, env_params, z_init, maxiter=solver_iters
    )

    positions = traj_info['positions']

    info = {
        'theta': theta,
        'z_star': z_star,
        'times': jnp.linspace(0, 1, len(positions)),  # Approximate times
        'converged': True,
    }

    return positions, info


# ============================================================================
# Training Step
# ============================================================================

def train_step(policy_params: Dict,
              policy: EvasionPolicy,
              env_params: Dict,
              adversary_traj: jnp.ndarray,
              adversary_level: int,
              opt_state,
              optimizer,
              num_steps: int = 30) -> Tuple[Dict, any, float, Dict]:
    """
    Single training step.

    Args:
        policy_params: Policy network parameters
        policy: Policy module
        env_params: Environment parameters
        adversary_traj: Pre-simulated adversary trajectory
        adversary_level: 1 (straight line) or 2 (proportional nav)
        opt_state: Optimizer state
        optimizer: Optax optimizer
        num_steps: Number of unrolled gradient steps

    Returns:
        updated_params: Updated policy parameters
        updated_opt_state: Updated optimizer state
        loss: Task loss value
        info: Training info dict
    """
    # Create initial guess
    z_init = create_initial_guess_from_env(env_params)

    # Create observation (use start position and velocity, initial adversary state)
    obs = get_observation_from_scenario(env_params, adversary_traj, time_idx=0)

    # Forward pass to get loss
    ego_traj, fwd_info = forward_pass_simple(
        policy_params, policy, obs, env_params, z_init, solver_iters=num_steps
    )

    task_loss, metrics = compute_task_loss(ego_traj, adversary_traj, env_params['xf'])

    # Compute gradients using REINFORCE-style estimation
    # (Finite differences on theta, then backprop through policy)

    # Get current theta
    theta_current = fwd_info['theta']

    # Compute gradient w.r.t. theta using finite differences
    def loss_for_theta(theta):
        """Loss as function of theta (cost weights)."""
        z_star, traj_info = solve_and_extract_trajectory(
            theta, env_params, z_init, maxiter=num_steps
        )
        loss_val, _ = compute_task_loss(traj_info['positions'], adversary_traj, env_params['xf'])
        return loss_val

    theta_keys = list(theta_current.keys())
    theta_grads = {}
    eps = 1e-3

    for key in theta_keys:
        theta_plus = {**theta_current, key: theta_current[key] + eps}
        loss_plus = loss_for_theta(theta_plus)
        theta_grads[key] = (loss_plus - task_loss) / eps

    # Now compute gradient w.r.t. policy params using chain rule
    # d(loss)/d(params) = d(loss)/d(theta) * d(theta)/d(params)
    def pseudo_loss(params):
        theta = policy.apply(params, obs)
        # Weight by estimated gradients
        weighted_sum = sum(theta_grads[k] * theta[k] for k in theta_keys)
        return weighted_sum

    grads = jax.grad(pseudo_loss)(policy_params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)

    loss = task_loss

    info = {
        'ego_trajectory': ego_traj,
        'adversary_trajectory': adversary_traj,  # Pass through
        'theta': fwd_info['theta'],
        'converged': fwd_info['converged'],
        'metrics': metrics,
    }

    return policy_params, opt_state, loss, info


# ============================================================================
# Main Training Loop
# ============================================================================

def train(num_episodes: int = 1000,
         learning_rate: float = 1e-3,
         adversary_level: int = 1,
         save_dir: str = 'training_results',
         visualize: bool = True,
         live_viz: bool = False,
         plot_every: int = 10,
         num_steps: int = 30,
         seed: int = 42) -> Tuple[Dict, TrainingVisualizer]:
    """
    Main training loop.

    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate
        adversary_level: 1 (straight line) or 2 (proportional nav)
        save_dir: Directory to save results
        visualize: Whether to visualize
        live_viz: Live visualization (requires display)
        plot_every: Plot every N episodes
        num_steps: Unrolled gradient steps
        seed: Random seed

    Returns:
        policy_params: Trained policy parameters
        visualizer: Training visualizer
    """
    print("=" * 70)
    print("TRAINING MIGHTY EVASION POLICY WITH VISUALIZATION")
    print("=" * 70)

    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Initialize policy
    policy, policy_params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=seed)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(policy_params)

    # Initialize visualizer
    viz = None
    if visualize:
        viz_dir = save_path / 'visualizations'
        viz = TrainingVisualizer(save_dir=str(viz_dir), live=live_viz, plot_every=plot_every)

    # Random key
    rng = jax.random.PRNGKey(seed)

    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Adversary level: {adversary_level}")
    print(f"  Unrolled steps: {num_steps}")
    print(f"  Visualization: {visualize} (live={live_viz}, plot_every={plot_every})")
    print(f"  Save directory: {save_path}")
    print()

    # Training loop
    for episode in range(num_episodes):
        t_start = time.perf_counter()

        # Sample scenario
        env_params, rng = sample_random_scenario(rng)

        # Simulate adversary (simple Level 1 for now - straight line toward goal)
        # Create a simple adversary trajectory
        adv_start = jnp.array([env_params['x0'][0] + 2.0, env_params['x0'][1], env_params['x0'][2]])
        adv_goal = env_params['xf']

        # Simple straight-line adversary
        num_adv_points = 50
        t_adv = jnp.linspace(0, 1, num_adv_points)
        adversary_traj = adv_start[None, :] + t_adv[:, None] * (adv_goal - adv_start)[None, :]

        # Training step
        policy_params, opt_state, loss, info = train_step(
            policy_params, policy, env_params, adversary_traj, adversary_level,
            opt_state, optimizer, num_steps=num_steps
        )

        t_episode = time.perf_counter() - t_start

        # Log to visualizer
        if viz is not None:
            viz.log_episode(
                episode=episode,
                loss=loss,
                ego_trajectory=np.array(info['ego_trajectory']),
                adversary_trajectory=np.array(info['adversary_trajectory']),
                ego_start=np.array(env_params['x0']),
                ego_goal=np.array(env_params['xf']),
                adv_start=np.array(adv_start),  # Use our adversary start
                converged=info['converged'],
                theta=info['theta'],
            )

        # Print summary
        if episode % 10 == 0:
            if viz is not None:
                viz.print_summary(episode)
            else:
                metrics = info['metrics']
                print(f"Episode {episode:4d} | "
                      f"Loss: {loss:7.3f} | "
                      f"Min dist: {metrics['min_dist']:5.2f}m | "
                      f"Goal dist: {metrics['goal_dist']:5.2f}m | "
                      f"Time: {t_episode:5.2f}s")

        # Save checkpoint
        if episode % 100 == 0 and episode > 0:
            checkpoint = {
                'episode': episode,
                'policy_params': policy_params,
                'opt_state': opt_state,
            }
            checkpoint_path = save_path / f"checkpoint_{episode:04d}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Final plots
    if viz is not None:
        viz.plot_training_curves()

    # Save final model
    final_path = save_path / "final_policy.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(policy_params, f)
    print(f"\n💾 Saved final model: {final_path}")

    if viz is not None:
        viz.close()

    return policy_params, viz


# Example usage
if __name__ == "__main__":
    # Quick test
    print("Running quick training test (20 episodes)...")
    print()

    params, viz = train(
        num_episodes=20,
        learning_rate=1e-3,
        adversary_level=1,
        visualize=True,
        live_viz=False,
        plot_every=1,
        num_steps=10,  # Reduced for speed
        seed=42
    )

    print("\n✓ Training test complete!")
