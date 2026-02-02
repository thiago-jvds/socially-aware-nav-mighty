"""
Phase 4: Scenario Generation and Environment Utilities

This module provides utilities for generating random evasion scenarios,
simulating adversary trajectories, and creating initial guesses for training.
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import numpy as np
from typing import Tuple, Dict, List


def sample_random_scenario(rng: jax.random.PRNGKey,
                          bounds: Dict = None) -> Tuple[Dict, jax.random.PRNGKey]:
    """
    Sample a random evasion scenario.

    Args:
        rng: JAX random key
        bounds: Optional dict with spatial bounds

    Returns:
        env_params: Environment parameters dict
        rng: Updated random key
    """
    if bounds is None:
        bounds = {
            'x_min': -5.0, 'x_max': 5.0,
            'y_min': -2.0, 'y_max': 10.0,
            'z_min': 0.5, 'z_max': 2.5,
            'v_max': 2.0,
        }

    # Split RNG keys
    keys = jax.random.split(rng, 10)
    rng = keys[0]

    # Sample start position
    x0 = jax.random.uniform(keys[1], (3,), minval=jnp.array([bounds['x_min'], bounds['y_min'], bounds['z_min']]),
                           maxval=jnp.array([bounds['x_max'], bounds['y_min'] + 2.0, bounds['z_max']]))

    # Sample goal position (forward in y direction)
    xf = jax.random.uniform(keys[2], (3,), minval=jnp.array([bounds['x_min'], bounds['y_max'] - 2.0, bounds['z_min']]),
                           maxval=jnp.array([bounds['x_max'], bounds['y_max'], bounds['z_max']]))

    # Sample initial velocity (generally forward)
    v0 = jax.random.uniform(keys[3], (3,), minval=jnp.array([-0.5, 0.5, -0.2]),
                           maxval=jnp.array([0.5, 1.5, 0.2]))

    # Final velocity (slow down at goal)
    vf = jax.random.uniform(keys[4], (3,), minval=jnp.array([-0.2, -0.2, -0.1]),
                           maxval=jnp.array([0.2, 0.2, 0.1]))

    # Zero accelerations at start and end
    a0 = jnp.zeros(3)
    af = jnp.zeros(3)

    # Generate waypoints along the path
    M = 3  # Number of segments
    global_wps = jnp.array([x0])

    for i in range(1, M):
        alpha = i / M
        # Linear interpolation with some random offset
        wp = x0 + alpha * (xf - x0)
        offset = jax.random.uniform(keys[4 + i], (3,), minval=-0.5, maxval=0.5)
        wp = wp + offset
        global_wps = jnp.vstack([global_wps, wp])

    global_wps = jnp.vstack([global_wps, xf])

    # Free control points indices (same structure as tests)
    free_cps_indices = [
        (1, 1), (1, 2), (1, 3), (1, 4),  # Seg 1: CPs 1,2,3,4
        (2, 1), (2, 2)                   # Seg 2: CPs 1,2
    ]

    # Static constraints (ground plane)
    A_stat = [
        jnp.array([[-0.5, 0.0, 0.1]]),  # Segment 0
        jnp.array([[-0.5, 0.5, 0.1]]),  # Segment 1
        jnp.array([[0.0, 0.5, 0.1]]),   # Segment 2
    ]
    b_stat = [
        jnp.array([-0.5]),
        jnp.array([1.0]),
        jnp.array([2.0]),
    ]

    # No dynamic obstacles for now (can add later)
    obstacles = []

    # Environment parameters
    env_params = {
        'x0': x0,
        'v0': v0,
        'a0': a0,
        'xf': xf,
        'vf': vf,
        'af': af,
        'wps': global_wps,
        'free_cps_indices': free_cps_indices,
        'A_stat': A_stat,
        'b_stat': b_stat,
        'obstacles': obstacles,
        'V_max': bounds['v_max'],
        'A_max': 100.0,
        'N_samp': 10,
        'N_per_seg': 3,
        'Co': 0.5,
        'Cw': 1.0,
        'c_ellipsoid': 2.0,
    }

    return env_params, rng


def create_initial_guess_from_env(env_params: Dict) -> jnp.ndarray:
    """
    Create initial decision vector from environment parameters.

    Args:
        env_params: Environment parameters dict

    Returns:
        z0: Initial decision vector
    """
    M = len(env_params['wps']) - 1
    F = len(env_params['free_cps_indices'])
    global_wps = env_params['wps']

    np.random.seed(42)  # Fixed seed for consistency

    # Free control points: interpolate between waypoints
    free_cps = []

    # Segment 1: CPs 1,2,3,4
    wp_start = global_wps[1]
    wp_end = global_wps[2]
    for cp_idx in [1, 2, 3, 4]:
        alpha = cp_idx / 5.0
        pt = wp_start + alpha * (wp_end - wp_start)
        pt = pt + np.random.randn(3) * 0.05
        free_cps.append(pt)

    # Segment 2: CPs 1,2
    wp_start = global_wps[2]
    wp_end = global_wps[3]
    for cp_idx in [1, 2]:
        alpha = cp_idx / 5.0
        pt = wp_start + alpha * (wp_end - wp_start)
        pt = pt + np.random.randn(3) * 0.05
        free_cps.append(pt)

    free_cps = np.array(free_cps).flatten()

    # Time slacks: segment-proportional durations
    sigma_slacks = []
    for s in range(M):
        dist = np.linalg.norm(global_wps[s + 1] - global_wps[s])
        T_guess = dist / 2.0  # Assume avg speed ~ 2.0
        sigma_slacks.append(np.log(max(T_guess, 0.5)))
    sigma_slacks = np.array(sigma_slacks)

    z0 = jnp.concatenate([jnp.array(free_cps), jnp.array(sigma_slacks)])
    return z0


def simulate_adversary_trajectory(env_params: Dict,
                                  rng: jax.random.PRNGKey,
                                  strategy: str = 'pursuit') -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
    """
    Simulate adversary trajectory.

    Args:
        env_params: Environment parameters
        rng: Random key
        strategy: 'pursuit', 'intercept', or 'random'

    Returns:
        adversary_traj: (N_total, 3) adversary trajectory positions
        rng: Updated random key
    """
    keys = jax.random.split(rng, 5)
    rng = keys[0]

    x0 = env_params['x0']
    xf = env_params['xf']
    N_total = env_params['N_samp']

    if strategy == 'pursuit':
        # Adversary starts offset and pursues toward goal
        adv_start = x0 + jax.random.uniform(keys[1], (3,), minval=jnp.array([1.0, -1.0, 0.0]),
                                            maxval=jnp.array([3.0, 2.0, 0.5]))
        adv_goal = xf + jax.random.uniform(keys[2], (3,), minval=-0.5, maxval=0.5)

        # Linear trajectory
        ts = jnp.linspace(0, 1, N_total)
        adversary_traj = adv_start[None, :] + ts[:, None] * (adv_goal - adv_start)[None, :]

    elif strategy == 'intercept':
        # Adversary tries to intercept ego path midway
        adv_start = x0 + jax.random.uniform(keys[1], (3,), minval=jnp.array([2.0, 0.0, 0.0]),
                                            maxval=jnp.array([4.0, 3.0, 0.5]))
        mid_point = (x0 + xf) / 2.0
        adv_goal = mid_point + jax.random.uniform(keys[2], (3,), minval=-1.0, maxval=1.0)

        ts = jnp.linspace(0, 1, N_total)
        adversary_traj = adv_start[None, :] + ts[:, None] * (adv_goal - adv_start)[None, :]

    elif strategy == 'random':
        # Random waypoints
        adv_wps = jax.random.uniform(keys[1], (4, 3),
                                     minval=jnp.array([-3.0, -2.0, 0.5]),
                                     maxval=jnp.array([3.0, 10.0, 2.0]))

        # Interpolate between waypoints
        ts = jnp.linspace(0, 3, N_total)
        seg_idx = jnp.floor(ts).astype(int)
        seg_idx = jnp.clip(seg_idx, 0, 2)
        local_t = ts - seg_idx

        adversary_traj = []
        for i in range(N_total):
            idx = int(seg_idx[i])
            t = float(local_t[i])
            pos = adv_wps[idx] + t * (adv_wps[idx + 1] - adv_wps[idx])
            adversary_traj.append(pos)
        adversary_traj = jnp.array(adversary_traj)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return adversary_traj, rng


def get_observation_from_scenario(env_params: Dict,
                                  adversary_traj: jnp.ndarray,
                                  time_idx: int = 0) -> jnp.ndarray:
    """
    Create observation from scenario at a specific time.

    Args:
        env_params: Environment parameters
        adversary_traj: (N, 3) adversary trajectory
        time_idx: Time index (0 = start)

    Returns:
        obs: (12,) observation vector
    """
    ego_pos = env_params['x0']
    ego_vel = env_params['v0']

    adv_pos = adversary_traj[min(time_idx, len(adversary_traj) - 1)]

    # Estimate adversary velocity from trajectory
    if time_idx < len(adversary_traj) - 1:
        adv_vel = adversary_traj[time_idx + 1] - adversary_traj[time_idx]
    else:
        adv_vel = jnp.zeros(3)

    obs = jnp.concatenate([ego_pos, ego_vel, adv_pos, adv_vel])
    return obs


def visualize_scenario(env_params: Dict, adversary_traj: jnp.ndarray = None):
    """
    Print scenario details (for debugging).

    Args:
        env_params: Environment parameters
        adversary_traj: Optional adversary trajectory
    """
    print("=" * 70)
    print("SCENARIO DETAILS")
    print("=" * 70)

    print(f"\nEgo agent:")
    print(f"  Start: {env_params['x0']}")
    print(f"  Goal:  {env_params['xf']}")
    print(f"  v0:    {env_params['v0']}")
    print(f"  vf:    {env_params['vf']}")

    print(f"\nWaypoints:")
    for i, wp in enumerate(env_params['wps']):
        print(f"  WP {i}: {wp}")

    if adversary_traj is not None:
        print(f"\nAdversary:")
        print(f"  Start: {adversary_traj[0]}")
        print(f"  End:   {adversary_traj[-1]}")
        print(f"  Trajectory length: {len(adversary_traj)}")

        # Compute closest approach
        ego_path = env_params['wps']
        min_dist = float('inf')
        for ego_wp in ego_path:
            dists = jnp.linalg.norm(adversary_traj - ego_wp[None, :], axis=1)
            min_dist = min(min_dist, float(jnp.min(dists)))

        print(f"  Min distance to ego path: {min_dist:.4f}")

    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SCENARIO GENERATION - Example Usage")
    print("=" * 70)

    # Sample random scenario
    rng = jax.random.PRNGKey(42)
    env_params, rng = sample_random_scenario(rng)

    print("\n1. Random scenario sampled")
    print(f"   Start: {env_params['x0']}")
    print(f"   Goal:  {env_params['xf']}")

    # Create initial guess
    z0 = create_initial_guess_from_env(env_params)
    print(f"\n2. Initial guess created")
    print(f"   Decision vector shape: {z0.shape}")

    # Simulate adversary
    adversary_traj, rng = simulate_adversary_trajectory(env_params, rng, strategy='pursuit')
    print(f"\n3. Adversary trajectory simulated")
    print(f"   Trajectory shape: {adversary_traj.shape}")
    print(f"   Start: {adversary_traj[0]}")
    print(f"   End:   {adversary_traj[-1]}")

    # Get observation
    obs = get_observation_from_scenario(env_params, adversary_traj, time_idx=0)
    print(f"\n4. Observation created")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Ego pos: {obs[:3]}")
    print(f"   Adv pos: {obs[6:9]}")

    # Visualize
    print()
    visualize_scenario(env_params, adversary_traj)

    # Test multiple samples
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE SCENARIOS")
    print("=" * 70)

    for i in range(3):
        env_params, rng = sample_random_scenario(rng)
        adversary_traj, rng = simulate_adversary_trajectory(env_params, rng, strategy='pursuit')
        obs = get_observation_from_scenario(env_params, adversary_traj)

        print(f"\nScenario {i+1}:")
        print(f"  Ego: {env_params['x0']} -> {env_params['xf']}")
        print(f"  Adv: {adversary_traj[0]} -> {adversary_traj[-1]}")

    print("\n✓ Scenario generation ready!")
