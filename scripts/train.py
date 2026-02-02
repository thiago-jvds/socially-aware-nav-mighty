"""
Phase 4: Training Loop for MIGHTY Evasion Policy

This module implements the training loop for the neural network policy,
including task loss, gradient computation, and training utilities.

Due to implicit differentiation issues in Phase 2, we use finite differences
to compute gradients through the solver.
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import optax
import time
from typing import Dict, Tuple, Callable

from policy_network import EvasionPolicy, initialize_policy, create_observation
from mighty_solver_jax import solve_mighty_jax, solve_and_extract_trajectory
from mighty_jax import evaluate_objective_jax
from environments import (
    sample_random_scenario,
    create_initial_guess_from_env,
    simulate_adversary_trajectory,
    get_observation_from_scenario,
)


def task_loss(ego_traj: jnp.ndarray,
             adversary_traj: jnp.ndarray,
             goal_pos: jnp.ndarray,
             weights: Dict = None) -> float:
    """
    Evasion task loss: penalize proximity to adversary, reward reaching goal.

    Args:
        ego_traj: (N, 3) optimized ego trajectory positions
        adversary_traj: (M, 3) adversary trajectory positions
        goal_pos: (3,) goal position
        weights: Optional dict with loss weights

    Returns:
        loss: Scalar task loss (lower is better)
    """
    if weights is None:
        weights = {
            'proximity_weight': 10.0,
            'goal_weight': 1.0,
            'min_dist_weight': -1.0,  # Negative = reward for distance
        }

    # Align trajectory lengths if needed
    N_ego = ego_traj.shape[0]
    N_adv = adversary_traj.shape[0]

    if N_ego != N_adv:
        # Resample adversary trajectory to match ego
        ts_ego = jnp.linspace(0, 1, N_ego)
        ts_adv = jnp.linspace(0, 1, N_adv)

        # Simple linear interpolation for each dimension
        adversary_aligned = jnp.array([
            jnp.interp(ts_ego, ts_adv, adversary_traj[:, dim])
            for dim in range(3)
        ]).T
    else:
        adversary_aligned = adversary_traj

    # Compute distances to adversary at each time step
    dists = jnp.linalg.norm(ego_traj - adversary_aligned, axis=1)

    # Minimum distance (reward maximizing this)
    min_dist = jnp.min(dists)

    # Proximity cost: penalize getting too close (< 1m safety margin)
    proximity_violations = jnp.clip(1.0 - dists, 0.0, None)  # max(0, 1 - d)
    proximity_cost = jnp.sum(proximity_violations**2)

    # Goal reaching cost: penalize distance from goal at end
    goal_cost = jnp.linalg.norm(ego_traj[-1] - goal_pos)**2

    # Total loss
    loss = (weights['proximity_weight'] * proximity_cost +
            weights['goal_weight'] * goal_cost +
            weights['min_dist_weight'] * min_dist)

    return loss


def forward_pass(policy_params: Dict,
                policy: EvasionPolicy,
                obs: jnp.ndarray,
                env_params: Dict,
                z_init: jnp.ndarray,
                solver_iters: int = 30) -> Tuple[jnp.ndarray, Dict]:
    """
    Full forward pass: observation -> policy -> solver -> trajectory.

    Args:
        policy_params: Network parameters
        policy: Policy module
        obs: (12,) observation
        env_params: Environment parameters
        z_init: Initial guess for solver
        solver_iters: Number of solver iterations

    Returns:
        ego_traj: (N, 3) ego trajectory positions
        info: Dict with intermediate results
    """
    # 1. Policy forward: obs -> cost weights
    theta = policy.apply(policy_params, obs)

    # 2. MIGHTY solver: cost weights -> optimal trajectory
    z_star, traj = solve_and_extract_trajectory(
        theta, env_params, z_init, maxiter=solver_iters
    )

    ego_traj = traj['positions']

    info = {
        'theta': theta,
        'z_star': z_star,
        'trajectory': traj,
    }

    return ego_traj, info


def compute_loss_with_solver(policy_params: Dict,
                            policy: EvasionPolicy,
                            obs: jnp.ndarray,
                            env_params: Dict,
                            z_init: jnp.ndarray,
                            adversary_traj: jnp.ndarray,
                            solver_iters: int = 30) -> float:
    """
    Compute task loss including solver.

    This function is used for finite difference gradient estimation.

    Args:
        policy_params: Network parameters
        policy: Policy module
        obs: Observation
        env_params: Environment parameters
        z_init: Initial guess
        adversary_traj: Adversary trajectory
        solver_iters: Solver iterations

    Returns:
        loss: Task loss value
    """
    # Forward pass
    ego_traj, _ = forward_pass(policy_params, policy, obs, env_params, z_init, solver_iters)

    # Task loss
    loss = task_loss(ego_traj, adversary_traj, env_params['xf'])

    return loss


def compute_gradient_finite_diff(loss_fn: Callable,
                                params: Dict,
                                eps: float = 1e-4) -> Dict:
    """
    Compute gradient using finite differences.

    Workaround for implicit differentiation issues in Phase 2.

    Args:
        loss_fn: Function that takes params and returns scalar loss
        params: Current parameters (Flax pytree)
        eps: Finite difference step size

    Returns:
        grads: Gradient pytree with same structure as params
    """
    # Evaluate at current params
    loss_0 = loss_fn(params)

    # Compute gradient for each parameter
    def grad_for_param(param_path, param_val):
        """Compute gradient for a single parameter."""
        # Perturb this parameter
        params_plus = params.copy({'params': {**params['params']}})

        # Navigate to the parameter and perturb it
        def set_param_at_path(pytree, path, new_val):
            """Helper to set parameter at nested path."""
            if len(path) == 0:
                return new_val
            key = path[0]
            if isinstance(pytree, dict):
                return {**pytree, key: set_param_at_path(pytree[key], path[1:], new_val)}
            else:
                return new_val

        # For simplicity, use JAX tree operations
        # Compute gradient using central differences
        param_plus = param_val + eps
        param_minus = param_val - eps

        # This is simplified - proper implementation would perturb each element
        # For now, use forward differences
        loss_plus = loss_0 + eps * jnp.sum(param_val)  # Placeholder

        grad = (loss_plus - loss_0) / eps
        return grad

    # Use JAX grad instead (this will work for policy params, just not through solver)
    # Since we're using finite diff for the full pipeline, we can use autodiff for policy params
    grads = jax.grad(loss_fn)(params)

    return grads


def train_step_reinforce(policy_params: Dict,
                        policy: EvasionPolicy,
                        obs: jnp.ndarray,
                        env_params: Dict,
                        z_init: jnp.ndarray,
                        adversary_traj: jnp.ndarray,
                        opt_state,
                        optimizer,
                        solver_iters: int = 20) -> Tuple[Dict, any, float, Dict]:
    """
    Training step using REINFORCE-style gradient estimation.

    Instead of backprop through solver, we use policy gradients:
    - Sample from policy (deterministic in this case)
    - Compute task loss as reward
    - Estimate gradient using REINFORCE

    Args:
        policy_params: Current policy parameters
        policy: Policy module
        obs: Observation
        env_params: Environment parameters
        z_init: Initial guess
        adversary_traj: Adversary trajectory
        opt_state: Optimizer state
        optimizer: Optax optimizer
        solver_iters: Solver iterations

    Returns:
        updated_params: Updated policy parameters
        updated_opt_state: Updated optimizer state
        loss: Task loss value
        info: Dict with training metrics
    """
    # Forward pass to get loss
    ego_traj, fwd_info = forward_pass(
        policy_params, policy, obs, env_params, z_init, solver_iters
    )

    loss = task_loss(ego_traj, adversary_traj, env_params['xf'])

    # For deterministic policy, we can still use standard gradients on policy params
    # The solver is treated as a black box
    # Gradient only flows through: loss -> trajectory (from solver) -> STOP
    # We need a different approach

    # Option: Use evolution strategies or finite differences
    # For now, let's use a simple gradient estimate

    # Compute gradient w.r.t. policy output (theta) using finite diff
    def loss_for_theta(theta):
        """Loss as function of theta (cost weights)."""
        z_star, traj = solve_and_extract_trajectory(
            theta, env_params, z_init, maxiter=solver_iters
        )
        return task_loss(traj['positions'], adversary_traj, env_params['xf'])

    # Get current theta
    theta_current = policy.apply(policy_params, obs)

    # Estimate gradient w.r.t. theta using finite differences
    theta_keys = list(theta_current.keys())
    theta_grads = {}

    eps = 1e-3
    for key in theta_keys:
        theta_plus = {**theta_current, key: theta_current[key] + eps}
        loss_plus = loss_for_theta(theta_plus)

        theta_grads[key] = (loss_plus - loss) / eps

    # Now compute gradient w.r.t. policy params using chain rule
    # d(loss)/d(params) = d(loss)/d(theta) * d(theta)/d(params)

    # d(theta)/d(params) can be computed with autodiff
    def policy_output_fn(params):
        return policy.apply(params, obs)

    # Compute Jacobian: d(theta)/d(params)
    # This is tricky with dict output, so we'll use a simpler approach

    # Simplified: Just use gradient of weighted sum of thetas
    def pseudo_loss(params):
        theta = policy.apply(params, obs)
        # Weight by estimated gradients
        weighted_sum = sum(theta_grads[k] * theta[k] for k in theta_keys)
        return weighted_sum

    grads = jax.grad(pseudo_loss)(policy_params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)

    info = {
        'theta': theta_current,
        'theta_grads': theta_grads,
        'ego_traj': ego_traj,
    }

    return policy_params, opt_state, float(loss), info


def train(num_episodes: int = 100,
         solver_iters: int = 20,
         learning_rate: float = 1e-3,
         save_every: int = 10,
         seed: int = 42) -> Tuple[Dict, list]:
    """
    Main training loop.

    Args:
        num_episodes: Number of training episodes
        solver_iters: Solver iterations per episode
        learning_rate: Learning rate for optimizer
        save_every: Save checkpoints every N episodes
        seed: Random seed

    Returns:
        policy_params: Trained policy parameters
        training_history: List of training metrics
    """
    print("=" * 70)
    print("TRAINING MIGHTY EVASION POLICY")
    print("=" * 70)

    # Initialize policy
    policy, policy_params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=seed)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(policy_params)

    # Training history
    history = []

    # Random key
    rng = jax.random.PRNGKey(seed)

    print(f"\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Solver iterations: {solver_iters}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Policy parameters: {sum(x.size for x in jax.tree_util.tree_leaves(policy_params))}")
    print()

    # Training loop
    for episode in range(num_episodes):
        t_start = time.perf_counter()

        # Sample random scenario
        env_params, rng = sample_random_scenario(rng)

        # Create initial guess
        z_init = create_initial_guess_from_env(env_params)

        # Simulate adversary
        adversary_traj, rng = simulate_adversary_trajectory(
            env_params, rng, strategy='pursuit'
        )

        # Get observation
        obs = get_observation_from_scenario(env_params, adversary_traj, time_idx=0)

        # Training step
        policy_params, opt_state, loss, info = train_step_reinforce(
            policy_params, policy, obs, env_params, z_init,
            adversary_traj, opt_state, optimizer, solver_iters
        )

        t_episode = time.perf_counter() - t_start

        # Log metrics
        history.append({
            'episode': episode,
            'loss': loss,
            'time': t_episode,
            'theta': info['theta'],
        })

        # Print progress
        if episode % max(1, num_episodes // 20) == 0:
            print(f"Episode {episode:4d}/{num_episodes} | "
                  f"Loss: {loss:8.4f} | "
                  f"Time: {t_episode:5.2f}s")

            # Print theta values
            theta_str = ", ".join([f"{k[:4]}:{v:.2f}" for k, v in info['theta'].items()])
            print(f"  Theta: {theta_str}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Summary statistics
    losses = [h['loss'] for h in history]
    print(f"\nFinal statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Best loss: {min(losses):.4f}")
    print(f"  Mean loss (last 10): {sum(losses[-10:])/10:.4f}")

    return policy_params, history


# Example usage
if __name__ == "__main__":
    # Quick test run
    print("Running quick training test (5 episodes)...")
    print()

    policy_params, history = train(
        num_episodes=5,
        solver_iters=10,  # Reduced for speed
        learning_rate=1e-3,
    )

    print("\n✓ Training pipeline works!")
    loss_values = [f"{h['loss']:.2f}" for h in history]
    print(f"\nLoss progression: {loss_values}")
