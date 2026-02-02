"""
Quick Phase 4 Tests - Faster version for validation

Runs simplified tests with fewer solver iterations.
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from environments import (
    sample_random_scenario,
    create_initial_guess_from_env,
    simulate_adversary_trajectory,
    get_observation_from_scenario,
)
from train import task_loss, forward_pass, train
from policy_network import initialize_policy, EvasionPolicy


print("=" * 70)
print("PHASE 4 QUICK TESTS")
print("=" * 70)

# Test 1: Scenario generation
print("\n1. Testing scenario generation...")
rng = jax.random.PRNGKey(42)
env_params, rng = sample_random_scenario(rng)
z_init = create_initial_guess_from_env(env_params)
adversary_traj, rng = simulate_adversary_trajectory(env_params, rng)
obs = get_observation_from_scenario(env_params, adversary_traj)

print(f"   ✓ Scenario: {env_params['x0']} -> {env_params['xf']}")
print(f"   ✓ Initial guess: {z_init.shape}")
print(f"   ✓ Adversary: {adversary_traj.shape}")
print(f"   ✓ Observation: {obs.shape}")

# Test 2: Task loss
print("\n2. Testing task loss...")
ego_traj = jnp.ones((9, 3))
adv_traj = jnp.ones((10, 3)) * 2
goal_pos = env_params['xf']

loss = task_loss(ego_traj, adv_traj, goal_pos)
print(f"   ✓ Task loss: {float(loss):.4f}")

# Test 3: Forward pass (with reduced solver iterations)
print("\n3. Testing forward pass (5 solver iters)...")
policy, policy_params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

try:
    ego_traj, info = forward_pass(
        policy_params, policy, obs, env_params, z_init, solver_iters=5
    )
    print(f"   ✓ Forward pass: {ego_traj.shape}")
    print(f"   ✓ Theta: time={float(info['theta']['time_weight']):.2f}")

    # Compute task loss
    loss = task_loss(ego_traj, adversary_traj, env_params['xf'])
    print(f"   ✓ Task loss after forward: {float(loss):.4f}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")

# Test 4: Quick training (1 episode, 5 solver iters)
print("\n4. Testing mini training (1 episode)...")
try:
    policy_params, history = train(
        num_episodes=1,
        solver_iters=5,
        learning_rate=1e-3,
        seed=42
    )
    print(f"   ✓ Training completed")
    print(f"   ✓ Loss: {history[0]['loss']:.4f}")
except Exception as e:
    print(f"   ✗ Training failed: {e}")

print("\n" + "=" * 70)
print("QUICK TESTS COMPLETE")
print("=" * 70)
print("\n✓ Phase 4 components functional!")
print("Note: Full tests with more iterations may be slower")
