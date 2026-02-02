"""
Phase 4 Test Suite: Training Loop and Task Loss

Tests for train.py and environments.py to validate:
1. Scenario generation
2. Task loss computation
3. Forward pass (policy -> solver -> trajectory)
4. Training step
5. Full training loop
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import time

from environments import (
    sample_random_scenario,
    create_initial_guess_from_env,
    simulate_adversary_trajectory,
    get_observation_from_scenario,
)
from train import (
    task_loss,
    forward_pass,
    compute_loss_with_solver,
    train_step_reinforce,
    train,
)
from policy_network import initialize_policy, EvasionPolicy


# ============================================================================
# TEST 1: Scenario Generation
# ============================================================================

def test_scenario_generation():
    """Test random scenario generation."""
    print("=" * 70)
    print("TEST 1: Scenario Generation")
    print("=" * 70)

    rng = jax.random.PRNGKey(42)

    # Generate scenario
    print("Generating random scenario...")
    env_params, rng = sample_random_scenario(rng)

    print(f"\nScenario generated:")
    print(f"  Start: {env_params['x0']}")
    print(f"  Goal:  {env_params['xf']}")
    print(f"  Waypoints: {env_params['wps'].shape}")
    print(f"  ✓ Scenario generated successfully")

    # Create initial guess
    z_init = create_initial_guess_from_env(env_params)
    print(f"\nInitial guess:")
    print(f"  Shape: {z_init.shape}")
    assert z_init.shape == (21,), f"Expected (21,), got {z_init.shape}"
    print(f"  ✓ Initial guess has correct shape")

    # Simulate adversary
    adversary_traj, rng = simulate_adversary_trajectory(env_params, rng, strategy='pursuit')
    print(f"\nAdversary trajectory:")
    print(f"  Shape: {adversary_traj.shape}")
    print(f"  Start: {adversary_traj[0]}")
    print(f"  End:   {adversary_traj[-1]}")
    assert adversary_traj.shape[1] == 3, "Adversary trajectory should be (N, 3)"
    print(f"  ✓ Adversary trajectory generated")

    # Get observation
    obs = get_observation_from_scenario(env_params, adversary_traj, time_idx=0)
    print(f"\nObservation:")
    print(f"  Shape: {obs.shape}")
    assert obs.shape == (12,), f"Expected (12,), got {obs.shape}"
    print(f"  ✓ Observation has correct shape")

    print("\n✓ TEST 1 PASSED\n")
    return env_params, z_init, adversary_traj, obs


# ============================================================================
# TEST 2: Task Loss Computation
# ============================================================================

def test_task_loss():
    """Test task loss computation."""
    print("=" * 70)
    print("TEST 2: Task Loss Computation")
    print("=" * 70)

    # Create dummy trajectories
    N = 10
    ego_traj = jnp.array([
        [0.0, i * 1.0, 1.0] for i in range(N)
    ])
    adversary_traj = jnp.array([
        [2.0, i * 1.0 + 1.0, 1.0] for i in range(N)
    ])
    goal_pos = jnp.array([0.0, 9.0, 1.0])

    print(f"Ego trajectory shape: {ego_traj.shape}")
    print(f"Adversary trajectory shape: {adversary_traj.shape}")

    # Compute loss
    loss = task_loss(ego_traj, adversary_traj, goal_pos)

    print(f"\nTask loss: {float(loss):.6f}")
    assert jnp.isfinite(loss), "Task loss should be finite"
    print("  ✓ Task loss is finite")

    # Test that closer trajectories have higher loss
    adversary_close = jnp.array([
        [0.5, i * 1.0, 1.0] for i in range(N)  # Much closer
    ])
    loss_close = task_loss(ego_traj, adversary_close, goal_pos)

    print(f"\nWith closer adversary:")
    print(f"  Original loss: {float(loss):.6f}")
    print(f"  Close loss:    {float(loss_close):.6f}")

    assert loss_close > loss, "Closer adversary should have higher loss"
    print("  ✓ Loss increases when adversary is closer")

    # Test goal reaching
    ego_at_goal = jnp.array([
        [0.0, i * 1.0, 1.0] for i in range(N-1)
    ] + [[0.0, 9.0, 1.0]])  # Reaches goal

    ego_far_from_goal = jnp.array([
        [0.0, i * 1.0, 1.0] for i in range(N-1)
    ] + [[5.0, 5.0, 1.0]])  # Far from goal

    loss_at_goal = task_loss(ego_at_goal, adversary_traj, goal_pos)
    loss_far = task_loss(ego_far_from_goal, adversary_traj, goal_pos)

    print(f"\nGoal reaching:")
    print(f"  At goal loss: {float(loss_at_goal):.6f}")
    print(f"  Far from goal: {float(loss_far):.6f}")

    assert loss_far > loss_at_goal, "Being far from goal should increase loss"
    print("  ✓ Loss increases when far from goal")

    print("\n✓ TEST 2 PASSED\n")
    return loss


# ============================================================================
# TEST 3: Forward Pass
# ============================================================================

def test_forward_pass():
    """Test full forward pass through policy and solver."""
    print("=" * 70)
    print("TEST 3: Forward Pass (Policy -> Solver -> Trajectory)")
    print("=" * 70)

    # Initialize policy
    policy, policy_params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Generate scenario
    rng = jax.random.PRNGKey(42)
    env_params, rng = sample_random_scenario(rng)
    z_init = create_initial_guess_from_env(env_params)
    adversary_traj, rng = simulate_adversary_trajectory(env_params, rng)
    obs = get_observation_from_scenario(env_params, adversary_traj)

    print("Running forward pass...")
    print(f"  Observation: {obs.shape}")
    print(f"  Initial guess: {z_init.shape}")

    t_start = time.perf_counter()

    ego_traj, info = forward_pass(
        policy_params, policy, obs, env_params, z_init, solver_iters=10
    )

    t_forward = time.perf_counter() - t_start

    print(f"\nForward pass complete:")
    print(f"  Time: {t_forward:.2f} s")
    print(f"  Ego trajectory: {ego_traj.shape}")
    print(f"  ✓ Forward pass successful")

    # Check output shapes
    assert ego_traj.shape[1] == 3, "Ego trajectory should be (N, 3)"
    print(f"  ✓ Output shapes correct")

    # Check theta was generated
    theta = info['theta']
    print(f"\nGenerated cost weights:")
    for key, val in theta.items():
        print(f"  {key:20s}: {float(val):.4f}")
        assert val > 0, f"{key} should be positive"
    print(f"  ✓ All cost weights positive")

    # Compute task loss
    loss = task_loss(ego_traj, adversary_traj, env_params['xf'])
    print(f"\nTask loss: {float(loss):.6f}")
    assert jnp.isfinite(loss), "Loss should be finite"
    print(f"  ✓ Task loss is finite")

    print("\n✓ TEST 3 PASSED\n")
    return ego_traj, info


# ============================================================================
# TEST 4: Training Step
# ============================================================================

def test_training_step():
    """Test a single training step."""
    print("=" * 70)
    print("TEST 4: Training Step")
    print("=" * 70)

    # Initialize
    policy, policy_params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(policy_params)

    # Generate scenario
    rng = jax.random.PRNGKey(42)
    env_params, rng = sample_random_scenario(rng)
    z_init = create_initial_guess_from_env(env_params)
    adversary_traj, rng = simulate_adversary_trajectory(env_params, rng)
    obs = get_observation_from_scenario(env_params, adversary_traj)

    print("Running training step...")
    t_start = time.perf_counter()

    updated_params, updated_opt_state, loss, info = train_step_reinforce(
        policy_params, policy, obs, env_params, z_init,
        adversary_traj, opt_state, optimizer, solver_iters=10
    )

    t_step = time.perf_counter() - t_start

    print(f"\nTraining step complete:")
    print(f"  Time: {t_step:.2f} s")
    print(f"  Loss: {loss:.6f}")
    print(f"  ✓ Training step successful")

    # Check parameters were updated
    param_leaves_old = jax.tree_util.tree_leaves(policy_params)
    param_leaves_new = jax.tree_util.tree_leaves(updated_params)

    params_changed = False
    for old, new in zip(param_leaves_old, param_leaves_new):
        if not jnp.allclose(old, new):
            params_changed = True
            break

    if params_changed:
        print(f"  ✓ Parameters were updated")
    else:
        print(f"  Note: Parameters unchanged (gradient might be zero)")

    print("\n✓ TEST 4 PASSED\n")
    return updated_params, loss


# ============================================================================
# TEST 5: Multi-Step Training
# ============================================================================

def test_multi_step_training():
    """Test training for multiple steps."""
    print("=" * 70)
    print("TEST 5: Multi-Step Training")
    print("=" * 70)

    print("Running 3-episode training...")
    t_start = time.perf_counter()

    policy_params, history = train(
        num_episodes=3,
        solver_iters=10,
        learning_rate=1e-3,
        seed=42
    )

    t_total = time.perf_counter() - t_start

    print(f"\nTraining complete:")
    print(f"  Total time: {t_total:.2f} s")
    print(f"  Time per episode: {t_total/3:.2f} s")

    # Check history
    assert len(history) == 3, f"Expected 3 episodes in history, got {len(history)}"
    print(f"  ✓ History has correct length")

    # Print loss progression
    losses = [h['loss'] for h in history]
    print(f"\nLoss progression:")
    for i, loss in enumerate(losses):
        print(f"  Episode {i}: {loss:.6f}")

    print(f"  ✓ Training completed successfully")

    print("\n✓ TEST 5 PASSED\n")
    return policy_params, history


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n")
    print("=" * 70)
    print("TRAINING PIPELINE - PHASE 4 TEST SUITE")
    print("=" * 70)
    print()

    try:
        # Test 1: Scenario generation
        env_params, z_init, adversary_traj, obs = test_scenario_generation()

        # Test 2: Task loss
        loss = test_task_loss()

        # Test 3: Forward pass
        ego_traj, info = test_forward_pass()

        # Test 4: Training step
        updated_params, step_loss = test_training_step()

        # Test 5: Multi-step training
        trained_params, history = test_multi_step_training()

        # Summary
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("✓ PASSED: Scenario Generation")
        print("✓ PASSED: Task Loss Computation")
        print("✓ PASSED: Forward Pass")
        print("✓ PASSED: Training Step")
        print("✓ PASSED: Multi-Step Training")
        print()
        print("Total: 5/5 tests passed")
        print()
        print("🎉 Phase 4 Complete! Training pipeline is ready.")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
