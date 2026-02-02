"""
Phase 3 Test Suite: Neural Network Policy

Tests for policy_network.py to validate:
1. Policy initialization and forward pass
2. Output shapes and value ranges
3. Gradient computation through policy
4. Integration with Phase 2 solver
5. Batch processing
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import time
from policy_network import (
    EvasionPolicy,
    EvasionPolicyWithGoal,
    create_observation,
    create_relative_observation,
    initialize_policy,
    count_parameters,
    policy_forward,
)
from mighty_solver_jax import solve_mighty_jax, solve_and_extract_trajectory
from mighty_solver_test import create_test_scenario, create_initial_guess


# ============================================================================
# TEST 1: Policy Initialization
# ============================================================================

def test_policy_initialization():
    """Test policy initialization and parameter counting."""
    print("=" * 70)
    print("TEST 1: Policy Initialization")
    print("=" * 70)

    # Initialize base policy
    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    print("Base EvasionPolicy:")
    print(f"  Hidden dimension: 64")
    print(f"  Total parameters: {count_parameters(params)}")

    # Expected: input(12) -> hidden1(64) -> hidden2(64) -> output(7)
    # Parameters: (12*64 + 64) + (64*64 + 64) + (64*7 + 7)
    expected_params = (12 * 64 + 64) + (64 * 64 + 64) + (64 * 7 + 7)
    actual_params = count_parameters(params)

    print(f"  Expected parameters: {expected_params}")
    print(f"  Actual parameters: {actual_params}")

    assert actual_params == expected_params, f"Parameter count mismatch: {actual_params} != {expected_params}"
    print("  ✓ Parameter count matches expected")

    # Test with different hidden dimension
    policy_large, params_large, _ = initialize_policy(EvasionPolicy, hidden_dim=128, seed=42)
    params_large_count = count_parameters(params_large)

    print(f"\nLarge policy (hidden_dim=128):")
    print(f"  Total parameters: {params_large_count}")
    assert params_large_count > actual_params, "Larger policy should have more parameters"
    print("  ✓ Larger hidden dimension increases parameter count")

    # Test extended policy
    policy_ext, params_ext, _ = initialize_policy(EvasionPolicyWithGoal, hidden_dim=64, seed=42)
    params_ext_count = count_parameters(params_ext)

    print(f"\nExtended policy (with goal offset):")
    print(f"  Total parameters: {params_ext_count}")
    assert params_ext_count > actual_params, "Extended policy should have more parameters"
    print("  ✓ Extended policy has additional parameters for goal head")

    print("\n✓ TEST 1 PASSED\n")
    return params, params_ext


# ============================================================================
# TEST 2: Forward Pass and Output Validation
# ============================================================================

def test_forward_pass():
    """Test forward pass and validate outputs."""
    print("=" * 70)
    print("TEST 2: Forward Pass and Output Validation")
    print("=" * 70)

    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Create test observation
    ego_pos = jnp.array([0.0, 0.0, 1.0])
    ego_vel = jnp.array([1.0, 0.5, 0.0])
    adv_pos = jnp.array([5.0, 2.0, 1.0])
    adv_vel = jnp.array([-0.5, 0.3, 0.0])

    obs = create_observation(ego_pos, ego_vel, adv_pos, adv_vel)

    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (12,), f"Observation should be (12,), got {obs.shape}"
    print("  ✓ Observation shape correct")

    # Forward pass
    theta = apply_fn(params, obs)

    print(f"\nOutput cost weights:")
    expected_keys = ['time_weight', 'dyn_weight', 'stat_weight', 'accel_weight',
                     'jerk_weight', 'vel_constr_weight', 'acc_constr_weight']

    for key in expected_keys:
        assert key in theta, f"Missing key: {key}"
        val = float(theta[key])
        print(f"  {key:20s}: {val:.6f}")

        # Check positivity
        assert val > 0, f"{key} should be positive, got {val}"

    print("  ✓ All 7 cost weights present and positive")

    # Check that weights are in reasonable ranges
    # Base values are [10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0]
    # With softplus, we expect values >= base values
    assert theta['time_weight'] >= 10.0, "time_weight should be >= 10.0"
    assert theta['vel_constr_weight'] >= 10.0, "vel_constr_weight should be >= 10.0"
    print("  ✓ Weights are above base values")

    print("\n✓ TEST 2 PASSED\n")
    return theta


# ============================================================================
# TEST 3: Batch Processing
# ============================================================================

def test_batch_processing():
    """Test policy with batched observations."""
    print("=" * 70)
    print("TEST 3: Batch Processing")
    print("=" * 70)

    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Create batch of observations
    batch_size = 5
    obs_batch = jnp.array([
        create_observation(
            jnp.array([i * 1.0, 0.0, 1.0]),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([5.0, i * 0.5, 1.0]),
            jnp.array([-0.5, 0.0, 0.0]),
        )
        for i in range(batch_size)
    ])

    print(f"Batch observation shape: {obs_batch.shape}")
    assert obs_batch.shape == (batch_size, 12), f"Expected ({batch_size}, 12), got {obs_batch.shape}"

    # Vectorized forward pass
    theta_batch = jax.vmap(lambda obs: apply_fn(params, obs))(obs_batch)

    print(f"\nBatch output shapes:")
    for key in theta_batch.keys():
        shape = theta_batch[key].shape
        print(f"  {key:20s}: {shape}")
        assert shape == (batch_size,), f"Expected ({batch_size},), got {shape}"

    print("  ✓ Batch processing works correctly")

    # Check that different observations produce different outputs
    weights_0 = theta_batch['time_weight'][0]
    weights_1 = theta_batch['time_weight'][1]
    print(f"\ntime_weight[0]: {float(weights_0):.6f}")
    print(f"time_weight[1]: {float(weights_1):.6f}")

    # They should be different (unless by extreme coincidence)
    if abs(weights_0 - weights_1) > 1e-6:
        print("  ✓ Different observations produce different outputs")
    else:
        print("  Note: Outputs are very similar (may need more diverse inputs)")

    print("\n✓ TEST 3 PASSED\n")
    return theta_batch


# ============================================================================
# TEST 4: Gradient Computation
# ============================================================================

def test_gradient_computation():
    """Test that we can compute gradients w.r.t. policy parameters."""
    print("=" * 70)
    print("TEST 4: Gradient Computation")
    print("=" * 70)

    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Create observation
    obs = create_observation(
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([1.0, 0.5, 0.0]),
        jnp.array([5.0, 2.0, 1.0]),
        jnp.array([-0.5, 0.3, 0.0]),
    )

    # Define loss: sum of all output weights
    def loss_fn(params_dict):
        theta = apply_fn(params_dict, obs)
        return sum(theta.values())

    print("Computing gradient of weight sum w.r.t. parameters...")
    t_start = time.perf_counter()

    grad = jax.grad(loss_fn)(params)

    t_grad = time.perf_counter() - t_start

    print(f"  Time: {t_grad*1000:.2f} ms")

    # Check that gradient is non-trivial
    grad_flat = jax.tree_util.tree_leaves(grad)
    grad_norms = [float(jnp.linalg.norm(g.flatten())) for g in grad_flat]
    total_grad_norm = float(jnp.sqrt(sum(n**2 for n in grad_norms)))

    print(f"  Total gradient norm: {total_grad_norm:.6f}")

    assert total_grad_norm > 0, "Gradient should be non-zero"
    print("  ✓ Gradient is non-trivial")

    # Check all gradients are finite
    all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_flat)
    assert all_finite, "All gradients should be finite"
    print("  ✓ All gradients are finite")

    print("\n✓ TEST 4 PASSED\n")
    return grad


# ============================================================================
# TEST 5: Integration with Solver
# ============================================================================

def test_solver_integration():
    """Test that policy outputs work with Phase 2 solver."""
    print("=" * 70)
    print("TEST 5: Integration with Phase 2 Solver")
    print("=" * 70)

    # Initialize policy
    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Create observation
    obs = create_observation(
        jnp.array([0.0, 0.0, 0.0]),  # ego at origin
        jnp.array([0.0, 1.0, 0.5]),  # moving forward
        jnp.array([2.0, 3.0, 1.0]),  # adversary ahead
        jnp.array([0.0, 0.0, 0.0]),  # stationary
    )

    # Get cost weights from policy
    theta = apply_fn(params, obs)

    print("Policy output (cost weights):")
    for key, val in theta.items():
        print(f"  {key:20s}: {float(val):.4f}")

    # Setup scenario for solver
    env_params, _, M, F = create_test_scenario()
    z0 = create_initial_guess(M, F, np.array(env_params['wps']))
    z0_jax = jnp.array(z0)

    # Solve with policy-generated weights
    print("\nSolving MIGHTY with policy weights...")
    t_start = time.perf_counter()

    z_star = solve_mighty_jax(theta, env_params, z0_jax, maxiter=30)

    t_solve = time.perf_counter() - t_start

    print(f"  Solve time: {t_solve:.2f} s")

    # Evaluate final cost
    from mighty_jax import evaluate_objective_jax
    final_cost = evaluate_objective_jax(z_star, theta, env_params)

    print(f"  Final cost: {final_cost:.6f}")

    assert jnp.isfinite(final_cost), "Final cost should be finite"
    print("  ✓ Solver works with policy-generated weights")

    # Extract trajectory
    z_star, traj = solve_and_extract_trajectory(theta, env_params, z0_jax, maxiter=30)

    print(f"\nTrajectory:")
    print(f"  Positions: {traj['positions'].shape}")
    print(f"  Duration: {float(traj['times'][-1]):.4f} s")

    print("  ✓ Trajectory extraction works")

    print("\n✓ TEST 5 PASSED\n")
    return z_star, traj


# ============================================================================
# TEST 6: Extended Policy with Goal Offset
# ============================================================================

def test_extended_policy():
    """Test extended policy that outputs goal offset."""
    print("=" * 70)
    print("TEST 6: Extended Policy with Goal Offset")
    print("=" * 70)

    policy_ext, params_ext, apply_fn_ext = initialize_policy(
        EvasionPolicyWithGoal, hidden_dim=64, seed=42
    )

    # Create observation
    obs = create_observation(
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([1.0, 0.5, 0.0]),
        jnp.array([5.0, 2.0, 1.0]),
        jnp.array([-0.5, 0.3, 0.0]),
    )

    # Forward pass
    theta_ext = apply_fn_ext(params_ext, obs)

    print("Extended policy output:")
    for key, val in theta_ext.items():
        if key == 'goal_offset':
            print(f"  {key:20s}: {val}")
            assert val.shape == (3,), f"goal_offset should be (3,), got {val.shape}"

            # Check bounded by max_goal_offset (0.5 by default)
            max_offset = jnp.max(jnp.abs(val))
            print(f"    Max absolute offset: {float(max_offset):.4f}")
            assert max_offset <= 0.5, f"goal_offset should be bounded by 0.5, got {max_offset}"
        else:
            print(f"  {key:20s}: {float(val):.4f}")

    print("  ✓ Extended policy outputs goal offset")
    print("  ✓ Goal offset is properly bounded")

    # Test gradient computation through goal offset
    def loss_fn(params_dict):
        theta = apply_fn_ext(params_dict, obs)
        return jnp.sum(theta['goal_offset']**2)

    grad = jax.grad(loss_fn)(params_ext)
    grad_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad))))

    print(f"\n  Gradient norm (goal offset loss): {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient should be non-zero"
    print("  ✓ Gradients flow through goal offset head")

    print("\n✓ TEST 6 PASSED\n")
    return theta_ext


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n")
    print("=" * 70)
    print("POLICY NETWORK - PHASE 3 TEST SUITE")
    print("=" * 70)
    print()

    try:
        # Test 1: Initialization
        params, params_ext = test_policy_initialization()

        # Test 2: Forward pass
        theta = test_forward_pass()

        # Test 3: Batch processing
        theta_batch = test_batch_processing()

        # Test 4: Gradient computation
        grad = test_gradient_computation()

        # Test 5: Solver integration
        z_star, traj = test_solver_integration()

        # Test 6: Extended policy
        theta_ext = test_extended_policy()

        # Summary
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("✓ PASSED: Policy Initialization")
        print("✓ PASSED: Forward Pass and Output Validation")
        print("✓ PASSED: Batch Processing")
        print("✓ PASSED: Gradient Computation")
        print("✓ PASSED: Integration with Phase 2 Solver")
        print("✓ PASSED: Extended Policy with Goal Offset")
        print()
        print("Total: 6/6 tests passed")
        print()
        print("🎉 Phase 3 Complete! Policy network is ready.")
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
