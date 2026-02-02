"""Debug script for Phase 2 solver issues."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from mighty_jax import evaluate_objective_jax
from mighty_solver_test import create_test_scenario, create_initial_guess

# Setup
env_params, theta_default, M, F = create_test_scenario()
z0 = create_initial_guess(M, F, np.array(env_params['wps']))
z0_jax = jnp.array(z0)

print("=" * 70)
print("Debugging Solver Issues")
print("=" * 70)

# Test 1: Evaluate cost
print("\n1. Testing cost evaluation...")
try:
    cost = evaluate_objective_jax(z0_jax, theta_default, env_params)
    print(f"   Cost: {cost:.6f}")
    print(f"   Is finite: {jnp.isfinite(cost)}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Evaluate gradient
print("\n2. Testing gradient evaluation...")
try:
    grad_fn = jax.grad(evaluate_objective_jax, argnums=0)
    grad = grad_fn(z0_jax, theta_default, env_params)
    print(f"   Gradient shape: {grad.shape}")
    print(f"   Gradient norm: {jnp.linalg.norm(grad):.6e}")
    print(f"   All finite: {jnp.all(jnp.isfinite(grad))}")
    print(f"   NaN count: {jnp.sum(jnp.isnan(grad))}")
    print(f"   Inf count: {jnp.sum(jnp.isinf(grad))}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try simple gradient descent step
print("\n3. Testing single gradient descent step...")
try:
    lr = 0.001
    grad = jax.grad(evaluate_objective_jax, argnums=0)(z0_jax, theta_default, env_params)
    z1 = z0_jax - lr * grad
    cost1 = evaluate_objective_jax(z1, theta_default, env_params)
    print(f"   New cost: {cost1:.6f}")
    print(f"   Is finite: {jnp.isfinite(cost1)}")
    print(f"   Cost decreased: {cost1 < cost}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try JAXopt with different settings
print("\n4. Testing JAXopt with verbose output...")
try:
    import jaxopt

    def cost_fn(z):
        return evaluate_objective_jax(z, theta_default, env_params)

    # Try with more conservative settings
    solver = jaxopt.LBFGS(
        fun=cost_fn,
        maxiter=10,
        tol=1e-3,
        implicit_diff=False,  # Disable for debugging
        jit=False,            # Disable JIT for better error messages
        unroll=False,
        linesearch='backtracking',  # Explicit linesearch
        maxls=20,  # More linesearch iterations
        decrease_factor=0.8,  # More conservative line search
    )

    result = solver.run(z0_jax)

    print(f"   Final cost: {result.state.value}")
    print(f"   Gradient norm: {result.state.error}")
    print(f"   Iterations: {result.state.iter_num}")
    print(f"   Converged: {result.state.error < 1e-3}")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Debug complete")
print("=" * 70)
