"""Detailed gradient debugging."""

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

print("Gradient Analysis")
print("=" * 70)

# Compute gradient
grad_fn = jax.grad(evaluate_objective_jax, argnums=0)
grad = grad_fn(z0_jax, theta_default, env_params)

grad_np = np.array(grad)

print(f"Decision vector z has {len(z0)} parameters:")
print(f"  Free CPs: {F*3} = {F} CPs × 3 dims")
print(f"  Time slacks: {M}")
print(f"  Total: {len(z0)}")
print()

print(f"Gradient statistics:")
print(f"  NaN count: {np.sum(np.isnan(grad_np))}")
print(f"  Inf count: {np.sum(np.isinf(grad_np))}")
print(f"  Finite count: {np.sum(np.isfinite(grad_np))}")
print()

print("NaN gradient indices:")
nan_indices = np.where(np.isnan(grad_np))[0]
for idx in nan_indices:
    if idx < F * 3:
        cp_idx = idx // 3
        dim = idx % 3
        dim_name = ['x', 'y', 'z'][dim]
        print(f"  {idx}: Free CP {cp_idx}, dimension {dim_name}")
    else:
        seg_idx = idx - F * 3
        print(f"  {idx}: Time slack for segment {seg_idx}")

print()
print("Finite gradient values:")
finite_indices = np.where(np.isfinite(grad_np))[0]
for idx in finite_indices:
    if idx < F * 3:
        cp_idx = idx // 3
        dim = idx % 3
        dim_name = ['x', 'y', 'z'][dim]
        print(f"  {idx}: Free CP {cp_idx}, {dim_name} = {grad_np[idx]:.6e}")
    else:
        seg_idx = idx - F * 3
        print(f"  {idx}: Time slack {seg_idx} = {grad_np[idx]:.6e}")
