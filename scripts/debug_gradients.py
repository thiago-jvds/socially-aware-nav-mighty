"""Debug script to isolate NaN gradient sources."""

import numpy as np
import jax
import jax.numpy as jnp
from mighty_jax import *
from mighty_jax_test import create_test_scenario, create_initial_guess

# Setup
env_params, theta_default, M, F = create_test_scenario()
z0 = create_initial_guess(M, F, np.array(env_params['wps']))
z0_jax = jnp.array(z0)

print("Testing individual cost term gradients...")
print("=" * 70)

# Reconstruct
CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                       env_params['a0'], env_params['xf'], env_params['vf'],
                       env_params['af'], env_params['wps'],
                       env_params['free_cps_indices'])

print(f"T values: {T}")
print(f"T min: {jnp.min(T)}, T max: {jnp.max(T)}")
print()

# Test each cost term individually
def test_cost_term(name, cost_fn):
    try:
        cost = float(cost_fn())
        grad_fn = jax.grad(lambda z: cost_fn())
        grad = grad_fn(z0_jax)
        grad_np = np.array(grad)
        n_nan = np.sum(np.isnan(grad_np))
        n_inf = np.sum(np.isinf(grad_np))
        print(f"{name:20s}: cost={cost:12.6f}, NaN={n_nan}, Inf={n_inf}")
        if n_nan > 0:
            print(f"  NaN indices: {np.where(np.isnan(grad_np))[0].tolist()}")
    except Exception as e:
        print(f"{name:20s}: ERROR - {e}")

# Test time cost
def cost_time():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_time_jax(T)

test_cost_term("J_time", cost_time)

# Test jerk cost
def cost_jerk():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_jerk_jax(CP, T)

test_cost_term("J_jerk", cost_jerk)

# Test acc cost
def cost_acc():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_acc_jax(CP, T)

test_cost_term("J_acc", cost_acc)

# Test dyn cost
def cost_dyn():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_dyn_jax(CP, T, env_params)

test_cost_term("J_dyn", cost_dyn)

# Test stat cost
def cost_stat():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_stat_jax(CP, env_params)

test_cost_term("J_stat", cost_stat)

# Test vel constraint cost
def cost_vel():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_vel_constr_jax(CP, T, env_params)

test_cost_term("J_vel_constr", cost_vel)

# Test acc constraint cost
def cost_acc_c():
    CP, T = reconstruct_jax(z0_jax, env_params['x0'], env_params['v0'],
                           env_params['a0'], env_params['xf'], env_params['vf'],
                           env_params['af'], env_params['wps'],
                           env_params['free_cps_indices'])
    return compute_J_acc_constr_jax(CP, T, env_params)

test_cost_term("J_acc_constr", cost_acc_c)

print("\n" + "=" * 70)
print("Testing full objective...")

# Test full objective
grad_fn = jax.grad(evaluate_objective_jax, argnums=0)
grad = grad_fn(z0_jax, theta_default, env_params)
grad_np = np.array(grad)
n_nan = np.sum(np.isnan(grad_np))
n_inf = np.sum(np.isinf(grad_np))

print(f"Full objective: NaN={n_nan}, Inf={n_inf}")
if n_nan > 0:
    print(f"NaN indices: {np.where(np.isnan(grad_np))[0].tolist()}")

print("\n" + "=" * 70)
print("Gradient analysis complete")
