# MIGHTY JAX Implementation

This directory contains the JAX implementation of the MIGHTY trajectory optimizer for differentiable training with neural networks.

## Phase 1: JAX Cost Function (COMPLETE)

### Files Created

1. **`mighty_jax.py`** - Full JAX implementation of MIGHTY cost function
   - `evaluate_objective_jax(z, theta, env_params)` - Main differentiable cost function
   - All individual cost terms: `compute_J_time_jax`, `compute_J_jerk_jax`, etc.
   - Trajectory utilities: `reconstruct_jax`, `eval_traj_jax`, `eval_traj_and_derivs_jax`
   - Obstacle handling: `fit_quintic_jax`, `f_obs_poly_jax`

2. **`mighty_jax_test.py`** - Comprehensive validation suite
   - Test 1: Cost value computation and structure
   - Test 2: Gradient computation via `jax.grad`
   - Test 3: JIT compilation and performance
   - Test 4: Gradient validation against finite differences

3. **`requirements_jax.txt`** - Python dependencies

## Installation

Install JAX and dependencies:

```bash
# CPU-only version (faster installation, good for development)
pip install -r requirements_jax.txt

# OR GPU version (if you have CUDA)
pip install --upgrade "jax[cuda12]"
pip install jaxopt numpy
```

## Usage

### Run Tests

```bash
cd /home/kkondo/code/mighty_ws/src/mighty/scripts
python3 mighty_jax_test.py
```

Expected output:
- ✓ Cost is finite
- ✓ All gradients are finite
- ✓ JIT compiled results are consistent
- ✓ Gradient matches finite differences

### Basic Example

```python
import jax.numpy as jnp
from mighty_jax import evaluate_objective_jax

# Define environment parameters
env_params = {
    'x0': jnp.array([0.0, 0.0, 0.0]),
    'v0': jnp.array([0.0, 1.0, 0.5]),
    'a0': jnp.array([0.0, 0.0, 0.0]),
    'xf': jnp.array([0.0, 7.0, 2.0]),
    'vf': jnp.array([0.0, 0.0, 0.0]),
    'af': jnp.array([0.0, 0.0, 0.0]),
    'wps': jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 3.0, 1.0],
                      [0.0, 5.0, 1.5],
                      [0.0, 7.0, 2.0]]),
    'free_cps_indices': [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    'obstacles': [],  # Add obstacles as needed
    'A_stat': [jnp.array([[-0.5, 0.0, 0.1]]) for _ in range(3)],
    'b_stat': [jnp.array([-0.5]) for _ in range(3)],
    'V_max': 2.0,
    'A_max': 100.0,
    'N_samp': 10,
    'Co': 0.5,
    'Cw': 1.0,
    'c_ellipsoid': 2.0,
}

# Define cost weights (from NN or defaults)
theta = {
    'time_weight': 10.0,
    'dyn_weight': 0.1,
    'stat_weight': 0.1,
    'accel_weight': 0.1,
    'jerk_weight': 0.1,
    'vel_constr_weight': 10.0,
    'acc_constr_weight': 1.0,
}

# Decision vector: [free_CPs (F×3), σ_slacks (M)]
z = jnp.concatenate([
    jnp.zeros(18),  # 6 free CPs × 3 dims
    jnp.log(2.0) * jnp.ones(3),  # 3 segments
])

# Evaluate cost
cost = evaluate_objective_jax(z, theta, env_params)
print(f"Cost: {cost}")

# Compute gradient
import jax
grad_fn = jax.grad(evaluate_objective_jax, argnums=0)
gradient = grad_fn(z, theta, env_params)
print(f"Gradient shape: {gradient.shape}")
```

### JIT Compilation for Performance

```python
import jax

# JIT compile for 10-100x speedup
jit_cost = jax.jit(evaluate_objective_jax, static_argnames=['env_params'])
jit_grad = jax.jit(jax.grad(evaluate_objective_jax, argnums=0),
                   static_argnames=['env_params'])

# First call compiles, subsequent calls are fast
cost = jit_cost(z, theta, env_params)
gradient = jit_grad(z, theta, env_params)
```

## Validation Against NumPy

To validate against the original NumPy implementation from the notebook:

1. Open `test_unconstrained_opt_3d.ipynb`
2. Run all cells to define the NumPy functions
3. Extract a test vector `z0` from the optimization results
4. Run the comparison:

```python
# In notebook
CP_numpy, T_numpy = reconstruct(z0, x0, v0, a0, xf, vf, af, global_wps)
J_numpy = evaluate_objective(z0)
g_numpy = compute_analytical_grad(z0)

# With JAX
CP_jax, T_jax = reconstruct_jax(jnp.array(z0), ...)
J_jax = evaluate_objective_jax(jnp.array(z0), theta, env_params)
g_jax = jax.grad(evaluate_objective_jax)(jnp.array(z0), theta, env_params)

# Compare
print(f"Cost error: {abs(J_numpy - float(J_jax)) / abs(J_numpy)}")
print(f"Gradient max rel error: {np.max(np.abs(g_numpy - np.array(g_jax)) / (np.abs(g_numpy) + 1e-12))}")
```

Expected results:
- Cost relative error: < 1e-10
- Gradient relative error: < 1e-5

## Implementation Notes

### Decision Vector Structure

```
z = [free_CPs, σ_slacks]
  = [CP[1,0], CP[1,1], ..., CP[2,2], log(T[0]), log(T[1]), log(T[2])]
```

For 3 segments with 6 free control points:
- Free CPs: 6 × 3 = 18 values
- Time slacks: 3 values
- Total: 21 values

### Cost Function Components

1. **Time cost**: `J_time = Σ T[s]`
2. **Jerk cost**: `J_jerk = Σ_s (3600/T[s]⁵) Σ_m ‖Δ³P[s,m]‖²`
3. **Acceleration cost**: `J_acc = Σ_s (400/T[s]³) Σ_m ‖Δ²P[s,m]‖²`
4. **Dynamic obstacle cost**: `J_dyn = Σ_k Σ_i max(Cw² - d²_ki, 0)³`
5. **Static obstacle cost**: `J_stat = Σ_{s,j,p} max(Co - h_sjp, 0)³`
6. **Velocity constraint**: `J_vel = Σ_i max(‖v_i‖ - V_max, 0)³`
7. **Acceleration constraint**: `J_acc_c = Σ_i max(‖a_i‖ - A_max, 0)³`

### JAX-Specific Considerations

1. **No Python control flow on JAX arrays**: All conditionals use `jnp.where` or `jnp.clip`
2. **Fixed-size loops are fine**: `for s in range(M)` where M is a Python int
3. **`jnp.searchsorted` works in forward mode**: Gradients flow through Bernstein evaluation
4. **All arrays use float64**: Enabled via `jax.config.update("jax_enable_x64", True)`

## Next Steps

### Phase 2: Differentiable Inner-Loop Optimizer

Wrap the cost function with JAXopt L-BFGS:

```python
import jaxopt

def solve_mighty_jax(theta, env_params, z_init):
    def cost_fn(z):
        return evaluate_objective_jax(z, theta, env_params)

    solver = jaxopt.LBFGS(
        fun=cost_fn,
        maxiter=50,
        tol=1e-5,
        implicit_diff=True  # Enables dz*/dtheta
    )
    result = solver.run(z_init)
    return result.params  # z* is differentiable w.r.t. theta
```

### Phase 3: Neural Network Policy

Train a policy network that maps observations to cost weights:

```python
import flax.linen as nn

class EvasionPolicy(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(64)(obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        raw_weights = nn.Dense(7)(x)

        base = jnp.array([10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0])
        weights = jax.nn.softplus(raw_weights) + base

        return {
            'time_weight': weights[0],
            'dyn_weight': weights[1],
            # ...
        }
```

## Architecture Overview

```
┌─────────────────┐
│  Observation    │ (ego state, adversary state)
│  (12D vector)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neural Network  │ (Flax/Optax)
│  Policy (JAX)   │
└────────┬────────┘
         │ θ (cost weights)
         ▼
┌─────────────────┐
│ MIGHTY Cost     │ (mighty_jax.py)
│ evaluate_obj_jax│
└────────┬────────┘
         │ J(z, θ)
         ▼
┌─────────────────┐
│ JAXopt L-BFGS   │ (Phase 2)
│ implicit_diff   │
└────────┬────────┘
         │ z* (optimal trajectory)
         ▼
┌─────────────────┐
│  Task Loss      │ (evasion quality)
│ Backprop → NN   │
└─────────────────┘
```

## References

- Original notebook: `test_unconstrained_opt_3d.ipynb`
- Spec: `DIFF_MIGHTY_RL_SPEC.md`
- JAXopt implicit differentiation: Blondel et al., NeurIPS 2022
