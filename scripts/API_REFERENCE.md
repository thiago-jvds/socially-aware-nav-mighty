# MIGHTY JAX API Reference

Quick reference for the JAX implementation of MIGHTY trajectory optimizer.

## Main Function

### `evaluate_objective_jax(z, theta, env_params)`

Compute differentiable cost for a trajectory.

**Args:**
- `z`: `(K,)` decision vector `[free_CPs (F×3), σ_slacks (M)]`
- `theta`: `dict` with cost weight keys:
  - `'time_weight'`: scalar
  - `'dyn_weight'`: scalar
  - `'stat_weight'`: scalar
  - `'accel_weight'`: scalar
  - `'jerk_weight'`: scalar
  - `'vel_constr_weight'`: scalar
  - `'acc_constr_weight'`: scalar
- `env_params`: `dict` with keys:
  - `'x0', 'v0', 'a0'`: `(3,)` start conditions
  - `'xf', 'vf', 'af'`: `(3,)` end conditions
  - `'wps'`: `(M+1, 3)` waypoints
  - `'free_cps_indices'`: `list` of `(seg, cp_idx)` tuples
  - `'obstacles'`: `list` of obstacle dicts
  - `'A_stat', 'b_stat'`: static constraint planes
  - `'V_max', 'A_max'`: velocity/acceleration limits
  - `'N_samp'`: number of trajectory samples
  - `'Co', 'Cw', 'c_ellipsoid'`: clearance parameters

**Returns:**
- `scalar`: Total weighted cost `J(z, θ)`

**Usage:**
```python
cost = evaluate_objective_jax(z, theta, env_params)

# Gradient w.r.t. decision vector
grad_z = jax.grad(evaluate_objective_jax, argnums=0)(z, theta, env_params)

# Gradient w.r.t. cost weights (for NN training)
grad_theta_fn = jax.grad(lambda t: evaluate_objective_jax(z, t, env_params))
grad_theta = grad_theta_fn(theta)

# JIT compile for speed
jit_objective = jax.jit(evaluate_objective_jax, static_argnames=['env_params'])
cost = jit_objective(z, theta, env_params)
```

---

## Trajectory Functions

### `reconstruct_jax(z, x0, v0, a0, xf, vf, af, wps, free_cps_indices)`

Convert decision vector to control points and segment times.

**Returns:**
- `CP`: `(M, 6, 3)` control points
- `T`: `(M,)` segment durations

### `eval_traj_jax(CP, T, N_samp)`

Sample trajectory positions.

**Returns:**
- `pts`: `(N_samp, 3)` positions

### `eval_traj_and_derivs_jax(CP, T, N_samp)`

Sample trajectory positions, velocities, and accelerations.

**Returns:**
- `pts`: `(N_samp, 3)` positions
- `vels`: `(N_samp, 3)` velocities
- `accs`: `(N_samp, 3)` accelerations

---

## Cost Term Functions

### `compute_J_time_jax(T)`

Time cost: sum of segment durations.

### `compute_J_jerk_jax(CP, T)`

Jerk cost: smoothness penalty via third-order differences.

### `compute_J_acc_jax(CP, T)`

Acceleration cost: smoothness penalty via second-order differences.

### `compute_J_dyn_jax(CP, T, env_params)`

Dynamic obstacle avoidance cost using ellipsoid distance metric.

### `compute_J_stat_jax(CP, env_params)`

Static obstacle avoidance cost using half-space constraints.

### `compute_J_vel_constr_jax(CP, T, env_params)`

Velocity constraint violation cost: `Σ max(||v|| - V_max, 0)³`

### `compute_J_acc_constr_jax(CP, T, env_params)`

Acceleration constraint violation cost: `Σ max(||a|| - A_max, 0)³`

---

## Obstacle Functions

### `fit_quintic_jax(x0, v0, a0, xf, vf, af, T)`

Fit quintic polynomial for dynamic obstacle trajectory.

**Returns:**
- `cx, cy, cz`: `(6,)` polynomial coefficients

### `f_obs_poly_jax(t, cx, cy, cz)`

Evaluate obstacle position at time(s) t.

**Returns:**
- positions: `(N, 3)` or `(3,)` depending on input

---

## Decision Vector Structure

```python
# For M segments with F free control points:
z = [
    free_CP[0,0], free_CP[0,1], free_CP[0,2],  # First free CP (3D)
    free_CP[1,0], free_CP[1,1], free_CP[1,2],  # Second free CP
    ...
    free_CP[F-1,0], free_CP[F-1,1], free_CP[F-1,2],  # Last free CP
    σ[0], σ[1], ..., σ[M-1]  # Time slacks (log of durations)
]

# Length: 3*F + M
```

**Example** (3 segments, 6 free CPs):
```python
F = len([(i, j) for i in range(1, 3) for j in range(3)])  # 6
M = 3
z = np.concatenate([
    np.zeros(F * 3),          # 18 free CP values
    np.log(2.0) * np.ones(M), # 3 time slacks
])
# Total length: 21
```

---

## Obstacle Dictionary Format

### Dynamic Obstacles

```python
obstacle = {
    'x0': np.array([...]),  # (3,) start position
    'v0': np.array([...]),  # (3,) start velocity
    'a0': np.array([...]),  # (3,) start acceleration
    'xf': np.array([...]),  # (3,) end position
    'vf': np.array([...]),  # (3,) end velocity
    'af': np.array([...]),  # (3,) end acceleration
    'T': 10.0,              # duration
}

env_params['obstacles'] = [obstacle1, obstacle2, ...]
```

### Static Obstacles

```python
# Half-space constraints: A @ p >= b
A_stat = [
    jnp.array([[-0.5, 0.0, 0.1]]),  # Segment 0 constraints (n_planes, 3)
    jnp.array([[-0.5, 0.5, 0.1]]),  # Segment 1 constraints
    jnp.array([[0.0, 0.5, 0.1]]),   # Segment 2 constraints
]

b_stat = [
    jnp.array([-0.5]),  # Segment 0 offsets (n_planes,)
    jnp.array([1.0]),   # Segment 1 offsets
    jnp.array([2.0]),   # Segment 2 offsets
]

env_params['A_stat'] = A_stat
env_params['b_stat'] = b_stat
```

---

## Complete Example

```python
import jax
import jax.numpy as jnp
from mighty_jax import evaluate_objective_jax

# Define environment
env_params = {
    'x0': jnp.array([0.0, 0.0, 0.0]),
    'v0': jnp.array([0.0, 1.0, 0.5]),
    'a0': jnp.array([0.0, 0.0, 0.0]),
    'xf': jnp.array([0.0, 7.0, 2.0]),
    'vf': jnp.array([0.0, 0.0, 0.0]),
    'af': jnp.array([0.0, 0.0, 0.0]),
    'wps': jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 5.0, 1.5],
        [0.0, 7.0, 2.0]
    ]),
    'free_cps_indices': [
        (0, 3), (0, 4), (0, 5),  # Segment 0, CPs 3-5
        (1, 1), (1, 2), (1, 3),  # Segment 1, CPs 1-3
    ],
    'obstacles': [
        {
            'x0': jnp.array([1.0, 3.0, 1.0]),
            'v0': jnp.array([-0.3, 0.3, 0.0]),
            'a0': jnp.array([0.0, 0.0, 0.0]),
            'xf': jnp.array([-2.0, 4.0, 1.5]),
            'vf': jnp.array([0.0, 0.0, 0.0]),
            'af': jnp.array([0.0, 0.0, 0.0]),
            'T': 10.0,
        }
    ],
    'A_stat': [jnp.array([[-0.5, 0.0, 0.1]]) for _ in range(3)],
    'b_stat': [jnp.array([-0.5]), jnp.array([1.0]), jnp.array([2.0])],
    'V_max': 2.0,
    'A_max': 100.0,
    'N_samp': 10,
    'Co': 0.5,
    'Cw': 1.0,
    'c_ellipsoid': 2.0,
}

# Define cost weights
theta = {
    'time_weight': 10.0,
    'dyn_weight': 0.1,
    'stat_weight': 0.1,
    'accel_weight': 0.1,
    'jerk_weight': 0.1,
    'vel_constr_weight': 10.0,
    'acc_constr_weight': 1.0,
}

# Create decision vector
M = 3  # segments
F = 6  # free control points
z = jnp.concatenate([
    jnp.zeros(F * 3),             # Free CPs
    jnp.log(2.0) * jnp.ones(M),   # Time slacks
])

# Evaluate cost
cost = evaluate_objective_jax(z, theta, env_params)
print(f"Cost: {cost}")

# Compute gradients
grad_z = jax.grad(evaluate_objective_jax, argnums=0)(z, theta, env_params)
print(f"Gradient norm: {jnp.linalg.norm(grad_z)}")

# JIT compile for performance
jit_cost = jax.jit(evaluate_objective_jax, static_argnames=['env_params'])
jit_grad = jax.jit(jax.grad(evaluate_objective_jax, argnums=0),
                   static_argnames=['env_params'])

# Fast evaluation
cost = jit_cost(z, theta, env_params)
grad = jit_grad(z, theta, env_params)
```

---

## Performance Tips

1. **Use JIT compilation:**
   ```python
   jit_fn = jax.jit(evaluate_objective_jax, static_argnames=['env_params'])
   ```

2. **Enable 64-bit precision:**
   ```python
   from jax import config
   config.update("jax_enable_x64", True)
   ```

3. **Batch evaluation:** Use `jax.vmap` for multiple scenarios:
   ```python
   batch_cost = jax.vmap(evaluate_objective_jax, in_axes=(0, None, None))
   costs = batch_cost(z_batch, theta, env_params)
   ```

4. **GPU acceleration:** JAX automatically uses GPU if available:
   ```python
   # Check device
   print(jax.devices())
   ```

---

## Common Patterns

### Optimize trajectory with gradients

```python
import optax

optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(z)

for _ in range(100):
    cost, grad = jax.value_and_grad(evaluate_objective_jax)(z, theta, env_params)
    updates, opt_state = optimizer.update(grad, opt_state)
    z = optax.apply_updates(z, updates)
```

### Train neural network weights

```python
def loss_fn(theta_flat):
    theta = unpack_theta(theta_flat)  # Convert to dict
    z_opt = solve_trajectory(theta)    # Run optimizer
    return task_loss(z_opt)            # Evaluate task performance

grad_theta = jax.grad(loss_fn)(theta_flat)
# Update neural network parameters...
```

### Sensitivity analysis

```python
# How does cost change with each weight?
def cost_vs_weight(weight_idx, value):
    theta_test = theta.copy()
    keys = list(theta.keys())
    theta_test[keys[weight_idx]] = value
    return evaluate_objective_jax(z, theta_test, env_params)

sensitivity = jax.grad(lambda w: cost_vs_weight(0, w))(10.0)
```

---

## Error Handling

Common issues and solutions:

**NaN/Inf in gradients:**
- Check for division by zero in segment times
- Ensure all inputs are finite
- Verify obstacle positions don't coincide with ego trajectory

**JIT compilation errors:**
- Ensure `env_params` is marked as static: `static_argnames=['env_params']`
- Check that all Python integers are not traced (use `int()` if needed)

**Slow performance:**
- First call includes compilation time (50-200ms)
- Subsequent calls should be fast (1-5ms)
- Use `jax.jit()` for repeated evaluations

**Shape mismatches:**
- Verify `z` has length `3*F + M`
- Check `wps` is `(M+1, 3)`
- Ensure `free_cps_indices` length matches `F`
