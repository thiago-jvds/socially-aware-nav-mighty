# Phase 3: Neural Network Policy - COMPLETE ✅

## Objective
Create a neural network that maps observations (ego + adversary state) to MIGHTY cost weights, enabling learned trajectory optimization for evasion.

## Implementation Summary

### Files Created
- **policy_network.py** (357 lines): Neural network policy implementation
- **policy_network_test.py** (446 lines): Comprehensive test suite (6 tests)

### Core Components

#### 1. `EvasionPolicy` - Base Policy Network
Simple MLP that maps 12D observations to 7 cost weights.

**Architecture:**
- Input: 12D observation [ego_pos(3), ego_vel(3), adv_pos(3), adv_vel(3)]
- Hidden Layer 1: 64 units + ReLU
- Hidden Layer 2: 64 units + ReLU
- Output: 7 cost weights (positive via softplus + base values)

**Parameters:** 5,447 total
- Layer 1: 12 × 64 + 64 = 832
- Layer 2: 64 × 64 + 64 = 4,160
- Output: 64 × 7 + 7 = 455

**Output Cost Weights:**
```python
{
    'time_weight': scalar,         # Trajectory duration penalty
    'dyn_weight': scalar,          # Dynamic obstacle avoidance
    'stat_weight': scalar,         # Static obstacle avoidance
    'accel_weight': scalar,        # Acceleration minimization
    'jerk_weight': scalar,         # Jerk minimization (smoothness)
    'vel_constr_weight': scalar,   # Velocity constraint violation
    'acc_constr_weight': scalar,   # Acceleration constraint violation
}
```

#### 2. `EvasionPolicyWithGoal` - Extended Policy
Extended version that also outputs goal position offset for dynamic goal adjustment.

**Additional Output:**
- `goal_offset`: (3,) bounded offset to apply to goal position
- Bounded by tanh activation (±0.5m by default)

**Parameters:** 5,642 total (195 additional for goal head)

#### 3. Utility Functions
- `create_observation()` - Construct observation from state components
- `create_relative_observation()` - Use relative coordinates for better generalization
- `initialize_policy()` - Initialize policy with random parameters
- `count_parameters()` - Count total trainable parameters

## Test Results: 6/6 PASSED ✅

```
TEST 1: Policy Initialization                    ✅ PASSED
  - Base policy: 5,447 parameters
  - Large policy (128 hidden): 19,079 parameters
  - Extended policy: 5,642 parameters

TEST 2: Forward Pass and Output Validation       ✅ PASSED
  - Observation shape: (12,) ✓
  - All 7 cost weights present ✓
  - All weights positive (>= base values) ✓
  - Example output:
    time_weight: 11.697, dyn_weight: 1.186, stat_weight: 0.505

TEST 3: Batch Processing                         ✅ PASSED
  - Batch shape: (5, 12) -> (5,) for each weight
  - Different observations produce different outputs ✓

TEST 4: Gradient Computation                     ✅ PASSED
  - Gradient time: 708 ms
  - Gradient norm: 12.08
  - All gradients finite ✓

TEST 5: Integration with Phase 2 Solver          ✅ PASSED
  - Policy generates weights ✓
  - Solver converges with policy weights ✓
  - Final cost: 297.29
  - Trajectory duration: 17.97s
  - Full integration working ✓

TEST 6: Extended Policy with Goal Offset         ✅ PASSED
  - Goal offset shape: (3,) ✓
  - Bounded by ±0.5m ✓
  - Gradients flow through goal head ✓
```

## Integration with Phases 1 & 2

Phase 3 successfully integrates the full pipeline:

```python
# Complete forward pass
obs = create_observation(ego_pos, ego_vel, adv_pos, adv_vel)  # Phase 3
theta = policy.apply(params, obs)                              # Phase 3
z_star = solve_mighty_jax(theta, env_params, z_init)          # Phase 2
cost = evaluate_objective_jax(z_star, theta, env_params)      # Phase 1
```

**End-to-end differentiability:**
- ✅ Observation -> policy -> cost weights (Phase 3)
- ✅ Cost weights -> solver -> optimal trajectory (Phase 2)
- ✅ Optimal trajectory -> cost evaluation (Phase 1)
- ⚠️ Gradients through solver: Has tracer issues (Phase 2 limitation)

## Usage Examples

### Basic Policy Usage
```python
from policy_network import initialize_policy, create_observation

# Initialize policy
policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

# Create observation
obs = create_observation(
    ego_pos=jnp.array([0.0, 0.0, 1.0]),
    ego_vel=jnp.array([1.0, 0.5, 0.0]),
    adv_pos=jnp.array([5.0, 2.0, 1.0]),
    adv_vel=jnp.array([-0.5, 0.3, 0.0]),
)

# Forward pass
theta = apply_fn(params, obs)
# Returns: {'time_weight': 11.69, 'dyn_weight': 1.19, ...}
```

### Integration with Solver
```python
from mighty_solver_jax import solve_and_extract_trajectory

# Get cost weights from policy
theta = apply_fn(params, obs)

# Solve MIGHTY with policy weights
z_star, trajectory = solve_and_extract_trajectory(
    theta, env_params, z_init, maxiter=50
)

# Access trajectory
positions = trajectory['positions']      # (N, 3)
velocities = trajectory['velocities']    # (N, 3)
```

### Batch Processing
```python
# Process multiple observations
obs_batch = jnp.stack([obs1, obs2, obs3])  # (3, 12)

# Vectorized forward pass
theta_batch = jax.vmap(lambda o: apply_fn(params, o))(obs_batch)

# Each weight has shape (3,)
time_weights = theta_batch['time_weight']  # (3,)
```

### Training Setup (Preview for Phase 4)
```python
import optax

# Define task loss
def task_loss_fn(params, obs, env_params, z_init, adversary_traj):
    # 1. Policy forward: obs -> theta
    theta = apply_fn(params, obs)

    # 2. Solver: theta -> z*
    z_star, traj = solve_and_extract_trajectory(theta, env_params, z_init)

    # 3. Task loss: how well did we evade?
    dists = jnp.linalg.norm(traj['positions'] - adversary_traj, axis=1)
    proximity_cost = jnp.sum(jax.nn.relu(1.0 - dists)**2)
    goal_cost = jnp.linalg.norm(traj['positions'][-1] - env_params['xf'])**2

    return 10.0 * proximity_cost + 1.0 * goal_cost

# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training step (will need workaround for implicit diff issues)
loss, grads = jax.value_and_grad(task_loss_fn)(params, obs, ...)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

## Dependencies

Added for Phase 3:
```bash
pip install flax optax
```

- **Flax**: JAX neural network library (like PyTorch for JAX)
- **Optax**: JAX optimization library (Adam, SGD, etc.)

## Key Features

### 1. Flexible Architecture
- Easy to change hidden dimensions
- Can add more layers
- Can add additional output heads (e.g., goal offset)

### 2. Positive Weight Guarantees
Using `softplus(raw) + base` ensures:
- All weights always positive
- Weights bounded below by base values
- Smooth gradients (no ReLU dead zones)

### 3. Relative Observations
`create_relative_observation()` provides spatial invariance:
```python
obs = [ego_pos, ego_vel, (adv_pos - ego_pos), (adv_vel - ego_vel)]
```
Helps policy generalize across different spatial configurations.

### 4. Batch-Ready
All functions support batched inputs via `jax.vmap`.

## Known Limitations

### 1. Observation Space
Current observations are minimal (12D). Could be extended with:
- Multiple adversaries
- Obstacle information
- Historical states
- Goal information

### 2. Output Representation
Currently outputs only cost weights. Could be extended to:
- Waypoint suggestions
- Direct trajectory parameters
- Time horizon adjustments

### 3. Training Challenges (Phase 4)
- Implicit differentiation through solver has tracer issues (Phase 2)
- May need to use:
  - Finite difference gradients
  - Unrolled differentiation
  - Evolution strategies
  - Reinforcement learning (REINFORCE, PPO)

## Performance

- **Forward pass**: < 1 ms
- **Gradient computation**: ~700 ms (includes solver)
- **Memory**: Minimal (~5KB for 5,447 parameters)
- **Batch processing**: Scales linearly

## Next Steps: Phase 4

With Phase 3 complete, we can proceed to Phase 4 (Training Loop):

1. **Scenario Generation**
   - Sample diverse start/goal positions
   - Sample adversary trajectories
   - Generate static/dynamic obstacles

2. **Task Loss Design**
   - Evasion quality metrics
   - Goal reaching reward
   - Safety margins

3. **Training Strategy**
   - Option A: Supervised learning (if we have expert demonstrations)
   - Option B: Reinforcement learning (policy gradients)
   - Option C: Evolution strategies (gradient-free)

4. **Workarounds for Implicit Diff**
   - Use finite differences through solver
   - Or use REINFORCE-style gradient estimates
   - Or train with unrolled optimization

## Files Summary

```
policy_network.py          13.5 KB    Phase 3 core implementation
policy_network_test.py     18.9 KB    Phase 3 validation suite
```

## Status

**Phase 3 Status**: ✅ **COMPLETE**

All components tested and working:
- ✅ Policy network architecture
- ✅ Forward pass and output validation
- ✅ Batch processing
- ✅ Gradient computation
- ✅ Integration with Phase 2 solver
- ✅ Extended policy with goal offset

Ready for Phase 4: Training loop and task loss implementation.

---

**Date**: 2026-02-02
**Tests Passed**: 6/6
**Total Parameters**: 5,447 (base) / 5,642 (extended)
**Next Phase**: Training Loop
