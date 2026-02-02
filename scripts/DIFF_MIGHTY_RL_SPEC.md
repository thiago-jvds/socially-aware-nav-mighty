# Differentiable MIGHTY + RL for Adversarial Evasion — Implementation Spec

## Project Summary

Build a differentiable training pipeline where a neural network learns to parameterize the MIGHTY trajectory optimizer for adversarial pursuit-evasion. Training is in Python/JAX. Deployment is C++ MIGHTY + exported NN (ONNX).

**Two-phase architecture:**
- **Training (Python/JAX):** NN outputs cost weights/evasion params → JAX-ified MIGHTY optimizes trajectory → task loss → backprop through everything → update NN
- **Deployment (C++):** NN forward pass (ONNX Runtime) → C++ MIGHTY plans → motor commands. No Python at runtime.

---

## Existing Codebase

The file `test_unconstrained_opt_3d.ipynb` contains a complete Python implementation of MIGHTY. Key components:

### Decision Vector
```
z = [free_CPs (F×3), σ_slacks (M)]
```
where:
- `free_CPs`: F free Bézier control points, each 3D → F×3 floats
- `σ_slacks`: M segment time slacks, where `T[s] = exp(σ[s])`
- For 3 segments with 6 free CPs: z has length 6×3 + 3 = 21

### Core Functions (NumPy, already implemented)
- `reconstruct(z, x0, v0, a0, xf, vf, af, wps)` → `(CP, T)` — builds (M,6,3) control points and (M,) durations from z
- `evaluate_objective(z)` → scalar cost — weighted sum of J_time, J_dyn, J_stat, J_acc, J_jerk, J_vel_constr, J_acc_constr
- `compute_analytical_grad(z)` → gradient vector — hand-coded analytical gradients
- `lbfgs_optimize(z0, func, grad, params)` → optimized z
- `tron_lbfgs_optimize(z0, func, grad, params)` → optimized z (trust-region variant)

### JAX Functions (partially implemented, used for gradient testing)
- `reconstruct_jax(z, x0, v0, a0, xf, vf, af, wps)` → `(CP, T)` ✅ DONE
- `eval_traj_jax(CP, T, N_samp)` → positions ✅ DONE
- `f_obs_poly_jax(t, cx, cy, cz)` → obstacle positions ✅ DONE
- Individual cost-term JAX functions for gradient testing (J_time, J_dyn, J_jerk, J_vel, J_acc) ✅ DONE but as separate test functions

### Cost Function Weights (current defaults)
```python
time_weight = 10.0
dyn_weight = 0.1
stat_weight = 0.1
accel_weight = 0.1
jerk_weight = 0.1
dyn_constr_vel_weight = 10.0
dyn_constr_acc_weight = 1.0
```

### Other Key Parameters
```python
V_max = 2.0          # max velocity
A_max = 100.0        # max acceleration
N_samp = 10          # trajectory sample points
Co = 0.5             # static obstacle clearance
Cw = 1.0             # dynamic obstacle clearance
c = 2.0              # ellipsoid z-axis factor
```

### C++ Implementation
A separate C++ MIGHTY implementation exists and has been cross-validated against the Python version (Cell 9 shows numerical comparison). The C++ version is the deployment target and should NOT be modified for training.

---

## Phase 1: JAX Differentiable MIGHTY Cost Function

**Goal:** Create a single JAX-differentiable `evaluate_objective_jax(z, theta)` that takes both the decision vector z AND neural network output theta, so `jax.grad` can differentiate through the entire pipeline.

### File: `mighty_jax.py`

Port the cost function from NumPy to JAX. The existing `reconstruct_jax` and `eval_traj_jax` are already done. You need to assemble them into a complete differentiable cost.

```python
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

def evaluate_objective_jax(z, theta, env_params):
    """
    Fully differentiable MIGHTY cost function.
    
    Args:
        z: (K,) decision vector [free_CPs, σ_slacks]
        theta: dict of NN outputs that parameterize the cost:
            'time_weight': scalar
            'dyn_weight': scalar  
            'stat_weight': scalar
            'accel_weight': scalar
            'jerk_weight': scalar
            'vel_constr_weight': scalar
            'acc_constr_weight': scalar
            # Optional future extensions:
            # 'goal_position': (3,) evasion waypoint
            # 'clearance': scalar Cw
        env_params: dict of fixed environment parameters:
            'x0', 'v0', 'a0': (3,) start boundary conditions
            'xf', 'vf', 'af': (3,) end boundary conditions
            'wps': (M+1, 3) global waypoints
            'free_cps_indices': list of (seg, cp_idx) tuples
            'obstacles': list of obstacle dicts
            'A_stat', 'b_stat': static constraint planes
            'V_max', 'A_max': dynamic limits
            'N_samp': int
            'Co', 'Cw', 'c_ellipsoid': clearance params
    
    Returns:
        scalar cost J(z, theta)
    """
    # Unpack env
    x0, v0, a0 = env_params['x0'], env_params['v0'], env_params['a0']
    xf, vf, af = env_params['xf'], env_params['vf'], env_params['af']
    wps = env_params['wps']
    
    # Reconstruct (already JAX-ified)
    CP, T = reconstruct_jax(z, x0, v0, a0, xf, vf, af, wps)
    
    # Compute each cost term (all in JAX)
    J_time = jnp.sum(T)
    J_dyn = compute_J_dyn_jax(CP, T, env_params)
    J_stat = compute_J_stat_jax(CP, env_params)
    J_acc = compute_J_acc_jax(CP, T)
    J_jerk = compute_J_jerk_jax(CP, T)
    J_vel_constr = compute_J_vel_constr_jax(CP, T, env_params)
    J_acc_constr = compute_J_acc_constr_jax(CP, T, env_params)
    
    # Weighted sum using NN-provided weights
    return (theta['time_weight'] * J_time
          + theta['dyn_weight'] * J_dyn
          + theta['stat_weight'] * J_stat
          + theta['accel_weight'] * J_acc
          + theta['jerk_weight'] * J_jerk
          + theta['vel_constr_weight'] * J_vel_constr
          + theta['acc_constr_weight'] * J_acc_constr)
```

### Individual JAX Cost Terms to Implement

Each of these already exists in NumPy in Cell 6. Port them to JAX operations. Reference the existing JAX test functions in Cells 10-23 for patterns.

**1. `compute_J_jerk_jax(CP, T)`**
```python
def compute_J_jerk_jax(CP, T):
    """J_jerk = sum_s 3600/T[s]^5 * sum_{m=0}^2 ||Delta3P[s,m]||^2"""
    M = CP.shape[0]
    J = 0.0
    for s in range(M):
        for m in range(3):
            D3 = CP[s, m+3] - 3*CP[s, m+2] + 3*CP[s, m+1] - CP[s, m]
            J += (3600.0 / T[s]**5) * jnp.dot(D3, D3)
    return J
```

**2. `compute_J_acc_jax(CP, T)`**
```python
def compute_J_acc_jax(CP, T):
    """J_acc = sum_s 400/T[s]^3 * sum_{m=0}^3 ||Delta2P[s,m]||^2"""
    M = CP.shape[0]
    J = 0.0
    for s in range(M):
        for m in range(4):
            D2 = CP[s, m+2] - 2*CP[s, m+1] + CP[s, m]
            J += (400.0 / T[s]**3) * jnp.dot(D2, D2)
    return J
```

**3. `compute_J_dyn_jax(CP, T, env_params)`** — dynamic obstacle cost
- Sample ego trajectory at N_samp points
- For each obstacle, compute polynomial trajectory
- Ellipsoid distance metric with penalty `max(Cw^2 - d^2, 0)^3`
- Use `jnp.clip` for the max operation (differentiable)

**4. `compute_J_stat_jax(CP, env_params)`** — static obstacle cost
- Loop over control points and half-space planes
- Cubic penalty: `max(Co - h, 0)^3` where `h = A·P - b`
- Use `jnp.clip` for the max

**5. `compute_J_vel_constr_jax(CP, T, env_params)`** — velocity violation
- Sample velocities at N_samp points using `eval_traj_and_derivs_jax`
- Penalty: `max(||v|| - V_max, 0)^3`

**6. `compute_J_acc_constr_jax(CP, T, env_params)`** — acceleration violation
- Same pattern as velocity but for acceleration norm

### Critical Implementation Notes for JAX Porting

1. **No Python control flow on JAX arrays.** Replace `if h > 0` with `jnp.where(h > 0, ..., 0.0)`. Replace `np.clip(..., 0, None)` with `jnp.clip(..., 0.0)` or `jax.nn.relu(...)`.

2. **Fixed-size loops are fine.** `for s in range(M)` where M is a Python int (not a JAX array) compiles fine under `jax.jit`. The loop gets unrolled at trace time.

3. **`jnp.searchsorted` is supported** but has limited grad support. For `eval_traj_jax`, you already handle this — the segment lookup uses `jnp.searchsorted` which works in forward but the grad flows through the Bernstein evaluation, not the indexing.

4. **Validate against NumPy.** For each JAX cost term, compare its output and JAX-computed gradient against the NumPy version and the hand-coded analytical gradient. The notebook already has this pattern in Cells 11-23. Every cost term should match to ~1e-6 relative error.

5. **`jax.jit` the full objective.** Once assembled, `jax.jit(evaluate_objective_jax)` should compile. Test that `jax.grad(evaluate_objective_jax)(z, theta, env_params)` returns finite gradients.

### File: `mighty_jax_test.py`

Validation script that compares JAX cost/gradient against NumPy cost/analytical gradient:

```python
def test_cost_agreement():
    """JAX cost should match NumPy cost to high precision."""
    z0 = ...  # test decision vector
    J_numpy = evaluate_objective(z0)  # from notebook
    J_jax = float(evaluate_objective_jax(jnp.array(z0), theta_default, env_params))
    assert abs(J_numpy - J_jax) / (abs(J_numpy) + 1e-12) < 1e-10

def test_gradient_agreement():
    """JAX autodiff gradient should match hand-coded analytical gradient."""
    z0 = ...
    g_analytical = compute_analytical_grad(z0)  # from notebook
    g_jax = jax.grad(evaluate_objective_jax)(jnp.array(z0), theta_default, env_params)
    rel_err = np.abs(g_analytical - np.array(g_jax)) / (np.abs(g_analytical) + 1e-12)
    assert rel_err.max() < 1e-5
```

---

## Phase 2: Differentiable Inner-Loop Optimizer

**Goal:** Replace the Python L-BFGS with a JAX-native optimizer that JAX can differentiate through.

### Option A: `jaxopt.LBFGS` with Implicit Differentiation (RECOMMENDED)

```python
import jaxopt

def solve_mighty_jax(theta, env_params, z_init):
    """
    Solve MIGHTY trajectory optimization in JAX.
    Returns optimal z* that is differentiable w.r.t. theta.
    """
    def cost_fn(z):
        return evaluate_objective_jax(z, theta, env_params)
    
    solver = jaxopt.LBFGS(
        fun=cost_fn,
        maxiter=50,        # match C++ iteration budget
        tol=1e-5,
        implicit_diff=True  # KEY: enables implicit differentiation
    )
    result = solver.run(z_init)
    return result.params  # z* that is differentiable w.r.t. theta
```

With `implicit_diff=True`, JAXopt automatically computes dz*/dtheta using the implicit function theorem at the solution. No unrolling, no Hessian storage. This is exactly the approach from Blondel et al. (NeurIPS 2022, "Efficient and Modular Implicit Differentiation").

### Option B: Unrolled Gradient Descent (simpler, more memory)

```python
def solve_mighty_unrolled(theta, env_params, z_init, num_steps=20, lr=0.01):
    """Fixed number of gradient descent steps, fully unrolled for autodiff."""
    z = z_init
    for _ in range(num_steps):
        g = jax.grad(evaluate_objective_jax)(z, theta, env_params)
        z = z - lr * g
    return z
```

Simpler but uses more memory (stores all intermediate states for backprop). Start with Option A.

---

## Phase 3: Neural Network Policy

**Goal:** NN observes adversary state, outputs cost weights for MIGHTY.

### File: `policy_network.py`

```python
import jax
import jax.numpy as jnp
import flax.linen as nn  # or use equinox

class EvasionPolicy(nn.Module):
    """
    Maps observation → MIGHTY cost weights.
    
    Observation: [ego_pos(3), ego_vel(3), adversary_pos(3), adversary_vel(3)] = 12D
    Output: cost weights (7 values, all positive via softplus)
    """
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        raw_weights = nn.Dense(7)(x)
        
        # Ensure positive weights via softplus, with sensible base values
        # Base values are the manual defaults from the notebook
        base = jnp.array([10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0])
        weights = jax.nn.softplus(raw_weights) + base
        
        return {
            'time_weight': weights[0],
            'dyn_weight': weights[1],
            'stat_weight': weights[2],
            'accel_weight': weights[3],
            'jerk_weight': weights[4],
            'vel_constr_weight': weights[5],
            'acc_constr_weight': weights[6],
        }
```

**Future extension:** NN also outputs evasion goal point or intermediate waypoints. Add a second head:
```python
goal_offset = nn.Dense(3)(x)  # offset from current goal
theta['xf'] = env_params['xf'] + 0.5 * jnp.tanh(goal_offset)  # bounded offset
```

---

## Phase 4: Training Loop

### File: `train.py`

```python
import jax
import jax.numpy as jnp
import optax  # JAX optimizer for NN

def task_loss(tau_star, adversary_traj, env_params):
    """
    Evasion task loss: penalize proximity to adversary, reward reaching goal.
    
    tau_star: optimized ego trajectory (N_samp, 3)
    adversary_traj: predicted adversary trajectory (N_samp, 3)
    """
    # Minimize negative distance to adversary (= maximize distance)
    dists = jnp.linalg.norm(tau_star - adversary_traj, axis=1)
    min_dist = jnp.min(dists)
    
    # Penalize getting too close
    proximity_cost = jnp.sum(jax.nn.relu(1.0 - dists)**2)
    
    # Reward reaching goal
    goal_cost = jnp.linalg.norm(tau_star[-1] - env_params['xf'])**2
    
    # Smooth trajectory cost (already handled by MIGHTY, but can add here)
    return -min_dist + 10.0 * proximity_cost + 1.0 * goal_cost


def train_step(nn_params, obs, env_params, z_init, adversary_traj, opt_state, optimizer):
    """Single training step with backprop through MIGHTY."""
    
    def loss_fn(nn_params):
        # 1. NN forward pass: obs → cost weights
        policy = EvasionPolicy()
        theta = policy.apply(nn_params, obs)
        
        # 2. MIGHTY solve: cost weights → optimal trajectory
        z_star = solve_mighty_jax(theta, env_params, z_init)
        
        # 3. Extract trajectory from z*
        CP, T = reconstruct_jax(z_star, env_params['x0'], env_params['v0'],
                                 env_params['a0'], env_params['xf'],
                                 env_params['vf'], env_params['af'],
                                 env_params['wps'])
        tau_star = eval_traj_jax(CP, T, env_params['N_samp'])
        
        # 4. Compute task loss
        return task_loss(tau_star, adversary_traj, env_params)
    
    # Backprop through everything: task_loss → trajectory → MIGHTY solve → NN
    loss, grads = jax.value_and_grad(loss_fn)(nn_params)
    updates, opt_state = optimizer.update(grads, opt_state, nn_params)
    nn_params = optax.apply_updates(nn_params, updates)
    
    return nn_params, opt_state, loss


# Main training loop
def train():
    # Initialize
    policy = EvasionPolicy()
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros(12)
    nn_params = policy.init(rng, dummy_obs)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nn_params)
    
    for episode in range(num_episodes):
        # Sample random scenario
        env_params = sample_scenario(rng)
        obs = get_observation(env_params)
        z_init = get_initial_guess(env_params)
        adversary_traj = simulate_adversary(env_params)
        
        nn_params, opt_state, loss = train_step(
            nn_params, obs, env_params, z_init, adversary_traj, opt_state, optimizer
        )
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
    
    return nn_params
```

---

## Phase 5: Export NN for C++ Deployment

### File: `export_model.py`

After training, export the NN weights for C++ inference. Two options:

### Option A: ONNX Export (Recommended)

```python
import jax
import jax.numpy as jnp
# Convert Flax model to ONNX via jax2onnx or via PyTorch bridge

# Approach: save weights as numpy, reload in PyTorch, export ONNX
def export_to_onnx(nn_params, filepath="evasion_policy.onnx"):
    import torch
    import torch.nn as torch_nn
    
    # Rebuild equivalent PyTorch model
    class PolicyTorch(torch_nn.Module):
        def __init__(self, params):
            super().__init__()
            self.fc1 = torch_nn.Linear(12, 64)
            self.fc2 = torch_nn.Linear(64, 64)
            self.fc3 = torch_nn.Linear(64, 7)
            # Load JAX weights into PyTorch layers
            self._load_jax_params(params)
        
        def forward(self, obs):
            x = torch.relu(self.fc1(obs))
            x = torch.relu(self.fc2(x))
            raw = self.fc3(x)
            base = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0])
            return torch.nn.functional.softplus(raw) + base
        
        def _load_jax_params(self, params):
            # Extract from Flax param dict and set as torch tensors
            pass  # implement based on actual param structure
    
    model = PolicyTorch(nn_params)
    model.eval()
    dummy = torch.randn(1, 12)
    torch.onnx.export(model, dummy, filepath, input_names=['obs'], output_names=['weights'])
```

### Option B: Raw Weight Export (Zero-dependency deployment)

```python
def export_weights_cpp(nn_params, filepath="policy_weights.h"):
    """Export as C++ header with Eigen matrices."""
    # Extract weight matrices from Flax params
    # Write as constexpr Eigen::Matrix definitions
    # The C++ side just does:
    #   h = (W1 * obs + b1).cwiseMax(0);  // ReLU
    #   h = (W2 * h + b2).cwiseMax(0);
    #   raw = W3 * h + b3;
    #   weights = softplus(raw) + base;
    pass
```

### C++ Integration Pattern

```cpp
// In your existing MIGHTY C++ ROS2 node:

#include <onnxruntime_cxx_api.h>

class EvasionPlanner {
    Ort::Session policy_session_;
    MightyOptimizer mighty_;

public:
    void plan(const State& ego, const State& adversary) {
        // 1. Build observation vector
        Eigen::VectorXf obs(12);
        obs << ego.pos, ego.vel, adversary.pos, adversary.vel;
        
        // 2. NN forward pass (~0.1ms)
        auto weights = run_policy(obs);  // ONNX inference
        
        // 3. Set MIGHTY cost weights from NN output
        mighty_.setTimeWeight(weights[0]);
        mighty_.setDynWeight(weights[1]);
        mighty_.setStatWeight(weights[2]);
        mighty_.setAccelWeight(weights[3]);
        mighty_.setJerkWeight(weights[4]);
        mighty_.setVelConstrWeight(weights[5]);
        mighty_.setAccConstrWeight(weights[6]);
        
        // 4. Run C++ MIGHTY as normal (~5ms)
        auto trajectory = mighty_.optimize(z_init);
        
        // 5. Execute first control point
        publish_command(trajectory);
    }
};
```

---

## Implementation Order

Execute these in order. Each phase should be testable independently.

### Step 1: `mighty_jax.py` — Port cost function to JAX
- Copy `reconstruct_jax`, `eval_traj_jax`, `f_obs_poly_jax` from notebook
- Implement each cost term in JAX
- Write validation tests comparing against NumPy
- Test: `jax.grad(evaluate_objective_jax)(z0, theta, env)` returns finite values

### Step 2: `mighty_jax_test.py` — Validate
- Cost value agreement: JAX vs NumPy < 1e-10 relative error
- Gradient agreement: JAX autodiff vs analytical grad < 1e-5 relative error
- JIT compilation: `jax.jit(evaluate_objective_jax)` compiles without error

### Step 3: `mighty_solver_jax.py` — Inner-loop optimizer
- Wrap `evaluate_objective_jax` with `jaxopt.LBFGS(implicit_diff=True)`
- Test: solving returns same z* as NumPy L-BFGS (to optimizer tolerance)
- Test: `jax.grad(solve_mighty_jax)(theta, env, z_init)` returns finite dz*/dtheta

### Step 4: `policy_network.py` — NN architecture
- Simple MLP: 12 → 64 → 64 → 7 (with softplus output)
- Test: forward pass produces reasonable cost weights

### Step 5: `train.py` — Training loop
- Single `train_step` with full backprop chain
- Start with simple scenarios (single adversary, straight-line pursuit)
- Curriculum: increase adversary speed/intelligence over training

### Step 6: `export_model.py` — ONNX export
- Export trained model
- Validate: ONNX model output matches JAX model output

### Step 7: C++ integration
- Add ONNX Runtime to MIGHTY CMakeLists.txt
- Load model, run inference, set weights, plan
- This is the only step that touches C++ code

---

## Dependencies

```bash
# Python (training)
pip install jax jaxlib jaxopt flax optax numpy

# For ONNX export
pip install torch onnx onnxruntime

# For testing
pip install pytest
```

---

## Key Technical Decisions

1. **Why JAX, not PyTorch?** The notebook already has JAX versions of the core functions (reconstruct_jax, eval_traj_jax). JAXopt provides implicit differentiation through optimizers out of the box. JAX's functional style also matches the mathematical structure of the cost function better.

2. **Why `jaxopt.LBFGS` with `implicit_diff=True`?** This uses the implicit function theorem at the optimizer solution to compute dz*/dtheta without storing the full optimization trajectory. Memory-efficient, mathematically clean, and well-tested. This is the approach from Blondel et al. (NeurIPS 2022).

3. **Why not differentiate through the custom L-BFGS in the notebook?** The custom L-BFGS (Cell 6) uses Python loops with NumPy — it can't be JAX-traced. We'd need to rewrite it in JAX anyway, at which point `jaxopt.LBFGS` is strictly better since it's already optimized and supports implicit diff.

4. **Why ONNX for deployment?** Zero Python dependency at runtime. ONNX Runtime C++ is lightweight, fast, and battle-tested in robotics. A 12→64→64→7 MLP inference takes <0.1ms.

5. **NN outputs cost weights, not raw actions.** This preserves all of MIGHTY's constraint-handling, feasibility guarantees, and trajectory smoothness. The NN only modulates HOW the optimizer trades off objectives — it can't produce physically infeasible trajectories.

---

## Adversary Model for Training

Start simple and escalate:

### Level 1: Constant-velocity adversary
Adversary flies straight toward ego's initial goal at constant speed. The NN learns basic evasion.

### Level 2: Proportional-navigation adversary
Adversary uses PN guidance (classical missile guidance) — aims to nullify line-of-sight rate. More realistic.

### Level 3: RL adversary (self-play)
Train a second NN that controls the adversary using the same architecture. Alternate training (ego improves against frozen adversary, then adversary improves against frozen ego). This produces robust evasion policies.

---

## File Structure

```
diff_mighty_rl/
├── mighty_jax.py          # Phase 1: JAX cost function
├── mighty_jax_test.py     # Phase 1: Validation tests
├── mighty_solver_jax.py   # Phase 2: jaxopt L-BFGS wrapper
├── policy_network.py      # Phase 3: Flax NN
├── train.py               # Phase 4: Training loop
├── export_model.py        # Phase 5: ONNX export
├── environments.py        # Scenario sampling, adversary models
├── utils.py               # Shared helpers
└── configs/
    └── default.yaml       # Hyperparameters
```

---

## References

- Blondel et al., "Efficient and Modular Implicit Differentiation" (NeurIPS 2022) — the theory behind jaxopt implicit_diff
- Amos et al., "Differentiable MPC for End-to-end Planning and Control" (NeurIPS 2018) — foundational differentiable MPC
- AC-MPC (Romero et al., 2024) — actor-critic with differentiable MPC for drone racing, closest architecture
- DiffTORI (Wan et al., NeurIPS 2024) — differentiable trajectory optimization for RL, state of the art
