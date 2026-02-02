# Training Visualization System - User Guide

## Overview

Three new files have been created for training and visualizing the MIGHTY evasion policy:

1. **visualize_training.py** - Training visualization module (3D trajectory plots, metrics)
2. **train_with_viz.py** - Training loop with integrated visualization
3. **run_training.py** - Command-line interface for easy training

## Quick Start

### Basic Usage

```bash
# Quick test (1 episode with visualization)
python run_training.py --episodes 1 --plot-every 1 --num-steps 3

# Short training run (headless, for remote servers)
python run_training.py --episodes 20 --plot-every 5 --num-steps 5

# Longer training (100 episodes)
python run_training.py --episodes 100 --plot-every 10 --num-steps 10
```

### Live Visualization (Local Machine Only)

```bash
# Show live matplotlib window during training
python run_training.py --episodes 20 --live --plot-every 1 --num-steps 5
```

**Note**: `--live` requires a display (X11/Wayland). Use headless mode (without `--live`) on remote servers.

### Disable Visualization (Fastest Training)

```bash
# No visualization, fastest training
python run_training.py --episodes 100 --no-viz --num-steps 10
```

## Command-Line Options

### Training Parameters

- `--episodes N`: Number of training episodes (default: 100)
- `--lr FLOAT`: Learning rate (default: 1e-3)
- `--adversary-level {1,2}`: Adversary difficulty
  - **Level 1**: Straight line (constant velocity) - easier to evade
  - **Level 2**: Proportional navigation (pursues ego) - harder to evade
- `--num-steps N`: Solver iterations per episode (default: 30)
  - Lower = faster but less accurate (try 5-10 for testing)
  - Higher = slower but more accurate (20-50 for production)
- `--seed N`: Random seed (default: 42)

### Visualization Parameters

- `--live`: Show live matplotlib window (requires display)
- `--plot-every N`: Visualize every N episodes (default: 10)
- `--no-viz`: Disable visualization entirely (faster)

### Output Parameters

- `--save-dir PATH`: Directory to save results (default: training_results)

## Output Files

After training, you'll find:

```
training_results/
├── final_policy.pkl              # Trained policy network parameters
├── checkpoint_0100.pkl           # Checkpoints every 100 episodes
├── checkpoint_0200.pkl
├── visualizations/
│   ├── episode_0000.png          # 3D trajectory plots per episode
│   ├── episode_0010.png
│   ├── episode_0020.png
│   ├── training_curves.png       # Loss/distance curves (updated every 50 episodes)
│   └── training_history.json     # Full training metrics
```

### Visualization Outputs

#### Trajectory Plots (episode_NNNN.png)
Each trajectory plot shows:
- **Blue line**: Ego trajectory (with arrow showing direction)
- **Red line**: Adversary trajectory (with arrow)
- **Blue circle**: Ego start position
- **Red circle**: Adversary start position
- **Green star**: Goal position
- **Black dashed line**: Minimum distance between ego and adversary
- **Title**: Episode number, loss, min distance, goal distance

#### Training Curves (training_curves.png)
Four subplots:
1. Training loss over episodes
2. Minimum ego-adversary distance (with 1m safety margin line)
3. Final distance to goal
4. Optimizer convergence rate (rolling window)

#### Training History (training_history.json)
Complete metrics for all episodes:
- Episode numbers
- Loss values
- Minimum distances
- Goal distances
- Convergence status
- Network outputs (theta values)

## Performance Notes

### Timing

Training is relatively slow due to gradient estimation:
- **~3-5 minutes per episode** with `--num-steps 3-5` (quick)
- **~10-15 minutes per episode** with `--num-steps 20-30` (quality)

**Why so slow?**
- Each episode requires 7-8 forward passes (finite difference gradient estimation on 7 cost weights)
- Each forward pass runs the MIGHTY solver (L-BFGS optimization)
- This is a known limitation from Phase 4 - implicit differentiation through the solver has JAX tracer issues

### Recommended Settings

**For Quick Testing:**
```bash
python run_training.py --episodes 10 --num-steps 3 --plot-every 2
# ~30-50 minutes total
```

**For Quality Training:**
```bash
python run_training.py --episodes 100 --num-steps 10 --plot-every 10
# ~8-16 hours total
```

**Overnight Training:**
```bash
python run_training.py --episodes 500 --num-steps 20 --plot-every 20 --no-viz
# ~1-2 days total
```

## Training Progress

During training, you'll see output like:

```
Episode    0 | Loss: 2217.716 (avg: 2217.716) | Min dist:  2.18m (avg:  2.18m) | Goal dist: 47.12m (avg: 47.12m) | Conv: True
Episode   10 | Loss: 1845.352 (avg: 1923.234) | Min dist:  3.42m (avg:  2.89m) | Goal dist: 42.85m (avg: 44.32m) | Conv: True
Episode   20 | Loss: 1523.188 (avg: 1672.145) | Min dist:  4.91m (avg:  3.67m) | Goal dist: 38.12m (avg: 40.15m) | Conv: True
```

### Interpreting Metrics

- **Loss**: Task loss (lower is better)
  - Loss = -min_dist + goal_weight * goal_dist² + proximity_penalty
  - Negative loss means the policy is maintaining good distance from adversary
- **Min dist**: Minimum distance between ego and adversary (higher is better)
  - Target: > 1.0m (safety margin)
  - Good: > 2.0m
  - Excellent: > 5.0m
- **Goal dist**: Final distance from ego to goal (lower is better)
  - Target: < 1.0m
  - Good: < 0.5m
  - Excellent: < 0.1m
- **Conv**: Whether solver converged (should be True)

### Training Expectations

Good training should show:
- **Loss decreasing** over episodes
- **Min dist increasing** (ego staying farther from adversary)
- **Goal dist decreasing** (ego reaching goal more accurately)

If loss is increasing or not improving after 50+ episodes:
- Try reducing learning rate: `--lr 5e-4` or `--lr 1e-4`
- Try increasing solver steps: `--num-steps 20` or `--num-steps 30`
- Check visualizations to see what's happening

## Loading Trained Model

```python
import pickle
from policy_network import EvasionPolicy

# Load trained parameters
with open('training_results/final_policy.pkl', 'rb') as f:
    policy_params = pickle.load(f)

# Use with policy network
policy = EvasionPolicy(hidden_dim=64)
theta = policy.apply(policy_params, observation)  # Get cost weights for MIGHTY
```

## Integration with Existing Code

The training system uses:
- **policy_network.py** (Phase 3): Neural network policy
- **mighty_solver_jax.py** (Phase 2): MIGHTY optimizer
- **mighty_jax.py** (Phase 1): Cost functions
- **environments.py** (Phase 4): Scenario generation

All phases are integrated and working together.

## Troubleshooting

### "No module named 'mpl_toolkits.mplot3d'"
matplotlib version conflict. Try:
```bash
pip install --upgrade matplotlib
```

### Training very slow
- Reduce `--num-steps` (try 3-5 for testing)
- Use `--no-viz` to disable visualization overhead
- This is expected - each episode requires multiple solver runs

### Out of memory
- Reduce `--num-steps`
- Run fewer episodes at a time
- Use checkpoints (saved every 100 episodes)

### Matplotlib backend errors (on remote server)
Use headless mode (don't use `--live` flag):
```bash
python run_training.py --episodes 100  # Saves plots to files
```

### Want faster gradients?
The current implementation uses finite differences for gradient estimation. Potential improvements:
1. Use evolution strategies (ES) instead of gradient descent
2. Implement custom VJP for implicit differentiation
3. Use a simpler differentiable solver (not L-BFGS)
4. Train with GPU (requires CUDA-enabled JAX)

## Advanced: Custom Training Loop

You can also use the training functions directly in Python:

```python
from train_with_viz import train
from visualize_training import TrainingVisualizer

# Custom training
policy_params, visualizer = train(
    num_episodes=50,
    learning_rate=1e-3,
    adversary_level=1,
    save_dir='my_results',
    visualize=True,
    live_viz=False,
    plot_every=5,
    num_steps=10,
    seed=42
)

# Access training history
history = visualizer.history
print(f"Final loss: {history['losses'][-1]:.3f}")
print(f"Best min distance: {max(history['min_distances']):.2f}m")
```

## Examples

### Example 1: Quick Test Run
```bash
python run_training.py --episodes 5 --num-steps 3 --plot-every 1
```
Expected time: ~15-25 minutes
Use this to verify everything works.

### Example 2: Short Training Session
```bash
python run_training.py --episodes 50 --num-steps 5 --plot-every 10
```
Expected time: ~3-4 hours
Good for initial policy learning.

### Example 3: Production Training
```bash
python run_training.py --episodes 200 --num-steps 20 --plot-every 20 --lr 5e-4
```
Expected time: ~1-2 days
For best quality policy.

### Example 4: Adversary Level 2 (Harder)
```bash
python run_training.py --episodes 100 --num-steps 10 --adversary-level 2
```
Train against a pursuit adversary (more challenging).

## Next Steps

After training:
1. **Export model** for C++ deployment (use Phase 5 tools):
   ```python
   from export_model import export_to_onnx, export_to_cpp_header
   export_to_onnx(policy_params, "evasion_policy.onnx")
   export_to_cpp_header(policy_params, "policy_weights.h")
   ```

2. **Evaluate model** on test scenarios:
   - Sample random scenarios
   - Run policy inference
   - Visualize results

3. **Deploy to robot**:
   - Load ONNX model in C++
   - Integrate with existing MIGHTY C++ code
   - Test on hardware

## Summary

You now have a complete training and visualization system for MIGHTY evasion policies:

✅ 3D trajectory visualization (ego vs adversary)
✅ Real-time or saved plots
✅ Comprehensive metrics logging
✅ Training curves and history
✅ Checkpoint saving every 100 episodes
✅ Easy command-line interface
✅ Headless mode for remote servers
✅ Live mode for local development

The system is ready to use. Start with a quick test run to verify everything works!
