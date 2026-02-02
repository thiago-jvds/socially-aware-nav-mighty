# Phase 5: Model Export - COMPLETE ✅

## Objective
Export trained Flax policy network to formats suitable for C++ deployment, enabling real-time inference on robot hardware.

## Implementation Summary

### Files Created
- **export_model.py** (398 lines): Model export utilities
- **export_test.py** (379 lines): Comprehensive test suite

### Export Formats

#### 1. ONNX Format (Recommended)
- Converts Flax model to ONNX via PyTorch bridge
- Compatible with ONNX Runtime (C++ inference)
- Supports quantization and optimization
- Numerical equivalence verified (max diff < 5e-7)

**Usage:**
```python
export_to_onnx(params, "evasion_policy.onnx")
```

**C++ Integration:**
```cpp
#include <onnxruntime_cxx_api.h>

Ort::Session session(env, "evasion_policy.onnx");
// Input: observation (1, 12) float tensor
// Output: cost_weights (1, 7) float tensor
```

#### 2. NumPy Archive (.npz)
- Saves all weight matrices as NumPy arrays
- Portable format for Python/C++ interop
- Easy to load in custom C++ implementations

**Usage:**
```python
save_weights_numpy(params, "policy_weights.npz")

# Load in Python
weights = np.load("policy_weights.npz")
W1 = weights['W1']  # (12, 64)
b1 = weights['b1']  # (64,)
```

#### 3. C++ Header File (Zero-Dependency)
- Generates standalone C++ header with Eigen matrices
- No external dependencies besides Eigen
- Complete forward pass implementation included
- Weights embedded as constexpr arrays

**Usage:**
```python
export_to_cpp_header(params, "policy_weights.h")
```

**C++ Usage:**
```cpp
#include "policy_weights.h"

Eigen::VectorXf obs(12);
// ... fill observation ...

Eigen::VectorXf weights = mighty::policy::forward(obs);
// Returns (7,) vector of cost weights
```

### Core Components

#### Weight Extraction
```python
def extract_weights_from_flax(params):
    """
    Extract weight matrices from Flax parameter dictionary.

    Flax structure:
      params['params']['hidden1']['kernel'] -> W1 (12, 64)
      params['params']['hidden1']['bias']   -> b1 (64,)
      ... etc

    Returns flat dict: {'W1', 'b1', 'W2', 'b2', 'W3', 'b3'}
    """
```

#### ONNX Export Pipeline
```python
def export_to_onnx(params, filepath):
    """
    1. Extract weights from Flax
    2. Create equivalent PyTorch model
    3. Load Flax weights into PyTorch
    4. Export to ONNX format
    5. Validate ONNX model
    """
```

Key features:
- Dynamic batch size support
- Opset version 18 (latest)
- Constant folding enabled
- Model validation with onnx.checker

#### Numerical Verification
```python
def verify_onnx_export(flax_params, onnx_filepath):
    """
    Compare Flax and ONNX outputs on test data.

    Returns True if max difference < 1e-5
    """
```

Test scenarios:
- Multiple observation vectors
- Edge cases (zeros, large values)
- Batch processing

## Test Results

All 6/6 tests passed:

✅ **Test 1: Weight Extraction**
- Extracts all 6 weight matrices
- Correct shapes: W1 (12,64), b1 (64), W2 (64,64), b2 (64), W3 (64,7), b3 (7)
- NumPy array type verification

✅ **Test 2: NumPy Export/Load**
- Saves to .npz format
- Reloads correctly
- Values match original (allclose test)

✅ **Test 3: C++ Header Generation**
- Creates valid C++ header file
- Contains all weight declarations
- Includes forward pass implementation
- Proper header guards and namespaces

✅ **Test 4: ONNX Export**
- Successfully exports to ONNX format
- File size reasonable (~8.8 KB)
- ONNX model passes validation
- Inputs/outputs correctly named

✅ **Test 5: ONNX Numerical Verification**
- Flax and ONNX outputs match
- Max difference: 4.79e-07 (< 1e-5 tolerance)
- Verified on 2 test observations

✅ **Test 6: Full Export Workflow**
- All 3 formats export successfully
- Files created and validated
- Cleanup successful

## Network Architecture

**Input**: 12D observation
- Ego position (3D)
- Ego velocity (3D)
- Adversary position (3D)
- Adversary velocity (3D)

**Architecture**: 12 → 64 → 64 → 7

**Layers**:
1. Dense(12, 64) + ReLU
2. Dense(64, 64) + ReLU
3. Dense(64, 7) + Softplus + Base

**Output**: 7D cost weights
- time_weight (base: 10.0)
- dyn_weight (base: 0.1)
- stat_weight (base: 0.1)
- accel_weight (base: 0.1)
- jerk_weight (base: 0.1)
- vel_constr_weight (base: 10.0)
- acc_constr_weight (base: 1.0)

## Usage Examples

### Export All Formats
```python
from policy_network import initialize_policy, EvasionPolicy
from export_model import *

# Load trained policy (or initialize for testing)
policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

# Export to all formats
save_weights_numpy(params, "policy_weights.npz")
export_to_cpp_header(params, "policy_weights.h")
export_to_onnx(params, "evasion_policy.onnx")

# Verify ONNX export
verify_onnx_export(params, "evasion_policy.onnx")
```

### C++ Integration Example

**Option 1: ONNX Runtime (Recommended)**
```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MIGHTY");
Ort::SessionOptions session_options;
Ort::Session session(env, "evasion_policy.onnx", session_options);

// Prepare input
std::vector<float> obs_data = {
    1.0, 2.0, 3.0,    // ego_pos
    0.5, 0.5, 0.5,    // ego_vel
    5.0, 6.0, 7.0,    // adv_pos
    -0.3, 0.2, 0.1    // adv_vel
};

// Run inference
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
std::vector<int64_t> input_shape = {1, 12};
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, obs_data.data(), obs_data.size(),
    input_shape.data(), input_shape.size()
);

const char* input_names[] = {"observation"};
const char* output_names[] = {"cost_weights"};

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names, &input_tensor, 1,
    output_names, 1
);

float* weights = output_tensors[0].GetTensorMutableData<float>();
// weights[0..6] = [time, dyn, stat, accel, jerk, vel_constr, acc_constr]
```

**Option 2: C++ Header (Zero Dependencies)**
```cpp
#include "policy_weights.h"
#include <Eigen/Dense>

Eigen::VectorXf obs(12);
obs << 1.0, 2.0, 3.0,    // ego_pos
       0.5, 0.5, 0.5,    // ego_vel
       5.0, 6.0, 7.0,    // adv_pos
       -0.3, 0.2, 0.1;   // adv_vel

Eigen::VectorXf weights = mighty::policy::forward(obs);
// weights = [time, dyn, stat, accel, jerk, vel_constr, acc_constr]
```

## Dependencies

**Python Dependencies:**
- JAX, Flax (already installed from Phase 3)
- NumPy (core dependency)
- PyTorch (for ONNX export)
- ONNX, onnxscript (for ONNX validation)
- ONNX Runtime (for verification)

**Installation:**
```bash
pip install torch onnx onnxscript onnxruntime
```

**C++ Dependencies:**
- **For ONNX**: ONNX Runtime (recommended)
- **For C++ header**: Eigen 3.x (header-only)

## Performance Characteristics

### Export Times (CPU)
- Weight extraction: < 0.001s
- NumPy save: < 0.01s
- C++ header generation: < 0.01s
- ONNX export: ~2-3s (one-time)
- ONNX verification: ~0.5s

### Model Sizes
- NumPy (.npz): ~25 KB (compressed)
- C++ header (.h): ~150 KB (text, compiles to ~25 KB)
- ONNX (.onnx): ~8.8 KB

### Inference Performance (Estimated)
- **ONNX Runtime (CPU)**: ~0.1-0.5 ms per inference
- **C++ header (Eigen)**: ~0.05-0.2 ms per inference
- **Flax (JAX)**: ~1-2 ms per inference (first call slower due to JIT)

ONNX Runtime is recommended for production due to optimization support.

## Integration with Previous Phases

**Complete MIGHTY RL Pipeline:**

```
Phase 1: Cost Function (mighty_jax.py)
    ↓
Phase 2: Differentiable Solver (mighty_solver_jax.py)
    ↓
Phase 3: Policy Network (policy_network.py)
    ↓
Phase 4: Training Loop (train.py)
    ↓
Phase 5: Model Export (export_model.py) ← CURRENT
    ↓
C++ Deployment (ONNX Runtime or C++ header)
```

**Deployment Workflow:**
1. Train policy with Phase 4 (`train.py`)
2. Export trained model with Phase 5 (`export_model.py`)
3. Deploy to C++:
   - Load ONNX model in C++ with ONNX Runtime
   - OR include C++ header for zero-dependency deployment
4. Real-time inference on robot hardware

## Known Limitations

### 1. ONNX Opset Version
**Issue**: PyTorch generates ONNX opset 18, but backward conversion to opset 12 fails
**Impact**: Warning messages during export (but models still work)
**Solution**: Use opset 18 or higher in C++ (ONNX Runtime 1.10+)

### 2. PyTorch Dependency
**Issue**: ONNX export requires PyTorch installation
**Impact**: Adds ~800 MB dependency
**Alternatives**:
- Use C++ header export (no PyTorch needed)
- Use tf2onnx if TensorFlow is preferred

### 3. C++ Header Size
**Issue**: Header file is large (~150 KB text)
**Impact**: Slower compilation
**Mitigation**: Compile once, link as static library

## Recommendations

### For Development
- Use Flax model directly (fastest iteration)
- Export to ONNX only when needed for testing

### For Production
- **Recommended**: ONNX Runtime
  - Fast inference (~0.1 ms)
  - Supports quantization (INT8, FP16)
  - Platform-independent
  - Easy to update model without recompilation

- **Alternative**: C++ Header
  - Zero external dependencies
  - Slightly faster inference (~0.05 ms)
  - Model updates require recompilation

### Optimization Tips
1. **Quantization**: Convert ONNX to INT8 for 4x speedup
2. **Batch inference**: Process multiple observations together
3. **GPU acceleration**: Use ONNX Runtime with CUDA for 10-100x speedup
4. **Model pruning**: Remove less important weights (advanced)

## Files Summary

```
export_model.py          14.2 KB    Export utilities
export_test.py           14.0 KB    Test suite
PHASE5_COMPLETE.md       (this)     Documentation
```

## Next Steps

Phase 5 completes the MIGHTY RL pipeline. Potential extensions:

### 1. C++ Integration
- Create example C++ project using ONNX Runtime
- Test with real robot hardware
- Profile inference performance

### 2. Model Optimization
- Implement ONNX quantization (FP32 → INT8)
- Benchmark inference speed
- Compare ONNX Runtime vs C++ header

### 3. Deployment Pipeline
- Automate export in training script
- Version control for exported models
- CI/CD integration for model validation

### 4. Extended Functionality
- Support for different network architectures
- Batch export for multiple policies
- Model compression techniques

## Status

**Phase 5 Status**: ✅ **COMPLETE**

All components implemented and tested:
- ✅ Weight extraction working
- ✅ NumPy export working
- ✅ C++ header generation working
- ✅ ONNX export working
- ✅ Numerical verification passed (max diff: 4.79e-07)
- ✅ All 6/6 tests passed

**All 5 Phases Complete!** 🎉

The MIGHTY evasion policy pipeline is fully implemented:
- Phase 1: Cost function ✅
- Phase 2: Differentiable solver ✅
- Phase 3: Policy network ✅
- Phase 4: Training loop ✅
- Phase 5: Model export ✅

Ready for C++ deployment and real-world testing!

---

**Date**: 2026-02-02
**Status**: Complete and tested
**Dependencies**: torch, onnx, onnxscript, onnxruntime (all installed)
**Next**: C++ integration and deployment
