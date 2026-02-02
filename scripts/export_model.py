"""
Phase 5: Model Export for C++ Deployment

This module provides utilities to export the trained Flax policy network
to formats suitable for C++ inference:
1. ONNX format (recommended - use with ONNX Runtime in C++)
2. Raw NumPy weights (for custom C++ implementation)
3. C++ header file (zero-dependency option)
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def extract_weights_from_flax(params: Dict) -> Dict[str, np.ndarray]:
    """
    Extract weight matrices from Flax parameter dictionary.

    Flax stores params in nested dict structure:
    params['params']['hidden1']['kernel'] = (12, 64)
    params['params']['hidden1']['bias'] = (64,)
    etc.

    Args:
        params: Flax parameter dictionary

    Returns:
        weights: Dict with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'
    """
    weights = {}

    # Extract Layer 1 (hidden1)
    weights['W1'] = np.array(params['params']['hidden1']['kernel'])  # (12, 64)
    weights['b1'] = np.array(params['params']['hidden1']['bias'])    # (64,)

    # Extract Layer 2 (hidden2)
    weights['W2'] = np.array(params['params']['hidden2']['kernel'])  # (64, 64)
    weights['b2'] = np.array(params['params']['hidden2']['bias'])    # (64,)

    # Extract Layer 3 (output)
    weights['W3'] = np.array(params['params']['output']['kernel'])   # (64, 7)
    weights['b3'] = np.array(params['params']['output']['bias'])     # (7,)

    return weights


def save_weights_numpy(params: Dict, filepath: str = "policy_weights.npz"):
    """
    Save weights as NumPy archive.

    Args:
        params: Flax parameter dictionary
        filepath: Output file path

    Example:
        # Save
        save_weights_numpy(params, "policy.npz")

        # Load in Python
        weights = np.load("policy.npz")
        W1 = weights['W1']
    """
    weights = extract_weights_from_flax(params)
    np.savez(filepath, **weights)
    print(f"Saved weights to {filepath}")
    print(f"  W1: {weights['W1'].shape}")
    print(f"  b1: {weights['b1'].shape}")
    print(f"  W2: {weights['W2'].shape}")
    print(f"  b2: {weights['b2'].shape}")
    print(f"  W3: {weights['W3'].shape}")
    print(f"  b3: {weights['b3'].shape}")


def export_to_cpp_header(params: Dict, filepath: str = "policy_weights.h"):
    """
    Export weights as C++ header file with Eigen matrices.

    Generates a header file that can be directly included in C++ code.
    Uses Eigen library for matrix operations.

    Args:
        params: Flax parameter dictionary
        filepath: Output header file path

    Example C++ usage:
        #include "policy_weights.h"
        Eigen::VectorXf forward(const Eigen::VectorXf& obs) {
            Eigen::VectorXf h1 = (W1 * obs + b1).cwiseMax(0);
            Eigen::VectorXf h2 = (W2 * h1 + b2).cwiseMax(0);
            Eigen::VectorXf raw = W3 * h2 + b3;
            return softplus(raw) + base_weights;
        }
    """
    weights = extract_weights_from_flax(params)

    def array_to_cpp_init(arr, name):
        """Convert numpy array to C++ initializer list."""
        if arr.ndim == 1:
            values = ', '.join([f'{x:.8f}f' for x in arr])
            return f"constexpr float {name}[{arr.shape[0]}] = {{{values}}};"
        elif arr.ndim == 2:
            # Row-major format
            lines = []
            for row in arr:
                values = ', '.join([f'{x:.8f}f' for x in row])
                lines.append(f"    {{{values}}}")
            rows_str = ',\n'.join(lines)
            return f"constexpr float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n{rows_str}\n}};"
        else:
            raise ValueError(f"Only 1D and 2D arrays supported, got {arr.ndim}D")

    # Generate header file
    header_content = f"""// AUTO-GENERATED: Policy network weights for MIGHTY evasion
// Generated from trained Flax model
// Use with Eigen library for matrix operations

#ifndef POLICY_WEIGHTS_H
#define POLICY_WEIGHTS_H

#include <Eigen/Dense>
#include <cmath>

namespace mighty {{
namespace policy {{

// Network architecture: 12 -> 64 -> 64 -> 7
// Activation: ReLU for hidden layers, softplus + base for output

// Layer 1: Input (12) -> Hidden1 (64)
{array_to_cpp_init(weights['W1'].T, 'W1')}  // Transposed for Eigen
{array_to_cpp_init(weights['b1'], 'b1')}

// Layer 2: Hidden1 (64) -> Hidden2 (64)
{array_to_cpp_init(weights['W2'].T, 'W2')}  // Transposed for Eigen
{array_to_cpp_init(weights['b2'], 'b2')}

// Layer 3: Hidden2 (64) -> Output (7)
{array_to_cpp_init(weights['W3'].T, 'W3')}  // Transposed for Eigen
{array_to_cpp_init(weights['b3'], 'b3')}

// Base weights (added after softplus)
constexpr float base_weights[7] = {{
    10.0f,  // time_weight
    0.1f,   // dyn_weight
    0.1f,   // stat_weight
    0.1f,   // accel_weight
    0.1f,   // jerk_weight
    10.0f,  // vel_constr_weight
    1.0f    // acc_constr_weight
}};

// Softplus activation: log(1 + exp(x))
inline float softplus(float x) {{
    return std::log1p(std::exp(x));
}}

// Forward pass
inline Eigen::VectorXf forward(const Eigen::VectorXf& obs) {{
    assert(obs.size() == 12 && "Observation must be 12D");

    // Convert C arrays to Eigen matrices
    Eigen::Map<const Eigen::Matrix<float, 12, 64, Eigen::RowMajor>> W1_mat(&W1[0][0]);
    Eigen::Map<const Eigen::VectorXf> b1_vec(b1, 64);

    Eigen::Map<const Eigen::Matrix<float, 64, 64, Eigen::RowMajor>> W2_mat(&W2[0][0]);
    Eigen::Map<const Eigen::VectorXf> b2_vec(b2, 64);

    Eigen::Map<const Eigen::Matrix<float, 64, 7, Eigen::RowMajor>> W3_mat(&W3[0][0]);
    Eigen::Map<const Eigen::VectorXf> b3_vec(b3, 7);

    Eigen::Map<const Eigen::VectorXf> base_vec(base_weights, 7);

    // Layer 1: ReLU(W1 * obs + b1)
    Eigen::VectorXf h1 = (W1_mat.transpose() * obs + b1_vec).cwiseMax(0.0f);

    // Layer 2: ReLU(W2 * h1 + b2)
    Eigen::VectorXf h2 = (W2_mat.transpose() * h1 + b2_vec).cwiseMax(0.0f);

    // Layer 3: softplus(W3 * h2 + b3) + base
    Eigen::VectorXf raw = W3_mat.transpose() * h2 + b3_vec;

    Eigen::VectorXf output(7);
    for (int i = 0; i < 7; ++i) {{
        output[i] = softplus(raw[i]) + base_weights[i];
    }}

    return output;
}}

}}  // namespace policy
}}  // namespace mighty

#endif  // POLICY_WEIGHTS_H
"""

    # Write to file
    with open(filepath, 'w') as f:
        f.write(header_content)

    print(f"Exported C++ header to {filepath}")
    print(f"  Include in C++ code: #include \"{Path(filepath).name}\"")
    print(f"  Forward pass: mighty::policy::forward(obs)")


def export_to_onnx(params: Dict, filepath: str = "evasion_policy.onnx"):
    """
    Export model to ONNX format via PyTorch bridge.

    ONNX is the recommended format for production deployment as it:
    - Works with ONNX Runtime (fast C++ inference)
    - Supports quantization and optimization
    - Is widely supported across platforms

    Args:
        params: Flax parameter dictionary
        filepath: Output ONNX file path

    Returns:
        True if export succeeded

    Requirements:
        pip install torch onnx onnxruntime
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch onnx")
        return False

    # Extract weights
    weights = extract_weights_from_flax(params)

    # Create PyTorch model
    class PolicyTorch(nn.Module):
        def __init__(self, weights_dict):
            super().__init__()
            self.fc1 = nn.Linear(12, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 7)

            # Load weights from Flax/JAX
            self.fc1.weight.data = torch.from_numpy(weights_dict['W1'].T).float()
            self.fc1.bias.data = torch.from_numpy(weights_dict['b1']).float()

            self.fc2.weight.data = torch.from_numpy(weights_dict['W2'].T).float()
            self.fc2.bias.data = torch.from_numpy(weights_dict['b2']).float()

            self.fc3.weight.data = torch.from_numpy(weights_dict['W3'].T).float()
            self.fc3.bias.data = torch.from_numpy(weights_dict['b3']).float()

            # Base weights
            self.register_buffer('base', torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0]))

        def forward(self, obs):
            x = torch.relu(self.fc1(obs))
            x = torch.relu(self.fc2(x))
            raw = self.fc3(x)
            # Softplus + base
            output = torch.nn.functional.softplus(raw) + self.base
            return output

    # Create model and set to eval mode
    model = PolicyTorch(weights)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 12)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=['observation'],
        output_names=['cost_weights'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'cost_weights': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to {filepath}")
    print(f"  Input: observation (batch_size, 12)")
    print(f"  Output: cost_weights (batch_size, 7)")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model is valid")
    except ImportError:
        print(f"  Note: Install 'onnx' package to validate model")
    except Exception as e:
        print(f"  Warning: ONNX validation error: {e}")

    return True


def verify_onnx_export(flax_params: Dict, onnx_filepath: str) -> bool:
    """
    Verify ONNX export by comparing outputs with Flax model.

    Args:
        flax_params: Original Flax parameters
        onnx_filepath: Path to exported ONNX model

    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime")
        return False

    from policy_network import EvasionPolicy

    # Create Flax policy
    policy = EvasionPolicy(hidden_dim=64)

    # Test observations
    test_obs = jnp.array([
        [1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 5.0, 6.0, 7.0, -0.3, 0.2, 0.1],
        [0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 3.0, 2.0, 1.0, -0.5, 0.3, 0.0],
    ])

    # Flax forward pass
    # Use the same order as ONNX: time, dyn, stat, accel, jerk, vel_constr, acc_constr
    weight_keys = ['time_weight', 'dyn_weight', 'stat_weight', 'accel_weight',
                   'jerk_weight', 'vel_constr_weight', 'acc_constr_weight']
    flax_outputs = []
    for obs in test_obs:
        theta = policy.apply(flax_params, obs)
        flax_output = jnp.array([theta[k] for k in weight_keys])
        flax_outputs.append(flax_output)
    flax_outputs = jnp.array(flax_outputs)

    # ONNX forward pass
    session = ort.InferenceSession(onnx_filepath)
    onnx_outputs = session.run(
        None,
        {'observation': np.array(test_obs, dtype=np.float32)}
    )[0]

    # Compare
    max_diff = float(np.max(np.abs(flax_outputs - onnx_outputs)))
    print(f"\nNumerical verification:")
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print(f"  ✓ Outputs match (tolerance: 1e-5)")
        return True
    else:
        print(f"  ✗ Outputs differ (max diff: {max_diff:.2e})")
        print(f"  Flax output: {flax_outputs[0]}")
        print(f"  ONNX output: {onnx_outputs[0]}")
        return False


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EXPORT - Example Usage")
    print("=" * 70)

    from policy_network import initialize_policy, EvasionPolicy

    # Initialize a policy (in practice, load trained params)
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    print("\n1. Exporting to NumPy (.npz)")
    save_weights_numpy(params, "policy_weights.npz")

    print("\n2. Exporting to C++ header")
    export_to_cpp_header(params, "policy_weights.h")

    print("\n3. Exporting to ONNX")
    success = export_to_onnx(params, "evasion_policy.onnx")

    if success:
        print("\n4. Verifying ONNX export")
        verify_onnx_export(params, "evasion_policy.onnx")

    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("\nGenerated files:")
    print("  - policy_weights.npz   (NumPy format)")
    print("  - policy_weights.h     (C++ header)")
    print("  - evasion_policy.onnx  (ONNX format)")
    print("\nUse ONNX format for C++ deployment with ONNX Runtime")
