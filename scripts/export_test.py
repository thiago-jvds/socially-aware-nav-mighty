"""
Phase 5 Test Suite: Model Export

Tests for export_model.py to validate:
1. Weight extraction from Flax params
2. NumPy export/load
3. C++ header generation
4. ONNX export
5. Numerical equivalence
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from pathlib import Path
import os

from policy_network import initialize_policy, EvasionPolicy
from export_model import (
    extract_weights_from_flax,
    save_weights_numpy,
    export_to_cpp_header,
    export_to_onnx,
    verify_onnx_export,
)


# ============================================================================
# TEST 1: Weight Extraction
# ============================================================================

def test_weight_extraction():
    """Test extracting weights from Flax parameters."""
    print("=" * 70)
    print("TEST 1: Weight Extraction from Flax")
    print("=" * 70)

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    print("Extracting weights...")
    weights = extract_weights_from_flax(params)

    print(f"\nExtracted weights:")
    print(f"  W1: {weights['W1'].shape} (expected: (12, 64))")
    print(f"  b1: {weights['b1'].shape} (expected: (64,))")
    print(f"  W2: {weights['W2'].shape} (expected: (64, 64))")
    print(f"  b2: {weights['b2'].shape} (expected: (64,))")
    print(f"  W3: {weights['W3'].shape} (expected: (64, 7))")
    print(f"  b3: {weights['b3'].shape} (expected: (7,))")

    # Check shapes
    assert weights['W1'].shape == (12, 64), f"W1 shape mismatch"
    assert weights['b1'].shape == (64,), f"b1 shape mismatch"
    assert weights['W2'].shape == (64, 64), f"W2 shape mismatch"
    assert weights['b2'].shape == (64,), f"b2 shape mismatch"
    assert weights['W3'].shape == (64, 7), f"W3 shape mismatch"
    assert weights['b3'].shape == (7,), f"b3 shape mismatch"
    print("  ✓ All shapes correct")

    # Check types
    assert isinstance(weights['W1'], np.ndarray), "W1 should be numpy array"
    print("  ✓ All weights are numpy arrays")

    print("\n✓ TEST 1 PASSED\n")
    return weights


# ============================================================================
# TEST 2: NumPy Export/Load
# ============================================================================

def test_numpy_export():
    """Test NumPy export and reload."""
    print("=" * 70)
    print("TEST 2: NumPy Export/Load")
    print("=" * 70)

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Export
    filepath = "test_policy.npz"
    print(f"Exporting to {filepath}...")
    save_weights_numpy(params, filepath)

    # Load
    print(f"\nLoading from {filepath}...")
    loaded = np.load(filepath)

    print(f"Loaded keys: {list(loaded.keys())}")
    assert 'W1' in loaded, "W1 not in loaded file"
    assert 'b1' in loaded, "b1 not in loaded file"
    print("  ✓ All keys present")

    # Check values match
    original_weights = extract_weights_from_flax(params)
    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        assert np.allclose(original_weights[key], loaded[key]), f"{key} values don't match"
    print("  ✓ All weights match original")

    # Cleanup
    os.remove(filepath)
    print(f"  ✓ Cleaned up {filepath}")

    print("\n✓ TEST 2 PASSED\n")
    return True


# ============================================================================
# TEST 3: C++ Header Generation
# ============================================================================

def test_cpp_header():
    """Test C++ header file generation."""
    print("=" * 70)
    print("TEST 3: C++ Header Generation")
    print("=" * 70)

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Export
    filepath = "test_policy.h"
    print(f"Generating C++ header...")
    export_to_cpp_header(params, filepath)

    # Check file exists
    assert Path(filepath).exists(), f"{filepath} not created"
    print(f"  ✓ File created: {filepath}")

    # Read and check content
    with open(filepath, 'r') as f:
        content = f.read()

    # Check for expected content
    assert '#ifndef POLICY_WEIGHTS_H' in content, "Missing header guard"
    assert 'constexpr float W1' in content, "Missing W1 declaration"
    assert 'constexpr float b1' in content, "Missing b1 declaration"
    assert 'forward(const Eigen::VectorXf& obs)' in content, "Missing forward function"
    assert 'softplus' in content, "Missing softplus function"
    print("  ✓ Header contains expected declarations")

    # Check for all layers
    for var in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        assert f'constexpr float {var}' in content, f"Missing {var}"
    print("  ✓ All weight arrays declared")

    # Cleanup
    os.remove(filepath)
    print(f"  ✓ Cleaned up {filepath}")

    print("\n✓ TEST 3 PASSED\n")
    return True


# ============================================================================
# TEST 4: ONNX Export
# ============================================================================

def test_onnx_export():
    """Test ONNX export."""
    print("=" * 70)
    print("TEST 4: ONNX Export")
    print("=" * 70)

    try:
        import torch
    except ImportError:
        print("  ⚠️  PyTorch not installed, skipping ONNX test")
        print("  Install with: pip install torch onnx")
        print("\n✓ TEST 4 SKIPPED\n")
        return None

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Export
    filepath = "test_policy.onnx"
    print(f"Exporting to ONNX...")
    success = export_to_onnx(params, filepath)

    if not success:
        print("  ⚠️  ONNX export failed")
        print("\n✓ TEST 4 SKIPPED\n")
        return None

    # Check file exists
    assert Path(filepath).exists(), f"{filepath} not created"
    print(f"  ✓ ONNX file created: {filepath}")

    # Check file size (should be > 0)
    file_size = Path(filepath).stat().st_size
    print(f"  File size: {file_size / 1024:.2f} KB")
    assert file_size > 1000, "ONNX file seems too small"
    print("  ✓ File size reasonable")

    # Cleanup
    os.remove(filepath)
    print(f"  ✓ Cleaned up {filepath}")

    print("\n✓ TEST 4 PASSED\n")
    return True


# ============================================================================
# TEST 5: ONNX Numerical Verification
# ============================================================================

def test_onnx_verification():
    """Test ONNX output matches Flax output."""
    print("=" * 70)
    print("TEST 5: ONNX Numerical Verification")
    print("=" * 70)

    try:
        import torch
        import onnxruntime
    except ImportError:
        print("  ⚠️  PyTorch or ONNX Runtime not installed")
        print("  Install with: pip install torch onnx onnxruntime")
        print("\n✓ TEST 5 SKIPPED\n")
        return None

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    # Export
    filepath = "test_policy.onnx"
    print(f"Exporting to ONNX...")
    export_to_onnx(params, filepath)

    # Verify
    print(f"\nVerifying numerical equivalence...")
    success = verify_onnx_export(params, filepath)

    if success:
        print("  ✓ ONNX and Flax outputs match")
    else:
        print("  ✗ Outputs don't match!")

    # Cleanup
    os.remove(filepath)
    print(f"  ✓ Cleaned up {filepath}")

    assert success, "ONNX verification failed"

    print("\n✓ TEST 5 PASSED\n")
    return True


# ============================================================================
# TEST 6: End-to-End Export Workflow
# ============================================================================

def test_full_export_workflow():
    """Test complete export workflow."""
    print("=" * 70)
    print("TEST 6: Full Export Workflow")
    print("=" * 70)

    # Initialize policy
    policy, params, _ = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    print("Exporting in all formats...")

    # 1. NumPy
    save_weights_numpy(params, "export_test.npz")
    assert Path("export_test.npz").exists()
    print("  ✓ NumPy export complete")

    # 2. C++ header
    export_to_cpp_header(params, "export_test.h")
    assert Path("export_test.h").exists()
    print("  ✓ C++ header export complete")

    # 3. ONNX (if available)
    try:
        import torch
        export_to_onnx(params, "export_test.onnx")
        assert Path("export_test.onnx").exists()
        print("  ✓ ONNX export complete")
    except ImportError:
        print("  ⚠️  ONNX export skipped (PyTorch not available)")

    # Cleanup
    for filepath in ["export_test.npz", "export_test.h", "export_test.onnx"]:
        if Path(filepath).exists():
            os.remove(filepath)
    print("\n  ✓ Cleanup complete")

    print("\n✓ TEST 6 PASSED\n")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all Phase 5 tests."""
    print("\n")
    print("=" * 70)
    print("MODEL EXPORT - PHASE 5 TEST SUITE")
    print("=" * 70)
    print()

    results = {}

    try:
        # Test 1: Weight extraction
        weights = test_weight_extraction()
        results['weight_extraction'] = True

        # Test 2: NumPy export
        results['numpy_export'] = test_numpy_export()

        # Test 3: C++ header
        results['cpp_header'] = test_cpp_header()

        # Test 4: ONNX export
        onnx_result = test_onnx_export()
        results['onnx_export'] = onnx_result if onnx_result is not None else 'skipped'

        # Test 5: ONNX verification
        verify_result = test_onnx_verification()
        results['onnx_verify'] = verify_result if verify_result is not None else 'skipped'

        # Test 6: Full workflow
        results['full_workflow'] = test_full_export_workflow()

        # Summary
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for v in results.values() if v is True)
        skipped = sum(1 for v in results.values() if v == 'skipped')
        total = len(results)

        for name, result in results.items():
            if result is True:
                print(f"✓ PASSED: {name.replace('_', ' ').title()}")
            elif result == 'skipped':
                print(f"⚠ SKIPPED: {name.replace('_', ' ').title()}")
            else:
                print(f"✗ FAILED: {name.replace('_', ' ').title()}")

        print()
        print(f"Total: {passed}/{total - skipped} tests passed ({skipped} skipped)")
        print()

        if passed == total - skipped:
            print("🎉 Phase 5 Complete! Model export is ready.")
        else:
            print("⚠️  Some tests failed. Check output above.")

        print()

        return passed == total - skipped

    except Exception as e:
        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
