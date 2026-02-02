#!/usr/bin/env python3
"""
Verify that the JAX implementation files are correctly structured.
This script checks the implementation without requiring JAX to be installed.
"""

import os
import sys
import ast
import inspect


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✓ {os.path.basename(filepath)} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {os.path.basename(filepath)} MISSING")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def extract_functions(filepath):
    """Extract function definitions from a Python file."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return functions
    except Exception as e:
        print(f"  ✗ Error parsing file: {e}")
        return []


def main():
    print("=" * 70)
    print("JAX MIGHTY Implementation Verification")
    print("=" * 70)
    print()

    # Check files exist
    print("1. Checking files...")
    files = [
        'mighty_jax.py',
        'mighty_jax_test.py',
        'requirements_jax.txt',
        'README_JAX.md',
        'setup_jax.sh',
    ]

    all_exist = all(check_file_exists(f) for f in files)
    print()

    if not all_exist:
        print("✗ Some files are missing!")
        return 1

    # Check Python syntax
    print("2. Checking Python syntax...")
    python_files = ['mighty_jax.py', 'mighty_jax_test.py']
    all_valid = True
    for pyfile in python_files:
        if check_python_syntax(pyfile):
            print(f"  ✓ {pyfile} has valid syntax")
        else:
            all_valid = False
    print()

    if not all_valid:
        print("✗ Some files have syntax errors!")
        return 1

    # Check mighty_jax.py structure
    print("3. Checking mighty_jax.py structure...")
    required_functions = [
        # Utilities
        'bernstein_basis_and_derivative_jax',
        'get_segment_and_tau_jax',
        'reconstruct_jax',
        'eval_traj_jax',
        'eval_traj_and_derivs_jax',
        'fit_quintic_jax',
        'f_obs_poly_jax',
        # Cost terms
        'compute_J_time_jax',
        'compute_J_jerk_jax',
        'compute_J_acc_jax',
        'compute_J_dyn_jax',
        'compute_J_stat_jax',
        'compute_J_vel_constr_jax',
        'compute_J_acc_constr_jax',
        # Main function
        'evaluate_objective_jax',
    ]

    functions = extract_functions('mighty_jax.py')
    missing = []
    for func in required_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} MISSING")
            missing.append(func)
    print()

    if missing:
        print(f"✗ Missing {len(missing)} required functions!")
        return 1

    # Check test file structure
    print("4. Checking mighty_jax_test.py structure...")
    test_functions = [
        'test_cost_value',
        'test_gradient_computation',
        'test_jit_compilation',
        'test_gradient_finite_differences',
        'run_all_tests',
    ]

    test_funcs = extract_functions('mighty_jax_test.py')
    missing_tests = []
    for func in test_functions:
        if func in test_funcs:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} MISSING")
            missing_tests.append(func)
    print()

    if missing_tests:
        print(f"✗ Missing {len(missing_tests)} required test functions!")
        return 1

    # Summary
    print("=" * 70)
    print("✓ All verification checks passed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Install JAX: ./setup_jax.sh")
    print("  2. Run tests: python3 mighty_jax_test.py")
    print("  3. See README_JAX.md for usage examples")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
