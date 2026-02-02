#!/bin/bash
# Setup script for JAX MIGHTY implementation

set -e

echo "======================================================================"
echo "MIGHTY JAX Setup"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo ""
    echo "WARNING: Not in a virtual environment."
    echo "It's recommended to create one first:"
    echo "  python3 -m venv venv_jax"
    echo "  source venv_jax/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Installing JAX and dependencies..."
echo ""

# Install CPU-only JAX (faster for development)
pip install --upgrade pip
pip install jax>=0.4.20 jaxlib>=0.4.20
pip install jaxopt>=0.8.0
pip install numpy>=1.24.0

echo ""
echo "======================================================================"
echo "Installation complete!"
echo "======================================================================"
echo ""
echo "To verify the installation, run:"
echo "  python3 mighty_jax_test.py"
echo ""
echo "For GPU support (optional), install with:"
echo "  pip install --upgrade 'jax[cuda12]'"
echo ""
