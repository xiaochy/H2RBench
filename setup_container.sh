#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    ninja-build \
    linux-libc-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-6 \
    libxt6 \
    libglib2.0-0 \
    curl

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Configuring git safe directories ==="
git config --global --add safe.directory /workspace/benchmark_new/polaris/src/curobo
git config --global --add safe.directory /workspace/benchmark_new/polaris/src/diff-gaussian-rasterization
git config --global --add safe.directory /workspace/benchmark_new/polaris/src/diff-surfel-rasterization
git config --global --add safe.directory /workspace/benchmark_new/polaris/src/simple-knn
git config --global --add safe.directory /workspace/benchmark_new/polaris

echo "=== Setting up PolaRiS-Hub symlink ==="
ln -sf /workspace/benchmark_new/polaris/PolaRiS-Hub \
       /workspace/benchmark_new/polaris/check_env_exp/PolaRiS-Hub

echo "=== Syncing dependencies ==="
cd /workspace/benchmark_new/polaris
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.1 uv sync --no-build-isolation-package flatdict

echo "=== Done! Activate with: source .venv/bin/activate ==="
