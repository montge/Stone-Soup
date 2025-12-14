# Change: Add GPU Acceleration Support

## Why
Stone Soup performs computationally intensive matrix operations for target tracking (Kalman filters, particle filters, etc.). Production environments and developer machines often have GPUs that could significantly accelerate these operations. Currently, the codebase uses NumPy exclusively, leaving GPU resources unused. Additionally, performance benchmarks don't track historical improvements or report regression alerts.

## What Changes
- **Array Backend Abstraction**: Create pluggable backend supporting NumPy (CPU) and CuPy (GPU)
- **Auto-Detection**: Automatically detect GPU availability and select optimal backend
- **Graceful Fallback**: Fall back to CPU when GPU unavailable or for incompatible operations
- **C Library GPU Support**: Optional CUDA/OpenCL support in libstonesoup via compile flags
- **Benchmark Improvements**: Track historical performance, report improvements/regressions
- **Local GPU Testing**: Support for running GPU-accelerated tests locally

## Impact
- Affected specs: New `performance` spec, modifications to `multi-lang-bindings`
- Affected code:
  - New `stonesoup/backend/` module for array backend abstraction
  - `stonesoup/types/array.py` - use backend instead of direct NumPy
  - `stonesoup/predictor/particle.py` - GPU-accelerated particle operations
  - `libstonesoup/` - optional CUDA kernels
  - `.github/workflows/benchmark.yml` - historical tracking
  - `bindings/python/benchmarks/` - GPU benchmarks
- New dependencies (optional): `cupy`, `cupy-cuda11x` or `cupy-cuda12x`
- Risk: Medium (optional feature, CPU fallback ensures no breakage)
