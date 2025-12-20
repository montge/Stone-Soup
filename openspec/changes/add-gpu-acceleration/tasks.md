## 1. Array Backend Abstraction
- [x] 1.1 Create `stonesoup/backend/__init__.py` with backend selection logic
- [x] 1.2 Create `stonesoup/backend/numpy_backend.py` wrapping NumPy operations
- [x] 1.3 Create `stonesoup/backend/cupy_backend.py` wrapping CuPy operations
- [x] 1.4 Implement `get_backend()` function with auto-detection
- [x] 1.5 Add `STONESOUP_BACKEND` environment variable override
- [x] 1.6 Add backend configuration to `stonesoup/config.py`

## 2. GPU Detection and Initialization
- [x] 2.1 Implement GPU availability detection (CUDA/ROCm)
- [x] 2.2 Implement GPU memory checking before large allocations
- [x] 2.3 Add graceful fallback when GPU out of memory
- [x] 2.4 Log backend selection at initialization
- [x] 2.5 Add `stonesoup.backend.is_gpu_available()` utility function

## 3. Core Type Updates
- [x] 3.1 Update `StateVector` to use backend abstraction
- [x] 3.2 Update `CovarianceMatrix` to use backend abstraction
- [x] 3.3 Update `Matrix` to use backend abstraction
- [x] 3.4 Ensure CPU/GPU array interoperability (copy when needed)
- [x] 3.5 Add `.to_cpu()` and `.to_gpu()` methods for explicit transfers

## 4. Accelerated Operations
- [x] 4.1 GPU-accelerate particle filter resampling
- [x] 4.2 GPU-accelerate batch Kalman predictions
- [x] 4.3 GPU-accelerate large matrix inversions
- [x] 4.4 GPU-accelerate Cholesky decomposition
- [x] 4.5 Add batch operation APIs for GPU efficiency

## 5. C Library GPU Support (Optional)
- [x] 5.1 Add `ENABLE_CUDA` CMake option to libstonesoup
- [x] 5.2 Create `libstonesoup/src/cuda/` directory structure
- [x] 5.3 Implement CUDA kernel for matrix multiplication
- [x] 5.4 Implement CUDA kernel for Kalman predict/update
- [x] 5.5 Add runtime GPU detection in C library
- [x] 5.6 Update Python bindings to expose GPU functions

## 6. Benchmark Improvements
- [x] 6.1 Add historical benchmark tracking to CI workflow (COMPLETE: benchmark.yml stores history)
- [x] 6.2 Compare current run against baseline (main branch) (COMPLETE: Uses cache for baseline)
- [x] 6.3 Report performance improvements/regressions in PR comments (COMPLETE: Auto-comments on PRs)
- [ ] 6.4 Add GPU benchmarks (when GPU available) - Requires GPU runner
- [x] 6.5 Store benchmark history in gh-pages branch (COMPLETE: JSON + markdown storage)
- [x] 6.6 Create benchmark visualization dashboard (COMPLETE: Interactive Chart.js dashboard)

## 7. Testing
- [x] 7.1 Add unit tests for backend abstraction
- [x] 7.2 Add integration tests for GPU operations (skip if no GPU)
- [x] 7.3 Add benchmark tests that verify no regression
- [x] 7.4 Test CPU/GPU result equivalence (numerical precision)
- [x] 7.5 Add local GPU test runner script

## 8. Documentation
- [x] 8.1 Document GPU installation requirements
- [x] 8.2 Document backend configuration options
- [x] 8.3 Add GPU acceleration to performance guide
- [x] 8.4 Document benchmark interpretation
- [x] 8.5 Add troubleshooting guide for GPU issues

## 9. CI/CD Updates
- [ ] 9.1 Add optional GPU runner for benchmarks (self-hosted or cloud) - Requires infrastructure
- [ ] 9.2 Add GPU-specific test job (conditional on runner capability) - Requires GPU runner
- [x] 9.3 Update benchmark workflow to track baseline comparisons (COMPLETE: benchmark.yml)
- [x] 9.4 Add performance regression alerts (COMPLETE: Warns on >10% regression)
