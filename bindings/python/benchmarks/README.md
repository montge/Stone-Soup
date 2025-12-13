# Stone Soup PyO3 Benchmarks

Performance benchmarks comparing PyO3 native bindings against pure Python implementations.

## Overview

These benchmarks measure the performance advantage of using Rust-based PyO3 bindings compared to pure Python implementations for common tracking operations.

## Running Benchmarks

### Prerequisites

```bash
pip install -e .[dev]  # Includes pytest-benchmark
```

### Quick Run

```bash
cd bindings/python/benchmarks
python benchmark_pyo3.py
```

### Options

```bash
python benchmark_pyo3.py --iterations 1000 --warmup 100 --output results.md
```

- `--iterations, -n`: Number of iterations per benchmark (default: 1000)
- `--warmup, -w`: Number of warmup iterations (default: 100)
- `--output, -o`: Output file for results (default: stdout)

## Benchmark Categories

### 1. StateVector Operations
- **Add**: Vector addition of two state vectors
- **Norm**: L2 norm computation

### 2. Matrix Operations
- **Multiply**: Matrix multiplication (50x50)
- **Inverse**: Matrix inversion (50x50)
- **Cholesky**: Cholesky decomposition (50x50)

### 3. Kalman Filter Operations
- **Predict**: Single prediction step
- **Full Cycle**: Complete predict-update cycle

### 4. Batch Processing Operations
- **Batch Norms**: Compute norms for 1000 state vectors
- **Weighted Mean**: Weighted average of 1000 state vectors
- **Vectorized Norms**: NumPy-style vectorized operations
- **Batch Traces**: Trace computation for 1000 covariance matrices

## Methodology

### Timing Approach
1. **Warmup Phase**: Run function multiple times to warm JIT/caches
2. **Measurement Phase**: Time each iteration using `time.perf_counter_ns()`
3. **Statistics**: Report mean time across all iterations

### Fair Comparison
- Pure Python implementations use NumPy for underlying operations
- Both implementations use the same input data
- Warmup ensures consistent starting conditions

### Interpreting Results
- **Speedup > 1.0x**: Native bindings are faster
- **Speedup < 1.0x**: Pure Python is faster (rare for computation-heavy ops)
- For small operations, Python overhead may dominate

## Expected Results

PyO3 typically shows significant speedups for:
- **Batch operations**: Avoiding Python loop overhead
- **Complex matrix operations**: Leveraging optimized Rust/BLAS
- **Repeated small operations**: Reducing Python object creation

PyO3 may show smaller gains for:
- **Single NumPy operations**: NumPy is already highly optimized
- **I/O-bound operations**: Not compute-limited

## CI Integration

Benchmarks run automatically on:
- Push to main branch
- Pull requests to main

Results are:
- Posted as PR comments (for PRs)
- Stored as artifacts
- Tracked historically on gh-pages (main branch only)

## Adding New Benchmarks

1. Create a benchmark class in `benchmark_pyo3.py`
2. Implement both `python_*` and `native_*` methods
3. Add timing calls in `run_benchmarks()`
4. Test locally before committing
