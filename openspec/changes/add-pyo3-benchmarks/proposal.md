# Change: Add PyO3 Performance Benchmarks

## Why
Demonstrate the performance advantage of PyO3/Rust bindings over pure Python implementations. This provides concrete evidence of the value multi-language bindings bring and helps track performance over time to prevent regressions.

## What Changes
- Add GitHub Actions workflow for automated benchmarking
- Implement benchmark suite comparing PyO3 vs pure Python for core operations
- Generate benchmark reports with performance comparisons
- Track performance trends across commits
- Add benchmark results to PR comments

## Impact
- Affected specs: testing-coverage
- Affected code: .github/workflows/, bindings/python/benchmarks/
- New dependencies: pytest-benchmark, asv (airspeed velocity)
