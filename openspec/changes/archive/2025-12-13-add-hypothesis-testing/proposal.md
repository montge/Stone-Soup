# Change: Add Hypothesis Property-Based Testing

## Why
The testing-coverage spec requires property-based testing for numerical algorithm validation (Requirement: Property-Based Testing, lines 107-121). Currently, no Hypothesis library imports exist in the codebase - all "hypothesis" references are the stonesoup.types.hypothesis module, not the testing library.

## What Changes
- Add Hypothesis library as a test dependency
- Create property-based tests for Kalman filter operations (covariance positive semi-definite)
- Create property-based tests for serialization roundtrips
- Create property-based tests for coordinate transformations
- Add Hypothesis profile configuration for CI
- Integrate with pytest execution

## Impact
- Affected specs: testing-coverage
- Affected code: pyproject.toml, stonesoup/tests/
- New dependencies: hypothesis>=6.0
