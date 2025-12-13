# Change: Implement Scilab Bindings

## Why
The multi-lang-bindings spec includes Scilab bindings (gateway interface, Xcos palette, ATOMS package), but no implementation exists. Scilab is widely used in academia and industry for numerical computation and simulation, making it an important target for Stone Soup integration.

## What Changes
- Create `bindings/scilab/` directory with gateway interface implementation
- Implement Scilab gateway functions for core Stone Soup operations (StateVector, Kalman filter)
- Create Xcos palette blocks for simulation integration
- Package as Scilab ATOMS module for distribution
- Add build and test infrastructure for Scilab bindings

## Impact
- Affected specs: multi-lang-bindings (implementation of existing spec)
- Affected code:
  - New `bindings/scilab/` directory
  - Potential updates to CI workflows for Scilab testing
  - Updates to `bindings/README.md`
