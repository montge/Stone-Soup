# Change: Add Simulink S-Function Blocks for Tracking

## Why
The multi-lang-bindings spec requires Simulink block library integration, but only stub documentation exists. Simulink/Xcos blocks enable visual simulation of tracking algorithms for education, prototyping, and hardware-in-the-loop testing. Cross-platform support (MATLAB Simulink, GNU Octave, Scilab Xcos) maximizes accessibility.

## What Changes
- Implement complete MATLAB MEX interface for libstonesoup
- Create Simulink S-function blocks for Kalman filter operations
- Create Simulink library (stonesoup_lib.slx) with palette
- Add Xcos palette for Scilab (completes Scilab bindings)
- Ensure GNU Octave MEX compatibility
- Add demo models demonstrating tracking scenarios

## Impact
- Affected specs: multi-lang-bindings (implements existing requirements)
- Affected code:
  - `bindings/matlab/` - MEX functions, MATLAB classes, Simulink blocks
  - `bindings/scilab/xcos/` - Xcos palette blocks
  - `bindings/octave/` - MEX compatibility layer
- New Simulink/Xcos blocks:
  - Kalman Predictor
  - Kalman Updater
  - State Vector Source
  - Gaussian State Display
