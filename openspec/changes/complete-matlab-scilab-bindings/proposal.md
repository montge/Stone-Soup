# Change: Complete MATLAB/Simulink and Scilab Bindings

## Why
The initial MATLAB/Simulink and Scilab binding implementations are functional but incomplete. Several tasks remain for CI integration, packaging, Octave compatibility testing, and documentation to make these bindings production-ready.

## What Changes
- Add Scilab to CI workflow with conditional execution
- Complete ATOMS package scripts (install/uninstall)
- Create Simulink library file (`stonesoup_lib.slx`) with block masks
- Add GNU Octave compatibility testing
- Create multi-target tracking demo models
- Add comprehensive documentation and user guides

## Impact
- Affected specs: multi-lang-bindings
- Affected code:
  - `bindings/scilab/` - CI, ATOMS packaging, help files
  - `bindings/matlab/` - Simulink library, Octave tests, documentation
  - `.github/workflows/ci.yml` - Scilab CI job
