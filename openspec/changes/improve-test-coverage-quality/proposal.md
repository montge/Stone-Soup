# Change: Improve Test Coverage and Code Quality to 90%+

## Why
The current test coverage is approximately 72-75%, which is below the 90% target defined in the testing-coverage specification. Additionally, SonarCloud analysis has identified security hotspots, reliability issues, and maintainability concerns that need to be addressed to meet quality gates and ensure production readiness.

## What Changes
- Add comprehensive test suites for under-tested modules (functions/underwater.py, functions/orbital.py, types/coordinates.py, etc.)
- **SECURITY**: Mark legitimate PRNG usage as safe (simulation, not cryptographic)
- **RELIABILITY**: Fix bare except clauses, ensure proper resource management
- **MAINTAINABILITY**: Address code duplication, add missing docstrings, reduce complexity
- Achieve 90%+ overall coverage and pass all SonarCloud quality gates

## Impact
- Affected specs: testing-coverage, security-tooling
- Affected code:
  - stonesoup/functions/*.py (coordinate, orbital, underwater modules)
  - stonesoup/types/*.py (state, coordinates modules)
  - stonesoup/architecture/*.py (generator module - PRNG fixes)
  - libstonesoup/src/*.c (particle filter - PRNG fixes)
- CI: Quality gates will pass after implementation

## Success Criteria
- [ ] Overall test coverage >= 90%
- [ ] New code coverage >= 95%
- [ ] Zero critical/blocker security issues
- [ ] Zero critical reliability issues
- [ ] SonarCloud quality gate: PASSED
