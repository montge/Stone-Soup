# Tasks: Improve Test Coverage and Code Quality

## 1. Security Fixes
- [x] 1.1 Add `# nosec B311` comments to Python PRNG usage (generator.py)
- [x] 1.2 Add `// NOSONAR` comments to C PRNG usage (particle.c)
- [ ] 1.3 Review and address any remaining security hotspots in SonarCloud

## 2. Test Coverage - High Priority Modules (test:source ratio < 40%)
- [x] 2.1 Add tests for `functions/coordinates.py` (2536 lines, NEW: 444 lines tests)
- [ ] 2.2 Expand tests for `plotter.py` (4006 lines, currently 772 lines tests ~19%)
- [ ] 2.3 Expand tests for `functions/__init__.py` (1224 lines, currently 473 lines tests ~39%)
- [ ] 2.4 Expand tests for `types/orbitalstate.py` (1063 lines, currently 254 lines tests ~24%)
- [ ] 2.5 Expand tests for `functions/orbital.py` (964 lines, currently 420 lines tests ~44%)

## 3. Test Coverage - Medium Priority Modules
- [ ] 3.1 Add/expand tests for `models/measurement/nonlinear.py` (1367 lines)
- [ ] 3.2 Add/expand tests for `updater/kalman.py` (1083 lines)
- [ ] 3.3 Add/expand tests for `predictor/kalman.py` (733 lines)
- [ ] 3.4 Add/expand tests for `smoother/network_graphs.py` (715 lines)
- [ ] 3.5 Add/expand tests for `metricgenerator/tracktotruthmetrics.py` (806 lines)

## 4. Test Coverage - Additional Modules
- [ ] 4.1 Add tests for `sensor/radar/radar.py` (941 lines)
- [ ] 4.2 Add tests for `models/transition/linear.py` (758 lines)
- [ ] 4.3 Add tests for `models/measurement/underwater.py` (735 lines)
- [ ] 4.4 Add tests for `architecture/__init__.py` (769 lines)

## 5. Reliability Fixes
- [ ] 5.1 Fix bare `except:` clauses - replace with specific exceptions
- [ ] 5.2 Ensure proper resource management (context managers for files)
- [ ] 5.3 Add null/None validation where needed

## 6. Maintainability Improvements
- [ ] 6.1 Add docstrings to public functions missing documentation
- [ ] 6.2 Refactor long functions (>100 lines) where practical
- [ ] 6.3 Replace magic numbers with named constants
- [ ] 6.4 Address TODO/FIXME comments where appropriate

## 7. Verification
- [ ] 7.1 Run full test suite and verify all tests pass
- [ ] 7.2 Verify overall coverage >= 90%
- [ ] 7.3 Verify SonarCloud quality gate passes
- [ ] 7.4 Update spec deltas and close change
