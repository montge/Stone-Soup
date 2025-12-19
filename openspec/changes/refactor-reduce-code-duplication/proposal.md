# Change: Reduce Code Duplication Across Bindings and Tests

## Why
SonarCloud analysis identifies code duplication as a maintainability concern. The multi-language SDK bindings contain significant duplicated patterns (~2,500 LOC) that increase maintenance burden and risk inconsistencies. Reducing duplication improves code quality metrics and aligns with SonarCloud's < 3% duplication threshold.

## What Changes
- **Java Bindings**: Consolidate common patterns (validation, equals/hashCode, factory methods) into base classes
- **Java Tests**: Create abstract test base class with parameterized patterns
- **C++ Tests**: Use Google Test TYPED_TEST macros for copy/move semantics tests
- **Python Types**: Extract validation utilities to dedicated module
- **Python Tests**: Consolidate repeated test fixtures

## Impact
- Affected specs: `ci-cd` (SonarCloud quality gates)
- Affected code:
  - `bindings/java/src/main/java/org/stonesoup/*.java` (~1,100 LOC)
  - `bindings/java/src/test/java/org/stonesoup/*Test.java` (~400 LOC)
  - `bindings/cpp/tests/*.cpp` (~400 LOC)
  - `stonesoup/types/*.py` (~50 LOC)
  - `stonesoup/**/tests/*.py` (~200 LOC)
- Estimated reduction: 2,000-2,500 LOC
- Risk: Low (refactoring with test coverage)
