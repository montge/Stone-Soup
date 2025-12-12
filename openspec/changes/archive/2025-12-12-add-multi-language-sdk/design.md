## Context

Stone Soup is used in defense and aerospace applications where multiple programming languages are mandated by certification requirements (DO-178C, DO-254, MIL-STD-498). The current Python-only implementation limits adoption in safety-critical embedded systems.

### Stakeholders
- Defense contractors requiring Ada/SPARK for avionics
- Automotive industry requiring MISRA-compliant C/C++
- Systems integrators requiring polyglot interoperability
- Research community requiring Python accessibility

### Constraints
- Numerical precision must be maintained across all bindings
- Memory safety critical for real-time systems
- Certification evidence required for safety-critical deployments
- Backward compatibility with existing Python API

## Goals / Non-Goals

### Goals
- Enable Stone Soup usage from 8+ programming languages
- Establish security scanning and quality gates
- Achieve 90%+ test coverage with traceability
- Support safety certification evidence generation
- Maintain Python API compatibility

### Non-Goals
- Complete rewrite of all algorithms in each language (use FFI)
- Support for legacy Python versions (<3.10)
- Real-time guarantees (application responsibility)
- Specific certification (provide evidence, not certification)

## Decisions

### Core Library Architecture
**Decision**: Extract performance-critical algorithms to a C core library (libstonesoup) with bindings for each target language.

**Rationale**:
- C provides the lowest common denominator for FFI
- Single implementation reduces maintenance burden
- Enables MISRA compliance for safety-critical C users
- Rust/Ada/Go can all call C with minimal overhead

**Alternatives considered**:
1. *Pure Rust core* - Better safety, but Ada/C interop more complex
2. *Language-specific implementations* - Maximum performance, but unsustainable maintenance
3. *WebAssembly core* - Portable, but poor real-time performance

### Binding Strategy per Language

| Language | Binding Approach | Rationale |
|----------|-----------------|-----------|
| Rust | cbindgen + safe wrappers | Zero-cost FFI, ownership safety |
| Python | PyO3 wrapping C | GIL-aware, numpy integration |
| Java | Panama FFI (JEP 454) | Modern, no JNI boilerplate |
| Ada | pragma Import + SPARK contracts | Formal verification support |
| C/C++ | Direct headers | Native access, MISRA compliance |
| Go | cgo | Standard Go FFI |
| Node.js | napi-rs | Rust safety for Node bindings |

### Python Linting Strategy
**Decision**: Use ruff as primary linter, black for formatting, retain flake8 for CI compatibility.

**Rationale**:
- ruff is 10-100x faster than flake8
- black provides deterministic formatting
- flake8 retained for existing toolchain compatibility

### SAST Strategy
**Decision**: Multi-tool approach with language-specific analyzers.

| Tool | Scope | Purpose |
|------|-------|---------|
| Bandit | Python | Security-focused SAST |
| Semgrep | Polyglot | Custom rules, taint analysis |
| cppcheck | C/C++ | Static analysis, MISRA |
| PC-lint | C/C++ | MISRA certification evidence |
| clippy | Rust | Lint + security patterns |
| cargo-audit | Rust | Dependency vulnerabilities |

### Coverage Strategy
**Decision**: Enforce coverage at multiple levels.

| Level | Target | Enforcement |
|-------|--------|-------------|
| Overall | 90%+ | CI gate, fail build |
| Branch | 80%+ | CI gate, fail build |
| Function | 80%+ | CI warning |
| New code | 95%+ | PR gate |

### Requirements Traceability
**Decision**: sphinx-needs integration with bidirectional linking.

**Approach**:
- Requirements defined as `.. req::` directives
- Tests linked via `.. test::` with `links` attribute
- Automated traceability matrix generation
- Export to ReqIF for tool interoperability

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| C core introduces memory safety risks | High | Extensive fuzzing, Rust wrappers for safe languages |
| MISRA compliance delays | Medium | Incremental adoption, core algorithms first |
| Numerical precision drift | High | Cross-language property tests, golden datasets |
| Maintenance burden of 8 bindings | High | Generated bindings where possible, CI matrix |
| Python API breakage | Medium | Compatibility shim layer, deprecation period |

## Migration Plan

### Phase 1: Foundation (capabilities: security-tooling, testing-coverage)
1. Add ruff, black, flake8 configuration
2. Add Bandit, Semgrep scanning
3. Establish coverage baselines and CI gates
4. Add sphinx-needs to documentation

### Phase 2: Core Extraction (capabilities: multi-lang-bindings)
1. Identify hot paths via profiling
2. Extract StateVector, CovarianceMatrix operations to C
3. Add Rust bindings + tests
4. Add Python PyO3 bindings (parallel to existing)

### Phase 3: Language Expansion
1. Add C/C++ headers + MISRA checking
2. Add Ada bindings + SPARK contracts
3. Add Java, Go, Node.js bindings
4. Cross-language integration tests

### Phase 4: Certification Support (capabilities: requirements-traceability)
1. Document all requirements in sphinx-needs
2. Link tests to requirements
3. Generate traceability evidence
4. Validate against DO-178C objectives

### Rollback
- Each phase independently deployable
- Python fallback always available
- Feature flags for C vs Python backends

## Open Questions

1. **MISRA subset**: Which MISRA C:2012 guidelines to enforce? Full compliance vs. safety-critical subset?
2. **Ada compiler support**: GNAT only, or also support proprietary compilers (Green Hills, AdaCore)?
3. **Java version**: Minimum Java 21 for Panama, or also support Java 11 with JNI fallback?
4. **Coverage tool**: Use coverage.py + language-specific tools, or unified platform (Codecov, SonarQube)?
5. **Requirements format**: Use OpenSpec requirements or migrate all to sphinx-needs?
