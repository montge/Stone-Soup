# Design: Scilab Bindings

## Context
Scilab is an open-source numerical computation environment similar to MATLAB. Stone Soup already has MATLAB/Octave bindings via MEX interface. Scilab uses a different gateway interface but serves a similar user base in academia and simulation environments.

## Goals / Non-Goals
**Goals:**
- Provide Scilab gateway functions for core Stone Soup operations
- Enable Xcos integration for simulation workflows
- Package as ATOMS module for easy installation
- Maintain API similarity with MATLAB/Octave bindings where practical

**Non-Goals:**
- Full feature parity with Python API (focus on core tracking operations)
- Support for Scilab versions older than 6.0
- Real-time simulation performance guarantees

## Decisions

### Gateway Interface Architecture
- **Decision**: Use Scilab's external module gateway interface (C-based)
- **Why**: Direct C integration with libstonesoup, no intermediate layer needed
- **Alternatives**:
  - SWIG-based bindings (rejected: Scilab support limited)
  - Pure Scilab implementation calling shared library (rejected: more complex, less performant)

### Directory Structure
```
bindings/scilab/
├── etc/                    # Module startup/quit scripts
│   ├── stonesoup.start
│   └── stonesoup.quit
├── sci_gateway/            # C gateway functions
│   ├── sci_stonesoup.c
│   └── builder_gateway.sce
├── macros/                 # Scilab wrapper functions
│   ├── StateVector.sci
│   ├── GaussianState.sci
│   └── buildmacros.sce
├── xcos/                   # Xcos palette (phase 2)
│   └── palette.xml
├── tests/                  # Unit tests
│   └── unit_tests/
├── demos/                  # Example scripts
├── help/                   # Help files
├── DESCRIPTION             # ATOMS metadata
└── builder.sce             # Main build script
```

### Data Type Mapping
| Stone Soup Type | Scilab Representation |
|-----------------|----------------------|
| stonesoup_state_vector_t | Column vector (double) |
| stonesoup_covariance_matrix_t | Square matrix (double) |
| stonesoup_gaussian_state_t | tlist with 'sv', 'covar', 'ts' fields |

### Error Handling
- Gateway functions return Scilab error codes via `Scierror()` function
- Map `stonesoup_error_t` codes to descriptive Scilab error messages
- Input validation at both gateway and macro levels

## Risks / Trade-offs

### Risk: Scilab Version Fragmentation
- Scilab 6.x introduced breaking API changes from 5.x
- **Mitigation**: Target Scilab 6.0+ only, test on 6.x and 2024.x releases

### Risk: Build System Complexity
- Scilab build system differs from standard CMake
- **Mitigation**: Use Scilab's native builder.sce approach, document build requirements

### Trade-off: API Design
- Chose procedural API over OOP-style to match Scilab conventions
- Trade-off: Less elegant than Python API, but more natural for Scilab users

## Migration Plan
Not applicable - new implementation, no migration needed.

## Open Questions
1. Should Xcos palette be included in initial release or phased?
2. Which Scilab versions should be officially supported in CI?
3. Should we aim for ATOMS registry publication?
