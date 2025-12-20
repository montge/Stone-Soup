# multi-lang-bindings Specification

## Purpose
Defines the multi-language SDK bindings for Stone Soup. This includes the core C library (libstonesoup) and bindings for Rust, Python (PyO3), Java, Ada, C++, Go, Node.js, MATLAB, GNU Octave, and Scilab.
## Requirements
### Requirement: Core C Library
The system SHALL provide a C library (libstonesoup) that implements core tracking algorithms with a stable ABI.

#### Scenario: StateVector operations available in C
- **WHEN** a C program includes stonesoup.h
- **THEN** StateVector creation, arithmetic, and transformation functions are available

#### Scenario: Kalman filter operations available in C
- **WHEN** a C program links against libstonesoup
- **THEN** Kalman predict and update operations are callable

#### Scenario: MISRA C:2012 compliance
- **WHEN** libstonesoup source code is analyzed with cppcheck MISRA addon
- **THEN** no mandatory rule violations are reported

### Requirement: Rust Bindings
The system SHALL provide safe Rust bindings for libstonesoup with zero-cost FFI abstractions.

#### Scenario: Rust StateVector type
- **WHEN** a Rust program uses stonesoup::StateVector
- **THEN** the type provides safe access to underlying C operations with ownership semantics

#### Scenario: Rust clippy compliance
- **WHEN** Rust bindings are analyzed with clippy
- **THEN** no warnings are reported at the default warning level

#### Scenario: Rust dependency security
- **WHEN** cargo-audit is run against Rust bindings
- **THEN** no known vulnerabilities are reported in dependencies

### Requirement: Python Bindings (PyO3)
The system SHALL provide Python bindings via PyO3 that integrate with NumPy and maintain API compatibility.

#### Scenario: NumPy array compatibility
- **WHEN** a Python program creates a StateVector from a numpy array
- **THEN** the data is shared without copying where possible

#### Scenario: Existing API compatibility
- **WHEN** existing Python code imports stonesoup types
- **THEN** the public API behavior is unchanged

#### Scenario: Performance improvement
- **WHEN** Kalman filter operations are benchmarked
- **THEN** PyO3 bindings perform at least 2x faster than pure Python for large state vectors

### Requirement: Java Bindings
The system SHALL provide Java bindings using Panama FFI (JEP 454) with JNI fallback for older Java versions.

#### Scenario: Panama FFI on Java 21+
- **WHEN** Java 21+ code uses StoneSoup classes
- **THEN** operations use Panama FFI without JNI overhead

#### Scenario: JNI fallback on Java 11-20
- **WHEN** Java 11-20 code uses StoneSoup classes
- **THEN** operations fall back to JNI bindings

#### Scenario: Java type safety
- **WHEN** Java code uses StateVector class
- **THEN** type-safe operations are available with proper null handling

### Requirement: Ada Bindings
The system SHALL provide Ada bindings with SPARK subset support for formal verification and explicit numeric range constraints.

#### Scenario: Ada specification files
- **WHEN** an Ada program withs Stone_Soup package
- **THEN** type-safe bindings for core operations are available

#### Scenario: SPARK contracts
- **WHEN** SPARK subset operations are analyzed with GNATprove
- **THEN** contracts are proven free of runtime exceptions

#### Scenario: Ada tasking safety
- **WHEN** multiple Ada tasks use Stone_Soup concurrently
- **THEN** no data races occur with proper protected object usage

#### Scenario: Explicit numeric ranges
- **WHEN** Ada types are defined for state vectors
- **THEN** explicit range constraints match tracking domain requirements

#### Scenario: Domain-specific type packages
- **WHEN** tracking in specific domain (undersea, orbital, interplanetary)
- **THEN** domain-specific Ada type packages with appropriate ranges are available

#### Scenario: Fixed-point arithmetic
- **WHEN** deterministic arithmetic is required for certification
- **THEN** fixed-point type variants are available for critical calculations

#### Scenario: Overflow prevention proofs
- **WHEN** SPARK analysis runs on numeric operations
- **THEN** range overflow/underflow is proven impossible within domain constraints

### Requirement: C++ Bindings
The system SHALL provide C++ header-only wrappers with RAII semantics for libstonesoup.

#### Scenario: RAII resource management
- **WHEN** a C++ StateVector object goes out of scope
- **THEN** underlying C resources are automatically released

#### Scenario: C++ exception safety
- **WHEN** a C operation fails
- **THEN** a C++ exception is thrown with descriptive error message

#### Scenario: MISRA C++:2023 compliance
- **WHEN** C++ bindings are analyzed for MISRA compliance
- **THEN** no mandatory rule violations are reported

### Requirement: Go Bindings
The system SHALL provide Go bindings via cgo with idiomatic Go types.

#### Scenario: Go module availability
- **WHEN** go get github.com/dstl/stone-soup/go is executed
- **THEN** the module is installed with all dependencies

#### Scenario: Go error handling
- **WHEN** a C operation fails
- **THEN** an error value is returned following Go conventions

#### Scenario: Go garbage collection compatibility
- **WHEN** Go code uses StateVector
- **THEN** C memory is properly freed when Go objects are collected

### Requirement: Node.js Bindings
The system SHALL provide Node.js bindings via napi-rs with TypeScript type definitions.

#### Scenario: TypeScript types
- **WHEN** TypeScript code imports @stonesoup/core
- **THEN** full type definitions are available for IDE support

#### Scenario: Async operation support
- **WHEN** long-running operations are called
- **THEN** Promise-based async APIs are available

#### Scenario: Node.js memory safety
- **WHEN** Node.js bindings are run under valgrind
- **THEN** no memory leaks are detected

### Requirement: MATLAB Bindings
The system SHALL provide MATLAB bindings via MEX interface for integration with MATLAB and Simulink.

#### Scenario: MEX function availability
- **WHEN** MATLAB code calls stonesoup MEX functions
- **THEN** core tracking operations execute via libstonesoup

#### Scenario: MATLAB array compatibility
- **WHEN** MATLAB arrays are passed to stonesoup functions
- **THEN** data is efficiently transferred via MEX API

#### Scenario: Simulink block library
- **WHEN** Simulink model uses Stone Soup blocks
- **THEN** S-function blocks wrap core algorithms for simulation

#### Scenario: MATLAB Coder compatibility
- **WHEN** MATLAB code is compiled with MATLAB Coder
- **THEN** generated C code links against libstonesoup

### Requirement: GNU Octave Bindings
The system SHALL provide GNU Octave bindings compatible with MATLAB interface.

#### Scenario: Octave MEX compatibility
- **WHEN** Octave loads stonesoup MEX files
- **THEN** same API as MATLAB is available

#### Scenario: Octave package
- **WHEN** pkg install stonesoup is run in Octave
- **THEN** stonesoup functions are available

#### Scenario: MATLAB script compatibility
- **WHEN** MATLAB scripts using stonesoup run in Octave
- **THEN** scripts execute with same results

### Requirement: Scilab Bindings
The system SHALL provide Scilab bindings via gateway interface.

#### Scenario: Scilab gateway functions
- **WHEN** Scilab code calls stonesoup functions
- **THEN** operations execute via C gateway interface

#### Scenario: Xcos palette
- **WHEN** Xcos model uses Stone Soup palette
- **THEN** tracking blocks are available for simulation

#### Scenario: Scilab ATOMS package
- **WHEN** atomsInstall stonesoup is run
- **THEN** stonesoup module is installed

#### Scenario: StateVector creation in Scilab
- **WHEN** Scilab code calls `stonesoup_state_vector_create(dim)`
- **THEN** a state vector of specified dimension is returned as a Scilab column vector

#### Scenario: Kalman filter operations in Scilab
- **WHEN** Scilab code calls `stonesoup_kalman_predict` or `stonesoup_kalman_update`
- **THEN** Kalman filter operations execute via libstonesoup with Scilab-native data types

#### Scenario: Error handling in Scilab
- **WHEN** a gateway function encounters an error
- **THEN** a descriptive Scilab error is raised with error code and message

#### Scenario: Scilab 6.x compatibility
- **WHEN** Scilab bindings are loaded in Scilab 6.0 or later
- **THEN** all gateway functions and macros work correctly

#### Scenario: Xcos Kalman Predictor block
- **WHEN** Xcos model includes Kalman Predictor block
- **THEN** block performs Kalman prediction with configurable parameters

#### Scenario: Xcos Kalman Updater block
- **WHEN** Xcos model includes Kalman Updater block
- **THEN** block performs Kalman update with measurement input

### Requirement: Multi-Scale Numeric Handling
The system SHALL support tracking across multiple spatial scales without precision loss.

#### Scenario: Scale transition
- **WHEN** object transitions between tracking domains (e.g., LEO to lunar)
- **THEN** coordinate system switches with appropriate precision for each domain

#### Scenario: Precision documentation
- **WHEN** API documentation is generated
- **THEN** numeric precision and range limits are clearly documented per domain

#### Scenario: Compile-time domain selection
- **WHEN** library is compiled for specific domain
- **THEN** numeric types are optimized for that domain's range requirements

### Requirement: C Library GPU Support
The libstonesoup C library SHALL optionally support GPU acceleration via CUDA.

#### Scenario: CUDA compilation option
- **WHEN** CMake is configured with -DENABLE_CUDA=ON
- **THEN** CUDA kernels are compiled and linked
- **AND** GPU functions are exported in library API

#### Scenario: Runtime GPU detection in C
- **WHEN** libstonesoup initializes with CUDA support
- **THEN** available CUDA devices are enumerated
- **AND** device capabilities are logged

#### Scenario: GPU-accelerated Kalman operations
- **WHEN** CUDA is available and enabled
- **THEN** Kalman predict/update can use GPU kernels
- **AND** batch operations benefit from parallelism

#### Scenario: Non-CUDA build compatibility
- **WHEN** CMake is configured without CUDA
- **THEN** library builds with CPU-only operations
- **AND** GPU functions return NOT_IMPLEMENTED error

### Requirement: Python Bindings GPU Passthrough
The Python bindings SHALL expose GPU acceleration from both CuPy and native backends.

#### Scenario: CuPy array acceptance
- **WHEN** CuPy array is passed to Stone Soup function
- **THEN** operation executes on GPU when supported
- **AND** result is returned as CuPy array

#### Scenario: Native GPU function exposure
- **WHEN** PyO3 bindings are built with CUDA support
- **THEN** GPU-accelerated native functions are available
- **AND** performance exceeds pure CuPy for supported operations

### Requirement: Scilab CI Integration
The system SHALL include Scilab bindings in CI with conditional execution.

#### Scenario: Scilab CI job execution
- **WHEN** CI runs and Scilab is available
- **THEN** Scilab gateway functions compile and tests execute

#### Scenario: Scilab unavailable gracefully skipped
- **WHEN** CI runs and Scilab is not installed
- **THEN** Scilab job is skipped without failing the pipeline

#### Scenario: Scilab version matrix
- **WHEN** CI runs Scilab tests
- **THEN** both Scilab 6.x and 2024.x are tested if available

### Requirement: Scilab ATOMS Packaging
The system SHALL provide complete ATOMS package scripts for Scilab distribution.

#### Scenario: ATOMS installation script
- **WHEN** user runs builder.sce
- **THEN** gateway functions compile and module installs to Scilab

#### Scenario: ATOMS uninstallation script
- **WHEN** user runs cleaner.sce
- **THEN** module is cleanly removed from Scilab installation

#### Scenario: ATOMS package validation
- **WHEN** ATOMS package structure is validated
- **THEN** DESCRIPTION, DESCRIPTION-FUNCTIONS, and required files are present

### Requirement: Scilab Help Documentation
The system SHALL provide Scilab help files for all public functions.

#### Scenario: Help file availability
- **WHEN** user types help stonesoup_state_vector_create in Scilab
- **THEN** function documentation is displayed

#### Scenario: Help file format
- **WHEN** help files are built
- **THEN** XML help files follow Scilab help format conventions

### Requirement: Simulink Block Library
The system SHALL provide a Simulink library file containing all Stone Soup blocks.

#### Scenario: Simulink library file
- **WHEN** user opens stonesoup_lib.slx in Simulink
- **THEN** library browser shows Stone Soup block palette

#### Scenario: Block masks and icons
- **WHEN** blocks are viewed in Simulink
- **THEN** custom icons and parameter masks are displayed

#### Scenario: Block documentation
- **WHEN** user clicks Help on a block
- **THEN** block documentation is displayed

### Requirement: GNU Octave Testing
The system SHALL verify MEX compatibility with GNU Octave.

#### Scenario: Octave MEX execution
- **WHEN** Octave loads stonesoup MEX files
- **THEN** basic operations execute successfully

#### Scenario: Octave test suite
- **WHEN** Octave test suite runs
- **THEN** all compatible functions pass tests

#### Scenario: Octave limitation documentation
- **WHEN** user reads Octave documentation
- **THEN** known limitations and workarounds are documented

### Requirement: Multi-Target Tracking Demos
The system SHALL provide multi-target tracking demonstration models.

#### Scenario: Simulink multi-target demo
- **WHEN** user opens multi-target Simulink demo
- **THEN** demo shows tracking of multiple simultaneous targets

#### Scenario: Xcos multi-target demo
- **WHEN** user opens multi-target Xcos demo
- **THEN** demo shows tracking of multiple simultaneous targets

#### Scenario: Demo documentation
- **WHEN** user reads demo documentation
- **THEN** step-by-step usage instructions are provided

### Requirement: Binding User Guides
The system SHALL provide comprehensive user guides for MATLAB/Simulink and Scilab/Xcos.

#### Scenario: Simulink user guide
- **WHEN** user accesses Simulink documentation
- **THEN** complete guide for using Stone Soup blocks is available

#### Scenario: Xcos user guide
- **WHEN** user accesses Xcos documentation
- **THEN** complete guide for using Stone Soup Xcos palette is available

#### Scenario: Troubleshooting guide
- **WHEN** user encounters common issues
- **THEN** troubleshooting documentation addresses the problem
