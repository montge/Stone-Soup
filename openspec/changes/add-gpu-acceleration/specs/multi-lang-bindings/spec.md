## ADDED Requirements

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
