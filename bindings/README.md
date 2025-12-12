# Stone Soup Language Bindings

This directory contains language bindings for the Stone Soup tracking framework, enabling use from multiple programming languages.

## Available Bindings

### Rust (`rust/`)
Native Rust bindings providing zero-cost abstractions over the Stone Soup core library.
- **Build**: `cargo build --release`
- **Test**: `cargo test`
- **Documentation**: `cargo doc --open`

### Python (`python/`)
High-performance Python bindings built with PyO3 and Maturin.
- **Build**: `maturin develop`
- **Install**: `pip install .`
- **Requirements**: Python >= 3.10

### Java (`java/`)
Java bindings using Project Panama FFI for native interop.
- **Build**: `mvn package`
- **Requirements**: Java 21+ (for Panama FFI)

### Go (`go/`)
Go bindings using cgo for C interoperability.
- **Build**: `go build`
- **Test**: `go test`

### Node.js (`nodejs/`)
JavaScript/TypeScript bindings using napi-rs.
- **Build**: `npm install && npm run build`
- **Test**: `npm test`

### Ada (`ada/`)
Ada bindings with strong typing guarantees.
- **Build**: `gprbuild -P stonesoup.gpr`

### MATLAB (`matlab/`)
MATLAB bindings for scientific computing workflows.
- **Usage**: Add to MATLAB path and use `stonesoup.*` functions

### Octave (`octave/`)
GNU Octave bindings compatible with MATLAB interface.
- **Installation**: Copy to Octave package directory

## Architecture

All bindings interface with a common C API layer (`libstonesoup`) that provides a stable ABI. Each language binding:

1. Links to `libstonesoup` (shared library)
2. Wraps C functions with idiomatic language constructs
3. Provides language-specific error handling and memory management
4. Maintains type safety where applicable

## Building

Each binding can be built independently. Refer to the README or build configuration in each subdirectory for language-specific instructions.

## Contributing

When adding new bindings:
1. Follow the existing directory structure
2. Provide build configuration appropriate to the language ecosystem
3. Include tests demonstrating core functionality
4. Document installation and usage instructions
