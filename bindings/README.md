# Stone Soup Language Bindings

This directory contains language bindings for the Stone Soup tracking framework, enabling use from multiple programming languages.

## Quick Start

### Prerequisites

1. Build the core C library first:
   ```bash
   cd libstonesoup
   mkdir -p build && cd build
   cmake .. -DBUILD_TESTS=ON
   make -j$(nproc)
   export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
   ```

2. Choose your language binding below.

## Available Bindings

### Rust (`rust/`)

Native Rust bindings providing zero-cost abstractions over the Stone Soup core library. Uses `nalgebra` for linear algebra operations.

**Features:**
- Type-safe StateVector, CovarianceMatrix, GaussianState
- Kalman filter predict/update operations
- Full documentation with examples
- Clippy-compliant code

**Build & Test:**
```bash
cd rust
cargo build --release
cargo test
cargo clippy
cargo doc --open
```

**Example:**
```rust
use stonesoup::{kalman, CovarianceMatrix, GaussianState, StateVector};
use nalgebra::DMatrix;

// Create initial state [x, vx]
let state = StateVector::from_vec(vec![0.0, 1.0]);
let covar = CovarianceMatrix::diagonal(&[1.0, 0.1]);
let prior = GaussianState::new(state, covar).unwrap();

// Constant velocity transition matrix
let F = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
let Q = CovarianceMatrix::diagonal(&[0.01, 0.1]);

// Predict
let predicted = kalman::predict(&prior, &F, &Q).unwrap();
```

### Python (`python/`)

High-performance Python bindings built with PyO3 and Maturin.

**Build & Install:**
```bash
cd python
pip install maturin
maturin develop --release
```

**Example:**
```python
import stonesoup_native as ssn
import numpy as np

# Create state with covariance
state = ssn.GaussianState(
    state_vector=np.array([0.0, 1.0, 0.0, 1.0]),
    covariance=np.eye(4)
)

# Predict with transition model
predicted = ssn.kalman_predict(state, F=transition_matrix, Q=process_noise)
```

### C++ (`cpp/`)

Modern C++17 RAII wrappers for the C library.

**Features:**
- Exception-safe memory management
- Iterator support for state vectors
- Full STL compatibility
- MISRA C++ compliant where applicable

**Build & Test:**
```bash
cd cpp
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

**Example:**
```cpp
#include <stonesoup/stonesoup.hpp>

using namespace stonesoup;

// Create a 4D Gaussian state
GaussianState state(4);
state.state(0) = 0.0;  // x position
state.state(1) = 1.0;  // x velocity
state.state(2) = 0.0;  // y position
state.state(3) = 1.0;  // y velocity
state.set_covariance_identity();

// Iterate over state vector
for (const auto& val : state) {
    std::cout << val << " ";
}
```

### Java (`java/`)

Java bindings using Project Panama FFI for native interop.

**Requirements:** Java 21+ (for Panama FFI)

**Build:**
```bash
cd java
mvn package
```

**Example:**
```java
import com.stonesoup.StateVector;
import com.stonesoup.GaussianState;

// Create state
var state = new GaussianState(4);
state.setState(0, 0.0);  // x
state.setState(1, 1.0);  // vx
state.setTimestamp(System.currentTimeMillis() / 1000.0);
```

### Go (`go/`)

Go bindings using cgo for C interoperability.

**Build & Test:**
```bash
cd go
export CGO_LDFLAGS="-L../../libstonesoup/build -lstonesoup"
export CGO_CFLAGS="-I../../libstonesoup/include"
go build
go test
```

**Example:**
```go
package main

import "github.com/dstl/stonesoup/bindings/go/stonesoup"

func main() {
    // Create Gaussian state
    state := stonesoup.NewGaussianState(4)
    defer state.Free()

    state.SetState(0, 0.0)  // x position
    state.SetState(1, 1.0)  // x velocity
    state.SetCovarianceIdentity()
}
```

### Node.js (`nodejs/`)

JavaScript/TypeScript bindings using napi-rs.

**Build:**
```bash
cd nodejs
npm install
npm run build
```

**Example:**
```typescript
import { GaussianState, kalmanPredict } from 'stonesoup';

// Create initial state
const state = new GaussianState({
  stateVector: [0.0, 1.0, 0.0, 1.0],
  covariance: [
    [1, 0, 0, 0],
    [0, 0.1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0.1]
  ]
});

// Predict
const predicted = kalmanPredict(state, transitionMatrix, processNoise);
```

### Scilab (`scilab/`)

Scilab bindings for numerical computation and Xcos simulation integration.

**Requirements:** Scilab 6.0+, libstonesoup

**Build:**
```bash
cd scilab
# Open Scilab and run:
exec("builder.sce", -1);
```

**Example:**
```scilab
// Load Stone Soup
exec("loader.sce", -1);

// Create Gaussian state
sv = [0; 1; 0; 1];  // [x, vx, y, vy]
P = eye(4, 4);
gs = GaussianState(sv, P);

// Constant velocity model
dt = 1.0;
F = constant_velocity_transition(2, dt);
Q = 0.1 * eye(4, 4);

// Kalman prediction
gs_pred = kalman_predict(gs, F, Q);

// Measurement update
H = position_measurement_matrix(2);
R = 0.5 * eye(2, 2);
measurement = [1.1; 1.2];
gs_post = kalman_update(gs_pred, measurement, H, R);
```

### Ada (`ada/`)

Ada bindings with strong typing guarantees and SPARK contracts.

**Build:**
```bash
cd ada
gprbuild -P stonesoup.gpr
```

**Example:**
```ada
with Stone_Soup.Types; use Stone_Soup.Types;
with Stone_Soup.Kalman;

procedure Example is
   State : Gaussian_State := Create_Gaussian_State(4);
begin
   Set_State_Element(State, 1, 0.0);  -- x position
   Set_State_Element(State, 2, 1.0);  -- x velocity
   Set_Covariance_Identity(State);
   Free_Gaussian_State(State);
end Example;
```

## Cross-Language Usage Examples

### Common Tracking Scenario

All bindings implement the same core algorithm, allowing you to choose the best language for your deployment:

1. **Research & Prototyping**: Use Python bindings with full Stone Soup ecosystem
2. **High Performance**: Use Rust or C++ bindings
3. **Enterprise Java**: Use Java bindings with Panama FFI
4. **Embedded Systems**: Use Ada bindings with SPARK verification
5. **Web Services**: Use Node.js bindings for REST APIs

### Interoperability via File Formats

State can be serialized to JSON for cross-language communication:

```json
{
  "type": "GaussianState",
  "state_vector": [0.0, 1.0, 0.0, 1.0],
  "covariance": [[1,0,0,0],[0,0.1,0,0],[0,0,1,0],[0,0,0,0.1]],
  "timestamp": 1702401234.567
}
```

## Architecture

All bindings interface with a common C API layer (`libstonesoup`) that provides a stable ABI:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Application Code                                 │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────────┤
│  Python │   Rust  │   C++   │   Java  │   Go    │ Node.js │   Scilab    │
│  (PyO3) │(nalgebra)│ (RAII) │(Panama) │  (cgo)  │(napi-rs)│  (gateway)  │
├─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────────┤
│            Ada (SPARK)  │  MATLAB (MEX)  │  GNU Octave (MEX)            │
├─────────────────────────────────────────────────────────────────────────┤
│                        libstonesoup (C API)                              │
│                   - StateVector operations                               │
│                   - CovarianceMatrix operations                          │
│                   - Kalman filter                                        │
│                   - Particle filter                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

Each language binding:
1. Links to `libstonesoup` (shared library)
2. Wraps C functions with idiomatic language constructs
3. Provides language-specific error handling and memory management
4. Maintains type safety where applicable

## Building All Bindings

```bash
# Build C library first
cd libstonesoup && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON && make -j$(nproc)
cd ../..

# Build each binding
cd bindings/rust && cargo build --release && cd ../..
cd bindings/cpp && mkdir -p build && cd build && cmake .. && make && cd ../../..
# ... etc
```

## Contributing

When adding new bindings:
1. Follow the existing directory structure
2. Provide build configuration appropriate to the language ecosystem
3. Include tests demonstrating core functionality
4. Implement at minimum: StateVector, CovarianceMatrix, GaussianState, Kalman predict/update
5. Document installation and usage instructions
6. Follow language-specific best practices and linting
