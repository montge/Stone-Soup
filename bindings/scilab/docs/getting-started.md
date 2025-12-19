# Stone Soup Scilab Bindings - Getting Started

This guide will help you get started with the Stone Soup Scilab bindings for target tracking applications.

## Prerequisites

- **Scilab 6.0 or later** (tested with 6.1.x and 2024.x)
- **C compiler** (gcc on Linux/macOS, MSVC on Windows)
- **libstonesoup** compiled and installed

## Installation

### Step 1: Build libstonesoup

First, build the core C library:

```bash
cd libstonesoup
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig  # Linux only
```

### Step 2: Build Scilab Bindings

Open Scilab and run:

```scilab
cd path/to/bindings/scilab
exec("builder.sce", -1);
```

This compiles the gateway functions and prepares the macros.

### Step 3: Load the Module

After building, load the module:

```scilab
exec("loader.sce", -1);
```

## Quick Start Example

Here's a complete example of tracking a moving object using a Kalman filter:

```scilab
// Load Stone Soup
exec("loader.sce", -1);

// Initial state: [x, vx, y, vy] = [0, 10, 0, 5]
// Object at origin, moving at 10 m/s in x and 5 m/s in y
initial_state = [0; 10; 0; 5];
initial_covar = diag([100, 10, 100, 10]);  // Uncertainty

// Create Gaussian state
state = GaussianState(initial_state, initial_covar);

// Constant velocity transition model
dt = 1.0;  // 1 second time step
F = [1, dt, 0, 0;   // x = x + vx*dt
     0, 1,  0, 0;   // vx = vx
     0, 0,  1, dt;  // y = y + vy*dt
     0, 0,  0, 1];  // vy = vy

// Process noise (acceleration uncertainty)
q = 0.1;  // acceleration noise
Q = q * [dt^3/3, dt^2/2, 0,      0;
         dt^2/2, dt,     0,      0;
         0,      0,      dt^3/3, dt^2/2;
         0,      0,      dt^2/2, dt];

// Measurement model (we can only measure position)
H = [1, 0, 0, 0;   // measure x
     0, 0, 1, 0];  // measure y

// Measurement noise
R = diag([1, 1]);  // 1 meter standard deviation

// Simulate some measurements
true_positions = [
    10, 5;    // t=1
    20, 10;   // t=2
    30, 15;   // t=3
    40, 20;   // t=4
    50, 25    // t=5
];

// Add noise to measurements
measurements = true_positions + 0.5 * rand(5, 2, "normal");

// Run the Kalman filter
disp("Running Kalman filter...");
disp("========================");

for t = 1:5
    // Predict
    [x_pred, P_pred] = kalman_predict(state.mean, state.covar, F, Q);

    // Get measurement
    z = measurements(t, :)';

    // Update
    [x_upd, P_upd] = kalman_update(x_pred, P_pred, z, H, R);

    // Store updated state
    state.mean = x_upd;
    state.covar = P_upd;

    // Display results
    mprintf("Time %d:\n", t);
    mprintf("  Measurement: [%.2f, %.2f]\n", z(1), z(2));
    mprintf("  Estimated:   [%.2f, %.2f]\n", x_upd(1), x_upd(3));
    mprintf("  Velocity:    [%.2f, %.2f]\n", x_upd(2), x_upd(4));
    mprintf("\n");
end

disp("Tracking complete!");
```

## Core Functions

### State Vector Operations

```scilab
// Create a state vector
sv = StateVector(4);           // 4D zero vector
sv = StateVector(4, 1.0);      // 4D vector filled with 1.0
sv = StateVector([1; 2; 3]);   // From existing data

// Operations
n = sv_norm(sv);               // Euclidean norm
sv3 = sv_add(sv1, sv2);        // Vector addition
sv3 = sv_subtract(sv1, sv2);   // Vector subtraction
sv2 = sv_scale(sv1, 2.0);      // Scalar multiplication
```

### Gaussian State

```scilab
// Create a Gaussian state
mean = [0; 0; 0; 0];
covar = eye(4, 4);
gs = GaussianState(mean, covar);

// Access components
gs.mean   // State vector
gs.covar  // Covariance matrix
```

### Kalman Filter

```scilab
// Prediction step
[x_pred, P_pred] = kalman_predict(x, P, F, Q);

// Update step
[x_upd, P_upd] = kalman_update(x_pred, P_pred, z, H, R);

// Using GaussianState directly
gs_pred = kalman_predict(gs, F, Q);
gs_upd = kalman_update(gs_pred, z, H, R);
```

## Using Xcos Blocks

Stone Soup provides Xcos palette blocks for visual simulation design.

### Loading the Palette

```scilab
// Load Stone Soup module
exec("loader.sce", -1);

// Load Xcos palette
exec("xcos/loader.sce", -1);

// Open Xcos
xcos();
```

### Available Blocks

1. **Kalman Predictor**: Performs prediction step
   - Inputs: State vector, Covariance matrix
   - Parameters: Transition matrix F, Process noise Q
   - Outputs: Predicted state, Predicted covariance

2. **Kalman Updater**: Performs measurement update
   - Inputs: Predicted state, Predicted covariance, Measurement
   - Parameters: Measurement matrix H, Measurement noise R
   - Outputs: Updated state, Updated covariance

3. **Constant Velocity Model**: Generates CV transition matrix
   - Parameters: Number of dimensions, Time step
   - Outputs: Transition matrix F, Process noise Q

## Common Patterns

### Multi-Target Tracking

```scilab
// Initialize multiple tracks
num_targets = 3;
tracks = list();
for i = 1:num_targets
    tracks(i) = GaussianState(zeros(4, 1), 100*eye(4, 4));
end

// Update each track with its measurement
for i = 1:num_targets
    [x_pred, P_pred] = kalman_predict(tracks(i).mean, tracks(i).covar, F, Q);
    [x_upd, P_upd] = kalman_update(x_pred, P_pred, measurements(i, :)', H, R);
    tracks(i).mean = x_upd;
    tracks(i).covar = P_upd;
end
```

### Saving and Loading States

```scilab
// Save state to file
state_data = struct("mean", gs.mean, "covar", gs.covar);
save("track_state.sod", "state_data");

// Load state from file
load("track_state.sod");
gs = GaussianState(state_data.mean, state_data.covar);
```

## Troubleshooting

### Build Errors

**"libstonesoup not found"**
- Ensure libstonesoup is installed: `sudo make install`
- Update library cache: `sudo ldconfig`
- Check `LD_LIBRARY_PATH` includes the library location

**"MEX compiler not found"**
- Install a C compiler (gcc, clang, or MSVC)
- On Windows, ensure compiler is in PATH

### Runtime Errors

**"Undefined function"**
- Run `exec("loader.sce", -1)` to load the module
- Verify the build completed successfully

**"Matrix dimension mismatch"**
- Ensure state vectors are column vectors (n x 1)
- Verify matrix dimensions match: F is (n x n), H is (m x n)

## Next Steps

- Explore the demo files in `demos/`
- Try the Xcos palette for visual simulation
- Read the API documentation with `help StateVector`

## Support

For issues and questions:
- GitHub: https://github.com/dstl/Stone-Soup/issues
- Documentation: https://stonesoup.readthedocs.io/
