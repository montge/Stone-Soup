# Stone Soup Simulink Library User Guide

This guide explains how to use the Stone Soup Simulink library for target tracking applications.

## Installation

### Prerequisites

- MATLAB R2020a or later
- Simulink
- Control System Toolbox (recommended)

### Setup

1. Add the Stone Soup MATLAB bindings to your MATLAB path:
   ```matlab
   addpath('/path/to/stonesoup/bindings/matlab');
   addpath('/path/to/stonesoup/bindings/matlab/simulink');
   ```

2. If the library file doesn't exist, create it:
   ```matlab
   cd /path/to/stonesoup/bindings/matlab/simulink
   create_stonesoup_lib
   ```

3. Open the library browser in Simulink and look for "stonesoup_lib" or open it directly:
   ```matlab
   open_system('stonesoup_lib')
   ```

## Library Blocks

The Stone Soup Simulink library contains the following blocks:

### Kalman Predictor

Performs the Kalman filter prediction step.

**Inputs:**
- `x` - State vector (N x 1)
- `P` - Covariance matrix (vectorized, N² x 1)

**Outputs:**
- `x_pred` - Predicted state vector
- `P_pred` - Predicted covariance (vectorized)

**Mask Parameters:**
- `F` - State transition matrix (N x N)
- `Q` - Process noise covariance matrix (N x N)

**Equations:**
```
x_pred = F * x
P_pred = F * P * F' + Q
```

### Kalman Updater

Performs the Kalman filter update step with a measurement.

**Inputs:**
- `x` - Predicted state vector (N x 1)
- `P` - Predicted covariance (vectorized, N² x 1)
- `z` - Measurement vector (M x 1)

**Outputs:**
- `x_post` - Updated (posterior) state vector
- `P_post` - Updated covariance (vectorized)

**Mask Parameters:**
- `H` - Measurement matrix (M x N)
- `R` - Measurement noise covariance (M x M)

**Equations:**
```
y = z - H * x           % Innovation
S = H * P * H' + R      % Innovation covariance
K = P * H' / S          % Kalman gain
x_post = x + K * y      % Updated state
P_post = (I - K*H) * P * (I - K*H)' + K * R * K'  % Joseph form
```

### Constant Velocity Model

Generates transition matrix F and process noise Q for constant velocity motion.

**Outputs:**
- `F` - State transition matrix (vectorized)
- `Q` - Process noise covariance (vectorized)

**Mask Parameters:**
- `ndim` - Number of spatial dimensions (default: 2)
- `dt` - Time step in seconds (default: 0.1)
- `q` - Process noise intensity (default: 0.01)

For 2D motion, the state is `[x, vx, y, vy]'` (position and velocity).

### Gaussian State

Provides initial state and covariance for the Kalman filter.

**Outputs:**
- `x0` - Initial state vector
- `P0` - Initial covariance (vectorized)

**Mask Parameters:**
- `x0` - Initial state vector (N x 1)
- `P0` - Initial covariance matrix (N x N)

## Creating a Tracking Model

### Basic Structure

A typical Kalman filter tracking model has the following structure:

```
[Initial State] --> [Predictor] --> [Updater] --> [Output]
                         ^              |
                         |              v
                    [Delay] <---- [Measurements]
```

### Step-by-Step Guide

1. **Add Initial State Block**
   - Drag "Gaussian State" from the library
   - Set initial state vector `x0`
   - Set initial covariance `P0`

2. **Add Predictor Block**
   - Drag "Kalman Predictor" from the library
   - Set transition matrix `F` or use "Constant Velocity Model"
   - Set process noise `Q`

3. **Add Updater Block**
   - Drag "Kalman Updater" from the library
   - Set measurement matrix `H`
   - Set measurement noise `R`

4. **Add Measurement Source**
   - Use "From Workspace" block to load measurement data
   - Or connect to a sensor simulation

5. **Add Feedback Loop**
   - Add Unit Delay blocks for state and covariance
   - Connect updater outputs back to predictor inputs

6. **Add Output Blocks**
   - Add Scope or To Workspace blocks to visualize/save results

## Example: 2D Target Tracking

```matlab
% Simulation parameters
dt = 0.1;           % Time step
state_dim = 4;      % [x, vx, y, vy]
meas_dim = 2;       % [x, y]

% Constant velocity model
F = [1 dt 0  0;
     0  1 0  0;
     0  0 1 dt;
     0  0 0  1];

q = 0.1;
Q_block = [dt^3/3, dt^2/2; dt^2/2, dt] * q;
Q = blkdiag(Q_block, Q_block);

% Measurement model (position only)
H = [1 0 0 0;
     0 0 1 0];
R = 0.5 * eye(2);

% Initial state
x0 = [0; 1; 0; 0.5];
P0 = diag([1, 0.1, 1, 0.1]);
```

## Covariance Vectorization

The Simulink blocks use vectorized covariance matrices to allow scalar signal routing. A covariance matrix P of size N x N is converted to a vector of length N² using column-major order (MATLAB's default):

```matlab
% Matrix to vector
P_vec = P(:);

% Vector to matrix
P = reshape(P_vec, N, N);
```

## Running the Demo

A complete demo is provided in the `demos` directory:

```matlab
% Run the MATLAB script demo
cd /path/to/stonesoup/bindings/matlab/simulink/demos
tracking_demo

% Create the Simulink model demo
create_tracking_demo_model
```

## Troubleshooting

### Block Not Found

If blocks show as broken links:
1. Ensure `stonesoup_lib.slx` exists
2. Add the Simulink directory to MATLAB path
3. Reload the library: `load_system('stonesoup_lib')`

### Dimension Mismatch Errors

- Check that matrix dimensions match (F: NxN, Q: NxN, H: MxN, R: MxM)
- Ensure covariance inputs are vectorized (N² x 1)
- Verify state vector is column format (N x 1)

### Numerical Issues

- Use the Joseph form for covariance update (already implemented)
- Check for positive definiteness of Q, R, and P0
- Consider using `sqrtm` for very ill-conditioned problems

## License

SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
SPDX-License-Identifier: MIT
