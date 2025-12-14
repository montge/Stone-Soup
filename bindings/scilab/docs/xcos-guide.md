# Stone Soup Xcos User Guide

This guide covers using Stone Soup tracking blocks in Scilab's Xcos visual simulation environment.

## Overview

Xcos is Scilab's graphical modeling and simulation environment (similar to Simulink). Stone Soup provides a palette of tracking blocks that can be used to build visual simulations of target tracking systems.

## Getting Started

### Loading the Palette

```scilab
// Load Stone Soup module
cd path/to/bindings/scilab
exec("loader.sce", -1);

// Load Xcos palette
exec("xcos/loader.sce", -1);

// Launch Xcos
xcos();
```

### Finding Stone Soup Blocks

1. Open Xcos from Scilab
2. In the Palette Browser (View → Palette Browser)
3. Look for "Stone Soup" palette
4. Drag blocks into your model

## Available Blocks

### STONESOUP_KALMAN_PREDICT

Performs the Kalman filter prediction step.

**Inputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | x | State vector (n x 1) |
| 2 | P | Covariance matrix (n x n) |

**Outputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | x_pred | Predicted state (n x 1) |
| 2 | P_pred | Predicted covariance (n x n) |

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| F | Transition matrix (n x n) | eye(4,4) |
| Q | Process noise covariance (n x n) | 0.01*eye(4,4) |

**Example Configuration:**
```
// For a 2D constant velocity model (4 states: x, vx, y, vy)
dt = 0.1;
F = [1, dt, 0, 0; 0, 1, 0, 0; 0, 0, 1, dt; 0, 0, 0, 1];
Q = 0.01 * eye(4, 4);
```

### STONESOUP_KALMAN_UPDATE

Performs the Kalman filter measurement update step.

**Inputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | x_pred | Predicted state (n x 1) |
| 2 | P_pred | Predicted covariance (n x n) |
| 3 | z | Measurement vector (m x 1) |

**Outputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | x_upd | Updated state (n x 1) |
| 2 | P_upd | Updated covariance (n x n) |
| 3 | innovation | Measurement innovation (m x 1) |

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| H | Measurement matrix (m x n) | [1,0,0,0; 0,0,1,0] |
| R | Measurement noise covariance (m x m) | eye(2,2) |

### STONESOUP_CONSTANT_VELOCITY

Generates transition matrix and process noise for constant velocity model.

**Inputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | dt | Time step (scalar) |

**Outputs:**
| Port | Name | Description |
|------|------|-------------|
| 1 | F | Transition matrix |
| 2 | Q | Process noise matrix |

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| ndim | Number of spatial dimensions | 2 |
| q | Process noise intensity | 0.1 |

## Building a Complete Tracker

### Step 1: Create the Model Structure

```
[Clock] → [dt] → [CONSTANT_VELOCITY] → [F, Q]
                                          ↓
[Initial State] → [KALMAN_PREDICT] ← ────┘
                         ↓
                    [x_pred, P_pred]
                         ↓
[Measurements] → [KALMAN_UPDATE]
                         ↓
                   [x_upd, P_upd]
                         ↓
                    [Display/Scope]
```

### Step 2: Configure Blocks

1. **Clock Block**: Set sample time to match your sensor rate
2. **Constant Velocity**: Set dimensions (2 for 2D, 3 for 3D)
3. **Kalman Predict**: F and Q from Constant Velocity output
4. **Kalman Update**: Configure H for your sensor type

### Step 3: Connect Feedback Loop

For continuous tracking, connect the output back to the input:
- Use a Unit Delay block between update output and predict input
- This creates the recursive estimation loop

## Example: 2D Position Tracking

### Model Description

Track a target moving in 2D using position-only measurements.

### Block Configuration

**Initial State Source:**
```scilab
x0 = [0; 0; 0; 0];  // [x, vx, y, vy]
P0 = diag([100, 10, 100, 10]);
```

**Constant Velocity Block:**
- ndim = 2 (2D tracking)
- q = 0.1 (process noise intensity)

**Kalman Update Block:**
```scilab
H = [1, 0, 0, 0;    // Measure x position
     0, 0, 1, 0];   // Measure y position
R = diag([1, 1]);   // 1m measurement noise
```

### Running the Simulation

1. Set simulation time (e.g., 100 seconds)
2. Configure solver (ODE or discrete)
3. Click "Start Simulation"
4. View results in scope blocks

## Advanced Topics

### Variable Time Steps

For irregular sensor updates, pass dt as an input:

```
[Sensor Trigger] → [Time Diff] → [dt input to CONSTANT_VELOCITY]
```

### Multiple Sensors

Fuse measurements from multiple sensors:

```
[Sensor 1] →┐
            ├→ [Measurement Selector] → [KALMAN_UPDATE]
[Sensor 2] →┘
```

Configure H and R based on which sensor provides data.

### Track Management

For multi-target tracking, use Scilab's superblock feature:
1. Create a single-target tracker as a superblock
2. Instantiate multiple copies
3. Add data association logic

## Troubleshooting

### "Block not found"
- Ensure palette is loaded: `exec("xcos/loader.sce", -1);`
- Restart Xcos after loading

### "Matrix dimension error"
- Verify state dimension matches between blocks
- Check that F, Q, H, R dimensions are consistent

### "Simulation unstable"
- Reduce process noise Q
- Check measurement noise R is not too small
- Verify transition matrix F is correct

### "No output from blocks"
- Check all required inputs are connected
- Verify input signals have correct dimensions
- Add scope blocks to debug intermediate values

## Best Practices

1. **Start Simple**: Begin with a basic 1D tracker, then extend
2. **Validate Incrementally**: Test each block individually
3. **Use Scopes**: Add scope blocks at each stage for debugging
4. **Document Parameters**: Keep notes on F, Q, H, R values
5. **Save Often**: Xcos models can be complex; save regularly

## Demo Models

Example Xcos models are available in `demos/`:

- `xcos_kalman_demo.sce` - Basic Kalman filter demonstration
- Load with: `exec("demos/xcos_kalman_demo.sce", -1);`

## See Also

- [Getting Started Guide](getting-started.md)
- [Scilab API Reference](../help/en_US/)
- [Stone Soup Documentation](https://stonesoup.readthedocs.io/)
