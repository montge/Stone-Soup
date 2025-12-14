# GNU Octave Compatibility Guide

This document describes the compatibility of Stone Soup MATLAB bindings with GNU Octave, including known limitations and workarounds.

## Overview

The Stone Soup MATLAB bindings are designed to be compatible with GNU Octave where possible. Octave provides a free, open-source alternative to MATLAB with high compatibility for numerical computing.

## Compatibility Status

| Feature | MATLAB | Octave | Notes |
|---------|--------|--------|-------|
| MEX functions | ✅ | ✅ | Use mkoctfile |
| Core classes | ✅ | ✅ | StateVector, GaussianState |
| Kalman filter | ✅ | ✅ | predict, update |
| Simulink blocks | ✅ | ❌ | Use Xcos instead |
| S-functions | ✅ | ❌ | Not supported |
| Class packages (+) | ✅ | ⚠️ | Limited support |
| Graphics | ✅ | ⚠️ | Minor differences |

## Building for Octave

### Prerequisites

- GNU Octave 6.0 or later
- Octave development headers (`octave-dev` package)
- C compiler (gcc)
- libstonesoup compiled and installed

### Build Process

```bash
# Install Octave development tools (Ubuntu/Debian)
sudo apt-get install octave octave-dev

# Build libstonesoup first
cd libstonesoup
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..

# Build Octave MEX files
cd bindings/matlab
octave --eval "run('make_octave.m')"
```

### Manual MEX Build

If the automatic build fails:

```bash
cd bindings/matlab/mex
mkoctfile --mex \
    -I../../../libstonesoup/include \
    -L../../../libstonesoup/build \
    -lstonesoup \
    -o stonesoup_mex \
    stonesoup_mex.c
```

## Known Limitations

### 1. No Simulink Support

**Issue**: Octave does not include Simulink or compatible simulation environment.

**Workaround**: Use Scilab's Xcos instead:
- Xcos provides similar visual block-diagram simulation
- Stone Soup provides an Xcos palette with equivalent blocks
- See `bindings/scilab/docs/xcos-guide.md`

### 2. Package Namespace Differences

**Issue**: MATLAB's `+package` directories work differently in Octave.

**Workaround**: Use fully qualified function names:
```octave
% Instead of:
stonesoup.kalman_predict(...)

% Use:
addpath('path/to/+stonesoup');
kalman_predict(...)
```

### 3. Class Property Access

**Issue**: Some MATLAB class features may not work identically.

**Workaround**: Use getter/setter methods when available:
```octave
% MATLAB style (may not work in Octave)
state.mean = new_value;

% Octave-compatible
state = set_mean(state, new_value);
```

### 4. Graphics Differences

**Issue**: Plot appearance and some graphics functions differ.

**Workaround**: Use basic plotting functions:
```octave
% Works in both
plot(x, y, 'b-');
xlabel('X');
ylabel('Y');
title('Plot');

% MATLAB-specific (may not work)
set(gca, 'PropertyName', value);
```

### 5. MEX File Extensions

**Issue**: MEX files have different extensions.

| Platform | MATLAB | Octave |
|----------|--------|--------|
| Linux | .mexa64 | .mex |
| macOS | .mexmaci64 | .mex |
| Windows | .mexw64 | .mex |

**Workaround**: Build separately for each environment.

### 6. Function Handles in Structures

**Issue**: Function handles stored in structures may behave differently.

**Workaround**: Pass function handles as separate arguments:
```octave
% Instead of:
config.transition_func = @my_transition;
result = track(config);

% Use:
result = track(config, @my_transition);
```

## Testing Octave Compatibility

### Running Tests

```octave
% Run from bindings/matlab directory
cd tests
test_gaussian_state
test_kalman_filter
```

### Compatibility Test Script

```octave
% test_octave_compatibility.m
disp('Testing Stone Soup Octave Compatibility');
disp('========================================');

% Test 1: Basic operations
disp('Test 1: StateVector creation');
try
    sv = [1; 2; 3; 4];
    disp('  PASS');
catch
    disp('  FAIL');
end

% Test 2: Gaussian state
disp('Test 2: GaussianState creation');
try
    mean = [0; 0; 0; 0];
    covar = eye(4);
    gs = struct('mean', mean, 'covar', covar);
    disp('  PASS');
catch
    disp('  FAIL');
end

% Test 3: Kalman prediction
disp('Test 3: Kalman prediction');
try
    x = [0; 1; 0; 1];
    P = eye(4);
    F = [1 0.1 0 0; 0 1 0 0; 0 0 1 0.1; 0 0 0 1];
    Q = 0.01 * eye(4);
    x_pred = F * x;
    P_pred = F * P * F' + Q;
    disp('  PASS');
catch
    disp('  FAIL');
end

% Test 4: Kalman update
disp('Test 4: Kalman update');
try
    z = [0.1; 0.2];
    H = [1 0 0 0; 0 0 1 0];
    R = 0.1 * eye(2);
    y = z - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_upd = x_pred + K * y;
    P_upd = (eye(4) - K * H) * P_pred;
    disp('  PASS');
catch
    disp('  FAIL');
end

disp('');
disp('Compatibility test complete.');
```

## Performance Considerations

### MEX vs Pure Octave

For small state vectors (< 10 elements), pure Octave code may be faster due to MEX call overhead. For larger problems, MEX provides significant speedup.

### Memory Management

Octave's memory management is similar to MATLAB, but:
- `clear` may behave slightly differently
- Large arrays should be pre-allocated
- Use `sparse` matrices when appropriate

## Recommended Workflow

1. **Develop in MATLAB** if available for initial prototyping
2. **Test in Octave** before deployment
3. **Use conditional code** for platform-specific features:

```octave
if exist('OCTAVE_VERSION', 'builtin')
    % Octave-specific code
    disp('Running in Octave');
else
    % MATLAB-specific code
    disp('Running in MATLAB');
end
```

## Getting Help

- Octave documentation: https://octave.org/doc/
- Octave-MATLAB compatibility: https://wiki.octave.org/FAQ#MATLAB_compatibility
- Stone Soup issues: https://github.com/dstl/Stone-Soup/issues
