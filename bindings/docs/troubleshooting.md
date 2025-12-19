# Stone Soup Bindings Troubleshooting Guide

This guide covers common issues and solutions when using Stone Soup language bindings.

## General Issues

### libstonesoup Not Found

**Symptoms:**
- "libstonesoup.so: cannot open shared object file"
- "Library not found" errors
- Linker errors during build

**Solutions:**

1. **Verify installation:**
   ```bash
   ls /usr/local/lib/libstonesoup*
   # Should show libstonesoup.so or similar
   ```

2. **Update library cache (Linux):**
   ```bash
   sudo ldconfig
   ```

3. **Set library path:**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   # Or for custom location:
   export LD_LIBRARY_PATH=/path/to/libstonesoup/build:$LD_LIBRARY_PATH
   ```

4. **macOS:**
   ```bash
   export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
   ```

5. **Windows:**
   - Add libstonesoup.dll location to PATH
   - Or copy DLL to application directory

### Header Files Not Found

**Symptoms:**
- "stonesoup/stonesoup.h: No such file or directory"
- Include path errors

**Solutions:**

1. **Check include path:**
   ```bash
   ls /usr/local/include/stonesoup/
   # Should show stonesoup.h
   ```

2. **Specify include path during build:**
   ```bash
   # Example for gcc
   gcc -I/path/to/libstonesoup/include ...
   ```

### Matrix Dimension Errors

**Symptoms:**
- "Matrix dimensions must agree"
- "Incompatible dimensions"

**Solutions:**

1. **Check state vector dimensions:**
   - State should be column vector (n x 1)
   - Not row vector (1 x n)

2. **Verify matrix shapes:**
   | Matrix | Shape | Description |
   |--------|-------|-------------|
   | x (state) | n x 1 | State vector |
   | P (covariance) | n x n | Must be square |
   | F (transition) | n x n | Must match state |
   | Q (process noise) | n x n | Must match state |
   | z (measurement) | m x 1 | Measurement vector |
   | H (measurement) | m x n | Maps state to measurement |
   | R (measurement noise) | m x m | Must match measurement |

3. **Debug dimensions:**
   ```python
   # Python
   print(f"x: {x.shape}, P: {P.shape}, F: {F.shape}")
   ```
   ```matlab
   % MATLAB/Octave
   disp(['x: ', mat2str(size(x)), ', P: ', mat2str(size(P))]);
   ```
   ```scilab
   // Scilab
   disp(size(x)); disp(size(P));
   ```

## Language-Specific Issues

### Python (PyO3)

**Issue: Import error after installation**
```python
ImportError: cannot import name 'stonesoup_native' from 'stonesoup'
```

**Solution:**
```bash
cd bindings/python
pip uninstall stonesoup-native
maturin develop --release
```

**Issue: NumPy compatibility**
```python
TypeError: expected numpy array
```

**Solution:**
```python
import numpy as np
# Ensure arrays are numpy arrays with correct dtype
x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
```

### Rust

**Issue: Linking errors**
```
error: linking with `cc` failed
```

**Solutions:**
1. Set library path:
   ```bash
   export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
   ```

2. Add to Cargo.toml:
   ```toml
   [build]
   rustflags = ["-L", "/usr/local/lib"]
   ```

**Issue: Runtime library not found**

**Solution:**
```bash
# Add to ~/.bashrc or equivalent
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Java

**Issue: UnsatisfiedLinkError**
```java
java.lang.UnsatisfiedLinkError: no stonesoup in java.library.path
```

**Solutions:**
1. Set java.library.path:
   ```bash
   java -Djava.library.path=/usr/local/lib MyApp
   ```

2. Or set LD_LIBRARY_PATH:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   java MyApp
   ```

### MATLAB/Octave

**Issue: MEX file not found**
```
Error using stonesoup_mex
Invalid MEX-file
```

**Solutions:**
1. Rebuild MEX file:
   ```matlab
   cd mex
   make('clean')
   make('all')
   ```

2. Check library path:
   ```matlab
   setenv('LD_LIBRARY_PATH', ['/usr/local/lib:', getenv('LD_LIBRARY_PATH')]);
   ```

**Issue: Octave mkoctfile fails**

**Solution:**
```bash
# Install development packages
sudo apt-get install octave-dev
```

### Scilab

**Issue: Gateway function undefined**
```
Undefined variable: stonesoup_state_vector_create
```

**Solutions:**
1. Load the module:
   ```scilab
   exec("loader.sce", -1);
   ```

2. Rebuild if needed:
   ```scilab
   exec("cleaner.sce", -1);
   exec("builder.sce", -1);
   exec("loader.sce", -1);
   ```

**Issue: Scilab crashes during build**

**Solutions:**
1. Check Scilab version (need 6.0+):
   ```scilab
   getversion()
   ```

2. Verify C compiler is available:
   ```bash
   gcc --version
   ```

### Go

**Issue: cgo errors**

**Solutions:**
1. Set CGO environment:
   ```bash
   export CGO_LDFLAGS="-L/usr/local/lib -lstonesoup"
   export CGO_CFLAGS="-I/usr/local/include"
   ```

2. Install pkg-config (if available):
   ```bash
   sudo apt-get install pkg-config
   ```

### Node.js

**Issue: Native module won't load**
```
Error: Cannot find module './stonesoup.node'
```

**Solutions:**
1. Rebuild:
   ```bash
   npm run build
   ```

2. Check Node.js version compatibility:
   ```bash
   node --version  # Should be 16+
   ```

### Ada

**Issue: gprbuild fails**

**Solutions:**
1. Install GNAT:
   ```bash
   sudo apt-get install gnat gprbuild
   ```

2. Set library path in project file:
   ```ada
   for Library_Options use ("-L/usr/local/lib", "-lstonesoup");
   ```

## Numerical Issues

### NaN or Inf Values

**Causes:**
- Covariance matrix became non-positive-definite
- Division by near-zero values
- Numerical overflow

**Solutions:**
1. Check covariance symmetry:
   ```python
   P = (P + P.T) / 2  # Force symmetry
   ```

2. Add small regularization:
   ```python
   P = P + 1e-10 * np.eye(P.shape[0])
   ```

3. Use Joseph form for covariance update (more stable):
   ```python
   # Instead of P = (I - K*H) * P
   I_KH = np.eye(n) - K @ H
   P = I_KH @ P @ I_KH.T + K @ R @ K.T
   ```

### Filter Divergence

**Symptoms:**
- Estimates diverge from truth
- Covariance grows unboundedly

**Solutions:**
1. Check process noise Q is not too small
2. Verify measurement noise R is realistic
3. Ensure transition matrix F is correct
4. Check for outlier measurements

## CI/Build Issues

### GitHub Actions Failures

**Issue: Tests fail in CI but pass locally**

**Solutions:**
1. Check environment differences
2. Ensure all dependencies are installed in CI
3. Use same compiler/runtime versions

**Issue: Coverage not reported**

**Solutions:**
1. Verify coverage tool is installed
2. Check coverage file paths in CI config
3. Ensure tests actually ran

## Getting More Help

1. **Check logs**: Most issues leave error messages
2. **Enable debug output**: Many bindings have verbose modes
3. **Search issues**: https://github.com/dstl/Stone-Soup/issues
4. **Ask for help**: Create a new issue with:
   - Operating system and version
   - Language/runtime version
   - Complete error message
   - Minimal reproduction steps
