# Stone Soup C Library Fuzzing

This directory contains fuzzing harnesses for testing the robustness of the
Stone Soup C library API boundaries.

## Overview

Fuzzing helps find:
- Memory safety issues (buffer overflows, use-after-free)
- Undefined behavior
- Crash bugs from unexpected inputs
- Edge cases in numerical code

## Building with libFuzzer (Clang)

```bash
# From libstonesoup directory
mkdir build-fuzz && cd build-fuzz
cmake -DCMAKE_C_COMPILER=clang -DENABLE_FUZZING=ON ..
make fuzz_kalman
```

## Building with AFL

```bash
# Install AFL++
apt install afl++

# Build with AFL compiler
cd fuzz
afl-gcc -g -O1 -fsanitize=address,undefined \
    -I../include fuzz_kalman.c ../src/*.c -lm -o fuzz_kalman_afl
```

## Running libFuzzer

```bash
# Basic run (runs until stopped with Ctrl+C)
./fuzz_kalman corpus/

# Time-limited run (60 seconds)
./fuzz_kalman -max_total_time=60 corpus/

# With options
./fuzz_kalman \
    -max_len=1024 \        # Maximum input size
    -jobs=4 \              # Number of parallel jobs
    -workers=4 \           # Number of worker processes
    -max_total_time=3600 \ # Run for 1 hour
    corpus/
```

## Running AFL

```bash
# Create input corpus
mkdir -p corpus findings
echo -ne '\x02\x00' > corpus/seed1

# Run AFL
afl-fuzz -i corpus/ -o findings/ ./fuzz_kalman_afl @@
```

## Fuzz Targets

### `fuzz_kalman.c`

Tests the following API boundaries:

1. **State Vector Operations**
   - `stonesoup_state_vector_create`
   - `stonesoup_vector_add`, `stonesoup_vector_subtract`
   - `stonesoup_vector_scale`, `stonesoup_vector_dot`

2. **Matrix Operations**
   - `stonesoup_covariance_matrix_create`
   - `stonesoup_matrix_add`, `stonesoup_matrix_multiply`
   - `stonesoup_matrix_cholesky`, `stonesoup_matrix_inverse`

3. **Kalman Filter**
   - `stonesoup_kalman_predict`
   - `stonesoup_kalman_update`

4. **Null Pointer Handling**
   - All free functions with NULL
   - Operations with NULL arguments

## Interpreting Results

### Crash Files

When a crash is found, libFuzzer saves the input to a file named
`crash-<hash>` in the current directory. Analyze with:

```bash
# Reproduce the crash
./fuzz_kalman crash-abc123

# Debug with GDB
gdb ./fuzz_kalman
(gdb) run crash-abc123
```

### Coverage

To get coverage information:

```bash
# Build with coverage
cmake -DCMAKE_C_COMPILER=clang \
      -DENABLE_FUZZING=ON \
      -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ..
make fuzz_kalman

# Run fuzzing
./fuzz_kalman -max_total_time=60 corpus/

# Generate coverage report
llvm-profdata merge -sparse default.profraw -o default.profdata
llvm-cov show ./fuzz_kalman -instr-profile=default.profdata
```

## CI Integration

Add to CI pipeline:

```yaml
fuzz-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install Clang
      run: sudo apt-get install -y clang
    - name: Build fuzzer
      run: |
        cd libstonesoup
        mkdir build && cd build
        cmake -DCMAKE_C_COMPILER=clang -DENABLE_FUZZING=ON ..
        make fuzz_kalman
    - name: Run fuzzing (5 minutes)
      run: |
        cd libstonesoup/build
        ./fuzz_kalman -max_total_time=300 corpus/
```
