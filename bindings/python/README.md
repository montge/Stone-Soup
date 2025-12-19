# Stone Soup PyO3 Bindings

High-performance Python bindings for the Stone Soup tracking framework,
built with PyO3 and Rust.

## Overview

This package provides Rust-powered implementations of core Stone Soup types
and algorithms, offering significant performance improvements for
computationally intensive operations like Kalman filtering and coordinate
transformations.

## Installation

```bash
pip install stonesoup-core
```

## Usage

```python
from stonesoup_core import StateVector, CovarianceMatrix

# Create a state vector
state = StateVector([1.0, 2.0, 3.0, 4.0])

# Perform operations with NumPy-compatible interface
import numpy as np
arr = np.array(state)
```

## Features

- **NumPy Integration**: Seamless conversion between Rust types and NumPy arrays
- **Zero-Copy When Possible**: Efficient memory sharing with Python
- **Thread-Safe**: Safe concurrent access from multiple Python threads
- **Type-Safe**: Full type annotations for IDE support

## Building from Source

Requires Rust 1.70+ and maturin:

```bash
cd bindings/python
pip install maturin
maturin develop
```

## License

MIT License - See LICENSE file in the repository root.
