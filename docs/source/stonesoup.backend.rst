Backend
=======

.. automodule:: stonesoup.backend
    :members:
    :undoc-members:
    :show-inheritance:

NumPy Backend
-------------

.. automodule:: stonesoup.backend.numpy_backend
    :members:
    :undoc-members:
    :show-inheritance:

CuPy Backend
------------

.. automodule:: stonesoup.backend.cupy_backend
    :members:
    :undoc-members:
    :show-inheritance:

GPU Acceleration Guide
----------------------

Stone Soup supports GPU acceleration via CuPy for computationally intensive
operations like batch Kalman filtering and particle filter resampling.

Installation
~~~~~~~~~~~~

To enable GPU acceleration, install CuPy for your CUDA version:

.. code-block:: bash

    # For CUDA 12.x
    pip install cupy-cuda12x

    # For CUDA 11.x
    pip install cupy-cuda11x

Backend Selection
~~~~~~~~~~~~~~~~~

Stone Soup automatically detects and uses the GPU when available. You can also
control the backend manually:

.. code-block:: python

    from stonesoup.backend import get_backend, set_backend, is_gpu_available

    # Check GPU availability
    if is_gpu_available():
        print("GPU is available!")

    # Auto-detected backend (prefers GPU)
    backend = get_backend()  # Returns "cupy" or "numpy"

    # Force a specific backend
    set_backend("numpy")  # Force CPU
    set_backend("cupy")   # Force GPU (raises ImportError if unavailable)

Environment Variable
~~~~~~~~~~~~~~~~~~~~

You can also control the backend via environment variable:

.. code-block:: bash

    # Force NumPy (CPU)
    export STONESOUP_BACKEND=numpy

    # Force CuPy (GPU)
    export STONESOUP_BACKEND=cupy

    # Auto-detect (default)
    export STONESOUP_BACKEND=auto

Array Operations
~~~~~~~~~~~~~~~~

Use ``get_array_module()`` to get the appropriate array library:

.. code-block:: python

    from stonesoup.backend import get_array_module

    xp = get_array_module()  # Returns numpy or cupy
    x = xp.array([1.0, 2.0, 3.0])
    y = xp.zeros((3, 3))

Data Transfer
~~~~~~~~~~~~~

Transfer arrays between CPU and GPU:

.. code-block:: python

    from stonesoup.backend import to_numpy, to_gpu
    import numpy as np

    # CPU to GPU
    arr_cpu = np.array([1.0, 2.0, 3.0])
    arr_gpu = to_gpu(arr_cpu)

    # GPU to CPU
    arr_cpu = to_numpy(arr_gpu)

Performance
~~~~~~~~~~~

GPU acceleration provides significant speedups for large batch operations:

+---------------+----------+----------+---------+
| Operation     | CPU (ms) | GPU (ms) | Speedup |
+===============+==========+==========+=========+
| Kalman 1k     | 3.4      | 0.6      | 5.8x    |
+---------------+----------+----------+---------+
| Kalman 10k    | 33       | 2.0      | 16x     |
+---------------+----------+----------+---------+
| Kalman 50k    | 166      | 5.7      | 29x     |
+---------------+----------+----------+---------+
| Resample 100k | 2.3      | 0.3      | 8.6x    |
+---------------+----------+----------+---------+

*Benchmarks on NVIDIA RTX A2000 with 6D state vectors*

Running GPU Tests
~~~~~~~~~~~~~~~~~

Run the GPU test suite locally:

.. code-block:: bash

    # Show GPU info
    python scripts/run_gpu_tests.py --info

    # Run tests
    python scripts/run_gpu_tests.py --test

    # Run benchmarks
    python scripts/run_gpu_tests.py --bench

Troubleshooting
~~~~~~~~~~~~~~~

**CuPy not found**

Ensure CuPy is installed for your CUDA version:

.. code-block:: bash

    pip install cupy-cuda12x  # Adjust for your CUDA version

**GPU not detected**

Check that NVIDIA drivers are installed and CUDA is available:

.. code-block:: bash

    nvidia-smi

**Out of memory errors**

For large batch operations, consider processing in smaller chunks:

.. code-block:: python

    # Process in batches of 10,000
    chunk_size = 10000
    for i in range(0, len(states), chunk_size):
        batch = states[i:i+chunk_size]
        process(batch)
