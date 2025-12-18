# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Backend abstraction for array operations.

This module provides a backend abstraction layer that allows Stone Soup
to use different array libraries (NumPy, CuPy) transparently. The backend
can be selected automatically based on availability or explicitly via
environment variable.

Configuration:
    Set the STONESOUP_BACKEND environment variable to override auto-detection:
    - "numpy": Force NumPy backend (CPU)
    - "cupy": Force CuPy backend (GPU)
    - "auto": Auto-detect (default)

Example:
    >>> from stonesoup.backend import get_backend, get_array_module
    >>> backend = get_backend()
    >>> xp = get_array_module()
    >>> x = xp.array([1.0, 2.0, 3.0])
"""

import logging
import os

logger = logging.getLogger(__name__)

# Global backend state
_current_backend: str | None = None
_array_module = None

# Available backends
BACKENDS = ["numpy", "cupy"]
DEFAULT_BACKEND = "numpy"


def _detect_backend() -> str:
    """Detect the best available backend.

    Returns:
        str: Name of the best available backend ("numpy" or "cupy")
    """
    # Check environment variable override
    env_backend = os.environ.get("STONESOUP_BACKEND", "auto").lower()

    if env_backend != "auto":
        if env_backend in BACKENDS:
            return env_backend
        else:
            logger.warning(
                f"Unknown backend '{env_backend}' in STONESOUP_BACKEND. "
                f"Valid options: {BACKENDS}. Falling back to auto-detection."
            )

    # Try CuPy first for GPU acceleration
    try:
        import cupy  # noqa: F401

        # Verify GPU is actually available
        cupy.cuda.runtime.getDeviceCount()
        logger.info("CuPy backend detected with GPU support")
        return "cupy"
    except ImportError:
        logger.debug("CuPy not installed, using NumPy backend")
    except Exception as e:
        logger.debug("CuPy available but GPU not accessible: %s", e)

    # Fall back to NumPy
    logger.debug("Using NumPy backend (CPU)")
    return "numpy"


def get_backend() -> str:
    """Get the current backend name.

    Returns:
        str: Current backend name ("numpy" or "cupy")
    """
    global _current_backend
    if _current_backend is None:
        _current_backend = _detect_backend()
    return _current_backend


def set_backend(backend: str) -> None:
    """Set the array backend.

    Args:
        backend: Backend name ("numpy" or "cupy")

    Raises:
        ValueError: If backend is not recognized
        ImportError: If requested backend is not available
    """
    global _current_backend, _array_module

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Valid options: {BACKENDS}")

    if backend == "cupy":
        try:
            import cupy  # noqa: F401

            cupy.cuda.runtime.getDeviceCount()
        except ImportError:
            raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        except Exception as e:
            raise ImportError(f"CuPy available but GPU not accessible: {e}")

    _current_backend = backend
    _array_module = None  # Reset cached module
    logger.info(f"Backend set to: {backend}")


def get_array_module():
    """Get the array module for the current backend.

    Returns:
        module: NumPy or CuPy module

    Example:
        >>> xp = get_array_module()
        >>> x = xp.array([1.0, 2.0, 3.0])
        >>> y = xp.zeros((3, 3))
    """
    global _array_module

    if _array_module is None:
        backend = get_backend()
        if backend == "cupy":
            import cupy

            _array_module = cupy
        else:
            import numpy

            _array_module = numpy

    return _array_module


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        bool: True if CuPy is installed and GPU is accessible
    """
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, Exception):
        return False


def to_numpy(array):
    """Convert array to NumPy array.

    Args:
        array: NumPy or CuPy array

    Returns:
        numpy.ndarray: NumPy array
    """
    if hasattr(array, "get"):
        # CuPy array - transfer to CPU
        return array.get()
    return array


def to_gpu(array):
    """Convert array to GPU array (CuPy).

    Args:
        array: NumPy or CuPy array

    Returns:
        cupy.ndarray: CuPy array

    Raises:
        ImportError: If CuPy is not available
    """
    if not is_gpu_available():
        raise ImportError("GPU not available. Install CuPy: pip install cupy-cuda12x")

    import cupy

    if isinstance(array, cupy.ndarray):
        return array
    return cupy.asarray(array)


def get_device_info() -> dict:
    """Get information about available compute devices.

    Returns:
        dict: Device information including GPU name, memory, etc.
    """
    info = {
        "backend": get_backend(),
        "gpu_available": is_gpu_available(),
        "gpu_devices": [],
    }

    if is_gpu_available():
        import cupy

        num_devices = cupy.cuda.runtime.getDeviceCount()
        for i in range(num_devices):
            with cupy.cuda.Device(i):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                info["gpu_devices"].append(
                    {
                        "id": i,
                        "name": (
                            props["name"].decode()
                            if isinstance(props["name"], bytes)
                            else props["name"]
                        ),
                        "total_memory_mb": props["totalGlobalMem"] // (1024 * 1024),
                    }
                )

    return info
