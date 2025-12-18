"""Provides an ability to generate and load configuration from YAML.

Stone Soup utilises YAML_ for configuration files. The :doc:`stonesoup.base`
feature of components is exploited in order to store the configuration of the
components used for a run.

.. _YAML: http://yaml.org/"""

import os
from abc import ABC, abstractmethod
from io import StringIO


class Configuration:
    pass


class BackendConfiguration:
    """Configuration for the Stone Soup compute backend.

    This class provides programmatic configuration of the array backend
    used for numerical operations. The backend can be either NumPy (CPU)
    or CuPy (GPU).

    Configuration precedence (highest to lowest):
        1. Explicit call to :meth:`set_backend`
        2. STONESOUP_BACKEND environment variable
        3. Auto-detection based on availability

    Example:
        >>> from stonesoup.config import BackendConfiguration
        >>> config = BackendConfiguration()
        >>> config.set_backend("cupy")  # Force GPU
        >>> config.set_backend("numpy")  # Force CPU
        >>> config.set_backend("auto")  # Auto-detect
    """

    VALID_BACKENDS = ("numpy", "cupy", "auto")

    def __init__(self):
        """Initialize backend configuration."""
        self._backend = None
        self._auto_transfer = True
        self._gpu_memory_threshold = 0.9  # 90% memory threshold for fallback

    @property
    def backend(self) -> str:
        """Get the current backend name.

        Returns:
            str: Current backend ("numpy" or "cupy")
        """
        if self._backend is None:
            from stonesoup.backend import get_backend

            return get_backend()
        return self._backend

    def set_backend(self, backend: str) -> None:
        """Set the compute backend.

        Args:
            backend: Backend name ("numpy", "cupy", or "auto")

        Raises:
            ValueError: If backend is not valid
            ImportError: If requested backend is not available
        """
        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. " f"Valid options: {self.VALID_BACKENDS}"
            )

        from stonesoup.backend import set_backend as _set_backend

        if backend == "auto":
            # Let backend module auto-detect
            from stonesoup.backend import _detect_backend

            backend = _detect_backend()

        _set_backend(backend)
        self._backend = backend

    @property
    def auto_transfer(self) -> bool:
        """Whether to automatically transfer arrays between CPU/GPU.

        When True, operations will automatically transfer data between
        CPU and GPU as needed. When False, a mismatch raises an error.

        Returns:
            bool: Auto-transfer setting
        """
        return self._auto_transfer

    @auto_transfer.setter
    def auto_transfer(self, value: bool) -> None:
        """Set auto-transfer setting."""
        self._auto_transfer = bool(value)

    @property
    def gpu_memory_threshold(self) -> float:
        """GPU memory threshold for CPU fallback (0.0 to 1.0).

        When GPU memory usage exceeds this threshold, operations may
        fall back to CPU to avoid out-of-memory errors.

        Returns:
            float: Memory threshold (default 0.9 = 90%)
        """
        return self._gpu_memory_threshold

    @gpu_memory_threshold.setter
    def gpu_memory_threshold(self, value: float) -> None:
        """Set GPU memory threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Memory threshold must be between 0.0 and 1.0")
        self._gpu_memory_threshold = value

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU backend is currently enabled.

        Returns:
            bool: True if using CuPy/GPU backend
        """
        return self.backend == "cupy"

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available (even if not currently enabled).

        Returns:
            bool: True if CuPy and GPU are available
        """
        from stonesoup.backend import is_gpu_available

        return is_gpu_available()

    def get_info(self) -> dict:
        """Get current backend configuration info.

        Returns:
            dict: Configuration information
        """
        from stonesoup.backend import get_device_info

        info = get_device_info()
        info.update(
            {
                "auto_transfer": self._auto_transfer,
                "gpu_memory_threshold": self._gpu_memory_threshold,
            }
        )
        return info

    @classmethod
    def from_env(cls) -> "BackendConfiguration":
        """Create configuration from environment variables.

        Environment variables:
            - STONESOUP_BACKEND: Backend selection (numpy/cupy/auto)
            - STONESOUP_AUTO_TRANSFER: Enable auto CPU/GPU transfer (true/false)
            - STONESOUP_GPU_MEMORY_THRESHOLD: Memory threshold (0.0-1.0)

        Returns:
            BackendConfiguration: Configured instance
        """
        config = cls()

        # Backend selection
        backend = os.environ.get("STONESOUP_BACKEND", "auto").lower()
        if backend in cls.VALID_BACKENDS:
            config.set_backend(backend)

        # Auto-transfer setting
        auto_transfer = os.environ.get("STONESOUP_AUTO_TRANSFER", "true").lower()
        config.auto_transfer = auto_transfer in ("true", "1", "yes")

        # Memory threshold
        threshold_str = os.environ.get("STONESOUP_GPU_MEMORY_THRESHOLD")
        if threshold_str:
            try:
                config.gpu_memory_threshold = float(threshold_str)
            except ValueError:
                pass  # Ignore invalid values

        return config


# Global backend configuration instance
backend_config = BackendConfiguration()


class ConfigurationFile(ABC):
    """Base configuration class."""

    @abstractmethod
    def dump(self, data, stream, *args, **kwargs):
        """Dump configuration to a stream."""
        raise NotImplementedError

    def dumps(self, data, *args, **kwargs):
        """Return configuration as a string."""
        stream = StringIO()
        self.dump(data, stream, *args, **kwargs)
        return stream.getvalue()

    @abstractmethod
    def load(self, stream):
        """Load configuration from a stream."""
        raise NotImplementedError
