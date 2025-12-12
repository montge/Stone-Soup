"""Coordinate System Types
=========================

Reference frames and ellipsoids for coordinate transformations.

This module provides:

- Reference ellipsoids (WGS84, GRS80, etc.) for geodetic calculations
- Reference frames (GCRS, J2000, ICRS) for celestial coordinate systems
- Kinematic state representations with position, velocity, and acceleration
- Frame transformation composition system for chaining transformations

Reference Ellipsoids
--------------------

Reference ellipsoids are mathematical models of the Earth's shape used for
geodetic coordinate transformations. The choice of ellipsoid affects position
accuracy, particularly for vertical (altitude) coordinates.

**Accuracy Characteristics:**

+-------------+------------------+------------------+---------------------------+
| Ellipsoid   | Semi-major (m)   | 1/f              | Notes                     |
+=============+==================+==================+===========================+
| WGS84       | 6378137.0        | 298.257223563    | GPS standard, global      |
+-------------+------------------+------------------+---------------------------+
| GRS80       | 6378137.0        | 298.257222101    | ITRS basis, ~0.1mm diff   |
+-------------+------------------+------------------+---------------------------+
| WGS72       | 6378135.0        | 298.26           | Legacy GPS, 2m offset     |
+-------------+------------------+------------------+---------------------------+
| PZ-90       | 6378136.0        | 298.257839303    | GLONASS, ~1m from WGS84   |
+-------------+------------------+------------------+---------------------------+
| CGCS2000    | 6378137.0        | 298.257222101    | China BeiDou, = GRS80     |
+-------------+------------------+------------------+---------------------------+

**Precision Considerations:**

1. **WGS84 vs GRS80**: The difference in flattening (1.46e-11) results in
   ~0.1 mm difference at the poles. For practical purposes, they are identical.

2. **WGS84 Realizations**: Different WGS84 realizations (G730, G873, G1150,
   G1674, G1762, G2139) have the same defining parameters but different
   reference frame origins and orientations at the centimeter level.

3. **Coordinate Accuracy**: Using the correct ellipsoid for your application:

   - GPS/GNSS: Use WGS84 (matches broadcast ephemeris)
   - GLONASS: Use PZ-90.11 (or WGS84 with ~1m error)
   - BeiDou: Use CGCS2000 (compatible with GRS80/WGS84)
   - High-precision geodesy: Use ITRF ellipsoid (GRS80-based)

4. **Vertical Accuracy**: Ellipsoid height differs from orthometric height
   (above mean sea level) by the geoid undulation, which can be ±100 m.

**Numerical Precision:**

- Semi-major axis: Defined to 0.1 mm precision
- Flattening: Defined to ~15 significant figures
- Derived quantities (b, e, e') maintain full double precision
- Iterative algorithms (ecef_to_geodetic) converge to 1e-12 radians (~0.006 mm)

Choosing an Ellipsoid
---------------------

For most Stone Soup applications, use the default ``WGS84`` ellipsoid:

.. code-block:: python

    from stonesoup.types.coordinates import WGS84
    from stonesoup.functions.coordinates import geodetic_to_ecef

    # Convert geodetic to ECEF using WGS84
    xyz = geodetic_to_ecef(lat_rad, lon_rad, alt_m, ellipsoid=WGS84)

Use a specific realization for high-precision applications:

.. code-block:: python

    from stonesoup.types.coordinates import WGS84_G2139

    # Use latest WGS84 realization for precision work
    xyz = geodetic_to_ecef(lat_rad, lon_rad, alt_m, ellipsoid=WGS84_G2139)

"""
from typing import ClassVar, Optional, Callable, Dict, List, Tuple, Type
from datetime import datetime
from abc import abstractmethod
from collections import defaultdict

import numpy as np

from ..base import Base, Property
from .array import StateVector


# =============================================================================
# Frame Transformation Composition System
# =============================================================================

class TransformationPath:
    """Represents a sequence of frame transformations.

    A transformation path is an ordered sequence of frames that defines
    how to transform from a source frame to a target frame through
    intermediate frames.

    Parameters
    ----------
    frames : list
        List of frame instances or types defining the transformation path.
    transforms : list
        List of transformation functions for each step.

    Examples
    --------
    >>> # Path from ENU -> ECEF -> ECI
    >>> path = TransformationPath(
    ...     frames=[enu_frame, ecef_frame, eci_frame],
    ...     transforms=[enu_to_ecef, ecef_to_eci]
    ... )

    """

    def __init__(self, frames: list, transforms: list = None):
        self.frames = frames
        self.transforms = transforms or []

    def __len__(self):
        return len(self.frames) - 1  # Number of transformation steps

    def __repr__(self):
        frame_names = [getattr(f, 'name', str(f)) for f in self.frames]
        return f"TransformationPath({' -> '.join(frame_names)})"

    def apply(self, position: np.ndarray, velocity: np.ndarray = None,
              timestamp: datetime = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all transformations in the path sequentially.

        Parameters
        ----------
        position : np.ndarray
            Initial position in source frame.
        velocity : np.ndarray, optional
            Initial velocity in source frame.
        timestamp : datetime, optional
            Time for time-dependent transformations.

        Returns
        -------
        position : np.ndarray
            Final position in target frame.
        velocity : np.ndarray or None
            Final velocity in target frame.

        """
        pos = position
        vel = velocity

        for i, transform in enumerate(self.transforms):
            pos, vel = transform(pos, vel, timestamp)

        return pos, vel


class FrameTransformationRegistry:
    """Registry for frame transformations with automatic path finding.

    This registry manages transformations between different reference frames
    and can automatically find transformation paths through intermediate
    frames when a direct transformation is not available.

    The registry uses a graph-based approach where frames are nodes and
    transformations are edges. It finds the shortest path between any
    two frames using breadth-first search.

    Examples
    --------
    >>> registry = FrameTransformationRegistry()
    >>> # Register direct transformations
    >>> registry.register(ENUFrame, 'ECEF', enu_to_ecef_transform)
    >>> registry.register('ECEF', ECIFrame, ecef_to_eci_transform)
    >>> # Find path from ENU to ECI (goes through ECEF)
    >>> path = registry.find_path(enu_frame, eci_frame)
    >>> pos_eci, vel_eci = path.apply(pos_enu, vel_enu, timestamp)

    """

    def __init__(self):
        # Graph of transformations: {source_key: {target_key: transform_func}}
        self._transforms: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        # Store frame type mappings
        self._frame_types: Dict[str, Type] = {}

    def _get_key(self, frame) -> str:
        """Get a unique key for a frame type or instance."""
        if isinstance(frame, str):
            return frame
        elif isinstance(frame, type):
            return frame.__name__
        else:
            return type(frame).__name__

    def register(self, source, target, transform: Callable,
                 bidirectional: bool = False,
                 inverse_transform: Callable = None):
        """Register a transformation between two frame types.

        Parameters
        ----------
        source : str or type or ReferenceFrame
            Source frame (type, instance, or string name).
        target : str or type or ReferenceFrame
            Target frame (type, instance, or string name).
        transform : callable
            Transformation function with signature:
            (position, velocity, timestamp) -> (position, velocity)
        bidirectional : bool, optional
            If True, also register the inverse transformation.
        inverse_transform : callable, optional
            Inverse transformation function. Required if bidirectional=True
            and transform is not invertible.

        """
        source_key = self._get_key(source)
        target_key = self._get_key(target)

        self._transforms[source_key][target_key] = transform

        # Store frame types for later instantiation
        if isinstance(source, type):
            self._frame_types[source_key] = source
        if isinstance(target, type):
            self._frame_types[target_key] = target

        if bidirectional:
            if inverse_transform is None:
                raise ValueError(
                    "inverse_transform required for bidirectional registration")
            self._transforms[target_key][source_key] = inverse_transform

    def get_direct_transform(self, source, target) -> Optional[Callable]:
        """Get a direct transformation if one exists.

        Parameters
        ----------
        source : str or type or ReferenceFrame
            Source frame.
        target : str or type or ReferenceFrame
            Target frame.

        Returns
        -------
        callable or None
            Transformation function, or None if not registered.

        """
        source_key = self._get_key(source)
        target_key = self._get_key(target)

        return self._transforms.get(source_key, {}).get(target_key)

    def find_path(self, source, target) -> Optional[TransformationPath]:
        """Find a transformation path between two frames.

        Uses breadth-first search to find the shortest path through
        intermediate frames.

        Parameters
        ----------
        source : str or type or ReferenceFrame
            Source frame.
        target : str or type or ReferenceFrame
            Target frame.

        Returns
        -------
        TransformationPath or None
            Path object containing the sequence of transformations,
            or None if no path exists.

        """
        source_key = self._get_key(source)
        target_key = self._get_key(target)

        if source_key == target_key:
            # Identity transformation
            return TransformationPath(
                frames=[source],
                transforms=[lambda p, v, t: (p, v)]
            )

        # BFS to find shortest path
        visited = {source_key}
        queue = [(source_key, [source_key])]

        while queue:
            current, path = queue.pop(0)

            for neighbor in self._transforms.get(current, {}):
                if neighbor == target_key:
                    # Found path
                    full_path = path + [neighbor]
                    transforms = []
                    for i in range(len(full_path) - 1):
                        transforms.append(
                            self._transforms[full_path[i]][full_path[i + 1]])
                    return TransformationPath(
                        frames=full_path,
                        transforms=transforms
                    )

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    def transform(self, source, target, position: np.ndarray,
                  velocity: np.ndarray = None,
                  timestamp: datetime = None) -> Tuple[np.ndarray, np.ndarray]:
        """Transform position/velocity between frames.

        Parameters
        ----------
        source : str or type or ReferenceFrame
            Source frame.
        target : str or type or ReferenceFrame
            Target frame.
        position : np.ndarray
            Position in source frame.
        velocity : np.ndarray, optional
            Velocity in source frame.
        timestamp : datetime, optional
            Time for time-dependent transformations.

        Returns
        -------
        position : np.ndarray
            Position in target frame.
        velocity : np.ndarray or None
            Velocity in target frame.

        Raises
        ------
        ValueError
            If no transformation path exists between frames.

        """
        path = self.find_path(source, target)
        if path is None:
            raise ValueError(
                f"No transformation path from {self._get_key(source)} "
                f"to {self._get_key(target)}")
        return path.apply(position, velocity, timestamp)

    def list_frames(self) -> List[str]:
        """List all registered frame types."""
        frames = set(self._transforms.keys())
        for targets in self._transforms.values():
            frames.update(targets.keys())
        return sorted(frames)

    def list_transforms(self) -> List[Tuple[str, str]]:
        """List all registered direct transformations."""
        transforms = []
        for source, targets in self._transforms.items():
            for target in targets:
                transforms.append((source, target))
        return transforms


# Global transformation registry instance
_global_registry = FrameTransformationRegistry()


def get_frame_registry() -> FrameTransformationRegistry:
    """Get the global frame transformation registry.

    Returns
    -------
    FrameTransformationRegistry
        The global registry instance.

    """
    return _global_registry


def register_transform(source, target, transform: Callable,
                       bidirectional: bool = False,
                       inverse_transform: Callable = None):
    """Register a transformation in the global registry.

    This is a convenience function for registering transformations
    without explicitly accessing the registry.

    Parameters
    ----------
    source : str or type or ReferenceFrame
        Source frame.
    target : str or type or ReferenceFrame
        Target frame.
    transform : callable
        Transformation function.
    bidirectional : bool, optional
        If True, also register inverse.
    inverse_transform : callable, optional
        Inverse transformation function.

    """
    _global_registry.register(source, target, transform,
                              bidirectional, inverse_transform)


def compose_transformations(*transforms: Callable) -> Callable:
    """Compose multiple transformation functions into one.

    Parameters
    ----------
    *transforms : callable
        Transformation functions to compose. Each should have signature:
        (position, velocity, timestamp) -> (position, velocity)

    Returns
    -------
    callable
        Composed transformation function.

    Examples
    --------
    >>> # Compose ENU->ECEF and ECEF->ECI into ENU->ECI
    >>> enu_to_eci = compose_transformations(enu_to_ecef, ecef_to_eci)
    >>> pos_eci, vel_eci = enu_to_eci(pos_enu, vel_enu, timestamp)

    """
    def composed(position, velocity=None, timestamp=None):
        pos, vel = position, velocity
        for transform in transforms:
            pos, vel = transform(pos, vel, timestamp)
        return pos, vel

    return composed


# =============================================================================
# Time-Varying Transformation Support
# =============================================================================


class TimeVaryingTransform:
    """Base class for time-varying coordinate transformations.

    Time-varying transformations are transformations that change as a function
    of time. This includes Earth rotation (ECI↔ECEF), precession, nutation,
    and other epoch-dependent effects.

    Subclasses should implement `get_rotation_matrix` and optionally
    `get_rotation_rate_matrix` for velocity transformations.

    Parameters
    ----------
    reference_epoch : datetime, optional
        Reference epoch for the transformation. If None, uses J2000.0.

    Examples
    --------
    >>> from datetime import datetime
    >>> transform = TimeVaryingTransform()
    >>> position = np.array([7000000.0, 0.0, 0.0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> # Subclass would implement actual transformation

    """

    def __init__(self, reference_epoch: datetime = None):
        if reference_epoch is None:
            self.reference_epoch = datetime(2000, 1, 1, 12, 0, 0)
        else:
            self.reference_epoch = reference_epoch

    def get_rotation_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get the rotation matrix for a given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the rotation matrix.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        """
        raise NotImplementedError("Subclass must implement get_rotation_matrix")

    def get_rotation_rate_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get the rotation rate matrix for velocity transformation.

        For a rotating frame with angular velocity ω, the rotation rate matrix
        is the skew-symmetric matrix [ω×], where × denotes cross-product.

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the rotation rate matrix.

        Returns
        -------
        np.ndarray
            3x3 rotation rate (skew-symmetric) matrix.

        """
        # Default: zero rotation rate (identity velocity transformation)
        return np.zeros((3, 3))

    def __call__(self, position: np.ndarray, velocity: np.ndarray = None,
                 timestamp: datetime = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the time-varying transformation.

        Parameters
        ----------
        position : np.ndarray
            Position vector to transform.
        velocity : np.ndarray, optional
            Velocity vector to transform.
        timestamp : datetime, optional
            Time at which to compute transformation. If None, uses reference epoch.

        Returns
        -------
        position : np.ndarray
            Transformed position.
        velocity : np.ndarray or None
            Transformed velocity, or None if not provided.

        """
        if timestamp is None:
            timestamp = self.reference_epoch

        R = self.get_rotation_matrix(timestamp)
        pos_transformed = R @ position

        if velocity is None:
            return pos_transformed, None

        # For rotating frames: v_target = R @ v_source + R_dot @ r_source
        # where R_dot = R @ [ω×] for constant angular velocity
        omega_matrix = self.get_rotation_rate_matrix(timestamp)
        vel_transformed = R @ velocity + omega_matrix @ position

        return pos_transformed, vel_transformed


class RotationRateTransform(TimeVaryingTransform):
    """Time-varying transformation with constant rotation rate.

    This class implements transformations for frames that rotate with
    a constant angular velocity, such as Earth rotation (ECI↔ECEF).

    Parameters
    ----------
    rotation_axis : np.ndarray
        Unit vector defining the rotation axis.
    rotation_rate : float
        Angular rotation rate in radians per second.
    reference_epoch : datetime, optional
        Reference epoch when rotation angle is zero.

    Examples
    --------
    >>> # Earth rotation (ECI to ECEF)
    >>> omega_earth = 7.2921150e-5  # rad/s
    >>> earth_rotation = RotationRateTransform(
    ...     rotation_axis=np.array([0.0, 0.0, 1.0]),
    ...     rotation_rate=omega_earth
    ... )
    >>> pos_ecef, vel_ecef = earth_rotation(pos_eci, vel_eci, timestamp)

    """

    def __init__(self, rotation_axis: np.ndarray, rotation_rate: float,
                 reference_epoch: datetime = None, initial_angle: float = 0.0):
        super().__init__(reference_epoch)
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        self.rotation_rate = rotation_rate
        self.initial_angle = initial_angle

    def get_rotation_angle(self, timestamp: datetime) -> float:
        """Compute rotation angle at given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the angle.

        Returns
        -------
        float
            Rotation angle in radians.

        """
        dt = (timestamp - self.reference_epoch).total_seconds()
        return self.initial_angle + self.rotation_rate * dt

    def get_rotation_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get rotation matrix using Rodrigues' formula.

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the rotation.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        """
        angle = self.get_rotation_angle(timestamp)
        k = self.rotation_axis

        # Skew-symmetric matrix [k×]
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])

        # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return R

    def get_rotation_rate_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get rotation rate matrix (ω × position contribution).

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the rotation rate.

        Returns
        -------
        np.ndarray
            3x3 rotation rate matrix.

        """
        # Angular velocity vector
        omega = self.rotation_rate * self.rotation_axis

        # Skew-symmetric matrix [ω×]
        omega_matrix = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])

        # R_dot @ position = R @ [ω×] @ position for constant rotation rate
        R = self.get_rotation_matrix(timestamp)

        return R @ omega_matrix


class InterpolatedTransform(TimeVaryingTransform):
    """Time-varying transformation using interpolation between epochs.

    This class supports transformations where rotation matrices are known
    at specific epochs and need to be interpolated for intermediate times.
    Uses spherical linear interpolation (SLERP) for smooth rotation.

    Parameters
    ----------
    epochs : list of datetime
        List of epochs at which rotation matrices are known.
    rotation_matrices : list of np.ndarray
        List of 3x3 rotation matrices at each epoch.
    extrapolate : bool, optional
        Whether to allow extrapolation beyond known epochs. Default False.

    Examples
    --------
    >>> epochs = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    >>> matrices = [np.eye(3), rotation_z(0.01)]  # Small rotation
    >>> transform = InterpolatedTransform(epochs, matrices)
    >>> # Interpolate at midpoint
    >>> pos_out, _ = transform(position, timestamp=datetime(2024, 1, 1, 12))

    """

    def __init__(self, epochs: List[datetime], rotation_matrices: List[np.ndarray],
                 extrapolate: bool = False):
        if len(epochs) != len(rotation_matrices):
            raise ValueError("epochs and rotation_matrices must have same length")
        if len(epochs) < 2:
            raise ValueError("At least 2 epochs required for interpolation")

        # Sort by epoch
        sorted_pairs = sorted(zip(epochs, rotation_matrices), key=lambda x: x[0])
        self.epochs = [e for e, _ in sorted_pairs]
        self.rotation_matrices = [r for _, r in sorted_pairs]
        self.extrapolate = extrapolate

        super().__init__(self.epochs[0])

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        # Shepperd's method for numerical stability
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)

        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def _slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        # Ensure shortest path
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot

        # Linear interpolation for very close quaternions
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        # SLERP
        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q0 + s1 * q1

    def get_rotation_matrix(self, timestamp: datetime) -> np.ndarray:
        """Interpolate rotation matrix at given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Time at which to compute the rotation.

        Returns
        -------
        np.ndarray
            Interpolated 3x3 rotation matrix.

        """
        # Find bracketing epochs
        if timestamp <= self.epochs[0]:
            if not self.extrapolate:
                return self.rotation_matrices[0].copy()
            # Extrapolate before first epoch
            idx = 0
        elif timestamp >= self.epochs[-1]:
            if not self.extrapolate:
                return self.rotation_matrices[-1].copy()
            # Extrapolate after last epoch
            idx = len(self.epochs) - 2
        else:
            # Find bracketing index
            for i in range(len(self.epochs) - 1):
                if self.epochs[i] <= timestamp <= self.epochs[i + 1]:
                    idx = i
                    break

        # Interpolation parameter
        t0 = self.epochs[idx]
        t1 = self.epochs[idx + 1]
        dt_total = (t1 - t0).total_seconds()
        dt = (timestamp - t0).total_seconds()
        t_param = dt / dt_total if dt_total > 0 else 0.0

        # Convert to quaternions
        q0 = self._rotation_to_quaternion(self.rotation_matrices[idx])
        q1 = self._rotation_to_quaternion(self.rotation_matrices[idx + 1])

        # SLERP interpolation
        q_interp = self._slerp(q0, q1, t_param)

        return self._quaternion_to_rotation(q_interp)


class EpochCachedTransform(TimeVaryingTransform):
    """Time-varying transformation with epoch caching.

    This class caches rotation matrices at specific epochs for efficiency
    when the same transformation is applied at the same time repeatedly.

    Parameters
    ----------
    base_transform : TimeVaryingTransform
        The underlying time-varying transformation.
    cache_size : int, optional
        Maximum number of epochs to cache. Default 100.

    Examples
    --------
    >>> base_transform = RotationRateTransform(...)
    >>> cached = EpochCachedTransform(base_transform, cache_size=50)
    >>> # First call computes and caches
    >>> pos1, vel1 = cached(position1, velocity1, timestamp)
    >>> # Second call uses cache
    >>> pos2, vel2 = cached(position2, velocity2, timestamp)

    """

    def __init__(self, base_transform: TimeVaryingTransform, cache_size: int = 100):
        self.base_transform = base_transform
        self.cache_size = cache_size
        self._rotation_cache: Dict[datetime, np.ndarray] = {}
        self._rate_cache: Dict[datetime, np.ndarray] = {}

        super().__init__(base_transform.reference_epoch)

    def get_rotation_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get rotation matrix, using cache if available."""
        if timestamp not in self._rotation_cache:
            if len(self._rotation_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest = next(iter(self._rotation_cache))
                del self._rotation_cache[oldest]

            self._rotation_cache[timestamp] = \
                self.base_transform.get_rotation_matrix(timestamp)

        return self._rotation_cache[timestamp]

    def get_rotation_rate_matrix(self, timestamp: datetime) -> np.ndarray:
        """Get rotation rate matrix, using cache if available."""
        if timestamp not in self._rate_cache:
            if len(self._rate_cache) >= self.cache_size:
                oldest = next(iter(self._rate_cache))
                del self._rate_cache[oldest]

            self._rate_cache[timestamp] = \
                self.base_transform.get_rotation_rate_matrix(timestamp)

        return self._rate_cache[timestamp]

    def clear_cache(self):
        """Clear all cached matrices."""
        self._rotation_cache.clear()
        self._rate_cache.clear()


class ReferenceEllipsoid(Base):
    r"""Reference Ellipsoid for geodetic coordinate systems.

    A reference ellipsoid is a mathematically defined surface that approximates the geoid,
    the true figure of the Earth. It is defined by its semi-major axis (equatorial radius)
    and flattening.

    The semi-minor axis :math:`b` and eccentricity :math:`e` are derived quantities:

    .. math::

        b &= a(1 - f)

        e^2 &= 2f - f^2 = 1 - \frac{b^2}{a^2}

    where :math:`a` is the semi-major axis and :math:`f` is the flattening.

    Examples
    --------
    >>> # WGS84 ellipsoid
    >>> wgs84 = ReferenceEllipsoid(
    ...     name="WGS84",
    ...     semi_major_axis=6378137.0,
    ...     flattening=1/298.257223563
    ... )
    >>> print(f"Semi-minor axis: {wgs84.semi_minor_axis:.3f} m")
    Semi-minor axis: 6356752.314 m
    >>> print(f"Eccentricity: {wgs84.eccentricity:.10f}")
    Eccentricity: 0.0818191908

    """

    name: str = Property(doc="Name of the ellipsoid (e.g., 'WGS84', 'GRS80')")
    semi_major_axis: float = Property(doc="Semi-major axis (equatorial radius) in meters")
    flattening: float = Property(doc="Flattening factor (dimensionless)")

    @property
    def semi_minor_axis(self) -> float:
        r"""Semi-minor axis (polar radius) in meters.

        Calculated as:

        .. math::

            b = a(1 - f)

        where :math:`a` is the semi-major axis and :math:`f` is the flattening.
        """
        return self.semi_major_axis * (1.0 - self.flattening)

    @property
    def eccentricity(self) -> float:
        r"""First eccentricity of the ellipsoid (dimensionless).

        Calculated as:

        .. math::

            e^2 = 2f - f^2

        where :math:`f` is the flattening.
        """
        return np.sqrt(2.0 * self.flattening - self.flattening ** 2)

    @property
    def eccentricity_squared(self) -> float:
        r"""Square of the first eccentricity (dimensionless).

        Calculated as:

        .. math::

            e^2 = 2f - f^2

        where :math:`f` is the flattening.
        """
        return 2.0 * self.flattening - self.flattening ** 2

    @property
    def second_eccentricity_squared(self) -> float:
        r"""Square of the second eccentricity (dimensionless).

        Calculated as:

        .. math::

            e'^2 = \frac{a^2 - b^2}{b^2} = \frac{e^2}{1 - e^2}

        where :math:`a` is the semi-major axis and :math:`b` is the semi-minor axis.
        """
        e2 = self.eccentricity_squared
        return e2 / (1.0 - e2)

    @property
    def linear_eccentricity(self) -> float:
        r"""Linear eccentricity in meters.

        The distance from the center to a focus of the ellipse:

        .. math::

            E = \sqrt{a^2 - b^2} = ae

        where :math:`a` is the semi-major axis, :math:`b` is the semi-minor axis,
        and :math:`e` is the eccentricity.
        """
        return self.semi_major_axis * self.eccentricity


# WGS84 Realizations (G-series)
# Reference: https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84

WGS84_G730: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G730)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G730) - World Geodetic System 1984, original realization (1987).

Reference epoch: 1994.0
"""

WGS84_G873: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G873)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G873) - Refined realization (1996).

Reference epoch: 1997.0
"""

WGS84_G1150: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G1150)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G1150) - Refined realization (2002).

Reference epoch: 2001.0
"""

WGS84_G1674: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G1674)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G1674) - Refined realization (2012).

Reference epoch: 2005.0
"""

WGS84_G1762: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G1762)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G1762) - Refined realization (2013).

Reference epoch: 2005.0
"""

WGS84_G2139: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS84 (G2139)",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257223563
)
"""WGS 84 (G2139) - Latest realization (2021).

Reference epoch: 2016.0
"""

# Default WGS84 is the latest realization
WGS84: ClassVar[ReferenceEllipsoid] = WGS84_G2139
"""WGS 84 - World Geodetic System 1984 (latest realization: G2139).

This is the most commonly used geodetic reference system for GPS and global mapping.
The ellipsoid parameters are:

- Semi-major axis: 6378137.0 m
- Flattening: 1/298.257223563

Note: All WGS84 realizations use the same ellipsoid parameters; they differ in
their reference frames and realization epochs.
"""

# GRS80 - Geodetic Reference System 1980
GRS80: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="GRS80",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257222101
)
"""GRS 80 - Geodetic Reference System 1980.

Used as the basis for NAD83 and many other national geodetic systems. Parameters:

- Semi-major axis: 6378137.0 m
- Flattening: 1/298.257222101

Note: GRS80 and WGS84 have identical semi-major axes but differ very slightly in
flattening (difference of ~0.1 mm in semi-minor axis).
"""

# WGS72 - World Geodetic System 1972
WGS72: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="WGS72",
    semi_major_axis=6378135.0,
    flattening=1.0 / 298.26
)
"""WGS 72 - World Geodetic System 1972 (predecessor to WGS84).

Parameters:

- Semi-major axis: 6378135.0 m
- Flattening: 1/298.26
"""

# PZ90 - Parametry Zemli 1990 (Russian Geodetic System)
PZ90: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="PZ90",
    semi_major_axis=6378136.0,
    flattening=1.0 / 298.257839303
)
"""PZ-90 - Parametry Zemli 1990 (Parameters of the Earth 1990).

Russian geodetic system used by GLONASS. Parameters:

- Semi-major axis: 6378136.0 m
- Flattening: 1/298.257839303
"""

# CGCS2000 - China Geodetic Coordinate System 2000
CGCS2000: ClassVar[ReferenceEllipsoid] = ReferenceEllipsoid(
    name="CGCS2000",
    semi_major_axis=6378137.0,
    flattening=1.0 / 298.257222101
)
"""CGCS2000 - China Geodetic Coordinate System 2000.

Official geodetic system of China. Parameters are identical to GRS80:

- Semi-major axis: 6378137.0 m
- Flattening: 1/298.257222101
"""


class ReferenceFrame(Base):
    """Base class for celestial and inertial reference frames.

    A reference frame defines a coordinate system for describing positions and velocities
    in space. This base class provides the interface for transforming between different
    reference frames.

    Examples
    --------
    >>> # Transform position and velocity from GCRS to J2000
    >>> from datetime import datetime
    >>> import numpy as np
    >>> gcrs = GCRS()
    >>> j2000 = J2000()
    >>> position = np.array([7000000.0, 0.0, 0.0])
    >>> velocity = np.array([0.0, 7500.0, 0.0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> pos_j2000, vel_j2000 = gcrs.transform_to(j2000, position, velocity, timestamp)

    """

    name: str = Property(doc="Name of the reference frame")

    @abstractmethod
    def transform_to(self, other_frame: 'ReferenceFrame', position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple[np.ndarray, np.ndarray]:
        """Transform position and velocity to another reference frame.

        Parameters
        ----------
        other_frame : ReferenceFrame
            Target reference frame
        position : numpy.ndarray
            Position vector in this frame as [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity vector in this frame as [vx, vy, vz] in m/s.
            If None, only position is transformed and velocity is returned as None.
        timestamp : datetime.datetime, optional
            Time at which the transformation is computed.
            Required for time-dependent transformations.

        Returns
        -------
        position : numpy.ndarray
            Position vector in the target frame as [x, y, z] in meters
        velocity : numpy.ndarray or None
            Velocity vector in the target frame as [vx, vy, vz] in m/s,
            or None if velocity was not provided

        Raises
        ------
        NotImplementedError
            If transformation between these frames is not supported

        """
        raise NotImplementedError(
            f"Transformation from {self.name} to {other_frame.name} not implemented"
        )


class GCRS(ReferenceFrame):
    """Geocentric Celestial Reference System (GCRS).

    The GCRS is a celestial reference system with its origin at the geocenter
    (Earth's center of mass). The axes are kinematically non-rotating with respect
    to distant quasars. GCRS is the IAU 2000 replacement for J2000 for high-precision
    applications.

    The GCRS accounts for:

    - Precession and nutation of Earth's axis
    - Gravitational deflection of light
    - Aberration

    For many applications, GCRS can be approximated as equivalent to J2000, with
    differences typically less than a few meters for near-Earth objects.

    References
    ----------
    .. [1] IAU 2000 Resolution B1.3, "Definition of the Celestial Reference System"
    .. [2] IERS Conventions (2010), IERS Technical Note No. 36

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> gcrs = GCRS()
    >>> position = np.array([7000000.0, 0.0, 0.0])
    >>> velocity = np.array([0.0, 7500.0, 0.0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)

    """

    name: str = Property(default="GCRS", doc="Name of the reference frame")

    def transform_to(self, other_frame: ReferenceFrame, position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple[np.ndarray, np.ndarray]:
        """Transform from GCRS to another reference frame.

        Parameters
        ----------
        other_frame : ReferenceFrame
            Target reference frame
        position : numpy.ndarray
            Position vector in GCRS as [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity vector in GCRS as [vx, vy, vz] in m/s
        timestamp : datetime.datetime, optional
            Time at which the transformation is computed

        Returns
        -------
        position : numpy.ndarray
            Position vector in the target frame
        velocity : numpy.ndarray or None
            Velocity vector in the target frame, or None if not provided

        """
        if isinstance(other_frame, GCRS):
            # No transformation needed
            return position.copy(), velocity.copy() if velocity is not None else None

        if isinstance(other_frame, J2000):
            # Import here to avoid circular dependency
            from ..functions.coordinates import gcrs_to_j2000
            return gcrs_to_j2000(position, velocity, timestamp)

        if isinstance(other_frame, ICRS):
            # GCRS and ICRS differ by frame bias only
            # For most applications, the difference is negligible (< 0.1 arcsec)
            # A more precise implementation would apply the frame bias rotation
            return position.copy(), velocity.copy() if velocity is not None else None

        raise NotImplementedError(
            f"Transformation from GCRS to {other_frame.name} not implemented"
        )


class J2000(ReferenceFrame):
    """J2000.0 Reference Frame (Mean Equator and Equinox at J2000.0).

    The J2000 reference frame is defined by the mean equator and equinox at the
    J2000.0 epoch (2000-01-01 12:00:00 TT). This is the classical inertial reference
    frame used in celestial mechanics and astrodynamics.

    The J2000 frame:

    - Origin: Geocenter (Earth's center of mass)
    - Fundamental plane: Earth's mean equator at J2000.0
    - Reference direction: Mean vernal equinox at J2000.0
    - Epoch: 2000-01-01 12:00:00 TT (Terrestrial Time)

    Note that J2000 is a mean frame (does not include nutation) and is being
    superseded by GCRS for high-precision applications. However, J2000 remains
    widely used in satellite orbit determination and spacecraft navigation.

    References
    ----------
    .. [1] Seidelmann, P. K., 1992, "Explanatory Supplement to the Astronomical
           Almanac," University Science Books.
    .. [2] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press.

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> j2000 = J2000()
    >>> position = np.array([7000000.0, 0.0, 0.0])
    >>> velocity = np.array([0.0, 7500.0, 0.0])

    """

    name: str = Property(default="J2000", doc="Name of the reference frame")

    def transform_to(self, other_frame: ReferenceFrame, position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple[np.ndarray, np.ndarray]:
        """Transform from J2000 to another reference frame.

        Parameters
        ----------
        other_frame : ReferenceFrame
            Target reference frame
        position : numpy.ndarray
            Position vector in J2000 as [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity vector in J2000 as [vx, vy, vz] in m/s
        timestamp : datetime.datetime, optional
            Time at which the transformation is computed

        Returns
        -------
        position : numpy.ndarray
            Position vector in the target frame
        velocity : numpy.ndarray or None
            Velocity vector in the target frame, or None if not provided

        """
        if isinstance(other_frame, J2000):
            # No transformation needed
            return position.copy(), velocity.copy() if velocity is not None else None

        if isinstance(other_frame, GCRS):
            # Import here to avoid circular dependency
            from ..functions.coordinates import j2000_to_gcrs
            return j2000_to_gcrs(position, velocity, timestamp)

        if isinstance(other_frame, ICRS):
            # J2000 to ICRS requires frame bias correction
            # For most applications, the difference is small
            from ..functions.coordinates import compute_frame_bias_matrix
            bias_matrix = compute_frame_bias_matrix()
            pos_icrs = bias_matrix @ position
            vel_icrs = bias_matrix @ velocity if velocity is not None else None
            return pos_icrs, vel_icrs

        raise NotImplementedError(
            f"Transformation from J2000 to {other_frame.name} not implemented"
        )


class ICRS(ReferenceFrame):
    """International Celestial Reference System (ICRS).

    The ICRS is the fundamental celestial reference system adopted by the IAU in 1997.
    It is realized by the positions of extragalactic radio sources (quasars) that
    define a kinematically non-rotating reference frame.

    Key properties:

    - Origin: Barycenter of the solar system
    - Axes: Aligned with extragalactic radio sources (quasi-inertial)
    - Epoch: J2000.0 (but the frame itself is not tied to any epoch)
    - Precession/Nutation: Not applicable (frame is fixed in space)

    The ICRS differs from J2000 by a small frame bias (less than 0.1 arcseconds)
    and is barycentric rather than geocentric. For near-Earth applications, ICRS
    is often treated as equivalent to GCRS.

    References
    ----------
    .. [1] IAU 1997 Resolution B2, "The International Celestial Reference System (ICRS)"
    .. [2] Fey, A. L., et al., 2015, "The Second Realization of the International
           Celestial Reference Frame by Very Long Baseline Interferometry,"
           Astronomical Journal, Vol. 150, No. 2.

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> icrs = ICRS()
    >>> position = np.array([7000000.0, 0.0, 0.0])
    >>> velocity = np.array([0.0, 7500.0, 0.0])

    """

    name: str = Property(default="ICRS", doc="Name of the reference frame")

    def transform_to(self, other_frame: ReferenceFrame, position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple[np.ndarray, np.ndarray]:
        """Transform from ICRS to another reference frame.

        Parameters
        ----------
        other_frame : ReferenceFrame
            Target reference frame
        position : numpy.ndarray
            Position vector in ICRS as [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity vector in ICRS as [vx, vy, vz] in m/s
        timestamp : datetime.datetime, optional
            Time at which the transformation is computed

        Returns
        -------
        position : numpy.ndarray
            Position vector in the target frame
        velocity : numpy.ndarray or None
            Velocity vector in the target frame, or None if not provided

        """
        if isinstance(other_frame, ICRS):
            # No transformation needed
            return position.copy(), velocity.copy() if velocity is not None else None

        if isinstance(other_frame, GCRS):
            # ICRS to GCRS: apply inverse frame bias
            # For most near-Earth applications, difference is negligible
            return position.copy(), velocity.copy() if velocity is not None else None

        if isinstance(other_frame, J2000):
            # ICRS to J2000 requires inverse frame bias correction
            from ..functions.coordinates import compute_frame_bias_matrix
            bias_matrix = compute_frame_bias_matrix()
            # Inverse is transpose for rotation matrix
            pos_j2000 = bias_matrix.T @ position
            vel_j2000 = bias_matrix.T @ velocity if velocity is not None else None
            return pos_j2000, vel_j2000

        raise NotImplementedError(
            f"Transformation from ICRS to {other_frame.name} not implemented"
        )


class KinematicState(Base):
    r"""Full kinematic state with position, velocity, and acceleration.

    A kinematic state represents the complete motion state of an object,
    including its position, velocity, and optionally acceleration. This class
    provides a unified interface for working with kinematic states across
    different coordinate systems.

    The state can be represented in different forms:

    - **Cartesian**: [x, y, z, vx, vy, vz, ax, ay, az]
    - **Spherical**: [r, θ, φ, ṙ, θ̇, φ̇, r̈, θ̈, φ̈]
    - **Geodetic**: [lat, lon, alt, vlat, vlon, valt, alat, alon, aalt]

    Parameters
    ----------
    position : numpy.ndarray
        Position vector [x, y, z] in meters
    velocity : numpy.ndarray, optional
        Velocity vector [vx, vy, vz] in m/s. Default is zeros.
    acceleration : numpy.ndarray, optional
        Acceleration vector [ax, ay, az] in m/s². Default is None (not tracked).
    timestamp : datetime.datetime, optional
        Time at which the state is defined
    frame : ReferenceFrame, optional
        Reference frame in which the state is defined

    Attributes
    ----------
    state_vector : numpy.ndarray
        Combined state vector. Shape depends on whether acceleration is tracked:
        - With acceleration: [x, y, z, vx, vy, vz, ax, ay, az] (9 elements)
        - Without acceleration: [x, y, z, vx, vy, vz] (6 elements)
    ndim : int
        Number of spatial dimensions (always 3)
    has_acceleration : bool
        Whether acceleration is tracked

    Examples
    --------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> # Create a kinematic state for a satellite
    >>> position = np.array([7000000.0, 0.0, 0.0])  # 7000 km along x-axis
    >>> velocity = np.array([0.0, 7500.0, 0.0])     # 7.5 km/s along y-axis
    >>> acceleration = np.array([0.0, 0.0, -9.8])   # Gravity along z-axis
    >>> state = KinematicState(
    ...     position=position,
    ...     velocity=velocity,
    ...     acceleration=acceleration,
    ...     timestamp=datetime(2024, 1, 1, 12, 0, 0)
    ... )
    >>> print(f"State dimension: {state.state_vector.shape[0]}")
    State dimension: 9

    Notes
    -----
    When transforming kinematic states between reference frames, velocity and
    acceleration require additional corrections beyond simple rotation:

    - **Velocity transformation** must account for the rotation rate of the
      target frame (Coriolis effect)
    - **Acceleration transformation** must account for both Coriolis acceleration
      and centripetal acceleration

    For ECEF to ECI transformations:

    .. math::

        \mathbf{v}_{ECI} = \mathbf{R} \mathbf{v}_{ECEF} + \boldsymbol{\omega} \times \mathbf{r}_{ECI}

    .. math::

        \mathbf{a}_{ECI} = \mathbf{R} \mathbf{a}_{ECEF} + 2\boldsymbol{\omega} \times \mathbf{v}_{ECI}
                          + \boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_{ECI})

    where :math:`\mathbf{R}` is the rotation matrix and :math:`\boldsymbol{\omega}` is
    Earth's rotation vector.

    """

    position: np.ndarray = Property(
        doc="Position vector [x, y, z] in meters"
    )
    velocity: np.ndarray = Property(
        default=None,
        doc="Velocity vector [vx, vy, vz] in m/s. Default is zeros."
    )
    acceleration: np.ndarray = Property(
        default=None,
        doc="Acceleration vector [ax, ay, az] in m/s². Default is None (not tracked)."
    )
    timestamp: Optional[datetime] = Property(
        default=None,
        doc="Time at which the state is defined"
    )
    frame: Optional[ReferenceFrame] = Property(
        default=None,
        doc="Reference frame in which the state is defined"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure position is numpy array
        if not isinstance(self.position, np.ndarray):
            object.__setattr__(self, 'position', np.asarray(self.position))
        # Ensure velocity is numpy array or zeros
        if self.velocity is None:
            object.__setattr__(self, 'velocity', np.zeros(3))
        elif not isinstance(self.velocity, np.ndarray):
            object.__setattr__(self, 'velocity', np.asarray(self.velocity))
        # Ensure acceleration is numpy array if provided
        if self.acceleration is not None and not isinstance(self.acceleration, np.ndarray):
            object.__setattr__(self, 'acceleration', np.asarray(self.acceleration))

    @property
    def state_vector(self) -> StateVector:
        """Combined state vector as StateVector.

        Returns
        -------
        StateVector
            Combined state vector:
            - With acceleration: [x, y, z, vx, vy, vz, ax, ay, az]
            - Without acceleration: [x, y, z, vx, vy, vz]
        """
        if self.has_acceleration:
            return StateVector(np.concatenate([
                self.position.flatten(),
                self.velocity.flatten(),
                self.acceleration.flatten()
            ]).reshape(-1, 1))
        else:
            return StateVector(np.concatenate([
                self.position.flatten(),
                self.velocity.flatten()
            ]).reshape(-1, 1))

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (always 3)."""
        return 3

    @property
    def has_acceleration(self) -> bool:
        """Whether acceleration is tracked."""
        return self.acceleration is not None

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector in m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def acceleration_magnitude(self) -> Optional[float]:
        """Magnitude of acceleration vector in m/s², or None if not tracked."""
        if self.acceleration is None:
            return None
        return float(np.linalg.norm(self.acceleration))

    @classmethod
    def from_state_vector(cls, state_vector: np.ndarray,
                          timestamp: datetime = None,
                          frame: ReferenceFrame = None,
                          has_acceleration: bool = True) -> 'KinematicState':
        """Create KinematicState from a combined state vector.

        Parameters
        ----------
        state_vector : numpy.ndarray
            Combined state vector. Must be either:
            - 6 elements: [x, y, z, vx, vy, vz]
            - 9 elements: [x, y, z, vx, vy, vz, ax, ay, az]
        timestamp : datetime.datetime, optional
            Time at which the state is defined
        frame : ReferenceFrame, optional
            Reference frame in which the state is defined
        has_acceleration : bool, optional
            If True and state_vector has 9 elements, extract acceleration.
            Default is True.

        Returns
        -------
        KinematicState
            New kinematic state instance

        Raises
        ------
        ValueError
            If state_vector doesn't have 6 or 9 elements
        """
        sv = np.asarray(state_vector).flatten()

        if len(sv) == 6:
            return cls(
                position=sv[0:3],
                velocity=sv[3:6],
                acceleration=None,
                timestamp=timestamp,
                frame=frame
            )
        elif len(sv) == 9 and has_acceleration:
            return cls(
                position=sv[0:3],
                velocity=sv[3:6],
                acceleration=sv[6:9],
                timestamp=timestamp,
                frame=frame
            )
        elif len(sv) == 9:
            return cls(
                position=sv[0:3],
                velocity=sv[3:6],
                acceleration=None,
                timestamp=timestamp,
                frame=frame
            )
        else:
            raise ValueError(
                f"State vector must have 6 or 9 elements, got {len(sv)}"
            )

    def transform_to(self, target_frame: ReferenceFrame) -> 'KinematicState':
        """Transform kinematic state to another reference frame.

        This method properly handles the transformation of position, velocity,
        and acceleration between reference frames, accounting for rotation
        effects where applicable.

        Parameters
        ----------
        target_frame : ReferenceFrame
            Target reference frame

        Returns
        -------
        KinematicState
            New kinematic state in the target frame

        Raises
        ------
        ValueError
            If current frame is not set
        NotImplementedError
            If transformation between frames is not supported
        """
        if self.frame is None:
            raise ValueError("Cannot transform state without a defined frame")

        if type(self.frame) == type(target_frame):
            # Same frame type, no transformation needed
            return KinematicState(
                position=self.position.copy(),
                velocity=self.velocity.copy(),
                acceleration=self.acceleration.copy() if self.acceleration is not None else None,
                timestamp=self.timestamp,
                frame=target_frame
            )

        # Use the frame's transform_to method for position and velocity
        new_pos, new_vel = self.frame.transform_to(
            target_frame,
            self.position,
            self.velocity,
            self.timestamp
        )

        # For acceleration, apply the same rotation (simplified approach)
        # A more complete implementation would include Coriolis and centripetal terms
        if self.acceleration is not None:
            # Get rotation matrix by comparing transformed unit vectors
            # This is a simplified approach; full implementation would compute
            # the proper acceleration transformation
            _, acc_transformed = self.frame.transform_to(
                target_frame,
                np.zeros(3),  # Dummy position
                self.acceleration,  # Treat acceleration like velocity for rotation
                self.timestamp
            )
            new_acc = acc_transformed
        else:
            new_acc = None

        return KinematicState(
            position=new_pos,
            velocity=new_vel,
            acceleration=new_acc,
            timestamp=self.timestamp,
            frame=target_frame
        )

    def propagate(self, dt: float) -> 'KinematicState':
        """Propagate the kinematic state forward in time.

        Uses simple kinematic equations:
        - position += velocity * dt + 0.5 * acceleration * dt²
        - velocity += acceleration * dt

        Parameters
        ----------
        dt : float
            Time step in seconds

        Returns
        -------
        KinematicState
            New kinematic state at time t + dt
        """
        from datetime import timedelta

        if self.has_acceleration:
            new_pos = self.position + self.velocity * dt + 0.5 * self.acceleration * dt**2
            new_vel = self.velocity + self.acceleration * dt
            new_acc = self.acceleration.copy()  # Acceleration assumed constant
        else:
            new_pos = self.position + self.velocity * dt
            new_vel = self.velocity.copy()
            new_acc = None

        new_timestamp = None
        if self.timestamp is not None:
            new_timestamp = self.timestamp + timedelta(seconds=dt)

        return KinematicState(
            position=new_pos,
            velocity=new_vel,
            acceleration=new_acc,
            timestamp=new_timestamp,
            frame=self.frame
        )


class RelativeFrame(ReferenceFrame):
    """Base class for relative motion reference frames.

    Relative frames are defined with respect to a reference object (typically
    a target spacecraft) and are used for proximity operations, rendezvous,
    and formation flying analysis.

    The reference object's state defines the origin and orientation of the
    relative frame. The frame rotates with the reference object's orbital motion.

    Common relative frames include:
    - RIC (Radial/In-track/Cross-track)
    - RSW (Radial/Along-track/Cross-track)
    - LVLH (Local Vertical Local Horizontal)

    Parameters
    ----------
    name : str
        Name of the reference frame
    reference_position : numpy.ndarray
        Position of the reference object [x, y, z] in meters (ECI or ECEF)
    reference_velocity : numpy.ndarray
        Velocity of the reference object [vx, vy, vz] in m/s

    Attributes
    ----------
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix from inertial to relative frame

    Notes
    -----
    The rotation matrix R transforms vectors from inertial to relative frame:

    .. math::

        \\mathbf{r}_{rel} = \\mathbf{R} (\\mathbf{r}_{inertial} - \\mathbf{r}_{ref})

    The inverse transforms from relative to inertial:

    .. math::

        \\mathbf{r}_{inertial} = \\mathbf{R}^T \\mathbf{r}_{rel} + \\mathbf{r}_{ref}

    """

    reference_position: np.ndarray = Property(
        doc="Position of the reference object [x, y, z] in meters (ECI)"
    )
    reference_velocity: np.ndarray = Property(
        doc="Velocity of the reference object [vx, vy, vz] in m/s (ECI)"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure arrays
        if not isinstance(self.reference_position, np.ndarray):
            object.__setattr__(self, 'reference_position',
                               np.asarray(self.reference_position))
        if not isinstance(self.reference_velocity, np.ndarray):
            object.__setattr__(self, 'reference_velocity',
                               np.asarray(self.reference_velocity))

    @property
    @abstractmethod
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from inertial frame to this relative frame."""
        raise NotImplementedError

    def inertial_to_relative(self, position: np.ndarray,
                             velocity: np.ndarray = None) -> tuple:
        """Transform position/velocity from inertial to relative frame.

        Parameters
        ----------
        position : numpy.ndarray
            Position in inertial frame [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity in inertial frame [vx, vy, vz] in m/s

        Returns
        -------
        rel_position : numpy.ndarray
            Position in relative frame [x, y, z] in meters
        rel_velocity : numpy.ndarray or None
            Velocity in relative frame [vx, vy, vz] in m/s, or None
        """
        R = self.rotation_matrix

        # Relative position
        delta_r = position - self.reference_position
        rel_pos = R @ delta_r

        if velocity is None:
            return rel_pos, None

        # Relative velocity (simplified - doesn't include frame rotation rate)
        delta_v = velocity - self.reference_velocity
        rel_vel = R @ delta_v

        return rel_pos, rel_vel

    def relative_to_inertial(self, position: np.ndarray,
                             velocity: np.ndarray = None) -> tuple:
        """Transform position/velocity from relative to inertial frame.

        Parameters
        ----------
        position : numpy.ndarray
            Position in relative frame [x, y, z] in meters
        velocity : numpy.ndarray, optional
            Velocity in relative frame [vx, vy, vz] in m/s

        Returns
        -------
        inertial_position : numpy.ndarray
            Position in inertial frame [x, y, z] in meters
        inertial_velocity : numpy.ndarray or None
            Velocity in inertial frame [vx, vy, vz] in m/s, or None
        """
        R = self.rotation_matrix
        R_inv = R.T  # Rotation matrix is orthogonal

        # Inertial position
        inertial_pos = R_inv @ position + self.reference_position

        if velocity is None:
            return inertial_pos, None

        # Inertial velocity
        inertial_vel = R_inv @ velocity + self.reference_velocity

        return inertial_pos, inertial_vel

    def transform_to(self, other_frame: ReferenceFrame, position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple[np.ndarray, np.ndarray]:
        """Transform position and velocity to another reference frame.

        For relative frames, this first transforms to the inertial frame,
        then to the target frame if needed.
        """
        if type(self) == type(other_frame):
            # Same frame type - direct transformation via inertial
            if isinstance(other_frame, RelativeFrame):
                # Transform to inertial then to other relative frame
                inertial_pos, inertial_vel = self.relative_to_inertial(position, velocity)
                return other_frame.inertial_to_relative(inertial_pos, inertial_vel)
            return position.copy(), velocity.copy() if velocity is not None else None

        if isinstance(other_frame, RelativeFrame):
            # Transform to inertial then to other relative frame
            inertial_pos, inertial_vel = self.relative_to_inertial(position, velocity)
            return other_frame.inertial_to_relative(inertial_pos, inertial_vel)

        # Transform to inertial frame
        return self.relative_to_inertial(position, velocity)


class RICFrame(RelativeFrame):
    """Radial/In-track/Cross-track (RIC) relative frame.

    The RIC frame is commonly used for proximity operations and rendezvous:

    - **R (Radial)**: Points along the position vector from Earth's center
      through the reference object (outward)
    - **I (In-track)**: Approximately along the velocity vector, perpendicular
      to R in the orbital plane
    - **C (Cross-track)**: Completes the right-handed system, normal to the
      orbital plane (along angular momentum vector)

    This frame rotates with the reference object's orbital motion.

    Parameters
    ----------
    reference_position : numpy.ndarray
        Position of the reference object [x, y, z] in meters (ECI)
    reference_velocity : numpy.ndarray
        Velocity of the reference object [vx, vy, vz] in m/s (ECI)

    Notes
    -----
    The RIC unit vectors are computed as:

    .. math::

        \\hat{R} = \\frac{\\mathbf{r}}{|\\mathbf{r}|}

        \\hat{C} = \\frac{\\mathbf{r} \\times \\mathbf{v}}{|\\mathbf{r} \\times \\mathbf{v}|}

        \\hat{I} = \\hat{C} \\times \\hat{R}

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press, Chapter 6.

    Examples
    --------
    >>> import numpy as np
    >>> # Reference object in circular orbit
    >>> ref_pos = np.array([7000000.0, 0.0, 0.0])  # 7000 km along x
    >>> ref_vel = np.array([0.0, 7546.0, 0.0])     # ~7.5 km/s along y
    >>> ric = RICFrame(
    ...     name="RIC",
    ...     reference_position=ref_pos,
    ...     reference_velocity=ref_vel
    ... )
    >>> # Transform chaser position to RIC
    >>> chaser_pos = np.array([7001000.0, 100.0, 50.0])  # 1 km ahead
    >>> rel_pos, _ = ric.inertial_to_relative(chaser_pos)
    >>> print(f"RIC position: R={rel_pos[0]:.1f}m, I={rel_pos[1]:.1f}m, C={rel_pos[2]:.1f}m")

    """

    name: str = Property(default="RIC", doc="Name of the reference frame")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from inertial frame to RIC frame."""
        r = self.reference_position
        v = self.reference_velocity

        # Radial unit vector (outward from Earth center)
        r_hat = r / np.linalg.norm(r)

        # Angular momentum direction (cross-track, normal to orbital plane)
        h = np.cross(r, v)
        c_hat = h / np.linalg.norm(h)

        # In-track unit vector (perpendicular to R in orbital plane)
        i_hat = np.cross(c_hat, r_hat)

        # Rotation matrix: rows are the unit vectors
        return np.array([r_hat, i_hat, c_hat])


class RSWFrame(RelativeFrame):
    """Radial/Along-track/Cross-track (RSW) relative frame.

    The RSW frame is similar to RIC with slightly different conventions:

    - **R (Radial)**: Points along the position vector from Earth's center
      through the reference object (outward)
    - **S (Along-track)**: Along the velocity direction, in the orbital plane
    - **W (Cross-track)**: Completes right-handed system, normal to orbital plane

    This is essentially the same as RIC but with different naming. The S axis
    is aligned with velocity rather than being strictly perpendicular to R.

    Parameters
    ----------
    reference_position : numpy.ndarray
        Position of the reference object [x, y, z] in meters (ECI)
    reference_velocity : numpy.ndarray
        Velocity of the reference object [vx, vy, vz] in m/s (ECI)

    Notes
    -----
    For circular orbits, RSW is identical to RIC. For elliptical orbits,
    there's a small difference because S aligns with velocity while I is
    perpendicular to R.

    References
    ----------
    .. [1] Schaub, H., and Junkins, J. L., 2009, "Analytical Mechanics of
           Space Systems," 2nd ed., AIAA Education Series.

    """

    name: str = Property(default="RSW", doc="Name of the reference frame")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from inertial frame to RSW frame."""
        r = self.reference_position
        v = self.reference_velocity

        # Radial unit vector (outward from Earth center)
        r_hat = r / np.linalg.norm(r)

        # Cross-track (normal to orbital plane)
        h = np.cross(r, v)
        w_hat = h / np.linalg.norm(h)

        # Along-track (in velocity direction, in orbital plane)
        # We could use v directly, but for consistency with RIC, use perpendicular to R
        s_hat = np.cross(w_hat, r_hat)

        # Rotation matrix: rows are the unit vectors
        return np.array([r_hat, s_hat, w_hat])


class LVLHFrame(RelativeFrame):
    """Local Vertical Local Horizontal (LVLH) relative frame.

    The LVLH frame is defined with the local vertical pointing toward Earth:

    - **X (Velocity/In-track)**: Along the velocity direction
    - **Y (Cross-track)**: Normal to the orbital plane
    - **Z (Nadir)**: Points toward Earth's center (local vertical, radially inward)

    This frame is commonly used for spacecraft attitude control and
    Earth observation missions where "down" is important.

    Parameters
    ----------
    reference_position : numpy.ndarray
        Position of the reference object [x, y, z] in meters (ECI)
    reference_velocity : numpy.ndarray
        Velocity of the reference object [vx, vy, vz] in m/s (ECI)

    Notes
    -----
    The LVLH frame differs from RIC/RSW primarily in that:

    1. The Z-axis points toward Earth (nadir) instead of away (zenith)
    2. The coordinate order is typically [velocity, cross-track, nadir]

    Some definitions have variations in axis ordering. This implementation
    uses [X=velocity, Y=cross-track, Z=nadir].

    References
    ----------
    .. [1] Wertz, J. R., 1978, "Spacecraft Attitude Determination and Control,"
           D. Reidel Publishing, Chapter 2.

    Examples
    --------
    >>> import numpy as np
    >>> # Reference object in circular orbit
    >>> ref_pos = np.array([7000000.0, 0.0, 0.0])
    >>> ref_vel = np.array([0.0, 7546.0, 0.0])
    >>> lvlh = LVLHFrame(
    ...     name="LVLH",
    ...     reference_position=ref_pos,
    ...     reference_velocity=ref_vel
    ... )

    """

    name: str = Property(default="LVLH", doc="Name of the reference frame")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from inertial frame to LVLH frame."""
        r = self.reference_position
        v = self.reference_velocity

        # Z-axis: nadir direction (toward Earth center, opposite to radial)
        z_hat = -r / np.linalg.norm(r)

        # Y-axis: cross-track (negative angular momentum direction)
        h = np.cross(r, v)
        y_hat = -h / np.linalg.norm(h)

        # X-axis: velocity direction (completes right-handed system)
        x_hat = np.cross(y_hat, z_hat)

        # Rotation matrix: rows are the unit vectors
        return np.array([x_hat, y_hat, z_hat])


def compute_relative_state(target_position: np.ndarray, target_velocity: np.ndarray,
                           chaser_position: np.ndarray, chaser_velocity: np.ndarray,
                           frame_type: str = 'RIC') -> tuple[np.ndarray, np.ndarray]:
    """Compute relative state of chaser with respect to target in a relative frame.

    This is a convenience function for computing relative position and velocity
    between two objects.

    Parameters
    ----------
    target_position : numpy.ndarray
        Position of target (reference) object [x, y, z] in meters (ECI)
    target_velocity : numpy.ndarray
        Velocity of target object [vx, vy, vz] in m/s (ECI)
    chaser_position : numpy.ndarray
        Position of chaser object [x, y, z] in meters (ECI)
    chaser_velocity : numpy.ndarray
        Velocity of chaser object [vx, vy, vz] in m/s (ECI)
    frame_type : str, optional
        Type of relative frame: 'RIC', 'RSW', or 'LVLH'. Default is 'RIC'.

    Returns
    -------
    rel_position : numpy.ndarray
        Relative position [x, y, z] in the specified frame (meters)
    rel_velocity : numpy.ndarray
        Relative velocity [vx, vy, vz] in the specified frame (m/s)

    Raises
    ------
    ValueError
        If frame_type is not recognized

    Examples
    --------
    >>> import numpy as np
    >>> # Target and chaser in similar orbits
    >>> target_pos = np.array([7000000.0, 0.0, 0.0])
    >>> target_vel = np.array([0.0, 7546.0, 0.0])
    >>> chaser_pos = np.array([7001000.0, 500.0, 100.0])
    >>> chaser_vel = np.array([0.0, 7546.0, 0.0])
    >>> rel_pos, rel_vel = compute_relative_state(
    ...     target_pos, target_vel, chaser_pos, chaser_vel, frame_type='RIC')
    >>> print(f"Relative position (RIC): {rel_pos}")

    """
    frame_classes = {
        'RIC': RICFrame,
        'RSW': RSWFrame,
        'LVLH': LVLHFrame
    }

    if frame_type.upper() not in frame_classes:
        raise ValueError(f"Unknown frame type '{frame_type}'. "
                         f"Must be one of: {list(frame_classes.keys())}")

    frame_class = frame_classes[frame_type.upper()]
    frame = frame_class(
        name=frame_type.upper(),
        reference_position=target_position,
        reference_velocity=target_velocity
    )

    return frame.inertial_to_relative(chaser_position, chaser_velocity)


# Earth rotation rate in rad/s (WGS84)
EARTH_ROTATION_RATE: float = 7.2921150e-5
"""Earth's rotation rate in radians per second (WGS84 value).

This is the angular velocity of Earth's rotation about its polar axis,
used in coordinate transformations between ECI and ECEF frames.

Reference: IERS Conventions (2010), Table 1.1
"""

# Earth's rotation vector (aligned with z-axis in ECI)
EARTH_ROTATION_VECTOR: np.ndarray = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
"""Earth's rotation vector in ECI coordinates [rad/s].

The rotation vector is aligned with the z-axis (polar axis) in the ECI frame.
Magnitude equals EARTH_ROTATION_RATE.
"""


def kinematic_ecef_to_eci(state: KinematicState, timestamp: datetime) -> KinematicState:
    r"""Transform a kinematic state from ECEF to ECI frame.

    This function properly transforms position, velocity, and acceleration
    from ECEF (Earth-Centered Earth-Fixed) to ECI (Earth-Centered Inertial)
    coordinates, accounting for Earth's rotation.

    Parameters
    ----------
    state : KinematicState
        Kinematic state in ECEF coordinates
    timestamp : datetime.datetime
        UTC timestamp for the transformation

    Returns
    -------
    KinematicState
        Kinematic state in ECI coordinates

    Notes
    -----
    The transformation equations are:

    **Position**:

    .. math::

        \mathbf{r}_{ECI} = \mathbf{R}^T \mathbf{r}_{ECEF}

    **Velocity** (includes Earth rotation):

    .. math::

        \mathbf{v}_{ECI} = \mathbf{R}^T \mathbf{v}_{ECEF} + \boldsymbol{\omega} \times \mathbf{r}_{ECI}

    **Acceleration** (includes Coriolis and centripetal):

    .. math::

        \mathbf{a}_{ECI} = \mathbf{R}^T \mathbf{a}_{ECEF}
                          + 2\boldsymbol{\omega} \times \mathbf{v}_{ECI}
                          + \boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_{ECI})

    where :math:`\mathbf{R}` is the ECEF-to-ECI rotation matrix and
    :math:`\boldsymbol{\omega} = [0, 0, \omega_E]` is Earth's rotation vector.

    Examples
    --------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> # ECEF state for an object on the equator
    >>> state_ecef = KinematicState(
    ...     position=np.array([6378137.0, 0.0, 0.0]),
    ...     velocity=np.array([0.0, 0.0, 0.0]),
    ...     acceleration=np.array([0.0, 0.0, 0.0])
    ... )
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_eci = kinematic_ecef_to_eci(state_ecef, timestamp)
    >>> # In ECI, the object has velocity due to Earth's rotation
    >>> print(f"ECI velocity magnitude: {state_eci.speed:.1f} m/s")
    ECI velocity magnitude: 465.1 m/s

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press, Chapter 3.
    .. [2] Montenbruck, O., and Gill, E., 2000, "Satellite Orbits: Models, Methods
           and Applications," Springer-Verlag, Section 5.2.

    """
    from ..functions.coordinates import ecef_to_eci

    # Transform position
    pos_eci = ecef_to_eci(state.position, timestamp)

    # Get rotation matrix (we can derive it from the transformation)
    # Compute ERA for rotation
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_centuries = dt / (86400.0 * 36525.0)
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_centuries)
    era = np.fmod(era, 2.0 * np.pi)

    cos_era = np.cos(era)
    sin_era = np.sin(era)

    # Rotation matrix from ECEF to ECI (transpose of ECI-to-ECEF)
    R = np.array([
        [cos_era, -sin_era, 0.0],
        [sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Transform velocity: v_eci = R * v_ecef + omega x r_eci
    vel_ecef_in_eci = R @ state.velocity
    omega_cross_r = np.cross(EARTH_ROTATION_VECTOR, pos_eci)
    vel_eci = vel_ecef_in_eci + omega_cross_r

    # Transform acceleration if present
    acc_eci = None
    if state.has_acceleration:
        # a_eci = R * a_ecef + 2*omega x v_eci + omega x (omega x r_eci)
        acc_ecef_in_eci = R @ state.acceleration
        coriolis = 2.0 * np.cross(EARTH_ROTATION_VECTOR, vel_eci)
        centripetal = np.cross(EARTH_ROTATION_VECTOR, omega_cross_r)
        acc_eci = acc_ecef_in_eci + coriolis + centripetal

    return KinematicState(
        position=pos_eci,
        velocity=vel_eci,
        acceleration=acc_eci,
        timestamp=timestamp,
        frame=None  # Could set to an ECI frame instance
    )


def kinematic_eci_to_ecef(state: KinematicState, timestamp: datetime) -> KinematicState:
    r"""Transform a kinematic state from ECI to ECEF frame.

    This function properly transforms position, velocity, and acceleration
    from ECI (Earth-Centered Inertial) to ECEF (Earth-Centered Earth-Fixed)
    coordinates, accounting for Earth's rotation.

    Parameters
    ----------
    state : KinematicState
        Kinematic state in ECI coordinates
    timestamp : datetime.datetime
        UTC timestamp for the transformation

    Returns
    -------
    KinematicState
        Kinematic state in ECEF coordinates

    Notes
    -----
    The transformation equations are the inverse of ECEF-to-ECI:

    **Position**:

    .. math::

        \mathbf{r}_{ECEF} = \mathbf{R} \mathbf{r}_{ECI}

    **Velocity**:

    .. math::

        \mathbf{v}_{ECEF} = \mathbf{R} (\mathbf{v}_{ECI} - \boldsymbol{\omega} \times \mathbf{r}_{ECI})

    **Acceleration**:

    .. math::

        \mathbf{a}_{ECEF} = \mathbf{R} [\mathbf{a}_{ECI}
                          - 2\boldsymbol{\omega} \times \mathbf{v}_{ECI}
                          - \boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_{ECI})]

    Examples
    --------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> # ECI state for a satellite
    >>> state_eci = KinematicState(
    ...     position=np.array([7000000.0, 0.0, 0.0]),
    ...     velocity=np.array([0.0, 7500.0, 0.0]),
    ...     acceleration=np.array([0.0, 0.0, -8.0])
    ... )
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_ecef = kinematic_eci_to_ecef(state_eci, timestamp)

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press, Chapter 3.

    """
    from ..functions.coordinates import eci_to_ecef

    # Compute ERA for rotation
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_centuries = dt / (86400.0 * 36525.0)
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_centuries)
    era = np.fmod(era, 2.0 * np.pi)

    cos_era = np.cos(era)
    sin_era = np.sin(era)

    # Rotation matrix from ECI to ECEF
    R = np.array([
        [cos_era, sin_era, 0.0],
        [-sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Transform position
    pos_ecef = R @ state.position

    # Transform velocity: v_ecef = R * (v_eci - omega x r_eci)
    omega_cross_r = np.cross(EARTH_ROTATION_VECTOR, state.position)
    vel_ecef = R @ (state.velocity - omega_cross_r)

    # Transform acceleration if present
    acc_ecef = None
    if state.has_acceleration:
        # a_ecef = R * [a_eci - 2*omega x v_eci - omega x (omega x r_eci)]
        coriolis = 2.0 * np.cross(EARTH_ROTATION_VECTOR, state.velocity)
        centripetal = np.cross(EARTH_ROTATION_VECTOR, omega_cross_r)
        acc_ecef = R @ (state.acceleration - coriolis - centripetal)

    return KinematicState(
        position=pos_ecef,
        velocity=vel_ecef,
        acceleration=acc_ecef,
        timestamp=timestamp,
        frame=None  # Could set to an ECEF frame instance
    )


# =============================================================================
# Proximity Operations Utilities
# =============================================================================

def compute_range(position1: np.ndarray, position2: np.ndarray) -> float:
    """Compute the range (distance) between two positions.

    Parameters
    ----------
    position1 : np.ndarray
        First position vector [x, y, z] in meters.
    position2 : np.ndarray
        Second position vector [x, y, z] in meters.

    Returns
    -------
    float
        Range (distance) in meters.

    Examples
    --------
    >>> import numpy as np
    >>> pos1 = np.array([1000.0, 0.0, 0.0])
    >>> pos2 = np.array([0.0, 1000.0, 0.0])
    >>> compute_range(pos1, pos2)  # doctest: +ELLIPSIS
    1414.213...

    """
    relative = np.asarray(position2) - np.asarray(position1)
    return float(np.linalg.norm(relative))


def compute_range_rate(position1: np.ndarray, velocity1: np.ndarray,
                       position2: np.ndarray, velocity2: np.ndarray) -> float:
    """Compute the range-rate (time derivative of range) between two objects.

    The range-rate is positive when the objects are separating and negative
    when they are approaching. This is the radial component of relative
    velocity along the line of sight.

    Parameters
    ----------
    position1 : np.ndarray
        First position vector [x, y, z] in meters.
    velocity1 : np.ndarray
        First velocity vector [vx, vy, vz] in meters/second.
    position2 : np.ndarray
        Second position vector [x, y, z] in meters.
    velocity2 : np.ndarray
        Second velocity vector [vx, vy, vz] in meters/second.

    Returns
    -------
    float
        Range-rate in meters/second. Positive = separating, negative = closing.

    Examples
    --------
    >>> import numpy as np
    >>> # Object 1 stationary at origin
    >>> pos1 = np.array([0.0, 0.0, 0.0])
    >>> vel1 = np.array([0.0, 0.0, 0.0])
    >>> # Object 2 moving toward origin
    >>> pos2 = np.array([1000.0, 0.0, 0.0])
    >>> vel2 = np.array([-10.0, 0.0, 0.0])  # Moving toward origin
    >>> compute_range_rate(pos1, vel1, pos2, vel2)
    -10.0

    """
    relative_pos = np.asarray(position2) - np.asarray(position1)
    relative_vel = np.asarray(velocity2) - np.asarray(velocity1)
    r = np.linalg.norm(relative_pos)

    if r < 1e-10:  # Avoid division by zero
        return 0.0

    # Range rate is projection of relative velocity onto line of sight
    line_of_sight = relative_pos / r
    return float(np.dot(relative_vel, line_of_sight))


def compute_closest_approach(position1: np.ndarray, velocity1: np.ndarray,
                             position2: np.ndarray, velocity2: np.ndarray,
                             max_time: float = 86400.0) -> tuple:
    """Compute the time and distance of closest approach.

    Assumes straight-line (constant velocity) motion for both objects.
    For orbital mechanics with gravitational effects, use dedicated
    conjunction analysis tools.

    Parameters
    ----------
    position1 : np.ndarray
        First position vector [x, y, z] in meters.
    velocity1 : np.ndarray
        First velocity vector [vx, vy, vz] in meters/second.
    position2 : np.ndarray
        Second position vector [x, y, z] in meters.
    velocity2 : np.ndarray
        Second velocity vector [vx, vy, vz] in meters/second.
    max_time : float, optional
        Maximum time to consider (seconds). Default is 86400.0 (one day).
        If closest approach occurs beyond this time, returns (max_time, range).

    Returns
    -------
    time_to_closest : float
        Time to closest approach in seconds. Returns 0 if closest approach
        is in the past or if objects are stationary relative to each other.
    closest_distance : float
        Distance at closest approach in meters.

    Notes
    -----
    For straight-line motion, the squared range as a function of time is:

    .. math::

        r^2(t) = |\\mathbf{r}_0 + \\mathbf{v}_{rel} t|^2

    Taking the derivative and setting to zero gives:

    .. math::

        t_{ca} = -\\frac{\\mathbf{r}_0 \\cdot \\mathbf{v}_{rel}}
                       {|\\mathbf{v}_{rel}|^2}

    Examples
    --------
    >>> import numpy as np
    >>> # Two objects approaching each other
    >>> pos1 = np.array([0.0, 0.0, 0.0])
    >>> vel1 = np.array([0.0, 0.0, 0.0])
    >>> pos2 = np.array([1000.0, 100.0, 0.0])  # Offset in y
    >>> vel2 = np.array([-10.0, 0.0, 0.0])  # Moving toward x=0
    >>> t_ca, d_ca = compute_closest_approach(pos1, vel1, pos2, vel2)
    >>> round(t_ca, 1)
    100.0
    >>> round(d_ca, 1)
    100.0

    """
    relative_pos = np.asarray(position2) - np.asarray(position1)
    relative_vel = np.asarray(velocity2) - np.asarray(velocity1)

    v_rel_sq = np.dot(relative_vel, relative_vel)

    if v_rel_sq < 1e-20:  # Objects are stationary relative to each other
        current_range = float(np.linalg.norm(relative_pos))
        return 0.0, current_range

    # Time of closest approach
    t_ca = -np.dot(relative_pos, relative_vel) / v_rel_sq

    # If closest approach is in the past, report current state
    if t_ca < 0:
        current_range = float(np.linalg.norm(relative_pos))
        return 0.0, current_range

    # If closest approach is beyond max_time, clamp
    if t_ca > max_time:
        future_pos = relative_pos + relative_vel * max_time
        return max_time, float(np.linalg.norm(future_pos))

    # Distance at closest approach
    pos_at_ca = relative_pos + relative_vel * t_ca
    d_ca = float(np.linalg.norm(pos_at_ca))

    return t_ca, d_ca


def compute_miss_distance(position1: np.ndarray, velocity1: np.ndarray,
                          position2: np.ndarray, velocity2: np.ndarray) -> float:
    """Compute the miss distance (minimum separation distance).

    The miss distance is the perpendicular distance between the two
    straight-line trajectories. This is useful for collision avoidance
    and conjunction assessment.

    Parameters
    ----------
    position1 : np.ndarray
        First position vector [x, y, z] in meters.
    velocity1 : np.ndarray
        First velocity vector [vx, vy, vz] in meters/second.
    position2 : np.ndarray
        Second position vector [x, y, z] in meters.
    velocity2 : np.ndarray
        Second velocity vector [vx, vy, vz] in meters/second.

    Returns
    -------
    float
        Miss distance in meters.

    Notes
    -----
    This is equivalent to the closest_distance from compute_closest_approach
    for straight-line motion, but computed more directly.

    Examples
    --------
    >>> import numpy as np
    >>> # Parallel trajectories
    >>> pos1 = np.array([0.0, 0.0, 0.0])
    >>> vel1 = np.array([10.0, 0.0, 0.0])
    >>> pos2 = np.array([0.0, 100.0, 0.0])
    >>> vel2 = np.array([10.0, 0.0, 0.0])
    >>> compute_miss_distance(pos1, vel1, pos2, vel2)
    100.0

    """
    _, miss_dist = compute_closest_approach(position1, velocity1,
                                            position2, velocity2)
    return miss_dist


def is_in_keep_out_zone(position: np.ndarray, zone_center: np.ndarray,
                        zone_radius: float) -> bool:
    """Check if a position is within a spherical keep-out zone.

    Parameters
    ----------
    position : np.ndarray
        Position to check [x, y, z] in meters.
    zone_center : np.ndarray
        Center of the keep-out zone [x, y, z] in meters.
    zone_radius : float
        Radius of the keep-out zone in meters.

    Returns
    -------
    bool
        True if position is inside the keep-out zone.

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.array([50.0, 0.0, 0.0])
    >>> center = np.array([0.0, 0.0, 0.0])
    >>> is_in_keep_out_zone(pos, center, 100.0)
    True
    >>> is_in_keep_out_zone(pos, center, 40.0)
    False

    """
    distance = compute_range(position, zone_center)
    return distance < zone_radius


def is_in_ellipsoidal_keep_out_zone(position: np.ndarray,
                                    zone_center: np.ndarray,
                                    semi_axes: np.ndarray,
                                    rotation_matrix: np.ndarray = None) -> bool:
    """Check if a position is within an ellipsoidal keep-out zone.

    This is useful for defining anisotropic safety zones, such as
    elongated zones along the velocity direction.

    Parameters
    ----------
    position : np.ndarray
        Position to check [x, y, z] in meters.
    zone_center : np.ndarray
        Center of the keep-out zone [x, y, z] in meters.
    semi_axes : np.ndarray
        Semi-axes of the ellipsoid [a, b, c] in meters along x, y, z
        (before rotation).
    rotation_matrix : np.ndarray, optional
        3x3 rotation matrix defining the ellipsoid orientation.
        If None, uses identity (axes aligned with frame).

    Returns
    -------
    bool
        True if position is inside the ellipsoidal keep-out zone.

    Notes
    -----
    The check uses the ellipsoid equation:

    .. math::

        \\frac{x'^2}{a^2} + \\frac{y'^2}{b^2} + \\frac{z'^2}{c^2} < 1

    where :math:`(x', y', z')` is the position in the ellipsoid's local frame.

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.array([80.0, 0.0, 0.0])
    >>> center = np.array([0.0, 0.0, 0.0])
    >>> semi_axes = np.array([100.0, 50.0, 50.0])  # Elongated in x
    >>> is_in_ellipsoidal_keep_out_zone(pos, center, semi_axes)
    True
    >>> pos = np.array([0.0, 80.0, 0.0])
    >>> is_in_ellipsoidal_keep_out_zone(pos, center, semi_axes)
    False

    """
    relative = np.asarray(position) - np.asarray(zone_center)

    if rotation_matrix is not None:
        # Transform to ellipsoid's local frame
        relative = rotation_matrix.T @ relative

    # Normalized ellipsoid equation
    semi = np.asarray(semi_axes)
    normalized_dist_sq = np.sum((relative / semi) ** 2)

    return normalized_dist_sq < 1.0


def compute_conjunction_geometry(position1: np.ndarray, velocity1: np.ndarray,
                                 position2: np.ndarray, velocity2: np.ndarray,
                                 covariance1: np.ndarray = None,
                                 covariance2: np.ndarray = None) -> dict:
    """Compute comprehensive conjunction geometry parameters.

    This function computes all standard conjunction assessment metrics
    for collision probability analysis between two objects.

    Parameters
    ----------
    position1 : np.ndarray
        First position vector [x, y, z] in meters.
    velocity1 : np.ndarray
        First velocity vector [vx, vy, vz] in meters/second.
    position2 : np.ndarray
        Second position vector [x, y, z] in meters.
    velocity2 : np.ndarray
        Second velocity vector [vx, vy, vz] in meters/second.
    covariance1 : np.ndarray, optional
        3x3 position covariance matrix for object 1 (meters^2).
    covariance2 : np.ndarray, optional
        3x3 position covariance matrix for object 2 (meters^2).

    Returns
    -------
    dict
        Dictionary containing:

        - 'range': Current range in meters.
        - 'range_rate': Range rate in meters/second.
        - 'time_to_closest_approach': Time to TCA in seconds.
        - 'miss_distance': Miss distance in meters.
        - 'relative_velocity_magnitude': Relative velocity magnitude in m/s.
        - 'approach_angle': Angle between relative position and velocity (rad).
        - 'combined_covariance': Combined 3x3 covariance (if provided).
        - 'mahalanobis_distance': Mahalanobis distance (if covariances given).

    Examples
    --------
    >>> import numpy as np
    >>> pos1 = np.array([7000000.0, 0.0, 0.0])
    >>> vel1 = np.array([0.0, 7500.0, 0.0])
    >>> pos2 = np.array([7001000.0, 500.0, 0.0])
    >>> vel2 = np.array([0.0, 7500.0, 100.0])
    >>> geometry = compute_conjunction_geometry(pos1, vel1, pos2, vel2)
    >>> 'range' in geometry
    True
    >>> 'miss_distance' in geometry
    True

    """
    pos1 = np.asarray(position1)
    vel1 = np.asarray(velocity1)
    pos2 = np.asarray(position2)
    vel2 = np.asarray(velocity2)

    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1

    current_range = float(np.linalg.norm(relative_pos))
    rel_vel_mag = float(np.linalg.norm(relative_vel))

    # Range rate
    range_rate = compute_range_rate(pos1, vel1, pos2, vel2)

    # Closest approach
    t_ca, miss_dist = compute_closest_approach(pos1, vel1, pos2, vel2)

    # Approach angle (angle between relative position and velocity)
    if current_range > 1e-10 and rel_vel_mag > 1e-10:
        cos_angle = np.dot(relative_pos, relative_vel) / (
            current_range * rel_vel_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        approach_angle = float(np.arccos(cos_angle))
    else:
        approach_angle = 0.0

    result = {
        'range': current_range,
        'range_rate': range_rate,
        'time_to_closest_approach': t_ca,
        'miss_distance': miss_dist,
        'relative_velocity_magnitude': rel_vel_mag,
        'approach_angle': approach_angle,
    }

    # Add covariance-based metrics if provided
    if covariance1 is not None and covariance2 is not None:
        combined_cov = np.asarray(covariance1) + np.asarray(covariance2)
        result['combined_covariance'] = combined_cov

        # Mahalanobis distance
        try:
            cov_inv = np.linalg.inv(combined_cov)
            mahal_sq = relative_pos @ cov_inv @ relative_pos
            result['mahalanobis_distance'] = float(np.sqrt(mahal_sq))
        except np.linalg.LinAlgError:
            result['mahalanobis_distance'] = None

    return result


# =============================================================================
# Topocentric Frames
# =============================================================================

class TopocentricFrame(ReferenceFrame):
    r"""Base class for topocentric (observer-centered) reference frames.

    Topocentric frames are centered at an observer's location on or near
    Earth's surface. They provide a local coordinate system useful for
    sensor-centric tracking applications.

    The observer position can be specified in geodetic coordinates
    (latitude, longitude, altitude) or ECEF coordinates.

    Parameters
    ----------
    name : str
        Human-readable name for the frame.
    latitude : float
        Observer's geodetic latitude in radians.
    longitude : float
        Observer's geodetic longitude in radians.
    altitude : float
        Observer's altitude above the reference ellipsoid in meters.
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid. Default is WGS84.

    """

    latitude: float = Property(
        doc="Observer's geodetic latitude in radians (positive North)")
    longitude: float = Property(
        doc="Observer's geodetic longitude in radians (positive East)")
    altitude: float = Property(
        default=0.0,
        doc="Observer's altitude above the reference ellipsoid in meters")
    ellipsoid: ReferenceEllipsoid = Property(
        default=WGS84,
        doc="Reference ellipsoid for the observer position")

    @property
    def observer_ecef(self) -> np.ndarray:
        """Observer position in ECEF coordinates (meters)."""
        from ..functions.coordinates import geodetic_to_ecef
        ecef = geodetic_to_ecef(self.latitude, self.longitude, self.altitude,
                                self.ellipsoid)
        return np.array(ecef)

    @property
    @abstractmethod
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from ECEF to this local frame."""
        raise NotImplementedError

    def ecef_to_local(self, position_ecef: np.ndarray,
                      velocity_ecef: np.ndarray = None) -> tuple:
        """Transform ECEF position/velocity to local topocentric frame.

        Parameters
        ----------
        position_ecef : np.ndarray
            Position in ECEF coordinates [x, y, z] in meters.
        velocity_ecef : np.ndarray, optional
            Velocity in ECEF coordinates [vx, vy, vz] in m/s.

        Returns
        -------
        position_local : np.ndarray
            Position in local frame.
        velocity_local : np.ndarray or None
            Velocity in local frame, or None if not provided.

        """
        pos_ecef = np.asarray(position_ecef)
        relative_pos = pos_ecef - self.observer_ecef
        R = self.rotation_matrix

        pos_local = R @ relative_pos

        vel_local = None
        if velocity_ecef is not None:
            vel_local = R @ np.asarray(velocity_ecef)

        return pos_local, vel_local

    def local_to_ecef(self, position_local: np.ndarray,
                      velocity_local: np.ndarray = None) -> tuple:
        """Transform local topocentric position/velocity to ECEF.

        Parameters
        ----------
        position_local : np.ndarray
            Position in local frame.
        velocity_local : np.ndarray, optional
            Velocity in local frame.

        Returns
        -------
        position_ecef : np.ndarray
            Position in ECEF coordinates [x, y, z] in meters.
        velocity_ecef : np.ndarray or None
            Velocity in ECEF coordinates, or None if not provided.

        """
        R = self.rotation_matrix
        R_inv = R.T  # Orthogonal matrix, inverse is transpose

        pos_ecef = R_inv @ np.asarray(position_local) + self.observer_ecef

        vel_ecef = None
        if velocity_local is not None:
            vel_ecef = R_inv @ np.asarray(velocity_local)

        return pos_ecef, vel_ecef

    def transform_to(self, other_frame: ReferenceFrame, position: np.ndarray,
                     velocity: np.ndarray = None,
                     timestamp: datetime = None) -> tuple:
        """Transform position and velocity to another reference frame.

        Transformations go through ECEF as an intermediate frame.

        Parameters
        ----------
        other_frame : ReferenceFrame
            Target reference frame.
        position : np.ndarray
            Position vector in this frame.
        velocity : np.ndarray, optional
            Velocity vector in this frame.
        timestamp : datetime, optional
            Time for time-dependent transformations.

        Returns
        -------
        position : np.ndarray
            Position in target frame.
        velocity : np.ndarray or None
            Velocity in target frame.

        """
        # First transform to ECEF
        pos_ecef, vel_ecef = self.local_to_ecef(position, velocity)

        # If target is another TopocentricFrame, use its ecef_to_local
        if isinstance(other_frame, TopocentricFrame):
            return other_frame.ecef_to_local(pos_ecef, vel_ecef)

        # Otherwise, raise not implemented (could extend for ECI, etc.)
        raise NotImplementedError(
            f"Transformation from {self.name} to {other_frame.name} not implemented"
        )


class SEZFrame(TopocentricFrame):
    r"""South-East-Zenith (SEZ) topocentric reference frame.

    The SEZ frame is centered at an observer's location with:

    - **S (South)**: Points toward the South pole along the local meridian
    - **E (East)**: Points East, tangent to the parallel of latitude
    - **Z (Zenith)**: Points radially outward from Earth's center

    This frame is commonly used in radar tracking applications where
    the zenith direction (elevation) is a natural reference.

    The rotation matrix from ECEF to SEZ is:

    .. math::

        \mathbf{R}_{SEZ} = \begin{bmatrix}
            \cos\phi \cos\lambda & \cos\phi \sin\lambda & -\sin\phi \\
            -\sin\lambda & \cos\lambda & 0 \\
            \sin\phi \cos\lambda & \sin\phi \sin\lambda & \cos\phi
        \end{bmatrix}

    where :math:`\phi` is latitude and :math:`\lambda` is longitude.

    Examples
    --------
    >>> import numpy as np
    >>> # Observer at 45°N, 0°E
    >>> frame = SEZFrame(
    ...     name="Radar Site",
    ...     latitude=np.radians(45.0),
    ...     longitude=0.0
    ... )
    >>> # Target directly above observer at 100km altitude
    >>> target_ecef = frame.observer_ecef + np.array([0, 0, 100000])
    >>> pos_sez, _ = frame.ecef_to_local(target_ecef)

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and
           Applications," 4th ed., Microcosm Press.

    """

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from ECEF to SEZ frame."""
        lat = self.latitude
        lon = self.longitude

        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        # SEZ rotation matrix
        R = np.array([
            [cos_lat * cos_lon, cos_lat * sin_lon, -sin_lat],
            [-sin_lon, cos_lon, 0.0],
            [sin_lat * cos_lon, sin_lat * sin_lon, cos_lat]
        ])
        return R


class ENUFrame(TopocentricFrame):
    r"""East-North-Up (ENU) topocentric reference frame.

    The ENU frame is centered at an observer's location with:

    - **E (East)**: Points East, tangent to the parallel of latitude
    - **N (North)**: Points North, tangent to the local meridian
    - **U (Up)**: Points radially outward (zenith direction)

    This is a right-handed coordinate system commonly used in geodesy
    and navigation applications.

    The rotation matrix from ECEF to ENU is:

    .. math::

        \mathbf{R}_{ENU} = \begin{bmatrix}
            -\sin\lambda & \cos\lambda & 0 \\
            -\sin\phi \cos\lambda & -\sin\phi \sin\lambda & \cos\phi \\
            \cos\phi \cos\lambda & \cos\phi \sin\lambda & \sin\phi
        \end{bmatrix}

    where :math:`\phi` is latitude and :math:`\lambda` is longitude.

    Note
    ----
    The Up direction is defined by the local vertical (normal to the
    reference ellipsoid), not the geocentric radial direction. For
    an oblate ellipsoid, these differ slightly except at the equator
    and poles.

    Examples
    --------
    >>> import numpy as np
    >>> # Observer at equator, prime meridian
    >>> frame = ENUFrame(
    ...     name="Ground Station",
    ...     latitude=0.0,
    ...     longitude=0.0
    ... )
    >>> # At equator: E points +Y, N points +Z, U points +X (in ECEF)

    See Also
    --------
    :class:`NEDFrame` : North-East-Down frame (aviation convention)
    :class:`SEZFrame` : South-East-Zenith frame (radar convention)

    """

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from ECEF to ENU frame."""
        lat = self.latitude
        lon = self.longitude

        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        # ENU rotation matrix
        R = np.array([
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        return R


class NEDFrame(TopocentricFrame):
    r"""North-East-Down (NED) topocentric reference frame.

    The NED frame is centered at an observer's location with:

    - **N (North)**: Points North, tangent to the local meridian
    - **E (East)**: Points East, tangent to the parallel of latitude
    - **D (Down)**: Points toward Earth's center (nadir direction)

    This is a right-handed coordinate system commonly used in aviation
    and aerospace applications.

    The rotation matrix from ECEF to NED is:

    .. math::

        \mathbf{R}_{NED} = \begin{bmatrix}
            -\sin\phi \cos\lambda & -\sin\phi \sin\lambda & \cos\phi \\
            -\sin\lambda & \cos\lambda & 0 \\
            -\cos\phi \cos\lambda & -\cos\phi \sin\lambda & -\sin\phi
        \end{bmatrix}

    where :math:`\phi` is latitude and :math:`\lambda` is longitude.

    Examples
    --------
    >>> import numpy as np
    >>> # Aircraft at 45°N, 90°E
    >>> frame = NEDFrame(
    ...     name="Aircraft",
    ...     latitude=np.radians(45.0),
    ...     longitude=np.radians(90.0),
    ...     altitude=10000.0
    ... )

    See Also
    --------
    :class:`ENUFrame` : East-North-Up frame (geodesy convention)

    """

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from ECEF to NED frame."""
        lat = self.latitude
        lon = self.longitude

        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        # NED rotation matrix
        R = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        return R


# =============================================================================
# Radar/Sensor Line-of-Sight Calculations
# =============================================================================

def compute_azimuth_elevation(observer_lat: float, observer_lon: float,
                              observer_alt: float,
                              target_ecef: np.ndarray,
                              ellipsoid: ReferenceEllipsoid = WGS84) -> tuple:
    """Compute azimuth and elevation angles from observer to target.

    Parameters
    ----------
    observer_lat : float
        Observer's geodetic latitude in radians.
    observer_lon : float
        Observer's geodetic longitude in radians.
    observer_alt : float
        Observer's altitude above the ellipsoid in meters.
    target_ecef : np.ndarray
        Target position in ECEF coordinates [x, y, z] in meters.
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid. Default is WGS84.

    Returns
    -------
    azimuth : float
        Azimuth angle in radians, measured clockwise from North
        (0 = North, π/2 = East, π = South, 3π/2 = West).
    elevation : float
        Elevation angle in radians above the local horizon
        (0 = horizon, π/2 = zenith, -π/2 = nadir).
    range_ : float
        Slant range (distance) to target in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Observer at equator, prime meridian
    >>> obs_lat, obs_lon, obs_alt = 0.0, 0.0, 0.0
    >>> # Target directly east at same altitude
    >>> from stonesoup.functions.coordinates import geodetic_to_ecef
    >>> target = geodetic_to_ecef(0.0, np.radians(1.0), 0.0)
    >>> az, el, rng = compute_azimuth_elevation(obs_lat, obs_lon, obs_alt, target)
    >>> round(np.degrees(az))  # Should be ~90 degrees (East)
    90

    """
    # Create ENU frame at observer location
    enu = ENUFrame(
        name="temp",
        latitude=observer_lat,
        longitude=observer_lon,
        altitude=observer_alt,
        ellipsoid=ellipsoid
    )

    # Transform target to ENU
    pos_enu, _ = enu.ecef_to_local(target_ecef)
    east, north, up = pos_enu

    # Compute range
    range_ = float(np.linalg.norm(pos_enu))

    if range_ < 1e-10:
        return 0.0, np.pi / 2, 0.0  # Target at observer, return zenith

    # Compute azimuth (clockwise from North)
    azimuth = float(np.arctan2(east, north))
    if azimuth < 0:
        azimuth += 2 * np.pi

    # Compute elevation
    horizontal_dist = np.sqrt(east**2 + north**2)
    elevation = float(np.arctan2(up, horizontal_dist))

    return azimuth, elevation, range_


def compute_look_angles(observer_lat: float, observer_lon: float,
                        observer_alt: float,
                        target_ecef: np.ndarray,
                        ellipsoid: ReferenceEllipsoid = WGS84) -> dict:
    """Compute comprehensive look angle information from observer to target.

    This function provides all common representations of the viewing
    geometry from an observer to a target.

    Parameters
    ----------
    observer_lat : float
        Observer's geodetic latitude in radians.
    observer_lon : float
        Observer's geodetic longitude in radians.
    observer_alt : float
        Observer's altitude above the ellipsoid in meters.
    target_ecef : np.ndarray
        Target position in ECEF coordinates [x, y, z] in meters.
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid. Default is WGS84.

    Returns
    -------
    dict
        Dictionary containing:

        - 'azimuth': Azimuth in radians (clockwise from North)
        - 'azimuth_deg': Azimuth in degrees
        - 'elevation': Elevation in radians above horizon
        - 'elevation_deg': Elevation in degrees
        - 'range': Slant range in meters
        - 'range_rate': Range rate (if target velocity provided), m/s
        - 'position_enu': [E, N, U] position vector in meters
        - 'position_sez': [S, E, Z] position vector in meters
        - 'position_ned': [N, E, D] position vector in meters
        - 'visible': True if elevation > 0

    Examples
    --------
    >>> import numpy as np
    >>> from stonesoup.functions.coordinates import geodetic_to_ecef
    >>> # Observer at 45°N, 0°E, sea level
    >>> obs_lat = np.radians(45.0)
    >>> obs_lon = 0.0
    >>> obs_alt = 0.0
    >>> # Target at 45°N, 1°E, 10km altitude
    >>> target = geodetic_to_ecef(np.radians(45.0), np.radians(1.0), 10000.0)
    >>> angles = compute_look_angles(obs_lat, obs_lon, obs_alt, target)
    >>> angles['visible']
    True

    """
    # Create frames
    enu = ENUFrame(name="ENU", latitude=observer_lat, longitude=observer_lon,
                   altitude=observer_alt, ellipsoid=ellipsoid)
    sez = SEZFrame(name="SEZ", latitude=observer_lat, longitude=observer_lon,
                   altitude=observer_alt, ellipsoid=ellipsoid)
    ned = NEDFrame(name="NED", latitude=observer_lat, longitude=observer_lon,
                   altitude=observer_alt, ellipsoid=ellipsoid)

    # Transform to all frames
    pos_enu, _ = enu.ecef_to_local(target_ecef)
    pos_sez, _ = sez.ecef_to_local(target_ecef)
    pos_ned, _ = ned.ecef_to_local(target_ecef)

    # Compute azimuth/elevation
    azimuth, elevation, range_ = compute_azimuth_elevation(
        observer_lat, observer_lon, observer_alt, target_ecef, ellipsoid)

    return {
        'azimuth': azimuth,
        'azimuth_deg': float(np.degrees(azimuth)),
        'elevation': elevation,
        'elevation_deg': float(np.degrees(elevation)),
        'range': range_,
        'position_enu': pos_enu,
        'position_sez': pos_sez,
        'position_ned': pos_ned,
        'visible': elevation > 0
    }


def ecef_to_aer(target_ecef: np.ndarray,
                observer_lat: float, observer_lon: float,
                observer_alt: float,
                ellipsoid: ReferenceEllipsoid = WGS84) -> tuple:
    """Convert ECEF position to Azimuth-Elevation-Range (AER) coordinates.

    This is a convenience function for radar/sensor applications.

    Parameters
    ----------
    target_ecef : np.ndarray
        Target position in ECEF [x, y, z] in meters.
    observer_lat : float
        Observer latitude in radians.
    observer_lon : float
        Observer longitude in radians.
    observer_alt : float
        Observer altitude in meters.
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid. Default is WGS84.

    Returns
    -------
    azimuth : float
        Azimuth in radians (clockwise from North).
    elevation : float
        Elevation in radians above horizon.
    range_ : float
        Slant range in meters.

    """
    return compute_azimuth_elevation(observer_lat, observer_lon, observer_alt,
                                     target_ecef, ellipsoid)


def aer_to_ecef(azimuth: float, elevation: float, range_: float,
                observer_lat: float, observer_lon: float,
                observer_alt: float,
                ellipsoid: ReferenceEllipsoid = WGS84) -> np.ndarray:
    """Convert Azimuth-Elevation-Range (AER) to ECEF coordinates.

    Parameters
    ----------
    azimuth : float
        Azimuth in radians (clockwise from North).
    elevation : float
        Elevation in radians above horizon.
    range_ : float
        Slant range in meters.
    observer_lat : float
        Observer latitude in radians.
    observer_lon : float
        Observer longitude in radians.
    observer_alt : float
        Observer altitude in meters.
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid. Default is WGS84.

    Returns
    -------
    np.ndarray
        Target position in ECEF [x, y, z] in meters.

    Examples
    --------
    >>> import numpy as np
    >>> # Target at azimuth 90° (East), elevation 45°, range 1000m
    >>> obs_lat, obs_lon, obs_alt = 0.0, 0.0, 0.0
    >>> ecef = aer_to_ecef(np.pi/2, np.pi/4, 1000.0, obs_lat, obs_lon, obs_alt)
    >>> ecef.shape
    (3,)

    """
    # Convert AER to ENU
    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)

    # ENU components
    east = range_ * cos_el * sin_az
    north = range_ * cos_el * cos_az
    up = range_ * sin_el

    pos_enu = np.array([east, north, up])

    # Create ENU frame and transform to ECEF
    enu = ENUFrame(name="temp", latitude=observer_lat, longitude=observer_lon,
                   altitude=observer_alt, ellipsoid=ellipsoid)
    pos_ecef, _ = enu.local_to_ecef(pos_enu)

    return pos_ecef
