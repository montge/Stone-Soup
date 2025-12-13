"""Underwater/sonar measurement models.

This module provides measurement models for underwater target tracking using
sonar and acoustic sensors.
"""

import numpy as np

from ...base import Property
from ...functions.underwater import (
    bearing_elevation_range2cart,
    cart2bearing_elevation_range,
    cart2depth_bearing_range,
    depth_bearing_range2cart,
)
from ...types.angle import Bearing, Elevation
from ...types.array import StateVector
from ..base import ReversibleModel
from .nonlinear import NonLinearGaussianMeasurement


class CartesianToDepthBearingRange(NonLinearGaussianMeasurement, ReversibleModel):
    """Measurement model for sonar measuring depth, bearing, and slant range.

    This model converts Cartesian state to underwater sonar measurements:
    - Depth: Positive downward from surface
    - Bearing: Azimuth from north, clockwise positive
    - Slant Range: 3D distance from sensor to target

    The state is assumed to be in ENU (East-North-Up) coordinates.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="3D offset of sensor from platform center. Defaults to zero offset.",
    )
    rotation_offset: StateVector = Property(
        default=None,
        doc="3D rotation offset (roll, pitch, yaw) of sensor. Defaults to zero.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.translation_offset is None:
            self.translation_offset = StateVector([0, 0, 0])
        if self.rotation_offset is None:
            self.rotation_offset = StateVector([0, 0, 0])

    @property
    def ndim_meas(self) -> int:
        """Number of measurement dimensions (depth, bearing, range)."""
        return 3

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Model function mapping state to measurement.

        Parameters
        ----------
        state : State
            State vector in Cartesian coordinates
        noise : bool or array_like
            If True, add noise. If array, use as noise vector.

        Returns
        -------
        StateVector
            Measurement vector [depth, bearing, slant_range]
        """
        # Extract position from state using mapping
        if isinstance(state.state_vector, StateVector):
            pos = state.state_vector[self.mapping, :]
        else:
            pos = np.array(state.state_vector)[self.mapping, :]

        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]

        # Apply sensor offset
        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        depth, bearing, slant_range = cart2depth_bearing_range(x, y, z)

        # Wrap bearing
        bearing = Bearing(bearing)

        meas = StateVector([[depth], [bearing], [slant_range]])

        if noise is True:
            meas = meas + self.rvs()
        elif noise is not False:
            meas = meas + noise

        return meas

    def jacobian(self, state, **kwargs):
        """Calculate Jacobian of measurement function.

        Parameters
        ----------
        state : State
            State at which to calculate Jacobian

        Returns
        -------
        numpy.ndarray
            Jacobian matrix (3 x ndim_state)
        """
        pos = state.state_vector[self.mapping, :]
        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]

        # Apply offset
        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        r_horiz = np.sqrt(x**2 + y**2)
        r_slant = np.sqrt(x**2 + y**2 + z**2)

        # Avoid division by zero
        if r_horiz < 1e-10:
            r_horiz = 1e-10
        if r_slant < 1e-10:
            r_slant = 1e-10

        # Jacobian: d[depth, bearing, range]/d[x, y, z]
        # depth = -z
        # bearing = atan2(x, y)
        # range = sqrt(x^2 + y^2 + z^2)

        jac = np.zeros((3, self.ndim_state))

        # d(depth)/d(x,y,z) = [0, 0, -1]
        jac[0, self.mapping[2]] = -1.0

        # d(bearing)/d(x,y,z) = [y/(x^2+y^2), -x/(x^2+y^2), 0]
        jac[1, self.mapping[0]] = y / r_horiz**2
        jac[1, self.mapping[1]] = -x / r_horiz**2

        # d(range)/d(x,y,z) = [x/r, y/r, z/r]
        jac[2, self.mapping[0]] = x / r_slant
        jac[2, self.mapping[1]] = y / r_slant
        jac[2, self.mapping[2]] = z / r_slant

        return jac

    def inverse_function(self, detection, **kwargs) -> StateVector:
        """Map measurement back to state space.

        Parameters
        ----------
        detection : Detection
            Detection with [depth, bearing, slant_range]

        Returns
        -------
        StateVector
            Partial state vector with position filled in
        """
        depth = detection.state_vector[0, 0]
        bearing = detection.state_vector[1, 0]
        slant_range = detection.state_vector[2, 0]

        x, y, z = depth_bearing_range2cart(depth, bearing, slant_range)

        # Apply inverse offset
        x += self.translation_offset[0, 0]
        y += self.translation_offset[1, 0]
        z += self.translation_offset[2, 0]

        state_vector = StateVector(np.zeros((self.ndim_state, 1)))
        state_vector[self.mapping[0], 0] = x
        state_vector[self.mapping[1], 0] = y
        state_vector[self.mapping[2], 0] = z

        return state_vector


class CartesianToBearingOnly(NonLinearGaussianMeasurement, ReversibleModel):
    """Passive sonar bearing-only measurement model.

    This model is used for passive sonar that can only determine the
    direction to a target, not the range. Commonly used for submarine
    tracking.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="3D offset of sensor from platform center.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.translation_offset is None:
            self.translation_offset = StateVector([0, 0, 0])

    @property
    def ndim_meas(self) -> int:
        """Number of measurement dimensions (bearing only)."""
        return 1

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Model function for bearing-only measurement.

        Parameters
        ----------
        state : State
            State vector in Cartesian coordinates
        noise : bool or array_like
            If True, add noise.

        Returns
        -------
        StateVector
            Measurement vector [bearing]
        """
        pos = state.state_vector[self.mapping, :]
        x, y = pos[0, 0], pos[1, 0]

        # Apply sensor offset
        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]

        bearing = np.arctan2(x, y)
        bearing = Bearing(bearing)

        meas = StateVector([[bearing]])

        if noise is True:
            meas = meas + self.rvs()
        elif noise is not False:
            meas = meas + noise

        return meas

    def jacobian(self, state, **kwargs):
        """Calculate Jacobian for bearing-only measurement."""
        pos = state.state_vector[self.mapping, :]
        x, y = pos[0, 0], pos[1, 0]

        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]

        r_sq = x**2 + y**2
        if r_sq < 1e-10:
            r_sq = 1e-10

        jac = np.zeros((1, self.ndim_state))
        jac[0, self.mapping[0]] = y / r_sq
        jac[0, self.mapping[1]] = -x / r_sq

        return jac

    def inverse_function(self, detection, **kwargs) -> StateVector:
        """Cannot fully reconstruct state from bearing only.

        Returns a unit vector in the bearing direction.
        """
        bearing = detection.state_vector[0, 0]

        # Return unit vector at bearing (range = 1)
        x = np.sin(bearing) + self.translation_offset[0, 0]
        y = np.cos(bearing) + self.translation_offset[1, 0]

        state_vector = StateVector(np.zeros((self.ndim_state, 1)))
        state_vector[self.mapping[0], 0] = x
        state_vector[self.mapping[1], 0] = y

        return state_vector


class CartesianToBearingElevationRange(NonLinearGaussianMeasurement, ReversibleModel):
    """3D sonar measurement model with bearing, elevation, and range.

    Used for active sonar systems that can measure azimuth, elevation
    (depression angle), and slant range to target.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="3D offset of sensor from platform center.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.translation_offset is None:
            self.translation_offset = StateVector([0, 0, 0])

    @property
    def ndim_meas(self) -> int:
        """Number of measurement dimensions (bearing, elevation, range)."""
        return 3

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Model function for 3D sonar measurement.

        Parameters
        ----------
        state : State
            State vector in Cartesian coordinates
        noise : bool or array_like
            If True, add noise.

        Returns
        -------
        StateVector
            Measurement vector [bearing, elevation, slant_range]
        """
        pos = state.state_vector[self.mapping, :]
        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]

        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        bearing, elevation, slant_range = cart2bearing_elevation_range(x, y, z)

        bearing = Bearing(bearing)
        elevation = Elevation(elevation)

        meas = StateVector([[bearing], [elevation], [slant_range]])

        if noise is True:
            meas = meas + self.rvs()
        elif noise is not False:
            meas = meas + noise

        return meas

    def jacobian(self, state, **kwargs):
        """Calculate Jacobian for 3D sonar measurement."""
        pos = state.state_vector[self.mapping, :]
        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]

        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        r_horiz = np.sqrt(x**2 + y**2)
        r_slant = np.sqrt(x**2 + y**2 + z**2)

        if r_horiz < 1e-10:
            r_horiz = 1e-10
        if r_slant < 1e-10:
            r_slant = 1e-10

        jac = np.zeros((3, self.ndim_state))

        # d(bearing)/d(x,y,z)
        jac[0, self.mapping[0]] = y / r_horiz**2
        jac[0, self.mapping[1]] = -x / r_horiz**2

        # d(elevation)/d(x,y,z)
        jac[1, self.mapping[0]] = -x * z / (r_horiz * r_slant**2)
        jac[1, self.mapping[1]] = -y * z / (r_horiz * r_slant**2)
        jac[1, self.mapping[2]] = r_horiz / r_slant**2

        # d(range)/d(x,y,z)
        jac[2, self.mapping[0]] = x / r_slant
        jac[2, self.mapping[1]] = y / r_slant
        jac[2, self.mapping[2]] = z / r_slant

        return jac

    def inverse_function(self, detection, **kwargs) -> StateVector:
        """Map measurement back to state space."""
        bearing = detection.state_vector[0, 0]
        elevation = detection.state_vector[1, 0]
        slant_range = detection.state_vector[2, 0]

        x, y, z = bearing_elevation_range2cart(bearing, elevation, slant_range)

        x += self.translation_offset[0, 0]
        y += self.translation_offset[1, 0]
        z += self.translation_offset[2, 0]

        state_vector = StateVector(np.zeros((self.ndim_state, 1)))
        state_vector[self.mapping[0], 0] = x
        state_vector[self.mapping[1], 0] = y
        state_vector[self.mapping[2], 0] = z

        return state_vector


class CartesianToBearingRangeDoppler(NonLinearGaussianMeasurement, ReversibleModel):
    """Active sonar with Doppler shift measurement.

    Measures bearing, range, and range-rate (Doppler) for moving targets.
    The velocity components must be included in the state vector.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="3D offset of sensor from platform center.",
    )
    velocity_mapping: tuple = Property(
        doc="Mapping from state vector to velocity components (vx, vy, vz).",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.translation_offset is None:
            self.translation_offset = StateVector([0, 0, 0])

    @property
    def ndim_meas(self) -> int:
        """Number of measurement dimensions (bearing, range, range_rate)."""
        return 3

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Model function for sonar with Doppler.

        Parameters
        ----------
        state : State
            State vector with position and velocity
        noise : bool or array_like
            If True, add noise.

        Returns
        -------
        StateVector
            Measurement vector [bearing, range, range_rate]
        """
        pos = state.state_vector[self.mapping, :]
        vel = state.state_vector[list(self.velocity_mapping), :]

        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]
        vx, vy, vz = vel[0, 0], vel[1, 0], vel[2, 0]

        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        r_slant = np.sqrt(x**2 + y**2 + z**2)
        if r_slant < 1e-10:
            r_slant = 1e-10

        bearing = np.arctan2(x, y)
        bearing = Bearing(bearing)

        # Range rate is velocity component along line of sight
        range_rate = (x * vx + y * vy + z * vz) / r_slant

        meas = StateVector([[bearing], [r_slant], [range_rate]])

        if noise is True:
            meas = meas + self.rvs()
        elif noise is not False:
            meas = meas + noise

        return meas

    def jacobian(self, state, **kwargs):
        """Calculate Jacobian for Doppler sonar measurement."""
        pos = state.state_vector[self.mapping, :]
        vel = state.state_vector[list(self.velocity_mapping), :]

        x, y, z = pos[0, 0], pos[1, 0], pos[2, 0]
        vx, vy, vz = vel[0, 0], vel[1, 0], vel[2, 0]

        x -= self.translation_offset[0, 0]
        y -= self.translation_offset[1, 0]
        z -= self.translation_offset[2, 0]

        r_horiz = np.sqrt(x**2 + y**2)
        r_slant = np.sqrt(x**2 + y**2 + z**2)

        if r_horiz < 1e-10:
            r_horiz = 1e-10
        if r_slant < 1e-10:
            r_slant = 1e-10

        jac = np.zeros((3, self.ndim_state))

        # d(bearing)/d(x,y)
        jac[0, self.mapping[0]] = y / r_horiz**2
        jac[0, self.mapping[1]] = -x / r_horiz**2

        # d(range)/d(x,y,z)
        jac[1, self.mapping[0]] = x / r_slant
        jac[1, self.mapping[1]] = y / r_slant
        jac[1, self.mapping[2]] = z / r_slant

        # d(range_rate)/d(x,y,z,vx,vy,vz)
        dot_product = x * vx + y * vy + z * vz
        jac[2, self.mapping[0]] = (vx * r_slant - x * dot_product / r_slant) / r_slant**2
        jac[2, self.mapping[1]] = (vy * r_slant - y * dot_product / r_slant) / r_slant**2
        jac[2, self.mapping[2]] = (vz * r_slant - z * dot_product / r_slant) / r_slant**2
        jac[2, self.velocity_mapping[0]] = x / r_slant
        jac[2, self.velocity_mapping[1]] = y / r_slant
        jac[2, self.velocity_mapping[2]] = z / r_slant

        return jac

    def inverse_function(self, detection, **kwargs) -> StateVector:
        """Map measurement back to state space (position only)."""
        bearing = detection.state_vector[0, 0]
        slant_range = detection.state_vector[1, 0]

        # Cannot determine elevation from 2D bearing + range
        # Assume horizontal (z=0)
        x = slant_range * np.sin(bearing) + self.translation_offset[0, 0]
        y = slant_range * np.cos(bearing) + self.translation_offset[1, 0]
        z = self.translation_offset[2, 0]

        state_vector = StateVector(np.zeros((self.ndim_state, 1)))
        state_vector[self.mapping[0], 0] = x
        state_vector[self.mapping[1], 0] = y
        state_vector[self.mapping[2], 0] = z

        return state_vector
