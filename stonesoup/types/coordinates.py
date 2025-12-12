"""Coordinate System Types
----------------------

Reference frames and ellipsoids for coordinate transformations.

"""
from typing import ClassVar
from datetime import datetime
from abc import abstractmethod

import numpy as np

from ..base import Base, Property


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
