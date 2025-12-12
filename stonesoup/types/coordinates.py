"""Coordinate System Types
----------------------

Reference frames and ellipsoids for coordinate transformations.

"""
from typing import ClassVar

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
