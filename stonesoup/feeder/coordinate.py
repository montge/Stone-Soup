"""Coordinate System Conversion Feeders

Feeders for converting between different coordinate systems using native
Stone Soup implementations (without external dependencies like pymap3d).

These converters transform state vectors between:
- ECEF (Earth-Centered Earth-Fixed) Cartesian coordinates
- Geodetic (latitude, longitude, altitude) coordinates
- ECI (Earth-Centered Inertial) coordinates
"""

import numpy as np

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..functions.coordinates import (
    ecef_to_eci,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
)
from ..types.coordinates import WGS84, ReferenceEllipsoid
from .base import DetectionFeeder, GroundTruthFeeder


class GeodeticToECEFConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts geodetic coordinates to ECEF (Earth-Centered Earth-Fixed).

    This feeder transforms state vectors from geodetic coordinates
    (latitude, longitude, altitude) to ECEF Cartesian coordinates (x, y, z).

    The geodetic coordinates are expected in:
    - Latitude: radians (positive North)
    - Longitude: radians (positive East)
    - Altitude: meters above the reference ellipsoid

    The output ECEF coordinates are in meters.

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with geodetic coordinates
    mapping : tuple of int
        Indices of (latitude, longitude, altitude) in the state vector.
        Default is (0, 1, 2).
    ellipsoid : ReferenceEllipsoid
        Reference ellipsoid for the transformation. Default is WGS84.

    Example
    -------
    >>> from stonesoup.feeder.coordinate import GeodeticToECEFConverter
    >>> converter = GeodeticToECEFConverter(
    ...     reader=geodetic_reader,
    ...     mapping=(0, 1, 2)  # lat, lon, alt indices
    ... )
    >>> for time, detections in converter:
    ...     # detections now have ECEF coordinates
    ...     pass

    Notes
    -----
    This uses a native implementation of the geodetic to ECEF transformation
    based on the standard equations, without requiring pymap3d.

    See Also
    --------
    :class:`ECEFToGeodeticConverter` : Inverse transformation
    :func:`~stonesoup.functions.coordinates.geodetic_to_ecef` : Underlying function

    """

    mapping: tuple = Property(
        default=(0, 1, 2),
        doc="Indices of (latitude, longitude, altitude) in the state vector. "
        "Default (0, 1, 2). Latitude and longitude should be in radians, "
        "altitude in meters.",
    )
    ellipsoid: ReferenceEllipsoid = Property(
        default=WGS84, doc="Reference ellipsoid for the transformation. Default is WGS84."
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                lat = state.state_vector[self.mapping[0], 0]
                lon = state.state_vector[self.mapping[1], 0]
                alt = state.state_vector[self.mapping[2], 0]

                ecef = geodetic_to_ecef(lat, lon, alt, self.ellipsoid)

                state.state_vector[self.mapping[0], 0] = ecef[0]
                state.state_vector[self.mapping[1], 0] = ecef[1]
                state.state_vector[self.mapping[2], 0] = ecef[2]

            yield time, states


class ECEFToGeodeticConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts ECEF (Earth-Centered Earth-Fixed) coordinates to geodetic.

    This feeder transforms state vectors from ECEF Cartesian coordinates
    (x, y, z) to geodetic coordinates (latitude, longitude, altitude).

    The ECEF coordinates are expected in meters.

    The output geodetic coordinates are:
    - Latitude: radians (positive North, range -π/2 to π/2)
    - Longitude: radians (positive East, range -π to π)
    - Altitude: meters above the reference ellipsoid

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with ECEF coordinates
    mapping : tuple of int
        Indices of (x, y, z) in the state vector. Default is (0, 1, 2).
    ellipsoid : ReferenceEllipsoid
        Reference ellipsoid for the transformation. Default is WGS84.

    Example
    -------
    >>> from stonesoup.feeder.coordinate import ECEFToGeodeticConverter
    >>> converter = ECEFToGeodeticConverter(
    ...     reader=ecef_reader,
    ...     mapping=(0, 1, 2)  # x, y, z indices
    ... )
    >>> for time, detections in converter:
    ...     # detections now have geodetic coordinates (lat, lon, alt)
    ...     pass

    Notes
    -----
    This uses Bowring's iterative method for the ECEF to geodetic conversion,
    which converges rapidly for all points except those very close to Earth's
    center.

    See Also
    --------
    :class:`GeodeticToECEFConverter` : Inverse transformation
    :func:`~stonesoup.functions.coordinates.ecef_to_geodetic` : Underlying function

    """

    mapping: tuple = Property(
        default=(0, 1, 2),
        doc="Indices of (x, y, z) in the state vector. Default (0, 1, 2). "
        "All values should be in meters.",
    )
    ellipsoid: ReferenceEllipsoid = Property(
        default=WGS84, doc="Reference ellipsoid for the transformation. Default is WGS84."
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                x = state.state_vector[self.mapping[0], 0]
                y = state.state_vector[self.mapping[1], 0]
                z = state.state_vector[self.mapping[2], 0]

                lat, lon, alt = ecef_to_geodetic(x, y, z, self.ellipsoid)

                state.state_vector[self.mapping[0], 0] = lat
                state.state_vector[self.mapping[1], 0] = lon
                state.state_vector[self.mapping[2], 0] = alt

            yield time, states


class ECIToECEFConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts ECI (Earth-Centered Inertial) coordinates to ECEF.

    This feeder transforms state vectors from ECI coordinates to ECEF
    coordinates using Earth's rotation. The transformation accounts for
    Earth Rotation Angle (ERA) at the timestamp of each state.

    Both input and output coordinates are in meters.

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with ECI coordinates
    mapping : tuple of int
        Indices of (x, y, z) in the state vector. Default is (0, 1, 2).

    Example
    -------
    >>> from stonesoup.feeder.coordinate import ECIToECEFConverter
    >>> converter = ECIToECEFConverter(
    ...     reader=eci_reader,
    ...     mapping=(0, 1, 2)
    ... )
    >>> for time, detections in converter:
    ...     # detections now have ECEF coordinates
    ...     pass

    Notes
    -----
    This uses the IAU 2000 Earth Rotation Angle model. For highest precision
    applications requiring precession, nutation, and polar motion corrections,
    additional transformations may be needed.

    The timestamp from each state is used to compute the rotation angle.
    States without timestamps will raise an error.

    See Also
    --------
    :class:`ECEFToECIConverter` : Inverse transformation
    :func:`~stonesoup.functions.coordinates.eci_to_ecef` : Underlying function

    """

    mapping: tuple = Property(
        default=(0, 1, 2),
        doc="Indices of (x, y, z) in the state vector. Default (0, 1, 2). "
        "All values should be in meters.",
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                timestamp = state.timestamp
                if timestamp is None:
                    raise ValueError("ECI to ECEF conversion requires a timestamp on each state")

                eci_coords = np.array(
                    [
                        state.state_vector[self.mapping[0], 0],
                        state.state_vector[self.mapping[1], 0],
                        state.state_vector[self.mapping[2], 0],
                    ]
                )

                ecef_coords = eci_to_ecef(eci_coords, timestamp)

                state.state_vector[self.mapping[0], 0] = ecef_coords[0]
                state.state_vector[self.mapping[1], 0] = ecef_coords[1]
                state.state_vector[self.mapping[2], 0] = ecef_coords[2]

            yield time, states


class ECEFToECIConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts ECEF coordinates to ECI (Earth-Centered Inertial).

    This feeder transforms state vectors from ECEF coordinates to ECI
    coordinates using Earth's rotation. The transformation accounts for
    Earth Rotation Angle (ERA) at the timestamp of each state.

    Both input and output coordinates are in meters.

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with ECEF coordinates
    mapping : tuple of int
        Indices of (x, y, z) in the state vector. Default is (0, 1, 2).

    Example
    -------
    >>> from stonesoup.feeder.coordinate import ECEFToECIConverter
    >>> converter = ECEFToECIConverter(
    ...     reader=ecef_reader,
    ...     mapping=(0, 1, 2)
    ... )
    >>> for time, detections in converter:
    ...     # detections now have ECI coordinates
    ...     pass

    Notes
    -----
    This uses the IAU 2000 Earth Rotation Angle model. The timestamp from
    each state is used to compute the rotation angle.

    See Also
    --------
    :class:`ECIToECEFConverter` : Inverse transformation
    :func:`~stonesoup.functions.coordinates.ecef_to_eci` : Underlying function

    """

    mapping: tuple = Property(
        default=(0, 1, 2),
        doc="Indices of (x, y, z) in the state vector. Default (0, 1, 2). "
        "All values should be in meters.",
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                timestamp = state.timestamp
                if timestamp is None:
                    raise ValueError("ECEF to ECI conversion requires a timestamp on each state")

                ecef_coords = np.array(
                    [
                        state.state_vector[self.mapping[0], 0],
                        state.state_vector[self.mapping[1], 0],
                        state.state_vector[self.mapping[2], 0],
                    ]
                )

                eci_coords = ecef_to_eci(ecef_coords, timestamp)

                state.state_vector[self.mapping[0], 0] = eci_coords[0]
                state.state_vector[self.mapping[1], 0] = eci_coords[1]
                state.state_vector[self.mapping[2], 0] = eci_coords[2]

            yield time, states


class GeodeticToECIConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts geodetic coordinates directly to ECI coordinates.

    This is a convenience feeder that combines geodetic to ECEF and ECEF to
    ECI transformations in a single step.

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with geodetic coordinates
    mapping : tuple of int
        Indices of (latitude, longitude, altitude) in the state vector.
        Default is (0, 1, 2).
    ellipsoid : ReferenceEllipsoid
        Reference ellipsoid for the transformation. Default is WGS84.

    Example
    -------
    >>> from stonesoup.feeder.coordinate import GeodeticToECIConverter
    >>> converter = GeodeticToECIConverter(
    ...     reader=geodetic_reader,
    ...     mapping=(0, 1, 2)
    ... )
    >>> for time, detections in converter:
    ...     # detections now have ECI coordinates
    ...     pass

    Notes
    -----
    Latitude and longitude should be in radians, altitude in meters.
    The timestamp from each state is used for the ECI transformation.

    """

    mapping: tuple = Property(
        default=(0, 1, 2),
        doc="Indices of (latitude, longitude, altitude) in the state vector. "
        "Default (0, 1, 2).",
    )
    ellipsoid: ReferenceEllipsoid = Property(
        default=WGS84, doc="Reference ellipsoid for the transformation. Default is WGS84."
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                timestamp = state.timestamp
                if timestamp is None:
                    raise ValueError(
                        "Geodetic to ECI conversion requires a timestamp on each state"
                    )

                lat = state.state_vector[self.mapping[0], 0]
                lon = state.state_vector[self.mapping[1], 0]
                alt = state.state_vector[self.mapping[2], 0]

                # First convert to ECEF
                ecef = geodetic_to_ecef(lat, lon, alt, self.ellipsoid)

                # Then convert to ECI
                eci_coords = ecef_to_eci(ecef, timestamp)

                state.state_vector[self.mapping[0], 0] = eci_coords[0]
                state.state_vector[self.mapping[1], 0] = eci_coords[1]
                state.state_vector[self.mapping[2], 0] = eci_coords[2]

            yield time, states


class ECIToGeodeticConverter(DetectionFeeder, GroundTruthFeeder):
    """Converts ECI coordinates directly to geodetic coordinates.

    This is a convenience feeder that combines ECI to ECEF and ECEF to
    geodetic transformations in a single step.

    Parameters
    ----------
    reader : Reader
        Source of detections/ground truth with ECI coordinates
    mapping : tuple of int
        Indices of (x, y, z) in the state vector. Default is (0, 1, 2).
    ellipsoid : ReferenceEllipsoid
        Reference ellipsoid for the transformation. Default is WGS84.

    Example
    -------
    >>> from stonesoup.feeder.coordinate import ECIToGeodeticConverter
    >>> converter = ECIToGeodeticConverter(
    ...     reader=eci_reader,
    ...     mapping=(0, 1, 2)
    ... )
    >>> for time, detections in converter:
    ...     # detections now have geodetic coordinates (lat, lon, alt)
    ...     pass

    Notes
    -----
    Output latitude and longitude are in radians, altitude in meters.
    The timestamp from each state is used for the ECI transformation.

    """

    mapping: tuple = Property(
        default=(0, 1, 2), doc="Indices of (x, y, z) in the state vector. Default (0, 1, 2)."
    )
    ellipsoid: ReferenceEllipsoid = Property(
        default=WGS84, doc="Reference ellipsoid for the transformation. Default is WGS84."
    )

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, states in self.reader:
            for state in states:
                timestamp = state.timestamp
                if timestamp is None:
                    raise ValueError(
                        "ECI to geodetic conversion requires a timestamp on each state"
                    )

                eci_coords = np.array(
                    [
                        state.state_vector[self.mapping[0], 0],
                        state.state_vector[self.mapping[1], 0],
                        state.state_vector[self.mapping[2], 0],
                    ]
                )

                # First convert to ECEF
                ecef_coords = eci_to_ecef(eci_coords, timestamp)

                # Then convert to geodetic
                lat, lon, alt = ecef_to_geodetic(
                    ecef_coords[0], ecef_coords[1], ecef_coords[2], self.ellipsoid
                )

                state.state_vector[self.mapping[0], 0] = lat
                state.state_vector[self.mapping[1], 0] = lon
                state.state_vector[self.mapping[2], 0] = alt

            yield time, states
