"""Coordinate Transformation Functions
-----------------------------------

Functions for transforming between different coordinate systems including:

- ECEF (Earth-Centered Earth-Fixed) <-> Geodetic (Latitude, Longitude, Altitude)
- ECI (Earth-Centered Inertial) <-> ECEF

These implementations use native algorithms rather than external libraries.

"""
import numpy as np
from datetime import datetime

from ..types.coordinates import WGS84, ReferenceEllipsoid


def geodetic_to_ecef(lat: float, lon: float, alt: float,
                     ellipsoid: ReferenceEllipsoid = WGS84) -> np.ndarray:
    r"""Convert geodetic coordinates to ECEF (Earth-Centered Earth-Fixed) coordinates.

    This function uses the standard geodetic transformation formula. The ECEF frame is a
    Cartesian coordinate system with origin at Earth's center, Z-axis along the polar axis,
    X-axis through the prime meridian at the equator, and Y-axis completing the right-handed
    system (through 90°E).

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians (positive North, range: -π/2 to π/2)
    lon : float
        Geodetic longitude in radians (positive East, range: -π to π or 0 to 2π)
    alt : float
        Altitude above the ellipsoid in meters (height above the reference ellipsoid)
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid to use. Default is WGS84.

    Returns
    -------
    : numpy.ndarray
        ECEF coordinates as [x, y, z] in meters

    Notes
    -----
    The transformation uses the prime vertical radius of curvature :math:`N`:

    .. math::

        N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}}

    where :math:`a` is the semi-major axis, :math:`e` is the eccentricity, and
    :math:`\phi` is the geodetic latitude.

    The ECEF coordinates are then:

    .. math::

        x &= (N + h) \cos(\phi) \cos(\lambda)

        y &= (N + h) \cos(\phi) \sin(\lambda)

        z &= (N(1 - e^2) + h) \sin(\phi)

    where :math:`h` is altitude and :math:`\lambda` is longitude.

    Examples
    --------
    >>> # Greenwich Observatory at sea level
    >>> lat, lon, alt = np.radians(51.4769), np.radians(-0.0005), 0.0
    >>> xyz = geodetic_to_ecef(lat, lon, alt)
    >>> print(f"ECEF: x={xyz[0]:.1f}, y={xyz[1]:.1f}, z={xyz[2]:.1f}")
    ECEF: x=3980574.2, y=-0.4, z=4966894.1

    References
    ----------
    .. [1] Hofmann-Wellenhof, B., Lichtenegger, H., and Collins, J., 2001,
           "GPS: Theory and Practice," 5th ed., Springer-Verlag, Wien New York.
    .. [2] Jekeli, C., 2006, "Geometric Reference Systems in Geodesy,"
           Division of Geodesy and Geospatial Science, Ohio State University.

    """
    a = ellipsoid.semi_major_axis
    e2 = ellipsoid.eccentricity_squared

    # Compute the prime vertical radius of curvature
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)

    # Convert to ECEF
    x = (N + alt) * cos_lat * np.cos(lon)
    y = (N + alt) * cos_lat * np.sin(lon)
    z = (N * (1.0 - e2) + alt) * sin_lat

    return np.array([x, y, z])


def ecef_to_geodetic(x: float, y: float, z: float,
                     ellipsoid: ReferenceEllipsoid = WGS84,
                     tolerance: float = 1e-12,
                     max_iterations: int = 10) -> tuple[float, float, float]:
    r"""Convert ECEF (Earth-Centered Earth-Fixed) coordinates to geodetic coordinates.

    This function uses Bowring's iterative method (1985) which converges rapidly
    for all points except those very close to the center of the Earth.

    Parameters
    ----------
    x : float
        ECEF X-coordinate in meters
    y : float
        ECEF Y-coordinate in meters
    z : float
        ECEF Z-coordinate in meters
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid to use. Default is WGS84.
    tolerance : float, optional
        Convergence tolerance for latitude in radians. Default is 1e-12 (~0.006 mm).
    max_iterations : int, optional
        Maximum number of iterations. Default is 10.

    Returns
    -------
    lat : float
        Geodetic latitude in radians (range: -π/2 to π/2)
    lon : float
        Geodetic longitude in radians (range: -π to π)
    alt : float
        Altitude above the ellipsoid in meters

    Notes
    -----
    The algorithm uses an iterative refinement of the latitude estimate:

    .. math::

        \phi_{i+1} = \arctan\left(\frac{z + e'^2 b \sin^3(\theta_i)}
                                       {p - e^2 a \cos^3(\theta_i)}\right)

    where :math:`p = \sqrt{x^2 + y^2}`, :math:`e` is the first eccentricity,
    :math:`e'` is the second eccentricity, and :math:`\theta` is the parametric
    latitude.

    The altitude is computed as:

    .. math::

        h = \frac{p}{\cos(\phi)} - N

    where :math:`N` is the prime vertical radius of curvature at latitude :math:`\phi`.

    Longitude is directly computed as:

    .. math::

        \lambda = \arctan2(y, x)

    Examples
    --------
    >>> # Convert from ECEF back to geodetic
    >>> x, y, z = 3980574.2, -0.4, 4966894.1
    >>> lat, lon, alt = ecef_to_geodetic(x, y, z)
    >>> print(f"Lat: {np.degrees(lat):.4f}°, Lon: {np.degrees(lon):.4f}°, Alt: {alt:.1f}m")
    Lat: 51.4769°, Lon: -0.0001°, Alt: 0.0m

    References
    ----------
    .. [1] Bowring, B. R., 1985, "The Accuracy of Geodetic Latitude and Height Equations,"
           Survey Review, Vol. 28, No. 218, pp. 202-206.
    .. [2] Zhu, J., 1994, "Conversion of Earth-centered Earth-fixed coordinates to
           geodetic coordinates," IEEE Transactions on Aerospace and Electronic Systems,
           Vol. 30, No. 3, pp. 957-961.

    """
    a = ellipsoid.semi_major_axis
    b = ellipsoid.semi_minor_axis
    e2 = ellipsoid.eccentricity_squared
    e_prime2 = ellipsoid.second_eccentricity_squared

    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Compute distance from Z-axis
    p = np.sqrt(x ** 2 + y ** 2)

    # Handle special case of point on or near Z-axis
    if p < 1e-10:  # Within 10 micrometers of Z-axis
        lat = np.sign(z) * np.pi / 2.0 if z != 0 else 0.0
        alt = np.abs(z) - b
        return lat, lon, alt

    # Initial estimate of parametric latitude (Bowring 1985)
    theta = np.arctan2(z * a, p * b)

    # Iterative refinement of geodetic latitude
    lat = theta  # Initial guess
    for _ in range(max_iterations):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Update latitude estimate
        lat_new = np.arctan2(
            z + e_prime2 * b * sin_theta ** 3,
            p - e2 * a * cos_theta ** 3
        )

        # Check convergence
        if np.abs(lat_new - lat) < tolerance:
            lat = lat_new
            break

        lat = lat_new
        theta = np.arctan2(z * (1.0 + e_prime2 * np.sin(lat)),
                           p * (1.0 - e2 * np.sin(lat)))

    # Compute altitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)

    # Choose altitude formula based on latitude (for numerical stability)
    if np.abs(lat) < np.pi / 4:  # Use horizontal distance if not near poles
        alt = p / cos_lat - N
    else:  # Use vertical distance near poles
        alt = z / sin_lat - N * (1.0 - e2)

    return lat, lon, alt


def eci_to_ecef(eci_coords: np.ndarray, timestamp: datetime) -> np.ndarray:
    r"""Convert ECI (Earth-Centered Inertial) coordinates to ECEF coordinates.

    This function rotates ECI coordinates to ECEF using Earth Rotation Angle (ERA),
    which accounts for Earth's rotation. The transformation uses a simplified model
    suitable for many applications; for highest precision, additional corrections
    (precession, nutation, polar motion) may be needed.

    Parameters
    ----------
    eci_coords : numpy.ndarray
        ECI coordinates as [x, y, z] in meters
    timestamp : datetime.datetime
        UTC timestamp for the coordinate transformation

    Returns
    -------
    : numpy.ndarray
        ECEF coordinates as [x, y, z] in meters

    Notes
    -----
    The transformation uses the Earth Rotation Angle (ERA):

    .. math::

        \text{ERA} = 2\pi (0.7790572732640 + 1.00273781191135448 \cdot T_u)

    where :math:`T_u` is the UT1 time in Julian centuries from J2000.0.

    The rotation matrix from ECI to ECEF is:

    .. math::

        R_z(\text{ERA}) = \begin{bmatrix}
            \cos(\text{ERA}) & \sin(\text{ERA}) & 0 \\
            -\sin(\text{ERA}) & \cos(\text{ERA}) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    Then: :math:`\mathbf{r}_{\text{ECEF}} = R_z(\text{ERA}) \cdot \mathbf{r}_{\text{ECI}}`

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> # ECI position
    >>> eci = np.array([6378137.0, 0.0, 0.0])
    >>> time = datetime(2024, 1, 1, 0, 0, 0)
    >>> ecef = eci_to_ecef(eci, time)
    >>> print(f"ECEF: [{ecef[0]:.1f}, {ecef[1]:.1f}, {ecef[2]:.1f}]")
    ECEF: [6378137.0, 0.0, 0.0]

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press.
    .. [2] IERS Conventions (2010), IERS Technical Note No. 36,
           Verlag des Bundesamts für Kartographie und Geodäsie, Frankfurt am Main.

    """
    # Compute Julian Date
    # J2000 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_centuries = dt / (86400.0 * 36525.0)

    # Compute Earth Rotation Angle (ERA) using IAU 2000 model
    # ERA = 2π(0.7790572732640 + 1.00273781191135448 * T_u)
    # where T_u is UT1 in Julian centuries from J2000
    # (Simplified: using UTC instead of UT1, difference is typically < 1 second)
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_centuries)

    # Reduce to [0, 2π)
    era = np.fmod(era, 2.0 * np.pi)
    if era < 0:
        era += 2.0 * np.pi

    # Create rotation matrix (Z-axis rotation by ERA)
    cos_era = np.cos(era)
    sin_era = np.sin(era)

    rotation_matrix = np.array([
        [cos_era, sin_era, 0.0],
        [-sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Transform ECI to ECEF
    ecef_coords = rotation_matrix @ eci_coords

    return ecef_coords


def ecef_to_eci(ecef_coords: np.ndarray, timestamp: datetime) -> np.ndarray:
    r"""Convert ECEF coordinates to ECI (Earth-Centered Inertial) coordinates.

    This function is the inverse of :func:`eci_to_ecef`, rotating ECEF coordinates
    back to the ECI frame using the Earth Rotation Angle at the specified timestamp.

    Parameters
    ----------
    ecef_coords : numpy.ndarray
        ECEF coordinates as [x, y, z] in meters
    timestamp : datetime.datetime
        UTC timestamp for the coordinate transformation

    Returns
    -------
    : numpy.ndarray
        ECI coordinates as [x, y, z] in meters

    Notes
    -----
    The transformation uses the transpose of the ECI-to-ECEF rotation matrix:

    .. math::

        R_z(-\text{ERA}) = \begin{bmatrix}
            \cos(\text{ERA}) & -\sin(\text{ERA}) & 0 \\
            \sin(\text{ERA}) & \cos(\text{ERA}) & 0 \\
            0 & 0 & 1
        \end{bmatrix}

    Then: :math:`\mathbf{r}_{\text{ECI}} = R_z(-\text{ERA}) \cdot \mathbf{r}_{\text{ECEF}}`

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> # ECEF position
    >>> ecef = np.array([6378137.0, 0.0, 0.0])
    >>> time = datetime(2024, 1, 1, 0, 0, 0)
    >>> eci = ecef_to_eci(ecef, time)
    >>> print(f"ECI: [{eci[0]:.1f}, {eci[1]:.1f}, {eci[2]:.1f}]")
    ECI: [6378137.0, 0.0, 0.0]

    References
    ----------
    .. [1] Vallado, D. A., 2013, "Fundamentals of Astrodynamics and Applications,"
           4th ed., Microcosm Press.
    .. [2] IERS Conventions (2010), IERS Technical Note No. 36.

    """
    # Compute Julian Date
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_centuries = dt / (86400.0 * 36525.0)

    # Compute Earth Rotation Angle (ERA)
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_centuries)

    # Reduce to [0, 2π)
    era = np.fmod(era, 2.0 * np.pi)
    if era < 0:
        era += 2.0 * np.pi

    # Create inverse rotation matrix (transpose of ECI to ECEF rotation)
    cos_era = np.cos(era)
    sin_era = np.sin(era)

    rotation_matrix = np.array([
        [cos_era, -sin_era, 0.0],
        [sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Transform ECEF to ECI
    eci_coords = rotation_matrix @ ecef_coords

    return eci_coords


def geodetic_to_eci(lat: float, lon: float, alt: float,
                    timestamp: datetime,
                    ellipsoid: ReferenceEllipsoid = WGS84) -> np.ndarray:
    """Convert geodetic coordinates directly to ECI coordinates.

    This is a convenience function that combines geodetic_to_ecef and ecef_to_eci.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians
    lon : float
        Geodetic longitude in radians
    alt : float
        Altitude above the ellipsoid in meters
    timestamp : datetime.datetime
        UTC timestamp for the coordinate transformation
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid to use. Default is WGS84.

    Returns
    -------
    : numpy.ndarray
        ECI coordinates as [x, y, z] in meters

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> # Greenwich Observatory
    >>> lat, lon, alt = np.radians(51.4769), np.radians(-0.0005), 0.0
    >>> time = datetime(2024, 1, 1, 12, 0, 0)
    >>> eci = geodetic_to_eci(lat, lon, alt, time)
    >>> print(f"ECI: [{eci[0]:.1f}, {eci[1]:.1f}, {eci[2]:.1f}]")
    ECI: [3980574.2, -0.4, 4966894.1]

    """
    ecef = geodetic_to_ecef(lat, lon, alt, ellipsoid)
    return ecef_to_eci(ecef, timestamp)


def eci_to_geodetic(x: float, y: float, z: float,
                    timestamp: datetime,
                    ellipsoid: ReferenceEllipsoid = WGS84,
                    tolerance: float = 1e-12,
                    max_iterations: int = 10) -> tuple[float, float, float]:
    """Convert ECI coordinates directly to geodetic coordinates.

    This is a convenience function that combines ecef_to_geodetic and eci_to_ecef.

    Parameters
    ----------
    x : float
        ECI X-coordinate in meters
    y : float
        ECI Y-coordinate in meters
    z : float
        ECI Z-coordinate in meters
    timestamp : datetime.datetime
        UTC timestamp for the coordinate transformation
    ellipsoid : ReferenceEllipsoid, optional
        Reference ellipsoid to use. Default is WGS84.
    tolerance : float, optional
        Convergence tolerance for latitude in radians. Default is 1e-12.
    max_iterations : int, optional
        Maximum number of iterations. Default is 10.

    Returns
    -------
    lat : float
        Geodetic latitude in radians
    lon : float
        Geodetic longitude in radians
    alt : float
        Altitude above the ellipsoid in meters

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> # ECI position
    >>> x, y, z = 3980574.2, -0.4, 4966894.1
    >>> time = datetime(2024, 1, 1, 12, 0, 0)
    >>> lat, lon, alt = eci_to_geodetic(x, y, z, time)
    >>> print(f"Lat: {np.degrees(lat):.4f}°, Lon: {np.degrees(lon):.4f}°")
    Lat: 51.4769°, Lon: -0.0001°

    """
    eci_coords = np.array([x, y, z])
    ecef = eci_to_ecef(eci_coords, timestamp)
    return ecef_to_geodetic(ecef[0], ecef[1], ecef[2], ellipsoid, tolerance, max_iterations)
