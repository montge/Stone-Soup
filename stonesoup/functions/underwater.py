"""Underwater/undersea tracking utility functions.

This module provides functions for underwater acoustics, coordinate transformations,
and environmental modeling for undersea target tracking.
"""

import numpy as np


def sound_speed_mackenzie(temperature, salinity, depth):
    """Calculate sound speed in seawater using Mackenzie equation.

    The Mackenzie equation (1981) is accurate for typical ocean conditions:
    - Temperature: -2 to 30 degrees C
    - Salinity: 25 to 40 PSU
    - Depth: 0 to 8000 m

    Parameters
    ----------
    temperature : float
        Water temperature in degrees Celsius
    salinity : float
        Salinity in PSU (Practical Salinity Units), typically ~35 for seawater
    depth : float
        Depth in meters (positive downward)

    Returns
    -------
    float
        Speed of sound in m/s

    References
    ----------
    Mackenzie, K.V. (1981). Nine-term equation for sound speed in the oceans.
    Journal of the Acoustical Society of America, 70(3), 807-812.
    """
    T = temperature
    S = salinity
    D = depth

    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D**3
    )

    return c


def sound_speed_unesco(temperature, salinity, pressure):
    """Calculate sound speed using UNESCO algorithm (Chen & Millero, 1977).

    More accurate than Mackenzie for extreme conditions.

    Parameters
    ----------
    temperature : float
        Water temperature in degrees Celsius
    salinity : float
        Salinity in PSU
    pressure : float
        Pressure in decibars (approximately depth in meters)

    Returns
    -------
    float
        Speed of sound in m/s
    """
    T = temperature
    S = salinity
    P = pressure / 10.0  # Convert to bars

    # Chen-Millero coefficients
    C00 = 1402.388
    C01 = 5.03830
    C02 = -5.81090e-2
    C03 = 3.3432e-4
    C04 = -1.47797e-6
    C05 = 3.1419e-9
    C10 = 0.153563
    C11 = 6.8999e-4
    C12 = -8.1829e-6
    C13 = 1.3632e-7
    C14 = -6.1260e-10
    C20 = 3.1260e-5
    C21 = -1.7111e-6
    C22 = 2.5986e-8
    C23 = -2.5353e-10
    C24 = 1.0415e-12
    C30 = -9.7729e-9
    C31 = 3.8513e-10
    C32 = -2.3654e-12

    A00 = 1.389
    A01 = -1.262e-2
    A02 = 7.166e-5
    A03 = 2.008e-6
    A04 = -3.21e-8
    A10 = 9.4742e-5
    A11 = -1.2583e-5
    A12 = -6.4928e-8
    A13 = 1.0515e-8
    A14 = -2.0142e-10
    A20 = -3.9064e-7
    A21 = 9.1061e-9
    A22 = -1.6009e-10
    A23 = 7.994e-12
    A30 = 1.100e-10
    A31 = 6.651e-12
    A32 = -3.391e-13

    B00 = -1.922e-2
    B01 = -4.42e-5
    B10 = 7.3637e-5
    B11 = 1.7950e-7

    D00 = 1.727e-3
    D10 = -7.9836e-6

    # Pure water sound speed
    Cw = (
        C00
        + C01 * T
        + C02 * T**2
        + C03 * T**3
        + C04 * T**4
        + C05 * T**5
        + (C10 + C11 * T + C12 * T**2 + C13 * T**3 + C14 * T**4) * P
        + (C20 + C21 * T + C22 * T**2 + C23 * T**3 + C24 * T**4) * P**2
        + (C30 + C31 * T + C32 * T**2) * P**3
    )

    # Salinity and pressure corrections
    A = (
        A00
        + A01 * T
        + A02 * T**2
        + A03 * T**3
        + A04 * T**4
        + (A10 + A11 * T + A12 * T**2 + A13 * T**3 + A14 * T**4) * P
        + (A20 + A21 * T + A22 * T**2 + A23 * T**3) * P**2
        + (A30 + A31 * T + A32 * T**2) * P**3
    )

    B = B00 + B01 * T + (B10 + B11 * T) * P

    D = D00 + D10 * P

    c = Cw + A * S + B * S ** (3 / 2) + D * S**2

    return c


def pressure_to_depth(pressure, latitude=45.0):
    """Convert pressure to depth using Fofonoff & Millard (1983).

    Parameters
    ----------
    pressure : float
        Pressure in decibars
    latitude : float, optional
        Latitude in degrees (default 45)

    Returns
    -------
    float
        Depth in meters
    """
    # Gravity variation with latitude
    x = np.sin(np.radians(latitude)) ** 2

    # Coefficients from Fofonoff & Millard
    c1 = 9.72659
    c2 = -2.2512e-5
    c3 = 2.279e-10
    c4 = -1.82e-15

    g = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x)

    depth = ((((c4 * pressure + c3) * pressure + c2) * pressure + c1) * pressure) / g

    return depth


def depth_to_pressure(depth, latitude=45.0):
    """Convert depth to pressure using iterative inversion.

    Parameters
    ----------
    depth : float
        Depth in meters
    latitude : float, optional
        Latitude in degrees (default 45)

    Returns
    -------
    float
        Pressure in decibars
    """
    # Initial guess: pressure ≈ depth (for seawater ~1 dbar/m)
    pressure = depth

    # Newton-Raphson iteration
    for _ in range(10):
        d = pressure_to_depth(pressure, latitude)
        if abs(d - depth) < 1e-6:
            break
        # Approximate derivative (pressure increases faster than depth)
        pressure = pressure + (depth - d)

    return pressure


def acoustic_attenuation(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0):
    """Calculate acoustic attenuation coefficient using Francois-Garrison model.

    Parameters
    ----------
    frequency : float
        Acoustic frequency in kHz
    temperature : float, optional
        Temperature in degrees Celsius (default 10)
    salinity : float, optional
        Salinity in PSU (default 35)
    depth : float, optional
        Depth in meters (default 0)
    ph : float, optional
        pH value (default 8.0)

    Returns
    -------
    float
        Attenuation coefficient in dB/km

    References
    ----------
    Francois, R.E. and Garrison, G.R. (1982). Sound absorption based on ocean
    measurements. Part II: Boric acid contribution and equation for total
    absorption. Journal of the Acoustical Society of America, 72(6), 1879-1890.
    """
    f = frequency  # kHz
    T = temperature
    S = salinity
    D = depth / 1000.0  # Convert to km
    c = sound_speed_mackenzie(T, S, depth)

    # Boric acid relaxation
    A1 = 8.86 / c * 10 ** (0.78 * ph - 5)
    f1 = 2.8 * np.sqrt(S / 35) * 10 ** (4 - 1245 / (T + 273))

    # Magnesium sulfate relaxation
    A2 = 21.44 * S / c * (1 + 0.025 * T)
    f2 = 8.17 * 10 ** (8 - 1990 / (T + 273)) / (1 + 0.0018 * (S - 35))

    # Pure water viscosity
    A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    if T <= 20:
        A3 = A3 - 3.80e-4 * (1 - D) + 1.23e-4 * (1 - D) ** 2

    # Total attenuation
    alpha = A1 * f1 * f**2 / (f1**2 + f**2) + A2 * f2 * f**2 / (f2**2 + f**2) + A3 * f**2

    return alpha


def transmission_loss(
    distance, frequency, depth_source=10.0, depth_target=10.0, temperature=10.0, salinity=35.0
):
    """Calculate one-way transmission loss for underwater acoustics.

    Uses spherical spreading plus absorption.

    Parameters
    ----------
    distance : float
        Distance in meters
    frequency : float
        Frequency in kHz
    depth_source : float, optional
        Source depth in meters (default 10)
    depth_target : float, optional
        Target depth in meters (default 10)
    temperature : float, optional
        Temperature in degrees Celsius (default 10)
    salinity : float, optional
        Salinity in PSU (default 35)

    Returns
    -------
    float
        Transmission loss in dB
    """
    if distance <= 0:
        return 0.0

    # Average depth for attenuation calculation
    avg_depth = (depth_source + depth_target) / 2.0

    # Absorption coefficient (dB/km)
    alpha = acoustic_attenuation(frequency, temperature, salinity, avg_depth)

    # Spherical spreading loss + absorption
    # TL = 20*log10(r) + alpha*r
    distance_km = distance / 1000.0
    tl = 20 * np.log10(distance) + alpha * distance_km

    return tl


def cart2depth_bearing_range(x, y, z):
    """Convert Cartesian coordinates to depth-bearing-range.

    Uses ENU (East-North-Up) convention where:
    - x: East (positive eastward)
    - y: North (positive northward)
    - z: Up (positive upward, so depth = -z)

    Parameters
    ----------
    x : float or array_like
        East coordinate in meters
    y : float or array_like
        North coordinate in meters
    z : float or array_like
        Up coordinate in meters (negative for underwater)

    Returns
    -------
    tuple
        (depth, bearing, slant_range) where:
        - depth: Depth in meters (positive downward)
        - bearing: Bearing in radians (0=North, positive clockwise)
        - slant_range: 3D distance in meters
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Depth is negative of z (up is positive, depth is positive downward)
    depth = -z

    # Slant range (3D distance)
    slant_range = np.sqrt(x**2 + y**2 + z**2)

    # Bearing (azimuth from north, clockwise positive)
    bearing = np.arctan2(x, y)

    return depth, bearing, slant_range


def depth_bearing_range2cart(depth, bearing, slant_range):
    """Convert depth-bearing-range to Cartesian coordinates.

    Parameters
    ----------
    depth : float or array_like
        Depth in meters (positive downward)
    bearing : float or array_like
        Bearing in radians (0=North, positive clockwise)
    slant_range : float or array_like
        3D slant range in meters

    Returns
    -------
    tuple
        (x, y, z) in ENU coordinates
    """
    depth = np.asarray(depth)
    bearing = np.asarray(bearing)
    slant_range = np.asarray(slant_range)

    # z is negative of depth
    z = -depth

    # Horizontal range from slant range and depth
    horizontal_range_sq = slant_range**2 - depth**2
    horizontal_range = np.sqrt(np.maximum(horizontal_range_sq, 0))

    # Cartesian from bearing and horizontal range
    x = horizontal_range * np.sin(bearing)
    y = horizontal_range * np.cos(bearing)

    return x, y, z


def cart2bearing_elevation_range(x, y, z):
    """Convert Cartesian to bearing-elevation-range (for 3D sonar).

    Parameters
    ----------
    x : float or array_like
        East coordinate in meters
    y : float or array_like
        North coordinate in meters
    z : float or array_like
        Up coordinate in meters

    Returns
    -------
    tuple
        (bearing, elevation, slant_range) where:
        - bearing: Azimuth from north in radians
        - elevation: Elevation angle in radians (positive upward)
        - slant_range: 3D distance in meters
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    horizontal_range = np.sqrt(x**2 + y**2)
    slant_range = np.sqrt(x**2 + y**2 + z**2)

    bearing = np.arctan2(x, y)
    elevation = np.arctan2(z, horizontal_range)

    return bearing, elevation, slant_range


def bearing_elevation_range2cart(bearing, elevation, slant_range):
    """Convert bearing-elevation-range to Cartesian.

    Parameters
    ----------
    bearing : float or array_like
        Azimuth from north in radians
    elevation : float or array_like
        Elevation angle in radians
    slant_range : float or array_like
        3D distance in meters

    Returns
    -------
    tuple
        (x, y, z) in ENU coordinates
    """
    bearing = np.asarray(bearing)
    elevation = np.asarray(elevation)
    slant_range = np.asarray(slant_range)

    horizontal_range = slant_range * np.cos(elevation)
    z = slant_range * np.sin(elevation)
    x = horizontal_range * np.sin(bearing)
    y = horizontal_range * np.cos(bearing)

    return x, y, z


# =============================================================================
# Geodetic Integration Functions (WGS84)
# =============================================================================

# WGS84 ellipsoid parameters
WGS84_SEMI_MAJOR_AXIS = 6378137.0  # meters
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_SEMI_MINOR_AXIS = WGS84_SEMI_MAJOR_AXIS * (1 - WGS84_FLATTENING)
WGS84_ECCENTRICITY_SQUARED = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2


def geodetic_to_ecef_underwater(lat, lon, depth):
    """Convert geodetic coordinates with depth to ECEF.

    This is the underwater extension of standard geodetic-to-ECEF conversion.
    Depth is measured positive downward from the WGS84 ellipsoid surface.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians (positive north)
    lon : float
        Geodetic longitude in radians (positive east)
    depth : float
        Depth in meters (positive downward from ellipsoid surface)

    Returns
    -------
    tuple
        (x, y, z) ECEF coordinates in meters

    Notes
    -----
    Depth is converted to negative altitude for ECEF calculation:
        altitude = -depth
    """
    a = WGS84_SEMI_MAJOR_AXIS
    e2 = WGS84_ECCENTRICITY_SQUARED

    # Depth is negative altitude
    alt = -depth

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    # Prime vertical radius of curvature
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)

    x = (N + alt) * cos_lat * np.cos(lon)
    y = (N + alt) * cos_lat * np.sin(lon)
    z = (N * (1.0 - e2) + alt) * sin_lat

    return x, y, z


def ecef_to_geodetic_underwater(x, y, z, tolerance=1e-12, max_iterations=10):
    """Convert ECEF coordinates to geodetic with depth.

    Uses Bowring's iterative method for the lat/lon conversion.

    Parameters
    ----------
    x : float
        ECEF X-coordinate in meters
    y : float
        ECEF Y-coordinate in meters
    z : float
        ECEF Z-coordinate in meters
    tolerance : float, optional
        Convergence tolerance for latitude in radians
    max_iterations : int, optional
        Maximum number of iterations

    Returns
    -------
    tuple
        (lat, lon, depth) where:
        - lat: Geodetic latitude in radians
        - lon: Geodetic longitude in radians
        - depth: Depth in meters (positive downward)
    """
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_ECCENTRICITY_SQUARED
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared

    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Horizontal distance from z-axis
    p = np.sqrt(x**2 + y**2)

    # Initial latitude estimate using parametric latitude
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    # Iterate to improve latitude
    for _ in range(max_iterations):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat**2)

        lat_new = np.arctan2(z + e2 * N * sin_lat, p)

        if np.abs(lat_new - lat) < tolerance:
            lat = lat_new
            break
        lat = lat_new

    # Compute altitude and convert to depth
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)

    if np.abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = np.abs(z) / np.abs(sin_lat) - N * (1 - e2)

    # Depth is negative altitude
    depth = -alt

    return lat, lon, depth


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_depth=0.0):
    """Convert ECEF coordinates to local ENU (East-North-Up).

    The ENU frame is centered at the reference point with:
    - East: Positive toward increasing longitude
    - North: Positive toward increasing latitude
    - Up: Positive away from Earth center (opposite to depth)

    Parameters
    ----------
    x : float
        ECEF X-coordinate in meters
    y : float
        ECEF Y-coordinate in meters
    z : float
        ECEF Z-coordinate in meters
    ref_lat : float
        Reference latitude in radians
    ref_lon : float
        Reference longitude in radians
    ref_depth : float, optional
        Reference depth in meters (default 0 = surface)

    Returns
    -------
    tuple
        (east, north, up) coordinates in meters
    """
    # Get reference point in ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef_underwater(ref_lat, ref_lon, ref_depth)

    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    sin_lat = np.sin(ref_lat)
    cos_lat = np.cos(ref_lat)
    sin_lon = np.sin(ref_lon)
    cos_lon = np.cos(ref_lon)

    # Rotation matrix ECEF to ENU
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def enu_to_ecef(east, north, up, ref_lat, ref_lon, ref_depth=0.0):
    """Convert local ENU coordinates to ECEF.

    Parameters
    ----------
    east : float
        East coordinate in meters
    north : float
        North coordinate in meters
    up : float
        Up coordinate in meters (positive away from Earth)
    ref_lat : float
        Reference latitude in radians
    ref_lon : float
        Reference longitude in radians
    ref_depth : float, optional
        Reference depth in meters (default 0 = surface)

    Returns
    -------
    tuple
        (x, y, z) ECEF coordinates in meters
    """
    # Get reference point in ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef_underwater(ref_lat, ref_lon, ref_depth)

    sin_lat = np.sin(ref_lat)
    cos_lat = np.cos(ref_lat)
    sin_lon = np.sin(ref_lon)
    cos_lon = np.cos(ref_lon)

    # Rotation matrix ENU to ECEF (transpose of ECEF to ENU)
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    return x, y, z


def geodetic_depth_to_enu(lat, lon, depth, ref_lat, ref_lon, ref_depth=0.0):
    """Convert geodetic coordinates with depth to local ENU.

    This is the primary function for converting geodetic underwater
    positions to the local ENU coordinate system used in tracking.

    Parameters
    ----------
    lat : float
        Target latitude in radians
    lon : float
        Target longitude in radians
    depth : float
        Target depth in meters (positive downward)
    ref_lat : float
        Reference latitude in radians
    ref_lon : float
        Reference longitude in radians
    ref_depth : float, optional
        Reference depth in meters (default 0 = surface)

    Returns
    -------
    tuple
        (east, north, up) coordinates in meters

    Notes
    -----
    In underwater tracking, 'up' is typically negative (below surface),
    and the reference point is often at the surface above the tracking area.
    """
    # Convert target to ECEF
    x, y, z = geodetic_to_ecef_underwater(lat, lon, depth)

    # Convert to ENU
    return ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_depth)


def enu_to_geodetic_depth(east, north, up, ref_lat, ref_lon, ref_depth=0.0):
    """Convert local ENU coordinates to geodetic with depth.

    Parameters
    ----------
    east : float
        East coordinate in meters
    north : float
        North coordinate in meters
    up : float
        Up coordinate in meters (negative for underwater)
    ref_lat : float
        Reference latitude in radians
    ref_lon : float
        Reference longitude in radians
    ref_depth : float, optional
        Reference depth in meters (default 0 = surface)

    Returns
    -------
    tuple
        (lat, lon, depth) where:
        - lat: Geodetic latitude in radians
        - lon: Geodetic longitude in radians
        - depth: Depth in meters (positive downward)
    """
    # Convert ENU to ECEF
    x, y, z = enu_to_ecef(east, north, up, ref_lat, ref_lon, ref_depth)

    # Convert ECEF to geodetic with depth
    return ecef_to_geodetic_underwater(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2, depth1=0.0, depth2=0.0):
    """Calculate distance between two underwater points.

    Uses the haversine formula for horizontal distance, then applies
    Pythagorean theorem with depth difference for 3D distance.

    Parameters
    ----------
    lat1 : float
        Latitude of first point in radians
    lon1 : float
        Longitude of first point in radians
    lat2 : float
        Latitude of second point in radians
    lon2 : float
        Longitude of second point in radians
    depth1 : float, optional
        Depth of first point in meters (default 0)
    depth2 : float, optional
        Depth of second point in meters (default 0)

    Returns
    -------
    float
        3D distance between points in meters
    """
    # Use mean Earth radius at the average depth
    avg_depth = (depth1 + depth2) / 2
    R = WGS84_SEMI_MAJOR_AXIS - avg_depth  # Approximate radius at depth

    # Haversine formula for horizontal distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    horizontal_dist = R * c

    # 3D distance including depth difference
    depth_diff = depth2 - depth1
    distance = np.sqrt(horizontal_dist**2 + depth_diff**2)

    return distance


# =============================================================================
# Temperature/Salinity Profile Functions
# =============================================================================


def create_isothermal_profile(temperature, salinity, max_depth=1000.0, num_points=100):
    """Create an isothermal (constant temperature) water column profile.

    Parameters
    ----------
    temperature : float
        Constant temperature in degrees Celsius
    salinity : float
        Constant salinity in PSU
    max_depth : float, optional
        Maximum depth in meters (default 1000m)
    num_points : int, optional
        Number of depth points (default 100)

    Returns
    -------
    dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays
    """
    depths = np.linspace(0, max_depth, num_points)
    temperatures = np.full(num_points, temperature)
    salinities = np.full(num_points, salinity)

    return {
        "depth": depths,
        "temperature": temperatures,
        "salinity": salinities,
    }


def create_thermocline_profile(
    surface_temp,
    deep_temp,
    thermocline_depth,
    thermocline_thickness,
    salinity=35.0,
    max_depth=1000.0,
    num_points=100,
):
    """Create a water column profile with a thermocline layer.

    The thermocline is modeled as a hyperbolic tangent transition between
    surface and deep water temperatures.

    Parameters
    ----------
    surface_temp : float
        Surface layer temperature in degrees Celsius
    deep_temp : float
        Deep water temperature in degrees Celsius
    thermocline_depth : float
        Center depth of thermocline in meters
    thermocline_thickness : float
        Thickness of thermocline transition in meters
    salinity : float, optional
        Constant salinity in PSU (default 35)
    max_depth : float, optional
        Maximum depth in meters (default 1000m)
    num_points : int, optional
        Number of depth points (default 100)

    Returns
    -------
    dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays

    Notes
    -----
    The temperature profile uses a tanh function:
        T(z) = (T_surface + T_deep)/2 - (T_surface - T_deep)/2 * tanh((z - z_therm) / thickness)
    """
    depths = np.linspace(0, max_depth, num_points)

    # Tanh transition for temperature
    temp_mean = (surface_temp + deep_temp) / 2
    temp_range = (surface_temp - deep_temp) / 2
    temperatures = temp_mean - temp_range * np.tanh(
        (depths - thermocline_depth) / thermocline_thickness
    )

    salinities = np.full(num_points, salinity)

    return {
        "depth": depths,
        "temperature": temperatures,
        "salinity": salinities,
    }


def create_mixed_layer_profile(
    mixed_layer_temp,
    mixed_layer_depth,
    thermocline_gradient,
    deep_temp,
    salinity=35.0,
    max_depth=1000.0,
    num_points=100,
):
    """Create a profile with a mixed surface layer and thermocline.

    This models a typical ocean structure with:
    - Warm, well-mixed surface layer
    - Sharp thermocline below mixed layer
    - Cold deep water

    Parameters
    ----------
    mixed_layer_temp : float
        Temperature of mixed layer in degrees Celsius
    mixed_layer_depth : float
        Depth of mixed layer in meters
    thermocline_gradient : float
        Temperature gradient in thermocline (deg C per meter, negative)
    deep_temp : float
        Deep water temperature in degrees Celsius
    salinity : float, optional
        Constant salinity in PSU (default 35)
    max_depth : float, optional
        Maximum depth in meters (default 1000m)
    num_points : int, optional
        Number of depth points (default 100)

    Returns
    -------
    dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays
    """
    depths = np.linspace(0, max_depth, num_points)
    temperatures = np.zeros(num_points)

    for i, d in enumerate(depths):
        if d <= mixed_layer_depth:
            # Mixed layer - constant temperature
            temperatures[i] = mixed_layer_temp
        else:
            # Below mixed layer - linear decrease until reaching deep temp
            temp = mixed_layer_temp + thermocline_gradient * (d - mixed_layer_depth)
            temperatures[i] = max(temp, deep_temp)

    salinities = np.full(num_points, salinity)

    return {
        "depth": depths,
        "temperature": temperatures,
        "salinity": salinities,
    }


def interpolate_profile(profile, depth):
    """Interpolate temperature and salinity at a given depth.

    Parameters
    ----------
    profile : dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays
    depth : float or array_like
        Depth(s) at which to interpolate

    Returns
    -------
    tuple
        (temperature, salinity) at the requested depth(s)
    """
    depth = np.asarray(depth)
    scalar_input = depth.ndim == 0
    depth = np.atleast_1d(depth)

    # Linear interpolation
    temperature = np.interp(depth, profile["depth"], profile["temperature"])
    salinity = np.interp(depth, profile["depth"], profile["salinity"])

    if scalar_input:
        return float(temperature[0]), float(salinity[0])
    return temperature, salinity


def sound_speed_profile(profile, depths=None):
    """Calculate sound speed profile from temperature/salinity profile.

    Parameters
    ----------
    profile : dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays
    depths : array_like, optional
        Depths at which to calculate sound speed. If None, uses profile depths.

    Returns
    -------
    dict
        Profile dictionary with 'depth' and 'sound_speed' arrays
    """
    depths = profile["depth"] if depths is None else np.asarray(depths)

    temperatures, salinities = interpolate_profile(profile, depths)

    # Calculate sound speed at each depth
    sound_speeds = np.array(
        [
            sound_speed_mackenzie(t, s, d)
            for t, s, d in zip(temperatures, salinities, depths, strict=False)
        ]
    )

    return {
        "depth": depths,
        "sound_speed": sound_speeds,
    }


def find_sound_channel_axis(profile):
    """Find the depth of the SOFAR channel axis (sound speed minimum).

    The SOFAR (Sound Fixing and Ranging) channel is a horizontal layer
    in the ocean where sound speed reaches a minimum. Sound waves
    trapped in this channel can travel very long distances.

    Parameters
    ----------
    profile : dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays

    Returns
    -------
    float
        Depth of sound speed minimum in meters, or None if no minimum found
    """
    svp = sound_speed_profile(profile)
    speeds = svp["sound_speed"]
    depths = svp["depth"]

    # Find minimum (excluding surface and bottom)
    if len(speeds) < 3:
        return None

    min_idx = np.argmin(speeds[1:-1]) + 1
    # Check if it's a true local minimum
    if speeds[min_idx] < speeds[min_idx - 1] and speeds[min_idx] < speeds[min_idx + 1]:
        return depths[min_idx]

    return None


def compute_sound_speed_gradient(profile, depth):
    """Compute the vertical gradient of sound speed at a given depth.

    Parameters
    ----------
    profile : dict
        Profile dictionary with 'depth', 'temperature', 'salinity' arrays
    depth : float
        Depth at which to compute gradient

    Returns
    -------
    float
        Sound speed gradient in (m/s)/m (positive = increasing with depth)
    """
    svp = sound_speed_profile(profile)

    # Find bracketing depths
    depths = svp["depth"]
    speeds = svp["sound_speed"]

    # Use central difference where possible
    idx = np.searchsorted(depths, depth)

    if idx == 0:
        idx = 1
    elif idx >= len(depths) - 1:
        idx = len(depths) - 2

    dz = depths[idx + 1] - depths[idx - 1]
    dc = speeds[idx + 1] - speeds[idx - 1]

    return dc / dz if dz != 0 else 0.0


# =============================================================================
# Ocean Current Models
# =============================================================================


def create_uniform_current(velocity_east, velocity_north, velocity_down=0.0):
    """Create a uniform (constant) ocean current field.

    Parameters
    ----------
    velocity_east : float
        Eastward velocity component in m/s
    velocity_north : float
        Northward velocity component in m/s
    velocity_down : float, optional
        Downward velocity component in m/s (default 0)

    Returns
    -------
    callable
        Function that returns (vx, vy, vz) at any position
    """

    def current_field(x, y, z):
        return velocity_east, velocity_north, velocity_down

    return current_field


def create_depth_varying_current(surface_velocity, decay_depth, direction_rad):
    """Create an exponentially decaying current with depth.

    Models wind-driven currents that are strongest at the surface
    and decay exponentially with depth (Ekman-like behavior).

    Parameters
    ----------
    surface_velocity : float
        Current speed at surface in m/s
    decay_depth : float
        e-folding depth in meters (depth at which velocity = surface/e)
    direction_rad : float
        Current direction in radians (0 = north, pi/2 = east)

    Returns
    -------
    callable
        Function that returns (vx, vy, vz) at any position
    """

    def current_field(x, y, z):
        # z is up, so depth = -z for underwater
        depth = max(-z, 0)
        speed = surface_velocity * np.exp(-depth / decay_depth)
        vx = speed * np.sin(direction_rad)
        vy = speed * np.cos(direction_rad)
        return vx, vy, 0.0

    return current_field


def create_shear_current(surface_velocity, shear_rate, direction_rad, max_depth=1000.0):
    """Create a current with linear velocity shear.

    Parameters
    ----------
    surface_velocity : float
        Current speed at surface in m/s
    shear_rate : float
        Rate of velocity decrease with depth in (m/s)/m
    direction_rad : float
        Current direction in radians (0 = north, pi/2 = east)
    max_depth : float, optional
        Depth below which current is zero (default 1000m)

    Returns
    -------
    callable
        Function that returns (vx, vy, vz) at any position
    """

    def current_field(x, y, z):
        depth = max(-z, 0)
        if depth >= max_depth:
            return 0.0, 0.0, 0.0
        speed = max(surface_velocity - shear_rate * depth, 0.0)
        vx = speed * np.sin(direction_rad)
        vy = speed * np.cos(direction_rad)
        return vx, vy, 0.0

    return current_field


def apply_current_to_velocity(vx, vy, vz, current_field, x, y, z):
    """Add ocean current to a velocity vector.

    Parameters
    ----------
    vx, vy, vz : float
        Velocity components in m/s (object velocity in still water)
    current_field : callable
        Current field function returning (vx, vy, vz) at position
    x, y, z : float
        Position in meters

    Returns
    -------
    tuple
        (vx_total, vy_total, vz_total) velocity relative to fixed frame
    """
    cx, cy, cz = current_field(x, y, z)
    return vx + cx, vy + cy, vz + cz


# =============================================================================
# Bathymetry Grid Support
# =============================================================================


def create_flat_bathymetry(depth, x_range, y_range, resolution=100.0):
    """Create a flat (constant depth) bathymetry grid.

    Parameters
    ----------
    depth : float
        Constant water depth in meters (positive)
    x_range : tuple
        (x_min, x_max) extent in meters
    y_range : tuple
        (y_min, y_max) extent in meters
    resolution : float, optional
        Grid cell size in meters (default 100m)

    Returns
    -------
    dict
        Bathymetry grid with 'x', 'y', 'depth' arrays
    """
    x = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y = np.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    depths = np.full(X.shape, depth)

    return {"x": x, "y": y, "depth": depths}


def create_sloped_bathymetry(
    shallow_depth,
    deep_depth,
    slope_start_x,
    slope_end_x,
    x_range,
    y_range,
    resolution=100.0,
):
    """Create a bathymetry grid with a linear slope.

    Models a continental shelf transitioning to deep water.

    Parameters
    ----------
    shallow_depth : float
        Depth in shallow region in meters
    deep_depth : float
        Depth in deep region in meters
    slope_start_x : float
        X coordinate where slope begins
    slope_end_x : float
        X coordinate where slope ends
    x_range : tuple
        (x_min, x_max) extent in meters
    y_range : tuple
        (y_min, y_max) extent in meters
    resolution : float, optional
        Grid cell size in meters (default 100m)

    Returns
    -------
    dict
        Bathymetry grid with 'x', 'y', 'depth' arrays
    """
    x = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y = np.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = np.meshgrid(x, y)

    depths = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            if xi <= slope_start_x:
                depths[i, j] = shallow_depth
            elif xi >= slope_end_x:
                depths[i, j] = deep_depth
            else:
                # Linear interpolation on slope
                t = (xi - slope_start_x) / (slope_end_x - slope_start_x)
                depths[i, j] = shallow_depth + t * (deep_depth - shallow_depth)

    return {"x": x, "y": y, "depth": depths}


def create_canyon_bathymetry(
    base_depth, canyon_depth, canyon_center_y, canyon_width, x_range, y_range, resolution=100.0
):
    """Create a bathymetry grid with an underwater canyon.

    Parameters
    ----------
    base_depth : float
        Base water depth in meters
    canyon_depth : float
        Additional depth of canyon center in meters
    canyon_center_y : float
        Y coordinate of canyon centerline
    canyon_width : float
        Width of canyon in meters (at half-depth)
    x_range : tuple
        (x_min, x_max) extent in meters
    y_range : tuple
        (y_min, y_max) extent in meters
    resolution : float, optional
        Grid cell size in meters (default 100m)

    Returns
    -------
    dict
        Bathymetry grid with 'x', 'y', 'depth' arrays
    """
    x = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y = np.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = np.meshgrid(x, y)

    # Gaussian canyon profile
    canyon_profile = canyon_depth * np.exp(
        -((Y - canyon_center_y) ** 2) / (2 * (canyon_width / 2) ** 2)
    )
    depths = base_depth + canyon_profile

    return {"x": x, "y": y, "depth": depths}


def interpolate_bathymetry(bathymetry, x, y):
    """Interpolate seafloor depth at a given position.

    Parameters
    ----------
    bathymetry : dict
        Bathymetry grid with 'x', 'y', 'depth' arrays
    x : float or array_like
        X coordinate(s) in meters
    y : float or array_like
        Y coordinate(s) in meters

    Returns
    -------
    float or ndarray
        Interpolated depth(s) in meters
    """
    from scipy.interpolate import RegularGridInterpolator

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    scalar_input = x_arr.ndim == 0 and y_arr.ndim == 0

    x_arr = np.atleast_1d(x_arr)
    y_arr = np.atleast_1d(y_arr)

    # Create interpolator
    interp = RegularGridInterpolator(
        (bathymetry["y"], bathymetry["x"]),
        bathymetry["depth"],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Interpolate
    points = np.column_stack([y_arr, x_arr])
    depths = interp(points)

    if scalar_input:
        return float(depths[0])
    return depths


def height_above_seafloor(x, y, z, bathymetry):
    """Calculate height above seafloor at a given position.

    Parameters
    ----------
    x : float
        X coordinate in meters
    y : float
        Y coordinate in meters
    z : float
        Z coordinate in meters (negative = underwater)
    bathymetry : dict
        Bathymetry grid

    Returns
    -------
    float
        Height above seafloor in meters (negative = below seafloor)
    """
    seafloor_depth = interpolate_bathymetry(bathymetry, x, y)
    # z is up, seafloor is at z = -seafloor_depth
    return z - (-seafloor_depth)


def is_in_water(x, y, z, bathymetry):
    """Check if a point is within the water column.

    Parameters
    ----------
    x : float
        X coordinate in meters
    y : float
        Y coordinate in meters
    z : float
        Z coordinate in meters (negative = underwater)
    bathymetry : dict
        Bathymetry grid

    Returns
    -------
    bool
        True if point is in water (below surface, above seafloor)
    """
    if z > 0:  # Above water
        return False
    return height_above_seafloor(x, y, z, bathymetry) >= 0


# =============================================================================
# Tidal Effect Models
# =============================================================================


def simple_tidal_offset(time_hours, amplitude, period_hours=12.42):
    """Calculate simple sinusoidal tidal height offset.

    Parameters
    ----------
    time_hours : float
        Time in hours from reference (e.g., high tide)
    amplitude : float
        Tidal amplitude in meters (half of tidal range)
    period_hours : float, optional
        Tidal period in hours (default 12.42 for M2 constituent)

    Returns
    -------
    float
        Tidal height offset in meters (positive = high tide)
    """
    omega = 2 * np.pi / period_hours
    return amplitude * np.cos(omega * time_hours)


def dual_constituent_tide(time_hours, amp_m2, amp_s2, phase_m2=0.0, phase_s2=0.0):
    """Calculate tide from M2 (lunar) and S2 (solar) constituents.

    This produces the spring-neap cycle typical of most tidal regions.

    Parameters
    ----------
    time_hours : float
        Time in hours from reference
    amp_m2 : float
        M2 (lunar semidiurnal) amplitude in meters
    amp_s2 : float
        S2 (solar semidiurnal) amplitude in meters
    phase_m2 : float, optional
        M2 phase offset in radians
    phase_s2 : float, optional
        S2 phase offset in radians

    Returns
    -------
    float
        Tidal height offset in meters
    """
    # M2 period = 12.42 hours, S2 period = 12.00 hours
    omega_m2 = 2 * np.pi / 12.42
    omega_s2 = 2 * np.pi / 12.00

    m2 = amp_m2 * np.cos(omega_m2 * time_hours + phase_m2)
    s2 = amp_s2 * np.cos(omega_s2 * time_hours + phase_s2)

    return m2 + s2


def tidal_current(time_hours, max_speed, direction_rad, period_hours=12.42, phase=0.0):
    """Calculate tidal current velocity.

    Models a simple reversing tidal current with sinusoidal variation.

    Parameters
    ----------
    time_hours : float
        Time in hours from reference
    max_speed : float
        Maximum current speed in m/s
    direction_rad : float
        Principal current direction in radians (0=north, pi/2=east)
    period_hours : float, optional
        Tidal period in hours
    phase : float, optional
        Phase offset in radians (0 = max flood at t=0)

    Returns
    -------
    tuple
        (vx, vy) current velocity components in m/s
    """
    omega = 2 * np.pi / period_hours
    speed = max_speed * np.sin(omega * time_hours + phase)

    vx = speed * np.sin(direction_rad)
    vy = speed * np.cos(direction_rad)

    return vx, vy


def apply_tide_to_depth(depth, time_hours, tidal_amplitude, period_hours=12.42):
    """Adjust water depth for tidal effects.

    Parameters
    ----------
    depth : float
        Mean water depth in meters
    time_hours : float
        Time in hours from reference
    tidal_amplitude : float
        Tidal amplitude in meters
    period_hours : float, optional
        Tidal period in hours

    Returns
    -------
    float
        Actual water depth accounting for tide
    """
    tide_offset = simple_tidal_offset(time_hours, tidal_amplitude, period_hours)
    return depth + tide_offset
