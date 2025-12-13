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
