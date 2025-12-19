#!/usr/bin/env python

"""
============================================
11 - Coordinate Systems and Transformations
============================================
"""

# %%
# This tutorial introduces Stone Soup's coordinate system capabilities, including:
#
# - Reference ellipsoids (WGS84, GRS80, etc.)
# - Geodetic to ECEF transformations
# - ECI to ECEF transformations at different precision levels
# - Using Earth Orientation Parameters for high-precision work
#
# Background
# ----------
#
# Coordinate transformations are fundamental to tracking applications. Stone Soup provides
# native implementations of common geodetic and celestial coordinate transformations without
# requiring external libraries.
#
# The key coordinate systems covered are:
#
# - **Geodetic**: Latitude, longitude, altitude relative to an ellipsoid
# - **ECEF**: Earth-Centered Earth-Fixed Cartesian coordinates
# - **ECI**: Earth-Centered Inertial coordinates (for space applications)
# - **GCRS**: Geocentric Celestial Reference System
#
# Reference Ellipsoids
# --------------------
#
# A reference ellipsoid is a mathematical model of the Earth's shape. Different applications
# use different ellipsoids:

# %%
from stonesoup.types.coordinates import (
    WGS84, WGS84_G2139, GRS80, WGS72, PZ90, CGCS2000
)

# WGS84 is the standard for GPS
print(f"WGS84 semi-major axis: {WGS84.semi_major_axis} m")
print(f"WGS84 flattening: 1/{1/WGS84.flattening:.9f}")
print(f"WGS84 eccentricity: {WGS84.eccentricity:.10f}")

# %%
# Different ellipsoids have slightly different parameters:

ellipsoids = [WGS84, GRS80, WGS72, PZ90, CGCS2000]
print("\nEllipsoid Comparison:")
print("-" * 60)
for ell in ellipsoids:
    print(f"{ell.name:15s}: a={ell.semi_major_axis:.1f} m, 1/f={1/ell.flattening:.6f}")

# %%
# For most applications, the differences are negligible (sub-millimeter at Earth's surface).
# Use WGS84 for GPS/GNSS applications, PZ90 for GLONASS, and CGCS2000 for BeiDou.
#
# Geodetic to ECEF Transformation
# -------------------------------
#
# Convert latitude, longitude, altitude to Earth-Centered Earth-Fixed (ECEF) coordinates:

# %%
import numpy as np
from stonesoup.functions.coordinates import geodetic_to_ecef, ecef_to_geodetic

# London coordinates
lat_london = np.radians(51.5074)  # Latitude in radians
lon_london = np.radians(-0.1278)  # Longitude in radians
alt_london = 11.0  # Altitude in meters (above ellipsoid)

# Convert to ECEF
xyz_london = geodetic_to_ecef(lat_london, lon_london, alt_london)
print(f"London in ECEF: X={xyz_london[0]:.1f}, Y={xyz_london[1]:.1f}, Z={xyz_london[2]:.1f} m")

# %%
# The inverse transformation recovers the geodetic coordinates:

lat_back, lon_back, alt_back = ecef_to_geodetic(xyz_london[0], xyz_london[1], xyz_london[2])
print(f"Recovered: lat={np.degrees(lat_back):.6f}°, lon={np.degrees(lon_back):.6f}°, alt={alt_back:.3f} m")

# %%
# Roundtrip accuracy is excellent (sub-millimeter):

lat_error = abs(lat_back - lat_london)
lon_error = abs(lon_back - lon_london)
alt_error = abs(alt_back - alt_london)
print(f"Roundtrip errors: lat={lat_error:.2e} rad, lon={lon_error:.2e} rad, alt={alt_error:.2e} m")

# %%
# ECI to ECEF Transformations
# ---------------------------
#
# For space applications, we need to transform between Earth-Centered Inertial (ECI)
# and ECEF frames. Stone Soup provides three precision levels:
#
# 1. **Simple** (ERA only): Fast but ~100 km accuracy
# 2. **Standard** (ERA + precession + nutation): ~1-10 m accuracy
# 3. **High-precision** (full EOP): ~1 cm accuracy
#
# Simple ECI-ECEF Transformation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
from datetime import datetime
from stonesoup.functions.coordinates import eci_to_ecef, ecef_to_eci

# A satellite position in ECI coordinates
pos_eci = np.array([7000000.0, 0.0, 0.0])  # 7000 km along X-axis
timestamp = datetime(2024, 7, 1, 12, 0, 0)

# Transform to ECEF (simple, ERA only)
pos_ecef_simple = eci_to_ecef(pos_eci, timestamp)
print(f"ECI position: {pos_eci}")
print(f"ECEF (simple): {pos_ecef_simple}")

# %%
# The position magnitude is preserved (pure rotation):

print(f"ECI magnitude: {np.linalg.norm(pos_eci):.1f} m")
print(f"ECEF magnitude: {np.linalg.norm(pos_ecef_simple):.1f} m")

# %%
# Standard ECI-ECEF Transformation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For better accuracy, use the full transformation with IAU precession and nutation models:

# %%
from stonesoup.functions.coordinates import eci_to_ecef_full, ecef_to_eci_full

# Include velocity
vel_eci = np.array([0.0, 7500.0, 0.0])  # ~7.5 km/s orbital velocity

pos_ecef_full, vel_ecef_full = eci_to_ecef_full(pos_eci, timestamp, vel_eci)
print(f"ECEF (full): {pos_ecef_full}")
print(f"ECEF velocity: {vel_ecef_full}")

# %%
# The difference between simple and full transformations is significant:

diff = np.linalg.norm(pos_ecef_full - pos_ecef_simple)
print(f"Difference between simple and full: {diff/1000:.1f} km")

# %%
# This ~40 km difference comes from precession over ~24 years since J2000.0.
#
# Roundtrip accuracy is excellent:

pos_recovered, vel_recovered = ecef_to_eci_full(pos_ecef_full, timestamp, vel_ecef_full)
pos_error = np.linalg.norm(pos_recovered - pos_eci)
vel_error = np.linalg.norm(vel_recovered - vel_eci)
print(f"Position roundtrip error: {pos_error:.2e} m")
print(f"Velocity roundtrip error: {vel_error:.2e} m/s")

# %%
# High-Precision ECI-ECEF with EOP
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For the highest accuracy (GNSS, precision orbit determination), use Earth Orientation
# Parameters (EOP) from IERS:

# %%
from stonesoup.functions.coordinates import (
    EarthOrientationParameters,
    eci_to_ecef_with_eop,
    ecef_to_eci_with_eop,
    datetime_to_mjd
)

# Create EOP data (normally loaded from IERS finals2000A.all file)
# MJD 60548 corresponds to 2024-08-26
eop = EarthOrientationParameters(
    epochs=np.array([60547.0, 60548.0, 60549.0]),
    x_p=np.array([0.10, 0.11, 0.12]),      # Polar motion X (arcsec)
    y_p=np.array([0.30, 0.31, 0.32]),      # Polar motion Y (arcsec)
    ut1_utc=np.array([-0.10, -0.10, -0.10]) # UT1-UTC (seconds)
)

print(f"EOP data range: MJD {eop.start_epoch} to {eop.end_epoch}")

# %%
# Transform using EOP:

timestamp_eop = datetime(2024, 8, 26, 0, 0, 0)
pos_ecef_eop, vel_ecef_eop = eci_to_ecef_with_eop(pos_eci, timestamp_eop, eop, vel_eci)

print(f"ECEF (with EOP): {pos_ecef_eop}")

# %%
# Compare with standard transformation at same epoch:

pos_ecef_std, _ = eci_to_ecef_full(pos_eci, timestamp_eop, vel_eci)
diff_eop = np.linalg.norm(pos_ecef_eop - pos_ecef_std)
print(f"Difference (EOP vs standard): {diff_eop:.1f} m")

# %%
# The ~100-200 m difference comes from:
#
# - Polar motion corrections (~20 m at Earth's surface)
# - UT1-UTC timing correction (~150 m at equator per 0.1s offset)
#
# Loading Real EOP Data
# ^^^^^^^^^^^^^^^^^^^^^
#
# For real applications, load EOP data from IERS:
#
# .. code-block:: python
#
#     # Download from https://datacenter.iers.org/data/9/finals2000A.all
#     eop = EarthOrientationParameters.from_finals2000a('finals2000A.all')
#
# Time-Varying Transformations
# ----------------------------
#
# ECI-ECEF transformations are time-dependent due to Earth's rotation:

# %%
t1 = datetime(2024, 1, 1, 0, 0, 0)
t2 = datetime(2024, 1, 1, 6, 0, 0)  # 6 hours later

pos_t1, _ = eci_to_ecef_full(pos_eci, t1)
pos_t2, _ = eci_to_ecef_full(pos_eci, t2)

angular_change = np.degrees(np.arccos(
    np.dot(pos_t1, pos_t2) / (np.linalg.norm(pos_t1) * np.linalg.norm(pos_t2))
))
print(f"Angular change in 6 hours: {angular_change:.1f}°")
print(f"(Expected: 90° for 6 hours of Earth rotation)")

# %%
# Choosing the Right Precision Level
# -----------------------------------
#
# +---------------------+------------------+------------------+
# | Application         | Function         | Accuracy         |
# +=====================+==================+==================+
# | Quick estimates     | eci_to_ecef      | ~100 km          |
# +---------------------+------------------+------------------+
# | General tracking    | eci_to_ecef_full | ~1-10 m          |
# +---------------------+------------------+------------------+
# | LEO satellites      | eci_to_ecef_full | ~1-10 m          |
# +---------------------+------------------+------------------+
# | Precision orbits    | eci_to_ecef_with_eop | ~1 cm        |
# +---------------------+------------------+------------------+
# | GNSS positioning    | eci_to_ecef_with_eop | ~1 cm        |
# +---------------------+------------------+------------------+
#
# Summary
# -------
#
# Stone Soup provides comprehensive coordinate transformation capabilities:
#
# - Multiple reference ellipsoids (WGS84, GRS80, PZ90, etc.)
# - Fast geodetic-ECEF conversions with sub-millimeter accuracy
# - Three levels of ECI-ECEF precision for different applications
# - EOP support for highest-precision transformations
#
# For most tracking applications, the standard `eci_to_ecef_full` transformation
# provides sufficient accuracy while maintaining good performance.

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/CoordinateSystems.png'
