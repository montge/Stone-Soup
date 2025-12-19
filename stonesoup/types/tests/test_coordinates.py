"""Test coordinate system types and transformations."""

import datetime

import numpy as np
import pytest

from stonesoup.functions.coordinates import (
    EarthOrientationParameters,
    apply_nutation,
    apply_precession_date_to_j2000,
    apply_precession_j2000_to_date,
    compute_frame_bias_matrix,
    compute_fundamental_arguments,
    compute_nutation_iau2000b,
    compute_nutation_matrix,
    compute_polar_motion_matrix,
    compute_precession_angles_iau2006,
    compute_precession_matrix_iau2006,
    datetime_to_mjd,
    ecef_to_eci,
    ecef_to_eci_full,
    ecef_to_eci_with_eop,
    ecef_to_geodetic,
    eci_to_ecef,
    eci_to_ecef_full,
    eci_to_ecef_with_eop,
    eci_to_geodetic,
    gcrs_to_j2000,
    geodetic_to_ecef,
    geodetic_to_eci,
    j2000_to_gcrs,
)
from stonesoup.types.coordinates import (
    CGCS2000,
    GCRS,
    GRS80,
    ICRS,
    J2000,
    PZ90,
    WGS72,
    WGS84,
    WGS84_G730,
    WGS84_G873,
    WGS84_G1150,
    WGS84_G1674,
    WGS84_G1762,
    WGS84_G2139,
    EpochCachedTransform,
    InterpolatedTransform,
    ReferenceEllipsoid,
    RotationRateTransform,
    TimeVaryingTransform,
)

# Tests for ReferenceEllipsoid class


def test_ellipsoid_creation():
    """Test creating a custom reference ellipsoid."""
    ellipsoid = ReferenceEllipsoid(
        name="Custom", semi_major_axis=6378137.0, flattening=1 / 298.257223563
    )
    assert ellipsoid.name == "Custom"
    assert ellipsoid.semi_major_axis == 6378137.0
    assert ellipsoid.flattening == 1 / 298.257223563


def test_ellipsoid_semi_minor_axis():
    """Test semi-minor axis calculation."""
    # WGS84 parameters
    a = 6378137.0
    f = 1 / 298.257223563
    expected_b = a * (1.0 - f)

    ellipsoid = ReferenceEllipsoid(name="Test", semi_major_axis=a, flattening=f)
    assert pytest.approx(ellipsoid.semi_minor_axis, rel=1e-10) == expected_b
    # Known value for WGS84
    assert pytest.approx(ellipsoid.semi_minor_axis, abs=0.001) == 6356752.314


def test_ellipsoid_eccentricity():
    """Test eccentricity calculations."""
    # WGS84 parameters
    a = 6378137.0
    f = 1 / 298.257223563
    expected_e = np.sqrt(2.0 * f - f**2)

    ellipsoid = ReferenceEllipsoid(name="Test", semi_major_axis=a, flattening=f)
    assert pytest.approx(ellipsoid.eccentricity, rel=1e-10) == expected_e
    # Known value for WGS84
    assert pytest.approx(ellipsoid.eccentricity, abs=1e-10) == 0.0818191908


def test_ellipsoid_eccentricity_squared():
    """Test squared eccentricity calculation."""
    a = 6378137.0
    f = 1 / 298.257223563
    expected_e2 = 2.0 * f - f**2

    ellipsoid = ReferenceEllipsoid(name="Test", semi_major_axis=a, flattening=f)
    assert pytest.approx(ellipsoid.eccentricity_squared, rel=1e-10) == expected_e2
    assert pytest.approx(ellipsoid.eccentricity_squared) == ellipsoid.eccentricity**2


def test_ellipsoid_second_eccentricity_squared():
    """Test second eccentricity squared calculation."""
    a = 6378137.0
    b = 6356752.314
    expected_e_prime2 = (a**2 - b**2) / b**2

    ellipsoid = ReferenceEllipsoid(name="Test", semi_major_axis=a, flattening=1 / 298.257223563)
    assert pytest.approx(ellipsoid.second_eccentricity_squared, abs=1e-8) == expected_e_prime2


def test_ellipsoid_linear_eccentricity():
    """Test linear eccentricity calculation."""
    a = 6378137.0
    f = 1 / 298.257223563

    ellipsoid = ReferenceEllipsoid(name="Test", semi_major_axis=a, flattening=f)
    expected_E = a * ellipsoid.eccentricity
    assert pytest.approx(ellipsoid.linear_eccentricity) == expected_E
    # Also verify against sqrt(a^2 - b^2)
    assert pytest.approx(ellipsoid.linear_eccentricity) == np.sqrt(
        a**2 - ellipsoid.semi_minor_axis**2
    )


# Tests for predefined ellipsoids


def test_wgs84_parameters():
    """Test WGS84 ellipsoid parameters."""
    assert WGS84.name == "WGS84 (G2139)"
    assert WGS84.semi_major_axis == 6378137.0
    assert WGS84.flattening == 1.0 / 298.257223563
    # Known value
    assert pytest.approx(WGS84.semi_minor_axis, abs=0.001) == 6356752.314
    assert pytest.approx(WGS84.eccentricity, abs=1e-10) == 0.0818191908


def test_wgs84_is_latest_realization():
    """Test that WGS84 is the latest realization (G2139)."""
    assert WGS84.name == WGS84_G2139.name
    assert WGS84.semi_major_axis == WGS84_G2139.semi_major_axis
    assert WGS84.flattening == WGS84_G2139.flattening


def test_wgs84_realizations_same_ellipsoid():
    """Test that all WGS84 realizations use the same ellipsoid parameters."""
    realizations = [WGS84_G730, WGS84_G873, WGS84_G1150, WGS84_G1674, WGS84_G1762, WGS84_G2139]

    for realization in realizations:
        assert realization.semi_major_axis == 6378137.0
        assert realization.flattening == 1.0 / 298.257223563


def test_grs80_parameters():
    """Test GRS80 ellipsoid parameters."""
    assert GRS80.name == "GRS80"
    assert GRS80.semi_major_axis == 6378137.0
    assert GRS80.flattening == 1.0 / 298.257222101
    # GRS80 and WGS84 differ very slightly in flattening
    assert GRS80.flattening != WGS84.flattening
    # But have same semi-major axis
    assert GRS80.semi_major_axis == WGS84.semi_major_axis
    # Difference in semi-minor axis should be ~0.1 mm
    b_diff = abs(GRS80.semi_minor_axis - WGS84.semi_minor_axis)
    assert b_diff < 0.0002  # Less than 0.2 mm


def test_wgs72_parameters():
    """Test WGS72 ellipsoid parameters."""
    assert WGS72.name == "WGS72"
    assert WGS72.semi_major_axis == 6378135.0
    assert WGS72.flattening == 1.0 / 298.26
    # WGS72 has different semi-major axis than WGS84
    assert WGS72.semi_major_axis != WGS84.semi_major_axis


def test_pz90_parameters():
    """Test PZ-90 ellipsoid parameters."""
    assert PZ90.name == "PZ90"
    assert PZ90.semi_major_axis == 6378136.0
    assert PZ90.flattening == 1.0 / 298.257839303


def test_cgcs2000_parameters():
    """Test CGCS2000 ellipsoid parameters."""
    assert CGCS2000.name == "CGCS2000"
    assert CGCS2000.semi_major_axis == 6378137.0
    assert CGCS2000.flattening == 1.0 / 298.257222101
    # CGCS2000 is identical to GRS80
    assert CGCS2000.semi_major_axis == GRS80.semi_major_axis
    assert CGCS2000.flattening == GRS80.flattening


@pytest.mark.parametrize(
    "ellipsoid",
    [
        WGS84,
        WGS84_G730,
        WGS84_G873,
        WGS84_G1150,
        WGS84_G1674,
        WGS84_G1762,
        WGS84_G2139,
        GRS80,
        WGS72,
        PZ90,
        CGCS2000,
    ],
)
def test_ellipsoid_properties_valid(ellipsoid):
    """Test that all predefined ellipsoids have valid derived properties."""
    # All properties should be positive
    assert ellipsoid.semi_major_axis > 0
    assert ellipsoid.semi_minor_axis > 0
    assert ellipsoid.flattening > 0
    assert ellipsoid.eccentricity > 0
    assert ellipsoid.eccentricity_squared > 0
    assert ellipsoid.second_eccentricity_squared > 0
    assert ellipsoid.linear_eccentricity > 0

    # Semi-minor axis should be less than semi-major axis
    assert ellipsoid.semi_minor_axis < ellipsoid.semi_major_axis

    # Eccentricity should be less than 1
    assert ellipsoid.eccentricity < 1

    # Flattening should be small for Earth
    assert ellipsoid.flattening < 0.01


# Tests for geodetic_to_ecef function


def test_geodetic_to_ecef_equator_prime_meridian():
    """Test conversion at equator and prime meridian."""
    # Latitude = 0, Longitude = 0, Altitude = 0
    lat, lon, alt = 0.0, 0.0, 0.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # At equator on prime meridian, x = semi_major_axis, y = 0, z = 0
    expected_x = WGS84.semi_major_axis
    assert pytest.approx(xyz[0], abs=0.01) == expected_x
    assert pytest.approx(xyz[1], abs=0.01) == 0.0
    assert pytest.approx(xyz[2], abs=0.01) == 0.0


def test_geodetic_to_ecef_north_pole():
    """Test conversion at North Pole."""
    # Latitude = 90°, Longitude = 0 (arbitrary), Altitude = 0
    lat, lon, alt = np.pi / 2, 0.0, 0.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # At North Pole, x = 0, y = 0, z = semi_minor_axis
    assert pytest.approx(xyz[0], abs=0.01) == 0.0
    assert pytest.approx(xyz[1], abs=0.01) == 0.0
    assert pytest.approx(xyz[2], abs=0.01) == WGS84.semi_minor_axis


def test_geodetic_to_ecef_south_pole():
    """Test conversion at South Pole."""
    # Latitude = -90°, Longitude = 0 (arbitrary), Altitude = 0
    lat, lon, alt = -np.pi / 2, 0.0, 0.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # At South Pole, x = 0, y = 0, z = -semi_minor_axis
    assert pytest.approx(xyz[0], abs=0.01) == 0.0
    assert pytest.approx(xyz[1], abs=0.01) == 0.0
    assert pytest.approx(xyz[2], abs=0.01) == -WGS84.semi_minor_axis


def test_geodetic_to_ecef_london():
    """Test conversion at London (Greenwich Observatory)."""
    # Greenwich Observatory: 51.4769°N, 0.0005°W, 0m altitude
    lat = np.radians(51.4769)
    lon = np.radians(-0.0005)
    alt = 0.0

    xyz = geodetic_to_ecef(lat, lon, alt)

    # Check that coordinates are reasonable for Greenwich
    # Should be in northern hemisphere (positive z), near prime meridian (small y)
    assert xyz[0] > 3900000.0  # X should be around 3.98M meters
    assert xyz[0] < 4100000.0
    assert abs(xyz[1]) < 100.0  # Y should be small (near prime meridian)
    assert xyz[2] > 4900000.0  # Z should be around 4.97M meters
    assert xyz[2] < 5100000.0


def test_geodetic_to_ecef_with_altitude():
    """Test conversion with non-zero altitude."""
    # At equator with 1000m altitude
    lat, lon, alt = 0.0, 0.0, 1000.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # X should be semi_major_axis + altitude
    expected_x = WGS84.semi_major_axis + 1000.0
    assert pytest.approx(xyz[0], abs=0.01) == expected_x
    assert pytest.approx(xyz[1], abs=0.01) == 0.0
    assert pytest.approx(xyz[2], abs=0.01) == 0.0


def test_geodetic_to_ecef_different_ellipsoid():
    """Test conversion with different ellipsoid."""
    lat, lon, alt = 0.0, 0.0, 0.0

    xyz_wgs84 = geodetic_to_ecef(lat, lon, alt, ellipsoid=WGS84)
    xyz_grs80 = geodetic_to_ecef(lat, lon, alt, ellipsoid=GRS80)

    # At equator, results should be slightly different due to different ellipsoids
    # But very close since WGS84 and GRS80 are very similar
    assert pytest.approx(xyz_wgs84[0], abs=0.01) == xyz_grs80[0]

    xyz_wgs72 = geodetic_to_ecef(lat, lon, alt, ellipsoid=WGS72)
    # WGS72 has different semi-major axis, so should differ
    assert abs(xyz_wgs84[0] - xyz_wgs72[0]) > 1.0


@pytest.mark.parametrize(
    "lat,lon",
    [
        (np.radians(0), np.radians(0)),  # Equator, Prime Meridian
        (np.radians(0), np.radians(90)),  # Equator, 90°E
        (np.radians(0), np.radians(180)),  # Equator, 180°
        (np.radians(0), np.radians(-90)),  # Equator, 90°W
        (np.radians(45), np.radians(0)),  # 45°N, Prime Meridian
        (np.radians(-45), np.radians(0)),  # 45°S, Prime Meridian
        (np.radians(30), np.radians(120)),  # 30°N, 120°E
    ],
)
def test_geodetic_to_ecef_various_locations(lat, lon):
    """Test that conversion produces valid results for various locations."""
    alt = 0.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # Check that result is reasonable
    distance = np.linalg.norm(xyz)
    # Distance should be approximately Earth's radius
    assert distance > WGS84.semi_minor_axis
    assert distance < WGS84.semi_major_axis + 100  # Within 100m


# Tests for ecef_to_geodetic function


def test_ecef_to_geodetic_equator_prime_meridian():
    """Test conversion from ECEF at equator and prime meridian."""
    x, y, z = WGS84.semi_major_axis, 0.0, 0.0
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-9) == 0.0
    assert pytest.approx(lon, abs=1e-9) == 0.0
    assert pytest.approx(alt, abs=0.01) == 0.0


def test_ecef_to_geodetic_north_pole():
    """Test conversion from ECEF at North Pole."""
    x, y, z = 0.0, 0.0, WGS84.semi_minor_axis
    lat, _lon, alt = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-9) == np.pi / 2
    assert pytest.approx(alt, abs=0.01) == 0.0


def test_ecef_to_geodetic_south_pole():
    """Test conversion from ECEF at South Pole."""
    x, y, z = 0.0, 0.0, -WGS84.semi_minor_axis
    lat, _lon, alt = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-9) == -np.pi / 2
    assert pytest.approx(alt, abs=0.01) == 0.0


def test_ecef_to_geodetic_london():
    """Test conversion from ECEF at London."""
    # ECEF coordinates for Greenwich Observatory
    x, y, z = 3980574.2, -0.4, 4966894.1
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    # Expected geodetic coordinates
    expected_lat = np.radians(51.4769)
    expected_lon = np.radians(-0.0005)

    assert pytest.approx(lat, abs=1e-4) == expected_lat
    assert pytest.approx(lon, abs=1e-4) == expected_lon
    assert pytest.approx(alt, abs=100.0) == 0.0


def test_ecef_to_geodetic_with_altitude():
    """Test conversion with non-zero altitude."""
    x, y, z = WGS84.semi_major_axis + 1000.0, 0.0, 0.0
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-9) == 0.0
    assert pytest.approx(lon, abs=1e-9) == 0.0
    assert pytest.approx(alt, abs=0.01) == 1000.0


def test_ecef_to_geodetic_different_ellipsoid():
    """Test conversion with different ellipsoid."""
    x, y, z = 6378137.0, 0.0, 0.0

    lat1, lon1, alt1 = ecef_to_geodetic(x, y, z, ellipsoid=WGS84)
    lat2, lon2, alt2 = ecef_to_geodetic(x, y, z, ellipsoid=WGS72)

    # Same ECEF point interpreted with different ellipsoids
    # should give different altitudes
    assert lat1 == lat2  # Latitude same at equator
    assert lon1 == lon2  # Longitude same at prime meridian
    assert alt1 != alt2  # Altitude different due to different ellipsoid


def test_ecef_to_geodetic_near_z_axis():
    """Test conversion for points very close to Z-axis."""
    # Very small x, y (within 10 micrometers of Z-axis)
    x, y, z = 1e-12, 1e-12, 6356752.0
    lat, _lon, _alt = ecef_to_geodetic(x, y, z)

    # Should be very close to North Pole
    assert pytest.approx(lat, abs=1e-6) == np.pi / 2


# Tests for round-trip conversions between geodetic and ECEF


@pytest.mark.parametrize(
    "lat,lon,alt",
    [
        (0.0, 0.0, 0.0),  # Equator, sea level
        (np.radians(51.4769), np.radians(-0.0005), 0.0),  # London
        (np.radians(40.7128), np.radians(-74.0060), 10.0),  # New York
        (np.radians(-33.8688), np.radians(151.2093), 0.0),  # Sydney
        (np.radians(35.6762), np.radians(139.6503), 40.0),  # Tokyo
        (np.pi / 2, 0.0, 0.0),  # North Pole
        (-np.pi / 2, 0.0, 0.0),  # South Pole
        (np.radians(45), np.radians(90), 1000.0),  # 45°N, 90°E, 1km alt
    ],
)
def test_geodetic_ecef_round_trip(lat, lon, alt):
    """Test that geodetic -> ECEF -> geodetic preserves values."""
    # Forward conversion
    xyz = geodetic_to_ecef(lat, lon, alt)

    # Reverse conversion
    lat2, lon2, alt2 = ecef_to_geodetic(xyz[0], xyz[1], xyz[2])

    # Should recover original values within tolerance
    # Tolerance of ~2e-7 radians is ~1.2 cm on Earth's surface
    assert pytest.approx(lat2, abs=2e-7) == lat
    # Longitude is undefined at poles, so skip check there
    if abs(lat) < np.pi / 2 - 1e-6:
        # Normalize longitude to [-pi, pi]
        lon_normalized = np.arctan2(np.sin(lon), np.cos(lon))
        lon2_normalized = np.arctan2(np.sin(lon2), np.cos(lon2))
        assert pytest.approx(lon2_normalized, abs=1e-7) == lon_normalized
    # Altitude tolerance of 0.5m accounts for iterative algorithm convergence
    assert pytest.approx(alt2, abs=0.5) == alt


@pytest.mark.parametrize(
    "x,y,z",
    [
        (6378137.0, 0.0, 0.0),  # On equator
        (0.0, 6378137.0, 0.0),  # On equator, 90°E
        (0.0, 0.0, 6356752.314),  # North Pole
        (3980574.2, -0.4, 4966894.1),  # London
        (4000000.0, 3000000.0, 4000000.0),  # Arbitrary point
    ],
)
def test_ecef_geodetic_round_trip(x, y, z):
    """Test that ECEF -> geodetic -> ECEF preserves values."""
    # Forward conversion
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    # Reverse conversion
    xyz = geodetic_to_ecef(lat, lon, alt)

    # Should recover original values within tolerance
    # Tolerance of 1 meter is acceptable for coordinate transformations
    assert pytest.approx(xyz[0], abs=1.0) == x
    assert pytest.approx(xyz[1], abs=1.0) == y
    assert pytest.approx(xyz[2], abs=1.0) == z


# Tests for ECI-ECEF transformations


def test_eci_to_ecef_j2000():
    """Test ECI to ECEF at J2000 epoch."""
    # At J2000 epoch (2000-01-01 12:00:00 UTC)
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Point at Earth's equatorial radius in ECI
    eci = np.array([6378137.0, 0.0, 0.0])
    ecef = eci_to_ecef(eci, timestamp)

    # At J2000 epoch, the rotation should be specific
    # (not necessarily identity due to Earth rotation)
    assert ecef.shape == (3,)
    # Check that magnitude is preserved
    assert pytest.approx(np.linalg.norm(ecef)) == np.linalg.norm(eci)


def test_ecef_to_eci_j2000():
    """Test ECEF to ECI at J2000 epoch."""
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Point at Earth's equatorial radius in ECEF
    ecef = np.array([6378137.0, 0.0, 0.0])
    eci = ecef_to_eci(ecef, timestamp)

    assert eci.shape == (3,)
    # Check that magnitude is preserved
    assert pytest.approx(np.linalg.norm(eci)) == np.linalg.norm(ecef)


@pytest.mark.parametrize(
    "eci_pos",
    [
        np.array([6378137.0, 0.0, 0.0]),
        np.array([0.0, 6378137.0, 0.0]),
        np.array([0.0, 0.0, 6378137.0]),
        np.array([4000000.0, 3000000.0, 4000000.0]),
    ],
)
def test_eci_ecef_round_trip(eci_pos):
    """Test that ECI -> ECEF -> ECI preserves values."""
    timestamp = datetime.datetime(2024, 6, 15, 18, 30, 0)

    ecef = eci_to_ecef(eci_pos, timestamp)
    eci_back = ecef_to_eci(ecef, timestamp)

    assert pytest.approx(eci_back[0], abs=1e-6) == eci_pos[0]
    assert pytest.approx(eci_back[1], abs=1e-6) == eci_pos[1]
    assert pytest.approx(eci_back[2], abs=1e-6) == eci_pos[2]


@pytest.mark.parametrize(
    "ecef_pos",
    [
        np.array([6378137.0, 0.0, 0.0]),
        np.array([0.0, 6378137.0, 0.0]),
        np.array([0.0, 0.0, 6378137.0]),
        np.array([4000000.0, 3000000.0, 4000000.0]),
    ],
)
def test_ecef_eci_round_trip(ecef_pos):
    """Test that ECEF -> ECI -> ECEF preserves values."""
    timestamp = datetime.datetime(2024, 6, 15, 18, 30, 0)

    eci = ecef_to_eci(ecef_pos, timestamp)
    ecef_back = eci_to_ecef(eci, timestamp)

    assert pytest.approx(ecef_back[0], abs=1e-6) == ecef_pos[0]
    assert pytest.approx(ecef_back[1], abs=1e-6) == ecef_pos[1]
    assert pytest.approx(ecef_back[2], abs=1e-6) == ecef_pos[2]


@pytest.mark.parametrize(
    "pos",
    [
        np.array([7000000.0, 0.0, 0.0]),
        np.array([0.0, 7000000.0, 0.0]),
        np.array([0.0, 0.0, 7000000.0]),
        np.array([4000000.0, 3000000.0, 5000000.0]),
    ],
)
def test_eci_ecef_rotation_preserves_magnitude(pos):
    """Test that ECI-ECEF transformations preserve vector magnitude."""
    timestamp = datetime.datetime(2024, 1, 1, 0, 0, 0)

    orig_mag = np.linalg.norm(pos)

    # ECI to ECEF should preserve magnitude
    ecef = eci_to_ecef(pos, timestamp)
    assert pytest.approx(np.linalg.norm(ecef), rel=1e-10) == orig_mag

    # ECEF to ECI should preserve magnitude
    eci = ecef_to_eci(pos, timestamp)
    assert pytest.approx(np.linalg.norm(eci), rel=1e-10) == orig_mag


@pytest.mark.parametrize(
    "z_pos",
    [
        np.array([0.0, 0.0, 6378137.0]),
        np.array([0.0, 0.0, -6378137.0]),
    ],
)
def test_eci_ecef_z_axis_unchanged(z_pos):
    """Test that points on Z-axis are unchanged by rotation."""
    timestamp = datetime.datetime(2024, 1, 1, 0, 0, 0)

    # Points on Z-axis should be unchanged (rotation is about Z-axis)
    ecef = eci_to_ecef(z_pos, timestamp)
    assert pytest.approx(ecef[0], abs=1e-10) == z_pos[0]
    assert pytest.approx(ecef[1], abs=1e-10) == z_pos[1]
    assert pytest.approx(ecef[2], abs=1e-10) == z_pos[2]


def test_eci_ecef_earth_rotation_effect():
    """Test that Earth rotation is accounted for over time."""
    # Two timestamps 6 hours apart
    time1 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    time2 = datetime.datetime(2024, 1, 1, 6, 0, 0)

    # Same ECI position
    eci = np.array([6378137.0, 0.0, 0.0])

    # Convert to ECEF at different times
    ecef1 = eci_to_ecef(eci, time1)
    ecef2 = eci_to_ecef(eci, time2)

    # ECEF coordinates should be different due to Earth rotation
    assert not np.allclose(ecef1, ecef2)
    # But magnitude should be the same
    assert pytest.approx(np.linalg.norm(ecef1)) == np.linalg.norm(ecef2)


# Tests for convenience functions


def test_geodetic_to_eci():
    """Test geodetic to ECI conversion."""
    lat = np.radians(51.4769)
    lon = np.radians(-0.0005)
    alt = 0.0
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    eci = geodetic_to_eci(lat, lon, alt, timestamp)

    # Verify it's equivalent to chaining geodetic_to_ecef and ecef_to_eci
    ecef = geodetic_to_ecef(lat, lon, alt)
    eci_expected = ecef_to_eci(ecef, timestamp)

    assert pytest.approx(eci[0], abs=1e-6) == eci_expected[0]
    assert pytest.approx(eci[1], abs=1e-6) == eci_expected[1]
    assert pytest.approx(eci[2], abs=1e-6) == eci_expected[2]


def test_eci_to_geodetic():
    """Test ECI to geodetic conversion."""
    x, y, z = 3980574.2, -0.4, 4966894.1
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    lat, lon, alt = eci_to_geodetic(x, y, z, timestamp)

    # Verify it's equivalent to chaining eci_to_ecef and ecef_to_geodetic
    eci = np.array([x, y, z])
    ecef = eci_to_ecef(eci, timestamp)
    lat_expected, lon_expected, alt_expected = ecef_to_geodetic(ecef[0], ecef[1], ecef[2])

    assert pytest.approx(lat, abs=1e-9) == lat_expected
    assert pytest.approx(lon, abs=1e-9) == lon_expected
    assert pytest.approx(alt, abs=0.01) == alt_expected


def test_geodetic_eci_round_trip():
    """Test geodetic -> ECI -> geodetic round trip."""
    lat = np.radians(40.7128)
    lon = np.radians(-74.0060)
    alt = 10.0
    timestamp = datetime.datetime(2024, 6, 15, 18, 30, 0)

    # Forward and back
    eci = geodetic_to_eci(lat, lon, alt, timestamp)
    lat2, lon2, alt2 = eci_to_geodetic(eci[0], eci[1], eci[2], timestamp)

    assert pytest.approx(lat2, abs=1e-7) == lat
    # Normalize longitudes
    lon_normalized = np.arctan2(np.sin(lon), np.cos(lon))
    lon2_normalized = np.arctan2(np.sin(lon2), np.cos(lon2))
    assert pytest.approx(lon2_normalized, abs=1e-7) == lon_normalized
    assert pytest.approx(alt2, abs=0.5) == alt


# Tests for edge cases


def test_geodetic_to_ecef_negative_altitude():
    """Test geodetic to ECEF with negative altitude (below ellipsoid)."""
    lat, lon, alt = 0.0, 0.0, -1000.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # X should be semi_major_axis - 1000
    expected_x = WGS84.semi_major_axis - 1000.0
    assert pytest.approx(xyz[0], abs=0.01) == expected_x


def test_ecef_to_geodetic_high_altitude():
    """Test ECEF to geodetic with very high altitude."""
    # Point at 400km altitude (ISS orbit)
    lat, lon, alt = 0.0, 0.0, 400000.0
    xyz = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_geodetic(xyz[0], xyz[1], xyz[2])

    assert pytest.approx(lat2, abs=1e-9) == lat
    assert pytest.approx(lon2, abs=1e-9) == lon
    assert pytest.approx(alt2, abs=1.0) == alt


def test_geodetic_to_ecef_extreme_latitudes():
    """Test geodetic to ECEF at extreme latitudes."""
    # Just below North Pole
    lat = np.pi / 2 - 1e-6
    lon = 0.0
    alt = 0.0
    xyz = geodetic_to_ecef(lat, lon, alt)

    # X and Y should be very small, Z should be close to semi_minor_axis
    assert abs(xyz[0]) < 10.0
    assert abs(xyz[1]) < 10.0
    assert pytest.approx(xyz[2], abs=1.0) == WGS84.semi_minor_axis


def test_longitude_wrapping():
    """Test that longitude values outside [-π, π] work correctly."""
    lat = 0.0
    alt = 0.0

    # Test longitude wrapping
    lon1 = np.radians(90)
    lon2 = np.radians(450)  # Same as 90°

    xyz1 = geodetic_to_ecef(lat, lon1, alt)
    xyz2 = geodetic_to_ecef(lat, lon2, alt)

    # Results should be very close
    assert pytest.approx(xyz1[0], abs=0.01) == xyz2[0]
    assert pytest.approx(xyz1[1], abs=0.01) == xyz2[1]
    assert pytest.approx(xyz1[2], abs=0.01) == xyz2[2]


def test_ecef_to_geodetic_convergence_tolerance():
    """Test that ecef_to_geodetic respects convergence tolerance."""
    x, y, z = 4000000.0, 3000000.0, 4000000.0

    # Test with different tolerances
    lat1, lon1, alt1 = ecef_to_geodetic(x, y, z, tolerance=1e-12)
    lat2, lon2, alt2 = ecef_to_geodetic(x, y, z, tolerance=1e-6)

    # Results should be very close but potentially different
    assert pytest.approx(lat1, abs=1e-5) == lat2
    assert pytest.approx(lon1, abs=1e-9) == lon2
    assert pytest.approx(alt1, abs=0.1) == alt2


# Tests for latitude type conversions
from stonesoup.functions.coordinates import (
    geocentric_to_geodetic_latitude,
    geodetic_to_geocentric_latitude,
    geodetic_to_parametric_latitude,
    parametric_to_geodetic_latitude,
)


def test_geodetic_to_geocentric_equator():
    """Test geodetic to geocentric conversion at equator."""
    # At equator, geodetic and geocentric latitude are both 0
    geodetic_lat = 0.0
    geocentric_lat = geodetic_to_geocentric_latitude(geodetic_lat)
    assert pytest.approx(geocentric_lat, abs=1e-15) == 0.0


def test_geodetic_to_geocentric_poles():
    """Test geodetic to geocentric conversion at poles."""
    # At poles, geodetic and geocentric latitude are identical
    # North pole
    geocentric_north = geodetic_to_geocentric_latitude(np.pi / 2)
    assert pytest.approx(geocentric_north, abs=1e-10) == np.pi / 2

    # South pole
    geocentric_south = geodetic_to_geocentric_latitude(-np.pi / 2)
    assert pytest.approx(geocentric_south, abs=1e-10) == -np.pi / 2


def test_geodetic_to_geocentric_45_degrees():
    """Test geodetic to geocentric conversion at 45°."""
    # Maximum difference is around 45° latitude
    geodetic_lat = np.radians(45.0)
    geocentric_lat = geodetic_to_geocentric_latitude(geodetic_lat)

    # Geocentric should be less than geodetic at mid-latitudes
    assert geocentric_lat < geodetic_lat

    # Known value: difference is about 11.5 arcminutes at 45°
    diff_arcmin = np.degrees(geodetic_lat - geocentric_lat) * 60
    assert pytest.approx(diff_arcmin, abs=0.1) == 11.54


def test_geodetic_to_geocentric_various_latitudes():
    """Test geodetic to geocentric conversion at various latitudes."""
    # Geocentric latitude is always smaller in magnitude than geodetic
    # (except at equator and poles)
    for lat_deg in [10, 20, 30, 40, 50, 60, 70, 80]:
        geodetic_lat = np.radians(lat_deg)
        geocentric_lat = geodetic_to_geocentric_latitude(geodetic_lat)

        # Geocentric should be less than geodetic in northern hemisphere
        assert geocentric_lat < geodetic_lat
        # But still positive
        assert geocentric_lat > 0

        # Test southern hemisphere (symmetric)
        geodetic_lat_south = np.radians(-lat_deg)
        geocentric_lat_south = geodetic_to_geocentric_latitude(geodetic_lat_south)

        # Should be symmetric
        assert pytest.approx(geocentric_lat_south, abs=1e-10) == -geocentric_lat


def test_geocentric_to_geodetic_round_trip():
    """Test that geocentric -> geodetic -> geocentric is identity."""
    for lat_deg in [0, 15, 30, 45, 60, 75, 89]:
        geodetic_lat = np.radians(lat_deg)
        geocentric_lat = geodetic_to_geocentric_latitude(geodetic_lat)
        recovered = geocentric_to_geodetic_latitude(geocentric_lat)

        assert pytest.approx(recovered, abs=1e-14) == geodetic_lat

        # Test negative latitude
        geodetic_lat_south = np.radians(-lat_deg)
        geocentric_lat_south = geodetic_to_geocentric_latitude(geodetic_lat_south)
        recovered_south = geocentric_to_geodetic_latitude(geocentric_lat_south)

        assert pytest.approx(recovered_south, abs=1e-14) == geodetic_lat_south


def test_geocentric_to_geodetic_equator():
    """Test geocentric to geodetic conversion at equator."""
    geocentric_lat = 0.0
    geodetic_lat = geocentric_to_geodetic_latitude(geocentric_lat)
    assert pytest.approx(geodetic_lat, abs=1e-15) == 0.0


def test_geocentric_to_geodetic_poles():
    """Test geocentric to geodetic conversion at poles."""
    # North pole
    geodetic_north = geocentric_to_geodetic_latitude(np.pi / 2)
    assert pytest.approx(geodetic_north, abs=1e-10) == np.pi / 2

    # South pole
    geodetic_south = geocentric_to_geodetic_latitude(-np.pi / 2)
    assert pytest.approx(geodetic_south, abs=1e-10) == -np.pi / 2


def test_geodetic_geocentric_different_ellipsoid():
    """Test geodetic/geocentric conversion with different ellipsoid."""
    geodetic_lat = np.radians(45.0)

    geocentric_wgs84 = geodetic_to_geocentric_latitude(geodetic_lat, ellipsoid=WGS84)
    geocentric_grs80 = geodetic_to_geocentric_latitude(geodetic_lat, ellipsoid=GRS80)

    # WGS84 and GRS80 have very similar eccentricities, so results should be close
    assert pytest.approx(geocentric_wgs84, abs=1e-8) == geocentric_grs80

    # WGS72 has different eccentricity, so result should be slightly different
    geocentric_wgs72 = geodetic_to_geocentric_latitude(geodetic_lat, ellipsoid=WGS72)
    # Small but noticeable difference
    assert pytest.approx(geocentric_wgs84, abs=1e-09) != geocentric_wgs72


def test_geodetic_to_parametric_equator():
    """Test geodetic to parametric conversion at equator."""
    geodetic_lat = 0.0
    parametric_lat = geodetic_to_parametric_latitude(geodetic_lat)
    assert pytest.approx(parametric_lat, abs=1e-15) == 0.0


def test_geodetic_to_parametric_poles():
    """Test geodetic to parametric conversion at poles."""
    # North pole
    parametric_north = geodetic_to_parametric_latitude(np.pi / 2)
    assert pytest.approx(parametric_north, abs=1e-10) == np.pi / 2

    # South pole
    parametric_south = geodetic_to_parametric_latitude(-np.pi / 2)
    assert pytest.approx(parametric_south, abs=1e-10) == -np.pi / 2


def test_geodetic_to_parametric_45_degrees():
    """Test geodetic to parametric conversion at 45°."""
    geodetic_lat = np.radians(45.0)
    parametric_lat = geodetic_to_parametric_latitude(geodetic_lat)

    # Parametric latitude should be less than geodetic at mid-latitudes
    assert parametric_lat < geodetic_lat

    # Difference should be smaller than geodetic-geocentric difference
    diff_geodetic_parametric = np.degrees(geodetic_lat - parametric_lat) * 60
    # Known value: about 5.76 arcminutes at 45°
    assert pytest.approx(diff_geodetic_parametric, abs=0.1) == 5.78


def test_parametric_to_geodetic_round_trip():
    """Test that parametric -> geodetic -> parametric is identity."""
    for lat_deg in [0, 15, 30, 45, 60, 75, 89]:
        geodetic_lat = np.radians(lat_deg)
        parametric_lat = geodetic_to_parametric_latitude(geodetic_lat)
        recovered = parametric_to_geodetic_latitude(parametric_lat)

        assert pytest.approx(recovered, abs=1e-14) == geodetic_lat


def test_latitude_ordering():
    """Test that latitude types are ordered correctly."""
    # For a given point (not at equator or poles):
    # geocentric < parametric < geodetic (for positive latitudes)
    geodetic_lat = np.radians(45.0)

    geocentric_lat = geodetic_to_geocentric_latitude(geodetic_lat)
    parametric_lat = geodetic_to_parametric_latitude(geodetic_lat)

    assert geocentric_lat < parametric_lat < geodetic_lat

    # Test negative latitude (ordering reversed)
    geodetic_lat_south = np.radians(-45.0)
    geocentric_lat_south = geodetic_to_geocentric_latitude(geodetic_lat_south)
    parametric_lat_south = geodetic_to_parametric_latitude(geodetic_lat_south)

    assert geocentric_lat_south > parametric_lat_south > geodetic_lat_south


# Tests for relative motion frames
from stonesoup.types.coordinates import (
    LVLHFrame,
    RICFrame,
    RSWFrame,
    compute_relative_state,
)


@pytest.fixture
def circular_orbit_state():
    """Reference state for a circular equatorial orbit at ~7000 km altitude."""
    # Position on x-axis, velocity along y-axis (circular orbit)
    ref_pos = np.array([7000000.0, 0.0, 0.0])  # 7000 km
    ref_vel = np.array([0.0, 7546.0, 0.0])  # ~7.5 km/s for circular orbit
    return ref_pos, ref_vel


@pytest.fixture
def inclined_orbit_state():
    """Reference state for an inclined orbit."""
    # 45-degree inclination orbit
    ref_pos = np.array([5000000.0, 3000000.0, 2000000.0])
    ref_vel = np.array([-2000.0, 5000.0, 3000.0])
    return ref_pos, ref_vel


def test_ric_frame_instantiation(circular_orbit_state):
    """Test RIC frame can be instantiated."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    assert ric.name == "RIC"
    assert np.allclose(ric.reference_position, ref_pos)
    assert np.allclose(ric.reference_velocity, ref_vel)


def test_ric_frame_rotation_matrix_orthogonal(circular_orbit_state):
    """Test that RIC rotation matrix is orthogonal."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    R = ric.rotation_matrix

    # R * R^T should be identity
    identity = R @ R.T
    assert np.allclose(identity, np.eye(3), atol=1e-10)

    # Determinant should be 1 (proper rotation)
    assert pytest.approx(np.linalg.det(R), abs=1e-10) == 1.0


def test_ric_frame_circular_orbit(circular_orbit_state):
    """Test RIC frame for circular equatorial orbit."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    R = ric.rotation_matrix

    # For circular equatorial orbit with r along x and v along y:
    # R-axis (radial) should be along x
    # I-axis (in-track) should be along y
    # C-axis (cross-track) should be along z

    # R-axis = first row = [1, 0, 0]
    assert np.allclose(R[0], [1, 0, 0], atol=1e-10)
    # I-axis = second row = [0, 1, 0]
    assert np.allclose(R[1], [0, 1, 0], atol=1e-10)
    # C-axis = third row = [0, 0, 1]
    assert np.allclose(R[2], [0, 0, 1], atol=1e-10)


def test_ric_inertial_to_relative_same_position(circular_orbit_state):
    """Test that reference position transforms to origin in RIC."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    # Reference position should transform to origin
    rel_pos, rel_vel = ric.inertial_to_relative(ref_pos, ref_vel)

    assert np.allclose(rel_pos, [0, 0, 0], atol=1e-10)
    assert np.allclose(rel_vel, [0, 0, 0], atol=1e-10)


def test_ric_inertial_to_relative_radial_offset(circular_orbit_state):
    """Test RIC frame with radial offset."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    # Chaser 1000m ahead in radial direction (larger orbit radius)
    chaser_pos = np.array([7001000.0, 0.0, 0.0])
    chaser_vel = ref_vel.copy()

    rel_pos, _ = ric.inertial_to_relative(chaser_pos, chaser_vel)

    # Should have positive R (radial), zero I and C
    assert pytest.approx(rel_pos[0], abs=1.0) == 1000.0  # R = +1000m
    assert pytest.approx(rel_pos[1], abs=1.0) == 0.0  # I = 0
    assert pytest.approx(rel_pos[2], abs=1.0) == 0.0  # C = 0


def test_ric_inertial_to_relative_intrack_offset(circular_orbit_state):
    """Test RIC frame with in-track offset."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    # Chaser 500m ahead in in-track direction
    chaser_pos = np.array([7000000.0, 500.0, 0.0])
    chaser_vel = ref_vel.copy()

    rel_pos, _ = ric.inertial_to_relative(chaser_pos, chaser_vel)

    # Should have positive I (in-track), zero R and C
    assert pytest.approx(rel_pos[0], abs=1.0) == 0.0  # R = 0
    assert pytest.approx(rel_pos[1], abs=1.0) == 500.0  # I = +500m
    assert pytest.approx(rel_pos[2], abs=1.0) == 0.0  # C = 0


def test_ric_inertial_to_relative_crosstrack_offset(circular_orbit_state):
    """Test RIC frame with cross-track offset."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    # Chaser 200m above in cross-track direction
    chaser_pos = np.array([7000000.0, 0.0, 200.0])
    chaser_vel = ref_vel.copy()

    rel_pos, _ = ric.inertial_to_relative(chaser_pos, chaser_vel)

    # Should have positive C (cross-track), zero R and I
    assert pytest.approx(rel_pos[0], abs=1.0) == 0.0  # R = 0
    assert pytest.approx(rel_pos[1], abs=1.0) == 0.0  # I = 0
    assert pytest.approx(rel_pos[2], abs=1.0) == 200.0  # C = +200m


def test_ric_round_trip(circular_orbit_state):
    """Test RIC inertial→relative→inertial round trip."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    # Arbitrary chaser position and velocity
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    # Transform to relative
    rel_pos, rel_vel = ric.inertial_to_relative(chaser_pos, chaser_vel)

    # Transform back to inertial
    recovered_pos, recovered_vel = ric.relative_to_inertial(rel_pos, rel_vel)

    assert np.allclose(recovered_pos, chaser_pos, atol=1e-6)
    assert np.allclose(recovered_vel, chaser_vel, atol=1e-6)


def test_rsw_frame_instantiation(circular_orbit_state):
    """Test RSW frame can be instantiated."""
    ref_pos, ref_vel = circular_orbit_state
    rsw = RSWFrame(name="RSW", reference_position=ref_pos, reference_velocity=ref_vel)

    assert rsw.name == "RSW"


def test_rsw_same_as_ric_for_circular(circular_orbit_state):
    """Test that RSW equals RIC for circular orbit."""
    ref_pos, ref_vel = circular_orbit_state

    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)
    rsw = RSWFrame(name="RSW", reference_position=ref_pos, reference_velocity=ref_vel)

    # For circular orbit, RIC and RSW should be identical
    assert np.allclose(ric.rotation_matrix, rsw.rotation_matrix, atol=1e-10)


def test_lvlh_frame_instantiation(circular_orbit_state):
    """Test LVLH frame can be instantiated."""
    ref_pos, ref_vel = circular_orbit_state
    lvlh = LVLHFrame(name="LVLH", reference_position=ref_pos, reference_velocity=ref_vel)

    assert lvlh.name == "LVLH"


def test_lvlh_frame_z_nadir(circular_orbit_state):
    """Test that LVLH Z-axis points toward Earth (nadir)."""
    ref_pos, ref_vel = circular_orbit_state
    lvlh = LVLHFrame(name="LVLH", reference_position=ref_pos, reference_velocity=ref_vel)

    R = lvlh.rotation_matrix
    z_axis = R[2]  # Third row

    # Z should point toward Earth center (opposite to position)
    expected_z = -ref_pos / np.linalg.norm(ref_pos)
    assert np.allclose(z_axis, expected_z, atol=1e-10)


def test_lvlh_round_trip(circular_orbit_state):
    """Test LVLH inertial→relative→inertial round trip."""
    ref_pos, ref_vel = circular_orbit_state
    lvlh = LVLHFrame(name="LVLH", reference_position=ref_pos, reference_velocity=ref_vel)

    # Arbitrary chaser position and velocity
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    # Transform to relative
    rel_pos, rel_vel = lvlh.inertial_to_relative(chaser_pos, chaser_vel)

    # Transform back to inertial
    recovered_pos, recovered_vel = lvlh.relative_to_inertial(rel_pos, rel_vel)

    assert np.allclose(recovered_pos, chaser_pos, atol=1e-6)
    assert np.allclose(recovered_vel, chaser_vel, atol=1e-6)


def test_compute_relative_state_ric(circular_orbit_state):
    """Test compute_relative_state convenience function with RIC."""
    ref_pos, ref_vel = circular_orbit_state
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    rel_pos, rel_vel = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="RIC"
    )

    # Verify by using RICFrame directly
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)
    expected_pos, expected_vel = ric.inertial_to_relative(chaser_pos, chaser_vel)

    assert np.allclose(rel_pos, expected_pos, atol=1e-10)
    assert np.allclose(rel_vel, expected_vel, atol=1e-10)


def test_compute_relative_state_rsw(circular_orbit_state):
    """Test compute_relative_state with RSW."""
    ref_pos, ref_vel = circular_orbit_state
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    rel_pos, rel_vel = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="RSW"
    )

    assert rel_pos is not None
    assert rel_vel is not None


def test_compute_relative_state_lvlh(circular_orbit_state):
    """Test compute_relative_state with LVLH."""
    ref_pos, ref_vel = circular_orbit_state
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    rel_pos, rel_vel = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="LVLH"
    )

    assert rel_pos is not None
    assert rel_vel is not None


def test_compute_relative_state_invalid_frame():
    """Test compute_relative_state with invalid frame type."""
    with pytest.raises(ValueError, match="Unknown frame type"):
        compute_relative_state(
            np.array([7000000.0, 0.0, 0.0]),
            np.array([0.0, 7546.0, 0.0]),
            np.array([7001000.0, 0.0, 0.0]),
            np.array([0.0, 7546.0, 0.0]),
            frame_type="INVALID",
        )


def test_compute_relative_state_case_insensitive(circular_orbit_state):
    """Test that frame_type is case insensitive."""
    ref_pos, ref_vel = circular_orbit_state
    chaser_pos = np.array([7001000.0, 500.0, 200.0])
    chaser_vel = np.array([10.0, 7550.0, 5.0])

    # Should work with different cases
    rel_pos1, _ = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="ric"
    )
    rel_pos2, _ = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="RIC"
    )
    rel_pos3, _ = compute_relative_state(
        ref_pos, ref_vel, chaser_pos, chaser_vel, frame_type="Ric"
    )

    assert np.allclose(rel_pos1, rel_pos2, atol=1e-10)
    assert np.allclose(rel_pos2, rel_pos3, atol=1e-10)


def test_ric_inclined_orbit(inclined_orbit_state):
    """Test RIC frame with inclined orbit."""
    ref_pos, ref_vel = inclined_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    R = ric.rotation_matrix

    # R should be orthogonal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    # Reference position should transform to origin
    rel_pos, _ = ric.inertial_to_relative(ref_pos, ref_vel)
    assert np.allclose(rel_pos, [0, 0, 0], atol=1e-10)


def test_relative_frame_position_only(circular_orbit_state):
    """Test transformation with position only (no velocity)."""
    ref_pos, ref_vel = circular_orbit_state
    ric = RICFrame(name="RIC", reference_position=ref_pos, reference_velocity=ref_vel)

    chaser_pos = np.array([7001000.0, 500.0, 200.0])

    rel_pos, rel_vel = ric.inertial_to_relative(chaser_pos, None)

    assert rel_pos is not None
    assert rel_vel is None


def test_relative_frame_transform_to_same_type(circular_orbit_state):
    """Test transform_to between two RIC frames."""
    ref_pos1, ref_vel1 = circular_orbit_state

    # Second reference object slightly offset
    ref_pos2 = np.array([7000000.0, 1000.0, 0.0])
    ref_vel2 = np.array([0.0, 7546.0, 0.0])

    ric1 = RICFrame(name="RIC1", reference_position=ref_pos1, reference_velocity=ref_vel1)
    ric2 = RICFrame(name="RIC2", reference_position=ref_pos2, reference_velocity=ref_vel2)

    # Position in first RIC frame
    pos_in_ric1 = np.array([100.0, 50.0, 20.0])

    # Transform to second RIC frame
    pos_in_ric2, _ = ric1.transform_to(ric2, pos_in_ric1)

    # Verify by manual calculation: RIC1 → inertial → RIC2
    inertial_pos, _ = ric1.relative_to_inertial(pos_in_ric1)
    expected_pos, _ = ric2.inertial_to_relative(inertial_pos)

    assert np.allclose(pos_in_ric2, expected_pos, atol=1e-6)


# =============================================================================
# Proximity Operations Tests
# =============================================================================

from ..coordinates import (
    compute_closest_approach,
    compute_conjunction_geometry,
    compute_miss_distance,
    compute_range,
    compute_range_rate,
    is_in_ellipsoidal_keep_out_zone,
    is_in_keep_out_zone,
)


def test_compute_range_basic():
    """Test range computation between two points."""
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])

    r = compute_range(pos1, pos2)
    assert pytest.approx(r, abs=1e-10) == 1000.0


def test_compute_range_diagonal():
    """Test range computation along diagonal."""
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 1000.0, 1000.0])

    r = compute_range(pos1, pos2)
    expected = np.sqrt(3) * 1000.0
    assert pytest.approx(r, abs=1e-10) == expected


def test_compute_range_symmetric():
    """Test range is symmetric (order doesn't matter)."""
    pos1 = np.array([100.0, 200.0, 300.0])
    pos2 = np.array([400.0, 500.0, 600.0])

    r1 = compute_range(pos1, pos2)
    r2 = compute_range(pos2, pos1)
    assert pytest.approx(r1, abs=1e-15) == r2


def test_compute_range_rate_approaching():
    """Test range rate for approaching objects."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([-10.0, 0.0, 0.0])  # Moving toward origin

    rr = compute_range_rate(pos1, vel1, pos2, vel2)
    assert pytest.approx(rr, abs=1e-10) == -10.0  # Negative = closing


def test_compute_range_rate_separating():
    """Test range rate for separating objects."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([10.0, 0.0, 0.0])  # Moving away from origin

    rr = compute_range_rate(pos1, vel1, pos2, vel2)
    assert pytest.approx(rr, abs=1e-10) == 10.0  # Positive = separating


def test_compute_range_rate_perpendicular():
    """Test range rate when velocity is perpendicular to line of sight."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([0.0, 10.0, 0.0])  # Moving perpendicular

    rr = compute_range_rate(pos1, vel1, pos2, vel2)
    assert pytest.approx(rr, abs=1e-10) == 0.0


def test_compute_range_rate_coincident():
    """Test range rate when objects are coincident."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([1.0, 0.0, 0.0])
    pos2 = np.array([0.0, 0.0, 0.0])
    vel2 = np.array([2.0, 0.0, 0.0])

    rr = compute_range_rate(pos1, vel1, pos2, vel2)
    assert rr == 0.0  # Should handle divide by zero


def test_compute_closest_approach_simple():
    """Test closest approach for simple case."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 100.0, 0.0])  # Offset in y
    vel2 = np.array([-10.0, 0.0, 0.0])  # Moving toward x=0

    t_ca, d_ca = compute_closest_approach(pos1, vel1, pos2, vel2)

    # Should reach x=0 at t=100s, at that point y=100m
    assert pytest.approx(t_ca, abs=0.1) == 100.0
    assert pytest.approx(d_ca, abs=0.1) == 100.0


def test_compute_closest_approach_stationary():
    """Test closest approach for stationary objects."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([0.0, 0.0, 0.0])

    t_ca, d_ca = compute_closest_approach(pos1, vel1, pos2, vel2)

    assert t_ca == 0.0
    assert pytest.approx(d_ca, abs=1e-10) == 1000.0


def test_compute_closest_approach_past():
    """Test closest approach when it's in the past (separating)."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([10.0, 0.0, 0.0])  # Moving away

    t_ca, d_ca = compute_closest_approach(pos1, vel1, pos2, vel2)

    assert t_ca == 0.0  # Closest approach is now
    assert pytest.approx(d_ca, abs=1e-10) == 1000.0


def test_compute_closest_approach_max_time():
    """Test closest approach when beyond max_time."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1e9, 100.0, 0.0])  # Very far away
    vel2 = np.array([-1.0, 0.0, 0.0])  # Slow approach

    t_ca, _d_ca = compute_closest_approach(pos1, vel1, pos2, vel2, max_time=1000.0)

    assert t_ca == 1000.0  # Clamped to max_time


def test_compute_miss_distance_parallel():
    """Test miss distance for parallel trajectories."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([10.0, 0.0, 0.0])
    pos2 = np.array([0.0, 100.0, 0.0])
    vel2 = np.array([10.0, 0.0, 0.0])

    d = compute_miss_distance(pos1, vel1, pos2, vel2)
    assert pytest.approx(d, abs=1e-10) == 100.0


def test_is_in_keep_out_zone_inside():
    """Test keep-out zone check when inside."""
    pos = np.array([50.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])

    assert is_in_keep_out_zone(pos, center, 100.0) is True


def test_is_in_keep_out_zone_outside():
    """Test keep-out zone check when outside."""
    pos = np.array([150.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])

    assert is_in_keep_out_zone(pos, center, 100.0) is False


def test_is_in_keep_out_zone_boundary():
    """Test keep-out zone check at boundary."""
    pos = np.array([100.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])

    # At exact boundary, should be False (< not <=)
    assert is_in_keep_out_zone(pos, center, 100.0) is False


def test_is_in_ellipsoidal_keep_out_zone_inside_x():
    """Test ellipsoidal zone when inside along x."""
    pos = np.array([80.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    semi_axes = np.array([100.0, 50.0, 50.0])

    assert is_in_ellipsoidal_keep_out_zone(pos, center, semi_axes)


def test_is_in_ellipsoidal_keep_out_zone_outside_y():
    """Test ellipsoidal zone when outside along shorter axis."""
    pos = np.array([0.0, 80.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    semi_axes = np.array([100.0, 50.0, 50.0])

    assert not is_in_ellipsoidal_keep_out_zone(pos, center, semi_axes)


def test_is_in_ellipsoidal_keep_out_zone_rotated():
    """Test ellipsoidal zone with rotation."""
    # 90 degree rotation about z-axis: x->y, y->-x
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([0.0, 0.0, 0.0])
    semi_axes = np.array([100.0, 50.0, 50.0])  # Long axis along x before rotation

    # After rotation, long axis is along y
    # Point along y at 80 should be inside
    pos_y = np.array([0.0, 80.0, 0.0])
    assert is_in_ellipsoidal_keep_out_zone(pos_y, center, semi_axes, R)

    # Point along x at 80 should be outside
    pos_x = np.array([80.0, 0.0, 0.0])
    assert not is_in_ellipsoidal_keep_out_zone(pos_x, center, semi_axes, R)


def test_compute_conjunction_geometry_basic():
    """Test basic conjunction geometry computation."""
    pos1 = np.array([7000000.0, 0.0, 0.0])
    vel1 = np.array([0.0, 7500.0, 0.0])
    pos2 = np.array([7001000.0, 500.0, 0.0])
    vel2 = np.array([0.0, 7500.0, 100.0])

    geometry = compute_conjunction_geometry(pos1, vel1, pos2, vel2)

    assert "range" in geometry
    assert "range_rate" in geometry
    assert "time_to_closest_approach" in geometry
    assert "miss_distance" in geometry
    assert "relative_velocity_magnitude" in geometry
    assert "approach_angle" in geometry

    # Range should be roughly 1000m
    assert geometry["range"] > 1000.0
    assert geometry["range"] < 1200.0


def test_compute_conjunction_geometry_with_covariance():
    """Test conjunction geometry with covariance matrices."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 0.0, 0.0])
    vel2 = np.array([0.0, 0.0, 0.0])

    cov1 = np.eye(3) * 100.0  # 10m sigma
    cov2 = np.eye(3) * 100.0

    geometry = compute_conjunction_geometry(pos1, vel1, pos2, vel2, cov1, cov2)

    assert "combined_covariance" in geometry
    assert "mahalanobis_distance" in geometry

    # Combined covariance should be sum
    expected_cov = cov1 + cov2
    assert np.allclose(geometry["combined_covariance"], expected_cov)

    # Mahalanobis distance should be range / sqrt(combined variance)
    # For isotropic case: sqrt(200) per axis
    expected_mahal = 1000.0 / np.sqrt(200.0)
    assert pytest.approx(geometry["mahalanobis_distance"], rel=1e-10) == expected_mahal


def test_compute_conjunction_geometry_approaching():
    """Test conjunction geometry for approaching objects."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1000.0, 100.0, 0.0])
    vel2 = np.array([-100.0, 0.0, 0.0])

    geometry = compute_conjunction_geometry(pos1, vel1, pos2, vel2)

    # Should be closing
    assert geometry["range_rate"] < 0

    # Time to closest approach should be ~10 seconds
    assert geometry["time_to_closest_approach"] > 0
    assert geometry["time_to_closest_approach"] < 15.0


# =============================================================================
# Topocentric Frame Tests
# =============================================================================

from ..coordinates import (
    ENUFrame,
    NEDFrame,
    SEZFrame,
    aer_to_ecef,
    compute_azimuth_elevation,
    compute_look_angles,
    ecef_to_aer,
)


def test_sez_frame_instantiation():
    """Test SEZ frame instantiation."""
    frame = SEZFrame(name="Test", latitude=np.radians(45.0), longitude=np.radians(90.0))
    assert frame.name == "Test"
    assert pytest.approx(frame.latitude, abs=1e-10) == np.radians(45.0)
    assert pytest.approx(frame.altitude, abs=1e-10) == 0.0


def test_enu_frame_instantiation():
    """Test ENU frame instantiation."""
    frame = ENUFrame(name="Test", latitude=0.0, longitude=0.0, altitude=100.0)
    assert frame.altitude == 100.0


def test_ned_frame_instantiation():
    """Test NED frame instantiation."""
    frame = NEDFrame(
        name="Test", latitude=np.radians(30.0), longitude=np.radians(-90.0), altitude=500.0
    )
    assert pytest.approx(frame.longitude, abs=1e-10) == np.radians(-90.0)


def test_topocentric_frame_observer_ecef():
    """Test observer ECEF position calculation."""
    frame = ENUFrame(name="Test", latitude=0.0, longitude=0.0, altitude=0.0)
    obs_ecef = frame.observer_ecef

    # At equator, prime meridian, should be at [a, 0, 0] approximately
    assert obs_ecef[0] > 6.3e6  # Near semi-major axis
    assert pytest.approx(obs_ecef[1], abs=1.0) == 0.0
    assert pytest.approx(obs_ecef[2], abs=1.0) == 0.0


def test_sez_rotation_matrix_orthogonal():
    """Test SEZ rotation matrix is orthogonal."""
    frame = SEZFrame(name="Test", latitude=np.radians(45.0), longitude=np.radians(45.0))
    R = frame.rotation_matrix

    # R @ R^T should equal identity
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    # det(R) should be +1
    assert pytest.approx(np.linalg.det(R), abs=1e-10) == 1.0


def test_enu_rotation_matrix_orthogonal():
    """Test ENU rotation matrix is orthogonal."""
    frame = ENUFrame(name="Test", latitude=np.radians(30.0), longitude=np.radians(-60.0))
    R = frame.rotation_matrix

    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert pytest.approx(np.linalg.det(R), abs=1e-10) == 1.0


def test_ned_rotation_matrix_orthogonal():
    """Test NED rotation matrix is orthogonal."""
    frame = NEDFrame(name="Test", latitude=np.radians(-45.0), longitude=np.radians(120.0))
    R = frame.rotation_matrix

    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert pytest.approx(np.linalg.det(R), abs=1e-10) == 1.0


def test_enu_at_equator_axes():
    """Test ENU axes at equator, prime meridian."""
    frame = ENUFrame(name="Test", latitude=0.0, longitude=0.0)

    # At equator, prime meridian:
    # E (East) should point toward +Y in ECEF
    # N (North) should point toward +Z in ECEF
    # U (Up) should point toward +X in ECEF

    # Test by transforming ECEF unit vectors
    # E = R @ [0,0,0] direction from [1,0,0]
    # For a point at [1,0,0] + epsilon*[0,1,0], local should be [epsilon,0,0]

    # East direction test: ECEF [0, 1, 0] in local should be mostly East
    obs_ecef = frame.observer_ecef
    east_point_ecef = obs_ecef + np.array([0, 1000, 0])
    pos_enu, _ = frame.ecef_to_local(east_point_ecef)
    # East component should be dominant
    assert pos_enu[0] > abs(pos_enu[1])
    assert pos_enu[0] > abs(pos_enu[2])


def test_ned_down_is_negative_up():
    """Test NED Down axis is opposite of ENU Up axis."""
    lat = np.radians(45.0)
    lon = np.radians(45.0)

    enu = ENUFrame(name="ENU", latitude=lat, longitude=lon)
    ned = NEDFrame(name="NED", latitude=lat, longitude=lon)

    # Transform same point
    target = enu.observer_ecef + np.array([0, 0, 1000])  # 1km offset in ECEF Z
    pos_enu, _ = enu.ecef_to_local(target)
    pos_ned, _ = ned.ecef_to_local(target)

    # ENU Up should be negative of NED Down
    assert pytest.approx(pos_enu[2], abs=1e-6) == -pos_ned[2]

    # ENU East should equal NED East
    assert pytest.approx(pos_enu[0], abs=1e-6) == pos_ned[1]

    # ENU North should equal NED North
    assert pytest.approx(pos_enu[1], abs=1e-6) == pos_ned[0]


def test_topocentric_round_trip():
    """Test ECEF -> local -> ECEF round trip."""
    frame = ENUFrame(
        name="Test", latitude=np.radians(40.0), longitude=np.radians(-75.0), altitude=100.0
    )

    # Original ECEF point
    target_ecef = geodetic_to_ecef(np.radians(40.5), np.radians(-74.5), 500.0)
    target_ecef = np.array(target_ecef)

    # Forward transform
    pos_local, _ = frame.ecef_to_local(target_ecef)

    # Inverse transform
    recovered_ecef, _ = frame.local_to_ecef(pos_local)

    # Should match original
    assert np.allclose(target_ecef, recovered_ecef, atol=1e-6)


def test_sez_round_trip():
    """Test SEZ frame round trip."""
    frame = SEZFrame(
        name="Test", latitude=np.radians(35.0), longitude=np.radians(135.0), altitude=50.0
    )

    target_ecef = geodetic_to_ecef(np.radians(36.0), np.radians(136.0), 1000.0)
    target_ecef = np.array(target_ecef)

    pos_sez, _ = frame.ecef_to_local(target_ecef)
    recovered_ecef, _ = frame.local_to_ecef(pos_sez)

    assert np.allclose(target_ecef, recovered_ecef, atol=1e-6)


def test_compute_azimuth_elevation_east():
    """Test azimuth/elevation for target due East."""
    obs_lat, obs_lon, obs_alt = 0.0, 0.0, 0.0

    # Target 1 degree East at same altitude
    target_ecef = geodetic_to_ecef(0.0, np.radians(1.0), 0.0)
    target_ecef = np.array(target_ecef)

    az, el, rng = compute_azimuth_elevation(obs_lat, obs_lon, obs_alt, target_ecef)

    # Azimuth should be ~90 degrees (East)
    assert pytest.approx(np.degrees(az), abs=1.0) == 90.0

    # Elevation should be ~0 (on horizon)
    assert abs(np.degrees(el)) < 5.0

    # Range should be > 0
    assert rng > 0


def test_compute_azimuth_elevation_north():
    """Test azimuth/elevation for target due North."""
    obs_lat, obs_lon, obs_alt = 0.0, 0.0, 0.0

    # Target 1 degree North
    target_ecef = geodetic_to_ecef(np.radians(1.0), 0.0, 0.0)
    target_ecef = np.array(target_ecef)

    az, _el, _rng = compute_azimuth_elevation(obs_lat, obs_lon, obs_alt, target_ecef)

    # Azimuth should be ~0 degrees (North)
    assert (
        pytest.approx(np.degrees(az), abs=1.0) == 0.0
        or pytest.approx(np.degrees(az), abs=1.0) == 360.0
    )


def test_compute_azimuth_elevation_zenith():
    """Test azimuth/elevation for target directly above."""
    obs_lat, obs_lon, obs_alt = np.radians(45.0), np.radians(90.0), 0.0

    # Target 100km directly above
    target_ecef = geodetic_to_ecef(np.radians(45.0), np.radians(90.0), 100000.0)
    target_ecef = np.array(target_ecef)

    _az, el, rng = compute_azimuth_elevation(obs_lat, obs_lon, obs_alt, target_ecef)

    # Elevation should be ~90 degrees (zenith)
    assert pytest.approx(np.degrees(el), abs=1.0) == 90.0

    # Range should be ~100km
    assert pytest.approx(rng, rel=0.01) == 100000.0


def test_compute_look_angles_comprehensive():
    """Test comprehensive look angle computation."""
    obs_lat = np.radians(45.0)
    obs_lon = 0.0
    obs_alt = 0.0

    # Target slightly offset
    target_ecef = geodetic_to_ecef(np.radians(46.0), np.radians(1.0), 10000.0)
    target_ecef = np.array(target_ecef)

    result = compute_look_angles(obs_lat, obs_lon, obs_alt, target_ecef)

    # Check all expected keys
    assert "azimuth" in result
    assert "azimuth_deg" in result
    assert "elevation" in result
    assert "elevation_deg" in result
    assert "range" in result
    assert "position_enu" in result
    assert "position_sez" in result
    assert "position_ned" in result
    assert "visible" in result

    # Should be visible (positive elevation)
    assert result["visible"] is True


def test_aer_to_ecef_round_trip():
    """Test AER -> ECEF -> AER round trip."""
    obs_lat = np.radians(40.0)
    obs_lon = np.radians(-74.0)
    obs_alt = 10.0

    # Original AER
    az_orig = np.radians(45.0)
    el_orig = np.radians(30.0)
    rng_orig = 50000.0

    # Convert to ECEF
    target_ecef = aer_to_ecef(az_orig, el_orig, rng_orig, obs_lat, obs_lon, obs_alt)

    # Convert back to AER
    az_back, el_back, rng_back = ecef_to_aer(target_ecef, obs_lat, obs_lon, obs_alt)

    assert pytest.approx(az_orig, abs=1e-6) == az_back
    assert pytest.approx(el_orig, abs=1e-6) == el_back
    assert pytest.approx(rng_orig, abs=1e-3) == rng_back


def test_aer_to_ecef_cardinal_directions():
    """Test AER conversion for cardinal directions."""
    obs_lat = 0.0
    obs_lon = 0.0
    obs_alt = 0.0

    # Due East, on horizon, 1000m range
    ecef_east = aer_to_ecef(np.pi / 2, 0.0, 1000.0, obs_lat, obs_lon, obs_alt)
    assert ecef_east.shape == (3,)

    # Due North, on horizon
    ecef_north = aer_to_ecef(0.0, 0.0, 1000.0, obs_lat, obs_lon, obs_alt)
    assert ecef_north.shape == (3,)


def test_velocity_transformation_topocentric():
    """Test velocity transformation in topocentric frame."""
    frame = ENUFrame(name="Test", latitude=np.radians(45.0), longitude=0.0, altitude=0.0)

    # ECEF velocity pointing in +Y direction
    vel_ecef = np.array([0.0, 100.0, 0.0])

    # Transform some position with this velocity
    pos_ecef = frame.observer_ecef + np.array([1000.0, 0.0, 0.0])
    pos_local, vel_local = frame.ecef_to_local(pos_ecef, vel_ecef)

    # Velocity should be non-zero
    assert np.linalg.norm(vel_local) > 0

    # Round trip velocity
    _, vel_back = frame.local_to_ecef(pos_local, vel_local)
    assert np.allclose(vel_ecef, vel_back, atol=1e-10)


def test_topocentric_with_altitude():
    """Test topocentric frame at altitude."""
    # Observer at 10km altitude
    frame = ENUFrame(
        name="Aircraft", latitude=np.radians(40.0), longitude=np.radians(-74.0), altitude=10000.0
    )

    # Target on ground below
    target_ground = geodetic_to_ecef(np.radians(40.0), np.radians(-74.0), 0.0)
    target_ground = np.array(target_ground)

    pos_enu, _ = frame.ecef_to_local(target_ground)

    # Target should be below (negative Up in ENU)
    assert pos_enu[2] < 0

    # Should be ~10km below
    assert pytest.approx(abs(pos_enu[2]), rel=0.01) == 10000.0


# =============================================================================
# Frame Transformation Composition Tests
# =============================================================================

from ..coordinates import (
    FrameTransformationRegistry,
    TransformationPath,
    compose_transformations,
    get_frame_registry,
    register_transform,
)


def test_transformation_path_creation():
    """Test TransformationPath creation."""
    path = TransformationPath(
        frames=["A", "B", "C"], transforms=[lambda p, v, t: (p * 2, v), lambda p, v, t: (p + 1, v)]
    )
    assert len(path) == 2
    assert "A" in repr(path)


def test_transformation_path_apply():
    """Test TransformationPath apply method."""

    # Create simple scaling transforms
    def scale_2(p, v, t):
        return p * 2, v * 2 if v is not None else None

    def add_10(p, v, t):
        return p + 10, v

    path = TransformationPath(frames=["A", "B", "C"], transforms=[scale_2, add_10])

    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.1, 0.2, 0.3])

    result_pos, result_vel = path.apply(pos, vel)

    # pos * 2 = [2, 4, 6], then +10 = [12, 14, 16]
    expected_pos = np.array([12.0, 14.0, 16.0])
    # vel * 2 = [0.2, 0.4, 0.6], unchanged by add_10
    expected_vel = np.array([0.2, 0.4, 0.6])

    assert np.allclose(result_pos, expected_pos)
    assert np.allclose(result_vel, expected_vel)


def test_transformation_path_no_velocity():
    """Test TransformationPath with position only."""

    def scale_2(p, v, t):
        return p * 2, v * 2 if v is not None else None

    path = TransformationPath(frames=["A", "B"], transforms=[scale_2])

    pos = np.array([1.0, 2.0, 3.0])
    result_pos, result_vel = path.apply(pos, None)

    assert np.allclose(result_pos, np.array([2.0, 4.0, 6.0]))
    assert result_vel is None


def test_registry_creation():
    """Test FrameTransformationRegistry creation."""
    registry = FrameTransformationRegistry()
    assert len(registry.list_frames()) == 0
    assert len(registry.list_transforms()) == 0


def test_registry_register_transform():
    """Test registering a transformation."""
    registry = FrameTransformationRegistry()

    def dummy_transform(p, v, t):
        return p, v

    registry.register("FrameA", "FrameB", dummy_transform)

    assert "FrameA" in registry.list_frames()
    assert "FrameB" in registry.list_frames()
    assert ("FrameA", "FrameB") in registry.list_transforms()


def test_registry_get_direct_transform():
    """Test getting a direct transformation."""
    registry = FrameTransformationRegistry()

    def dummy_transform(p, v, t):
        return p * 2, v

    registry.register("A", "B", dummy_transform)

    transform = registry.get_direct_transform("A", "B")
    assert transform is not None

    pos = np.array([1.0, 2.0, 3.0])
    result, _ = transform(pos, None, None)
    assert np.allclose(result, pos * 2)

    # No direct transform from B to A
    assert registry.get_direct_transform("B", "A") is None


def test_registry_bidirectional():
    """Test bidirectional registration."""
    registry = FrameTransformationRegistry()

    def forward(p, v, t):
        return p * 2, v

    def inverse(p, v, t):
        return p / 2, v

    registry.register("A", "B", forward, bidirectional=True, inverse_transform=inverse)

    assert ("A", "B") in registry.list_transforms()
    assert ("B", "A") in registry.list_transforms()


def test_registry_bidirectional_requires_inverse():
    """Test that bidirectional requires inverse_transform."""
    registry = FrameTransformationRegistry()

    def forward(p, v, t):
        return p * 2, v

    with pytest.raises(ValueError):
        registry.register("A", "B", forward, bidirectional=True)


def test_registry_find_path_direct():
    """Test finding a direct path."""
    registry = FrameTransformationRegistry()

    def transform(p, v, t):
        return p * 2, v

    registry.register("A", "B", transform)

    path = registry.find_path("A", "B")
    assert path is not None
    assert len(path) == 1


def test_registry_find_path_identity():
    """Test finding path to same frame."""
    registry = FrameTransformationRegistry()

    path = registry.find_path("A", "A")
    assert path is not None

    pos = np.array([1.0, 2.0, 3.0])
    result, _ = path.apply(pos, None)
    assert np.allclose(result, pos)


def test_registry_find_path_multi_hop():
    """Test finding a multi-hop path."""
    registry = FrameTransformationRegistry()

    def a_to_b(p, v, t):
        return p + 1, v

    def b_to_c(p, v, t):
        return p * 2, v

    def c_to_d(p, v, t):
        return p - 5, v

    registry.register("A", "B", a_to_b)
    registry.register("B", "C", b_to_c)
    registry.register("C", "D", c_to_d)

    # Find path from A to D (should be A -> B -> C -> D)
    path = registry.find_path("A", "D")
    assert path is not None
    assert len(path) == 3

    pos = np.array([10.0, 10.0, 10.0])
    # (10 + 1) = 11, * 2 = 22, - 5 = 17
    result, _ = path.apply(pos, None)
    assert np.allclose(result, np.array([17.0, 17.0, 17.0]))


def test_registry_find_path_not_found():
    """Test when no path exists."""
    registry = FrameTransformationRegistry()

    def transform(p, v, t):
        return p, v

    registry.register("A", "B", transform)
    registry.register("C", "D", transform)

    # No path from A to D
    path = registry.find_path("A", "D")
    assert path is None


def test_registry_transform_method():
    """Test the transform convenience method."""
    registry = FrameTransformationRegistry()

    def a_to_b(p, v, t):
        return p * 2, v * 2 if v is not None else None

    registry.register("A", "B", a_to_b)

    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.1, 0.2, 0.3])

    result_pos, result_vel = registry.transform("A", "B", pos, vel)

    assert np.allclose(result_pos, pos * 2)
    assert np.allclose(result_vel, vel * 2)


def test_registry_transform_no_path():
    """Test transform raises error when no path."""
    registry = FrameTransformationRegistry()

    pos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="No transformation path"):
        registry.transform("A", "B", pos)


def test_compose_transformations():
    """Test compose_transformations function."""

    def scale_2(p, v, t):
        return p * 2, v * 2 if v is not None else None

    def add_10(p, v, t):
        return p + 10, v

    composed = compose_transformations(scale_2, add_10)

    pos = np.array([5.0, 5.0, 5.0])
    result, _ = composed(pos)

    # 5 * 2 = 10, + 10 = 20
    assert np.allclose(result, np.array([20.0, 20.0, 20.0]))


def test_compose_transformations_with_velocity():
    """Test composed transformation preserves velocity."""

    def scale_2(p, v, t):
        return p * 2, v * 2 if v is not None else None

    def add_10(p, v, t):
        return p + 10, v + 1 if v is not None else None

    composed = compose_transformations(scale_2, add_10)

    pos = np.array([1.0, 1.0, 1.0])
    vel = np.array([0.5, 0.5, 0.5])

    result_pos, result_vel = composed(pos, vel)

    # pos: 1*2=2, +10=12
    assert np.allclose(result_pos, np.array([12.0, 12.0, 12.0]))
    # vel: 0.5*2=1.0, +1=2.0
    assert np.allclose(result_vel, np.array([2.0, 2.0, 2.0]))


def test_global_registry():
    """Test global registry access."""
    registry = get_frame_registry()
    assert isinstance(registry, FrameTransformationRegistry)

    # Global registry should be the same instance
    registry2 = get_frame_registry()
    assert registry is registry2


def test_register_transform_function():
    """Test register_transform convenience function."""
    # Get initial state
    registry = get_frame_registry()
    initial_transforms = len(registry.list_transforms())

    def test_transform(p, v, t):
        return p, v

    # Register with unique names to avoid conflicts
    register_transform("TestFrameX", "TestFrameY", test_transform)

    # Should have one more transform
    assert len(registry.list_transforms()) == initial_transforms + 1


def test_registry_with_frame_types():
    """Test registry works with frame types."""
    registry = FrameTransformationRegistry()

    def transform(p, v, t):
        return p * 2, v

    # Register using class types
    registry.register(ENUFrame, NEDFrame, transform)

    # Should be able to find path using class names
    path = registry.find_path("ENUFrame", "NEDFrame")
    assert path is not None


def test_registry_with_frame_instances():
    """Test registry works with frame instances."""
    registry = FrameTransformationRegistry()

    enu = ENUFrame(name="ENU", latitude=0.0, longitude=0.0)
    ned = NEDFrame(name="NED", latitude=0.0, longitude=0.0)

    def transform(p, v, t):
        return p * 2, v

    # Register using instances (gets class name as key)
    registry.register(enu, ned, transform)

    # Should be able to find path
    path = registry.find_path(enu, ned)
    assert path is not None


# =============================================================================
# Tests for Inertial Frame Conversions (5.6)
# =============================================================================


def test_gcrs_creation():
    """Test GCRS frame instantiation."""
    gcrs = GCRS()
    assert gcrs.name == "GCRS"


def test_j2000_creation():
    """Test J2000 frame instantiation."""
    j2000 = J2000()
    assert j2000.name == "J2000"


def test_icrs_creation():
    """Test ICRS frame instantiation."""
    icrs = ICRS()
    assert icrs.name == "ICRS"


def test_gcrs_transform_to_self():
    """Test GCRS transform to itself."""
    gcrs = GCRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_out, vel_out = gcrs.transform_to(gcrs, position, velocity, timestamp)

    np.testing.assert_array_almost_equal(pos_out, position)
    np.testing.assert_array_almost_equal(vel_out, velocity)


def test_j2000_transform_to_self():
    """Test J2000 transform to itself."""
    j2000 = J2000()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_out, vel_out = j2000.transform_to(j2000, position, velocity, timestamp)

    np.testing.assert_array_almost_equal(pos_out, position)
    np.testing.assert_array_almost_equal(vel_out, velocity)


def test_icrs_transform_to_self():
    """Test ICRS transform to itself."""
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_out, vel_out = icrs.transform_to(icrs, position, velocity, timestamp)

    np.testing.assert_array_almost_equal(pos_out, position)
    np.testing.assert_array_almost_equal(vel_out, velocity)


def test_gcrs_to_j2000_no_timestamp():
    """Test GCRS to J2000 without timestamp (simplified)."""
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_j2000, vel_j2000 = gcrs_to_j2000(position, velocity)

    # Without timestamp, should be approximately the same
    np.testing.assert_array_almost_equal(pos_j2000, position)
    np.testing.assert_array_almost_equal(vel_j2000, velocity)


def test_j2000_to_gcrs_no_timestamp():
    """Test J2000 to GCRS without timestamp (simplified)."""
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_gcrs, vel_gcrs = j2000_to_gcrs(position, velocity)

    # Without timestamp, should be approximately the same
    np.testing.assert_array_almost_equal(pos_gcrs, position)
    np.testing.assert_array_almost_equal(vel_gcrs, velocity)


def test_gcrs_j2000_roundtrip_no_timestamp():
    """Test GCRS to J2000 to GCRS round-trip without timestamp."""
    position = np.array([7000000.0, 1000000.0, 500000.0])
    velocity = np.array([100.0, 7500.0, 200.0])

    pos_j2000, vel_j2000 = gcrs_to_j2000(position, velocity)
    pos_back, vel_back = j2000_to_gcrs(pos_j2000, vel_j2000)

    np.testing.assert_array_almost_equal(pos_back, position)
    np.testing.assert_array_almost_equal(vel_back, velocity)


def test_gcrs_to_j2000_with_timestamp():
    """Test GCRS to J2000 with timestamp."""
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_j2000, vel_j2000 = gcrs_to_j2000(position, velocity, timestamp)

    # The transformation should produce valid results
    assert pos_j2000 is not None
    assert vel_j2000 is not None
    assert len(pos_j2000) == 3
    assert len(vel_j2000) == 3

    # Precession rate is ~50 arcsec/year, so over 24 years = ~1200 arcsec
    # At 7000 km range: 7e6 * sin(1200/3600 deg) ≈ 40 km
    pos_diff = np.linalg.norm(pos_j2000 - position)
    assert pos_diff < 50000.0  # Conservative bound (50 km)


def test_j2000_to_gcrs_with_timestamp():
    """Test J2000 to GCRS with timestamp."""
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_gcrs, vel_gcrs = j2000_to_gcrs(position, velocity, timestamp)

    # The transformation should produce valid results
    assert pos_gcrs is not None
    assert vel_gcrs is not None
    assert len(pos_gcrs) == 3
    assert len(vel_gcrs) == 3


def test_gcrs_j2000_roundtrip_with_timestamp():
    """Test GCRS to J2000 to GCRS round-trip with timestamp."""
    position = np.array([7000000.0, 1000000.0, 500000.0])
    velocity = np.array([100.0, 7500.0, 200.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_j2000, vel_j2000 = gcrs_to_j2000(position, velocity, timestamp)
    pos_back, vel_back = j2000_to_gcrs(pos_j2000, vel_j2000, timestamp)

    # Round-trip should preserve position and velocity
    np.testing.assert_array_almost_equal(pos_back, position, decimal=3)
    np.testing.assert_array_almost_equal(vel_back, velocity, decimal=3)


def test_gcrs_j2000_roundtrip_at_epoch():
    """Test GCRS to J2000 round-trip at J2000 epoch."""
    position = np.array([7000000.0, 1000000.0, 500000.0])
    velocity = np.array([100.0, 7500.0, 200.0])
    # At J2000 epoch, transformation should be identity
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    pos_j2000, vel_j2000 = gcrs_to_j2000(position, velocity, timestamp)
    pos_back, vel_back = j2000_to_gcrs(pos_j2000, vel_j2000, timestamp)

    # At J2000 epoch, should be very close to identity
    np.testing.assert_array_almost_equal(pos_back, position, decimal=6)
    np.testing.assert_array_almost_equal(vel_back, velocity, decimal=6)


def test_compute_frame_bias_matrix():
    """Test compute_frame_bias_matrix function."""
    bias_matrix = compute_frame_bias_matrix()

    # Should be 3x3 matrix
    assert bias_matrix.shape == (3, 3)

    # Should be approximately orthogonal (rotation matrix)
    identity = bias_matrix @ bias_matrix.T
    np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)

    # Determinant should be approximately 1
    det = np.linalg.det(bias_matrix)
    assert pytest.approx(det, abs=1e-10) == 1.0


def test_frame_bias_matrix_small_rotation():
    """Test frame bias matrix represents a small rotation."""
    bias_matrix = compute_frame_bias_matrix()

    # Frame bias is very small (< 0.1 arcseconds)
    # The matrix should be very close to identity
    diff_from_identity = np.linalg.norm(bias_matrix - np.eye(3))
    # Typically order of 1e-7 radians
    assert diff_from_identity < 1e-5


def test_frame_bias_applied_to_position():
    """Test frame bias matrix applied to position."""
    bias_matrix = compute_frame_bias_matrix()

    # Near-Earth position
    position = np.array([7000000.0, 0.0, 0.0])

    # Apply bias
    pos_transformed = bias_matrix @ position

    # Difference should be very small (< 1 meter for near-Earth)
    diff = np.linalg.norm(pos_transformed - position)
    # Frame bias of ~0.1 arcsec means ~3 cm per 7000 km
    assert diff < 1.0  # Less than 1 meter


def test_gcrs_transform_to_j2000_class():
    """Test GCRS class transform_to J2000."""
    gcrs = GCRS()
    j2000 = J2000()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_j2000, vel_j2000 = gcrs.transform_to(j2000, position, velocity, timestamp)

    # Should produce valid transformation
    assert pos_j2000 is not None
    assert vel_j2000 is not None


def test_j2000_transform_to_gcrs_class():
    """Test J2000 class transform_to GCRS."""
    gcrs = GCRS()
    j2000 = J2000()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_gcrs, vel_gcrs = j2000.transform_to(gcrs, position, velocity, timestamp)

    # Should produce valid transformation
    assert pos_gcrs is not None
    assert vel_gcrs is not None


def test_j2000_transform_to_icrs():
    """Test J2000 to ICRS transformation."""
    j2000 = J2000()
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_icrs, _vel_icrs = j2000.transform_to(icrs, position, velocity)

    # Frame bias is very small
    diff = np.linalg.norm(pos_icrs - position)
    assert diff < 1.0  # Less than 1 meter


def test_icrs_transform_to_j2000():
    """Test ICRS to J2000 transformation."""
    j2000 = J2000()
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_j2000, _vel_j2000 = icrs.transform_to(j2000, position, velocity)

    # Frame bias is very small
    diff = np.linalg.norm(pos_j2000 - position)
    assert diff < 1.0  # Less than 1 meter


def test_j2000_icrs_roundtrip():
    """Test J2000 to ICRS to J2000 round-trip."""
    j2000 = J2000()
    icrs = ICRS()
    position = np.array([7000000.0, 1000000.0, 500000.0])
    velocity = np.array([100.0, 7500.0, 200.0])

    pos_icrs, vel_icrs = j2000.transform_to(icrs, position, velocity)
    pos_back, vel_back = icrs.transform_to(j2000, pos_icrs, vel_icrs)

    # Round-trip should preserve position and velocity
    np.testing.assert_array_almost_equal(pos_back, position, decimal=6)
    np.testing.assert_array_almost_equal(vel_back, velocity, decimal=6)


def test_gcrs_transform_to_icrs():
    """Test GCRS to ICRS transformation (approximately identical)."""
    gcrs = GCRS()
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_icrs, vel_icrs = gcrs.transform_to(icrs, position, velocity)

    # For practical purposes, GCRS ≈ ICRS
    np.testing.assert_array_almost_equal(pos_icrs, position)
    np.testing.assert_array_almost_equal(vel_icrs, velocity)


def test_icrs_transform_to_gcrs():
    """Test ICRS to GCRS transformation (approximately identical)."""
    gcrs = GCRS()
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    pos_gcrs, vel_gcrs = icrs.transform_to(gcrs, position, velocity)

    # For practical purposes, ICRS ≈ GCRS
    np.testing.assert_array_almost_equal(pos_gcrs, position)
    np.testing.assert_array_almost_equal(vel_gcrs, velocity)


def test_gcrs_velocity_only():
    """Test GCRS transform without velocity."""
    gcrs = GCRS()
    j2000 = J2000()
    position = np.array([7000000.0, 0.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_j2000, vel_j2000 = gcrs.transform_to(j2000, position, None, timestamp)

    assert pos_j2000 is not None
    assert vel_j2000 is None


def test_j2000_velocity_only():
    """Test J2000 transform without velocity."""
    gcrs = GCRS()
    j2000 = J2000()
    position = np.array([7000000.0, 0.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    pos_gcrs, vel_gcrs = j2000.transform_to(gcrs, position, None, timestamp)

    assert pos_gcrs is not None
    assert vel_gcrs is None


def test_icrs_velocity_only():
    """Test ICRS transform without velocity."""
    j2000 = J2000()
    icrs = ICRS()
    position = np.array([7000000.0, 0.0, 0.0])

    pos_j2000, vel_j2000 = icrs.transform_to(j2000, position, None)

    assert pos_j2000 is not None
    assert vel_j2000 is None


def test_gcrs_transform_not_implemented():
    """Test GCRS transform_to raises for unsupported frame."""
    gcrs = GCRS()
    enu = ENUFrame(name="ENU", latitude=0.0, longitude=0.0)
    position = np.array([7000000.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError):
        gcrs.transform_to(enu, position)


def test_j2000_transform_not_implemented():
    """Test J2000 transform_to raises for unsupported frame."""
    j2000 = J2000()
    enu = ENUFrame(name="ENU", latitude=0.0, longitude=0.0)
    position = np.array([7000000.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError):
        j2000.transform_to(enu, position)


def test_icrs_transform_not_implemented():
    """Test ICRS transform_to raises for unsupported frame."""
    icrs = ICRS()
    enu = ENUFrame(name="ENU", latitude=0.0, longitude=0.0)
    position = np.array([7000000.0, 0.0, 0.0])

    with pytest.raises(NotImplementedError):
        icrs.transform_to(enu, position)


# =============================================================================
# Tests for Time-Varying Transformations (2.7)
# =============================================================================


def test_time_varying_transform_base_class():
    """Test TimeVaryingTransform base class."""
    transform = TimeVaryingTransform()

    # Default reference epoch is J2000.0
    assert transform.reference_epoch == datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Base class should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        transform.get_rotation_matrix(datetime.datetime.now())


def test_time_varying_transform_custom_epoch():
    """Test TimeVaryingTransform with custom reference epoch."""
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)
    transform = TimeVaryingTransform(reference_epoch=epoch)

    assert transform.reference_epoch == epoch


def test_rotation_rate_transform_creation():
    """Test RotationRateTransform instantiation."""
    axis = np.array([0.0, 0.0, 1.0])
    rate = 0.001  # rad/s

    transform = RotationRateTransform(rotation_axis=axis, rotation_rate=rate)

    assert transform.rotation_rate == rate
    np.testing.assert_array_almost_equal(transform.rotation_axis, axis)


def test_rotation_rate_transform_angle():
    """Test RotationRateTransform angle computation."""
    axis = np.array([0.0, 0.0, 1.0])
    rate = 0.001  # rad/s
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)

    transform = RotationRateTransform(
        rotation_axis=axis, rotation_rate=rate, reference_epoch=epoch
    )

    # At reference epoch, angle should be 0
    angle_0 = transform.get_rotation_angle(epoch)
    assert pytest.approx(angle_0, abs=1e-12) == 0.0

    # After 1000 seconds, angle should be 1 radian
    later = epoch + datetime.timedelta(seconds=1000)
    angle_1 = transform.get_rotation_angle(later)
    assert pytest.approx(angle_1, abs=1e-12) == 1.0


def test_rotation_rate_transform_matrix_z_axis():
    """Test RotationRateTransform rotation matrix about z-axis."""
    axis = np.array([0.0, 0.0, 1.0])
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)

    # Create transform with known angle
    transform = RotationRateTransform(
        rotation_axis=axis,
        rotation_rate=1.0,  # 1 rad/s for easy calculation
        reference_epoch=epoch,
    )

    # At t = π/2 seconds, rotation should be 90 degrees
    t_90deg = epoch + datetime.timedelta(seconds=np.pi / 2)
    R = transform.get_rotation_matrix(t_90deg)

    # Expected rotation by 90° about z-axis
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Note: datetime microsecond resolution causes small errors
    np.testing.assert_array_almost_equal(R, expected, decimal=6)


def test_rotation_rate_transform_position():
    """Test RotationRateTransform on position."""
    axis = np.array([0.0, 0.0, 1.0])
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)

    transform = RotationRateTransform(rotation_axis=axis, rotation_rate=1.0, reference_epoch=epoch)

    # Position along x-axis
    position = np.array([7000000.0, 0.0, 0.0])

    # At t = π/2, position should rotate to y-axis
    t_90deg = epoch + datetime.timedelta(seconds=np.pi / 2)
    pos_out, vel_out = transform(position, timestamp=t_90deg)

    expected_pos = np.array([0.0, 7000000.0, 0.0])
    # datetime microsecond resolution causes small position errors
    # Check relative error is < 1e-6 (about 7 meters on 7000 km)
    np.testing.assert_allclose(pos_out, expected_pos, atol=10.0)  # 10 meter tolerance
    assert vel_out is None


def test_rotation_rate_transform_velocity():
    """Test RotationRateTransform on velocity."""
    axis = np.array([0.0, 0.0, 1.0])
    omega = 7.2921150e-5  # Earth rotation rate
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)

    transform = RotationRateTransform(
        rotation_axis=axis, rotation_rate=omega, reference_epoch=epoch
    )

    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])

    _pos_out, vel_out = transform(position, velocity, timestamp=epoch)

    # Velocity should include ω×r contribution
    assert vel_out is not None
    assert len(vel_out) == 3


def test_rotation_rate_transform_earth_rotation():
    """Test RotationRateTransform with Earth rotation rate."""
    axis = np.array([0.0, 0.0, 1.0])
    omega_earth = 7.2921150e-5  # rad/s
    epoch = datetime.datetime(2024, 1, 1, 0, 0, 0)

    transform = RotationRateTransform(
        rotation_axis=axis, rotation_rate=omega_earth, reference_epoch=epoch
    )

    # After one sidereal day, rotation should be 2π
    sidereal_day = 86164.0905  # seconds
    one_day_later = epoch + datetime.timedelta(seconds=sidereal_day)
    angle = transform.get_rotation_angle(one_day_later)

    assert pytest.approx(angle, rel=1e-4) == 2 * np.pi


def test_interpolated_transform_creation():
    """Test InterpolatedTransform instantiation."""
    epochs = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
    matrices = [np.eye(3), np.eye(3)]

    transform = InterpolatedTransform(epochs, matrices)

    assert len(transform.epochs) == 2
    assert len(transform.rotation_matrices) == 2


def test_interpolated_transform_requires_two_epochs():
    """Test InterpolatedTransform requires at least 2 epochs."""
    with pytest.raises(ValueError, match="At least 2 epochs"):
        InterpolatedTransform([datetime.datetime.now()], [np.eye(3)])


def test_interpolated_transform_mismatched_lengths():
    """Test InterpolatedTransform with mismatched lengths."""
    epochs = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
    matrices = [np.eye(3)]

    with pytest.raises(ValueError, match="same length"):
        InterpolatedTransform(epochs, matrices)


def test_interpolated_transform_identity():
    """Test InterpolatedTransform with identity matrices."""
    epochs = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
    matrices = [np.eye(3), np.eye(3)]

    transform = InterpolatedTransform(epochs, matrices)

    position = np.array([7000000.0, 0.0, 0.0])
    mid_time = datetime.datetime(2024, 1, 1, 12)

    pos_out, _ = transform(position, timestamp=mid_time)

    np.testing.assert_array_almost_equal(pos_out, position)


def test_interpolated_transform_rotation():
    """Test InterpolatedTransform with actual rotation."""
    # Create a 90-degree rotation about z-axis
    R_0 = np.eye(3)
    R_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    epochs = [
        datetime.datetime(2024, 1, 1, 0, 0, 0),
        datetime.datetime(2024, 1, 1, 0, 0, 10),  # 10 seconds later
    ]
    matrices = [R_0, R_90]

    transform = InterpolatedTransform(epochs, matrices)

    # At midpoint (45 degrees), x should rotate partially toward y
    position = np.array([1.0, 0.0, 0.0])
    mid_time = datetime.datetime(2024, 1, 1, 0, 0, 5)

    pos_out, _ = transform(position, timestamp=mid_time)

    # At 45 degrees, x and y components should be equal
    assert pytest.approx(abs(pos_out[0]), abs=0.1) == pytest.approx(abs(pos_out[1]), abs=0.1)


def test_interpolated_transform_boundary_clamping():
    """Test InterpolatedTransform boundary clamping."""
    epochs = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
    R_0 = np.eye(3)
    R_1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # 90 deg about z
    matrices = [R_0, R_1]

    transform = InterpolatedTransform(epochs, matrices, extrapolate=False)

    # Before first epoch, should use first matrix
    before = datetime.datetime(2023, 12, 31)
    R_before = transform.get_rotation_matrix(before)
    np.testing.assert_array_almost_equal(R_before, R_0)

    # After last epoch, should use last matrix
    after = datetime.datetime(2024, 1, 3)
    R_after = transform.get_rotation_matrix(after)
    np.testing.assert_array_almost_equal(R_after, R_1)


def test_interpolated_transform_quaternion_conversion():
    """Test InterpolatedTransform quaternion conversion is reversible."""
    transform = InterpolatedTransform(
        [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)], [np.eye(3), np.eye(3)]
    )

    # Test with various rotation matrices
    test_matrices = [
        np.eye(3),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90° about z
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # 90° about x
    ]

    for R in test_matrices:
        q = transform._rotation_to_quaternion(R)
        R_back = transform._quaternion_to_rotation(q)
        np.testing.assert_array_almost_equal(R, R_back, decimal=10)


def test_epoch_cached_transform_creation():
    """Test EpochCachedTransform instantiation."""
    base = RotationRateTransform(rotation_axis=np.array([0, 0, 1]), rotation_rate=0.001)
    cached = EpochCachedTransform(base, cache_size=50)

    assert cached.cache_size == 50
    assert len(cached._rotation_cache) == 0


def test_epoch_cached_transform_caching():
    """Test EpochCachedTransform caches results."""
    base = RotationRateTransform(rotation_axis=np.array([0, 0, 1]), rotation_rate=0.001)
    cached = EpochCachedTransform(base, cache_size=50)

    timestamp = datetime.datetime(2024, 1, 1)

    # First call should compute and cache
    R1 = cached.get_rotation_matrix(timestamp)
    assert len(cached._rotation_cache) == 1

    # Second call should use cache
    R2 = cached.get_rotation_matrix(timestamp)
    assert len(cached._rotation_cache) == 1

    np.testing.assert_array_equal(R1, R2)


def test_epoch_cached_transform_cache_eviction():
    """Test EpochCachedTransform evicts old entries."""
    base = RotationRateTransform(rotation_axis=np.array([0, 0, 1]), rotation_rate=0.001)
    cached = EpochCachedTransform(base, cache_size=3)

    base_time = datetime.datetime(2024, 1, 1)

    # Fill cache beyond capacity
    for i in range(5):
        t = base_time + datetime.timedelta(seconds=i)
        cached.get_rotation_matrix(t)

    # Cache should not exceed size
    assert len(cached._rotation_cache) <= 3


def test_epoch_cached_transform_clear_cache():
    """Test EpochCachedTransform clear_cache method."""
    base = RotationRateTransform(rotation_axis=np.array([0, 0, 1]), rotation_rate=0.001)
    cached = EpochCachedTransform(base, cache_size=50)

    timestamp = datetime.datetime(2024, 1, 1)
    cached.get_rotation_matrix(timestamp)
    cached.get_rotation_rate_matrix(timestamp)

    assert len(cached._rotation_cache) == 1
    assert len(cached._rate_cache) == 1

    cached.clear_cache()

    assert len(cached._rotation_cache) == 0
    assert len(cached._rate_cache) == 0


def test_epoch_cached_transform_full_pipeline():
    """Test EpochCachedTransform full transformation."""
    base = RotationRateTransform(rotation_axis=np.array([0, 0, 1]), rotation_rate=0.001)
    cached = EpochCachedTransform(base, cache_size=50)

    position = np.array([7000000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 1, 1)

    pos_out1, vel_out1 = cached(position, velocity, timestamp)
    pos_out2, vel_out2 = cached(position, velocity, timestamp)

    np.testing.assert_array_equal(pos_out1, pos_out2)
    np.testing.assert_array_equal(vel_out1, vel_out2)


def test_rotation_rate_transform_initial_angle():
    """Test RotationRateTransform with initial angle."""
    axis = np.array([0, 0, 1])
    rate = 1.0
    epoch = datetime.datetime(2024, 1, 1)
    initial_angle = np.pi / 4  # 45 degrees

    transform = RotationRateTransform(
        rotation_axis=axis, rotation_rate=rate, reference_epoch=epoch, initial_angle=initial_angle
    )

    # At reference epoch, angle should be initial_angle
    angle = transform.get_rotation_angle(epoch)
    assert pytest.approx(angle, abs=1e-12) == initial_angle


def test_rotation_rate_transform_non_z_axis():
    """Test RotationRateTransform with non-z rotation axis."""
    # Rotation about x-axis
    axis = np.array([1, 0, 0])
    rate = 1.0
    epoch = datetime.datetime(2024, 1, 1)

    transform = RotationRateTransform(
        rotation_axis=axis, rotation_rate=rate, reference_epoch=epoch
    )

    # At t = π/2, rotation should be 90 degrees about x
    t_90deg = epoch + datetime.timedelta(seconds=np.pi / 2)
    R = transform.get_rotation_matrix(t_90deg)

    # For rotation about x-axis by 90°:
    # x stays the same, y -> z, z -> -y
    expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # datetime microsecond resolution causes small errors
    np.testing.assert_array_almost_equal(R, expected, decimal=6)


def test_rotation_rate_transform_axis_normalization():
    """Test RotationRateTransform normalizes rotation axis."""
    # Non-unit axis
    axis = np.array([0, 0, 2])  # Length 2
    rate = 1.0

    transform = RotationRateTransform(rotation_axis=axis, rotation_rate=rate)

    # Axis should be normalized
    np.testing.assert_array_almost_equal(transform.rotation_axis, np.array([0, 0, 1]))


def test_time_varying_transform_default_rate_matrix():
    """Test TimeVaryingTransform default rotation rate is zero."""

    # Need a concrete subclass to test
    class SimpleTransform(TimeVaryingTransform):
        def get_rotation_matrix(self, timestamp):
            return np.eye(3)

    transform = SimpleTransform()
    rate_matrix = transform.get_rotation_rate_matrix(datetime.datetime.now())

    np.testing.assert_array_equal(rate_matrix, np.zeros((3, 3)))


# =============================================================================
# Tests for IAU 2006 Precession Model
# =============================================================================


def test_precession_angles_at_j2000():
    """Test precession angles at J2000.0 epoch (T=0)."""
    angles = compute_precession_angles_iau2006(0.0)

    # At J2000.0, epsilon_A should equal the J2000 obliquity
    epsilon_0 = 84381.406 * np.pi / (180.0 * 3600.0)  # radians
    np.testing.assert_allclose(angles["epsilon_A"], epsilon_0, rtol=1e-10)

    # At T=0, equatorial precession angles should be small (just constant terms)
    # zeta_A and z_A have constant terms of ~2.65" = 1.285e-5 rad
    assert abs(angles["zeta_A"]) < 2e-5  # 2.65" ~ 1.3e-5 rad
    assert abs(angles["z_A"]) < 2e-5
    assert abs(angles["theta_A"]) < 1e-10  # Should be essentially zero


def test_precession_angles_signs():
    """Test that precession angles have correct signs over time."""
    T = 0.25  # About 25 years from J2000

    angles = compute_precession_angles_iau2006(T)

    # Epsilon_A should decrease over time (obliquity is decreasing)
    epsilon_0 = 84381.406 * np.pi / (180.0 * 3600.0)
    assert angles["epsilon_A"] < epsilon_0

    # zeta_A should be positive for positive T
    assert angles["zeta_A"] > 0

    # z_A should be positive for positive T
    assert angles["z_A"] > 0

    # theta_A should be positive for positive T
    assert angles["theta_A"] > 0


def test_precession_matrix_orthogonal():
    """Test that precession matrix is orthogonal."""
    T = 0.5  # 50 years from J2000

    P = compute_precession_matrix_iau2006(T)

    # P @ P.T should be identity
    np.testing.assert_allclose(P @ P.T, np.eye(3), atol=1e-14)

    # det(P) should be 1
    np.testing.assert_allclose(np.linalg.det(P), 1.0, atol=1e-14)


def test_precession_matrix_identity_at_j2000():
    """Test that precession matrix is identity at J2000.0."""
    T = 0.0

    P = compute_precession_matrix_iau2006(T)

    # Should be very close to identity (only small constant terms)
    np.testing.assert_allclose(P, np.eye(3), atol=1e-8)


def test_precession_roundtrip():
    """Test that J2000->date->J2000 transformation is reversible."""
    pos_j2000 = np.array([7000000.0, 1000000.0, 500000.0])
    vel_j2000 = np.array([1000.0, 7500.0, 500.0])
    timestamp = datetime.datetime(2024, 6, 15, 12, 0, 0)

    # Forward transformation
    pos_date, vel_date = apply_precession_j2000_to_date(pos_j2000, timestamp, vel_j2000)

    # Inverse transformation
    pos_recovered, vel_recovered = apply_precession_date_to_j2000(pos_date, timestamp, vel_date)

    np.testing.assert_allclose(pos_recovered, pos_j2000, rtol=1e-14)
    np.testing.assert_allclose(vel_recovered, vel_j2000, rtol=1e-14)


def test_precession_magnitude():
    """Test that precession causes expected magnitude of position change."""
    pos_j2000 = np.array([7000000.0, 0.0, 0.0])  # 7000 km along x-axis
    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)  # ~24 years from J2000

    pos_date, _ = apply_precession_j2000_to_date(pos_j2000, timestamp)

    # Position magnitude should be preserved
    np.testing.assert_allclose(np.linalg.norm(pos_date), np.linalg.norm(pos_j2000), rtol=1e-14)

    # Position should change by tens of km over 24 years at 7000 km range
    pos_diff = np.linalg.norm(pos_date - pos_j2000)
    assert 10000 < pos_diff < 100000  # 10-100 km change expected


def test_precession_method_ecliptic():
    """Test ecliptic precession method."""
    T = 0.25

    P_equatorial = compute_precession_matrix_iau2006(T, method="equatorial")
    P_ecliptic = compute_precession_matrix_iau2006(T, method="ecliptic")

    # Both methods should produce orthogonal matrices
    np.testing.assert_allclose(P_equatorial @ P_equatorial.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(P_ecliptic @ P_ecliptic.T, np.eye(3), atol=1e-14)


def test_precession_invalid_method():
    """Test that invalid method raises error."""
    with pytest.raises(ValueError, match="Unknown method"):
        compute_precession_matrix_iau2006(0.25, method="invalid")


def test_precession_velocity_only():
    """Test precession with velocity but no position."""
    vel_j2000 = np.array([1000.0, 7500.0, 500.0])
    timestamp = datetime.datetime(2024, 6, 15, 12, 0, 0)

    # Forward with no position (just test it doesn't crash)
    _pos_date, vel_date = apply_precession_j2000_to_date(
        np.array([0.0, 0.0, 0.0]), timestamp, vel_j2000
    )

    assert vel_date is not None
    # Velocity magnitude should be preserved
    np.testing.assert_allclose(np.linalg.norm(vel_date), np.linalg.norm(vel_j2000), rtol=1e-14)


def test_precession_no_velocity():
    """Test precession without velocity."""
    pos_j2000 = np.array([7000000.0, 0.0, 0.0])
    timestamp = datetime.datetime(2024, 6, 15, 12, 0, 0)

    pos_date, vel_date = apply_precession_j2000_to_date(pos_j2000, timestamp)

    assert vel_date is None
    assert pos_date is not None


# =============================================================================
# Tests for IAU 2000B Nutation Model
# =============================================================================


def test_fundamental_arguments_at_j2000():
    """Test fundamental arguments at J2000.0 epoch."""
    args = compute_fundamental_arguments(0.0)

    # At T=0, arguments should be their initial values (in radians)
    # These are the constant terms from the polynomial expressions
    arcsec_to_rad = np.pi / (180.0 * 3600.0)

    l_0 = 485868.249036 * arcsec_to_rad
    lp_0 = 1287104.79305 * arcsec_to_rad
    F_0 = 335779.526232 * arcsec_to_rad
    D_0 = 1072260.70369 * arcsec_to_rad
    Om_0 = 450160.398036 * arcsec_to_rad

    np.testing.assert_allclose(args["l"], l_0, rtol=1e-10)
    np.testing.assert_allclose(args["lp"], lp_0, rtol=1e-10)
    np.testing.assert_allclose(args["F"], F_0, rtol=1e-10)
    np.testing.assert_allclose(args["D"], D_0, rtol=1e-10)
    np.testing.assert_allclose(args["Om"], Om_0, rtol=1e-10)


def test_nutation_magnitude():
    """Test that nutation values are in expected range."""
    # Test at various epochs
    for T in [0.0, 0.1, 0.25, 0.5]:
        dpsi, deps = compute_nutation_iau2000b(T)

        # Convert to arcseconds
        dpsi_arcsec = np.degrees(dpsi) * 3600
        deps_arcsec = np.degrees(deps) * 3600

        # Nutation in longitude should be roughly ±20 arcsec
        assert -25 < dpsi_arcsec < 25

        # Nutation in obliquity should be roughly ±10 arcsec
        assert -15 < deps_arcsec < 15


def test_nutation_matrix_orthogonal():
    """Test that nutation matrix is orthogonal."""
    T = 0.25

    N = compute_nutation_matrix(T)

    # N @ N.T should be identity
    np.testing.assert_allclose(N @ N.T, np.eye(3), atol=1e-14)

    # det(N) should be 1
    np.testing.assert_allclose(np.linalg.det(N), 1.0, atol=1e-14)


def test_nutation_matrix_close_to_identity():
    """Test that nutation matrix is close to identity (small rotation)."""
    T = 0.25

    N = compute_nutation_matrix(T)

    # Nutation is a small rotation, so matrix should be close to identity
    # Off-diagonal elements should be small (< 1e-4 rad ~ 20 arcsec)
    assert np.allclose(N, np.eye(3), atol=1e-4)


def test_nutation_roundtrip():
    """Test that nutation transformation is reversible."""
    pos_mean = np.array([7000000.0, 1000000.0, 500000.0])
    vel_mean = np.array([1000.0, 7500.0, 500.0])
    T = 0.25

    # Forward: mean to true
    pos_true, vel_true = apply_nutation(pos_mean, T, vel_mean)

    # Inverse: true to mean
    pos_recovered, vel_recovered = apply_nutation(pos_true, T, vel_true, inverse=True)

    np.testing.assert_allclose(pos_recovered, pos_mean, rtol=1e-14)
    np.testing.assert_allclose(vel_recovered, vel_mean, rtol=1e-14)


def test_nutation_position_magnitude_preserved():
    """Test that nutation preserves position magnitude."""
    pos = np.array([7000000.0, 1000000.0, 500000.0])
    T = 0.25

    pos_nutated, _ = apply_nutation(pos, T)

    np.testing.assert_allclose(np.linalg.norm(pos_nutated), np.linalg.norm(pos), rtol=1e-14)


def test_nutation_effect_magnitude():
    """Test that nutation effect has expected magnitude."""
    pos = np.array([7000000.0, 0.0, 0.0])  # 7000 km
    T = 0.25

    pos_nutated, _ = apply_nutation(pos, T)

    # Nutation effect depends on direction - along x-axis, the effect is smaller
    # than perpendicular. Typical nutation is ~10-20 arcsec total.
    # At 7000 km, expect 1-1000 m displacement depending on direction
    pos_diff = np.linalg.norm(pos_nutated - pos)
    assert 0 < pos_diff < 1000  # Some measurable change, less than 1 km


def test_nutation_no_velocity():
    """Test nutation without velocity."""
    pos = np.array([7000000.0, 0.0, 0.0])
    T = 0.25

    pos_out, vel_out = apply_nutation(pos, T)

    assert vel_out is None
    assert pos_out is not None


# =============================================================================
# Tests for Full ECI↔ECEF Transformations (with Precession + Nutation)
# =============================================================================


def test_eci_to_ecef_full_basic():
    """Test basic full ECI to ECEF transformation."""
    pos_eci = np.array([7000000.0, 0.0, 0.0])
    vel_eci = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    pos_ecef, vel_ecef = eci_to_ecef_full(pos_eci, timestamp, vel_eci)

    # Position magnitude should be preserved
    np.testing.assert_allclose(np.linalg.norm(pos_ecef), np.linalg.norm(pos_eci), rtol=1e-10)

    # Both outputs should be returned
    assert pos_ecef is not None
    assert vel_ecef is not None
    assert len(pos_ecef) == 3
    assert len(vel_ecef) == 3


def test_ecef_to_eci_full_basic():
    """Test basic full ECEF to ECI transformation."""
    pos_ecef = np.array([7000000.0, 0.0, 0.0])
    vel_ecef = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    pos_eci, vel_eci = ecef_to_eci_full(pos_ecef, timestamp, vel_ecef)

    # Position magnitude should be preserved
    np.testing.assert_allclose(np.linalg.norm(pos_eci), np.linalg.norm(pos_ecef), rtol=1e-10)

    assert pos_eci is not None
    assert vel_eci is not None


def test_full_eci_ecef_roundtrip():
    """Test that full ECI->ECEF->ECI returns original state."""
    pos_eci = np.array([7000000.0, 1000000.0, 500000.0])
    vel_eci = np.array([1000.0, 7500.0, 500.0])
    timestamp = datetime.datetime(2024, 6, 15, 6, 30, 0)

    # Forward transformation
    pos_ecef, vel_ecef = eci_to_ecef_full(pos_eci, timestamp, vel_eci)

    # Inverse transformation
    pos_recovered, vel_recovered = ecef_to_eci_full(pos_ecef, timestamp, vel_ecef)

    # Position roundtrip error should be very small
    np.testing.assert_allclose(pos_recovered, pos_eci, atol=1e-4)

    # Velocity roundtrip error should be very small
    np.testing.assert_allclose(vel_recovered, vel_eci, atol=1e-7)


def test_full_ecef_eci_roundtrip():
    """Test that full ECEF->ECI->ECEF returns original state."""
    pos_ecef = np.array([6378137.0, 1000000.0, 2000000.0])
    vel_ecef = np.array([-500.0, 7000.0, 1000.0])
    timestamp = datetime.datetime(2024, 3, 21, 0, 0, 0)

    # Forward transformation
    pos_eci, vel_eci = ecef_to_eci_full(pos_ecef, timestamp, vel_ecef)

    # Inverse transformation
    pos_recovered, vel_recovered = eci_to_ecef_full(pos_eci, timestamp, vel_eci)

    np.testing.assert_allclose(pos_recovered, pos_ecef, atol=1e-4)
    np.testing.assert_allclose(vel_recovered, vel_ecef, atol=1e-7)


def test_full_differs_from_simple():
    """Test that full transformation differs from simple ERA-only."""
    pos_eci = np.array([7000000.0, 0.0, 0.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    # Full transformation (with precession + nutation)
    pos_ecef_full, _ = eci_to_ecef_full(pos_eci, timestamp)

    # Simple transformation (ERA only)
    pos_ecef_simple = eci_to_ecef(pos_eci, timestamp)

    # They should differ due to precession (~24 years from J2000)
    # Expected ~40-50 km difference at 7000 km range
    diff = np.linalg.norm(pos_ecef_full - pos_ecef_simple)
    assert diff > 10000  # More than 10 km difference
    assert diff < 100000  # Less than 100 km difference


def test_full_transformation_time_variation():
    """Test that full ECI-ECEF transformation varies with time."""
    pos_eci = np.array([7000000.0, 0.0, 0.0])
    t1 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    t2 = datetime.datetime(2024, 1, 1, 6, 0, 0)  # 6 hours later

    pos_ecef_1, _ = eci_to_ecef_full(pos_eci, t1)
    pos_ecef_2, _ = eci_to_ecef_full(pos_eci, t2)

    # 6 hours = 90 degrees of Earth rotation
    # For 7000 km radius, expect ~9900 km displacement
    pos_diff = np.linalg.norm(pos_ecef_1 - pos_ecef_2)
    assert pos_diff > 5000000  # At least 5000 km difference


def test_full_position_only():
    """Test full transformation with position only (no velocity)."""
    pos_eci = np.array([7000000.0, 1000000.0, 500000.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    pos_ecef, vel_ecef = eci_to_ecef_full(pos_eci, timestamp)

    assert pos_ecef is not None
    assert vel_ecef is None

    # Roundtrip
    pos_recovered, vel_recovered = ecef_to_eci_full(pos_ecef, timestamp)

    assert vel_recovered is None
    np.testing.assert_allclose(pos_recovered, pos_eci, atol=1e-4)


def test_full_velocity_transformation():
    """Test that velocity transformation includes omega×r term."""
    pos_eci = np.array([7000000.0, 0.0, 0.0])  # Along x-axis
    vel_eci = np.array([0.0, 0.0, 0.0])  # Zero velocity in ECI
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    _pos_ecef, vel_ecef = eci_to_ecef_full(pos_eci, timestamp, vel_eci)

    # In ECEF, a stationary ECI point should have velocity
    # due to Earth's rotation (omega × r)
    # omega = 7.292115e-5 rad/s, |r| = 7000 km
    # |omega × r| ~ 7.292115e-5 * 7e6 ~ 510 m/s
    vel_mag = np.linalg.norm(vel_ecef)
    assert 400 < vel_mag < 600  # Expect ~500 m/s


def test_full_precession_effect():
    """Test that precession has expected magnitude over ~24 years."""
    pos_eci = np.array([7000000.0, 0.0, 0.0])

    # At J2000.0, precession should be zero
    t_j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    pos_ecef_j2000, _ = eci_to_ecef_full(pos_eci, t_j2000)
    pos_ecef_simple_j2000 = eci_to_ecef(pos_eci, t_j2000)

    # At J2000.0, full and simple should be very close
    diff_j2000 = np.linalg.norm(pos_ecef_j2000 - pos_ecef_simple_j2000)
    assert diff_j2000 < 1000  # Less than 1 km at epoch

    # At 2024, precession (~24 years) creates significant difference
    t_2024 = datetime.datetime(2024, 7, 1, 12, 0, 0)
    pos_ecef_2024, _ = eci_to_ecef_full(pos_eci, t_2024)
    pos_ecef_simple_2024 = eci_to_ecef(pos_eci, t_2024)

    diff_2024 = np.linalg.norm(pos_ecef_2024 - pos_ecef_simple_2024)
    assert diff_2024 > 30000  # More than 30 km after 24 years


# =============================================================================
# Tests for Earth Orientation Parameters (EOP) Interface
# =============================================================================


def test_eop_creation():
    """Test creating EOP data container."""
    epochs = np.array([60000.0, 60001.0, 60002.0])
    x_p = np.array([0.10, 0.11, 0.12])
    y_p = np.array([0.30, 0.31, 0.32])
    ut1_utc = np.array([-0.10, -0.09, -0.08])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    assert eop.start_epoch == 60000.0
    assert eop.end_epoch == 60002.0
    assert len(eop.epochs) == 3


def test_eop_interpolation():
    """Test EOP interpolation."""
    epochs = np.array([60000.0, 60001.0, 60002.0])
    x_p = np.array([0.10, 0.12, 0.14])
    y_p = np.array([0.30, 0.32, 0.34])
    ut1_utc = np.array([-0.10, -0.08, -0.06])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    # Interpolate at midpoint
    x_p_i, y_p_i, ut1_utc_i = eop.interpolate(60000.5)

    np.testing.assert_allclose(x_p_i, 0.11, rtol=1e-10)
    np.testing.assert_allclose(y_p_i, 0.31, rtol=1e-10)
    np.testing.assert_allclose(ut1_utc_i, -0.09, rtol=1e-10)


def test_eop_interpolation_at_bounds():
    """Test EOP interpolation at data boundaries."""
    epochs = np.array([60000.0, 60001.0])
    x_p = np.array([0.10, 0.12])
    y_p = np.array([0.30, 0.32])
    ut1_utc = np.array([-0.10, -0.08])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    # At start
    x_p_i, y_p_i, ut1_utc_i = eop.interpolate(60000.0)
    np.testing.assert_allclose(x_p_i, 0.10)
    np.testing.assert_allclose(y_p_i, 0.30)

    # At end
    x_p_i, y_p_i, _ut1_utc_i = eop.interpolate(60001.0)
    np.testing.assert_allclose(x_p_i, 0.12)
    np.testing.assert_allclose(y_p_i, 0.32)


def test_eop_interpolation_out_of_range():
    """Test that EOP interpolation raises error for out of range epoch."""
    epochs = np.array([60000.0, 60001.0])
    x_p = np.array([0.10, 0.12])
    y_p = np.array([0.30, 0.32])
    ut1_utc = np.array([-0.10, -0.08])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    with pytest.raises(ValueError):
        eop.interpolate(59999.0)  # Before start

    with pytest.raises(ValueError):
        eop.interpolate(60002.0)  # After end


def test_eop_invalid_lengths():
    """Test that EOP raises error for mismatched array lengths."""
    epochs = np.array([60000.0, 60001.0])
    x_p = np.array([0.10, 0.12, 0.14])  # Wrong length
    y_p = np.array([0.30, 0.32])
    ut1_utc = np.array([-0.10, -0.08])

    with pytest.raises(ValueError):
        EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)


def test_eop_interpolate_full():
    """Test EOP full interpolation with optional data."""
    epochs = np.array([60000.0, 60001.0])
    x_p = np.array([0.10, 0.12])
    y_p = np.array([0.30, 0.32])
    ut1_utc = np.array([-0.10, -0.08])
    lod = np.array([1.0, 1.2])
    dX = np.array([0.001, 0.002])

    eop = EarthOrientationParameters(
        epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc, lod=lod, dX=dX
    )

    result = eop.interpolate_full(60000.5)

    assert "x_p" in result
    assert "y_p" in result
    assert "ut1_utc" in result
    assert "lod" in result
    assert "dX" in result
    assert "dY" not in result  # Not provided

    np.testing.assert_allclose(result["lod"], 1.1, rtol=1e-10)
    np.testing.assert_allclose(result["dX"], 0.0015, rtol=1e-10)


def test_datetime_to_mjd():
    """Test datetime to MJD conversion."""
    # J2000.0 epoch has MJD = 51544.5
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    mjd = datetime_to_mjd(j2000)
    np.testing.assert_allclose(mjd, 51544.5, rtol=1e-10)

    # One day later
    day_after = datetime.datetime(2000, 1, 2, 12, 0, 0)
    mjd_day = datetime_to_mjd(day_after)
    np.testing.assert_allclose(mjd_day, 51545.5, rtol=1e-10)


def test_polar_motion_matrix_identity():
    """Test polar motion matrix with zero parameters."""
    W = compute_polar_motion_matrix(0.0, 0.0, 0.0)

    np.testing.assert_allclose(W, np.eye(3), atol=1e-14)


def test_polar_motion_matrix_orthogonal():
    """Test that polar motion matrix is orthogonal."""
    # Typical polar motion values (in radians)
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    x_p = 0.1 * arcsec_to_rad  # ~0.1 arcsec
    y_p = 0.3 * arcsec_to_rad  # ~0.3 arcsec

    W = compute_polar_motion_matrix(x_p, y_p)

    # W @ W.T should be identity
    np.testing.assert_allclose(W @ W.T, np.eye(3), atol=1e-14)

    # det(W) should be 1
    np.testing.assert_allclose(np.linalg.det(W), 1.0, atol=1e-14)


def test_polar_motion_matrix_small_rotation():
    """Test that polar motion is a small rotation."""
    # Typical polar motion is < 1 arcsec
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    x_p = 0.2 * arcsec_to_rad
    y_p = 0.4 * arcsec_to_rad

    W = compute_polar_motion_matrix(x_p, y_p)

    # Matrix should be close to identity
    assert np.allclose(W, np.eye(3), atol=1e-5)


def test_eop_transformation_roundtrip():
    """Test ECI<->ECEF with EOP roundtrip."""
    # Create simple EOP data
    # MJD 60548 = 2024-08-26
    epochs = np.array([60547.0, 60548.0, 60549.0])
    x_p = np.array([0.10, 0.11, 0.12])
    y_p = np.array([0.30, 0.31, 0.32])
    ut1_utc = np.array([-0.10, -0.10, -0.10])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    # Use a timestamp within the EOP range
    # MJD 60548 = 2024-08-26
    timestamp = datetime.datetime(2024, 8, 26, 0, 0, 0)

    pos_eci = np.array([7000000.0, 1000000.0, 500000.0])
    vel_eci = np.array([1000.0, 7500.0, 500.0])

    # Forward transformation
    pos_ecef, vel_ecef = eci_to_ecef_with_eop(pos_eci, timestamp, eop, vel_eci)

    # Inverse transformation
    pos_recovered, vel_recovered = ecef_to_eci_with_eop(pos_ecef, timestamp, eop, vel_ecef)

    # Position roundtrip error should be very small
    np.testing.assert_allclose(pos_recovered, pos_eci, atol=1e-4)

    # Velocity roundtrip error should be very small
    np.testing.assert_allclose(vel_recovered, vel_eci, atol=1e-7)


def test_eop_transformation_position_magnitude():
    """Test that EOP transformation preserves position magnitude."""
    # MJD 60548 = 2024-08-26
    epochs = np.array([60547.0, 60549.0])
    x_p = np.array([0.10, 0.11])
    y_p = np.array([0.30, 0.31])
    ut1_utc = np.array([-0.10, -0.10])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    timestamp = datetime.datetime(2024, 8, 26, 0, 0, 0)
    pos_eci = np.array([7000000.0, 0.0, 0.0])

    pos_ecef, _ = eci_to_ecef_with_eop(pos_eci, timestamp, eop)

    np.testing.assert_allclose(np.linalg.norm(pos_ecef), np.linalg.norm(pos_eci), rtol=1e-10)


def test_eop_differs_from_full():
    """Test that EOP transformation differs from standard full transformation."""
    # MJD 60548 = 2024-08-26
    epochs = np.array([60547.0, 60549.0])
    x_p = np.array([0.20, 0.21])  # ~0.2 arcsec polar motion
    y_p = np.array([0.40, 0.41])  # ~0.4 arcsec polar motion
    ut1_utc = np.array([-0.30, -0.30])  # 300ms UT1-UTC offset

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    timestamp = datetime.datetime(2024, 8, 26, 0, 0, 0)
    pos_eci = np.array([7000000.0, 0.0, 0.0])

    # With EOP
    pos_ecef_eop, _ = eci_to_ecef_with_eop(pos_eci, timestamp, eop)

    # Without EOP (using full transformation)
    pos_ecef_full, _ = eci_to_ecef_full(pos_eci, timestamp)

    # They should differ due to polar motion and UT1 corrections
    # Polar motion of ~0.5 arcsec at 7000 km gives ~17 m offset
    # UT1-UTC of 0.3s gives ~150 m offset at equator
    diff = np.linalg.norm(pos_ecef_eop - pos_ecef_full)
    assert diff > 10  # At least 10 m difference
    assert diff < 500  # But less than 500 m


# =============================================================================
# Performance Benchmarks for Coordinate Transformations
# =============================================================================
# These tests measure transformation performance.
# Run with: pytest -k benchmark -v -s


@pytest.mark.benchmark
def test_benchmark_geodetic_to_ecef():
    """Benchmark geodetic to ECEF transformation performance."""
    import time

    # Test data
    n_iterations = 10000
    lat = np.radians(45.0)
    lon = np.radians(-75.0)
    alt = 1000.0

    # Warmup
    for _ in range(100):
        geodetic_to_ecef(lat, lon, alt)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        geodetic_to_ecef(lat, lon, alt)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\ngeodetic_to_ecef: {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    # Should be fast - less than 100 μs per call
    assert us_per_call < 100


@pytest.mark.benchmark
def test_benchmark_ecef_to_geodetic():
    """Benchmark ECEF to geodetic transformation performance."""
    import time

    n_iterations = 10000
    x, y, z = 4000000.0, 3000000.0, 4500000.0

    # Warmup
    for _ in range(100):
        ecef_to_geodetic(x, y, z)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        ecef_to_geodetic(x, y, z)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\necef_to_geodetic: {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    # Iterative algorithm, allow up to 200 μs
    assert us_per_call < 200


@pytest.mark.benchmark
def test_benchmark_eci_to_ecef_simple():
    """Benchmark simple ECI to ECEF transformation (ERA only)."""
    import time

    n_iterations = 10000
    pos = np.array([7000000.0, 0.0, 0.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    # Warmup
    for _ in range(100):
        eci_to_ecef(pos, timestamp)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        eci_to_ecef(pos, timestamp)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\neci_to_ecef (simple): {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    assert us_per_call < 100


@pytest.mark.benchmark
def test_benchmark_eci_to_ecef_full():
    """Benchmark full ECI to ECEF transformation (precession + nutation)."""
    import time

    n_iterations = 1000
    pos = np.array([7000000.0, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 7, 1, 12, 0, 0)

    # Warmup
    for _ in range(100):
        eci_to_ecef_full(pos, timestamp, vel)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        eci_to_ecef_full(pos, timestamp, vel)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\neci_to_ecef_full: {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    # More complex, allow up to 2000 μs (2 ms) for CI variability
    assert us_per_call < 2000


@pytest.mark.benchmark
def test_benchmark_eci_to_ecef_with_eop():
    """Benchmark high-precision ECI to ECEF transformation with EOP."""
    import time

    # Create EOP data
    epochs = np.array([60547.0, 60548.0, 60549.0])
    x_p = np.array([0.10, 0.11, 0.12])
    y_p = np.array([0.30, 0.31, 0.32])
    ut1_utc = np.array([-0.10, -0.10, -0.10])

    eop = EarthOrientationParameters(epochs=epochs, x_p=x_p, y_p=y_p, ut1_utc=ut1_utc)

    n_iterations = 1000
    pos = np.array([7000000.0, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 8, 26, 0, 0, 0)

    # Warmup
    for _ in range(100):
        eci_to_ecef_with_eop(pos, timestamp, eop, vel)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        eci_to_ecef_with_eop(pos, timestamp, eop, vel)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\neci_to_ecef_with_eop: {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    # Full transformation with EOP, allow up to 1500 μs
    assert us_per_call < 1500


@pytest.mark.benchmark
def test_benchmark_precession_matrix():
    """Benchmark precession matrix computation."""
    import time

    n_iterations = 10000
    T = 0.25  # Julian centuries

    # Warmup
    for _ in range(100):
        compute_precession_matrix_iau2006(T)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        compute_precession_matrix_iau2006(T)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(
        f"\ncompute_precession_matrix_iau2006: {us_per_call:.2f} μs/call ({n_iterations} iterations)"
    )

    assert us_per_call < 100


@pytest.mark.benchmark
def test_benchmark_nutation():
    """Benchmark nutation computation."""
    import time

    n_iterations = 1000
    T = 0.25

    # Warmup
    for _ in range(100):
        compute_nutation_iau2000b(T)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        compute_nutation_iau2000b(T)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / n_iterations) * 1e6
    print(f"\ncompute_nutation_iau2000b: {us_per_call:.2f} μs/call ({n_iterations} iterations)")

    # 77-term series, allow up to 1500 μs for CI variability
    assert us_per_call < 1500


@pytest.mark.benchmark
def test_benchmark_transformation_summary():
    """Print summary of all transformation benchmarks."""
    import time

    results = {}

    # Test parameters
    lat, lon, alt = np.radians(45.0), np.radians(-75.0), 1000.0
    x, y, z = 4000000.0, 3000000.0, 4500000.0
    pos = np.array([7000000.0, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    timestamp = datetime.datetime(2024, 8, 26, 0, 0, 0)
    T = 0.25

    # EOP data
    epochs = np.array([60547.0, 60548.0, 60549.0])
    eop = EarthOrientationParameters(
        epochs=epochs,
        x_p=np.array([0.10, 0.11, 0.12]),
        y_p=np.array([0.30, 0.31, 0.32]),
        ut1_utc=np.array([-0.10, -0.10, -0.10]),
    )

    # Benchmark functions
    benchmarks = [
        ("geodetic_to_ecef", lambda: geodetic_to_ecef(lat, lon, alt), 10000),
        ("ecef_to_geodetic", lambda: ecef_to_geodetic(x, y, z), 10000),
        ("eci_to_ecef (simple)", lambda: eci_to_ecef(pos, timestamp), 10000),
        ("ecef_to_eci (simple)", lambda: ecef_to_eci(pos, timestamp), 10000),
        ("eci_to_ecef_full", lambda: eci_to_ecef_full(pos, timestamp, vel), 1000),
        ("ecef_to_eci_full", lambda: ecef_to_eci_full(pos, timestamp, vel), 1000),
        ("eci_to_ecef_with_eop", lambda: eci_to_ecef_with_eop(pos, timestamp, eop, vel), 1000),
        ("precession_matrix", lambda: compute_precession_matrix_iau2006(T), 10000),
        ("nutation_iau2000b", lambda: compute_nutation_iau2000b(T), 1000),
        ("nutation_matrix", lambda: compute_nutation_matrix(T), 1000),
    ]

    print("\n" + "=" * 60)
    print("Coordinate Transformation Performance Benchmarks")
    print("=" * 60)

    for name, func, n_iter in benchmarks:
        # Warmup
        for _ in range(100):
            func()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iter):
            func()
        elapsed = time.perf_counter() - start

        us_per_call = (elapsed / n_iter) * 1e6
        results[name] = us_per_call
        print(f"{name:30s}: {us_per_call:8.2f} μs/call")

    print("=" * 60)

    # Verify all are reasonably fast
    assert all(v < 2000 for v in results.values())


# =============================================================================
# Validation Against pymap3d (Optional)
# =============================================================================
# These tests compare our implementations against pymap3d for validation.
# Skip if pymap3d is not installed.

try:
    import pymap3d

    HAS_PYMAP3D = True
except ImportError:
    HAS_PYMAP3D = False


@pytest.mark.skipif(not HAS_PYMAP3D, reason="pymap3d not installed")
def test_validate_geodetic_to_ecef_against_pymap3d():
    """Validate geodetic to ECEF transformation against pymap3d."""
    # Test cases covering different locations
    test_cases = [
        (0.0, 0.0, 0.0),  # Equator, prime meridian
        (np.radians(45), np.radians(-75), 1000.0),  # Mid-latitude
        (np.radians(89), np.radians(0), 10000.0),  # Near pole
        (np.radians(-30), np.radians(120), 5000.0),  # Southern hemisphere
    ]

    for lat, lon, alt in test_cases:
        # Our implementation
        xyz_ours = geodetic_to_ecef(lat, lon, alt)

        # pymap3d (uses degrees)
        x, y, z = pymap3d.geodetic2ecef(np.degrees(lat), np.degrees(lon), alt)
        xyz_pymap3d = np.array([x, y, z])

        # Should match within millimeter
        np.testing.assert_allclose(
            xyz_ours,
            xyz_pymap3d,
            atol=1e-3,
            err_msg=f"Mismatch at lat={np.degrees(lat)}, lon={np.degrees(lon)}, alt={alt}",
        )


@pytest.mark.skipif(not HAS_PYMAP3D, reason="pymap3d not installed")
def test_validate_ecef_to_geodetic_against_pymap3d():
    """Validate ECEF to geodetic transformation against pymap3d."""
    test_cases = [
        (6378137.0, 0.0, 0.0),  # On equator
        (4000000.0, 3000000.0, 4500000.0),  # Generic
        (0.0, 0.0, 6356752.0),  # Near pole
        (-5000000.0, -3000000.0, 2000000.0),  # Southern hemisphere
    ]

    for x, y, z in test_cases:
        # Our implementation
        lat_ours, lon_ours, alt_ours = ecef_to_geodetic(x, y, z)

        # pymap3d (returns degrees)
        lat_pm, lon_pm, alt_pm = pymap3d.ecef2geodetic(x, y, z)
        lat_pymap3d = np.radians(lat_pm)
        lon_pymap3d = np.radians(lon_pm)

        # Latitude and longitude should match within micro-degrees
        np.testing.assert_allclose(
            lat_ours, lat_pymap3d, atol=1e-12, err_msg=f"Latitude mismatch at ({x}, {y}, {z})"
        )
        np.testing.assert_allclose(
            lon_ours, lon_pymap3d, atol=1e-12, err_msg=f"Longitude mismatch at ({x}, {y}, {z})"
        )
        # Altitude should match within 1 meter (different algorithms have small differences)
        np.testing.assert_allclose(
            alt_ours, alt_pm, atol=1.0, err_msg=f"Altitude mismatch at ({x}, {y}, {z})"
        )


@pytest.mark.skipif(not HAS_PYMAP3D, reason="pymap3d not installed")
def test_validate_roundtrip_against_pymap3d():
    """Validate roundtrip accuracy of our implementation."""
    # Generate random test points
    np.random.seed(42)
    n_tests = 100

    for _ in range(n_tests):
        lat = np.random.uniform(-89, 89)  # degrees
        lon = np.random.uniform(-180, 180)  # degrees
        alt = np.random.uniform(-500, 100000)  # meters

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Our roundtrip
        xyz = geodetic_to_ecef(lat_rad, lon_rad, alt)
        lat2, _lon2, alt2 = ecef_to_geodetic(xyz[0], xyz[1], xyz[2])

        # pymap3d roundtrip
        x, y, z = pymap3d.geodetic2ecef(lat, lon, alt)
        lat_pm, _lon_pm, alt_pm = pymap3d.ecef2geodetic(x, y, z)

        # Our roundtrip should be as accurate as pymap3d (with tolerance for algorithm differences)
        # Different geodetic algorithms can have differences up to ~1e-4 degrees (~10m) at high latitudes
        assert abs(np.degrees(lat2) - lat) < abs(lat_pm - lat) + 1e-4
        assert abs(alt2 - alt) < abs(alt_pm - alt) + 10.0
