"""Test coordinate system types and transformations."""
import datetime

import numpy as np
import pytest

from stonesoup.types.coordinates import (
    ReferenceEllipsoid,
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
)
from stonesoup.functions.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    eci_to_ecef,
    ecef_to_eci,
    geodetic_to_eci,
    eci_to_geodetic,
)


# Tests for ReferenceEllipsoid class


def test_ellipsoid_creation():
    """Test creating a custom reference ellipsoid."""
    ellipsoid = ReferenceEllipsoid(
        name="Custom",
        semi_major_axis=6378137.0,
        flattening=1/298.257223563
    )
    assert ellipsoid.name == "Custom"
    assert ellipsoid.semi_major_axis == 6378137.0
    assert ellipsoid.flattening == 1/298.257223563


def test_ellipsoid_semi_minor_axis():
    """Test semi-minor axis calculation."""
    # WGS84 parameters
    a = 6378137.0
    f = 1/298.257223563
    expected_b = a * (1.0 - f)

    ellipsoid = ReferenceEllipsoid(
        name="Test",
        semi_major_axis=a,
        flattening=f
    )
    assert pytest.approx(ellipsoid.semi_minor_axis, rel=1e-10) == expected_b
    # Known value for WGS84
    assert pytest.approx(ellipsoid.semi_minor_axis, abs=0.001) == 6356752.314


def test_ellipsoid_eccentricity():
    """Test eccentricity calculations."""
    # WGS84 parameters
    a = 6378137.0
    f = 1/298.257223563
    expected_e = np.sqrt(2.0 * f - f ** 2)

    ellipsoid = ReferenceEllipsoid(
        name="Test",
        semi_major_axis=a,
        flattening=f
    )
    assert pytest.approx(ellipsoid.eccentricity, rel=1e-10) == expected_e
    # Known value for WGS84
    assert pytest.approx(ellipsoid.eccentricity, abs=1e-10) == 0.0818191908


def test_ellipsoid_eccentricity_squared():
    """Test squared eccentricity calculation."""
    a = 6378137.0
    f = 1/298.257223563
    expected_e2 = 2.0 * f - f ** 2

    ellipsoid = ReferenceEllipsoid(
        name="Test",
        semi_major_axis=a,
        flattening=f
    )
    assert pytest.approx(ellipsoid.eccentricity_squared, rel=1e-10) == expected_e2
    assert pytest.approx(ellipsoid.eccentricity_squared) == \
        ellipsoid.eccentricity ** 2


def test_ellipsoid_second_eccentricity_squared():
    """Test second eccentricity squared calculation."""
    a = 6378137.0
    b = 6356752.314
    expected_e_prime2 = (a**2 - b**2) / b**2

    ellipsoid = ReferenceEllipsoid(
        name="Test",
        semi_major_axis=a,
        flattening=1/298.257223563
    )
    assert pytest.approx(ellipsoid.second_eccentricity_squared, abs=1e-8) == \
        expected_e_prime2


def test_ellipsoid_linear_eccentricity():
    """Test linear eccentricity calculation."""
    a = 6378137.0
    f = 1/298.257223563

    ellipsoid = ReferenceEllipsoid(
        name="Test",
        semi_major_axis=a,
        flattening=f
    )
    expected_E = a * ellipsoid.eccentricity
    assert pytest.approx(ellipsoid.linear_eccentricity) == expected_E
    # Also verify against sqrt(a^2 - b^2)
    assert pytest.approx(ellipsoid.linear_eccentricity) == \
        np.sqrt(a**2 - ellipsoid.semi_minor_axis**2)


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
    realizations = [WGS84_G730, WGS84_G873, WGS84_G1150,
                   WGS84_G1674, WGS84_G1762, WGS84_G2139]

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


@pytest.mark.parametrize('ellipsoid', [
    WGS84, WGS84_G730, WGS84_G873, WGS84_G1150, WGS84_G1674,
    WGS84_G1762, WGS84_G2139, GRS80, WGS72, PZ90, CGCS2000
])
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


@pytest.mark.parametrize('lat,lon', [
    (np.radians(0), np.radians(0)),      # Equator, Prime Meridian
    (np.radians(0), np.radians(90)),     # Equator, 90°E
    (np.radians(0), np.radians(180)),    # Equator, 180°
    (np.radians(0), np.radians(-90)),    # Equator, 90°W
    (np.radians(45), np.radians(0)),     # 45°N, Prime Meridian
    (np.radians(-45), np.radians(0)),    # 45°S, Prime Meridian
    (np.radians(30), np.radians(120)),   # 30°N, 120°E
])
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
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-9) == np.pi / 2
    assert pytest.approx(alt, abs=0.01) == 0.0


def test_ecef_to_geodetic_south_pole():
    """Test conversion from ECEF at South Pole."""
    x, y, z = 0.0, 0.0, -WGS84.semi_minor_axis
    lat, lon, alt = ecef_to_geodetic(x, y, z)

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
    lat, lon, alt = ecef_to_geodetic(x, y, z)

    # Should be very close to North Pole
    assert pytest.approx(lat, abs=1e-6) == np.pi / 2


# Tests for round-trip conversions between geodetic and ECEF


@pytest.mark.parametrize('lat,lon,alt', [
    (0.0, 0.0, 0.0),                              # Equator, sea level
    (np.radians(51.4769), np.radians(-0.0005), 0.0),  # London
    (np.radians(40.7128), np.radians(-74.0060), 10.0),  # New York
    (np.radians(-33.8688), np.radians(151.2093), 0.0),  # Sydney
    (np.radians(35.6762), np.radians(139.6503), 40.0),  # Tokyo
    (np.pi/2, 0.0, 0.0),                          # North Pole
    (-np.pi/2, 0.0, 0.0),                         # South Pole
    (np.radians(45), np.radians(90), 1000.0),     # 45°N, 90°E, 1km alt
])
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
    if abs(lat) < np.pi/2 - 1e-6:
        # Normalize longitude to [-pi, pi]
        lon_normalized = np.arctan2(np.sin(lon), np.cos(lon))
        lon2_normalized = np.arctan2(np.sin(lon2), np.cos(lon2))
        assert pytest.approx(lon2_normalized, abs=1e-7) == lon_normalized
    # Altitude tolerance of 0.5m accounts for iterative algorithm convergence
    assert pytest.approx(alt2, abs=0.5) == alt


@pytest.mark.parametrize('x,y,z', [
    (6378137.0, 0.0, 0.0),           # On equator
    (0.0, 6378137.0, 0.0),           # On equator, 90°E
    (0.0, 0.0, 6356752.314),         # North Pole
    (3980574.2, -0.4, 4966894.1),    # London
    (4000000.0, 3000000.0, 4000000.0),  # Arbitrary point
])
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


@pytest.mark.parametrize('eci_pos', [
    np.array([6378137.0, 0.0, 0.0]),
    np.array([0.0, 6378137.0, 0.0]),
    np.array([0.0, 0.0, 6378137.0]),
    np.array([4000000.0, 3000000.0, 4000000.0]),
])
def test_eci_ecef_round_trip(eci_pos):
    """Test that ECI -> ECEF -> ECI preserves values."""
    timestamp = datetime.datetime(2024, 6, 15, 18, 30, 0)

    ecef = eci_to_ecef(eci_pos, timestamp)
    eci_back = ecef_to_eci(ecef, timestamp)

    assert pytest.approx(eci_back[0], abs=1e-6) == eci_pos[0]
    assert pytest.approx(eci_back[1], abs=1e-6) == eci_pos[1]
    assert pytest.approx(eci_back[2], abs=1e-6) == eci_pos[2]


@pytest.mark.parametrize('ecef_pos', [
    np.array([6378137.0, 0.0, 0.0]),
    np.array([0.0, 6378137.0, 0.0]),
    np.array([0.0, 0.0, 6378137.0]),
    np.array([4000000.0, 3000000.0, 4000000.0]),
])
def test_ecef_eci_round_trip(ecef_pos):
    """Test that ECEF -> ECI -> ECEF preserves values."""
    timestamp = datetime.datetime(2024, 6, 15, 18, 30, 0)

    eci = ecef_to_eci(ecef_pos, timestamp)
    ecef_back = eci_to_ecef(eci, timestamp)

    assert pytest.approx(ecef_back[0], abs=1e-6) == ecef_pos[0]
    assert pytest.approx(ecef_back[1], abs=1e-6) == ecef_pos[1]
    assert pytest.approx(ecef_back[2], abs=1e-6) == ecef_pos[2]


@pytest.mark.parametrize('pos', [
    np.array([7000000.0, 0.0, 0.0]),
    np.array([0.0, 7000000.0, 0.0]),
    np.array([0.0, 0.0, 7000000.0]),
    np.array([4000000.0, 3000000.0, 5000000.0]),
])
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


@pytest.mark.parametrize('z_pos', [
    np.array([0.0, 0.0, 6378137.0]),
    np.array([0.0, 0.0, -6378137.0]),
])
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
    lat_expected, lon_expected, alt_expected = ecef_to_geodetic(
        ecef[0], ecef[1], ecef[2]
    )

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
    lat = np.pi/2 - 1e-6
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
