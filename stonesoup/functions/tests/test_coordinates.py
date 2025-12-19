"""Tests for coordinate transformation functions.

This module tests the coordinate transformation functions including:
- ECEF <-> Geodetic conversions
- ECI <-> ECEF conversions
- Latitude conversions (geodetic, geocentric, parametric)
- Frame transformations with precession/nutation
"""

from datetime import datetime

import numpy as np

from stonesoup.functions.coordinates import (
    apply_nutation,
    compute_frame_bias_matrix,
    compute_fundamental_arguments,
    compute_nutation_matrix,
    compute_polar_motion_matrix,
    datetime_to_mjd,
    ecef_to_eci,
    ecef_to_eci_full,
    ecef_to_geodetic,
    eci_to_ecef,
    eci_to_ecef_full,
    eci_to_geodetic,
    geocentric_to_geodetic_latitude,
    geodetic_to_ecef,
    geodetic_to_eci,
    geodetic_to_geocentric_latitude,
    geodetic_to_parametric_latitude,
    parametric_to_geodetic_latitude,
)
from stonesoup.types.coordinates import WGS84


class TestGeodeticECEF:
    """Tests for geodetic to ECEF conversions."""

    def test_geodetic_to_ecef_origin(self):
        """Test conversion at equator/prime meridian."""
        # At equator, prime meridian, zero altitude
        ecef = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert ecef.shape == (3,)
        # X should be approximately semi-major axis
        assert np.isclose(ecef[0], WGS84.semi_major_axis, rtol=1e-6)
        assert np.isclose(ecef[1], 0.0, atol=1e-6)
        assert np.isclose(ecef[2], 0.0, atol=1e-6)

    def test_geodetic_to_ecef_north_pole(self):
        """Test conversion at North Pole."""
        ecef = geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
        assert ecef.shape == (3,)
        # At pole, X and Y should be ~0, Z should be semi-minor axis
        assert np.isclose(ecef[0], 0.0, atol=1e-6)
        assert np.isclose(ecef[1], 0.0, atol=1e-6)
        assert np.isclose(ecef[2], WGS84.semi_minor_axis, rtol=1e-6)

    def test_geodetic_to_ecef_south_pole(self):
        """Test conversion at South Pole."""
        ecef = geodetic_to_ecef(-np.pi / 2, 0.0, 0.0)
        assert ecef.shape == (3,)
        assert np.isclose(ecef[0], 0.0, atol=1e-6)
        assert np.isclose(ecef[1], 0.0, atol=1e-6)
        assert np.isclose(ecef[2], -WGS84.semi_minor_axis, rtol=1e-6)

    def test_geodetic_to_ecef_with_altitude(self):
        """Test conversion with altitude."""
        altitude = 10000.0  # 10 km altitude
        ecef = geodetic_to_ecef(0.0, 0.0, altitude)
        expected_x = WGS84.semi_major_axis + altitude
        assert np.isclose(ecef[0], expected_x, rtol=1e-6)

    def test_geodetic_to_ecef_arbitrary_point(self):
        """Test conversion at arbitrary point (London)."""
        lat = np.radians(51.5074)  # London latitude
        lon = np.radians(-0.1278)  # London longitude
        alt = 0.0
        ecef = geodetic_to_ecef(lat, lon, alt)
        # Result should be approximately at Earth's surface
        radius = np.sqrt(ecef[0] ** 2 + ecef[1] ** 2 + ecef[2] ** 2)
        assert WGS84.semi_minor_axis < radius < WGS84.semi_major_axis + 100

    def test_ecef_to_geodetic_origin(self):
        """Test ECEF to geodetic at equator/prime meridian."""
        ecef = np.array([WGS84.semi_major_axis, 0.0, 0.0])
        lat, lon, alt = ecef_to_geodetic(ecef)
        assert np.isclose(lat, 0.0, atol=1e-10)
        assert np.isclose(lon, 0.0, atol=1e-10)
        assert np.isclose(alt, 0.0, atol=1.0)  # Sub-meter accuracy

    def test_ecef_to_geodetic_north_pole(self):
        """Test ECEF to geodetic at North Pole."""
        ecef = np.array([0.0, 0.0, WGS84.semi_minor_axis])
        lat, lon, alt = ecef_to_geodetic(ecef)
        assert np.isclose(lat, np.pi / 2, atol=1e-10)
        assert np.isclose(alt, 0.0, atol=1.0)

    def test_geodetic_ecef_roundtrip(self):
        """Test roundtrip conversion geodetic -> ECEF -> geodetic."""
        lat_orig = np.radians(45.0)
        lon_orig = np.radians(90.0)
        alt_orig = 5000.0

        ecef = geodetic_to_ecef(lat_orig, lon_orig, alt_orig)
        lat, lon, alt = ecef_to_geodetic(ecef)

        assert np.isclose(lat, lat_orig, atol=1e-10)
        assert np.isclose(lon, lon_orig, atol=1e-10)
        assert np.isclose(alt, alt_orig, atol=0.01)  # cm accuracy


class TestECIECEF:
    """Tests for ECI to ECEF conversions."""

    def test_eci_to_ecef_returns_correct_shape(self):
        """Test that ECI to ECEF returns correct shape."""
        eci_coords = np.array([7000e3, 0.0, 0.0])
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        ecef = eci_to_ecef(eci_coords, timestamp)
        assert ecef.shape == (3,)

    def test_ecef_to_eci_returns_correct_shape(self):
        """Test that ECEF to ECI returns correct shape."""
        ecef_coords = np.array([7000e3, 0.0, 0.0])
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        eci = ecef_to_eci(ecef_coords, timestamp)
        assert eci.shape == (3,)

    def test_eci_ecef_roundtrip(self):
        """Test roundtrip ECI -> ECEF -> ECI."""
        eci_orig = np.array([7000e3, 1000e3, 500e3])
        timestamp = datetime(2024, 6, 15, 18, 30, 0)

        ecef = eci_to_ecef(eci_orig, timestamp)
        eci_back = ecef_to_eci(ecef, timestamp)

        assert np.allclose(eci_orig, eci_back, rtol=1e-10)

    def test_eci_ecef_preserves_magnitude(self):
        """Test that transformation preserves vector magnitude."""
        eci_coords = np.array([7000e3, 2000e3, 1000e3])
        timestamp = datetime(2024, 3, 20, 0, 0, 0)

        ecef = eci_to_ecef(eci_coords, timestamp)

        eci_mag = np.linalg.norm(eci_coords)
        ecef_mag = np.linalg.norm(ecef)

        assert np.isclose(eci_mag, ecef_mag, rtol=1e-10)

    def test_eci_to_ecef_different_times(self):
        """Test ECI to ECEF produces different results at different times."""
        eci_coords = np.array([7000e3, 0.0, 0.0])
        timestamp1 = datetime(2024, 1, 1, 0, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 6, 0, 0)  # 6 hours later

        ecef1 = eci_to_ecef(eci_coords, timestamp1)
        ecef2 = eci_to_ecef(eci_coords, timestamp2)

        # Results should differ due to Earth rotation
        assert not np.allclose(ecef1, ecef2)


class TestGeodeticECI:
    """Tests for geodetic to ECI conversions."""

    def test_geodetic_to_eci_returns_correct_shape(self):
        """Test geodetic to ECI returns correct shape."""
        lat = np.radians(45.0)
        lon = np.radians(-90.0)
        alt = 100e3
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        eci = geodetic_to_eci(lat, lon, alt, timestamp)
        assert eci.shape == (3,)

    def test_eci_to_geodetic_returns_correct_shape(self):
        """Test ECI to geodetic returns correct values."""
        eci_coords = np.array([7000e3, 0.0, 0.0])
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        lat, lon, alt = eci_to_geodetic(eci_coords, timestamp)

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(alt, float)

    def test_geodetic_eci_roundtrip(self):
        """Test roundtrip geodetic -> ECI -> geodetic."""
        lat_orig = np.radians(30.0)
        lon_orig = np.radians(-45.0)
        alt_orig = 200e3
        timestamp = datetime(2024, 6, 1, 6, 0, 0)

        eci = geodetic_to_eci(lat_orig, lon_orig, alt_orig, timestamp)
        lat, lon, alt = eci_to_geodetic(eci, timestamp)

        assert np.isclose(lat, lat_orig, atol=1e-9)
        assert np.isclose(lon, lon_orig, atol=1e-9)
        assert np.isclose(alt, alt_orig, atol=1.0)  # meter accuracy


class TestLatitudeConversions:
    """Tests for latitude conversion functions."""

    def test_geodetic_to_geocentric_at_equator(self):
        """Test geodetic to geocentric at equator (should be zero)."""
        geocentric = geodetic_to_geocentric_latitude(0.0)
        assert np.isclose(geocentric, 0.0, atol=1e-15)

    def test_geodetic_to_geocentric_at_pole(self):
        """Test geodetic to geocentric at poles (should be pi/2 or -pi/2)."""
        geocentric_north = geodetic_to_geocentric_latitude(np.pi / 2)
        geocentric_south = geodetic_to_geocentric_latitude(-np.pi / 2)
        assert np.isclose(geocentric_north, np.pi / 2, atol=1e-10)
        assert np.isclose(geocentric_south, -np.pi / 2, atol=1e-10)

    def test_geocentric_to_geodetic_at_equator(self):
        """Test geocentric to geodetic at equator."""
        geodetic = geocentric_to_geodetic_latitude(0.0)
        assert np.isclose(geodetic, 0.0, atol=1e-15)

    def test_geodetic_geocentric_roundtrip(self):
        """Test roundtrip geodetic -> geocentric -> geodetic."""
        geodetic_orig = np.radians(45.0)
        geocentric = geodetic_to_geocentric_latitude(geodetic_orig)
        geodetic_back = geocentric_to_geodetic_latitude(geocentric)
        assert np.isclose(geodetic_orig, geodetic_back, atol=1e-12)

    def test_geodetic_to_parametric_at_equator(self):
        """Test geodetic to parametric at equator."""
        parametric = geodetic_to_parametric_latitude(0.0)
        assert np.isclose(parametric, 0.0, atol=1e-15)

    def test_parametric_to_geodetic_at_equator(self):
        """Test parametric to geodetic at equator."""
        geodetic = parametric_to_geodetic_latitude(0.0)
        assert np.isclose(geodetic, 0.0, atol=1e-15)

    def test_geodetic_parametric_roundtrip(self):
        """Test roundtrip geodetic -> parametric -> geodetic."""
        geodetic_orig = np.radians(60.0)
        parametric = geodetic_to_parametric_latitude(geodetic_orig)
        geodetic_back = parametric_to_geodetic_latitude(parametric)
        assert np.isclose(geodetic_orig, geodetic_back, atol=1e-12)

    def test_latitude_ordering(self):
        """Test that geocentric < parametric < geodetic for positive latitudes."""
        geodetic = np.radians(45.0)
        geocentric = geodetic_to_geocentric_latitude(geodetic)
        parametric = geodetic_to_parametric_latitude(geodetic)

        # For oblate Earth, geocentric < parametric < geodetic
        assert geocentric < parametric < geodetic


class TestFrameTransformations:
    """Tests for frame transformation matrices."""

    def test_frame_bias_matrix_shape(self):
        """Test frame bias matrix has correct shape."""
        B = compute_frame_bias_matrix()
        assert B.shape == (3, 3)

    def test_frame_bias_matrix_orthogonal(self):
        """Test frame bias matrix is orthogonal."""
        B = compute_frame_bias_matrix()
        identity = np.eye(3)
        assert np.allclose(B @ B.T, identity, atol=1e-12)
        assert np.allclose(B.T @ B, identity, atol=1e-12)

    def test_fundamental_arguments_returns_dict(self):
        """Test fundamental arguments returns expected keys."""
        T = 0.5  # Half a century from J2000
        args = compute_fundamental_arguments(T)
        expected_keys = {"l", "l_prime", "F", "D", "Omega"}
        assert expected_keys <= set(args.keys())

    def test_nutation_matrix_shape(self):
        """Test nutation matrix has correct shape."""
        T = 0.1
        N = compute_nutation_matrix(T)
        assert N.shape == (3, 3)

    def test_nutation_matrix_orthogonal(self):
        """Test nutation matrix is orthogonal."""
        T = 0.2
        N = compute_nutation_matrix(T)
        identity = np.eye(3)
        assert np.allclose(N @ N.T, identity, atol=1e-10)

    def test_polar_motion_matrix_shape(self):
        """Test polar motion matrix has correct shape."""
        W = compute_polar_motion_matrix(0.0, 0.0)
        assert W.shape == (3, 3)

    def test_polar_motion_matrix_identity_at_zero(self):
        """Test polar motion matrix is identity when x_p=y_p=0."""
        W = compute_polar_motion_matrix(0.0, 0.0)
        identity = np.eye(3)
        assert np.allclose(W, identity, atol=1e-12)


class TestFullTransformations:
    """Tests for full ECI-ECEF transformations with precession/nutation."""

    def test_eci_to_ecef_full_returns_correct_shape(self):
        """Test full ECI to ECEF returns correct shape."""
        eci_coords = np.array([7000e3, 1000e3, 500e3])
        timestamp = datetime(2024, 6, 15, 12, 0, 0)

        ecef = eci_to_ecef_full(eci_coords, timestamp)
        assert ecef.shape == (3,)

    def test_ecef_to_eci_full_returns_correct_shape(self):
        """Test full ECEF to ECI returns correct shape."""
        ecef_coords = np.array([7000e3, 1000e3, 500e3])
        timestamp = datetime(2024, 6, 15, 12, 0, 0)

        eci = ecef_to_eci_full(ecef_coords, timestamp)
        assert eci.shape == (3,)

    def test_eci_ecef_full_roundtrip(self):
        """Test roundtrip full ECI -> ECEF -> ECI."""
        eci_orig = np.array([6800e3, 1500e3, 700e3])
        timestamp = datetime(2024, 3, 21, 0, 0, 0)

        ecef = eci_to_ecef_full(eci_orig, timestamp)
        eci_back = ecef_to_eci_full(ecef, timestamp)

        assert np.allclose(eci_orig, eci_back, rtol=1e-9)

    def test_full_preserves_magnitude(self):
        """Test full transformation preserves vector magnitude."""
        eci_coords = np.array([7000e3, 2000e3, 1000e3])
        timestamp = datetime(2024, 9, 22, 12, 0, 0)

        ecef = eci_to_ecef_full(eci_coords, timestamp)

        eci_mag = np.linalg.norm(eci_coords)
        ecef_mag = np.linalg.norm(ecef)

        assert np.isclose(eci_mag, ecef_mag, rtol=1e-10)


class TestNutationApplication:
    """Tests for nutation application."""

    def test_apply_nutation_returns_correct_shape(self):
        """Test apply_nutation returns correct shape."""
        coords = np.array([7000e3, 1000e3, 500e3])
        timestamp = datetime(2024, 1, 1, 0, 0, 0)

        result = apply_nutation(coords, timestamp)
        assert result.shape == (3,)

    def test_apply_nutation_preserves_magnitude(self):
        """Test apply_nutation preserves vector magnitude."""
        coords = np.array([7000e3, 2000e3, 1000e3])
        timestamp = datetime(2024, 6, 21, 12, 0, 0)

        result = apply_nutation(coords, timestamp)

        orig_mag = np.linalg.norm(coords)
        result_mag = np.linalg.norm(result)

        assert np.isclose(orig_mag, result_mag, rtol=1e-10)


class TestDatetimeToMJD:
    """Tests for datetime to MJD conversion."""

    def test_datetime_to_mjd_j2000(self):
        """Test MJD at J2000 epoch."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        mjd = datetime_to_mjd(j2000)
        # J2000 MJD is 51544.5
        assert np.isclose(mjd, 51544.5, atol=0.001)

    def test_datetime_to_mjd_ordering(self):
        """Test that later dates have larger MJD."""
        date1 = datetime(2020, 1, 1, 0, 0, 0)
        date2 = datetime(2024, 1, 1, 0, 0, 0)

        mjd1 = datetime_to_mjd(date1)
        mjd2 = datetime_to_mjd(date2)

        assert mjd2 > mjd1

    def test_datetime_to_mjd_one_day_difference(self):
        """Test that one day difference equals 1.0 in MJD."""
        date1 = datetime(2024, 6, 15, 12, 0, 0)
        date2 = datetime(2024, 6, 16, 12, 0, 0)

        mjd1 = datetime_to_mjd(date1)
        mjd2 = datetime_to_mjd(date2)

        assert np.isclose(mjd2 - mjd1, 1.0, atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_geodetic_ecef_near_poles(self):
        """Test geodetic to ECEF near poles doesn't cause issues."""
        # Very close to North Pole
        lat = np.radians(89.99999)
        lon = np.radians(45.0)
        alt = 0.0

        ecef = geodetic_to_ecef(lat, lon, alt)
        assert np.all(np.isfinite(ecef))

    def test_geodetic_ecef_date_line(self):
        """Test geodetic to ECEF at date line (180 degrees)."""
        lat = np.radians(0.0)
        lon = np.pi  # 180 degrees
        alt = 0.0

        ecef = geodetic_to_ecef(lat, lon, alt)
        assert np.all(np.isfinite(ecef))
        assert ecef[0] < 0  # X should be negative at 180 degrees

    def test_eci_ecef_at_midnight(self):
        """Test ECI-ECEF at midnight UTC."""
        eci_coords = np.array([7000e3, 0.0, 0.0])
        timestamp = datetime(2024, 1, 1, 0, 0, 0)

        ecef = eci_to_ecef(eci_coords, timestamp)
        assert np.all(np.isfinite(ecef))

    def test_high_altitude_conversion(self):
        """Test conversion at very high altitude (GEO)."""
        lat = np.radians(0.0)
        lon = np.radians(0.0)
        alt = 35786e3  # GEO altitude

        ecef = geodetic_to_ecef(lat, lon, alt)
        lat_back, lon_back, alt_back = ecef_to_geodetic(ecef)

        assert np.isclose(lat, lat_back, atol=1e-10)
        assert np.isclose(lon, lon_back, atol=1e-10)
        assert np.isclose(alt, alt_back, rtol=1e-6)
