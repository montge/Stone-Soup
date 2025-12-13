"""Tests for underwater/undersea tracking utility functions."""

import numpy as np
import pytest
from pytest import approx

from ..underwater import (
    acoustic_attenuation,
    apply_current_to_velocity,
    apply_tide_to_depth,
    bearing_elevation_range2cart,
    cart2bearing_elevation_range,
    cart2depth_bearing_range,
    compute_sound_speed_gradient,
    create_canyon_bathymetry,
    create_depth_varying_current,
    create_flat_bathymetry,
    create_isothermal_profile,
    create_mixed_layer_profile,
    create_shear_current,
    create_sloped_bathymetry,
    create_thermocline_profile,
    create_uniform_current,
    depth_bearing_range2cart,
    depth_to_pressure,
    dual_constituent_tide,
    ecef_to_enu,
    ecef_to_geodetic_underwater,
    enu_to_ecef,
    enu_to_geodetic_depth,
    find_sound_channel_axis,
    geodetic_depth_to_enu,
    geodetic_to_ecef_underwater,
    haversine_distance,
    height_above_seafloor,
    interpolate_bathymetry,
    interpolate_profile,
    is_in_water,
    pressure_to_depth,
    simple_tidal_offset,
    sound_speed_mackenzie,
    sound_speed_profile,
    sound_speed_unesco,
    tidal_current,
    transmission_loss,
)

# =============================================================================
# Sound Speed Tests
# =============================================================================


def test_mackenzie_standard_conditions():
    """Test Mackenzie equation at standard ocean conditions."""
    # Standard conditions: T=10°C, S=35 PSU, D=0m
    c = sound_speed_mackenzie(temperature=10.0, salinity=35.0, depth=0.0)
    # Expected ~1490 m/s at surface
    assert 1480 < c < 1500


def test_mackenzie_temperature_effect():
    """Sound speed should increase with temperature."""
    c_cold = sound_speed_mackenzie(temperature=5.0, salinity=35.0, depth=0.0)
    c_warm = sound_speed_mackenzie(temperature=20.0, salinity=35.0, depth=0.0)
    assert c_warm > c_cold


def test_mackenzie_depth_effect():
    """Sound speed should increase with depth (pressure)."""
    c_shallow = sound_speed_mackenzie(temperature=10.0, salinity=35.0, depth=0.0)
    c_deep = sound_speed_mackenzie(temperature=10.0, salinity=35.0, depth=1000.0)
    assert c_deep > c_shallow


def test_mackenzie_salinity_effect():
    """Sound speed should increase with salinity."""
    c_fresh = sound_speed_mackenzie(temperature=10.0, salinity=30.0, depth=0.0)
    c_salty = sound_speed_mackenzie(temperature=10.0, salinity=40.0, depth=0.0)
    assert c_salty > c_fresh


def test_mackenzie_known_value():
    """Test against known reference value."""
    # At T=10°C, S=35 PSU, D=0m, expect ~1490 m/s
    c = sound_speed_mackenzie(temperature=10.0, salinity=35.0, depth=0.0)
    assert c == approx(1490, rel=0.01)


def test_unesco_standard_conditions():
    """Test UNESCO algorithm at standard conditions."""
    c = sound_speed_unesco(temperature=10.0, salinity=35.0, pressure=0.0)
    assert 1480 < c < 1500


def test_unesco_temperature_effect():
    """Sound speed should increase with temperature."""
    c_cold = sound_speed_unesco(temperature=5.0, salinity=35.0, pressure=0.0)
    c_warm = sound_speed_unesco(temperature=20.0, salinity=35.0, pressure=0.0)
    assert c_warm > c_cold


def test_mackenzie_unesco_agreement():
    """Mackenzie and UNESCO should give similar results."""
    c_mack = sound_speed_mackenzie(temperature=15.0, salinity=35.0, depth=100.0)
    c_unesco = sound_speed_unesco(temperature=15.0, salinity=35.0, pressure=100.0)
    # Should agree within 1%
    assert c_mack == approx(c_unesco, rel=0.01)


# =============================================================================
# Pressure-Depth Conversion Tests
# =============================================================================


def test_pressure_to_depth_surface():
    """Zero pressure should give zero depth."""
    depth = pressure_to_depth(0.0)
    assert depth == approx(0.0, abs=1e-6)


def test_pressure_to_depth_approximate():
    """Pressure in dbar ≈ depth in meters for seawater."""
    depth = pressure_to_depth(100.0)
    # Should be approximately 100m (within 2%)
    assert depth == approx(100.0, rel=0.02)


def test_pressure_to_depth_deep():
    """Test pressure-depth at greater depths."""
    depth = pressure_to_depth(4000.0)
    # At 4000 dbar, depth should be around 3932m (slightly less than 4000m)
    assert 3900 < depth < 4000


def test_pressure_to_depth_latitude_effect():
    """Depth should vary slightly with latitude."""
    depth_equator = pressure_to_depth(1000.0, latitude=0.0)
    depth_pole = pressure_to_depth(1000.0, latitude=90.0)
    # Gravity is stronger at poles, so same pressure = less depth
    assert depth_pole < depth_equator


def test_depth_to_pressure_roundtrip():
    """depth_to_pressure(pressure_to_depth(p)) ≈ p."""
    original_pressure = 500.0
    depth = pressure_to_depth(original_pressure)
    recovered_pressure = depth_to_pressure(depth)
    assert recovered_pressure == approx(original_pressure, rel=1e-4)


@pytest.mark.parametrize("latitude", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_depth_to_pressure_various_latitudes(latitude):
    """Test roundtrip at different latitudes."""
    depth = 200.0
    pressure = depth_to_pressure(depth, latitude)
    recovered_depth = pressure_to_depth(pressure, latitude)
    assert recovered_depth == approx(depth, rel=1e-4)


# =============================================================================
# Acoustic Attenuation Tests
# =============================================================================


def test_attenuation_positive():
    """Attenuation should always be positive."""
    alpha = acoustic_attenuation(frequency=10.0)
    assert alpha > 0


def test_attenuation_frequency_dependence():
    """Attenuation should increase with frequency."""
    alpha_low = acoustic_attenuation(frequency=1.0)
    alpha_high = acoustic_attenuation(frequency=100.0)
    assert alpha_high > alpha_low


def test_attenuation_standard_conditions():
    """Test attenuation at standard conditions."""
    # At 10 kHz, typical attenuation ~1 dB/km
    alpha = acoustic_attenuation(frequency=10.0, temperature=10.0, salinity=35.0)
    assert 0.5 < alpha < 2.0


def test_attenuation_low_frequency():
    """Low frequency should have low attenuation."""
    alpha = acoustic_attenuation(frequency=0.1)
    assert alpha < 0.1  # Very low attenuation at 100 Hz


# =============================================================================
# Transmission Loss Tests
# =============================================================================


def test_transmission_loss_zero_distance():
    """Zero distance should give zero loss."""
    tl = transmission_loss(distance=0.0, frequency=10.0)
    assert tl == 0.0


def test_transmission_loss_increasing():
    """TL should increase with distance."""
    tl_near = transmission_loss(distance=100.0, frequency=10.0)
    tl_far = transmission_loss(distance=1000.0, frequency=10.0)
    assert tl_far > tl_near


def test_transmission_loss_spherical_spreading():
    """At low frequency, TL should approximate 20*log10(r)."""
    # At very low frequency, absorption is negligible
    distance = 1000.0
    tl = transmission_loss(distance=distance, frequency=0.01)
    # Should be close to spherical spreading
    expected = 20 * np.log10(distance)
    assert tl == approx(expected, rel=0.1)


def test_transmission_loss_frequency_effect():
    """Higher frequency should have higher TL at same distance."""
    tl_low = transmission_loss(distance=5000.0, frequency=1.0)
    tl_high = transmission_loss(distance=5000.0, frequency=50.0)
    assert tl_high > tl_low


# =============================================================================
# Coordinate Transform Tests
# =============================================================================


def test_cart2depth_bearing_range_origin():
    """Point at origin should have zero depth, bearing, and range."""
    depth, bearing, slant_range = cart2depth_bearing_range(0.0, 0.0, 0.0)
    assert depth == 0.0
    assert slant_range == 0.0


def test_cart2depth_bearing_range_depth_sign():
    """Negative z (underwater) should give positive depth."""
    depth, bearing, slant_range = cart2depth_bearing_range(0.0, 0.0, -100.0)
    assert depth == 100.0


def test_cart2depth_bearing_range_north():
    """Point due north should have bearing = 0."""
    depth, bearing, slant_range = cart2depth_bearing_range(0.0, 100.0, 0.0)
    assert bearing == approx(0.0, abs=1e-10)


def test_cart2depth_bearing_range_east():
    """Point due east should have bearing = pi/2."""
    depth, bearing, slant_range = cart2depth_bearing_range(100.0, 0.0, 0.0)
    assert bearing == approx(np.pi / 2, abs=1e-10)


def test_cart2depth_bearing_range_south():
    """Point due south should have bearing = pi or -pi."""
    depth, bearing, slant_range = cart2depth_bearing_range(0.0, -100.0, 0.0)
    assert abs(bearing) == approx(np.pi, abs=1e-10)


def test_cart2depth_bearing_range_west():
    """Point due west should have bearing = -pi/2."""
    depth, bearing, slant_range = cart2depth_bearing_range(-100.0, 0.0, 0.0)
    assert bearing == approx(-np.pi / 2, abs=1e-10)


def test_depth_bearing_range_roundtrip():
    """cart2depth_bearing_range and inverse should be consistent."""
    x, y, z = 100.0, 200.0, -50.0
    depth, bearing, slant_range = cart2depth_bearing_range(x, y, z)
    x_back, y_back, z_back = depth_bearing_range2cart(depth, bearing, slant_range)
    assert x_back == approx(x, rel=1e-10)
    assert y_back == approx(y, rel=1e-10)
    assert z_back == approx(z, rel=1e-10)


def test_depth_bearing_range_array_input():
    """Should handle array inputs."""
    x = np.array([100.0, 200.0, 300.0])
    y = np.array([100.0, 200.0, 300.0])
    z = np.array([-50.0, -100.0, -150.0])
    depth, bearing, slant_range = cart2depth_bearing_range(x, y, z)
    assert depth.shape == (3,)
    assert bearing.shape == (3,)
    assert slant_range.shape == (3,)


def test_cart2bearing_elevation_range_horizontal():
    """Horizontal point should have zero elevation."""
    bearing, elevation, slant_range = cart2bearing_elevation_range(100.0, 100.0, 0.0)
    assert elevation == approx(0.0, abs=1e-10)


def test_cart2bearing_elevation_range_up():
    """Point directly above should have elevation = pi/2."""
    bearing, elevation, slant_range = cart2bearing_elevation_range(0.0, 0.0, 100.0)
    assert elevation == approx(np.pi / 2, abs=1e-10)


def test_cart2bearing_elevation_range_down():
    """Point directly below should have elevation = -pi/2."""
    bearing, elevation, slant_range = cart2bearing_elevation_range(0.0, 0.0, -100.0)
    assert elevation == approx(-np.pi / 2, abs=1e-10)


def test_bearing_elevation_range_roundtrip():
    """cart2bearing_elevation_range and inverse should be consistent."""
    x, y, z = 100.0, 200.0, 50.0
    bearing, elevation, slant_range = cart2bearing_elevation_range(x, y, z)
    x_back, y_back, z_back = bearing_elevation_range2cart(bearing, elevation, slant_range)
    assert x_back == approx(x, rel=1e-10)
    assert y_back == approx(y, rel=1e-10)
    assert z_back == approx(z, rel=1e-10)


@pytest.mark.parametrize(
    "x,y,z",
    [
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
        (100.0, 100.0, 100.0),
        (-50.0, 75.0, -25.0),
        (1000.0, 2000.0, -500.0),
    ],
)
def test_coordinate_roundtrip_parametrized(x, y, z):
    """Test coordinate roundtrip for various positions."""
    # Test depth-bearing-range roundtrip
    depth, bearing, slant_range = cart2depth_bearing_range(x, y, z)
    x1, y1, z1 = depth_bearing_range2cart(depth, bearing, slant_range)
    assert x1 == approx(x, rel=1e-10, abs=1e-10)
    assert y1 == approx(y, rel=1e-10, abs=1e-10)
    assert z1 == approx(z, rel=1e-10, abs=1e-10)

    # Test bearing-elevation-range roundtrip
    bearing2, elevation, slant_range2 = cart2bearing_elevation_range(x, y, z)
    x2, y2, z2 = bearing_elevation_range2cart(bearing2, elevation, slant_range2)
    assert x2 == approx(x, rel=1e-10, abs=1e-10)
    assert y2 == approx(y, rel=1e-10, abs=1e-10)
    assert z2 == approx(z, rel=1e-10, abs=1e-10)


def test_slant_range_consistency():
    """Slant range should be same from both coordinate systems."""
    x, y, z = 100.0, 200.0, -50.0
    _, _, slant_range1 = cart2depth_bearing_range(x, y, z)
    _, _, slant_range2 = cart2bearing_elevation_range(x, y, z)
    assert slant_range1 == approx(slant_range2, rel=1e-10)
    # Should equal 3D distance
    expected = np.sqrt(x**2 + y**2 + z**2)
    assert slant_range1 == approx(expected, rel=1e-10)


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_zero_temperature_sound_speed():
    """Sound speed at 0°C should still be reasonable."""
    c = sound_speed_mackenzie(temperature=0.0, salinity=35.0, depth=0.0)
    assert 1400 < c < 1500


def test_negative_temperature_sound_speed():
    """Sound speed at -2°C (sea water freezing point) should work."""
    c = sound_speed_mackenzie(temperature=-2.0, salinity=35.0, depth=0.0)
    assert 1400 < c < 1500


def test_high_depth_sound_speed():
    """Sound speed at great depth should be higher."""
    c = sound_speed_mackenzie(temperature=2.0, salinity=35.0, depth=5000.0)
    assert c > 1500  # Deep water, high pressure


def test_attenuation_very_high_frequency():
    """Attenuation at high frequency should be very high."""
    alpha = acoustic_attenuation(frequency=500.0)
    assert alpha > 50  # High attenuation at 500 kHz (~60 dB/km)


def test_transmission_loss_small_distance():
    """TL at very small distance should be small but positive."""
    tl = transmission_loss(distance=1.0, frequency=10.0)
    assert tl > 0
    assert tl < 10  # Should be reasonable for 1m


# =============================================================================
# Geodetic Integration Tests
# =============================================================================


def test_geodetic_to_ecef_at_equator_surface():
    """Test ECEF conversion at equator, prime meridian, surface."""
    lat, lon, depth = 0.0, 0.0, 0.0
    x, y, z = geodetic_to_ecef_underwater(lat, lon, depth)
    # At equator, prime meridian: x = semi-major axis, y = 0, z = 0
    assert x == approx(6378137.0, rel=1e-6)
    assert y == approx(0.0, abs=1e-6)
    assert z == approx(0.0, abs=1e-6)


def test_geodetic_to_ecef_at_pole():
    """Test ECEF conversion at north pole."""
    lat, lon, depth = np.pi / 2, 0.0, 0.0
    x, y, z = geodetic_to_ecef_underwater(lat, lon, depth)
    # At pole: x = 0, y = 0, z = semi-minor axis
    # z = N(1-e2) = a(1-e2) / sqrt(1-e2) = a*sqrt(1-e2) for pole
    assert x == approx(0.0, abs=1e-6)
    assert y == approx(0.0, abs=1e-6)
    assert z > 6356000.0  # Should be close to semi-minor axis


def test_geodetic_to_ecef_with_depth():
    """Depth should reduce ECEF distance from Earth center."""
    lat, lon = np.radians(45.0), np.radians(0.0)
    x_surface, y_surface, z_surface = geodetic_to_ecef_underwater(lat, lon, 0.0)
    x_deep, y_deep, z_deep = geodetic_to_ecef_underwater(lat, lon, 1000.0)

    r_surface = np.sqrt(x_surface**2 + y_surface**2 + z_surface**2)
    r_deep = np.sqrt(x_deep**2 + y_deep**2 + z_deep**2)

    # Underwater point should be closer to Earth center
    assert r_deep < r_surface


def test_ecef_to_geodetic_roundtrip():
    """Test ECEF to geodetic roundtrip conversion."""
    lat_orig = np.radians(51.4769)  # Greenwich
    lon_orig = np.radians(-0.0005)
    depth_orig = 100.0

    x, y, z = geodetic_to_ecef_underwater(lat_orig, lon_orig, depth_orig)
    lat, lon, depth = ecef_to_geodetic_underwater(x, y, z)

    assert lat == approx(lat_orig, rel=1e-10)
    assert lon == approx(lon_orig, rel=1e-10)
    assert depth == approx(depth_orig, rel=1e-6)


@pytest.mark.parametrize(
    "lat_deg,lon_deg,depth",
    [
        (0.0, 0.0, 0.0),
        (45.0, 90.0, 500.0),
        (-30.0, 120.0, 2000.0),
        (89.0, -45.0, 100.0),
        (-89.0, 180.0, 3000.0),
    ],
)
def test_ecef_geodetic_roundtrip_parametrized(lat_deg, lon_deg, depth):
    """Test roundtrip at various locations."""
    lat_orig = np.radians(lat_deg)
    lon_orig = np.radians(lon_deg)

    x, y, z = geodetic_to_ecef_underwater(lat_orig, lon_orig, depth)
    lat, lon, depth_back = ecef_to_geodetic_underwater(x, y, z)

    assert lat == approx(lat_orig, rel=1e-10, abs=1e-10)
    assert lon == approx(lon_orig, rel=1e-10, abs=1e-10)
    assert depth_back == approx(depth, rel=1e-6, abs=1e-6)


def test_enu_at_reference_is_zero():
    """Point at reference should have ENU = (0, 0, 0)."""
    ref_lat = np.radians(45.0)
    ref_lon = np.radians(-122.0)
    ref_depth = 0.0

    x, y, z = geodetic_to_ecef_underwater(ref_lat, ref_lon, ref_depth)
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_depth)

    assert e == approx(0.0, abs=1e-6)
    assert n == approx(0.0, abs=1e-6)
    assert u == approx(0.0, abs=1e-6)


def test_enu_north_direction():
    """Point 1km north should have north ~= 1000m."""
    ref_lat = np.radians(45.0)
    ref_lon = np.radians(-122.0)

    # Move ~1km north (approx 0.009 degrees at 45 lat)
    target_lat = ref_lat + 0.009 * np.pi / 180
    target_lon = ref_lon
    target_depth = 0.0

    x, y, z = geodetic_to_ecef_underwater(target_lat, target_lon, target_depth)
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, 0.0)

    assert abs(e) < 1.0  # Should be nearly zero east
    assert n > 900 and n < 1100  # Should be ~1000m north
    assert abs(u) < 1.0  # Should be nearly zero up


def test_enu_east_direction():
    """Point 1km east should have east ~= 1000m."""
    ref_lat = np.radians(45.0)
    ref_lon = np.radians(-122.0)

    # Move ~1km east (approx 0.013 degrees at 45 lat)
    target_lat = ref_lat
    target_lon = ref_lon + 0.013 * np.pi / 180
    target_depth = 0.0

    x, y, z = geodetic_to_ecef_underwater(target_lat, target_lon, target_depth)
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, 0.0)

    assert e > 900 and e < 1100  # Should be ~1000m east
    assert abs(n) < 1.0  # Should be nearly zero north
    assert abs(u) < 1.0  # Should be nearly zero up


def test_enu_down_direction():
    """Point 100m deeper should have up ~= -100m."""
    ref_lat = np.radians(45.0)
    ref_lon = np.radians(-122.0)
    ref_depth = 0.0

    target_lat = ref_lat
    target_lon = ref_lon
    target_depth = 100.0

    x, y, z = geodetic_to_ecef_underwater(target_lat, target_lon, target_depth)
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_depth)

    assert abs(e) < 0.1  # Should be zero east
    assert abs(n) < 0.1  # Should be zero north
    assert u == approx(-100.0, rel=1e-4)  # Should be -100m (down)


def test_enu_to_ecef_roundtrip():
    """Test ENU to ECEF roundtrip."""
    ref_lat = np.radians(34.0)
    ref_lon = np.radians(-118.0)
    ref_depth = 50.0

    east_orig, north_orig, up_orig = 500.0, -300.0, -150.0

    x, y, z = enu_to_ecef(east_orig, north_orig, up_orig, ref_lat, ref_lon, ref_depth)
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_depth)

    assert e == approx(east_orig, rel=1e-10)
    assert n == approx(north_orig, rel=1e-10)
    assert u == approx(up_orig, rel=1e-10)


def test_geodetic_depth_to_enu_basic():
    """Test geodetic to ENU conversion."""
    ref_lat = np.radians(45.0)
    ref_lon = np.radians(0.0)

    # Target at same location but 200m deep
    target_lat = ref_lat
    target_lon = ref_lon
    target_depth = 200.0

    e, n, u = geodetic_depth_to_enu(target_lat, target_lon, target_depth, ref_lat, ref_lon, 0.0)

    assert abs(e) < 0.1
    assert abs(n) < 0.1
    assert u == approx(-200.0, rel=1e-4)


def test_enu_to_geodetic_depth_roundtrip():
    """Test ENU to geodetic roundtrip."""
    ref_lat = np.radians(40.0)
    ref_lon = np.radians(-74.0)
    ref_depth = 0.0

    # Start with geodetic coordinates
    target_lat = np.radians(40.01)
    target_lon = np.radians(-73.99)
    target_depth = 300.0

    # Convert to ENU
    e, n, u = geodetic_depth_to_enu(
        target_lat, target_lon, target_depth, ref_lat, ref_lon, ref_depth
    )

    # Convert back to geodetic
    lat, lon, depth = enu_to_geodetic_depth(e, n, u, ref_lat, ref_lon, ref_depth)

    assert lat == approx(target_lat, rel=1e-10)
    assert lon == approx(target_lon, rel=1e-10)
    assert depth == approx(target_depth, rel=1e-6)


def test_haversine_distance_surface():
    """Test haversine distance at surface."""
    # Two points ~111km apart (1 degree at equator)
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 0.0, np.radians(1.0)

    dist = haversine_distance(lat1, lon1, lat2, lon2)

    # 1 degree at equator is ~111km
    assert dist == approx(111320.0, rel=0.01)


def test_haversine_distance_same_point():
    """Distance to same point should be zero."""
    lat, lon = np.radians(45.0), np.radians(-90.0)
    dist = haversine_distance(lat, lon, lat, lon)
    assert dist == approx(0.0, abs=1e-6)


def test_haversine_distance_vertical():
    """Distance for same lat/lon but different depth."""
    lat, lon = np.radians(30.0), np.radians(60.0)
    dist = haversine_distance(lat, lon, lat, lon, depth1=0.0, depth2=1000.0)
    assert dist == approx(1000.0, rel=1e-4)


def test_haversine_distance_3d():
    """Test 3D distance with both horizontal and vertical components."""
    lat1, lon1, depth1 = np.radians(45.0), np.radians(0.0), 0.0
    lat2, lon2, depth2 = np.radians(45.0), np.radians(0.01), 500.0

    dist = haversine_distance(lat1, lon1, lat2, lon2, depth1, depth2)

    # Should be greater than either horizontal or vertical distance alone
    horiz_dist = haversine_distance(lat1, lon1, lat2, lon2, 0.0, 0.0)
    assert dist > horiz_dist
    assert dist > 500.0
    assert dist < horiz_dist + 500.0  # Triangle inequality


# =============================================================================
# Temperature/Salinity Profile Tests
# =============================================================================


def test_isothermal_profile_structure():
    """Isothermal profile should have correct structure."""
    profile = create_isothermal_profile(15.0, 35.0, max_depth=500.0, num_points=50)
    assert "depth" in profile
    assert "temperature" in profile
    assert "salinity" in profile
    assert len(profile["depth"]) == 50
    assert profile["depth"][0] == 0.0
    assert profile["depth"][-1] == 500.0


def test_isothermal_profile_constant():
    """Isothermal profile should have constant T and S."""
    profile = create_isothermal_profile(18.0, 36.0)
    assert np.all(profile["temperature"] == 18.0)
    assert np.all(profile["salinity"] == 36.0)


def test_thermocline_profile_surface_temp():
    """Thermocline profile should have correct surface temperature."""
    profile = create_thermocline_profile(
        surface_temp=25.0,
        deep_temp=5.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
    )
    # Near surface should be close to surface temp
    assert profile["temperature"][0] == approx(25.0, rel=0.1)


def test_thermocline_profile_deep_temp():
    """Thermocline profile should approach deep temperature."""
    profile = create_thermocline_profile(
        surface_temp=25.0,
        deep_temp=5.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
        max_depth=1000.0,
    )
    # Deep water should be close to deep temp
    assert profile["temperature"][-1] == approx(5.0, rel=0.1)


def test_thermocline_profile_transition():
    """Temperature should decrease through thermocline."""
    profile = create_thermocline_profile(
        surface_temp=25.0,
        deep_temp=5.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
    )
    # Temperature should monotonically decrease
    for i in range(1, len(profile["temperature"])):
        assert profile["temperature"][i] <= profile["temperature"][i - 1]


def test_mixed_layer_profile_constant_surface():
    """Mixed layer should have constant temperature."""
    profile = create_mixed_layer_profile(
        mixed_layer_temp=20.0,
        mixed_layer_depth=50.0,
        thermocline_gradient=-0.1,
        deep_temp=4.0,
    )
    # Find points in mixed layer
    in_mixed = profile["depth"] <= 50.0
    mixed_temps = profile["temperature"][in_mixed]
    assert np.all(mixed_temps == 20.0)


def test_mixed_layer_profile_thermocline():
    """Temperature should decrease below mixed layer."""
    profile = create_mixed_layer_profile(
        mixed_layer_temp=20.0,
        mixed_layer_depth=50.0,
        thermocline_gradient=-0.1,
        deep_temp=4.0,
    )
    # Below mixed layer, temperature should decrease
    below_mixed = profile["depth"] > 50.0
    below_temps = profile["temperature"][below_mixed]
    assert below_temps[0] < 20.0
    assert below_temps[-1] == approx(4.0, rel=0.01)


def test_interpolate_profile_scalar():
    """Profile interpolation should work for scalar depth."""
    profile = create_isothermal_profile(15.0, 35.0, max_depth=1000.0)
    temp, sal = interpolate_profile(profile, 500.0)
    assert temp == 15.0
    assert sal == 35.0


def test_interpolate_profile_array():
    """Profile interpolation should work for array of depths."""
    profile = create_isothermal_profile(15.0, 35.0, max_depth=1000.0)
    depths = np.array([100.0, 200.0, 300.0])
    temps, sals = interpolate_profile(profile, depths)
    assert len(temps) == 3
    assert np.all(temps == 15.0)


def test_interpolate_profile_thermocline():
    """Interpolation should work in thermocline."""
    profile = create_thermocline_profile(
        surface_temp=25.0,
        deep_temp=5.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
    )
    temp, _ = interpolate_profile(profile, 100.0)
    # At thermocline center, temp should be mean
    assert temp == approx(15.0, rel=0.1)


def test_sound_speed_profile_shape():
    """Sound speed profile should have correct shape."""
    profile = create_isothermal_profile(15.0, 35.0, max_depth=500.0, num_points=50)
    svp = sound_speed_profile(profile)
    assert len(svp["depth"]) == 50
    assert len(svp["sound_speed"]) == 50


def test_sound_speed_profile_reasonable_values():
    """Sound speeds should be in reasonable range."""
    profile = create_thermocline_profile(
        surface_temp=20.0,
        deep_temp=4.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
    )
    svp = sound_speed_profile(profile)
    # All sound speeds should be between 1400 and 1600 m/s
    assert np.all(svp["sound_speed"] > 1400)
    assert np.all(svp["sound_speed"] < 1600)


def test_sound_speed_gradient_sign():
    """Gradient should be negative in thermocline (temp effect dominates)."""
    profile = create_thermocline_profile(
        surface_temp=25.0,
        deep_temp=5.0,
        thermocline_depth=100.0,
        thermocline_thickness=30.0,
    )
    gradient = compute_sound_speed_gradient(profile, 100.0)
    # In strong thermocline, temperature effect dominates -> negative gradient
    assert gradient < 0


def test_sound_speed_gradient_deep():
    """Gradient should be positive in deep water (pressure effect)."""
    profile = create_isothermal_profile(4.0, 35.0, max_depth=5000.0)
    gradient = compute_sound_speed_gradient(profile, 3000.0)
    # In isothermal water, pressure effect dominates -> positive gradient
    assert gradient > 0


def test_find_sound_channel_no_minimum():
    """Isothermal profile should not have sound channel."""
    profile = create_isothermal_profile(4.0, 35.0, max_depth=1000.0)
    axis = find_sound_channel_axis(profile)
    # No minimum in isothermal water (sound speed increases with depth)
    assert axis is None


# =============================================================================
# Ocean Current Model Tests
# =============================================================================


def test_uniform_current_constant():
    """Uniform current should be constant everywhere."""
    current = create_uniform_current(1.0, 0.5, 0.0)
    vx1, vy1, vz1 = current(0, 0, 0)
    vx2, vy2, vz2 = current(1000, 2000, -500)
    assert vx1 == vx2 == 1.0
    assert vy1 == vy2 == 0.5
    assert vz1 == vz2 == 0.0


def test_depth_varying_current_surface():
    """Depth-varying current should be strongest at surface."""
    current = create_depth_varying_current(2.0, 100.0, np.pi / 2)  # East
    vx, vy, vz = current(0, 0, 0)  # Surface
    assert vx == approx(2.0, rel=1e-6)
    assert vy == approx(0.0, abs=1e-10)


def test_depth_varying_current_decay():
    """Current should decay exponentially with depth."""
    current = create_depth_varying_current(2.0, 100.0, np.pi / 2)
    vx_surface, _, _ = current(0, 0, 0)
    vx_deep, _, _ = current(0, 0, -100.0)  # 100m depth
    # At e-folding depth, velocity should be 1/e of surface
    assert vx_deep == approx(vx_surface / np.e, rel=0.01)


def test_shear_current_surface():
    """Shear current should have maximum at surface."""
    current = create_shear_current(1.5, 0.001, 0.0)  # North
    vx, vy, vz = current(0, 0, 0)
    assert vx == approx(0.0, abs=1e-10)
    assert vy == approx(1.5, rel=1e-6)


def test_shear_current_linear_decrease():
    """Shear current should decrease linearly."""
    current = create_shear_current(1.0, 0.001, 0.0)
    _, vy1, _ = current(0, 0, 0)
    _, vy2, _ = current(0, 0, -500.0)  # 500m depth
    # Velocity should decrease by 0.001 * 500 = 0.5
    assert vy2 == approx(0.5, rel=0.01)


def test_shear_current_zero_at_max_depth():
    """Shear current should be zero at max depth."""
    current = create_shear_current(1.0, 0.001, 0.0, max_depth=500.0)
    vx, vy, vz = current(0, 0, -600.0)  # Below max depth
    assert vx == 0.0
    assert vy == 0.0
    assert vz == 0.0


def test_apply_current_to_velocity():
    """Apply current should add velocity components."""
    current = create_uniform_current(1.0, 2.0, 0.0)
    vx, vy, vz = apply_current_to_velocity(3.0, 4.0, -1.0, current, 0, 0, 0)
    assert vx == 4.0  # 3.0 + 1.0
    assert vy == 6.0  # 4.0 + 2.0
    assert vz == -1.0  # -1.0 + 0.0


def test_current_direction_north():
    """Current with direction=0 should flow north."""
    current = create_depth_varying_current(1.0, 100.0, 0.0)
    vx, vy, vz = current(0, 0, 0)
    assert vx == approx(0.0, abs=1e-10)
    assert vy == approx(1.0, rel=1e-6)


def test_current_direction_east():
    """Current with direction=pi/2 should flow east."""
    current = create_depth_varying_current(1.0, 100.0, np.pi / 2)
    vx, vy, vz = current(0, 0, 0)
    assert vx == approx(1.0, rel=1e-6)
    assert vy == approx(0.0, abs=1e-10)


# =============================================================================
# Bathymetry Grid Tests
# =============================================================================


def test_flat_bathymetry_structure():
    """Flat bathymetry should have correct structure."""
    bathy = create_flat_bathymetry(100.0, (0, 1000), (0, 1000), resolution=100.0)
    assert "x" in bathy
    assert "y" in bathy
    assert "depth" in bathy
    assert bathy["depth"].ndim == 2


def test_flat_bathymetry_constant():
    """Flat bathymetry should have constant depth."""
    bathy = create_flat_bathymetry(150.0, (0, 500), (0, 500))
    assert np.all(bathy["depth"] == 150.0)


def test_flat_bathymetry_grid_extent():
    """Flat bathymetry grid should cover specified extent."""
    bathy = create_flat_bathymetry(100.0, (0, 1000), (-500, 500), resolution=100.0)
    assert bathy["x"][0] == 0.0
    assert bathy["x"][-1] >= 1000.0
    assert bathy["y"][0] == -500.0
    assert bathy["y"][-1] >= 500.0


def test_sloped_bathymetry_shallow_region():
    """Sloped bathymetry should have shallow depth before slope."""
    bathy = create_sloped_bathymetry(
        shallow_depth=50.0,
        deep_depth=500.0,
        slope_start_x=200.0,
        slope_end_x=400.0,
        x_range=(0, 600),
        y_range=(0, 200),
    )
    # Points before slope should have shallow depth
    shallow_idx = bathy["x"] <= 200.0
    for j in range(bathy["depth"].shape[0]):
        assert np.all(bathy["depth"][j, shallow_idx] == 50.0)


def test_sloped_bathymetry_deep_region():
    """Sloped bathymetry should have deep depth after slope."""
    bathy = create_sloped_bathymetry(
        shallow_depth=50.0,
        deep_depth=500.0,
        slope_start_x=200.0,
        slope_end_x=400.0,
        x_range=(0, 600),
        y_range=(0, 200),
    )
    # Points after slope should have deep depth
    deep_idx = bathy["x"] >= 400.0
    for j in range(bathy["depth"].shape[0]):
        assert np.all(bathy["depth"][j, deep_idx] == 500.0)


def test_sloped_bathymetry_transition():
    """Sloped bathymetry should transition smoothly."""
    bathy = create_sloped_bathymetry(
        shallow_depth=50.0,
        deep_depth=500.0,
        slope_start_x=200.0,
        slope_end_x=400.0,
        x_range=(0, 600),
        y_range=(0, 200),
        resolution=50.0,
    )
    # Midpoint of slope should have average depth
    mid_x = 300.0
    mid_idx = np.argmin(np.abs(bathy["x"] - mid_x))
    expected_depth = (50.0 + 500.0) / 2
    assert bathy["depth"][0, mid_idx] == approx(expected_depth, rel=0.1)


def test_canyon_bathymetry_base_depth():
    """Canyon bathymetry should have base depth away from canyon."""
    bathy = create_canyon_bathymetry(
        base_depth=100.0,
        canyon_depth=200.0,
        canyon_center_y=500.0,
        canyon_width=100.0,
        x_range=(0, 1000),
        y_range=(0, 1000),
    )
    # Far from canyon center, depth should be near base depth
    far_y_idx = np.argmin(np.abs(bathy["y"] - 0.0))  # y=0, far from center at 500
    assert bathy["depth"][far_y_idx, 0] == approx(100.0, rel=0.01)


def test_canyon_bathymetry_canyon_center():
    """Canyon bathymetry should be deepest at canyon center."""
    bathy = create_canyon_bathymetry(
        base_depth=100.0,
        canyon_depth=200.0,
        canyon_center_y=500.0,
        canyon_width=100.0,
        x_range=(0, 1000),
        y_range=(0, 1000),
    )
    # At canyon center, depth should be base + canyon depth
    center_y_idx = np.argmin(np.abs(bathy["y"] - 500.0))
    expected_max_depth = 100.0 + 200.0
    assert bathy["depth"][center_y_idx, 0] == approx(expected_max_depth, rel=0.01)


def test_interpolate_bathymetry_exact_point():
    """Interpolation at grid point should return exact value."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500), resolution=100.0)
    depth = interpolate_bathymetry(bathy, 100.0, 100.0)
    assert depth == approx(100.0, rel=1e-6)


def test_interpolate_bathymetry_between_points():
    """Interpolation between grid points should give intermediate value."""
    bathy = create_sloped_bathymetry(
        shallow_depth=50.0,
        deep_depth=150.0,
        slope_start_x=0.0,
        slope_end_x=1000.0,
        x_range=(0, 1000),
        y_range=(0, 200),
        resolution=100.0,
    )
    # At midpoint, depth should be average
    depth = interpolate_bathymetry(bathy, 500.0, 100.0)
    assert depth == approx(100.0, rel=0.1)


def test_interpolate_bathymetry_array():
    """Interpolation should work for arrays of points."""
    bathy = create_flat_bathymetry(200.0, (0, 1000), (0, 1000))
    x = np.array([100.0, 200.0, 300.0])
    y = np.array([100.0, 200.0, 300.0])
    depths = interpolate_bathymetry(bathy, x, y)
    assert len(depths) == 3
    assert np.all(depths == approx(200.0, rel=1e-6))


def test_height_above_seafloor_at_surface():
    """Point at surface should have height = seafloor depth."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    hab = height_above_seafloor(250.0, 250.0, 0.0, bathy)
    assert hab == approx(100.0, rel=1e-6)


def test_height_above_seafloor_underwater():
    """Point underwater should have reduced height."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    hab = height_above_seafloor(250.0, 250.0, -50.0, bathy)
    assert hab == approx(50.0, rel=1e-6)


def test_height_above_seafloor_at_bottom():
    """Point at seafloor should have height = 0."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    hab = height_above_seafloor(250.0, 250.0, -100.0, bathy)
    assert hab == approx(0.0, abs=1e-6)


def test_height_above_seafloor_below_bottom():
    """Point below seafloor should have negative height."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    hab = height_above_seafloor(250.0, 250.0, -150.0, bathy)
    assert hab == approx(-50.0, rel=1e-6)


def test_is_in_water_surface():
    """Point at surface should be in water."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    assert is_in_water(250.0, 250.0, 0.0, bathy) is True


def test_is_in_water_underwater():
    """Point underwater should be in water."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    assert is_in_water(250.0, 250.0, -50.0, bathy) is True


def test_is_in_water_above_surface():
    """Point above surface should not be in water."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    assert is_in_water(250.0, 250.0, 10.0, bathy) is False


def test_is_in_water_at_bottom():
    """Point at seafloor should be in water."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    assert is_in_water(250.0, 250.0, -100.0, bathy) is True


def test_is_in_water_below_bottom():
    """Point below seafloor should not be in water."""
    bathy = create_flat_bathymetry(100.0, (0, 500), (0, 500))
    assert is_in_water(250.0, 250.0, -150.0, bathy) is False


# =============================================================================
# Tidal Effect Tests
# =============================================================================


def test_simple_tidal_offset_high_tide():
    """At t=0, tide should be at maximum (high tide)."""
    offset = simple_tidal_offset(0.0, amplitude=1.0)
    assert offset == approx(1.0, rel=1e-6)


def test_simple_tidal_offset_low_tide():
    """At t=period/2, tide should be at minimum (low tide)."""
    period = 12.42
    offset = simple_tidal_offset(period / 2, amplitude=1.0, period_hours=period)
    assert offset == approx(-1.0, rel=1e-6)


def test_simple_tidal_offset_mid_tide():
    """At t=period/4, tide should be at zero."""
    period = 12.42
    offset = simple_tidal_offset(period / 4, amplitude=1.0, period_hours=period)
    assert offset == approx(0.0, abs=1e-6)


def test_simple_tidal_offset_periodicity():
    """Tide should repeat after one period."""
    period = 12.42
    offset1 = simple_tidal_offset(0.0, amplitude=1.5, period_hours=period)
    offset2 = simple_tidal_offset(period, amplitude=1.5, period_hours=period)
    assert offset1 == approx(offset2, rel=1e-6)


def test_simple_tidal_offset_amplitude():
    """Amplitude should scale the offset."""
    offset1 = simple_tidal_offset(0.0, amplitude=1.0)
    offset2 = simple_tidal_offset(0.0, amplitude=2.0)
    assert offset2 == approx(2 * offset1, rel=1e-6)


def test_dual_constituent_tide_at_zero():
    """At t=0 with zero phases, both constituents should be at max."""
    offset = dual_constituent_tide(0.0, amp_m2=1.0, amp_s2=0.5)
    assert offset == approx(1.5, rel=1e-6)  # 1.0 + 0.5


def test_dual_constituent_tide_spring_tide():
    """Spring tide occurs when M2 and S2 are in phase."""
    # Both at high, zero phase offset
    offset = dual_constituent_tide(0.0, amp_m2=1.0, amp_s2=0.5, phase_m2=0.0, phase_s2=0.0)
    assert offset == approx(1.5, rel=1e-6)


def test_dual_constituent_tide_neap_tide():
    """Neap tide occurs when M2 and S2 are out of phase."""
    # M2 at high, S2 at low (phase = pi)
    offset = dual_constituent_tide(0.0, amp_m2=1.0, amp_s2=0.5, phase_m2=0.0, phase_s2=np.pi)
    assert offset == approx(0.5, rel=1e-6)  # 1.0 - 0.5


def test_dual_constituent_tide_m2_only():
    """With amp_s2=0, should behave like simple M2 tide."""
    offset1 = dual_constituent_tide(6.21, amp_m2=1.0, amp_s2=0.0)
    offset2 = simple_tidal_offset(6.21, amplitude=1.0, period_hours=12.42)
    assert offset1 == approx(offset2, rel=1e-6)


def test_tidal_current_max_flood():
    """At phase=0, current should be at maximum flood."""
    vx, vy = tidal_current(0.0, max_speed=1.0, direction_rad=np.pi / 2, phase=0.0)
    # sin(0) = 0, so at t=0 current is 0
    assert vx == approx(0.0, abs=1e-10)
    assert vy == approx(0.0, abs=1e-10)


def test_tidal_current_quarter_period():
    """At t=period/4, current should be at maximum."""
    period = 12.42
    vx, vy = tidal_current(period / 4, max_speed=1.0, direction_rad=np.pi / 2, period_hours=period)
    # sin(pi/2) = 1, flowing east
    assert vx == approx(1.0, rel=1e-6)
    assert vy == approx(0.0, abs=1e-10)


def test_tidal_current_half_period():
    """At t=period/2, current should reverse to zero crossing."""
    period = 12.42
    vx, vy = tidal_current(period / 2, max_speed=1.0, direction_rad=np.pi / 2, period_hours=period)
    # sin(pi) = 0
    assert vx == approx(0.0, abs=1e-10)
    assert vy == approx(0.0, abs=1e-10)


def test_tidal_current_direction_north():
    """Current with direction=0 should flow north."""
    period = 12.42
    vx, vy = tidal_current(period / 4, max_speed=1.0, direction_rad=0.0, period_hours=period)
    assert vx == approx(0.0, abs=1e-10)
    assert vy == approx(1.0, rel=1e-6)


def test_tidal_current_periodicity():
    """Tidal current should repeat after one period."""
    period = 12.42
    vx1, vy1 = tidal_current(1.0, max_speed=0.5, direction_rad=np.pi / 4, period_hours=period)
    vx2, vy2 = tidal_current(
        1.0 + period, max_speed=0.5, direction_rad=np.pi / 4, period_hours=period
    )
    assert vx1 == approx(vx2, rel=1e-6)
    assert vy1 == approx(vy2, rel=1e-6)


def test_apply_tide_to_depth_high_tide():
    """At high tide, water depth should increase."""
    depth = apply_tide_to_depth(100.0, 0.0, tidal_amplitude=2.0)
    assert depth == approx(102.0, rel=1e-6)


def test_apply_tide_to_depth_low_tide():
    """At low tide, water depth should decrease."""
    period = 12.42
    depth = apply_tide_to_depth(100.0, period / 2, tidal_amplitude=2.0, period_hours=period)
    assert depth == approx(98.0, rel=1e-6)


def test_apply_tide_to_depth_mean_tide():
    """At mean tide, depth should equal mean depth."""
    period = 12.42
    depth = apply_tide_to_depth(100.0, period / 4, tidal_amplitude=2.0, period_hours=period)
    assert depth == approx(100.0, abs=1e-6)


def test_apply_tide_to_depth_amplitude():
    """Larger amplitude should cause larger depth variation."""
    depth1 = apply_tide_to_depth(100.0, 0.0, tidal_amplitude=1.0)
    depth2 = apply_tide_to_depth(100.0, 0.0, tidal_amplitude=3.0)
    assert depth1 == approx(101.0, rel=1e-6)
    assert depth2 == approx(103.0, rel=1e-6)
