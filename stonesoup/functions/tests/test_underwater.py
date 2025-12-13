"""Tests for underwater/undersea tracking utility functions."""

import numpy as np
import pytest
from pytest import approx

from ..underwater import (
    acoustic_attenuation,
    bearing_elevation_range2cart,
    cart2bearing_elevation_range,
    cart2depth_bearing_range,
    depth_bearing_range2cart,
    depth_to_pressure,
    pressure_to_depth,
    sound_speed_mackenzie,
    sound_speed_unesco,
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
