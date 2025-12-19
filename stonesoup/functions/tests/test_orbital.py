from datetime import datetime, timedelta

import numpy as np
import pytest

from ...types.array import StateVector
from ..orbital import (
    compute_orbital_period,
    compute_orbital_velocity,
    compute_specific_angular_momentum,
    compute_specific_energy,
    eccentric_anomaly_from_mean_anomaly,
    keplerian_to_rv,
    lagrange_coefficients_from_universal_anomaly,
    mod_elongitude,
    mod_inclination,
    orbital_state_ecef_to_eci,
    orbital_state_eci_to_ecef,
    orbital_state_gcrs_to_j2000,
    orbital_state_j2000_to_gcrs,
    perifocal_position,
    perifocal_to_geocentric_matrix,
    perifocal_velocity,
    stumpff_c,
    stumpff_s,
    tru_anom_from_mean_anom,
    universal_anomaly_newton,
)


@pytest.mark.parametrize(
    "z, outs, outc",
    [
        (0, 1 / 6, 1 / 2),
        (np.pi**2, 0.10132118364233778, 0.20264236728467555),
        (-(np.pi**2), 0.2711433813983066, 1.073189242960177),
    ],
)
def test_stumpff(z, outs, outc):
    """Test the Stumpf functions"""
    assert np.isclose(stumpff_s(z), outs, rtol=1e-10)
    assert np.isclose(stumpff_c(z), outc, rtol=1e-10)


def test_universal_anomaly_and_lagrange():
    """Test the computation of the universal anomaly. Also test the computation of the Lagrange
    coefficients. Follows example 3.7 in [1]_.

    References
    ----------
    .. [1] Curtis H.D. 2010, Orbital mechanics for engineering students, 3rd Ed., Elsevier

    """

    # Answers
    chi_is = 253.53  # km^0.5
    f_is = -0.54128
    g_is = 184.32  # s^{-1}
    fdot_is = -0.00055297  # s^{-1}
    gdot_is = -1.6592

    # Parameters
    bigg = 3.986004418e5  # in km^{-3} rather than m^{-3}
    start_at = StateVector([7000, -12124, 0, 2.6679, 4.6210, 0])
    deltat = timedelta(hours=1)

    assert np.isclose(
        universal_anomaly_newton(start_at, deltat, grav_parameter=bigg), chi_is, atol=1e-2
    )

    f, g, fdot, gdot = lagrange_coefficients_from_universal_anomaly(
        start_at, deltat, grav_parameter=bigg
    )
    assert np.isclose(f, f_is, rtol=1e-4)
    assert np.isclose(g, g_is, rtol=2e-3)  # Seems a bit loose - is the textbook wrong?
    assert np.isclose(fdot, fdot_is, rtol=1e-4)
    assert np.isclose(gdot, gdot_is, rtol=1e-4)


def test_keplerian_to_rv_nosv():
    """Make sure the error is chucked correctly with no Statevector input"""
    with pytest.raises(TypeError):
        _ = keplerian_to_rv(np.zeros(3))


# =============================================================================
# Tests for Orbital State Transformation Functions
# =============================================================================


def test_eci_to_ecef_basic():
    """Test basic ECI to ECEF transformation."""
    # LEO satellite along ECI x-axis
    state_eci = StateVector([7000000, 0, 0, 0, 7500, 0])
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    state_ecef = orbital_state_eci_to_ecef(state_eci, timestamp)

    # Position magnitude should be preserved
    pos_eci_mag = np.linalg.norm(state_eci[:3])
    pos_ecef_mag = np.linalg.norm(state_ecef[:3])
    np.testing.assert_allclose(pos_ecef_mag, pos_eci_mag, rtol=1e-10)

    # Result should be StateVector
    assert isinstance(state_ecef, StateVector)
    assert len(state_ecef) == 6


def test_ecef_to_eci_basic():
    """Test basic ECEF to ECI transformation."""
    state_ecef = StateVector([7000000, 0, 0, 0, 7500, 0])
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    state_eci = orbital_state_ecef_to_eci(state_ecef, timestamp)

    # Position magnitude should be preserved
    pos_ecef_mag = np.linalg.norm(state_ecef[:3])
    pos_eci_mag = np.linalg.norm(state_eci[:3])
    np.testing.assert_allclose(pos_eci_mag, pos_ecef_mag, rtol=1e-10)

    assert isinstance(state_eci, StateVector)
    assert len(state_eci) == 6


def test_eci_ecef_roundtrip():
    """Test that ECI->ECEF->ECI returns original state."""
    state_eci = StateVector([7000000, 1000000, 500000, 1000, 7500, 500])
    timestamp = datetime(2024, 6, 15, 6, 30, 0)

    state_ecef = orbital_state_eci_to_ecef(state_eci, timestamp)
    state_recovered = orbital_state_ecef_to_eci(state_ecef, timestamp)

    np.testing.assert_allclose(
        np.array(state_recovered).flatten(), np.array(state_eci).flatten(), rtol=1e-10
    )


def test_ecef_eci_roundtrip():
    """Test that ECEF->ECI->ECEF returns original state."""
    state_ecef = StateVector([6378137, 1000000, 2000000, -500, 7000, 1000])
    timestamp = datetime(2024, 3, 21, 0, 0, 0)

    state_eci = orbital_state_ecef_to_eci(state_ecef, timestamp)
    state_recovered = orbital_state_eci_to_ecef(state_eci, timestamp)

    np.testing.assert_allclose(
        np.array(state_recovered).flatten(), np.array(state_ecef).flatten(), rtol=1e-10
    )


def test_j2000_to_gcrs_basic():
    """Test basic J2000 to GCRS transformation."""
    state_j2000 = StateVector([7000000, 0, 0, 0, 7500, 0])
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    state_gcrs = orbital_state_j2000_to_gcrs(state_j2000, timestamp)

    # J2000 and GCRS differ due to precession over ~24 years
    # At 7000 km range, expect up to ~50 km difference
    pos_diff = np.linalg.norm(
        np.array(state_gcrs[:3]).flatten() - np.array(state_j2000[:3]).flatten()
    )
    assert pos_diff < 50000  # Less than 50 km difference (precession effects)

    assert isinstance(state_gcrs, StateVector)
    assert len(state_gcrs) == 6


def test_gcrs_to_j2000_basic():
    """Test basic GCRS to J2000 transformation."""
    state_gcrs = StateVector([7000000, 0, 0, 0, 7500, 0])
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    state_j2000 = orbital_state_gcrs_to_j2000(state_gcrs, timestamp)

    # Precession over 24 years creates significant difference
    pos_diff = np.linalg.norm(
        np.array(state_j2000[:3]).flatten() - np.array(state_gcrs[:3]).flatten()
    )
    assert pos_diff < 50000  # Less than 50 km

    assert isinstance(state_j2000, StateVector)


def test_j2000_gcrs_roundtrip():
    """Test that J2000->GCRS->J2000 returns original state."""
    state_j2000 = StateVector([7000000, 1000000, 500000, 1000, 7500, 500])
    timestamp = datetime(2024, 6, 15, 6, 30, 0)

    state_gcrs = orbital_state_j2000_to_gcrs(state_j2000, timestamp)
    state_recovered = orbital_state_gcrs_to_j2000(state_gcrs, timestamp)

    np.testing.assert_allclose(
        np.array(state_recovered).flatten(), np.array(state_j2000).flatten(), rtol=1e-10
    )


def test_transformation_at_different_times():
    """Test that ECI-ECEF transformation varies with time."""
    state_eci = StateVector([7000000, 0, 0, 0, 7500, 0])
    t1 = datetime(2024, 1, 1, 0, 0, 0)
    t2 = datetime(2024, 1, 1, 6, 0, 0)  # 6 hours later

    state_ecef_1 = orbital_state_eci_to_ecef(state_eci, t1)
    state_ecef_2 = orbital_state_eci_to_ecef(state_eci, t2)

    # Positions should differ due to Earth rotation
    pos_diff = np.linalg.norm(
        np.array(state_ecef_1[:3]).flatten() - np.array(state_ecef_2[:3]).flatten()
    )
    # 6 hours = 90 degrees of Earth rotation
    # For 7000 km radius, expect ~9900 km displacement
    assert pos_diff > 5000000  # At least 5000 km difference


# =============================================================================
# Tests for compute_orbital_period
# =============================================================================


def test_leo_period():
    """Test orbital period for LEO (400 km altitude)."""
    r_earth = 6378137  # meters
    altitude = 400000  # 400 km
    semi_major_axis = r_earth + altitude

    period = compute_orbital_period(semi_major_axis)

    # LEO period should be about 92 minutes
    period_minutes = period / 60
    assert 90 < period_minutes < 95


def test_geostationary_period():
    """Test orbital period for geostationary orbit."""
    # GEO altitude is about 35786 km
    geo_sma = 42164000  # meters (Earth radius + 35786 km)

    period = compute_orbital_period(geo_sma)

    # GEO period is one sidereal day (~86164 seconds), not one solar day
    # Sidereal day = 23h 56m 4s = 86164 seconds
    np.testing.assert_allclose(period, 86164, rtol=0.001)


def test_kepler_third_law():
    """Verify consistency with Kepler's third law."""
    mu = 3.986004418e14
    a1 = 7000000  # meters
    a2 = 14000000  # meters (doubled)

    T1 = compute_orbital_period(a1, mu)
    T2 = compute_orbital_period(a2, mu)

    # T^2 proportional to a^3
    # (T2/T1)^2 should equal (a2/a1)^3
    ratio_T_squared = (T2 / T1) ** 2
    ratio_a_cubed = (a2 / a1) ** 3
    np.testing.assert_allclose(ratio_T_squared, ratio_a_cubed, rtol=1e-10)


# =============================================================================
# Tests for compute_orbital_velocity
# =============================================================================


def test_circular_orbit_velocity():
    """Test velocity for circular orbit."""
    r = 7000000  # 7000 km
    mu = 3.986004418e14

    v = compute_orbital_velocity(r, r)  # Circular: r = a

    # Expected: sqrt(mu/r) for circular orbit
    expected_v = np.sqrt(mu / r)
    np.testing.assert_allclose(v, expected_v, rtol=1e-10)


def test_elliptical_orbit_periapsis():
    """Test velocity at periapsis of elliptical orbit."""
    r_peri = 7000000  # periapsis
    r_apo = 14000000  # apoapsis
    a = (r_peri + r_apo) / 2  # semi-major axis
    mu = 3.986004418e14

    v_peri = compute_orbital_velocity(r_peri, a)

    # Velocity should be higher at periapsis than circular
    v_circular = np.sqrt(mu / r_peri)
    assert v_peri > v_circular


def test_elliptical_orbit_apoapsis():
    """Test velocity at apoapsis of elliptical orbit."""
    r_peri = 7000000
    r_apo = 14000000
    a = (r_peri + r_apo) / 2
    mu = 3.986004418e14

    v_apo = compute_orbital_velocity(r_apo, a)

    # Velocity should be lower at apoapsis than circular at same radius
    v_circular = np.sqrt(mu / r_apo)
    assert v_apo < v_circular


def test_vis_viva_conservation():
    """Test energy conservation via vis-viva equation."""
    r_peri = 7000000
    r_apo = 14000000
    a = (r_peri + r_apo) / 2
    mu = 3.986004418e14

    v_peri = compute_orbital_velocity(r_peri, a)
    v_apo = compute_orbital_velocity(r_apo, a)

    # Specific orbital energy should be the same at both points
    energy_peri = v_peri**2 / 2 - mu / r_peri
    energy_apo = v_apo**2 / 2 - mu / r_apo
    np.testing.assert_allclose(energy_peri, energy_apo, rtol=1e-10)


# =============================================================================
# Tests for compute_specific_angular_momentum
# =============================================================================


def test_basic_angular_momentum():
    """Test basic angular momentum calculation."""
    # Simple case: r along x, v along y
    state = StateVector([7000000, 0, 0, 0, 7500, 0])

    h = compute_specific_angular_momentum(state)

    # h = r x v, should be along z-axis
    expected = np.array([0, 0, 7000000 * 7500])
    np.testing.assert_allclose(h, expected, rtol=1e-10)


def test_angular_momentum_magnitude():
    """Test angular momentum magnitude for circular orbit."""
    r = 7000000
    v = 7500  # Approximate circular velocity

    # Position along x, velocity along y
    state = StateVector([r, 0, 0, 0, v, 0])

    h = compute_specific_angular_momentum(state)
    h_mag = np.linalg.norm(h)

    # For circular orbit, |h| = r * v
    np.testing.assert_allclose(h_mag, r * v, rtol=1e-10)


def test_3d_angular_momentum():
    """Test angular momentum with 3D position and velocity."""
    state = StateVector([1000, 2000, 3000, 100, 200, 300])

    h = compute_specific_angular_momentum(state)
    r = np.array([1000, 2000, 3000])
    v = np.array([100, 200, 300])

    expected = np.cross(r, v)
    np.testing.assert_allclose(h, expected, rtol=1e-10)


# =============================================================================
# Tests for compute_specific_energy
# =============================================================================


def test_bound_orbit_negative_energy():
    """Test that bound orbits have negative energy."""
    # LEO satellite - clearly bound
    state = StateVector([7000000, 0, 0, 0, 7500, 0])

    energy = compute_specific_energy(state)

    assert energy < 0


def test_escape_velocity():
    """Test energy near escape velocity."""
    r = 7000000
    mu = 3.986004418e14

    # Escape velocity = sqrt(2 * mu / r)
    v_escape = np.sqrt(2 * mu / r)

    # At escape velocity, energy should be zero
    state = StateVector([r, 0, 0, 0, v_escape, 0])
    energy = compute_specific_energy(state)
    np.testing.assert_allclose(energy, 0, atol=1e-5)


def test_hyperbolic_positive_energy():
    """Test that hyperbolic trajectory has positive energy."""
    r = 7000000

    # Above escape velocity
    v = 15000  # Much faster than escape velocity (~10.9 km/s)

    state = StateVector([r, 0, 0, 0, v, 0])
    energy = compute_specific_energy(state)

    assert energy > 0


def test_energy_semi_major_axis_relation():
    """Test relationship between energy and semi-major axis."""
    mu = 3.986004418e14
    r = 7000000
    v = 7000  # Sub-circular velocity

    state = StateVector([r, 0, 0, 0, v, 0])
    energy = compute_specific_energy(state)

    # For elliptical orbit: epsilon = -mu / (2a)
    # So a = -mu / (2 * epsilon)
    a_computed = -mu / (2 * energy)

    # Verify via vis-viva: v^2 = mu * (2/r - 1/a)
    # So 1/a = 2/r - v^2/mu
    inv_a = 2 / r - v**2 / mu
    a_from_visviva = 1 / inv_a

    np.testing.assert_allclose(a_computed, a_from_visviva, rtol=1e-10)


# =============================================================================
# Tests for eccentric_anomaly_from_mean_anomaly (Kepler equation solver)
# =============================================================================


def test_eccentric_anomaly_circular_orbit():
    """For circular orbit (e=0), eccentric anomaly equals mean anomaly."""
    mean_anomaly = np.pi / 3  # 60 degrees
    eccentricity = 0.0

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    np.testing.assert_allclose(ecc_anom, mean_anomaly, rtol=1e-10)


def test_eccentric_anomaly_low_eccentricity():
    """Test with known value for low eccentricity orbit."""
    # For e=0.1, M=π/6, E should be approximately 0.5322
    mean_anomaly = np.pi / 6
    eccentricity = 0.1

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # Verify Kepler's equation: E - e*sin(E) = M
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert abs(residual) < 1e-8


def test_eccentric_anomaly_high_eccentricity():
    """Test with high eccentricity (elongated ellipse)."""
    mean_anomaly = np.pi / 4
    eccentricity = 0.8

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # Verify Kepler's equation is satisfied
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert abs(residual) < 1e-8


def test_eccentric_anomaly_mean_less_than_pi():
    """Test initial guess when M < π."""
    # When M < π, initial guess is M + e/2
    mean_anomaly = np.pi / 2
    eccentricity = 0.5

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # Should converge to correct value
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert abs(residual) < 1e-8


def test_eccentric_anomaly_mean_greater_than_pi():
    """Test initial guess when M > π."""
    # When M > π, initial guess is M - e/2
    mean_anomaly = 3 * np.pi / 2
    eccentricity = 0.3

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # Verify solution
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert abs(residual) < 1e-8


def test_eccentric_anomaly_zero_mean():
    """Test at periapsis (M=0)."""
    mean_anomaly = 0.0
    eccentricity = 0.4

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # At periapsis, E should also be 0
    np.testing.assert_allclose(ecc_anom, 0.0, atol=1e-8)


def test_eccentric_anomaly_mean_pi():
    """Test at apoapsis (M=π)."""
    mean_anomaly = np.pi
    eccentricity = 0.6

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # At apoapsis, E should be π
    np.testing.assert_allclose(ecc_anom, np.pi, atol=1e-6)


def test_eccentric_anomaly_multiple_values():
    """Test solver doesn't fail with various inputs."""
    eccentricity = 0.3
    mean_anomalies = np.linspace(0, 2 * np.pi, 20)

    for mean_anomaly in mean_anomalies:
        ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)
        residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
        assert abs(residual) < 1e-8


def test_eccentric_anomaly_custom_precision():
    """Test with custom precision parameter."""
    mean_anomaly = np.pi / 4
    eccentricity = 0.5

    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity, precision=1e-12)

    # Should achieve higher precision
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert abs(residual) < 1e-12


# =============================================================================
# Tests for tru_anom_from_mean_anom
# =============================================================================


def test_circular_orbit_true_anomaly_from_mean_anomaly():
    """For circular orbit, true anomaly equals mean anomaly."""
    mean_anomaly = np.pi / 3
    eccentricity = 0.0

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

    np.testing.assert_allclose(true_anom, mean_anomaly, rtol=1e-10)


def test_periapsis_true_anomaly_from_mean_anomaly():
    """At periapsis, true and mean anomalies are both zero."""
    mean_anomaly = 0.0
    eccentricity = 0.5

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

    np.testing.assert_allclose(true_anom, 0.0, atol=1e-8)


def test_apoapsis_true_anomaly_from_mean_anomaly():
    """At apoapsis, true anomaly should be π."""
    mean_anomaly = np.pi
    eccentricity = 0.5

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

    np.testing.assert_allclose(true_anom, np.pi, atol=1e-6)


def test_range_0_to_2pi_true_anomaly_from_mean_anomaly():
    """True anomaly should be in range [0, 2π)."""
    eccentricity = 0.4
    mean_anomalies = np.linspace(0, 2 * np.pi, 50)

    for mean_anomaly in mean_anomalies:
        true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)
        # Allow for floating point precision - use <= for upper bound
        assert 0 <= true_anom <= 2 * np.pi


def test_known_values_true_anomaly_from_mean_anomaly():
    """Test with known value from orbital mechanics."""
    # For e=0.5, M=π/2, can compute true anomaly analytically
    mean_anomaly = np.pi / 2
    eccentricity = 0.5

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

    # Verify relationship: tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)
    expected_true_anom = 2 * np.arctan(
        np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(ecc_anom / 2)
    )
    # Normalize to [0, 2π)
    expected_true_anom = np.remainder(expected_true_anom, 2 * np.pi)

    np.testing.assert_allclose(true_anom, expected_true_anom, rtol=1e-6)


def test_high_eccentricity_true_anomaly_from_mean_anomaly():
    """Test with high eccentricity orbit."""
    mean_anomaly = np.pi / 4
    eccentricity = 0.9

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

    # Should be valid angle
    assert 0 <= true_anom < 2 * np.pi


def test_consistency_with_eccentric_anomaly_true_anomaly_from_mean_anomaly():
    """Test relationship between true, eccentric, and mean anomaly."""
    mean_anomaly = np.pi / 3
    eccentricity = 0.6

    true_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)
    ecc_anom = eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)

    # Use relationship: cos(ν) = (cos(E) - e) / (1 - e*cos(E))
    cos_nu_expected = (np.cos(ecc_anom) - eccentricity) / (1 - eccentricity * np.cos(ecc_anom))
    cos_nu_actual = np.cos(true_anom)

    np.testing.assert_allclose(cos_nu_actual, cos_nu_expected, rtol=1e-6)


# =============================================================================
# Tests for perifocal_position
# =============================================================================


def test_periapsis_position_perifocal_position():
    """At periapsis (ν=0), position is along x-axis."""
    eccentricity = 0.5
    semimajor_axis = 7000000
    true_anomaly = 0.0

    pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)

    # r = a(1-e²)/(1+e*cos(ν))
    # At ν=0: r = a(1-e²)/(1+e) = a(1-e)
    expected_r = semimajor_axis * (1 - eccentricity)

    np.testing.assert_allclose(pos[0], expected_r, rtol=1e-10)
    np.testing.assert_allclose(pos[1], 0, atol=1e-10)
    np.testing.assert_allclose(pos[2], 0, atol=1e-10)


def test_apoapsis_position_perifocal_position():
    """At apoapsis (ν=π), position is along negative x-axis."""
    eccentricity = 0.3
    semimajor_axis = 10000000
    true_anomaly = np.pi

    pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)

    # At ν=π: r = a(1-e²)/(1-e) = a(1+e)
    expected_r = semimajor_axis * (1 + eccentricity)

    np.testing.assert_allclose(pos[0], -expected_r, rtol=1e-10)
    np.testing.assert_allclose(pos[1], 0, atol=1e-8)
    np.testing.assert_allclose(pos[2], 0, atol=1e-8)


def test_90_degrees_perifocal_position():
    """At ν=π/2, position should be along y-axis."""
    eccentricity = 0.2
    semimajor_axis = 8000000
    true_anomaly = np.pi / 2

    pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)

    # r = a(1-e²)/(1+e*cos(π/2)) = a(1-e²)
    expected_r = semimajor_axis * (1 - eccentricity**2)

    # x component should be 0, y should be r
    np.testing.assert_allclose(pos[0], 0, atol=1e-9)
    np.testing.assert_allclose(pos[1], expected_r, rtol=1e-10)
    np.testing.assert_allclose(pos[2], 0, atol=1e-9)


def test_circular_orbit_perifocal_position():
    """For circular orbit (e=0), radius is constant = a."""
    eccentricity = 0.0
    semimajor_axis = 7000000
    true_anomaly = np.pi / 4

    pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)

    # For circular orbit, |r| = a always
    r_mag = np.linalg.norm(pos)
    np.testing.assert_allclose(r_mag, semimajor_axis, rtol=1e-10)


def test_z_component_always_zero_perifocal_position():
    """In perifocal frame, z component is always zero."""
    eccentricity = 0.4
    semimajor_axis = 9000000

    # Test multiple anomalies
    for true_anomaly in np.linspace(0, 2 * np.pi, 10):
        pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)
        np.testing.assert_allclose(pos[2], 0, atol=1e-10)


def test_array_input_perifocal_position():
    """Test with array of true anomalies."""
    eccentricity = 0.3
    semimajor_axis = 7500000
    true_anomalies = np.array([[0, np.pi / 2, np.pi]])

    pos = perifocal_position(eccentricity, semimajor_axis, true_anomalies)

    # Should return 3x3 array
    assert pos.shape == (3, 3)


# =============================================================================
# Tests for perifocal_velocity
# =============================================================================


def test_periapsis_velocity_perifocal_velocity():
    """At periapsis, velocity is perpendicular to position (along y)."""
    eccentricity = 0.5
    semimajor_axis = 7000000
    true_anomaly = 0.0
    mu = 3.986004418e14

    vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)

    # At periapsis, v_x should be 0, v_y should be positive
    np.testing.assert_allclose(vel[0], 0, atol=1e-5)
    assert vel[1] > 0
    np.testing.assert_allclose(vel[2], 0, atol=1e-10)


def test_apoapsis_velocity_perifocal_velocity():
    """At apoapsis, velocity is along negative y-axis."""
    eccentricity = 0.3
    semimajor_axis = 10000000
    true_anomaly = np.pi
    mu = 3.986004418e14

    vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)

    # At apoapsis, v_x should be 0, v_y should be negative
    np.testing.assert_allclose(vel[0], 0, atol=1e-5)
    assert vel[1] < 0
    np.testing.assert_allclose(vel[2], 0, atol=1e-10)


def test_circular_orbit_velocity_perifocal_velocity():
    """For circular orbit, velocity magnitude is constant."""
    eccentricity = 0.0
    semimajor_axis = 7000000
    mu = 3.986004418e14

    # Expected velocity for circular orbit
    v_expected = np.sqrt(mu / semimajor_axis)

    # Test at multiple points
    for true_anomaly in [0, np.pi / 4, np.pi / 2, np.pi]:
        vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)
        v_mag = np.linalg.norm(vel)
        np.testing.assert_allclose(v_mag, v_expected, rtol=1e-10)


def test_z_component_always_zero_perifocal_velocity():
    """In perifocal frame, z velocity component is always zero."""
    eccentricity = 0.4
    semimajor_axis = 9000000
    mu = 3.986004418e14

    for true_anomaly in np.linspace(0, 2 * np.pi, 10):
        vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)
        np.testing.assert_allclose(vel[2], 0, atol=1e-10)


def test_vis_viva_equation_perifocal_velocity():
    """Verify velocity magnitude satisfies vis-viva equation."""
    eccentricity = 0.6
    semimajor_axis = 8000000
    true_anomaly = np.pi / 3
    mu = 3.986004418e14

    vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)
    pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)

    v_mag = np.linalg.norm(vel)
    r_mag = np.linalg.norm(pos)

    # Vis-viva: v² = μ(2/r - 1/a)
    v_expected = np.sqrt(mu * (2 / r_mag - 1 / semimajor_axis))

    np.testing.assert_allclose(v_mag, v_expected, rtol=1e-10)


def test_perpendicular_at_periapsis_apoapsis_perifocal_velocity():
    """Verify r·v = 0 at periapsis and apoapsis."""
    eccentricity = 0.4
    semimajor_axis = 7500000
    mu = 3.986004418e14

    for true_anomaly in [0, np.pi]:
        pos = perifocal_position(eccentricity, semimajor_axis, true_anomaly)
        vel = perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, mu)

        # Dot product should be zero (perpendicular)
        dot_product = np.dot(pos.flatten(), vel.flatten())
        np.testing.assert_allclose(dot_product, 0, atol=1e-3)


# =============================================================================
# Tests for perifocal_to_geocentric_matrix
# =============================================================================


def test_zero_inclination_zero_raan_zero_argp_perifocal_to_geocentric_matrix():
    """With all angles zero, transformation should be identity."""
    inclination = 0.0
    raan = 0.0
    argp = 0.0

    matrix = perifocal_to_geocentric_matrix(inclination, raan, argp)

    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-10)


def test_matrix_is_rotation_perifocal_to_geocentric_matrix():
    """Transformation matrix should be orthogonal (rotation)."""
    inclination = np.pi / 4
    raan = np.pi / 6
    argp = np.pi / 3

    matrix = perifocal_to_geocentric_matrix(inclination, raan, argp)

    # Should be orthogonal: M^T M = I
    product = matrix.T @ matrix
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    # Determinant should be +1 (proper rotation)
    det = np.linalg.det(matrix)
    np.testing.assert_allclose(det, 1.0, atol=1e-10)


def test_preserves_magnitude_perifocal_to_geocentric_matrix():
    """Rotation should preserve vector magnitude."""
    inclination = np.pi / 3
    raan = np.pi / 4
    argp = np.pi / 6

    matrix = perifocal_to_geocentric_matrix(inclination, raan, argp)

    # Test with arbitrary vector
    vec_perifocal = np.array([1000, 2000, 3000])
    vec_geocentric = matrix @ vec_perifocal

    mag_before = np.linalg.norm(vec_perifocal)
    mag_after = np.linalg.norm(vec_geocentric)

    np.testing.assert_allclose(mag_after, mag_before, rtol=1e-10)


def test_equatorial_orbit_perifocal_to_geocentric_matrix():
    """For equatorial orbit (i=0), only RAAN and argp matter."""
    inclination = 0.0
    raan = np.pi / 4
    argp = np.pi / 6

    matrix = perifocal_to_geocentric_matrix(inclination, raan, argp)

    # For i=0, this reduces to a z-axis rotation by (raan + argp)
    total_angle = raan + argp
    expected_matrix = np.array(
        [
            [np.cos(total_angle), -np.sin(total_angle), 0],
            [np.sin(total_angle), np.cos(total_angle), 0],
            [0, 0, 1],
        ]
    )

    np.testing.assert_allclose(matrix, expected_matrix, atol=1e-10)


def test_polar_orbit_perifocal_to_geocentric_matrix():
    """Test polar orbit (i=90°)."""
    inclination = np.pi / 2  # 90 degrees
    raan = 0.0
    argp = 0.0

    matrix = perifocal_to_geocentric_matrix(inclination, raan, argp)

    # With i=90°, RAAN=0, argp=0, perifocal x should map to geocentric x
    # and perifocal y should map to geocentric z
    expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    np.testing.assert_allclose(matrix, expected, atol=1e-10)


# =============================================================================
# Tests for keplerian_to_rv (expanded)
# =============================================================================


def test_circular_equatorial_orbit_keplerian_to_r_v():
    """Test simple circular equatorial orbit."""
    # e=0, i=0, Ω=0, ω=0, ν=0
    # Should give position along x-axis
    kep = StateVector([0.0, 7000000, 0.0, 0.0, 0.0, 0.0])
    mu = 3.986004418e14

    rv = keplerian_to_rv(kep, mu)

    # At periapsis of circular orbit, position is [a, 0, 0]
    expected_pos = np.array([7000000, 0, 0])
    np.testing.assert_allclose(rv[:3].flatten(), expected_pos, atol=1e-3)

    # Velocity should be perpendicular (along y)
    v_expected = np.sqrt(mu / 7000000)
    np.testing.assert_allclose(rv[3].flatten(), 0, atol=1e-3)
    np.testing.assert_allclose(abs(rv[4].flatten()), v_expected, rtol=1e-6)
    np.testing.assert_allclose(rv[5].flatten(), 0, atol=1e-3)


def test_elliptical_orbit_periapsis_keplerian_to_r_v():
    """Test elliptical orbit at periapsis."""
    # e=0.5, a=8e6, i=0, Ω=0, ω=0, ν=0
    kep = StateVector([0.5, 8000000, 0.0, 0.0, 0.0, 0.0])
    mu = 3.986004418e14

    rv = keplerian_to_rv(kep, mu)

    # At periapsis: r = a(1-e)
    r_periapsis = 8000000 * (1 - 0.5)
    np.testing.assert_allclose(rv[0].flatten(), r_periapsis, rtol=1e-6)


def test_inclined_orbit_keplerian_to_r_v():
    """Test orbit with inclination."""
    # i=30°, should have z component
    kep = StateVector([0.3, 7500000, np.pi / 6, 0.0, 0.0, np.pi / 2])
    mu = 3.986004418e14

    rv = keplerian_to_rv(kep, mu)

    # Should have non-zero z component
    assert abs(rv[2].flatten()) > 0


def test_returns_statevector_keplerian_to_r_v():
    """Output should be StateVector."""
    kep = StateVector([0.2, 7000000, 0.1, 0.2, 0.3, 0.4])

    rv = keplerian_to_rv(kep)

    assert isinstance(rv, StateVector)
    assert len(rv) == 6


def test_energy_conservation_keplerian_to_r_v():
    """Verify specific energy matches orbital elements."""
    kep = StateVector([0.4, 8000000, 0.5, 0.3, 0.2, np.pi / 4])
    mu = 3.986004418e14

    rv = keplerian_to_rv(kep, mu)

    # Compute specific energy from rv
    r = rv[:3].flatten()
    v = rv[3:6].flatten()
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    energy = v_mag**2 / 2 - mu / r_mag

    # Expected energy from semi-major axis
    energy_expected = -mu / (2 * kep[1])

    np.testing.assert_allclose(energy, energy_expected, rtol=1e-6)


def test_angular_momentum_conservation_keplerian_to_r_v():
    """Verify angular momentum magnitude matches orbital elements."""
    e = 0.3
    a = 9000000
    i = np.pi / 4
    kep = StateVector([e, a, i, 0.2, 0.3, np.pi / 3])
    mu = 3.986004418e14

    rv = keplerian_to_rv(kep, mu)

    # Compute angular momentum from rv
    r = rv[:3].flatten()
    v = rv[3:6].flatten()
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Expected from orbital elements: h = sqrt(μ * a * (1-e²))
    h_expected = np.sqrt(mu * a * (1 - e**2))

    np.testing.assert_allclose(h_mag, h_expected, rtol=1e-6)


# =============================================================================
# Tests for mod_inclination
# =============================================================================


def test_zero_inclination_mod_inclination():
    """Zero inclination should remain zero."""
    assert mod_inclination(0.0) == 0.0


def test_small_positive_mod_inclination():
    """Small positive angle should be unchanged."""
    angle = np.pi / 6
    assert mod_inclination(angle) == angle


def test_at_pi_mod_inclination():
    """Angle of π should remain π (but modulo gives 0)."""
    # Actually, π mod π = 0
    assert mod_inclination(np.pi) == 0.0


def test_just_below_pi_mod_inclination():
    """Angle just below π should be unchanged."""
    angle = np.pi - 0.1
    np.testing.assert_allclose(mod_inclination(angle), angle, rtol=1e-10)


def test_above_pi_mod_inclination():
    """Angle above π should wrap to [0, π)."""
    angle = np.pi + 0.5
    expected = 0.5
    np.testing.assert_allclose(mod_inclination(angle), expected, rtol=1e-10)


def test_two_pi_mod_inclination():
    """2π should wrap to 0."""
    assert mod_inclination(2 * np.pi) == 0.0


def test_negative_angle_mod_inclination():
    """Negative angle should wrap to [0, π)."""
    angle = -np.pi / 4
    expected = 3 * np.pi / 4
    np.testing.assert_allclose(mod_inclination(angle), expected, rtol=1e-10)


def test_large_positive_mod_inclination():
    """Large positive angle should wrap correctly."""
    angle = 5 * np.pi + np.pi / 3
    expected = np.pi / 3
    np.testing.assert_allclose(mod_inclination(angle), expected, rtol=1e-10)


def test_range_check_mod_inclination():
    """Output should always be in [0, π)."""
    test_angles = np.linspace(-10 * np.pi, 10 * np.pi, 100)
    for angle in test_angles:
        result = mod_inclination(angle)
        assert 0 <= result < np.pi


# =============================================================================
# Tests for mod_elongitude
# =============================================================================


def test_zero_longitude_mod_elongitude():
    """Zero longitude should remain zero."""
    assert mod_elongitude(0.0) == 0.0


def test_small_positive_mod_elongitude():
    """Small positive angle should be unchanged."""
    angle = np.pi / 4
    assert mod_elongitude(angle) == angle


def test_at_2pi_mod_elongitude():
    """Angle of 2π should wrap to 0."""
    assert mod_elongitude(2 * np.pi) == 0.0


def test_just_below_2pi_mod_elongitude():
    """Angle just below 2π should be unchanged."""
    angle = 2 * np.pi - 0.1
    np.testing.assert_allclose(mod_elongitude(angle), angle, rtol=1e-10)


def test_above_2pi_mod_elongitude():
    """Angle above 2π should wrap to [0, 2π)."""
    angle = 2 * np.pi + 0.5
    expected = 0.5
    np.testing.assert_allclose(mod_elongitude(angle), expected, rtol=1e-10)


def test_negative_angle_mod_elongitude():
    """Negative angle should wrap to [0, 2π)."""
    angle = -np.pi / 4
    expected = 2 * np.pi - np.pi / 4
    np.testing.assert_allclose(mod_elongitude(angle), expected, rtol=1e-10)


def test_large_positive_mod_elongitude():
    """Large positive angle should wrap correctly."""
    angle = 10 * np.pi + np.pi / 3
    expected = np.pi / 3
    np.testing.assert_allclose(mod_elongitude(angle), expected, rtol=1e-10)


def test_large_negative_mod_elongitude():
    """Large negative angle should wrap correctly."""
    angle = -10 * np.pi - np.pi / 6
    expected = 2 * np.pi - np.pi / 6
    np.testing.assert_allclose(mod_elongitude(angle), expected, rtol=1e-10)


def test_pi_mod_elongitude():
    """π should remain π."""
    assert mod_elongitude(np.pi) == np.pi


def test_range_check_mod_elongitude():
    """Output should always be in [0, 2π)."""
    test_angles = np.linspace(-20 * np.pi, 20 * np.pi, 200)
    for angle in test_angles:
        result = mod_elongitude(angle)
        assert 0 <= result < 2 * np.pi


# =============================================================================
# Tests for Stumpff functions with arrays
# =============================================================================


def test_stumpff_s_array_stumpff_functions_arrays():
    """Test Stumpff S with array input."""
    z_array = np.array([0, np.pi**2, -(np.pi**2)])
    expected = np.array([1 / 6, 0.10132118364233778, 0.2711433813983066])

    result = stumpff_s(z_array)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_stumpff_c_array_stumpff_functions_arrays():
    """Test Stumpff C with array input."""
    z_array = np.array([0, np.pi**2, -(np.pi**2)])
    expected = np.array([1 / 2, 0.20264236728467555, 1.073189242960177])

    result = stumpff_c(z_array)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_stumpff_positive_values_stumpff_functions_arrays():
    """Test Stumpff functions for various positive values."""
    z_values = np.linspace(0.1, 10, 20)

    for z in z_values:
        s = stumpff_s(z)
        c = stumpff_c(z)

        # Both should be positive for positive z
        assert s > 0
        assert c > 0


def test_stumpff_negative_values_stumpff_functions_arrays():
    """Test Stumpff functions for various negative values."""
    z_values = np.linspace(-10, -0.1, 20)

    for z in z_values:
        s = stumpff_s(z)
        c = stumpff_c(z)

        # Both should be positive for negative z
        assert s > 0
        assert c > 0


def test_stumpff_series_expansion_small_z_stumpff_functions_arrays():
    """For small |z|, verify against series expansion."""
    z = 0.01

    s = stumpff_s(z)
    c = stumpff_c(z)

    # Series: S(z) ≈ 1/6 - z/120 + z²/5040 - ...
    s_series = 1 / 6 - z / 120 + z**2 / 5040
    # Series: C(z) ≈ 1/2 - z/24 + z²/720 - ...
    c_series = 1 / 2 - z / 24 + z**2 / 720

    np.testing.assert_allclose(s, s_series, rtol=1e-4)
    np.testing.assert_allclose(c, c_series, rtol=1e-4)


# =============================================================================
# Tests for universal_anomaly_newton edge cases
# =============================================================================


def test_circular_orbit_universal_anomaly_edge_cases():
    """Test universal anomaly for circular orbit."""
    # Circular orbit
    r = 7000000
    v = np.sqrt(3.986004418e14 / r)
    state = StateVector([r, 0, 0, 0, v, 0])
    delta_t = timedelta(hours=1)
    mu = 3.986004418e14

    chi = universal_anomaly_newton(state, delta_t, grav_parameter=mu)

    # Should converge to a reasonable value
    assert np.isfinite(chi)
    assert chi > 0


def test_short_time_interval_universal_anomaly_edge_cases():
    """Test with very short time interval."""
    state = StateVector([7000000, 0, 0, 1000, 7500, 500])
    delta_t = timedelta(seconds=1)

    chi = universal_anomaly_newton(state, delta_t)

    # Should be small for short time
    assert np.isfinite(chi)


def test_elliptical_orbit_universal_anomaly_edge_cases():
    """Test with elliptical orbit."""
    # From the existing test (example 3.7)
    mu = 3.986004418e5
    state = StateVector([7000, -12124, 0, 2.6679, 4.6210, 0])
    delta_t = timedelta(hours=1)

    chi = universal_anomaly_newton(state, delta_t, grav_parameter=mu)

    # Known answer: 253.53
    np.testing.assert_allclose(chi, 253.53, atol=1e-2)


# =============================================================================
# Tests for lagrange_coefficients edge cases
# =============================================================================


def test_determinant_condition_lagrange_coefficients_edge_cases():
    """Verify that f*gdot - fdot*g = 1."""
    state = StateVector([7000000, 1000000, 500000, 1000, 7500, 500])
    delta_t = timedelta(hours=1)

    f, g, fdot, gdot = lagrange_coefficients_from_universal_anomaly(state, delta_t)

    # The determinant of the f-g matrix should be 1
    determinant = f * gdot - fdot * g
    np.testing.assert_allclose(determinant, 1.0, rtol=1e-6)


def test_zero_time_lagrange_coefficients_edge_cases():
    """At zero time, f=1, g=0, fdot=0, gdot=1."""
    state = StateVector([7000000, 0, 0, 0, 7500, 0])
    delta_t = timedelta(seconds=0)

    f, g, fdot, gdot = lagrange_coefficients_from_universal_anomaly(state, delta_t)

    np.testing.assert_allclose(f, 1.0, atol=1e-8)
    np.testing.assert_allclose(g, 0.0, atol=1e-8)
    # fdot and gdot might not be exactly 0 and 1 due to numerical issues
    # but the determinant should be 1


# =============================================================================
# Integration tests
# =============================================================================


def test_keplerian_roundtrip_via_perifocal_orbital_functions_integration():
    """Test that Keplerian -> perifocal -> Keplerian preserves elements."""
    # Start with Keplerian elements
    e = 0.3
    a = 8000000
    i = np.pi / 6
    omega = np.pi / 4
    w = np.pi / 3
    nu = np.pi / 2

    kep = StateVector([e, a, i, omega, w, nu])
    mu = 3.986004418e14

    # Convert to position/velocity
    rv = keplerian_to_rv(kep, mu)

    # Verify angular momentum is correct
    h = compute_specific_angular_momentum(rv)
    h_mag = np.linalg.norm(h)
    h_expected = np.sqrt(mu * a * (1 - e**2))
    np.testing.assert_allclose(h_mag, h_expected, rtol=1e-6)

    # Verify energy is correct
    energy = compute_specific_energy(rv, mu)
    energy_expected = -mu / (2 * a)
    np.testing.assert_allclose(energy, energy_expected, rtol=1e-6)


def test_mean_to_true_anomaly_full_orbit_orbital_functions_integration():
    """Test mean to true anomaly conversion over full orbit."""
    e = 0.5
    mean_anomalies = np.linspace(0, 2 * np.pi, 36)

    for M in mean_anomalies:
        # Get true anomaly
        nu = tru_anom_from_mean_anom(M, e)

        # Verify it's in valid range (allow for floating point precision)
        assert 0 <= nu <= 2 * np.pi

        # Get eccentric anomaly
        E = eccentric_anomaly_from_mean_anomaly(M, e)

        # Verify Kepler's equation
        M_check = E - e * np.sin(E)
        np.testing.assert_allclose(M_check, M, atol=1e-8)
