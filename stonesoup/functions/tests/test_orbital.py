import pytest
import numpy as np
from datetime import datetime, timedelta

from ..orbital import (
    stumpff_c, stumpff_s, universal_anomaly_newton,
    lagrange_coefficients_from_universal_anomaly, keplerian_to_rv,
    orbital_state_eci_to_ecef, orbital_state_ecef_to_eci,
    orbital_state_j2000_to_gcrs, orbital_state_gcrs_to_j2000,
    compute_orbital_period, compute_orbital_velocity,
    compute_specific_angular_momentum, compute_specific_energy
)
from ...types.array import StateVector


@pytest.mark.parametrize(
    "z, outs, outc",
    [
        (0, 1/6, 1/2),
        (np.pi**2, 0.10132118364233778, 0.20264236728467555),
        (-(np.pi**2), 0.2711433813983066, 1.073189242960177),
    ]
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

    assert np.isclose(universal_anomaly_newton(start_at, deltat, grav_parameter=bigg),
                      chi_is, atol=1e-2)

    f, g, fdot, gdot = lagrange_coefficients_from_universal_anomaly(start_at, deltat,
                                                                    grav_parameter=bigg)
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
        np.array(state_recovered).flatten(),
        np.array(state_eci).flatten(),
        rtol=1e-10
    )


def test_ecef_eci_roundtrip():
    """Test that ECEF->ECI->ECEF returns original state."""
    state_ecef = StateVector([6378137, 1000000, 2000000, -500, 7000, 1000])
    timestamp = datetime(2024, 3, 21, 0, 0, 0)

    state_eci = orbital_state_ecef_to_eci(state_ecef, timestamp)
    state_recovered = orbital_state_eci_to_ecef(state_eci, timestamp)

    np.testing.assert_allclose(
        np.array(state_recovered).flatten(),
        np.array(state_ecef).flatten(),
        rtol=1e-10
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
        np.array(state_recovered).flatten(),
        np.array(state_j2000).flatten(),
        rtol=1e-10
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
    mu = 3.986004418e14

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
    inv_a = 2/r - v**2/mu
    a_from_visviva = 1 / inv_a

    np.testing.assert_allclose(a_computed, a_from_visviva, rtol=1e-10)
