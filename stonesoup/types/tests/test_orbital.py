r"""Test the various constructions of the orbital state vector. Take a known orbital state vector
and check the various parameterisations.

Example 4.3 from Curtis. Take the orbital state vector as input and check the various output
parameterisations. The input state vector is:

    .. math::

        \mathbf{r} = [-6045 \, -3490 \, 2500] \mathrm{km}

        \dot{\mathbf{r}} = [-3.457 \, 6.618 \, 2.553] \mathrm{km s^{-1}}

Selected outputs should be:

    magnitude of the specific orbital angular momentum, :math:`h = 58,310 \mathrm{km^2 s^{-1}}`

For the Keplerian elements
    semi-major axis, :math:`8788 \mathrm{km}`

    eccentricity, :math:`0.1712`

    inclination, :math:`2.674 \mathrm{rad}`

    longitude of ascending node, :math:`4.456 \mathrm{rad}`

    argument of periapsis, :math:`0.3503 \mathrm{rad}`

    true anomaly, :math:`0.4965 \mathrm{rad}`

TLE stuff
    eccentric anomaly, :math:`0.4202 \mathrm{rad}`

    mean anomaly, :math:`0.3504 \mathrm{rad}`

    period, :math:`8201 \mathrm{s}`

    mean motion, :math:`0.0007662 \mathrm{rad} \, \mathrm{s}^{-1}`

Equinoctial
     horizontal component of eccentricity, :math:`h = -0.1704`

     vertical component of eccentricity, :math:`k = 0.01605`

     horizontal component of the inclination, :math:`p =

     vertical component of the inclination :math:`q`

      mean longitude



"""

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("astropy")

from ...types.array import StateVector, StateVectors
from ..orbitalstate import OrbitalState

# Time
dtime = datetime.now()

# Orbital state vector in km and km/s
orb_st_vec = StateVector([-6045, -3490, 2500, -3.457, 6.618, 2.533])
# Initialise an equivalent StateVectors object
orb_st_vec2 = StateVector([-3756.52, 5626.22, 488.986, -4.20561, -2.29107, -5.98629])
orb_st_vecs = StateVectors([orb_st_vec, orb_st_vec2])

cartesian_s = OrbitalState(orb_st_vec, coordinates="Cartesian")
# ensure that the Gravitational parameter is in km^3 s^-2
cartesian_s.grav_parameter = cartesian_s.grav_parameter / 1e9
# Equivalent StateVectors object
cartesian_ss = OrbitalState(
    orb_st_vecs, coordinates="Cartesian", grav_parameter=cartesian_s.grav_parameter
)

# The Keplarian elements should be (to 4sf)
out_kep = StateVector([0.1712, 8788, 2.674, 4.456, 0.3503, 0.4965])
out_kep2 = StateVector([0.0003700, 6783, 0.9013, 5.358, 4.411, 4.922])
keplerian_s = OrbitalState(
    out_kep, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter, timestamp=dtime
)
out_keps = StateVectors([out_kep, out_kep2])
keplerian_ss = OrbitalState(
    out_kep2, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter, timestamp=dtime
)

# The TLE should be (to 4sf)
out_tle = StateVector([2.674, 4.456, 0.1712, 0.3503, 0.3504, 0.0007662])
out_tle2 = StateVector([0.9013, 5.358, 0.0003700, 4.411, 4.922, 0.001130])

# Equinoctial elements are (again, 4sf)
out_equ = StateVector([8788, -0.1704, 0.01605, -4.062, -1.065, 5.157])
out_equ2 = StateVector([6783, -0.0001250, -0.0003483, -0.3864, 0.2913, 2.125])


def test_incorrect_initialisation():
    """Run a bunch of tests to show that initialisations with the wrong parameters will fail."""

    bad_stvec = orb_st_vec[0:4]
    with pytest.raises(ValueError):
        OrbitalState(bad_stvec)

    with pytest.raises(ValueError):
        OrbitalState(orb_st_vec, coordinates="Nonsense")

    with pytest.raises(TypeError):
        OrbitalState(None, metadata=None, coordinates="TLE")

    # Push the relevant quantities outside of their limits one at a time
    bad_out_kep = np.copy(out_kep)
    bad_out_kep[0] = 1.2
    with pytest.raises(ValueError):
        OrbitalState(bad_out_kep, coordinates="keplerian")
    bad_out_tle = np.copy(out_tle)
    bad_out_tle[2] = 1.2
    with pytest.raises(ValueError):
        OrbitalState(bad_out_tle, coordinates="TLE")
    bad_out_equ = np.copy(out_equ)
    bad_out_equ[2] = -1.5
    with pytest.raises(ValueError):
        OrbitalState(bad_out_equ, coordinates="Equinoctial")
    bad_out_equ[1] = -1.5
    with pytest.raises(ValueError):
        OrbitalState(bad_out_equ, coordinates="Equinoctial")


# The next three tests ensure that the initialisations in different forms
# yield the same results
def test_kep_cart():
    # Test that Keplerian initialisation yields same state vector and state vectors
    # Firstly just flipping back and forth
    keplerian_sn = OrbitalState(
        cartesian_s.keplerian_elements,
        coordinates="keplerian",
        grav_parameter=cartesian_s.grav_parameter,
    )
    assert np.allclose(cartesian_s.state_vector, keplerian_sn.cartesian_state_vector, rtol=1e-4)

    keplerian_ssn = OrbitalState(
        cartesian_ss.keplerian_elements,
        coordinates="keplerian",
        grav_parameter=cartesian_s.grav_parameter,
    )

    # independent initialisation
    assert np.allclose(keplerian_s.state_vector, orb_st_vec, rtol=2e-3)
    assert np.allclose(keplerian_ssn.state_vector, orb_st_vecs, rtol=2e-3)

    # Test timestamp
    assert keplerian_s.epoch == dtime


def test_tle_cart():
    # Test that the TLE initialisation delivers the correct elements
    tle_sn = OrbitalState(
        cartesian_s.two_line_element, coordinates="TLE", grav_parameter=cartesian_s.grav_parameter
    )

    # Note that we need to convert to floats to do the comparison because np.allclose invokes the
    # np.isfinite() function which throws an error on Angle types
    assert np.allclose(
        np.float64(cartesian_s.two_line_element), np.float64(tle_sn.two_line_element), rtol=1e-3
    )

    # StateVectors equivalent
    tle_ssn = OrbitalState(
        cartesian_ss.two_line_element,
        coordinates="twolineelement",
        grav_parameter=cartesian_ss.grav_parameter,
    )
    assert np.allclose(
        np.float64(cartesian_ss.equinoctial_elements),
        np.float64(tle_ssn.equinoctial_elements),
        rtol=1e-3,
    )


def test_equ_cart():
    # Test that the equinoctial initialisation delivers the correct elements
    equ_sn = OrbitalState(
        cartesian_s.equinoctial_elements,
        coordinates="equinoctial",
        grav_parameter=cartesian_s.grav_parameter,
    )
    assert np.allclose(
        np.float64(cartesian_s.equinoctial_elements),
        np.float64(equ_sn.equinoctial_elements),
        rtol=1e-3,
    )

    # StateVectors equivalent
    equ_ssn = OrbitalState(
        cartesian_ss.equinoctial_elements,
        coordinates="Equinoctial",
        grav_parameter=cartesian_ss.grav_parameter,
    )
    assert np.allclose(
        np.float64(cartesian_ss.equinoctial_elements),
        np.float64(equ_ssn.equinoctial_elements),
        rtol=1e-3,
    )


# Now we need to test that the output is actually correct.
# Test Cartesian input and Keplerian output on known equivalents
def test_cart_kep():
    # Simple assertion
    assert np.all(cartesian_s.state_vector == orb_st_vec)
    # Check Keplerian elements come out right
    assert np.allclose(np.float64(cartesian_s.keplerian_elements), out_kep, rtol=1e-3)

    # This isn't tested elsewhere so do it here
    assert cartesian_s.specific_orbital_energy == cartesian_ss.specific_orbital_energy[0, 0]


# The test TLE output
def test_cart_tle():
    assert np.allclose(np.float64(cartesian_s.two_line_element), out_tle, rtol=1e-3)


# Test some specific quantities
def test_tle_via_metadata():
    """Initiate the orbitstate from a TLE (like you'd get from SpaceTrack).
    The TLE is an test TLE from copied verbatim with the following Cartesian
    state"""

    outstate = StateVector(
        [
            -3.75652102e06,
            5.62622198e06,
            4.88985712e05,
            -4.20560647e03,
            -2.29106828e03,
            -5.98628657e03,
        ]
    )

    lin1 = "1 25544U 98067A   18182.57105324 +.00001714 +00000-0 +33281-4 0  9991"
    lin2 = "2 25544 051.6426 307.0095 0003698 252.8831 281.8833 15.53996196120757"

    tle_metadata = {"line_1": lin1, "line_2": lin2}
    tle_state = OrbitalState(None, coordinates="TwoLineElement", metadata=tle_metadata)

    assert np.allclose(tle_state.state_vector, outstate, rtol=1e-4)

    # Check the dictionary contains items
    tle_dictionary = tle_state.tle_dict
    assert bool(tle_dictionary)


# Additional comprehensive tests for OrbitalState


def test_property_accessors():
    """Test that all the scalar property accessors work correctly."""
    # Test epoch property
    assert cartesian_s.epoch == cartesian_s.timestamp

    # Test range property
    expected_range = np.sqrt(orb_st_vec[0] ** 2 + orb_st_vec[1] ** 2 + orb_st_vec[2] ** 2)
    assert np.allclose(cartesian_s.range, expected_range, rtol=1e-4)

    # Test speed property
    expected_speed = np.sqrt(orb_st_vec[3] ** 2 + orb_st_vec[4] ** 2 + orb_st_vec[5] ** 2)
    assert np.allclose(cartesian_s.speed, expected_speed, rtol=1e-4)

    # Test cartesian_state_vector property
    assert np.allclose(cartesian_s.cartesian_state_vector, orb_st_vec, rtol=1e-4)

    # Test magnitude of specific angular momentum
    expected_h = 58310  # km^2/s
    assert np.allclose(cartesian_s.mag_specific_angular_momentum, expected_h, rtol=1e-2)

    # Test period (should be ~8201 s)
    assert np.allclose(cartesian_s.period, 8201, rtol=1e-2)

    # Test mean motion (should be ~0.0007662 rad/s)
    assert np.allclose(cartesian_s.mean_motion, 0.0007662, rtol=1e-2)

    # Test eccentric anomaly (should be ~0.4202 rad)
    assert np.allclose(np.float64(cartesian_s.eccentric_anomaly), 0.4202, rtol=1e-2)

    # Test mean anomaly (should be ~0.3504 rad)
    assert np.allclose(np.float64(cartesian_s.mean_anomaly), 0.3504, rtol=1e-2)


def test_equinoctial_elements():
    """Test equinoctial element computation."""
    # Test horizontal component of eccentricity (h = -0.1704)
    assert np.allclose(cartesian_s.equinoctial_h, -0.1704, rtol=1e-2)

    # Test vertical component of eccentricity (k = 0.01605)
    assert np.allclose(cartesian_s.equinoctial_k, 0.01605, rtol=1e-2)

    # Test complete equinoctial elements vector
    assert np.allclose(np.float64(cartesian_s.equinoctial_elements), out_equ, rtol=1e-2)


def test_circular_orbit():
    """Test handling of circular orbits (zero eccentricity)."""
    # Create a circular orbit (ISS-like orbit, 400 km altitude)
    # Using consistent units: position in meters, velocity in m/s, grav_parameter in m^3/s^2
    r_earth = 6378000  # m
    altitude = 400000  # m
    radius = r_earth + altitude
    grav_parameter_m3_s2 = 398600.4418e9  # m^3/s^2 (standard Earth GM)
    velocity = np.sqrt(grav_parameter_m3_s2 / radius)  # m/s (circular orbit velocity)

    # Circular orbit in equatorial plane
    circular_vec = StateVector([radius, 0, 0, 0, velocity, 0])
    circular_state = OrbitalState(
        circular_vec, coordinates="Cartesian", grav_parameter=grav_parameter_m3_s2
    )

    # Eccentricity should be very close to 0
    assert circular_state.eccentricity < 1e-6

    # For circular orbit, argument of periapsis should be set to 0 (by convention)
    # Note: due to numerical precision, we check for small eccentricity instead of exact 0
    if circular_state.eccentricity < np.finfo(float).eps:
        assert np.allclose(np.float64(circular_state.argument_periapsis), 0.0, atol=1e-6)

    # Semi-major axis should equal radius
    assert np.allclose(circular_state.semimajor_axis, radius, rtol=1e-4)


def test_zero_inclination():
    """Test handling of zero inclination (equatorial orbit)."""
    # Create an orbit in the equatorial plane
    r_earth = 6378  # km
    altitude = 400  # km
    radius = r_earth + altitude
    velocity = np.sqrt(cartesian_s.grav_parameter / (radius * 1000))

    equatorial_vec = StateVector([radius * 1000, 0, 0, 0, velocity, 0])
    equatorial_state = OrbitalState(
        equatorial_vec, coordinates="Cartesian", grav_parameter=cartesian_s.grav_parameter
    )

    # Inclination should be very close to 0
    assert np.allclose(np.float64(equatorial_state.inclination), 0.0, atol=1e-6)

    # For zero inclination, node line is set to [1, 0, 0] by convention
    # Verify longitude of ascending node behaves correctly
    assert np.isfinite(np.float64(equatorial_state.longitude_ascending_node))


def test_polar_orbit():
    """Test handling of polar orbits (90 degree inclination)."""
    # Create a polar orbit
    r_earth = 6378  # km
    altitude = 800  # km
    radius = r_earth + altitude
    velocity = np.sqrt(cartesian_s.grav_parameter / (radius * 1000))

    # Position along x-axis, velocity along z-axis (polar)
    polar_vec = StateVector([radius * 1000, 0, 0, 0, 0, velocity])
    polar_state = OrbitalState(
        polar_vec, coordinates="Cartesian", grav_parameter=cartesian_s.grav_parameter
    )

    # Inclination should be close to pi/2
    assert np.allclose(np.float64(polar_state.inclination), np.pi / 2, rtol=1e-4)


def test_highly_eccentric_orbit():
    """Test handling of highly eccentric orbits."""
    # Create Keplerian elements for a highly eccentric orbit (e.g., Molniya orbit)
    ecc = 0.7  # High eccentricity
    sma = 26600  # km
    inc = np.radians(63.4)  # Molniya inclination
    raan = 0.0
    argp = 0.0
    tran = 0.0  # At periapsis

    molniya_kep = StateVector([ecc, sma, inc, raan, argp, tran])
    molniya_state = OrbitalState(
        molniya_kep, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter
    )

    # Verify eccentricity is preserved
    assert np.allclose(molniya_state.eccentricity, ecc, rtol=1e-4)

    # Verify semi-major axis is preserved
    assert np.allclose(molniya_state.semimajor_axis, sma, rtol=1e-4)


def test_statevectors_consistency():
    """Test that StateVectors objects work consistently with StateVector."""
    # Test that properties work for both single and multiple vectors
    assert np.allclose(cartesian_s.eccentricity, cartesian_ss.eccentricity[0, 0], rtol=1e-4)
    assert np.allclose(cartesian_s.semimajor_axis, cartesian_ss.semimajor_axis[0, 0], rtol=1e-4)
    assert np.allclose(
        np.float64(cartesian_s.inclination),
        np.float64(cartesian_ss.inclination)[0, 0],
        rtol=1e-4,
    )
    assert np.allclose(
        np.float64(cartesian_s.longitude_ascending_node),
        np.float64(cartesian_ss.longitude_ascending_node)[0, 0],
        rtol=1e-4,
    )


def test_vector_properties():
    """Test vector properties like specific angular momentum and eccentricity vector."""
    # Test specific angular momentum is perpendicular to position and velocity
    h_vec = cartesian_s.specific_angular_momentum
    r_vec = cartesian_s.state_vector[0:3]
    v_vec = cartesian_s.state_vector[3:6]

    # h should be perpendicular to r (dot product ~ 0)
    assert np.allclose(np.dot(h_vec.flatten(), r_vec.flatten()), 0, atol=1e-6)

    # h should be perpendicular to v (dot product ~ 0)
    assert np.allclose(np.dot(h_vec.flatten(), v_vec.flatten()), 0, atol=1e-6)


def test_energy_conservation():
    """Test that specific orbital energy is computed correctly."""
    # Specific orbital energy should equal -mu / (2a)
    expected_energy = -cartesian_s.grav_parameter / (2 * cartesian_s.semimajor_axis)
    assert np.allclose(cartesian_s.specific_orbital_energy, expected_energy, rtol=1e-4)


def test_keplerian_elements_vector():
    """Test that complete Keplerian elements vector is correctly constructed."""
    kep_vec = cartesian_s.keplerian_elements

    # Check shape
    assert kep_vec.shape == (6, 1)

    # Check individual elements
    assert np.allclose(kep_vec[0], cartesian_s.eccentricity, rtol=1e-4)
    assert np.allclose(kep_vec[1], cartesian_s.semimajor_axis, rtol=1e-4)
    assert np.allclose(np.float64(kep_vec[2]), np.float64(cartesian_s.inclination), rtol=1e-4)
    assert np.allclose(
        np.float64(kep_vec[3]), np.float64(cartesian_s.longitude_ascending_node), rtol=1e-4
    )
    assert np.allclose(
        np.float64(kep_vec[4]), np.float64(cartesian_s.argument_periapsis), rtol=1e-4
    )
    assert np.allclose(np.float64(kep_vec[5]), np.float64(cartesian_s.true_anomaly), rtol=1e-4)


def test_two_line_element_vector():
    """Test that TLE vector is correctly constructed."""
    tle_vec = cartesian_s.two_line_element

    # Check shape
    assert tle_vec.shape == (6, 1)

    # Check individual elements match expected output
    assert np.allclose(np.float64(tle_vec), out_tle, rtol=1e-3)


def test_coordinate_system_enum():
    """Test CoordinateSystem enum functionality."""
    from ..orbitalstate import CoordinateSystem

    # Test case insensitivity
    assert CoordinateSystem("cartesian") == CoordinateSystem.CARTESIAN
    assert CoordinateSystem("CARTESIAN") == CoordinateSystem.CARTESIAN
    assert CoordinateSystem("Cartesian") == CoordinateSystem.CARTESIAN

    # Test TLE aliases
    assert CoordinateSystem("TLE") == CoordinateSystem.TLE
    assert CoordinateSystem("tle") == CoordinateSystem.TLE
    assert CoordinateSystem("twolineelement") == CoordinateSystem.TLE
    assert CoordinateSystem("TwoLineElement") == CoordinateSystem.TLE

    # Test invalid value
    with pytest.raises(ValueError):
        CoordinateSystem("invalid")


def test_tle_metadata_properties():
    """Test that TLE metadata properties are correctly set."""
    lin1 = "1 25544U 98067A   18182.57105324 +.00001714 +00000-0 +33281-4 0  9991"
    lin2 = "2 25544 051.6426 307.0095 0003698 252.8831 281.8833 15.53996196120757"

    tle_metadata = {"line_1": lin1, "line_2": lin2}
    tle_state = OrbitalState(None, coordinates="TLE", metadata=tle_metadata)

    # Check that metadata properties are set
    assert tle_state.catalogue_number == 25544
    assert tle_state.classification == "U"
    assert tle_state.international_designator == "98067A  "
    assert tle_state.ephemeris_type == 0
    assert tle_state.element_set_number == 999


def test_tle_dict_output():
    """Test that tle_dict property produces valid TLE strings."""
    lin1 = "1 25544U 98067A   18182.57105324 +.00001714 +00000-0 +33281-4 0  9991"
    lin2 = "2 25544 051.6426 307.0095 0003698 252.8831 281.8833 15.53996196120757"

    tle_metadata = {"line_1": lin1, "line_2": lin2}
    tle_state = OrbitalState(None, coordinates="TLE", metadata=tle_metadata)

    # Get the TLE dictionary
    tle_dict = tle_state.tle_dict

    # Check that dictionary has required keys
    assert "line_1" in tle_dict
    assert "line_2" in tle_dict

    # Check that lines start correctly
    assert tle_dict["line_1"].startswith("1 25544")
    assert tle_dict["line_2"].startswith("2 25544")

    # Check that lines have correct length (69 characters + checksum)
    assert len(tle_dict["line_1"]) == 69
    assert len(tle_dict["line_2"]) == 69


def test_grav_parameter_default():
    """Test that default gravitational parameter is Earth's."""
    state = OrbitalState(orb_st_vec, coordinates="Cartesian")
    # Default should be Earth's gravitational parameter in m^3/s^2
    assert state.grav_parameter == 3.986004418e14


def test_grav_parameter_custom():
    """Test that custom gravitational parameter can be set."""
    # Use Moon's gravitational parameter
    moon_mu = 4.9028e12  # m^3/s^2
    state = OrbitalState(orb_st_vec, coordinates="Cartesian", grav_parameter=moon_mu)
    assert state.grav_parameter == moon_mu


def test_metadata_preservation():
    """Test that metadata is preserved through initialization."""
    metadata = {"source": "test", "satellite_name": "TestSat"}
    state = OrbitalState(orb_st_vec, coordinates="Cartesian", metadata=metadata)
    assert state.metadata == metadata


def test_timestamp_preservation():
    """Test that timestamp is preserved correctly."""
    state = OrbitalState(orb_st_vec, coordinates="Cartesian", timestamp=dtime)
    assert state.timestamp == dtime
    assert state.epoch == dtime


# Tests for reference frame transformations (OrbitalState specific)


def test_reference_frame_default():
    """Test that default reference frame is J2000."""
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(orb_st_vec, coordinates="Cartesian")
    assert state.reference_frame == ReferenceFrameType.J2000


def test_reference_frame_enum():
    """Test ReferenceFrameType enum functionality."""
    from ..orbitalstate import ReferenceFrameType

    # Test case insensitivity
    assert ReferenceFrameType("j2000") == ReferenceFrameType.J2000
    assert ReferenceFrameType("J2000") == ReferenceFrameType.J2000
    assert ReferenceFrameType("gcrs") == ReferenceFrameType.GCRS
    assert ReferenceFrameType("GCRS") == ReferenceFrameType.GCRS
    assert ReferenceFrameType("icrs") == ReferenceFrameType.ICRS
    assert ReferenceFrameType("ICRS") == ReferenceFrameType.ICRS

    # Test ECI is a valid enum member (conceptually similar to J2000 but a separate value)
    assert ReferenceFrameType("eci") == ReferenceFrameType.ECI
    assert ReferenceFrameType("ECI") == ReferenceFrameType.ECI
    # ECI is a separate enum value, not an alias
    assert ReferenceFrameType.ECI.value == "ECI"
    assert ReferenceFrameType.J2000.value == "J2000"

    # Test invalid value
    with pytest.raises(ValueError):
        ReferenceFrameType("invalid")


def test_orbital_state_position_velocity_properties():
    """Test position and velocity property accessors."""
    state = OrbitalState(orb_st_vec, coordinates="Cartesian")

    # Check position property
    expected_position = np.array(orb_st_vec[0:3]).flatten()
    assert np.allclose(state.position, expected_position)

    # Check velocity property
    expected_velocity = np.array(orb_st_vec[3:6]).flatten()
    assert np.allclose(state.velocity, expected_velocity)


def test_transform_to_same_frame():
    """Test that transforming to the same frame returns the original state."""
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(
        orb_st_vec, coordinates="Cartesian", reference_frame=ReferenceFrameType.J2000
    )
    transformed = state.transform_to_frame(ReferenceFrameType.J2000)

    # Should return the same state
    assert transformed is state


def test_transform_frame_string_input():
    """Test that transform_to_frame accepts string inputs."""
    state = OrbitalState(orb_st_vec, coordinates="Cartesian", timestamp=dtime)

    # Should accept string
    transformed = state.transform_to_frame("GCRS", timestamp=dtime)
    from ..orbitalstate import ReferenceFrameType

    assert transformed.reference_frame == ReferenceFrameType.GCRS


def test_orbital_state_to_j2000():
    """Test to_j2000 convenience method."""
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(
        orb_st_vec, coordinates="Cartesian", reference_frame=ReferenceFrameType.GCRS
    )
    j2000_state = state.to_j2000(timestamp=dtime)

    assert j2000_state.reference_frame == ReferenceFrameType.J2000


def test_orbital_state_to_gcrs():
    """Test to_gcrs convenience method."""
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(
        orb_st_vec, coordinates="Cartesian", reference_frame=ReferenceFrameType.J2000
    )
    gcrs_state = state.to_gcrs(timestamp=dtime)

    assert gcrs_state.reference_frame == ReferenceFrameType.GCRS


def test_orbital_state_to_icrs():
    """Test to_icrs convenience method."""
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(
        orb_st_vec, coordinates="Cartesian", reference_frame=ReferenceFrameType.J2000
    )
    icrs_state = state.to_icrs(timestamp=dtime)

    assert icrs_state.reference_frame == ReferenceFrameType.ICRS


def test_get_frame_instance():
    """Test get_frame_instance method."""
    from ...types.coordinates import J2000
    from ..orbitalstate import ReferenceFrameType

    state = OrbitalState(
        orb_st_vec, coordinates="Cartesian", reference_frame=ReferenceFrameType.J2000
    )
    frame = state.get_frame_instance()

    assert isinstance(frame, J2000)


# Tests for GaussianOrbitalState


def test_gaussian_orbital_state_creation():
    """Test creation of GaussianOrbitalState."""
    from ..orbitalstate import GaussianOrbitalState

    covariance = np.eye(6) * 100  # Simple diagonal covariance

    gaussian_state = GaussianOrbitalState(
        state_vector=orb_st_vec, covar=covariance, coordinates="Cartesian"
    )

    # Check that it has both orbital and Gaussian properties
    assert gaussian_state.state_vector is not None
    assert gaussian_state.covar is not None
    assert np.allclose(gaussian_state.covar, covariance)


def test_gaussian_orbital_state_properties():
    """Test that GaussianOrbitalState has access to orbital properties."""
    from ..orbitalstate import GaussianOrbitalState

    covariance = np.eye(6) * 100

    gaussian_state = GaussianOrbitalState(
        state_vector=orb_st_vec, covar=covariance, coordinates="Cartesian"
    )

    # Ensure covariance is properly set (convert to km units for consistency)
    gaussian_state.grav_parameter = gaussian_state.grav_parameter / 1e9

    # Should have access to orbital properties
    assert gaussian_state.eccentricity is not None
    assert gaussian_state.semimajor_axis is not None
    assert gaussian_state.inclination is not None


def test_gaussian_orbital_state_keplerian():
    """Test GaussianOrbitalState initialized from Keplerian elements."""
    from ..orbitalstate import GaussianOrbitalState

    covariance = np.eye(6) * 0.01  # Small covariance for Keplerian elements

    gaussian_state = GaussianOrbitalState(
        state_vector=out_kep,
        covar=covariance,
        coordinates="Keplerian",
        grav_parameter=cartesian_s.grav_parameter,
    )

    # Should convert to Cartesian internally
    assert gaussian_state.state_vector is not None
    assert len(gaussian_state.state_vector) == 6


# Tests for ParticleOrbitalState


def test_particle_orbital_state_creation():
    """Test creation of ParticleOrbitalState."""
    from ..orbitalstate import ParticleOrbitalState

    # Create particles (multiple state vectors with weights)
    n_particles = 100
    particles = StateVectors([orb_st_vec for _ in range(n_particles)])
    weights = np.ones(n_particles) / n_particles

    particle_state = ParticleOrbitalState(
        state_vector=particles,
        weight=weights,
        coordinates="Cartesian",
        parent=None,
        particle_list=None,
    )

    # Check that it has both orbital and particle properties
    assert particle_state.state_vector is not None
    assert particle_state.weight is not None
    assert len(particle_state.weight) == n_particles


def test_particle_orbital_state_properties():
    """Test that ParticleOrbitalState has access to orbital properties."""
    from ..orbitalstate import ParticleOrbitalState

    n_particles = 10
    particles = StateVectors([orb_st_vec for _ in range(n_particles)])
    weights = np.ones(n_particles) / n_particles

    particle_state = ParticleOrbitalState(
        state_vector=particles,
        weight=weights,
        coordinates="Cartesian",
        parent=None,
        particle_list=None,
    )

    # Ensure gravitational parameter is set correctly (convert to km units)
    particle_state.grav_parameter = particle_state.grav_parameter / 1e9

    # Should have access to orbital properties (returns arrays)
    assert particle_state.eccentricity is not None
    assert particle_state.semimajor_axis is not None
    assert particle_state.inclination is not None


# Edge case tests


def test_retrograde_orbit():
    """Test handling of retrograde orbits (inclination > 90 degrees)."""
    # Create a retrograde orbit (inclination > 90 degrees)
    ecc = 0.1
    sma = 8000  # km
    inc = np.radians(120)  # Retrograde
    raan = 0.0
    argp = 0.0
    tran = 0.0

    retro_kep = StateVector([ecc, sma, inc, raan, argp, tran])
    retro_state = OrbitalState(
        retro_kep, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter
    )

    # Verify inclination is preserved
    assert np.allclose(np.float64(retro_state.inclination), inc, rtol=1e-4)


def test_near_parabolic_orbit():
    """Test handling of near-parabolic orbits (eccentricity close to 1)."""
    # Create a highly eccentric orbit (close to parabolic)
    ecc = 0.99  # Very high eccentricity
    sma = 20000  # km
    inc = np.radians(45)
    raan = 0.0
    argp = 0.0
    tran = 0.0

    parabolic_kep = StateVector([ecc, sma, inc, raan, argp, tran])
    parabolic_state = OrbitalState(
        parabolic_kep, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter
    )

    # Verify eccentricity is preserved
    assert np.allclose(parabolic_state.eccentricity, ecc, rtol=1e-4)


def test_anomaly_conversions():
    """Test conversions between different anomaly types."""
    # Create an orbit at a specific true anomaly
    true_anom_input = np.radians(45)
    ecc = 0.2
    sma = 8000  # km
    inc = np.radians(30)
    raan = 0.0
    argp = 0.0

    kep_vec = StateVector([ecc, sma, inc, raan, argp, true_anom_input])
    state = OrbitalState(
        kep_vec, coordinates="Keplerian", grav_parameter=cartesian_s.grav_parameter
    )

    # Check that true anomaly is preserved
    assert np.allclose(np.float64(state.true_anomaly), true_anom_input, rtol=1e-3)

    # Check that eccentric anomaly is computed
    ecc_anom = state.eccentric_anomaly
    assert ecc_anom is not None

    # Check that mean anomaly is computed
    mean_anom = state.mean_anomaly
    assert mean_anom is not None

    # Verify Kepler's equation: M = E - e*sin(E)
    expected_mean = np.float64(ecc_anom) - ecc * np.sin(np.float64(ecc_anom))
    assert np.allclose(np.float64(mean_anom), expected_mean, rtol=1e-4)


def test_multiple_revolutions():
    """Test that angles are properly bounded to [0, 2pi]."""
    # Create an orbit and verify all angles are in valid ranges
    state = OrbitalState(orb_st_vec, coordinates="Cartesian")
    state.grav_parameter = state.grav_parameter / 1e9

    # All longitude-type angles should be in [0, 2pi]
    assert 0 <= np.float64(state.longitude_ascending_node) < 2 * np.pi
    assert 0 <= np.float64(state.argument_periapsis) < 2 * np.pi
    assert 0 <= np.float64(state.true_anomaly) < 2 * np.pi
    assert 0 <= np.float64(state.eccentric_anomaly) < 2 * np.pi
    assert 0 <= np.float64(state.mean_longitude) < 4 * np.pi  # Can exceed 2pi as sum

    # Inclination should be in [0, pi]
    assert 0 <= np.float64(state.inclination) <= np.pi


def test_equinoctial_roundtrip():
    """Test that equinoctial elements can be used for initialization and retrieval."""
    # Get equinoctial elements from a Cartesian state
    equ_elements = cartesian_s.equinoctial_elements

    # Create new state from equinoctial elements
    equ_state = OrbitalState(
        equ_elements, coordinates="Equinoctial", grav_parameter=cartesian_s.grav_parameter
    )

    # Compare Cartesian state vectors (should be very close)
    assert np.allclose(
        equ_state.cartesian_state_vector, cartesian_s.cartesian_state_vector, rtol=1e-3
    )


def test_tle_roundtrip():
    """Test that TLE elements can be used for initialization and retrieval."""
    # Get TLE from Cartesian state
    tle_elements = cartesian_s.two_line_element

    # Create new state from TLE
    tle_state = OrbitalState(
        tle_elements, coordinates="TLE", grav_parameter=cartesian_s.grav_parameter
    )

    # Compare TLE elements (should be very close)
    assert np.allclose(np.float64(tle_state.two_line_element), np.float64(tle_elements), rtol=1e-3)
