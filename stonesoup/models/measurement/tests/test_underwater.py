"""Tests for underwater/sonar measurement models."""

import numpy as np
import pytest
from pytest import approx

from ....types.array import CovarianceMatrix, StateVector
from ....types.detection import Detection
from ....types.state import State
from ..underwater import (
    CartesianToBearingElevationRange,
    CartesianToBearingOnly,
    CartesianToBearingRangeDoppler,
    CartesianToDepthBearingRange,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def depth_bearing_range_model():
    """Create standard depth-bearing-range test model."""
    return CartesianToDepthBearingRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=CovarianceMatrix(np.diag([1.0, 0.01, 1.0])),
    )


@pytest.fixture
def bearing_only_model():
    """Create standard bearing-only test model."""
    return CartesianToBearingOnly(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=CovarianceMatrix([[0.01]]),
    )


@pytest.fixture
def bearing_elevation_range_model():
    """Create standard bearing-elevation-range test model."""
    return CartesianToBearingElevationRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=CovarianceMatrix(np.diag([0.01, 0.01, 1.0])),
    )


@pytest.fixture
def doppler_model():
    """Create standard Doppler sonar test model."""
    return CartesianToBearingRangeDoppler(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=(1, 3, 5),
        noise_covar=CovarianceMatrix(np.diag([0.01, 1.0, 0.1])),
    )


# =============================================================================
# CartesianToDepthBearingRange Tests
# =============================================================================


def test_depth_bearing_range_ndim_meas(depth_bearing_range_model):
    """Model should have 3 measurement dimensions."""
    assert depth_bearing_range_model.ndim_meas == 3


def test_depth_bearing_range_function_at_origin(depth_bearing_range_model):
    """Test measurement at origin."""
    state = State(StateVector([0, 0, 0, 0, 0, 0]))
    meas = depth_bearing_range_model.function(state, noise=False)
    assert meas[0, 0] == 0  # depth
    assert meas[2, 0] == 0  # range


def test_depth_bearing_range_function_underwater(depth_bearing_range_model):
    """Test measurement for underwater target."""
    # Target at (100, 200, -50) - 50m below surface
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    meas = depth_bearing_range_model.function(state, noise=False)

    assert meas[0, 0] == approx(50)  # depth = -z = 50
    # Bearing from north = atan2(100, 200)
    expected_bearing = np.arctan2(100, 200)
    assert meas[1, 0] == approx(expected_bearing)
    # Slant range
    expected_range = np.sqrt(100**2 + 200**2 + 50**2)
    assert meas[2, 0] == approx(expected_range)


def test_depth_bearing_range_function_due_north(depth_bearing_range_model):
    """Target due north should have bearing = 0."""
    state = State(StateVector([0, 0, 1000, 0, 0, 0]))
    meas = depth_bearing_range_model.function(state, noise=False)
    assert meas[1, 0] == approx(0, abs=1e-10)


def test_depth_bearing_range_function_due_east(depth_bearing_range_model):
    """Target due east should have bearing = pi/2."""
    state = State(StateVector([1000, 0, 0, 0, 0, 0]))
    meas = depth_bearing_range_model.function(state, noise=False)
    assert meas[1, 0] == approx(np.pi / 2, abs=1e-10)


def test_depth_bearing_range_function_with_noise(depth_bearing_range_model):
    """Test that noise is added correctly."""
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    np.random.seed(42)
    meas_noisy = depth_bearing_range_model.function(state, noise=True)
    meas_clean = depth_bearing_range_model.function(state, noise=False)
    # Should be different due to noise
    assert not np.allclose(meas_noisy, meas_clean)


def test_depth_bearing_range_jacobian_shape(depth_bearing_range_model):
    """Jacobian should have correct shape."""
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    jac = depth_bearing_range_model.jacobian(state)
    assert jac.shape == (3, 6)


def test_depth_bearing_range_jacobian_numerical(depth_bearing_range_model):
    """Jacobian should match numerical approximation."""
    state = State(StateVector([100.0, 0.0, 200.0, 0.0, -50.0, 0.0]))
    jac = depth_bearing_range_model.jacobian(state)

    # Numerical Jacobian
    eps = 1e-7
    num_jac = np.zeros((3, 6))
    for i in range(6):
        sv_plus = np.array(state.state_vector, dtype=float).copy()
        sv_plus[i, 0] += eps
        sv_minus = np.array(state.state_vector, dtype=float).copy()
        sv_minus[i, 0] -= eps
        meas_plus = np.array(
            depth_bearing_range_model.function(State(StateVector(sv_plus)), noise=False),
            dtype=float,
        )
        meas_minus = np.array(
            depth_bearing_range_model.function(State(StateVector(sv_minus)), noise=False),
            dtype=float,
        )
        num_jac[:, i] = ((meas_plus - meas_minus) / (2 * eps)).ravel()

    np.testing.assert_allclose(jac, num_jac, atol=1e-5)


def test_depth_bearing_range_inverse_function(depth_bearing_range_model):
    """Inverse function should recover position."""
    original_state = State(StateVector([100, 0, 200, 0, -50, 0]))
    meas = depth_bearing_range_model.function(original_state, noise=False)
    detection = Detection(meas)
    recovered = depth_bearing_range_model.inverse_function(detection)

    assert recovered[0, 0] == approx(100, rel=1e-10)
    assert recovered[2, 0] == approx(200, rel=1e-10)
    assert recovered[4, 0] == approx(-50, rel=1e-10)


def test_depth_bearing_range_with_translation_offset():
    """Test model with sensor offset."""
    offset = StateVector([10, 20, 5])
    model = CartesianToDepthBearingRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=CovarianceMatrix(np.diag([1.0, 0.01, 1.0])),
        translation_offset=offset,
    )
    # Target at (10, 20, 5) should be at origin relative to sensor
    state = State(StateVector([10, 0, 20, 0, 5, 0]))
    meas = model.function(state, noise=False)
    assert meas[2, 0] == approx(0, abs=1e-10)  # range = 0


# =============================================================================
# CartesianToBearingOnly Tests
# =============================================================================


def test_bearing_only_ndim_meas(bearing_only_model):
    """Model should have 1 measurement dimension."""
    assert bearing_only_model.ndim_meas == 1


def test_bearing_only_function_due_north(bearing_only_model):
    """Target due north should have bearing = 0."""
    state = State(StateVector([0, 0, 1000, 0]))
    meas = bearing_only_model.function(state, noise=False)
    assert meas[0, 0] == approx(0, abs=1e-10)


def test_bearing_only_function_due_east(bearing_only_model):
    """Target due east should have bearing = pi/2."""
    state = State(StateVector([1000, 0, 0, 0]))
    meas = bearing_only_model.function(state, noise=False)
    assert meas[0, 0] == approx(np.pi / 2, abs=1e-10)


def test_bearing_only_function_due_south(bearing_only_model):
    """Target due south should have bearing = +/- pi."""
    state = State(StateVector([0, 0, -1000, 0]))
    meas = bearing_only_model.function(state, noise=False)
    assert abs(meas[0, 0]) == approx(np.pi, abs=1e-10)


def test_bearing_only_function_due_west(bearing_only_model):
    """Target due west should have bearing = -pi/2."""
    state = State(StateVector([-1000, 0, 0, 0]))
    meas = bearing_only_model.function(state, noise=False)
    assert meas[0, 0] == approx(-np.pi / 2, abs=1e-10)


def test_bearing_only_jacobian_shape(bearing_only_model):
    """Jacobian should have correct shape."""
    state = State(StateVector([100, 0, 200, 0]))
    jac = bearing_only_model.jacobian(state)
    assert jac.shape == (1, 4)


def test_bearing_only_jacobian_numerical(bearing_only_model):
    """Jacobian should match numerical approximation."""
    state = State(StateVector([100.0, 0.0, 200.0, 0.0]))
    jac = bearing_only_model.jacobian(state)

    eps = 1e-7
    num_jac = np.zeros((1, 4))
    for i in range(4):
        sv_plus = np.array(state.state_vector, dtype=float).copy()
        sv_plus[i, 0] += eps
        sv_minus = np.array(state.state_vector, dtype=float).copy()
        sv_minus[i, 0] -= eps
        meas_plus = np.array(
            bearing_only_model.function(State(StateVector(sv_plus)), noise=False), dtype=float
        )
        meas_minus = np.array(
            bearing_only_model.function(State(StateVector(sv_minus)), noise=False), dtype=float
        )
        diff = meas_plus - meas_minus
        # Handle angle wrapping
        while diff[0, 0] > np.pi:
            diff[0, 0] -= 2 * np.pi
        while diff[0, 0] < -np.pi:
            diff[0, 0] += 2 * np.pi
        num_jac[:, i] = (diff / (2 * eps)).ravel()

    np.testing.assert_allclose(jac, num_jac, atol=1e-5)


# =============================================================================
# CartesianToBearingElevationRange Tests
# =============================================================================


def test_bearing_elevation_range_ndim_meas(bearing_elevation_range_model):
    """Model should have 3 measurement dimensions."""
    assert bearing_elevation_range_model.ndim_meas == 3


def test_bearing_elevation_range_function_horizontal(bearing_elevation_range_model):
    """Horizontal target should have elevation = 0."""
    state = State(StateVector([100, 0, 100, 0, 0, 0]))
    meas = bearing_elevation_range_model.function(state, noise=False)
    assert meas[1, 0] == approx(0, abs=1e-10)


def test_bearing_elevation_range_function_above(bearing_elevation_range_model):
    """Target directly above should have elevation = pi/2."""
    state = State(StateVector([0, 0, 0, 0, 100, 0]))
    meas = bearing_elevation_range_model.function(state, noise=False)
    assert meas[1, 0] == approx(np.pi / 2, abs=1e-10)


def test_bearing_elevation_range_function_below(bearing_elevation_range_model):
    """Target directly below should have elevation = -pi/2."""
    state = State(StateVector([0, 0, 0, 0, -100, 0]))
    meas = bearing_elevation_range_model.function(state, noise=False)
    assert meas[1, 0] == approx(-np.pi / 2, abs=1e-10)


def test_bearing_elevation_range_function_3d(bearing_elevation_range_model):
    """Test full 3D measurement."""
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    meas = bearing_elevation_range_model.function(state, noise=False)

    # Bearing from north
    expected_bearing = np.arctan2(100, 200)
    assert meas[0, 0] == approx(expected_bearing)

    # Slant range
    expected_range = np.sqrt(100**2 + 200**2 + 50**2)
    assert meas[2, 0] == approx(expected_range)


def test_bearing_elevation_range_jacobian_shape(bearing_elevation_range_model):
    """Jacobian should have correct shape."""
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    jac = bearing_elevation_range_model.jacobian(state)
    assert jac.shape == (3, 6)


def test_bearing_elevation_range_inverse_function(bearing_elevation_range_model):
    """Inverse function should recover position."""
    original_state = State(StateVector([100, 0, 200, 0, -50, 0]))
    meas = bearing_elevation_range_model.function(original_state, noise=False)
    detection = Detection(meas)
    recovered = bearing_elevation_range_model.inverse_function(detection)

    assert recovered[0, 0] == approx(100, rel=1e-10)
    assert recovered[2, 0] == approx(200, rel=1e-10)
    assert recovered[4, 0] == approx(-50, rel=1e-10)


# =============================================================================
# CartesianToBearingRangeDoppler Tests
# =============================================================================


def test_doppler_ndim_meas(doppler_model):
    """Model should have 3 measurement dimensions."""
    assert doppler_model.ndim_meas == 3


def test_doppler_function_stationary_target(doppler_model):
    """Stationary target should have zero range rate."""
    state = State(StateVector([100, 0, 200, 0, -50, 0]))
    meas = doppler_model.function(state, noise=False)
    assert meas[2, 0] == approx(0, abs=1e-10)


def test_doppler_function_approaching_target(doppler_model):
    """Target moving toward sensor should have negative range rate."""
    # Target at (100, 0) moving toward origin at (-10, 0)
    state = State(StateVector([100, -10, 0, 0, 0, 0]))
    meas = doppler_model.function(state, noise=False)
    assert meas[2, 0] < 0  # Approaching = negative range rate


def test_doppler_function_receding_target(doppler_model):
    """Target moving away should have positive range rate."""
    # Target at (100, 0) moving away at (10, 0)
    state = State(StateVector([100, 10, 0, 0, 0, 0]))
    meas = doppler_model.function(state, noise=False)
    assert meas[2, 0] > 0  # Receding = positive range rate


def test_doppler_function_range_rate_value(doppler_model):
    """Test specific range rate calculation."""
    # Target at (100, 0, 0) moving radially outward at 5 m/s
    state = State(StateVector([100, 5, 0, 0, 0, 0]))
    meas = doppler_model.function(state, noise=False)
    assert meas[1, 0] == approx(100)  # range
    assert meas[2, 0] == approx(5)  # range rate = radial velocity


def test_doppler_function_tangential_velocity(doppler_model):
    """Tangential velocity should not affect range rate."""
    # Target at (100, 0, 0) moving tangentially at (0, 5, 0)
    state = State(StateVector([100, 0, 0, 5, 0, 0]))
    meas = doppler_model.function(state, noise=False)
    assert meas[2, 0] == approx(0, abs=1e-10)


def test_doppler_jacobian_shape(doppler_model):
    """Jacobian should have correct shape."""
    state = State(StateVector([100, 5, 200, 3, -50, -2]))
    jac = doppler_model.jacobian(state)
    assert jac.shape == (3, 6)


def test_doppler_jacobian_numerical(doppler_model):
    """Jacobian should match numerical approximation."""
    state = State(StateVector([100.0, 5.0, 200.0, 3.0, -50.0, -2.0]))
    jac = doppler_model.jacobian(state)

    eps = 1e-7
    num_jac = np.zeros((3, 6))
    for i in range(6):
        sv_plus = np.array(state.state_vector, dtype=float).copy()
        sv_plus[i, 0] += eps
        sv_minus = np.array(state.state_vector, dtype=float).copy()
        sv_minus[i, 0] -= eps
        meas_plus = np.array(
            doppler_model.function(State(StateVector(sv_plus)), noise=False), dtype=float
        )
        meas_minus = np.array(
            doppler_model.function(State(StateVector(sv_minus)), noise=False), dtype=float
        )
        diff = meas_plus - meas_minus
        # Handle angle wrapping for bearing
        while diff[0, 0] > np.pi:
            diff[0, 0] -= 2 * np.pi
        while diff[0, 0] < -np.pi:
            diff[0, 0] += 2 * np.pi
        num_jac[:, i] = (diff / (2 * eps)).ravel()

    np.testing.assert_allclose(jac, num_jac, atol=1e-5)


# =============================================================================
# Roundtrip Tests
# =============================================================================


@pytest.mark.parametrize(
    "x,y,z",
    [
        (100.0, 200.0, -50.0),
        (500.0, 0.0, -100.0),
        (0.0, 1000.0, -200.0),
        (-300.0, 400.0, -75.0),
        (1000.0, 2000.0, -500.0),
    ],
)
def test_depth_bearing_range_roundtrip(x, y, z):
    """Test roundtrip for depth-bearing-range model."""
    model = CartesianToDepthBearingRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=CovarianceMatrix(np.diag([1.0, 0.01, 1.0])),
    )
    state = State(StateVector([x, 0, y, 0, z, 0]))
    meas = model.function(state, noise=False)
    detection = Detection(meas)
    recovered = model.inverse_function(detection)

    assert recovered[0, 0] == approx(x, rel=1e-10, abs=1e-10)
    assert recovered[2, 0] == approx(y, rel=1e-10, abs=1e-10)
    assert recovered[4, 0] == approx(z, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(
    "x,y,z",
    [
        (100.0, 200.0, 50.0),
        (500.0, 0.0, -100.0),
        (0.0, 1000.0, 200.0),
        (-300.0, 400.0, -75.0),
    ],
)
def test_bearing_elevation_range_roundtrip(x, y, z):
    """Test roundtrip for bearing-elevation-range model."""
    model = CartesianToBearingElevationRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=CovarianceMatrix(np.diag([0.01, 0.01, 1.0])),
    )
    state = State(StateVector([x, 0, y, 0, z, 0]))
    meas = model.function(state, noise=False)
    detection = Detection(meas)
    recovered = model.inverse_function(detection)

    assert recovered[0, 0] == approx(x, rel=1e-10, abs=1e-10)
    assert recovered[2, 0] == approx(y, rel=1e-10, abs=1e-10)
    assert recovered[4, 0] == approx(z, rel=1e-10, abs=1e-10)


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.parametrize(
    "model_class",
    [
        CartesianToDepthBearingRange,
        CartesianToBearingOnly,
        CartesianToBearingElevationRange,
    ],
)
def test_none_covar_raises(model_class):
    """Models should raise error with None covariance."""
    with pytest.raises(ValueError, match="Covariance should have ndim of 2"):
        model_class(ndim_state=6, mapping=[0, 2, 4], noise_covar=None)
