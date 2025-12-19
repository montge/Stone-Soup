"""Comprehensive tests for nonlinear measurement models.

This test file provides extensive coverage for the nonlinear measurement models,
including edge cases, Jacobian calculations, and inverse functions.
"""

import numpy as np
import pytest
from pytest import approx

from ....types.angle import Bearing, Elevation
from ....types.array import StateVector
from ....types.state import State
from ..nonlinear import (
    Cartesian2DToBearing,
    CartesianToAzimuthElevationRange,
    CartesianToBearingRange,
    CartesianToBearingRangeRate,
    CartesianToBearingRangeRate2D,
    CartesianToElevationBearing,
    CartesianToElevationBearingRange,
    CartesianToElevationBearingRangeRate,
    CombinedReversibleGaussianMeasurementModel,
)


# CartesianToBearingRange Tests
def test_bearing_range_initialization():
    """Test CartesianToBearingRange model initialization."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2))
    assert model.ndim_state == 4
    assert model.ndim_meas == 2
    assert np.array_equal(model.mapping, [0, 2])


def test_bearing_range_with_offsets():
    """Test CartesianToBearingRange initialization with translation and rotation offsets."""
    translation_offset = StateVector([[10], [20]])
    rotation_offset = StateVector([[0], [0], [np.pi / 4]])

    model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2),
        translation_offset=translation_offset,
        rotation_offset=rotation_offset,
    )

    assert np.allclose(model.translation_offset, translation_offset)
    assert np.allclose(model.rotation_offset, rotation_offset)


def test_bearing_range_target_at_origin():
    """Test CartesianToBearingRange when target is at sensor origin."""
    model = CartesianToBearingRange(ndim_state=2, mapping=[0, 1], noise_covar=np.eye(2))

    state = State(StateVector([[0], [0]]))
    measurement = model.function(state, noise=False)

    # Range should be zero
    assert measurement[1, 0] == approx(0.0)


def test_bearing_range_target_on_positive_x_axis():
    """Test CartesianToBearingRange when target is on positive x-axis."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))

    state = State(StateVector([[10], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Bearing should be 0
    assert float(measurement[0, 0]) == approx(0.0)
    # Range should be 10
    assert measurement[1, 0] == approx(10.0)


def test_bearing_range_target_on_positive_y_axis():
    """Test CartesianToBearingRange when target is on positive y-axis."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))

    state = State(StateVector([[0], [0], [10], [0]]))
    measurement = model.function(state, noise=False)

    # Bearing should be π/2
    assert float(measurement[0, 0]) == approx(np.pi / 2)
    # Range should be 10
    assert measurement[1, 0] == approx(10.0)


def test_bearing_range_target_on_negative_x_axis():
    """Test CartesianToBearingRange when target is on negative x-axis."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))

    state = State(StateVector([[-10], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Bearing should be ±π
    assert abs(float(measurement[0, 0])) == approx(np.pi)
    # Range should be 10
    assert measurement[1, 0] == approx(10.0)


def test_bearing_range_inverse_function():
    """Test CartesianToBearingRange inverse function."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))

    original_state = StateVector([[10], [0], [5], [0]])
    state = State(original_state)

    # Forward then inverse
    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    # Check that position is recovered (velocities are not)
    assert recovered_state[0, 0] == approx(original_state[0, 0])
    assert recovered_state[2, 0] == approx(original_state[2, 0])


def test_bearing_range_inverse_function_with_rotation():
    """Test CartesianToBearingRange inverse function with rotation."""
    model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2),
        rotation_offset=StateVector([[0], [0], [np.pi / 4]]),
    )

    original_state = StateVector([[10], [0], [0], [0]])
    state = State(original_state)

    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    assert recovered_state[0, 0] == approx(original_state[0, 0])
    assert recovered_state[2, 0] == approx(original_state[2, 0])


def test_bearing_range_inverse_raises_with_3d_rotation():
    """Test that CartesianToBearingRange inverse function raises error with unsupported 3D rotation."""
    model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2),
        rotation_offset=StateVector([[0.5], [0], [0]]),  # x-rotation not supported
    )

    measurement = State(StateVector([[0.5], [10]]))

    with pytest.raises(RuntimeError, match="2D space"):
        model.inverse_function(measurement)


def test_bearing_range_with_translation_offset():
    """Test CartesianToBearingRange function with translation offset."""
    translation_offset = StateVector([[5], [3]])
    model = CartesianToBearingRange(
        ndim_state=2,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        translation_offset=translation_offset,
    )

    # Target at (8, 7) with sensor at (5, 3) means relative position is (3, 4)
    state = State(StateVector([[8], [7]]))
    measurement = model.function(state, noise=False)

    # Expected bearing: atan2(4, 3)
    expected_bearing = np.arctan2(4, 3)
    # Expected range: sqrt(3^2 + 4^2) = 5
    expected_range = 5.0

    assert float(measurement[0, 0]) == approx(expected_bearing)
    assert measurement[1, 0] == approx(expected_range)


def test_bearing_range_jacobian_numerical():
    """Test CartesianToBearingRange Jacobian is consistent with numerical differentiation."""
    model = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))

    state = State(StateVector([[10], [1], [5], [0.5]]))

    # Compute Jacobian using the model
    jac = model.jacobian(state)

    # Compute numerical Jacobian
    epsilon = 1e-6
    numerical_jac = np.zeros((2, 4))

    for i in range(4):
        state_plus = state.state_vector.copy()
        state_minus = state.state_vector.copy()
        state_plus[i] += epsilon
        state_minus[i] -= epsilon

        meas_plus = model.function(State(state_plus), noise=False)
        meas_minus = model.function(State(state_minus), noise=False)

        # Handle angle wrapping for bearing
        diff = meas_plus - meas_minus
        if i in [0, 2]:  # positions affect bearing
            # Unwrap bearing difference
            diff[0, 0] = float(Bearing(diff[0, 0]))

        numerical_jac[:, i] = (diff / (2 * epsilon)).ravel()

    assert np.allclose(jac, numerical_jac, atol=1e-5)


# CartesianToElevationBearingRange Tests
def test_elevation_bearing_range_initialization():
    """Test CartesianToElevationBearingRange model initialization."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(3)
    )
    assert model.ndim_state == 6
    assert model.ndim_meas == 3
    assert np.array_equal(model.mapping, [0, 2, 4])


def test_elevation_bearing_range_target_on_x_axis():
    """Test CartesianToElevationBearingRange when target is on positive x-axis."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.1, 0.5])
    )

    state = State(StateVector([[10], [0], [0], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Elevation should be 0
    assert float(measurement[0, 0]) == approx(0.0)
    # Bearing should be 0
    assert float(measurement[1, 0]) == approx(0.0)
    # Range should be 10
    assert measurement[2, 0] == approx(10.0)


def test_elevation_bearing_range_target_on_z_axis():
    """Test CartesianToElevationBearingRange when target is on positive z-axis."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.1, 0.5])
    )

    state = State(StateVector([[0], [0], [0], [0], [10], [0]]))
    measurement = model.function(state, noise=False)

    # Elevation should be π/2
    assert float(measurement[0, 0]) == approx(np.pi / 2)
    # Range should be 10
    assert measurement[2, 0] == approx(10.0)


def test_elevation_bearing_range_45_degrees():
    """Test CartesianToElevationBearingRange when target is at 45 degrees elevation."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.1, 0.5])
    )

    # Target at (sqrt(2), 0, sqrt(2)) has 45 degree elevation
    x = np.sqrt(2)
    z = np.sqrt(2)
    state = State(StateVector([[x], [0], [0], [0], [z], [0]]))
    measurement = model.function(state, noise=False)

    # Elevation should be π/4
    assert float(measurement[0, 0]) == approx(np.pi / 4, abs=1e-6)
    # Bearing should be 0
    assert float(measurement[1, 0]) == approx(0.0)
    # Range should be 2
    assert measurement[2, 0] == approx(2.0)


def test_elevation_bearing_range_inverse_function():
    """Test CartesianToElevationBearingRange inverse function."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.1, 0.5])
    )

    original_state = StateVector([[10], [0], [5], [0], [3], [0]])
    state = State(original_state)

    # Forward then inverse
    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    # Check that position is recovered
    assert recovered_state[0, 0] == approx(original_state[0, 0], abs=1e-10)
    assert recovered_state[2, 0] == approx(original_state[2, 0], abs=1e-10)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-10)


def test_elevation_bearing_range_inverse_with_rotation():
    """Test CartesianToElevationBearingRange inverse function with rotation."""
    model = CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=np.eye(3),
        rotation_offset=StateVector([[0.1], [0.2], [0.3]]),
    )

    original_state = StateVector([[10], [0], [5], [0], [3], [0]])
    state = State(original_state)

    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    assert recovered_state[0, 0] == approx(original_state[0, 0], abs=1e-10)
    assert recovered_state[2, 0] == approx(original_state[2, 0], abs=1e-10)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-10)


def test_elevation_bearing_range_inverse_with_translation():
    """Test CartesianToElevationBearingRange inverse function with translation offset."""
    translation_offset = StateVector([[100], [50], [25]])
    model = CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=np.eye(3),
        translation_offset=translation_offset,
    )

    original_state = StateVector([[110], [0], [55], [0], [28], [0]])
    state = State(original_state)

    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    assert recovered_state[0, 0] == approx(original_state[0, 0], abs=1e-10)
    assert recovered_state[2, 0] == approx(original_state[2, 0], abs=1e-10)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-10)


def test_elevation_bearing_range_jacobian_numerical():
    """Test CartesianToElevationBearingRange Jacobian with numerical differentiation."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.1, 0.5])
    )

    state = State(StateVector([[10], [1], [5], [0.5], [3], [0.2]]))

    # Compute Jacobian using the model
    jac = model.jacobian(state)

    # Compute numerical Jacobian
    epsilon = 1e-6
    numerical_jac = np.zeros((3, 6))

    for i in range(6):
        state_plus = state.state_vector.copy()
        state_minus = state.state_vector.copy()
        state_plus[i] += epsilon
        state_minus[i] -= epsilon

        meas_plus = model.function(State(state_plus), noise=False)
        meas_minus = model.function(State(state_minus), noise=False)

        # Handle angle wrapping
        diff = meas_plus - meas_minus
        if i in [0, 2, 4]:  # positions affect angles
            diff[0, 0] = float(Elevation(diff[0, 0]))
            diff[1, 0] = float(Bearing(diff[1, 0]))

        numerical_jac[:, i] = (diff / (2 * epsilon)).ravel()

    assert np.allclose(jac, numerical_jac, atol=1e-5)


# CartesianToBearingRangeRate Tests
def test_bearing_range_rate_initialization():
    """Test CartesianToBearingRangeRate model initialization."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.eye(3),
    )
    assert model.ndim_state == 6
    assert model.ndim_meas == 3
    assert np.array_equal(model.mapping, [0, 2, 4])
    assert np.array_equal(model.velocity_mapping, [1, 3, 5])


def test_bearing_range_rate_stationary_target():
    """Test CartesianToBearingRangeRate measurement of stationary target with stationary sensor."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    state = State(StateVector([[10], [0], [5], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be 0 for stationary target
    assert measurement[2, 0] == approx(0.0)


def test_bearing_range_rate_approaching_target():
    """Test CartesianToBearingRangeRate measurement of approaching target."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    # Target at (10, 0, 0) moving towards origin at -5 m/s in x-direction
    state = State(StateVector([[10], [-5], [0], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be negative (approaching)
    assert measurement[2, 0] < 0


def test_bearing_range_rate_receding_target():
    """Test CartesianToBearingRangeRate measurement of receding target."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    # Target at (10, 0, 0) moving away at 5 m/s in x-direction
    state = State(StateVector([[10], [5], [0], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be positive (receding)
    assert measurement[2, 0] > 0


def test_bearing_range_rate_tangential_motion():
    """Test CartesianToBearingRangeRate measurement of target with tangential motion."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    # Target at (10, 0, 0) moving in y-direction
    state = State(StateVector([[10], [0], [0], [5], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be approximately 0 for pure tangential motion
    assert measurement[2, 0] == approx(0.0)


def test_bearing_range_rate_sensor_velocity():
    """Test CartesianToBearingRangeRate measurement with moving sensor."""
    model = CartesianToBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.eye(3),
        velocity=StateVector([[5], [0], [0]]),  # Sensor moving at 5 m/s in x
    )

    # Stationary target at (10, 0, 0)
    state = State(StateVector([[10], [0], [0], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Relative motion: sensor approaching target, so range rate should be negative
    assert measurement[2, 0] < 0


# CartesianToElevationBearingRangeRate Tests
def test_elevation_bearing_range_rate_initialization():
    """Test CartesianToElevationBearingRangeRate model initialization."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.eye(4),
    )
    assert model.ndim_state == 6
    assert model.ndim_meas == 4


def test_elevation_bearing_range_rate_stationary():
    """Test CartesianToElevationBearingRangeRate measurement of stationary target."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.05, 0.1, 0.5, 1.0]),
    )

    state = State(StateVector([[10], [0], [5], [0], [3], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be 0
    assert measurement[3, 0] == approx(0.0)


def test_elevation_bearing_range_rate_approaching():
    """Test CartesianToElevationBearingRangeRate measurement of approaching target."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.05, 0.1, 0.5, 1.0]),
    )

    # Target at (10, 5, 3) moving towards origin
    state = State(StateVector([[10], [-5], [5], [-2.5], [3], [-1.5]]))
    measurement = model.function(state, noise=False)

    # Range rate should be negative (approaching)
    assert measurement[3, 0] < 0


def test_elevation_bearing_range_rate_inverse_function():
    """Test CartesianToElevationBearingRangeRate inverse function."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.05, 0.1, 0.5, 1.0]),
    )

    original_state = StateVector([[10], [2], [5], [1], [3], [0.5]])
    state = State(original_state)

    # Forward then inverse
    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    # Check that position is recovered
    assert recovered_state[0, 0] == approx(original_state[0, 0], abs=1e-10)
    assert recovered_state[2, 0] == approx(original_state[2, 0], abs=1e-10)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-10)


def test_elevation_bearing_range_rate_jacobian_analytic():
    """Test CartesianToElevationBearingRangeRate analytic Jacobian matches numerical."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.diag([0.05, 0.1, 0.5, 1.0]),
    )

    state = State(StateVector([[10], [2], [5], [1], [3], [0.5]]))

    # Analytic Jacobian
    jac_analytic = model.jacobian(state)

    # Numerical Jacobian
    epsilon = 1e-8
    numerical_jac = np.zeros((4, 6))

    for i in range(6):
        state_plus = state.state_vector.copy()
        state_minus = state.state_vector.copy()
        state_plus[i] += epsilon
        state_minus[i] -= epsilon

        meas_plus = model.function(State(state_plus), noise=False)
        meas_minus = model.function(State(state_minus), noise=False)

        # Handle angle wrapping
        diff = meas_plus - meas_minus
        diff[0, 0] = float(Elevation(diff[0, 0]))
        diff[1, 0] = float(Bearing(diff[1, 0]))

        numerical_jac[:, i] = (diff / (2 * epsilon)).ravel()

    assert np.allclose(jac_analytic, numerical_jac, atol=1e-4, rtol=1e-5)


def test_elevation_bearing_range_rate_jacobian_with_rotation():
    """Test CartesianToElevationBearingRangeRate Jacobian with rotation offset."""
    model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.eye(4),
        rotation_offset=StateVector([[0.1], [0.2], [0.3]]),
    )

    state = State(StateVector([[10], [2], [5], [1], [3], [0.5]]))

    # Analytic Jacobian
    jac_analytic = model.jacobian(state)

    # Numerical Jacobian
    epsilon = 1e-8
    numerical_jac = np.zeros((4, 6))

    for i in range(6):
        state_plus = state.state_vector.copy()
        state_minus = state.state_vector.copy()
        state_plus[i] += epsilon
        state_minus[i] -= epsilon

        meas_plus = model.function(State(state_plus), noise=False)
        meas_minus = model.function(State(state_minus), noise=False)

        diff = meas_plus - meas_minus
        diff[0, 0] = float(Elevation(diff[0, 0]))
        diff[1, 0] = float(Bearing(diff[1, 0]))

        numerical_jac[:, i] = (diff / (2 * epsilon)).ravel()

    assert np.allclose(jac_analytic, numerical_jac, atol=1e-4, rtol=1e-5)


# CartesianToAzimuthElevationRange Tests
def test_azimuth_elevation_range_initialization():
    """Test CartesianToAzimuthElevationRange model initialization."""
    model = CartesianToAzimuthElevationRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(3)
    )
    assert model.ndim_state == 6
    assert model.ndim_meas == 3


def test_azimuth_elevation_range_target_on_z_axis():
    """Test CartesianToAzimuthElevationRange when target is on positive z-axis (broadside)."""
    model = CartesianToAzimuthElevationRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.05, 0.5])
    )

    state = State(StateVector([[0], [0], [0], [0], [10], [0]]))
    measurement = model.function(state, noise=False)

    # Azimuth should be 0
    assert float(measurement[0, 0]) == approx(0.0)
    # Elevation should be 0
    assert float(measurement[1, 0]) == approx(0.0)
    # Range should be 10
    assert measurement[2, 0] == approx(10.0)


def test_azimuth_elevation_range_inverse_function():
    """Test CartesianToAzimuthElevationRange inverse function."""
    model = CartesianToAzimuthElevationRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.05, 0.5])
    )

    original_state = StateVector([[5], [0], [5], [0], [10], [0]])
    state = State(original_state)

    # Forward then inverse
    measurement = model.function(state, noise=False)
    recovered_state = model.inverse_function(State(measurement))

    # Check that position is recovered
    assert recovered_state[0, 0] == approx(original_state[0, 0], abs=1e-10)
    assert recovered_state[2, 0] == approx(original_state[2, 0], abs=1e-10)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-10)


# Cartesian2DToBearing Tests
def test_2d_bearing_initialization():
    """Test Cartesian2DToBearing model initialization."""
    model = Cartesian2DToBearing(ndim_state=4, mapping=[0, 2], noise_covar=np.array([[0.1]]))
    assert model.ndim_state == 4
    assert model.ndim_meas == 1


def test_2d_bearing_target_on_x_axis():
    """Test Cartesian2DToBearing when target is on positive x-axis."""
    model = Cartesian2DToBearing(ndim_state=4, mapping=[0, 2], noise_covar=np.array([[0.1]]))

    state = State(StateVector([[10], [0], [0], [0]]))
    measurement = model.function(state, noise=False)

    # Bearing should be 0
    assert float(measurement[0, 0]) == approx(0.0)


def test_2d_bearing_target_on_y_axis():
    """Test Cartesian2DToBearing when target is on positive y-axis."""
    model = Cartesian2DToBearing(ndim_state=4, mapping=[0, 2], noise_covar=np.array([[0.1]]))

    state = State(StateVector([[0], [0], [10], [0]]))
    measurement = model.function(state, noise=False)

    # Bearing should be π/2
    assert float(measurement[0, 0]) == approx(np.pi / 2)


# CartesianToElevationBearing Tests
def test_elevation_bearing_initialization():
    """Test CartesianToElevationBearing model initialization."""
    model = CartesianToElevationBearing(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(2))
    assert model.ndim_state == 6
    assert model.ndim_meas == 2


# CartesianToBearingRangeRate2D Tests
def test_bearing_range_rate_2d_initialization():
    """Test CartesianToBearingRangeRate2D model initialization."""
    model = CartesianToBearingRangeRate2D(
        ndim_state=4,
        mapping=[0, 2],
        velocity_mapping=[1, 3],
        noise_covar=np.eye(3),
    )
    assert model.ndim_state == 4
    assert model.ndim_meas == 3
    assert np.array_equal(model.mapping, [0, 2])
    assert np.array_equal(model.velocity_mapping, [1, 3])


def test_bearing_range_rate_2d_stationary():
    """Test CartesianToBearingRangeRate2D measurement of stationary target."""
    model = CartesianToBearingRangeRate2D(
        ndim_state=4,
        mapping=[0, 2],
        velocity_mapping=[1, 3],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    state = State(StateVector([[10], [0], [5], [0]]))
    measurement = model.function(state, noise=False)

    # Range rate should be 0
    assert measurement[2, 0] == approx(0.0)


def test_bearing_range_rate_2d_approaching():
    """Test CartesianToBearingRangeRate2D measurement of approaching target."""
    model = CartesianToBearingRangeRate2D(
        ndim_state=4,
        mapping=[0, 2],
        velocity_mapping=[1, 3],
        noise_covar=np.diag([0.1, 0.5, 1.0]),
    )

    # Target at (10, 5) moving towards origin
    state = State(StateVector([[10], [-5], [5], [-2.5]]))
    measurement = model.function(state, noise=False)

    # Range rate should be negative (approaching)
    assert measurement[2, 0] < 0


# CombinedReversibleGaussianMeasurementModel Tests
def test_combined_model_incompatible_ndim():
    """Test CombinedReversibleGaussianMeasurementModel with incompatible models."""
    model1 = CartesianToBearingRange(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2))
    model2 = CartesianToElevationBearing(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(2))

    # This should raise an error because models have different ndim_state
    with pytest.raises(ValueError, match="ndim_state"):
        CombinedReversibleGaussianMeasurementModel(model_list=[model1, model2])


def test_combined_model_function():
    """Test CombinedReversibleGaussianMeasurementModel function."""
    # Create two compatible models
    model1 = CartesianToBearingRange(ndim_state=6, mapping=[0, 2], noise_covar=np.eye(2))
    model2 = CartesianToElevationBearing(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(2))

    combined = CombinedReversibleGaussianMeasurementModel(model_list=[model1, model2])

    state = State(StateVector([[10], [0], [5], [0], [3], [0]]))
    measurement = combined.function(state, noise=False)

    # Should have 4 measurements: bearing, range, elevation, bearing
    assert measurement.shape == (4, 1)


def test_combined_model_covariance():
    """Test CombinedReversibleGaussianMeasurementModel covariance."""
    model1 = CartesianToBearingRange(ndim_state=6, mapping=[0, 2], noise_covar=np.diag([0.1, 0.5]))
    model2 = CartesianToElevationBearing(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.05, 0.15])
    )

    combined = CombinedReversibleGaussianMeasurementModel(model_list=[model1, model2])

    covar = combined.covar()

    # Should be block diagonal
    expected_covar = np.diag([0.1, 0.5, 0.05, 0.15])
    assert np.allclose(covar, expected_covar)


def test_combined_model_inverse_function():
    """Test CombinedReversibleGaussianMeasurementModel inverse function."""
    model1 = CartesianToBearingRange(ndim_state=6, mapping=[0, 2], noise_covar=np.eye(2))
    model2 = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(3)
    )

    combined = CombinedReversibleGaussianMeasurementModel(model_list=[model1, model2])

    original_state = StateVector([[10], [0], [5], [0], [3], [0]])
    state = State(original_state)

    measurement = combined.function(state, noise=False)
    recovered_state = combined.inverse_function(State(measurement))

    # Note: Combined model sums the inverse functions
    # model1 uses mapping [0, 2], model2 uses mapping [0, 2, 4]
    # So positions [0] and [2] are doubled (both models contribute)
    # but position [4] is only from model2
    assert recovered_state[0, 0] == approx(2 * original_state[0, 0], abs=1e-9)
    assert recovered_state[2, 0] == approx(2 * original_state[2, 0], abs=1e-9)
    assert recovered_state[4, 0] == approx(original_state[4, 0], abs=1e-9)


# Edge Cases and Boundary Conditions
def test_very_small_range():
    """Test behavior with very small range."""
    model = CartesianToBearingRange(ndim_state=2, mapping=[0, 1], noise_covar=np.eye(2))

    # Target very close to origin
    state = State(StateVector([[1e-10], [1e-10]]))
    measurement = model.function(state, noise=False)

    # Range should be very small
    assert measurement[1, 0] < 1e-9


def test_very_large_range():
    """Test behavior with very large range."""
    model = CartesianToBearingRange(ndim_state=2, mapping=[0, 1], noise_covar=np.eye(2))

    # Target very far away
    state = State(StateVector([[1e10], [1e10]]))
    measurement = model.function(state, noise=False)

    # Range should be very large
    assert measurement[1, 0] > 1e10


def test_negative_coordinates():
    """Test behavior with negative coordinates."""
    model = CartesianToElevationBearingRange(
        ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(3)
    )

    state = State(StateVector([[-10], [0], [-5], [0], [-3], [0]]))
    measurement = model.function(state, noise=False)

    # Should produce valid measurements
    assert measurement.shape == (3, 1)
    assert measurement[2, 0] > 0  # Range should always be positive


def test_angle_wrapping():
    """Test angle wrapping in measurements."""
    model = CartesianToBearingRange(ndim_state=2, mapping=[0, 1], noise_covar=np.eye(2))

    # Target at various angles
    for angle in [0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2]:
        x = 10 * np.cos(angle)
        y = 10 * np.sin(angle)
        state = State(StateVector([[x], [y]]))
        measurement = model.function(state, noise=False)

        # Bearing should be within [-π, π]
        bearing = float(measurement[0, 0])
        assert -np.pi <= bearing <= np.pi
