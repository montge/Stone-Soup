# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Tests for stonesoup-core Python bindings.

These tests verify the PyO3-based Rust bindings for Stone Soup.
"""

import numpy as np
import pytest

# Try to import the core module - skip all tests if not built
try:
    from stonesoup_core import (
        CovarianceMatrix,
        Detection,
        GaussianState,
        StateVector,
        Track,
        initialize,
        kalman_predict,
        kalman_update,
    )

    HAS_CORE = True
except ImportError:
    HAS_CORE = False

pytestmark = pytest.mark.skipif(not HAS_CORE, reason="stonesoup_core not built")


class TestStateVector:
    """Tests for StateVector class."""

    def test_create_from_list(self):
        """Test creating state vector from Python list."""
        sv = StateVector([1.0, 2.0, 3.0])
        assert sv.dim == 3
        assert len(sv) == 3

    def test_create_zeros(self):
        """Test creating zero state vector."""
        sv = StateVector.zeros(4)
        assert sv.dim == 4
        assert sv[0] == 0.0
        assert sv[3] == 0.0

    def test_from_numpy(self):
        """Test creating state vector from numpy array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        sv = StateVector.from_numpy(arr)
        assert sv.dim == 4
        assert sv[0] == 1.0
        assert sv[3] == 4.0

    def test_indexing(self):
        """Test getting and setting elements."""
        sv = StateVector([1.0, 2.0, 3.0])
        assert sv[0] == 1.0
        assert sv[2] == 3.0

        sv[1] = 5.0
        assert sv[1] == 5.0

    def test_index_out_of_bounds(self):
        """Test that out-of-bounds indexing raises error."""
        sv = StateVector([1.0, 2.0])
        with pytest.raises(ValueError):
            _ = sv[5]

    def test_to_list(self):
        """Test converting to Python list."""
        sv = StateVector([1.0, 2.0, 3.0])
        lst = sv.to_list()
        assert lst == [1.0, 2.0, 3.0]

    def test_to_numpy(self):
        """Test converting to numpy array."""
        sv = StateVector([1.0, 2.0, 3.0])
        arr = sv.to_numpy()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_norm(self):
        """Test Euclidean norm computation."""
        sv = StateVector([3.0, 4.0])
        assert sv.norm() == 5.0

    def test_addition(self):
        """Test state vector addition."""
        sv1 = StateVector([1.0, 2.0])
        sv2 = StateVector([3.0, 4.0])
        result = sv1 + sv2
        assert result.to_list() == [4.0, 6.0]

    def test_addition_dimension_mismatch(self):
        """Test that mismatched dimensions raise error."""
        sv1 = StateVector([1.0, 2.0])
        sv2 = StateVector([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            _ = sv1 + sv2

    def test_subtraction(self):
        """Test state vector subtraction."""
        sv1 = StateVector([3.0, 4.0])
        sv2 = StateVector([1.0, 1.0])
        result = sv1 - sv2
        assert result.to_list() == [2.0, 3.0]

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        sv = StateVector([1.0, 2.0])
        result = sv * 2.0
        assert result.to_list() == [2.0, 4.0]

    def test_repr(self):
        """Test string representation."""
        sv = StateVector([1.0, 2.0])
        assert "StateVector" in repr(sv)
        assert "dims=2" in repr(sv)


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_create_from_list(self):
        """Test creating covariance matrix from nested list."""
        cov = CovarianceMatrix([[1.0, 0.0], [0.0, 1.0]])
        assert cov.dim == 2
        assert cov.rows == 2
        assert cov.cols == 2

    def test_create_identity(self):
        """Test creating identity matrix."""
        cov = CovarianceMatrix.identity(3)
        assert cov.dim == 3
        assert cov[(0, 0)] == 1.0
        assert cov[(1, 1)] == 1.0
        assert cov[(0, 1)] == 0.0

    def test_create_diagonal(self):
        """Test creating diagonal matrix."""
        cov = CovarianceMatrix.diagonal([1.0, 2.0, 3.0])
        assert cov.dim == 3
        assert cov[(0, 0)] == 1.0
        assert cov[(1, 1)] == 2.0
        assert cov[(2, 2)] == 3.0
        assert cov[(0, 1)] == 0.0

    def test_create_zeros(self):
        """Test creating zero matrix."""
        cov = CovarianceMatrix.zeros(3)
        assert cov.dim == 3
        assert cov[(0, 0)] == 0.0

    def test_from_numpy(self):
        """Test creating from numpy array."""
        arr = np.eye(3)
        cov = CovarianceMatrix.from_numpy(arr)
        assert cov.dim == 3
        assert cov[(0, 0)] == 1.0

    def test_non_square_rejected(self):
        """Test that non-square matrices are rejected."""
        arr = np.ones((2, 3))
        with pytest.raises(ValueError):
            CovarianceMatrix.from_numpy(arr)

    def test_indexing(self):
        """Test getting and setting elements."""
        cov = CovarianceMatrix.identity(2)
        assert cov[(0, 0)] == 1.0

        cov[(0, 1)] = 0.5
        assert cov[(0, 1)] == 0.5

    def test_to_list(self):
        """Test converting to nested Python list."""
        cov = CovarianceMatrix.identity(2)
        lst = cov.to_list()
        assert lst == [[1.0, 0.0], [0.0, 1.0]]

    def test_to_numpy(self):
        """Test converting to numpy array."""
        cov = CovarianceMatrix.identity(2)
        arr = cov.to_numpy()
        np.testing.assert_array_equal(arr, np.eye(2))

    def test_trace(self):
        """Test trace computation."""
        cov = CovarianceMatrix.diagonal([1.0, 2.0, 3.0])
        assert cov.trace() == 6.0

    def test_determinant_1x1(self):
        """Test determinant for 1x1 matrix."""
        cov = CovarianceMatrix([[5.0]])
        assert cov.determinant() == 5.0

    def test_determinant_2x2(self):
        """Test determinant for 2x2 matrix."""
        cov = CovarianceMatrix([[1.0, 2.0], [3.0, 4.0]])
        # det = 1*4 - 2*3 = -2
        assert cov.determinant() == -2.0

    def test_addition(self):
        """Test matrix addition."""
        cov1 = CovarianceMatrix.identity(2)
        cov2 = CovarianceMatrix.identity(2)
        result = cov1 + cov2
        assert result[(0, 0)] == 2.0

    def test_subtraction(self):
        """Test matrix subtraction."""
        cov1 = CovarianceMatrix.diagonal([2.0, 2.0])
        cov2 = CovarianceMatrix.identity(2)
        result = cov1 - cov2
        assert result[(0, 0)] == 1.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        cov = CovarianceMatrix.identity(2)
        result = cov * 3.0
        assert result[(0, 0)] == 3.0

    def test_repr(self):
        """Test string representation."""
        cov = CovarianceMatrix.identity(3)
        assert "CovarianceMatrix" in repr(cov)
        assert "3x3" in repr(cov)


class TestGaussianState:
    """Tests for GaussianState class."""

    def test_create(self):
        """Test creating Gaussian state."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov)

        assert state.dim == 2
        assert state.timestamp is None

    def test_create_with_timestamp(self):
        """Test creating Gaussian state with timestamp."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov, timestamp=1.5)

        assert state.timestamp == 1.5

    def test_dimension_mismatch_rejected(self):
        """Test that mismatched dimensions are rejected."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(3)

        with pytest.raises(ValueError):
            GaussianState(sv, cov)

    def test_from_numpy(self):
        """Test creating from numpy arrays."""
        state_arr = np.array([1.0, 2.0, 3.0, 4.0])
        covar_arr = np.eye(4)

        state = GaussianState.from_numpy(state_arr, covar_arr, timestamp=0.0)
        assert state.dim == 4
        assert state.timestamp == 0.0

    def test_state_numpy(self):
        """Test getting state as numpy array."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov)

        arr = state.state_numpy()
        np.testing.assert_array_equal(arr, [1.0, 2.0])

    def test_covar_numpy(self):
        """Test getting covariance as numpy array."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov)

        arr = state.covar_numpy()
        np.testing.assert_array_equal(arr, np.eye(2))

    def test_repr(self):
        """Test string representation."""
        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov)

        assert "GaussianState" in repr(state)
        assert "dims=2" in repr(state)


class TestDetection:
    """Tests for Detection class."""

    def test_create(self):
        """Test creating detection."""
        det = Detection([1.0, 2.0], 0.5)
        assert det.timestamp == 0.5

    def test_from_numpy(self):
        """Test creating from numpy array."""
        arr = np.array([1.0, 2.0])
        det = Detection.from_numpy(arr, 1.0)
        assert det.timestamp == 1.0

    def test_measurement(self):
        """Test getting measurement."""
        det = Detection([3.0, 4.0], 0.0)
        meas = det.measurement
        assert meas.dim == 2

    def test_measurement_numpy(self):
        """Test getting measurement as numpy."""
        det = Detection([3.0, 4.0], 0.0)
        arr = det.measurement_numpy()
        np.testing.assert_array_equal(arr, [3.0, 4.0])

    def test_repr(self):
        """Test string representation."""
        det = Detection([1.0, 2.0], 0.5)
        assert "Detection" in repr(det)


class TestTrack:
    """Tests for Track class."""

    def test_create(self):
        """Test creating track."""
        track = Track("track-1")
        assert track.id == "track-1"
        assert len(track) == 0

    def test_append(self):
        """Test appending states."""
        track = Track("track-1")

        sv = StateVector([1.0, 2.0])
        cov = CovarianceMatrix.identity(2)
        state = GaussianState(sv, cov, timestamp=0.0)

        track.append(state)
        assert len(track) == 1

    def test_indexing(self):
        """Test getting states by index."""
        track = Track("track-1")

        sv1 = StateVector([1.0, 2.0])
        sv2 = StateVector([3.0, 4.0])
        cov = CovarianceMatrix.identity(2)

        track.append(GaussianState(sv1, cov, timestamp=0.0))
        track.append(GaussianState(sv2, cov, timestamp=1.0))

        state0 = track[0]
        assert state0.timestamp == 0.0

        state1 = track[1]
        assert state1.timestamp == 1.0

    def test_latest(self):
        """Test getting latest state."""
        track = Track("track-1")

        sv1 = StateVector([1.0, 2.0])
        sv2 = StateVector([3.0, 4.0])
        cov = CovarianceMatrix.identity(2)

        track.append(GaussianState(sv1, cov, timestamp=0.0))
        track.append(GaussianState(sv2, cov, timestamp=1.0))

        latest = track.latest()
        assert latest.timestamp == 1.0

    def test_latest_empty_raises(self):
        """Test that latest() on empty track raises error."""
        track = Track("track-1")
        with pytest.raises(RuntimeError):
            track.latest()

    def test_repr(self):
        """Test string representation."""
        track = Track("track-1")
        assert "Track" in repr(track)
        assert "track-1" in repr(track)


class TestKalmanFilter:
    """Tests for Kalman filter functions."""

    def test_initialize(self):
        """Test initialize function."""
        # Should not raise
        initialize()

    def test_kalman_predict_1d(self):
        """Test Kalman predict with 1D state."""
        sv = StateVector([1.0])
        cov = CovarianceMatrix([[1.0]])
        prior = GaussianState(sv, cov, timestamp=0.0)

        F = [[1.0]]  # Identity transition
        Q = CovarianceMatrix([[0.1]])

        predicted = kalman_predict(prior, F, Q)
        assert predicted.dim == 1

    def test_kalman_predict_2d_cv(self):
        """Test Kalman predict with constant velocity model."""
        # State: [x, vx]
        sv = StateVector([0.0, 1.0])  # x=0, vx=1
        cov = CovarianceMatrix.identity(2)
        prior = GaussianState(sv, cov, timestamp=0.0)

        # Constant velocity transition, dt=1
        F = [[1.0, 1.0], [0.0, 1.0]]
        Q = CovarianceMatrix.diagonal([0.01, 0.01])

        predicted = kalman_predict(prior, F, Q)

        # After dt=1 with vx=1, x should be ~1
        state_arr = predicted.state_numpy()
        assert abs(state_arr[0] - 1.0) < 0.01
        assert abs(state_arr[1] - 1.0) < 0.01

    def test_kalman_predict_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        sv = StateVector([0.0, 1.0])
        cov = CovarianceMatrix.identity(2)
        prior = GaussianState(sv, cov)

        F = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # 3x3
        Q = CovarianceMatrix.identity(2)

        with pytest.raises(ValueError):
            kalman_predict(prior, F, Q)

    def test_kalman_update_1d(self):
        """Test Kalman update with 1D measurement."""
        sv = StateVector([1.0])
        cov = CovarianceMatrix([[1.0]])
        predicted = GaussianState(sv, cov)

        z = StateVector([1.1])
        H = [[1.0]]
        R = CovarianceMatrix([[0.1]])

        posterior = kalman_update(predicted, z, H, R)

        # Updated state should be between prediction and measurement
        state_arr = posterior.state_numpy()
        assert 1.0 <= state_arr[0] <= 1.1

    def test_kalman_update_2d(self):
        """Test Kalman update with 2D measurement."""
        # State: [x, vx, y, vy]
        sv = StateVector([0.0, 1.0, 0.0, 1.0])
        cov = CovarianceMatrix.identity(4)
        predicted = GaussianState(sv, cov)

        # Measure position only
        z = StateVector([0.5, 0.5])
        H = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        R = CovarianceMatrix.diagonal([0.1, 0.1])

        posterior = kalman_update(predicted, z, H, R)
        assert posterior.dim == 4

    def test_kalman_complete_cycle(self):
        """Test complete predict-update cycle."""
        # Initial state: x=0, vx=1
        sv = StateVector([0.0, 1.0])
        cov = CovarianceMatrix.diagonal([1.0, 0.1])
        state = GaussianState(sv, cov, timestamp=0.0)

        # Transition (constant velocity, dt=1)
        F = [[1.0, 1.0], [0.0, 1.0]]
        Q = CovarianceMatrix.diagonal([0.01, 0.01])

        # Measurement model (observe position only)
        H = [[1.0, 0.0]]
        R = CovarianceMatrix([[0.1]])

        # Predict
        predicted = kalman_predict(state, F, Q)

        # Update with measurement
        z = StateVector([1.2])
        posterior = kalman_update(predicted, z, H, R)

        # Final position should be close to measurement
        final_state = posterior.state_numpy()
        assert 0.8 <= final_state[0] <= 1.4


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_tracking_workflow(self):
        """Test complete tracking workflow."""
        # Create track
        track = Track("target-1")

        # Initial state
        state = GaussianState(
            StateVector([0.0, 1.0, 0.0, 1.0]), CovarianceMatrix.identity(4), timestamp=0.0
        )
        track.append(state)

        # Kalman filter matrices
        dt = 1.0
        F = [[1.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]]
        Q = CovarianceMatrix.diagonal([0.01, 0.1, 0.01, 0.1])
        H = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        R = CovarianceMatrix.diagonal([0.5, 0.5])

        # Process measurements
        measurements = [
            ([1.0, 1.0], 1.0),
            ([2.0, 2.0], 2.0),
            ([3.0, 3.0], 3.0),
        ]

        for meas, timestamp in measurements:
            # Predict
            state = kalman_predict(state, F, Q)

            # Update
            z = StateVector(meas)
            state = kalman_update(state, z, H, R)

            # Store in track
            state.timestamp = timestamp
            track.append(state)

        # Verify track
        assert len(track) == 4
        final = track.latest()
        final_state = final.state_numpy()

        # Final position should be close to (3, 3)
        assert 2.5 <= final_state[0] <= 3.5
        assert 2.5 <= final_state[2] <= 3.5
