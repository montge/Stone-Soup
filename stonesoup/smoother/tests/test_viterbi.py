"""Tests for Viterbi Smoother"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from stonesoup.smoother.viterbi import ViterbiSmoother
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState, State
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity
)
from stonesoup.models.measurement.linear import LinearGaussian


@pytest.fixture
def transition_model():
    """Create a simple transition model for testing."""
    return CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.1), ConstantVelocity(0.1)]
    )


@pytest.fixture
def measurement_model():
    """Create a simple measurement model for testing."""
    return LinearGaussian(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2) * 0.5
    )


@pytest.fixture
def simple_track(measurement_model):
    """Create a simple track with GaussianStateUpdate objects."""
    start = datetime.now()
    track = Track()

    # Create some simple states
    times = [start + timedelta(seconds=i) for i in range(5)]
    state_vectors = [
        np.array([[1.0], [1.0], [2.0], [1.5]]),
        np.array([[2.0], [1.0], [3.5], [1.5]]),
        np.array([[3.0], [1.0], [5.0], [1.5]]),
        np.array([[4.0], [1.0], [6.5], [1.5]]),
        np.array([[5.0], [1.0], [8.0], [1.5]])
    ]

    for state_vec, timestamp in zip(state_vectors, times):
        covar = np.eye(4) * 0.1
        detection = Detection(state_vec[[0, 2]], timestamp=timestamp)
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    return track


def test_viterbi_smoother_instantiation(transition_model, measurement_model):
    """Test that ViterbiSmoother can be instantiated with required models."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    assert smoother.transition_model is transition_model
    assert smoother.measurement_model is measurement_model
    assert smoother.num_states == 100  # Default value
    assert smoother.state_bounds is None  # Default value


def test_viterbi_smoother_instantiation_with_params(transition_model, measurement_model):
    """Test ViterbiSmoother instantiation with custom parameters."""
    bounds = [(0.0, 10.0), (-1.0, 2.0), (0.0, 10.0), (-1.0, 2.0)]

    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=50,
        state_bounds=bounds
    )

    assert smoother.num_states == 50
    assert smoother.state_bounds == bounds


def test_viterbi_smoother_empty_track(transition_model, measurement_model):
    """Test that ViterbiSmoother raises ValueError for empty track."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    empty_track = Track()

    with pytest.raises(ValueError, match="Cannot smooth an empty track"):
        smoother.smooth(empty_track)


def test_viterbi_smoother_smooth_basic(transition_model, measurement_model, simple_track):
    """Test basic smoothing operation on a simple track."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=10  # Use smaller grid for faster testing
    )

    # Smooth the track
    smoothed_track = smoother.smooth(simple_track)

    # Check that we get a track back
    assert isinstance(smoothed_track, Track)

    # Check that smoothed track has same length as input
    assert len(smoothed_track) == len(simple_track)

    # Check that all states are GaussianStateUpdate (same type as input)
    for state in smoothed_track:
        assert isinstance(state, GaussianStateUpdate)

    # Check that timestamps are preserved
    for orig_state, smooth_state in zip(simple_track, smoothed_track):
        assert orig_state.timestamp == smooth_state.timestamp


def test_viterbi_smoother_discretize_state_space(transition_model, measurement_model,
                                                   simple_track):
    """Test state space discretization."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=10
    )

    # Test discretization without bounds (auto-estimated)
    grid_points = smoother._discretize_state_space(simple_track)

    assert len(grid_points) == 10  # num_states
    assert grid_points[0].shape[0] == 4  # State dimension

    # Test discretization with explicit bounds
    smoother.state_bounds = [(0.0, 10.0), (0.0, 2.0), (0.0, 10.0), (0.0, 2.0)]
    grid_points_bounded = smoother._discretize_state_space(simple_track)

    assert len(grid_points_bounded) == 10
    assert grid_points_bounded[0].shape[0] == 4


def test_viterbi_smoother_log_transition_probability(transition_model, measurement_model):
    """Test log transition probability computation."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    state_from = np.array([[1.0], [1.0], [2.0], [1.5]])
    state_to = np.array([[2.0], [1.0], [3.5], [1.5]])
    time_interval = timedelta(seconds=1)

    log_prob = smoother._log_transition_probability(state_from, state_to, time_interval)

    # Should be a finite number (not inf or nan)
    assert np.isfinite(log_prob)
    # Should be a real number (log probability can be positive or negative)
    assert isinstance(log_prob, (float, np.floating))


def test_viterbi_smoother_log_observation_likelihood(transition_model, measurement_model):
    """Test log observation likelihood computation."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    state = np.array([[1.0], [1.0], [2.0], [1.5]])
    detection = Detection(np.array([[1.1], [2.1]]), timestamp=datetime.now())

    log_likelihood = smoother._log_observation_likelihood(state, detection)

    # Should be finite
    assert np.isfinite(log_likelihood)
    # Should be negative (log probability)
    assert log_likelihood <= 0.0


def test_viterbi_smoother_log_observation_likelihood_with_state(transition_model,
                                                                 measurement_model):
    """Test log observation likelihood with state instead of detection."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    state = np.array([[1.0], [1.0], [2.0], [1.5]])
    # Pass a state with full state vector
    measurement_state = GaussianState(
        np.array([[1.1], [1.0], [2.1], [1.5]]),
        np.eye(4),
        timestamp=datetime.now()
    )

    log_likelihood = smoother._log_observation_likelihood(state, measurement_state)

    assert np.isfinite(log_likelihood)


def test_viterbi_smoother_with_predictions(transition_model, measurement_model):
    """Test smoothing with prediction states (no measurements)."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=10
    )

    start = datetime.now()
    track = Track()

    # Create track with a prediction in the middle
    times = [start + timedelta(seconds=i) for i in range(3)]
    state_vectors = [
        np.array([[1.0], [1.0], [2.0], [1.5]]),
        np.array([[2.0], [1.0], [3.5], [1.5]]),
        np.array([[3.0], [1.0], [5.0], [1.5]])
    ]

    for i, (state_vec, timestamp) in enumerate(zip(state_vectors, times)):
        covar = np.eye(4) * 0.1
        if i == 1:
            # Add a prediction without measurement
            state = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        else:
            detection = Detection(state_vec[[0, 2]], timestamp=timestamp)
            pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
            hypothesis = SingleHypothesis(pred, detection)
            state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    # Should not raise error
    smoothed_track = smoother.smooth(track)

    assert len(smoothed_track) == 3
    # Check that prediction is still a prediction
    assert isinstance(smoothed_track[1], GaussianStatePrediction)


def test_viterbi_smoother_preserves_state_type(transition_model, measurement_model):
    """Test that smoother preserves the state type."""
    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=5
    )

    start = datetime.now()
    track = Track()

    # Create track with different state types
    state_vec = np.array([[1.0], [1.0], [2.0], [1.5]])
    covar = np.eye(4) * 0.1
    timestamp = start

    # GaussianState (not Update or Prediction)
    state = GaussianState(state_vec, covar, timestamp=timestamp)
    track.append(state)

    smoothed_track = smoother.smooth(track)

    # Should preserve GaussianState type
    assert type(smoothed_track[0]).__name__ == type(state).__name__


def test_viterbi_smoother_with_custom_bounds(transition_model, measurement_model):
    """Test smoother with custom state bounds."""
    bounds = [(0.0, 10.0), (0.0, 2.0), (0.0, 10.0), (0.0, 2.0)]

    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=5,
        state_bounds=bounds
    )

    start = datetime.now()
    track = Track()

    times = [start + timedelta(seconds=i) for i in range(3)]
    state_vectors = [
        np.array([[1.0], [1.0], [2.0], [1.5]]),
        np.array([[2.0], [1.0], [3.5], [1.5]]),
        np.array([[3.0], [1.0], [5.0], [1.5]])
    ]

    for state_vec, timestamp in zip(state_vectors, times):
        covar = np.eye(4) * 0.1
        state = GaussianState(state_vec, covar, timestamp=timestamp)
        track.append(state)

    smoothed_track = smoother.smooth(track)

    assert len(smoothed_track) == 3
    # Smoothed states should be within bounds (approximately)
    for state in smoothed_track:
        for i, (low, high) in enumerate(bounds):
            # Allow some margin since we're using a discrete grid
            assert low - 1.0 <= state.state_vector[i, 0] <= high + 1.0


def test_viterbi_smoother_non_gaussian_transition():
    """Test smoother with non-Gaussian transition model (no covar method)."""
    # Create a mock transition model without covar
    mock_transition = Mock(spec=['function'])  # Only has function attribute
    mock_transition.function = Mock(return_value=np.array([[1.0], [1.0], [2.0], [1.5]]))

    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2) * 0.5
    )

    smoother = ViterbiSmoother(
        transition_model=mock_transition,
        measurement_model=measurement_model,
        num_states=5
    )

    state_from = np.array([[1.0], [1.0], [2.0], [1.5]])
    state_to = np.array([[2.0], [1.0], [3.5], [1.5]])
    time_interval = timedelta(seconds=1)

    # Should fall back to distance-based scoring (negative distance)
    log_prob = smoother._log_transition_probability(state_from, state_to, time_interval)

    assert np.isfinite(log_prob)
    assert isinstance(log_prob, (float, np.floating))


def test_viterbi_smoother_numerical_stability():
    """Test that smoother handles numerical edge cases."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.1), ConstantVelocity(0.1)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2) * 0.5
    )

    smoother = ViterbiSmoother(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_states=10
    )

    # Create track with very similar states (potential numerical issues)
    start = datetime.now()
    track = Track()

    times = [start + timedelta(seconds=i) for i in range(3)]
    state_vectors = [
        np.array([[1.0], [1.0], [2.0], [1.5]]),
        np.array([[1.001], [1.0], [2.001], [1.5]]),  # Very close
        np.array([[1.002], [1.0], [2.002], [1.5]])   # Very close
    ]

    for state_vec, timestamp in zip(state_vectors, times):
        covar = np.eye(4) * 0.1
        state = GaussianState(state_vec, covar, timestamp=timestamp)
        track.append(state)

    # Should handle without errors
    smoothed_track = smoother.smooth(track)

    assert len(smoothed_track) == 3
    # All states should have finite values
    for state in smoothed_track:
        assert np.all(np.isfinite(state.state_vector))
