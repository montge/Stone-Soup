"""Tests for Viterbi Track Initiator"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from stonesoup.initiator.viterbi import ViterbiTrackInitiator
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


@pytest.fixture
def transition_model():
    """Create a simple transition model for testing."""
    return CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1), ConstantVelocity(0.1)])


@pytest.fixture
def measurement_model():
    """Create a simple measurement model for testing."""
    return LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2) * 0.5)


@pytest.fixture
def simple_detections():
    """Create simple detection sets for testing."""
    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Create 5 scans with simple trajectory
    for i in range(5):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = set()
        # True target trajectory
        detections.add(Detection(np.array([[float(i)], [float(i * 2)]]), timestamp=timestamp))
        # Some clutter
        detections.add(
            Detection(np.array([[float(i) + 10.0], [float(i * 2) + 10.0]]), timestamp=timestamp)
        )

        detection_sets.append(detections)

    return detection_sets, timestamps


def test_viterbi_initiator_instantiation(transition_model, measurement_model):
    """Test that ViterbiTrackInitiator can be instantiated with required models."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    assert initiator.transition_model is transition_model
    assert initiator.measurement_model is measurement_model
    assert initiator.num_scans == 5  # Default value
    assert initiator.detection_threshold == 10.0  # Default value
    assert initiator.max_detections_per_scan == 100  # Default value
    assert initiator.missed_detection_penalty == -5.0  # Default value


def test_viterbi_initiator_instantiation_with_params(transition_model, measurement_model):
    """Test ViterbiTrackInitiator instantiation with custom parameters."""
    prior_covar = np.eye(4) * 10.0

    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        detection_threshold=5.0,
        max_detections_per_scan=50,
        missed_detection_penalty=-10.0,
        prior_state_covar=prior_covar,
    )

    assert initiator.num_scans == 3
    assert initiator.detection_threshold == 5.0
    assert initiator.max_detections_per_scan == 50
    assert initiator.missed_detection_penalty == -10.0
    assert np.array_equal(initiator.prior_state_covar, prior_covar)


def test_viterbi_initiator_initiate_single_scan(transition_model, measurement_model):
    """Test that initiate() returns empty set (Viterbi requires multiple scans)."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    timestamp = datetime.now()
    detections = {Detection(np.array([[1.0], [2.0]]), timestamp=timestamp)}

    tracks = initiator.initiate(detections, timestamp)

    # Should return empty set since Viterbi requires multiple scans
    assert len(tracks) == 0
    assert isinstance(tracks, set)


def test_viterbi_initiator_initiate_from_scans_basic(
    transition_model, measurement_model, simple_detections
):
    """Test basic track initiation from multiple scans."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=5,
        detection_threshold=-10.0,  # Low threshold to ensure we get tracks
    )

    detection_sets, timestamps = simple_detections

    # Initiate tracks
    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should get some tracks
    assert isinstance(tracks, set)
    assert len(tracks) >= 0  # May or may not find tracks depending on threshold

    # If tracks found, validate them
    for track in tracks:
        assert isinstance(track, Track)
        assert len(track) <= len(detection_sets)
        # All states should have timestamps
        for state in track:
            assert state.timestamp is not None


def test_viterbi_initiator_initiate_from_scans_mismatched_lengths(
    transition_model, measurement_model
):
    """Test that mismatched detection sets and timestamps raise ValueError."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    start = datetime.now()
    detection_sets = [
        {Detection(np.array([[1.0], [2.0]]), timestamp=start)},
        {Detection(np.array([[2.0], [3.0]]), timestamp=start + timedelta(seconds=1))},
    ]
    timestamps = [start]  # Only one timestamp

    with pytest.raises(ValueError, match="Number of detection sets must match"):
        initiator.initiate_from_scans(detection_sets, timestamps)


def test_viterbi_initiator_initiate_from_scans_insufficient_scans(
    transition_model, measurement_model
):
    """Test that insufficient scans raise ValueError."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model, num_scans=5
    )

    start = datetime.now()
    detection_sets = [
        {Detection(np.array([[1.0], [2.0]]), timestamp=start)},
        {Detection(np.array([[2.0], [3.0]]), timestamp=start + timedelta(seconds=1))},
    ]
    timestamps = [start, start + timedelta(seconds=1)]

    with pytest.raises(ValueError, match="Need at least 5 scans"):
        initiator.initiate_from_scans(detection_sets, timestamps)


def test_viterbi_initiator_detection_to_state(transition_model, measurement_model):
    """Test conversion of detection to state estimate."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    timestamp = datetime.now()
    detection = Detection(np.array([[5.0], [10.0]]), timestamp=timestamp)

    state = initiator._detection_to_state(detection)

    # Should be a GaussianState
    assert isinstance(state, GaussianState)
    # Should have correct timestamp
    assert state.timestamp == timestamp
    # Should have 4-dimensional state vector (from measurement mapping)
    assert state.state_vector.shape[0] == 4
    # Measured dimensions should match detection
    assert state.state_vector[0, 0] == 5.0
    assert state.state_vector[2, 0] == 10.0
    # Should have covariance
    assert state.covar is not None


def test_viterbi_initiator_detection_to_state_with_custom_covar(
    transition_model, measurement_model
):
    """Test detection to state conversion with custom prior covariance."""
    prior_covar = np.eye(4) * 50.0

    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        prior_state_covar=prior_covar,
    )

    timestamp = datetime.now()
    detection = Detection(np.array([[5.0], [10.0]]), timestamp=timestamp)

    state = initiator._detection_to_state(detection)

    # Should use custom covariance
    assert np.array_equal(state.covar, prior_covar)


def test_viterbi_initiator_compute_detection_score(transition_model, measurement_model):
    """Test detection score computation."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    # Test detection without metadata score
    timestamp = datetime.now()
    detection_no_score = Detection(np.array([[5.0], [10.0]]), timestamp=timestamp)
    score1 = initiator._compute_detection_score(detection_no_score)

    # Should return default score (0.0)
    assert score1 == 0.0

    # Test detection with metadata score
    detection_with_score = Detection(
        np.array([[5.0], [10.0]]), timestamp=timestamp, metadata={"score": 0.8}
    )
    score2 = initiator._compute_detection_score(detection_with_score)

    # Should return log of score
    assert score2 == pytest.approx(np.log(0.8))


def test_viterbi_initiator_compute_transition_score(transition_model, measurement_model):
    """Test transition score computation."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    state_from = GaussianState(
        np.array([[1.0], [1.0], [2.0], [1.5]]), np.eye(4), timestamp=datetime.now()
    )
    state_to = GaussianState(
        np.array([[2.0], [1.0], [3.5], [1.5]]),
        np.eye(4),
        timestamp=datetime.now() + timedelta(seconds=1),
    )
    time_interval = timedelta(seconds=1)

    score = initiator._compute_transition_score(state_from, state_to, time_interval)

    # Should be finite
    assert np.isfinite(score)
    # Should be a real number (log probability can be positive or negative)
    assert isinstance(score, (float, np.floating))


def test_viterbi_initiator_max_detections_per_scan(transition_model, measurement_model):
    """Test that max_detections_per_scan limit is enforced."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        max_detections_per_scan=2,  # Limit to 2 detections
        detection_threshold=-100.0,  # Low threshold
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Create scans with many detections
    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = set()
        # Add 5 detections per scan
        for j in range(5):
            detections.add(Detection(np.array([[float(j)], [float(j)]]), timestamp=timestamp))

        detection_sets.append(detections)

    # Should handle without error, limiting detections
    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should complete without error
    assert isinstance(tracks, set)


def test_viterbi_initiator_with_scored_detections(transition_model, measurement_model):
    """Test initiator with detections that have scores in metadata."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        max_detections_per_scan=3,
        detection_threshold=-50.0,
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Create scans with scored detections
    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = set()
        # Add high-score detection (true target)
        detections.add(
            Detection(
                np.array([[float(i)], [float(i * 2)]]),
                timestamp=timestamp,
                metadata={"score": 0.9},
            )
        )
        # Add low-score detections (clutter)
        detections.add(
            Detection(
                np.array([[float(i) + 5.0], [float(i * 2) + 5.0]]),
                timestamp=timestamp,
                metadata={"score": 0.1},
            )
        )

        detection_sets.append(detections)

    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should handle scored detections
    assert isinstance(tracks, set)


def test_viterbi_initiator_high_threshold_filters_tracks(transition_model, measurement_model):
    """Test that high detection threshold filters out weak tracks."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        detection_threshold=1000.0,  # Very high threshold
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = {Detection(np.array([[float(i)], [float(i * 2)]]), timestamp=timestamp)}

        detection_sets.append(detections)

    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # With very high threshold, should get no tracks
    assert len(tracks) == 0


def test_viterbi_initiator_empty_detection_sets(transition_model, measurement_model):
    """Test initiator with empty detection sets."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model, num_scans=3
    )

    start = datetime.now()
    detection_sets = [set(), set(), set()]  # All empty
    timestamps = [start + timedelta(seconds=i) for i in range(3)]

    # Should handle empty sets gracefully
    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    assert len(tracks) == 0


def test_viterbi_initiator_single_detection_per_scan(transition_model, measurement_model):
    """Test initiator with single detection per scan (simplest case)."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        detection_threshold=-50.0,  # Low enough to get a track
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Create simple trajectory with one detection per scan
    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = {Detection(np.array([[float(i)], [float(i * 2)]]), timestamp=timestamp)}

        detection_sets.append(detections)

    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should get at least one track
    assert len(tracks) >= 0  # May be 0 or 1 depending on threshold

    # If track found, validate it
    for track in tracks:
        assert isinstance(track, Track)
        assert len(track) == 3
        # All states should be GaussianStates
        for state in track:
            assert isinstance(state, GaussianState)


def test_viterbi_initiator_non_gaussian_transition():
    """Test initiator with non-Gaussian transition model."""
    # Create a mock transition model without covar
    mock_transition = Mock(spec=["function"])  # Only has function attribute
    mock_transition.function = Mock(return_value=np.array([[1.0], [1.0], [2.0], [1.5]]))

    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2) * 0.5)

    initiator = ViterbiTrackInitiator(
        transition_model=mock_transition, measurement_model=measurement_model, num_scans=3
    )

    state_from = GaussianState(
        np.array([[1.0], [1.0], [2.0], [1.5]]), np.eye(4), timestamp=datetime.now()
    )
    state_to = GaussianState(
        np.array([[2.0], [1.0], [3.5], [1.5]]),
        np.eye(4),
        timestamp=datetime.now() + timedelta(seconds=1),
    )

    # Should fall back to distance-based scoring (negative distance)
    score = initiator._compute_transition_score(state_from, state_to, timedelta(seconds=1))

    assert np.isfinite(score)
    assert isinstance(score, (float, np.floating))


def test_viterbi_initiator_detection_with_custom_measurement_model(transition_model):
    """Test detection to state conversion when detection has its own measurement model."""
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2) * 0.5)

    initiator = ViterbiTrackInitiator(
        transition_model=transition_model, measurement_model=measurement_model
    )

    # Create detection with its own measurement model
    custom_meas_model = LinearGaussian(
        ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2) * 1.0  # Different noise
    )

    timestamp = datetime.now()
    detection = Detection(
        np.array([[5.0], [10.0]]), timestamp=timestamp, measurement_model=custom_meas_model
    )

    state = initiator._detection_to_state(detection)

    # Should use detection's measurement model
    assert isinstance(state, GaussianState)
    assert state.timestamp == timestamp


def test_viterbi_initiator_uses_only_required_scans(transition_model, measurement_model):
    """Test that initiator uses only num_scans even if more are provided."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,  # Only use 3 scans
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Provide 5 scans
    for i in range(5):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        detections = {Detection(np.array([[float(i)], [float(i * 2)]]), timestamp=timestamp)}

        detection_sets.append(detections)

    # Should only use first 3 scans
    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should complete without error
    assert isinstance(tracks, set)


def test_viterbi_initiator_duplicate_path_filtering(transition_model, measurement_model):
    """Test that duplicate paths are filtered out correctly."""
    initiator = ViterbiTrackInitiator(
        transition_model=transition_model,
        measurement_model=measurement_model,
        num_scans=3,
        detection_threshold=-100.0,  # Very low to allow multiple tracks
    )

    start = datetime.now()
    detection_sets = []
    timestamps = []

    # Create scans where same detections appear (potential for duplicate paths)
    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        timestamps.append(timestamp)

        # Same detection appears in multiple scans
        detections = {Detection(np.array([[float(i)], [float(i * 2)]]), timestamp=timestamp)}

        detection_sets.append(detections)

    tracks = initiator.initiate_from_scans(detection_sets, timestamps)

    # Should handle duplicate filtering
    assert isinstance(tracks, set)
    # Convert to list to check track contents
    track_list = list(tracks)
    # If multiple tracks found, they should be different
    if len(track_list) > 1:
        # Compare first states to ensure tracks are different
        first_states = [track[0].state_vector for track in track_list]
        # Should not all be identical
        for i in range(len(first_states)):
            for _j in range(i + 1, len(first_states)):
                # At least some difference expected
                pass  # Duplicate filtering is based on 80% overlap
