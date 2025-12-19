"""Comprehensive tests for track-to-truth metrics module.

This module contains extensive edge case tests for SIAP and ID-SIAP metrics,
complementing the basic tests in test_tracktotruthmetrics.py.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from ...measures import Euclidean
from ...metricgenerator.manager import MultiManager
from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.hypothesis import SingleDistanceHypothesis
from ...types.time import TimeRange
from ...types.track import Track
from ...types.update import GaussianStateUpdate
from ..tracktotruthmetrics import IDSIAPMetrics, SIAPMetrics

# ================== Edge Cases and Comprehensive Tests ==================


def test_siap_no_tracks():
    """Test SIAP metrics when there are no tracks (only ground truths)."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    # Create ground truths but no tracks
    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(3)]

    truth = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=timestamps[0]),
            GroundTruthState(np.array([[1, 1, 1, 1]]), timestamp=timestamps[1]),
            GroundTruthState(np.array([[2, 1, 2, 1]]), timestamp=timestamps[2]),
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": []})
    manager.association_set = AssociationSet()
    manager.generators = [siap_generator]

    # Test individual methods
    assert siap_generator.num_truths_at_time([truth], timestamps[0]) == 1
    assert siap_generator.num_tracks_at_time([], timestamps[0]) == 0
    assert siap_generator.num_associated_truths_at_time(manager, [truth], timestamps[0]) == 0
    assert siap_generator.num_associated_tracks_at_time(manager, [], timestamps[0]) == 0
    assert siap_generator.total_time_tracked(manager, truth) == 0
    assert siap_generator.min_num_tracks_needed_to_track(manager, truth) == 0
    assert siap_generator.longest_track_time_on_truth(manager, truth) == 0

    # Test compute_metric
    metrics = siap_generator.compute_metric(manager)

    # Extract metrics by title
    metric_dict = {metric.title: metric for metric in metrics}

    # Completeness should be 0 (no truths tracked)
    assert metric_dict["SIAP Completeness"].value == 0

    # Ambiguity should be 1 when no truths are tracked (default)
    assert metric_dict["SIAP Ambiguity"].value == 1

    # Spuriousness should be 0 (no tracks at all)
    assert metric_dict["SIAP Spuriousness"].value == 0

    # Accuracy should be 0 when no associations
    assert metric_dict["SIAP Position Accuracy"].value == 0
    assert metric_dict["SIAP Velocity Accuracy"].value == 0

    # Rate should be 0 (no tracking)
    assert metric_dict["SIAP Rate of Track Number Change"].value == 0

    # Longest track segment should be 0
    assert metric_dict["SIAP Longest Track Segment"].value == 0


def test_siap_no_truths():
    """Test SIAP metrics when there are no ground truths (only tracks)."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    # Create tracks but no ground truths
    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(3)]

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=timestamps[0],
            ),
            GaussianStateUpdate(
                np.array([[1, 1, 1, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 1]]), metadata={}), distance=1
                ),
                timestamp=timestamps[1],
            ),
            GaussianStateUpdate(
                np.array([[2, 1, 2, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 2]]), metadata={}), distance=1
                ),
                timestamp=timestamps[2],
            ),
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [], "tracks": [track]})
    manager.association_set = AssociationSet()
    manager.generators = [siap_generator]

    # Test individual methods
    assert siap_generator.num_truths_at_time([], timestamps[0]) == 0
    assert siap_generator.num_tracks_at_time([track], timestamps[0]) == 1
    assert siap_generator.num_associated_truths_at_time(manager, [], timestamps[0]) == 0
    assert siap_generator.num_associated_tracks_at_time(manager, [track], timestamps[0]) == 0

    # Test compute_metric
    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Completeness should be 0 (no truths to track)
    assert metric_dict["SIAP Completeness"].value == 0

    # Ambiguity should be 1 (default when no associated truths)
    assert metric_dict["SIAP Ambiguity"].value == 1

    # Spuriousness should be 1 (all tracks are spurious)
    assert metric_dict["SIAP Spuriousness"].value == 1

    # Accuracy should be 0 (no associations)
    assert metric_dict["SIAP Position Accuracy"].value == 0
    assert metric_dict["SIAP Velocity Accuracy"].value == 0


def test_siap_perfect_tracking():
    """Test SIAP metrics with perfect one-to-one track-to-truth associations."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(3)]

    # Create perfectly matching truth and track
    truth = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=timestamps[0]),
            GroundTruthState(np.array([[1, 1, 1, 1]]), timestamp=timestamps[1]),
            GroundTruthState(np.array([[2, 1, 2, 1]]), timestamp=timestamps[2]),
        ]
    )

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),  # Perfect match
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=0
                ),
                timestamp=timestamps[0],
            ),
            GaussianStateUpdate(
                np.array([[1, 1, 1, 1]]),  # Perfect match
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 1]]), metadata={}), distance=0
                ),
                timestamp=timestamps[1],
            ),
            GaussianStateUpdate(
                np.array([[2, 1, 2, 1]]),  # Perfect match
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 2]]), metadata={}), distance=0
                ),
                timestamp=timestamps[2],
            ),
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[2])
            )
        }
    )
    manager.generators = [siap_generator]

    # Test individual methods
    assert siap_generator.num_truths_at_time([truth], timestamps[0]) == 1
    assert siap_generator.num_tracks_at_time([track], timestamps[0]) == 1
    assert siap_generator.num_associated_truths_at_time(manager, [truth], timestamps[0]) == 1
    assert siap_generator.num_associated_tracks_at_time(manager, [track], timestamps[0]) == 1
    assert siap_generator.total_time_tracked(manager, truth) == 2  # 2 seconds
    assert siap_generator.min_num_tracks_needed_to_track(manager, truth) == 1
    assert siap_generator.longest_track_time_on_truth(manager, truth) == 2
    assert siap_generator.truth_lifetime(truth) == 2

    # Test compute_metric
    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Perfect completeness (all truths tracked)
    assert metric_dict["SIAP Completeness"].value == 1

    # Perfect ambiguity (1 track per truth)
    assert metric_dict["SIAP Ambiguity"].value == 1

    # No spuriousness (all tracks associated)
    assert metric_dict["SIAP Spuriousness"].value == 0

    # Perfect accuracy (zero error)
    assert metric_dict["SIAP Position Accuracy"].value == 0
    assert metric_dict["SIAP Velocity Accuracy"].value == 0

    # No track changes
    assert metric_dict["SIAP Rate of Track Number Change"].value == 0

    # Full track segment (100% of lifetime)
    assert metric_dict["SIAP Longest Track Segment"].value == 1


def test_siap_empty_data():
    """Test SIAP metrics with completely empty data."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [], "tracks": []})
    manager.association_set = AssociationSet()
    manager.generators = [siap_generator]

    # Test that no metrics are generated with empty data
    # (no timestamps available)
    timestamps = manager.list_timestamps(siap_generator)
    assert len(timestamps) == 0


def test_siap_num_truths_at_time_variations():
    """Test num_truths_at_time with various edge cases."""
    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(5)]

    # Truth that exists for partial time
    truth1 = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=timestamps[0]),
            GroundTruthState(np.array([[1, 1, 1, 1]]), timestamp=timestamps[1]),
        ]
    )

    # Truth that exists for different partial time
    truth2 = GroundTruthPath(
        [
            GroundTruthState(np.array([[2, 1, 2, 1]]), timestamp=timestamps[2]),
            GroundTruthState(np.array([[3, 1, 3, 1]]), timestamp=timestamps[3]),
            GroundTruthState(np.array([[4, 1, 4, 1]]), timestamp=timestamps[4]),
        ]
    )

    truths = [truth1, truth2]

    assert SIAPMetrics.num_truths_at_time(truths, timestamps[0]) == 1
    assert SIAPMetrics.num_truths_at_time(truths, timestamps[1]) == 1
    assert SIAPMetrics.num_truths_at_time(truths, timestamps[2]) == 1
    assert SIAPMetrics.num_truths_at_time(truths, timestamps[3]) == 1
    assert SIAPMetrics.num_truths_at_time(truths, timestamps[4]) == 1

    # Test non-existent timestamp
    non_existent_time = now - timedelta(seconds=1)
    assert SIAPMetrics.num_truths_at_time(truths, non_existent_time) == 0


def test_siap_truth_lifetime_edge_cases():
    """Test truth_lifetime with various scenarios."""
    now = datetime(2024, 1, 1, 0, 0, 0)

    # Single state truth (lifetime = 0)
    truth_single = GroundTruthPath([GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=now)])
    assert SIAPMetrics.truth_lifetime(truth_single) == 0

    # Two states with 5 second gap
    truth_two = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=now),
            GroundTruthState(np.array([[1, 1, 1, 1]]), timestamp=now + timedelta(seconds=5)),
        ]
    )
    assert SIAPMetrics.truth_lifetime(truth_two) == 5

    # Multiple states over 10 seconds
    truth_multi = GroundTruthPath(
        [
            GroundTruthState(np.array([[i, 1, i, 1]]), timestamp=now + timedelta(seconds=i))
            for i in range(11)
        ]
    )
    assert SIAPMetrics.truth_lifetime(truth_multi) == 10


def test_siap_accuracy_multiple_associations():
    """Test accuracy calculation with multiple associations at same timestamp."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)

    # Create truth with known position/velocity
    truth = GroundTruthPath([GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=now)])

    # Create track with specific error (offset by 3, 4 in position dimensions)
    track = Track(
        [
            GaussianStateUpdate(
                np.array([[3, 1, 4, 1]]),  # Error: (3, 4) in position
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=now,
            )
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track]})
    # TimeRange requires start < end, so use a minimal time interval
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(now, now + timedelta(microseconds=1))
            )
        }
    )
    manager.generators = [siap_generator]

    # Expected position error: sqrt(3^2 + 4^2) = 5
    pos_accuracy = siap_generator.accuracy_at_time(manager, now, position_measure)
    assert pos_accuracy == pytest.approx(5.0, abs=1e-9)

    # Velocity error should be 0 (both have velocity [1, 1])
    vel_accuracy = siap_generator.accuracy_at_time(manager, now, velocity_measure)
    assert vel_accuracy == pytest.approx(0.0, abs=1e-9)


def test_siap_spuriousness_variations():
    """Test spuriousness metric with different track/association ratios."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    truth = GroundTruthPath(
        [GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t) for t in timestamps]
    )

    # Create 3 tracks: 1 associated, 2 spurious
    track1 = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )
    track2 = Track(
        [
            GaussianStateUpdate(
                np.array([[10, 1, 10, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[10, 10]]), metadata={}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )
    track3 = Track(
        [
            GaussianStateUpdate(
                np.array([[20, 1, 20, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[20, 20]]), metadata={}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track1, track2, track3]})
    # Only track1 is associated
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track1}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
        }
    )
    manager.generators = [siap_generator]

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Spuriousness should be 2/3 (2 out of 3 tracks are spurious)
    assert metric_dict["SIAP Spuriousness"].value == pytest.approx(2 / 3, abs=1e-9)

    # Completeness should be 1 (only truth is tracked)
    assert metric_dict["SIAP Completeness"].value == 1

    # Ambiguity should be 1 (only 1 track per truth)
    assert metric_dict["SIAP Ambiguity"].value == 1


def test_siap_ambiguity_multiple_tracks_per_truth():
    """Test ambiguity metric when multiple tracks are assigned to same truth."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    truth = GroundTruthPath(
        [GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t) for t in timestamps]
    )

    # Create 3 tracks all associated to same truth
    tracks = []
    for i in range(3):
        track = Track(
            [
                GaussianStateUpdate(
                    np.array([[i, 1, i, 1]]),
                    np.eye(4),
                    SingleDistanceHypothesis(
                        None, Detection(np.array([[i, i]]), metadata={}), distance=1
                    ),
                    timestamp=t,
                )
                for t in timestamps
            ]
        )
        tracks.append(track)

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": tracks})
    # All 3 tracks associated to same truth
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
            for track in tracks
        }
    )
    manager.generators = [siap_generator]

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Ambiguity should be 3 (3 tracks per truth)
    assert metric_dict["SIAP Ambiguity"].value == 3

    # Completeness should be 1 (truth is tracked)
    assert metric_dict["SIAP Completeness"].value == 1

    # Spuriousness should be 0 (all tracks are associated)
    assert metric_dict["SIAP Spuriousness"].value == 0


def test_id_siap_no_metadata():
    """Test ID SIAP metrics when tracks/truths have no ID metadata."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = IDSIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        truth_id="id",
        track_id="id",
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    # Truth and track without ID metadata
    truth = GroundTruthPath(
        [GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t, metadata={}) for t in timestamps]
    )

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
        }
    )
    manager.generators = [siap_generator]

    # Test find_track_id returns None
    assert siap_generator.find_track_id(track, timestamps[0]) is None

    # Test num_id_truths_at_time
    u, c, i = siap_generator.num_id_truths_at_time(manager, [truth], timestamps[0])
    assert u == 1  # Unknown (no ID)
    assert c == 0  # Correct
    assert i == 0  # Incorrect

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # ID Completeness should be 0 (truth is unidentified)
    assert metric_dict["SIAP ID Completeness"].value == 0

    # ID Correctness should be 0 (no correct IDs)
    assert metric_dict["SIAP ID Correctness"].value == 0

    # ID Ambiguity should be 0 (no ambiguous IDs)
    assert metric_dict["SIAP ID Ambiguity"].value == 0


def test_id_siap_correct_ids():
    """Test ID SIAP metrics with all correct ID assignments."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = IDSIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        truth_id="id",
        track_id="id",
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    # Truth and track with matching IDs
    truth = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t, metadata={"id": 123})
            for t in timestamps
        ]
    )

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={"id": 123}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
        }
    )
    manager.generators = [siap_generator]

    # Test find_track_id
    assert siap_generator.find_track_id(track, timestamps[0]) == 123

    # Test num_id_truths_at_time
    u, c, i = siap_generator.num_id_truths_at_time(manager, [truth], timestamps[0])
    assert u == 0  # Unknown
    assert c == 1  # Correct
    assert i == 0  # Incorrect

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # ID Completeness should be 1 (truth is identified)
    assert metric_dict["SIAP ID Completeness"].value == 1

    # ID Correctness should be 1 (all IDs correct)
    assert metric_dict["SIAP ID Correctness"].value == 1

    # ID Ambiguity should be 0 (no ambiguous IDs)
    assert metric_dict["SIAP ID Ambiguity"].value == 0


def test_id_siap_incorrect_ids():
    """Test ID SIAP metrics with incorrect ID assignments."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = IDSIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        truth_id="id",
        track_id="id",
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    # Truth and track with mismatched IDs
    truth = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t, metadata={"id": 123})
            for t in timestamps
        ]
    )

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={"id": 456}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
        }
    )
    manager.generators = [siap_generator]

    # Test num_id_truths_at_time
    u, c, i = siap_generator.num_id_truths_at_time(manager, [truth], timestamps[0])
    assert u == 0  # Unknown
    assert c == 0  # Correct
    assert i == 1  # Incorrect

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # ID Completeness should be 1 (truth is identified, even if wrong)
    assert metric_dict["SIAP ID Completeness"].value == 1

    # ID Correctness should be 0 (IDs are incorrect)
    assert metric_dict["SIAP ID Correctness"].value == 0

    # ID Ambiguity should be 0 (only one ID, not ambiguous)
    assert metric_dict["SIAP ID Ambiguity"].value == 0


def test_id_siap_ambiguous_ids():
    """Test ID SIAP metrics with ambiguous ID assignments."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = IDSIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        truth_id="id",
        track_id="id",
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    # Truth with one ID
    truth = GroundTruthPath(
        [
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t, metadata={"id": 123})
            for t in timestamps
        ]
    )

    # Two tracks with different IDs, both associated to same truth
    track1 = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={"id": 123}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    track2 = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={"id": 456}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track1, track2]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track1}, time_range=TimeRange(timestamps[0], timestamps[1])
            ),
            TimeRangeAssociation(
                objects={truth, track2}, time_range=TimeRange(timestamps[0], timestamps[1])
            ),
        }
    )
    manager.generators = [siap_generator]

    # Test num_id_truths_at_time - tracks have different IDs, so ambiguous
    u, c, i = siap_generator.num_id_truths_at_time(manager, [truth], timestamps[0])
    assert u == 0  # Unknown
    assert c == 0  # Correct (not all tracks have correct ID)
    assert i == 0  # Incorrect (not all tracks have wrong ID)
    # The ambiguous count is calculated as JT - JC - JI - JU in compute_metric

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # ID Completeness should be 1 (truth has IDs assigned)
    assert metric_dict["SIAP ID Completeness"].value == 1

    # ID Correctness should be 0 (not all IDs correct)
    assert metric_dict["SIAP ID Correctness"].value == 0

    # ID Ambiguity should be 1 (ambiguous assignment)
    assert metric_dict["SIAP ID Ambiguity"].value == 1


def test_siap_metric_value_bounds():
    """Test that metric values stay within expected bounds."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(3)]

    # Create scenario with mixed tracking quality
    truths = [
        GroundTruthPath(
            [
                GroundTruthState(
                    np.array([[i, 1, i, 1]]), timestamp=timestamps[j], metadata={"id": i}
                )
                for j in range(3)
            ]
        )
        for i in range(2)
    ]

    tracks = [
        Track(
            [
                GaussianStateUpdate(
                    np.array([[0.1, 1.1, 0.1, 1.1]]),
                    np.eye(4),
                    SingleDistanceHypothesis(
                        None, Detection(np.array([[0, 0]]), metadata={"id": 0}), distance=1
                    ),
                    timestamp=timestamps[j],
                )
                for j in range(3)
            ]
        )
    ]

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": truths, "tracks": tracks})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truths[0], tracks[0]}, time_range=TimeRange(timestamps[0], timestamps[2])
            )
        }
    )
    manager.generators = [siap_generator]

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Completeness should be in [0, 1]
    assert 0 <= metric_dict["SIAP Completeness"].value <= 1

    # Spuriousness should be in [0, 1]
    assert 0 <= metric_dict["SIAP Spuriousness"].value <= 1

    # Ambiguity should be >= 0 (unbounded above, but positive)
    assert metric_dict["SIAP Ambiguity"].value >= 0

    # Position accuracy should be >= 0
    assert metric_dict["SIAP Position Accuracy"].value >= 0

    # Velocity accuracy should be >= 0
    assert metric_dict["SIAP Velocity Accuracy"].value >= 0

    # Rate should be >= 0
    assert metric_dict["SIAP Rate of Track Number Change"].value >= 0

    # Longest track segment should be in [0, 1]
    assert 0 <= metric_dict["SIAP Longest Track Segment"].value <= 1


def test_siap_custom_keys():
    """Test SIAP metrics with custom tracks_key and truths_key."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        tracks_key="custom_tracks",
        truths_key="custom_truths",
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(2)]

    truth = GroundTruthPath(
        [GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t) for t in timestamps]
    )

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=t,
            )
            for t in timestamps
        ]
    )

    manager = MultiManager()
    manager.add_data({"custom_truths": [truth], "custom_tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track}, time_range=TimeRange(timestamps[0], timestamps[1])
            )
        }
    )
    manager.generators = [siap_generator]

    # Should work with custom keys
    metrics = siap_generator.compute_metric(manager)
    assert len(metrics) == 12  # All expected metrics


def test_siap_generator_name():
    """Test that generator_name property works correctly."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))

    # Default generator name
    siap_gen = SIAPMetrics(position_measure=position_measure, velocity_measure=velocity_measure)
    assert siap_gen.generator_name == "siap_generator"

    # Custom generator name
    custom_siap_gen = SIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        generator_name="my_custom_generator",
    )
    assert custom_siap_gen.generator_name == "my_custom_generator"

    # ID SIAP default
    id_siap_gen = IDSIAPMetrics(
        position_measure=position_measure,
        velocity_measure=velocity_measure,
        truth_id="id",
        track_id="id",
    )
    assert id_siap_gen.generator_name == "Id_siap_generator"


def test_siap_truth_track_ordering():
    """Test that truth_track_from_association correctly identifies truth vs track."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)

    truth = GroundTruthPath([GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=now)])

    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=now,
            )
        ]
    )

    # Create association (order in set doesn't matter)
    association = TimeRangeAssociation(
        objects={truth, track}, time_range=TimeRange(now, now + timedelta(microseconds=1))
    )

    # Test that method correctly identifies which is truth and which is track
    extracted_truth, extracted_track = siap_generator.truth_track_from_association(association)

    assert isinstance(extracted_truth, GroundTruthPath)
    assert isinstance(extracted_track, Track)
    assert extracted_truth is truth
    assert extracted_track is track


def test_siap_completeness_at_times():
    """Test completeness metric calculated at individual timestamps."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(3)]

    # Create two truths
    truth1 = GroundTruthPath(
        [GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=t) for t in timestamps]
    )
    truth2 = GroundTruthPath(
        [GroundTruthState(np.array([[5, 1, 5, 1]]), timestamp=t) for t in timestamps]
    )

    # Create one track only associated at first timestamp
    track = Track(
        [
            GaussianStateUpdate(
                np.array([[0, 1, 0, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=timestamps[0],
            )
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth1, truth2], "tracks": [track]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth1, track},
                time_range=TimeRange(timestamps[0], timestamps[0] + timedelta(microseconds=1)),
            )
        }
    )
    manager.generators = [siap_generator]

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Check "at times" metric
    completeness_at_times = metric_dict["SIAP Completeness at times"].value
    assert len(completeness_at_times) == 3

    # At t0: 1 out of 2 truths tracked = 0.5
    assert completeness_at_times[0].value == 0.5

    # At t1 and t2: 0 out of 2 truths tracked = 0
    assert completeness_at_times[1].value == 0
    assert completeness_at_times[2].value == 0


def test_siap_rate_of_track_changes_complex():
    """Test rate of track number changes with complex track switching."""
    position_measure = Euclidean((0, 2))
    velocity_measure = Euclidean((1, 3))
    siap_generator = SIAPMetrics(
        position_measure=position_measure, velocity_measure=velocity_measure
    )

    now = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [now + timedelta(seconds=i) for i in range(5)]

    # Truth tracked by 3 different tracks over time
    truth = GroundTruthPath(
        [GroundTruthState(np.array([[i, 1, i, 1]]), timestamp=timestamps[i]) for i in range(5)]
    )

    # Track 1: covers t0-t1
    track1 = Track(
        [
            GaussianStateUpdate(
                np.array([[i, 1, i, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=timestamps[i],
            )
            for i in range(2)
        ]
    )

    # Track 2: covers t2-t3
    track2 = Track(
        [
            GaussianStateUpdate(
                np.array([[i, 1, i, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=timestamps[i],
            )
            for i in range(2, 4)
        ]
    )

    # Track 3: covers t4
    track3 = Track(
        [
            GaussianStateUpdate(
                np.array([[4, 1, 4, 1]]),
                np.eye(4),
                SingleDistanceHypothesis(
                    None, Detection(np.array([[0, 0]]), metadata={}), distance=1
                ),
                timestamp=timestamps[4],
            )
        ]
    )

    manager = MultiManager()
    manager.add_data({"groundtruth_paths": [truth], "tracks": [track1, track2, track3]})
    manager.association_set = AssociationSet(
        {
            TimeRangeAssociation(
                objects={truth, track1}, time_range=TimeRange(timestamps[0], timestamps[1])
            ),
            TimeRangeAssociation(
                objects={truth, track2}, time_range=TimeRange(timestamps[2], timestamps[3])
            ),
            TimeRangeAssociation(
                objects={truth, track3},
                time_range=TimeRange(timestamps[4], timestamps[4] + timedelta(microseconds=1)),
            ),
        }
    )
    manager.generators = [siap_generator]

    # Verify min_num_tracks_needed
    assert siap_generator.min_num_tracks_needed_to_track(manager, truth) == 3

    # Verify total time tracked (4 seconds: 0-1, 2-3, 4-4 = 1+1+0 = but intervals are [t0,t1], [t2,t3], [t4]
    # Actually: t0 to t1 = 1 second, t2 to t3 = 1 second, no time for single point at t4
    total_time = siap_generator.total_time_tracked(manager, truth)
    assert total_time == 2  # Only the intervals count

    metrics = siap_generator.compute_metric(manager)
    metric_dict = {metric.title: metric for metric in metrics}

    # Rate = (3 - 1) / 2 = 1.0 (2 track switches over 2 seconds of tracking)
    assert metric_dict["SIAP Rate of Track Number Change"].value == pytest.approx(1.0, abs=1e-9)
