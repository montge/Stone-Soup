import datetime

import pytest
import numpy as np

from ..general import OneToOneAssociator
from ...measures.base import BaseMeasure, SetComparisonMeasure
from ...measures.state import Euclidean, EuclideanWeighted
from ...types.association import Association, AssociationSet
from ...types.detection import Detection
from ...types.state import State, GaussianState
from ...types.track import Track


class SimpleMeasure(BaseMeasure):
    """Simple measure that returns the sum of absolute differences for testing."""

    def __call__(self, item1, item2):
        """Calculate simple distance measure between two states."""
        if hasattr(item1, 'state_vector') and hasattr(item2, 'state_vector'):
            state_vector1 = getattr(item1, 'mean', item1.state_vector)
            state_vector2 = getattr(item2, 'mean', item2.state_vector)
            return float(np.sum(np.abs(state_vector1 - state_vector2)))
        return None


class ConditionalMeasure(BaseMeasure):
    """Measure that returns None for specific conditions (for testing None handling)."""

    def __call__(self, item1, item2):
        """Return None if items are far apart, otherwise return distance."""
        if hasattr(item1, 'state_vector') and hasattr(item2, 'state_vector'):
            state_vector1 = getattr(item1, 'mean', item1.state_vector)
            state_vector2 = getattr(item2, 'mean', item2.state_vector)
            distance = float(np.sum(np.abs(state_vector1 - state_vector2)))
            # Return None if distance is too large
            if distance > 20:
                return None
            return distance
        return None


class MaximiseMeasure(BaseMeasure):
    """Measure for maximisation testing (e.g., similarity scores)."""

    def __call__(self, item1, item2):
        """Calculate similarity measure (higher is better)."""
        if hasattr(item1, 'state_vector') and hasattr(item2, 'state_vector'):
            state_vector1 = getattr(item1, 'mean', item1.state_vector)
            state_vector2 = getattr(item2, 'mean', item2.state_vector)
            distance = float(np.sum(np.abs(state_vector1 - state_vector2)))
            # Return negative distance as similarity (closer = higher similarity)
            return -distance
        return None


@pytest.fixture()
def simple_measure():
    return SimpleMeasure()


@pytest.fixture()
def euclidean_measure():
    return Euclidean()


@pytest.fixture()
def conditional_measure():
    return ConditionalMeasure()


@pytest.fixture()
def maximise_measure():
    return MaximiseMeasure()


@pytest.fixture()
def weighted_measure():
    return EuclideanWeighted(weighting=np.array([1, 0.5]))


@pytest.fixture()
def set_comparison_measure():
    return SetComparisonMeasure()


@pytest.fixture()
def simple_states():
    """Create simple state objects for testing."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
        State(state_vector=np.array([[20], [20]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[11], [11]]), timestamp=timestamp),
        State(state_vector=np.array([[21], [21]]), timestamp=timestamp),
    ]
    return states_a, states_b


@pytest.fixture()
def simple_tracks():
    """Create simple tracks for testing."""
    timestamp = datetime.datetime.now()
    tracks = [
        Track([GaussianState(np.array([[0], [0], [0], [0]]),
                            np.diag([1, 0.1, 1, 0.1]), timestamp)]),
        Track([GaussianState(np.array([[10], [0], [10], [0]]),
                            np.diag([1, 0.1, 1, 0.1]), timestamp)]),
        Track([GaussianState(np.array([[20], [0], [20], [0]]),
                            np.diag([1, 0.1, 1, 0.1]), timestamp)]),
    ]
    return tracks


@pytest.fixture()
def simple_detections():
    """Create simple detections for testing."""
    timestamp = datetime.datetime.now()
    detections = [
        Detection(np.array([[1], [1]]), timestamp=timestamp),
        Detection(np.array([[11], [11]]), timestamp=timestamp),
        Detection(np.array([[21], [21]]), timestamp=timestamp),
    ]
    return detections


# Test OneToOneAssociator initialization and properties

def test_basic_instantiation(simple_measure):
    """Test basic instantiation with required parameters."""
    associator = OneToOneAssociator(measure=simple_measure)
    assert associator.measure is simple_measure
    assert associator.maximise_measure is False
    assert associator.association_threshold == 1e10


def test_instantiation_with_threshold(simple_measure):
    """Test instantiation with custom association threshold."""
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=5.0
    )
    assert associator.association_threshold == 5.0


def test_instantiation_with_maximise(simple_measure):
    """Test instantiation with maximise_measure flag."""
    associator = OneToOneAssociator(
        measure=simple_measure,
        maximise_measure=True
    )
    assert associator.maximise_measure is True
    assert associator.association_threshold == -1e10


def test_instantiation_maximise_with_threshold(simple_measure):
    """Test instantiation with both maximise and custom threshold."""
    associator = OneToOneAssociator(
        measure=simple_measure,
        maximise_measure=True,
        association_threshold=-10.0
    )
    assert associator.maximise_measure is True
    assert associator.association_threshold == -10.0


def test_fail_value_property_minimise(simple_measure):
    """Test fail_value property with minimise measure."""
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=5.0
    )
    assert associator.fail_value == 5.0


def test_fail_value_property_maximise(simple_measure):
    """Test fail_value property with maximise measure."""
    associator = OneToOneAssociator(
        measure=simple_measure,
        maximise_measure=True,
        association_threshold=-5.0
    )
    assert associator.fail_value == -5.0


# Test the associate method of OneToOneAssociator

def test_associate_basic(simple_measure, simple_states):
    """Test basic association with equal number of objects."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # All objects should be associated
    assert len(associations.associations) == 3
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0

    # Verify associations are correct (closest pairs)
    associated_pairs = [(assoc.objects) for assoc in associations.associations]
    # Each association should have 2 objects
    for pair in associated_pairs:
        assert len(pair) == 2


def test_associate_with_threshold(simple_measure, simple_states):
    """Test association with threshold that filters some associations."""
    states_a, states_b = simple_states
    # Threshold of 1.5 should only allow the first pair (distance ~2)
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=1.5
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # No associations should meet the threshold
    assert len(associations.associations) == 0
    assert len(unassociated_a) == 3
    assert len(unassociated_b) == 3


def test_associate_with_high_threshold(simple_measure, simple_states):
    """Test association with high threshold that allows all associations."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=100.0
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # All should be associated
    assert len(associations.associations) == 3
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0


def test_associate_empty_first_collection(simple_measure, simple_states):
    """Test association with empty first collection."""
    _, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        [], states_b
    )

    assert len(associations.associations) == 0
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == len(states_b)


def test_associate_empty_second_collection(simple_measure, simple_states):
    """Test association with empty second collection."""
    states_a, _ = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, []
    )

    assert len(associations.associations) == 0
    assert len(unassociated_a) == len(states_a)
    assert len(unassociated_b) == 0


def test_associate_both_empty(simple_measure):
    """Test association with both collections empty."""
    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        [], []
    )

    assert len(associations.associations) == 0
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0


def test_associate_unequal_sizes(simple_measure):
    """Test association with unequal collection sizes."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[11], [11]]), timestamp=timestamp),
        State(state_vector=np.array([[21], [21]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should have 2 associations (limited by smaller collection)
    assert len(associations.associations) == 2
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 1


def test_associate_with_euclidean_measure(euclidean_measure):
    """Test association using Euclidean distance measure."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[5], [5]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[6], [6]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(measure=euclidean_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    assert len(associations.associations) == 2
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0


def test_associate_with_gaussian_states(euclidean_measure):
    """Test association with GaussianState objects."""
    timestamp = datetime.datetime.now()
    states_a = [
        GaussianState(np.array([[0], [0]]), np.eye(2), timestamp),
        GaussianState(np.array([[10], [10]]), np.eye(2), timestamp),
    ]
    states_b = [
        GaussianState(np.array([[1], [1]]), np.eye(2), timestamp),
        GaussianState(np.array([[11], [11]]), np.eye(2), timestamp),
    ]

    associator = OneToOneAssociator(measure=euclidean_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    assert len(associations.associations) == 2


# Test OneToOneAssociator with maximise_measure=True

def test_maximise_basic(maximise_measure):
    """Test association with maximisation objective."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[11], [11]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(
        measure=maximise_measure,
        maximise_measure=True
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should associate closest pairs (highest similarity scores)
    assert len(associations.associations) == 2
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0


def test_maximise_with_threshold(maximise_measure):
    """Test maximisation with threshold filtering."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[50], [50]]), timestamp=timestamp),
    ]

    # Threshold of -5 means similarity must be greater than -5
    # (i.e., distance must be less than 5)
    associator = OneToOneAssociator(
        measure=maximise_measure,
        maximise_measure=True,
        association_threshold=-5.0
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Only the first pair should meet the threshold (distance ~2)
    # The second pair has distance ~80
    assert len(associations.associations) == 1
    assert len(unassociated_a) == 1
    assert len(unassociated_b) == 1


# Test the individual_weighting method

def test_individual_weighting_minimise(simple_measure):
    """Test individual_weighting with minimise measure."""
    timestamp = datetime.datetime.now()
    state_a = State(state_vector=np.array([[0], [0]]), timestamp=timestamp)
    state_b = State(state_vector=np.array([[1], [1]]), timestamp=timestamp)

    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=10.0
    )

    weight = associator.individual_weighting(state_a, state_b)

    # Should return the measure value (2.0 in this case)
    assert weight == 2.0


def test_individual_weighting_with_none(conditional_measure):
    """Test individual_weighting when measure returns None."""
    timestamp = datetime.datetime.now()
    state_a = State(state_vector=np.array([[0], [0]]), timestamp=timestamp)
    # This state is far away, so measure will return None
    state_b = State(state_vector=np.array([[100], [100]]), timestamp=timestamp)

    associator = OneToOneAssociator(
        measure=conditional_measure,
        association_threshold=10.0
    )

    weight = associator.individual_weighting(state_a, state_b)

    # Should return fail_value when measure returns None
    assert weight == associator.fail_value


def test_individual_weighting_maximise(maximise_measure):
    """Test individual_weighting with maximise measure."""
    timestamp = datetime.datetime.now()
    state_a = State(state_vector=np.array([[0], [0]]), timestamp=timestamp)
    state_b = State(state_vector=np.array([[1], [1]]), timestamp=timestamp)

    associator = OneToOneAssociator(
        measure=maximise_measure,
        maximise_measure=True,
        association_threshold=-10.0
    )

    weight = associator.individual_weighting(state_a, state_b)

    # Should return max of measure value and fail_value
    expected = max(-2.0, -10.0)
    assert weight == expected


def test_individual_weighting_caps_to_threshold_minimise(simple_measure):
    """Test that individual_weighting caps to threshold for minimise."""
    timestamp = datetime.datetime.now()
    state_a = State(state_vector=np.array([[0], [0]]), timestamp=timestamp)
    state_b = State(state_vector=np.array([[100], [100]]), timestamp=timestamp)

    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=5.0
    )

    weight = associator.individual_weighting(state_a, state_b)

    # Actual measure would be 200, but should cap at threshold
    assert weight == min(200.0, 5.0)


def test_individual_weighting_caps_to_threshold_maximise(maximise_measure):
    """Test that individual_weighting caps to threshold for maximise."""
    timestamp = datetime.datetime.now()
    state_a = State(state_vector=np.array([[0], [0]]), timestamp=timestamp)
    state_b = State(state_vector=np.array([[100], [100]]), timestamp=timestamp)

    associator = OneToOneAssociator(
        measure=maximise_measure,
        maximise_measure=True,
        association_threshold=-5.0
    )

    weight = associator.individual_weighting(state_a, state_b)

    # Actual measure would be -200, should cap at threshold -5.0
    assert weight == max(-200.0, -5.0)


# Test the association_dict method

def test_association_dict_basic(simple_measure, simple_states):
    """Test association_dict with all objects associated."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    assoc_dict = associator.association_dict(states_a, states_b)

    # Should have entries for all objects
    assert len(assoc_dict) == 6

    # Each object should be a key
    for state in states_a + states_b:
        assert state in assoc_dict

    # Check bidirectional associations
    for state_a in states_a:
        associated_b = assoc_dict[state_a]
        if associated_b is not None:
            assert assoc_dict[associated_b] == state_a


def test_association_dict_with_unassociated(simple_measure):
    """Test association_dict with some unassociated objects."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[100], [100]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=50.0
    )

    assoc_dict = associator.association_dict(states_a, states_b)

    # Should have entries for all objects
    assert len(assoc_dict) == 3

    # states_a[0] should associate with states_b[0]
    assert assoc_dict[states_a[0]] == states_b[0]
    assert assoc_dict[states_b[0]] == states_a[0]

    # states_b[1] should not be associated (too far)
    assert assoc_dict[states_b[1]] is None


def test_association_dict_empty_collections(simple_measure):
    """Test association_dict with empty collections."""
    associator = OneToOneAssociator(measure=simple_measure)

    assoc_dict = associator.association_dict([], [])

    assert len(assoc_dict) == 0


def test_association_dict_one_empty(simple_measure, simple_states):
    """Test association_dict with one empty collection."""
    states_a, _ = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    assoc_dict = associator.association_dict(states_a, [])

    # All objects should map to None
    assert len(assoc_dict) == len(states_a)
    for state in states_a:
        assert assoc_dict[state] is None


# Test OneToOneAssociator with Track objects

def test_associate_tracks_and_detections(euclidean_measure):
    """Test associating tracks with detections."""
    timestamp = datetime.datetime.now()

    tracks = {
        Track([GaussianState(np.array([[0], [0]]), np.eye(2), timestamp)]),
        Track([GaussianState(np.array([[10], [10]]), np.eye(2), timestamp)]),
    }

    detections = {
        Detection(np.array([[1], [1]]), timestamp=timestamp),
        Detection(np.array([[11], [11]]), timestamp=timestamp),
    }

    # Use Euclidean with mapping to position dimensions
    measure = Euclidean(mapping=[0, 1])
    associator = OneToOneAssociator(measure=measure)

    # Get current states from tracks
    track_states = {track[-1] for track in tracks}

    associations, unassoc_tracks, unassoc_detections = associator.associate(
        track_states, detections
    )

    assert len(associations.associations) == 2
    assert len(unassoc_tracks) == 0
    assert len(unassoc_detections) == 0


# Test OneToOneAssociator with set collections

def test_associate_with_sets(simple_measure):
    """Test that association works with set collections."""
    timestamp = datetime.datetime.now()
    states_a = {
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    }
    states_b = {
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[11], [11]]), timestamp=timestamp),
    }

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    assert len(associations.associations) == 2
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0

    # Verify return types are sets
    assert isinstance(unassociated_a, set)
    assert isinstance(unassociated_b, set)


# Test edge cases and error conditions

def test_single_object_each(simple_measure):
    """Test association with single object in each collection."""
    timestamp = datetime.datetime.now()
    states_a = [State(state_vector=np.array([[0], [0]]), timestamp=timestamp)]
    states_b = [State(state_vector=np.array([[1], [1]]), timestamp=timestamp)]

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    assert len(associations.associations) == 1
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0


def test_very_large_threshold(simple_measure):
    """Test with very large association threshold."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[1000], [1000]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[1001], [1001]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=1e15
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # All should associate despite large distances
    assert len(associations.associations) == 2


def test_very_small_threshold(simple_measure):
    """Test with very small association threshold."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[0.001], [0.001]]), timestamp=timestamp),
        State(state_vector=np.array([[10.001], [10.001]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=0.001
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # None should associate (all distances > 0.001)
    assert len(associations.associations) == 0
    assert len(unassociated_a) == 2
    assert len(unassociated_b) == 2


def test_threshold_exactly_at_boundary(simple_measure):
    """Test threshold exactly at the boundary of association."""
    timestamp = datetime.datetime.now()
    states_a = [State(state_vector=np.array([[0], [0]]), timestamp=timestamp)]
    states_b = [State(state_vector=np.array([[1], [1]]), timestamp=timestamp)]

    # Distance is exactly 2.0
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=2.0
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should not associate (threshold is non-inclusive)
    assert len(associations.associations) == 0
    assert len(unassociated_a) == 1
    assert len(unassociated_b) == 1


def test_threshold_just_above_boundary(simple_measure):
    """Test threshold just above the boundary."""
    timestamp = datetime.datetime.now()
    states_a = [State(state_vector=np.array([[0], [0]]), timestamp=timestamp)]
    states_b = [State(state_vector=np.array([[1], [1]]), timestamp=timestamp)]

    # Distance is 2.0, threshold is 2.1
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=2.1
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should associate
    assert len(associations.associations) == 1


def test_all_measure_returns_none(conditional_measure):
    """Test when measure returns None for all pairs."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[100], [100]]), timestamp=timestamp),
    ]

    # Conditional measure returns None for distances > 20
    associator = OneToOneAssociator(measure=conditional_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # No associations should be made
    assert len(associations.associations) == 0
    assert len(unassociated_a) == 1
    assert len(unassociated_b) == 1


# Test that return types are correct

def test_association_set_type(simple_measure, simple_states):
    """Test that first return value is AssociationSet."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    associations, _, _ = associator.associate(states_a, states_b)

    assert isinstance(associations, AssociationSet)


def test_association_objects_type(simple_measure, simple_states):
    """Test that individual associations are Association objects."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    associations, _, _ = associator.associate(states_a, states_b)

    for assoc in associations.associations:
        assert isinstance(assoc, Association)


def test_unassociated_are_sets(simple_measure, simple_states):
    """Test that unassociated returns are sets."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    _, unassociated_a, unassociated_b = associator.associate(states_a, states_b)

    assert isinstance(unassociated_a, set)
    assert isinstance(unassociated_b, set)


def test_association_dict_return_type(simple_measure, simple_states):
    """Test that association_dict returns a dict."""
    states_a, states_b = simple_states
    associator = OneToOneAssociator(measure=simple_measure)

    assoc_dict = associator.association_dict(states_a, states_b)

    assert isinstance(assoc_dict, dict)


# Test complex real-world scenarios

def test_many_to_few_association(simple_measure):
    """Test scenario with many objects in first collection, few in second."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[i], [i]]), timestamp=timestamp)
        for i in range(10)
    ]
    states_b = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[5], [5]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Only 2 associations possible
    assert len(associations.associations) == 2
    assert len(unassociated_a) == 8
    assert len(unassociated_b) == 0


def test_few_to_many_association(simple_measure):
    """Test scenario with few objects in first collection, many in second."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[1], [1]]), timestamp=timestamp),
        State(state_vector=np.array([[5], [5]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[i], [i]]), timestamp=timestamp)
        for i in range(10)
    ]

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Only 2 associations possible
    assert len(associations.associations) == 2
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 8


def test_clustered_objects(simple_measure):
    """Test with clustered objects where optimal assignment matters."""
    timestamp = datetime.datetime.now()
    # Two clusters: one at (0,0), one at (10,10)
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[0.5], [0.5]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[0.2], [0.2]]), timestamp=timestamp),
        State(state_vector=np.array([[10.2], [10.2]]), timestamp=timestamp),
        State(state_vector=np.array([[10.5], [10.5]]), timestamp=timestamp),
    ]

    associator = OneToOneAssociator(measure=simple_measure)

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should find optimal one-to-one assignment
    assert len(associations.associations) == 3
    assert len(unassociated_a) == 0
    assert len(unassociated_b) == 0

    # Calculate total distance of all associations
    total_distance = 0
    for assoc in associations.associations:
        obj_list = list(assoc.objects)
        sv1 = getattr(obj_list[0], 'mean', obj_list[0].state_vector)
        sv2 = getattr(obj_list[1], 'mean', obj_list[1].state_vector)
        distance = float(np.sum(np.abs(sv1 - sv2)))
        total_distance += distance

    # The optimal assignment should have a relatively small total distance
    # Worst case would be all cross-cluster assignments (~30), optimal is ~1.4
    assert total_distance < 25.0


def test_mixed_thresholds(simple_measure):
    """Test scenario where some pairs meet threshold and others don't."""
    timestamp = datetime.datetime.now()
    states_a = [
        State(state_vector=np.array([[0], [0]]), timestamp=timestamp),
        State(state_vector=np.array([[10], [10]]), timestamp=timestamp),
        State(state_vector=np.array([[20], [20]]), timestamp=timestamp),
    ]
    states_b = [
        State(state_vector=np.array([[0.5], [0.5]]), timestamp=timestamp),
        State(state_vector=np.array([[10.5], [10.5]]), timestamp=timestamp),
        State(state_vector=np.array([[100], [100]]), timestamp=timestamp),
    ]

    # Threshold of 2.0 should allow first two pairs but not the third
    associator = OneToOneAssociator(
        measure=simple_measure,
        association_threshold=2.0
    )

    associations, unassociated_a, unassociated_b = associator.associate(
        states_a, states_b
    )

    # Should get 2 associations, 1 unassociated in each
    assert len(associations.associations) == 2
    assert len(unassociated_a) == 1
    assert len(unassociated_b) == 1
