"""Tests for Graph-based Viterbi Smoother"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from stonesoup.smoother.graph_viterbi import GraphViterbiSmoother
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
def simple_graph_dict():
    """Create a simple dictionary-based graph."""
    return {
        'nodes': {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
        },
        'edges': [(0, 1), (1, 2), (2, 3), (3, 0)]
    }


@pytest.fixture
def linear_graph_dict():
    """Create a simple linear graph for testing."""
    return {
        'nodes': {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
            3: (3.0, 0.0),
            4: (4.0, 0.0),
        },
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4)]
    }


@pytest.fixture
def simple_track(measurement_model):
    """Create a simple track following a linear path."""
    start = datetime.now()
    track = Track()

    # Create states roughly along x-axis (matching linear_graph)
    times = [start + timedelta(seconds=i) for i in range(5)]
    positions = [
        (0.1, 0.1),
        (1.1, 0.1),
        (2.0, 0.0),
        (3.1, 0.0),
        (4.0, 0.0)
    ]

    for (x, y), timestamp in zip(positions, times):
        state_vec = np.array([[x], [0.9], [y], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(
            np.array([[x], [y]]),
            timestamp=timestamp
        )
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    return track


def test_graph_viterbi_instantiation_dict(transition_model, measurement_model,
                                          simple_graph_dict):
    """Test instantiation with dictionary graph."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    assert smoother.graph is simple_graph_dict
    assert smoother.transition_model is transition_model
    assert smoother.measurement_model is measurement_model
    assert smoother.off_road_penalty == -10.0  # Default
    assert smoother.max_snap_distance is None  # Default


def test_graph_viterbi_instantiation_with_params(transition_model, measurement_model,
                                                  simple_graph_dict):
    """Test instantiation with custom parameters."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model,
        off_road_penalty=-20.0,
        max_snap_distance=2.0
    )

    assert smoother.off_road_penalty == -20.0
    assert smoother.max_snap_distance == 2.0


def test_graph_viterbi_parse_graph_dict(transition_model, measurement_model,
                                        simple_graph_dict):
    """Test that dictionary graph is parsed correctly."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Check nodes are parsed
    assert len(smoother._node_positions) == 4
    assert 0 in smoother._node_positions
    assert 1 in smoother._node_positions
    assert 2 in smoother._node_positions
    assert 3 in smoother._node_positions

    # Check edges are parsed (should be bidirectional)
    assert (0, 1) in smoother._edges
    assert (1, 0) in smoother._edges
    assert (1, 2) in smoother._edges
    assert (2, 1) in smoother._edges


def test_graph_viterbi_invalid_dict_graph(transition_model, measurement_model):
    """Test that invalid dictionary graphs raise errors."""
    # Missing 'nodes' key
    with pytest.raises(ValueError, match="must contain 'nodes' and 'edges' keys"):
        GraphViterbiSmoother(
            graph={'edges': [(0, 1)]},
            transition_model=transition_model,
            measurement_model=measurement_model
        )

    # Missing 'edges' key
    with pytest.raises(ValueError, match="must contain 'nodes' and 'edges' keys"):
        GraphViterbiSmoother(
            graph={'nodes': {0: (0.0, 0.0)}},
            transition_model=transition_model,
            measurement_model=measurement_model
        )


def test_graph_viterbi_empty_graph(transition_model, measurement_model):
    """Test that empty graph raises error."""
    with pytest.raises(ValueError, match="must contain at least one node"):
        GraphViterbiSmoother(
            graph={'nodes': {}, 'edges': []},
            transition_model=transition_model,
            measurement_model=measurement_model
        )


def test_graph_viterbi_invalid_graph_type(transition_model, measurement_model):
    """Test that invalid graph types raise errors."""
    with pytest.raises(TypeError, match="must be a NetworkX graph or dictionary"):
        GraphViterbiSmoother(
            graph="not a graph",
            transition_model=transition_model,
            measurement_model=measurement_model
        )

    with pytest.raises(TypeError, match="must be a NetworkX graph or dictionary"):
        GraphViterbiSmoother(
            graph=[(0, 1), (1, 2)],
            transition_model=transition_model,
            measurement_model=measurement_model
        )


def test_graph_viterbi_find_nearest_node(transition_model, measurement_model,
                                          simple_graph_dict):
    """Test finding nearest node to a measurement."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Measurement near node 0
    meas = Detection(np.array([[0.1], [0.1]]), timestamp=datetime.now())
    nearest, distance = smoother._find_nearest_node(meas)
    assert nearest == 0
    assert distance < 0.2

    # Measurement near node 2
    meas2 = Detection(np.array([[0.9], [1.1]]), timestamp=datetime.now())
    nearest2, distance2 = smoother._find_nearest_node(meas2)
    assert nearest2 == 2
    assert distance2 < 0.2


def test_graph_viterbi_get_neighbors(transition_model, measurement_model,
                                      simple_graph_dict):
    """Test getting neighbors of a node."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Node 0 is connected to 1 and 3
    neighbors_0 = smoother._get_neighbors(0)
    assert 1 in neighbors_0
    assert 3 in neighbors_0
    assert 2 not in neighbors_0

    # Node 1 is connected to 0 and 2
    neighbors_1 = smoother._get_neighbors(1)
    assert 0 in neighbors_1
    assert 2 in neighbors_1


def test_graph_viterbi_empty_track(transition_model, measurement_model,
                                   simple_graph_dict):
    """Test that smoothing empty track raises error."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    with pytest.raises(ValueError, match="Cannot smooth an empty track"):
        smoother.smooth(Track())


def test_graph_viterbi_smooth_basic(transition_model, measurement_model,
                                    linear_graph_dict, simple_track):
    """Test basic smoothing operation."""
    smoother = GraphViterbiSmoother(
        graph=linear_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    smoothed_track = smoother.smooth(simple_track)

    # Check that we get a track back
    assert isinstance(smoothed_track, Track)

    # Check that smoothed track has same length as input
    assert len(smoothed_track) == len(simple_track)

    # Check that timestamps are preserved
    for orig_state, smooth_state in zip(simple_track, smoothed_track):
        assert orig_state.timestamp == smooth_state.timestamp

    # All smoothed states should have state vectors
    for state in smoothed_track:
        assert state.state_vector is not None
        assert state.state_vector.shape[0] == 4


def test_graph_viterbi_smooth_constrained_to_graph(transition_model, measurement_model,
                                                    linear_graph_dict):
    """Test that smoothed states are constrained to graph nodes."""
    smoother = GraphViterbiSmoother(
        graph=linear_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Create track with states off the graph
    start = datetime.now()
    track = Track()

    times = [start + timedelta(seconds=i) for i in range(3)]
    for i, timestamp in enumerate(times):
        # States not exactly on graph nodes
        state_vec = np.array([[float(i) + 0.3], [1.0], [0.5], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(
            np.array([[float(i) + 0.3], [0.5]]),
            timestamp=timestamp
        )
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    smoothed_track = smoother.smooth(track)

    # Smoothed positions should be on graph nodes
    node_positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]

    for state in smoothed_track:
        pos = (state.state_vector[0, 0], state.state_vector[2, 0])
        # Position should match one of the graph nodes
        matches_node = any(
            np.isclose(pos[0], n[0], atol=0.01) and np.isclose(pos[1], n[1], atol=0.01)
            for n in node_positions
        )
        assert matches_node, f"Position {pos} does not match any graph node"


def test_graph_viterbi_single_state_track(transition_model, measurement_model,
                                          simple_graph_dict):
    """Test smoothing a track with single state."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    track = Track()
    timestamp = datetime.now()
    state_vec = np.array([[0.1], [0.0], [0.1], [0.0]])
    covar = np.eye(4) * 0.1
    detection = Detection(np.array([[0.1], [0.1]]), timestamp=timestamp)
    pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
    hypothesis = SingleHypothesis(pred, detection)
    state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
    track.append(state)

    smoothed_track = smoother.smooth(track)

    assert len(smoothed_track) == 1
    # Should snap to nearest node (node 0)
    assert np.isclose(smoothed_track[0].state_vector[0, 0], 0.0, atol=0.01)
    assert np.isclose(smoothed_track[0].state_vector[2, 0], 0.0, atol=0.01)


def test_graph_viterbi_log_transition_probability(transition_model, measurement_model,
                                                   simple_graph_dict):
    """Test log transition probability computation."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    time_interval = timedelta(seconds=1)
    ndim = 4

    # Connected nodes should have finite probability
    log_prob = smoother._log_transition_probability(0, 1, time_interval, ndim)
    assert np.isfinite(log_prob)

    # Unconnected nodes should have -inf probability
    log_prob_unconnected = smoother._log_transition_probability(0, 2, time_interval, ndim)
    assert log_prob_unconnected == -np.inf


def test_graph_viterbi_log_observation_likelihood(transition_model, measurement_model,
                                                   simple_graph_dict):
    """Test log observation likelihood computation."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    ndim = 4

    # Detection near node 0
    detection = Detection(np.array([[0.0], [0.0]]), timestamp=datetime.now())
    log_likelihood = smoother._log_observation_likelihood(0, detection, ndim)

    # Should be finite and relatively high (measurement matches node)
    assert np.isfinite(log_likelihood)

    # Detection far from node 0 (at node 2)
    detection_far = Detection(np.array([[1.0], [1.0]]), timestamp=datetime.now())
    log_likelihood_far = smoother._log_observation_likelihood(0, detection_far, ndim)

    # Likelihood should be lower for far detection
    assert log_likelihood_far < log_likelihood


def test_graph_viterbi_off_road_penalty(transition_model, measurement_model,
                                        linear_graph_dict):
    """Test off-road penalty application."""
    smoother = GraphViterbiSmoother(
        graph=linear_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model,
        max_snap_distance=0.5,
        off_road_penalty=-50.0
    )

    # Create track with one detection way off the graph
    start = datetime.now()
    track = Track()

    positions = [
        (0.0, 0.0),     # On graph
        (1.0, 0.0),     # On graph
        (2.0, 5.0),     # Way off graph (y=5)
        (3.0, 0.0),     # On graph
    ]

    for i, (x, y) in enumerate(positions):
        timestamp = start + timedelta(seconds=i)
        state_vec = np.array([[x], [1.0], [y], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(np.array([[x], [y]]), timestamp=timestamp)
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    # Smoothing should still work
    smoothed_track = smoother.smooth(track)
    assert len(smoothed_track) == 4


def test_graph_viterbi_with_predictions(transition_model, measurement_model,
                                        linear_graph_dict):
    """Test smoothing with prediction states (no measurements)."""
    smoother = GraphViterbiSmoother(
        graph=linear_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    start = datetime.now()
    track = Track()

    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        state_vec = np.array([[float(i)], [1.0], [0.0], [0.0]])
        covar = np.eye(4) * 0.1

        if i == 1:
            # Add a prediction without measurement
            state = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        else:
            detection = Detection(
                np.array([[float(i)], [0.0]]),
                timestamp=timestamp
            )
            pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
            hypothesis = SingleHypothesis(pred, detection)
            state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)

        track.append(state)

    # Should handle predictions
    smoothed_track = smoother.smooth(track)
    assert len(smoothed_track) == 3


def test_graph_viterbi_preserves_state_type(transition_model, measurement_model,
                                            simple_graph_dict):
    """Test that smoother preserves state types."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    track = Track()
    timestamp = datetime.now()
    state_vec = np.array([[0.0], [0.0], [0.0], [0.0]])
    covar = np.eye(4) * 0.1

    # Add a GaussianState
    state = GaussianState(state_vec, covar, timestamp=timestamp)
    track.append(state)

    smoothed_track = smoother.smooth(track)

    assert isinstance(smoothed_track[0], GaussianState)


def test_graph_viterbi_node_position_retrieval(transition_model, measurement_model,
                                                simple_graph_dict):
    """Test node position retrieval with state dimension extension."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Get node 1 position (1.0, 0.0) extended to 4D
    state = smoother._get_node_position(1, ndim=4)

    assert state.shape == (4, 1)
    assert state[0, 0] == 1.0  # x
    assert state[1, 0] == 0.0  # vx (zero)
    # y position might be at index 2 depending on measurement model mapping


def test_graph_viterbi_square_graph_path(transition_model, measurement_model,
                                          simple_graph_dict):
    """Test path finding on square graph."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Create track that should follow path 0 -> 1 -> 2
    start = datetime.now()
    track = Track()

    positions = [
        (0.0, 0.0),   # Near node 0
        (1.0, 0.0),   # Near node 1
        (1.0, 1.0),   # Near node 2
    ]

    for i, (x, y) in enumerate(positions):
        timestamp = start + timedelta(seconds=i)
        state_vec = np.array([[x + 0.05], [0.0], [y + 0.05], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(np.array([[x + 0.05], [y + 0.05]]), timestamp=timestamp)
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    smoothed_track = smoother.smooth(track)

    # Check that smoothed path follows graph
    assert len(smoothed_track) == 3

    # First state should be at node 0
    assert np.isclose(smoothed_track[0].state_vector[0, 0], 0.0, atol=0.01)

    # Second state should be at node 1
    assert np.isclose(smoothed_track[1].state_vector[0, 0], 1.0, atol=0.01)

    # Third state should be at node 2
    assert np.isclose(smoothed_track[2].state_vector[0, 0], 1.0, atol=0.01)


def test_graph_viterbi_numerical_stability(transition_model, measurement_model,
                                           linear_graph_dict):
    """Test numerical stability with many states."""
    smoother = GraphViterbiSmoother(
        graph=linear_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Create a longer track
    start = datetime.now()
    track = Track()

    for i in range(20):
        timestamp = start + timedelta(seconds=i)
        x = min(i / 5.0, 4.0)  # Move along graph
        state_vec = np.array([[x], [0.2], [0.0], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(np.array([[x], [0.0]]), timestamp=timestamp)
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    # Should handle without numerical issues
    smoothed_track = smoother.smooth(track)

    assert len(smoothed_track) == 20
    # All states should have finite values
    for state in smoothed_track:
        assert np.all(np.isfinite(state.state_vector))


def test_graph_viterbi_with_state_measurement(transition_model, measurement_model,
                                               simple_graph_dict):
    """Test smoothing when track contains State objects instead of Updates."""
    smoother = GraphViterbiSmoother(
        graph=simple_graph_dict,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    start = datetime.now()
    track = Track()

    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        state_vec = np.array([[float(i % 2)], [0.0], [float(i // 2)], [0.0]])
        covar = np.eye(4) * 0.1
        state = GaussianState(state_vec, covar, timestamp=timestamp)
        track.append(state)

    smoothed_track = smoother.smooth(track)
    assert len(smoothed_track) == 3


# Test with NetworkX if available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
def test_graph_viterbi_networkx_graph(transition_model, measurement_model):
    """Test instantiation with NetworkX graph."""
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_node(2, pos=(1.0, 1.0))
    G.add_edges_from([(0, 1), (1, 2)])

    smoother = GraphViterbiSmoother(
        graph=G,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    assert len(smoother._node_positions) == 3
    assert (0, 1) in smoother._edges
    assert (1, 2) in smoother._edges


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
def test_graph_viterbi_networkx_missing_pos(transition_model, measurement_model):
    """Test that NetworkX graph with missing pos attribute raises error."""
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1)  # No pos attribute
    G.add_edge(0, 1)

    with pytest.raises(ValueError, match="missing 'pos' attribute"):
        GraphViterbiSmoother(
            graph=G,
            transition_model=transition_model,
            measurement_model=measurement_model
        )


@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not installed")
def test_graph_viterbi_networkx_smooth(transition_model, measurement_model):
    """Test smoothing with NetworkX graph."""
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_node(2, pos=(2.0, 0.0))
    G.add_edges_from([(0, 1), (1, 2)])

    smoother = GraphViterbiSmoother(
        graph=G,
        transition_model=transition_model,
        measurement_model=measurement_model
    )

    # Create simple track
    start = datetime.now()
    track = Track()

    for i in range(3):
        timestamp = start + timedelta(seconds=i)
        state_vec = np.array([[float(i)], [1.0], [0.0], [0.0]])
        covar = np.eye(4) * 0.1
        detection = Detection(np.array([[float(i)], [0.0]]), timestamp=timestamp)
        pred = GaussianStatePrediction(state_vec, covar, timestamp=timestamp)
        hypothesis = SingleHypothesis(pred, detection)
        state = GaussianStateUpdate(state_vec, covar, hypothesis, timestamp=timestamp)
        track.append(state)

    smoothed_track = smoother.smooth(track)
    assert len(smoothed_track) == 3
