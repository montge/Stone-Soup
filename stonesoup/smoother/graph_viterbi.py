"""Graph-based Viterbi Smoother for constrained tracking.

This module implements a Viterbi algorithm variant that constrains state
transitions to follow a graph structure, such as a road or rail network.
This is particularly useful for tracking vehicles on known infrastructure.
"""
import copy
from typing import Union, Dict, Tuple, Set

import numpy as np

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..types.prediction import Prediction
from ..types.state import State, GaussianState
from ..types.track import Track
from ..types.update import Update
from .base import Smoother


class GraphViterbiSmoother(Smoother):
    r"""Graph-Constrained Viterbi Algorithm Smoother

    Implements a variant of the Viterbi algorithm that constrains state transitions
    to follow edges in a graph structure, such as a road or rail network. This is
    particularly effective for tracking vehicles or targets that are constrained to
    follow known infrastructure.

    The algorithm operates similarly to standard Viterbi but with key differences:

    1. **Discrete State Space**: States are discrete graph nodes rather than
       continuous values. Each node represents a position in the network.

    2. **Graph-Constrained Transitions**: Transitions are only allowed between
       nodes connected by edges in the graph. This enforces that paths follow
       the network structure.

    3. **Off-Graph Penalty**: Detections that are far from any graph node can
       be handled with a configurable penalty, allowing for graceful handling
       of off-network detections (e.g., GPS errors, temporary off-road travel).

    The Viterbi algorithm operates in three stages:

    1. **Forward Pass**:
       For each time step :math:`k`, compute the maximum log-probability of
       being at each node :math:`n_k`:

       .. math::

           \delta_k(n_k) = \max_{n_{k-1}} [\delta_{k-1}(n_{k-1}) +
                           \log P(n_k|n_{k-1}) + \log P(z_k|n_k)]

       where transitions :math:`P(n_k|n_{k-1})` are only non-zero for edges
       in the graph.

    2. **Termination**:
       Find the most likely final node:

       .. math::

           n_K^* = \arg\max_{n_K} \delta_K(n_K)

    3. **Backtracking**:
       Trace back through stored pointers to recover the optimal path:

       .. math::

           n_k^* = \psi_{k+1}(n_{k+1}^*) \quad \text{for } k = K-1,...,1

    **Graph Format**: The graph can be provided as:
    - A NetworkX graph object with nodes having 'pos' attribute (x, y coordinates)
    - A dictionary with 'nodes' and 'edges' keys where:
      - 'nodes': dict mapping node_id -> (x, y) position
      - 'edges': list of (node_id1, node_id2) tuples

    **Numerical Stability**: All computations use log-space arithmetic to avoid
    numerical underflow.

    Parameters
    ----------
    graph : networkx.Graph or dict
        Graph structure representing the network. If NetworkX graph, nodes should
        have a 'pos' attribute with (x, y) coordinates. If dict, should contain
        'nodes' and 'edges' keys.
    transition_model : TransitionModel
        Transition model for computing transition probabilities between connected nodes
    measurement_model : MeasurementModel
        Measurement model for computing observation likelihoods at nodes
    off_road_penalty : float
        Log-probability penalty for detections far from graph nodes. Default is -10.0.
        More negative values make off-graph states less likely.
    max_snap_distance : float
        Maximum distance (in state space units) to snap detections to nearest node.
        Detections farther than this receive the off_road_penalty. Default is None
        (no distance limit).

    Example
    -------
    >>> import networkx as nx
    >>> from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel,\
    ...                                                 ConstantVelocity
    >>> from stonesoup.models.measurement.linear import LinearGaussian
    >>> import numpy as np
    >>>
    >>> # Create a simple road network graph
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0.0, 0.0))
    >>> G.add_node(1, pos=(1.0, 0.0))
    >>> G.add_node(2, pos=(1.0, 1.0))
    >>> G.add_node(3, pos=(0.0, 1.0))
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    >>>
    >>> # Define models
    >>> transition_model = CombinedLinearGaussianTransitionModel(
    ...     [ConstantVelocity(0.1), ConstantVelocity(0.1)])
    >>> measurement_model = LinearGaussian(
    ...     ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2)*0.5)
    >>>
    >>> # Create smoother
    >>> smoother = GraphViterbiSmoother(
    ...     graph=G,
    ...     transition_model=transition_model,
    ...     measurement_model=measurement_model,
    ...     off_road_penalty=-10.0)
    >>>
    >>> # Smooth a track
    >>> # track = ... (create a track from filtered states)
    >>> # smoothed_track = smoother.smooth(track)

    Notes
    -----
    - This implementation is particularly effective for urban tracking, railway
      monitoring, or any scenario where targets follow known infrastructure
    - The algorithm complexity is :math:`O(K \cdot E)` where :math:`K` is the
      track length and :math:`E` is the average number of edges per node
    - For large graphs, consider spatial indexing (e.g., KD-tree) for efficient
      nearest-node lookups

    References
    ----------
    .. [1] Viterbi, A. J. (1967). "Error bounds for convolutional codes and an
           asymptotically optimum decoding algorithm". IEEE Transactions on
           Information Theory. 13 (2): 260–269.
    .. [2] Newson, P. and Krumm, J. (2009). "Hidden Markov map matching through
           noise and sparseness". Proceedings of the 17th ACM SIGSPATIAL
           International Conference on Advances in Geographic Information Systems.
    .. [3] Raymond, R., et al. (2012). "Map matching with travel time constraints".
           IEEE Transactions on Intelligent Transportation Systems. 13 (1): 131-141.

    """

    graph: Union[object, Dict] = Property(
        doc="Graph structure representing the road/rail network. Can be a NetworkX "
            "graph with nodes having 'pos' attribute, or a dictionary with 'nodes' "
            "and 'edges' keys. Nodes should have associated positions (x, y coordinates)."
    )
    transition_model: TransitionModel = Property(
        doc="Transition model for state evolution. Used to compute transition "
            "probabilities between connected graph nodes."
    )
    measurement_model: MeasurementModel = Property(
        doc="Measurement model for observations. Used to compute observation "
            "likelihoods at each graph node."
    )
    off_road_penalty: float = Property(
        default=-10.0,
        doc="Log-probability penalty applied to detections that are far from any "
            "graph node (beyond max_snap_distance). More negative values make "
            "off-graph detections less likely in the optimal path. Default is -10.0."
    )
    max_snap_distance: float = Property(
        default=None,
        doc="Maximum distance (in state space units) for snapping detections to "
            "nearest graph node. Detections farther than this receive the "
            "off_road_penalty. If None, all detections are snapped to nearest node "
            "regardless of distance. Default is None."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_positions = {}
        self._edges = set()
        self._parse_graph()

    def _parse_graph(self):
        """Parse graph structure into internal format.

        Extracts node positions and edges from the provided graph structure,
        whether it's a NetworkX graph or a dictionary.
        """
        # Check if it's a NetworkX graph (duck typing)
        if hasattr(self.graph, 'nodes') and hasattr(self.graph, 'edges'):
            # NetworkX graph
            for node, data in self.graph.nodes(data=True):
                if 'pos' in data:
                    self._node_positions[node] = np.array(data['pos']).reshape(-1, 1)
                else:
                    raise ValueError(f"Node {node} missing 'pos' attribute")

            for u, v in self.graph.edges():
                self._edges.add((u, v))
                self._edges.add((v, u))  # Make undirected
        elif isinstance(self.graph, dict):
            # Dictionary format
            if 'nodes' not in self.graph or 'edges' not in self.graph:
                raise ValueError("Graph dictionary must contain 'nodes' and 'edges' keys")

            for node_id, pos in self.graph['nodes'].items():
                self._node_positions[node_id] = np.array(pos).reshape(-1, 1)

            for u, v in self.graph['edges']:
                self._edges.add((u, v))
                self._edges.add((v, u))  # Make undirected
        else:
            raise TypeError("Graph must be a NetworkX graph or dictionary")

        if not self._node_positions:
            raise ValueError("Graph must contain at least one node with position")

    def _get_node_position(self, node_id, ndim):
        """Get position vector for a node, extended to state space dimension.

        Parameters
        ----------
        node_id : hashable
            Node identifier
        ndim : int
            State vector dimension

        Returns
        -------
        state_vector : numpy.ndarray
            State vector with position filled in (other dims zero)
        """
        pos = self._node_positions[node_id]
        state_vector = np.zeros((ndim, 1))
        # Fill in position dimensions (assuming first dims are position)
        pos_dim = min(pos.shape[0], ndim)
        state_vector[:pos_dim, :] = pos[:pos_dim, :]
        return state_vector

    def _find_nearest_node(self, measurement):
        """Find nearest graph node to a measurement.

        Parameters
        ----------
        measurement : Detection or State
            Measurement or state containing position information

        Returns
        -------
        nearest_node : hashable
            ID of nearest node
        distance : float
            Distance to nearest node
        """
        # Extract position from measurement
        if hasattr(measurement, 'state_vector'):
            # Map to measurement space if needed
            if hasattr(self.measurement_model, 'mapping'):
                mapping = self.measurement_model.mapping
                # Get position dimensions from mapping
                meas_vec = measurement.state_vector[mapping[:2], :]
            else:
                meas_vec = measurement.state_vector[:2, :]
        else:
            meas_vec = measurement[:2, :]

        # Find nearest node
        min_distance = float('inf')
        nearest_node = None

        for node_id, pos in self._node_positions.items():
            distance = np.linalg.norm(pos[:2, :] - meas_vec[:2, :])
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id

        return nearest_node, min_distance

    def _get_neighbors(self, node_id):
        """Get neighboring nodes connected by edges.

        Parameters
        ----------
        node_id : hashable
            Node identifier

        Returns
        -------
        neighbors : set
            Set of neighboring node IDs
        """
        neighbors = set()
        for u, v in self._edges:
            if u == node_id:
                neighbors.add(v)
        return neighbors

    def _log_transition_probability(self, node_from, node_to, time_interval, ndim):
        """Compute log transition probability between nodes.

        Parameters
        ----------
        node_from : hashable
            Starting node ID
        node_to : hashable
            Target node ID
        time_interval : datetime.timedelta
            Time difference between states
        ndim : int
            State vector dimension

        Returns
        -------
        log_prob : float
            Log probability of transition (or -inf if not connected)
        """
        # Check if nodes are connected
        if (node_from, node_to) not in self._edges:
            return -np.inf  # Not connected

        # Get node positions
        state_from = self._get_node_position(node_from, ndim)
        state_to = self._get_node_position(node_to, ndim)

        # Compute predicted state from transition model
        predicted = self.transition_model.function(
            State(state_from), time_interval=time_interval)

        # Compute log-likelihood of reaching state_to from prediction
        if hasattr(self.transition_model, 'covar'):
            # Gaussian transition model
            covar = self.transition_model.covar(time_interval=time_interval)
            diff = state_to - predicted
            try:
                # Log of multivariate normal PDF
                log_det = np.linalg.slogdet(covar)[1]
                inv_covar = np.linalg.inv(covar)
                mahalanobis = float((diff.T @ inv_covar @ diff).item())
                log_prob = -0.5 * (ndim * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                log_prob = -np.inf
        else:
            # For non-Gaussian models, use simple Euclidean distance
            distance = np.linalg.norm(state_to - predicted)
            log_prob = -distance  # Simple negative distance

        return log_prob

    def _log_observation_likelihood(self, node_id, measurement, ndim):
        """Compute log observation likelihood at a node.

        Parameters
        ----------
        node_id : hashable
            Node identifier
        measurement : Detection or State
            Measurement/detection or state
        ndim : int
            State vector dimension

        Returns
        -------
        log_likelihood : float
            Log likelihood of observation given state at node
        """
        # Get state at node
        state_vector = self._get_node_position(node_id, ndim)

        # Predicted measurement from state
        predicted_meas = self.measurement_model.function(State(state_vector))

        # Extract measurement vector
        if hasattr(measurement, 'state_vector'):
            # Map state to measurement space if needed
            if measurement.state_vector.shape[0] == predicted_meas.shape[0]:
                meas_vec = measurement.state_vector
            else:
                # Use measurement model mapping
                mapping = self.measurement_model.mapping
                meas_vec = measurement.state_vector[mapping, :]
        else:
            meas_vec = measurement

        # Compute log-likelihood
        if hasattr(self.measurement_model, 'covar'):
            # Gaussian measurement model
            covar = self.measurement_model.covar()
            diff = meas_vec - predicted_meas
            try:
                ndim_meas = len(diff)
                log_det = np.linalg.slogdet(covar)[1]
                inv_covar = np.linalg.inv(covar)
                mahalanobis = float((diff.T @ inv_covar @ diff).item())
                log_likelihood = -0.5 * (ndim_meas * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                log_likelihood = -np.inf
        else:
            # For non-Gaussian models
            distance = np.linalg.norm(meas_vec - predicted_meas)
            log_likelihood = -distance

        return log_likelihood

    def smooth(self, track, **kwargs):
        r"""Perform graph-constrained Viterbi algorithm on track.

        Implements the Viterbi algorithm with state transitions constrained to
        follow edges in the graph structure. Returns the maximum a posteriori
        (MAP) path through the graph.

        Parameters
        ----------
        track : Track
            Input track containing filtered states and associated measurements

        Returns
        -------
        smoothed_track : Track
            Shallow copy of input track with states replaced by optimal
            graph-constrained sequence

        Raises
        ------
        ValueError
            If track is empty or contains invalid states

        Notes
        -----
        The algorithm complexity is :math:`O(K \cdot E)` where :math:`K` is the
        track length and :math:`E` is the average number of edges per node.

        All computations use log-domain arithmetic for numerical stability.
        """
        if len(track) == 0:
            raise ValueError("Cannot smooth an empty track")

        # Get state dimension
        ndim = track[0].state_vector.shape[0]

        # Get list of all node IDs
        node_list = list(self._node_positions.keys())
        num_nodes = len(node_list)

        # Create node ID to index mapping
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

        # Initialize storage for forward pass
        delta = []  # delta[k][node_idx] = max log-probability
        psi = []    # psi[k][node_idx] = argmax previous node

        # Extract measurements and timestamps
        measurements = []
        timestamps = []
        for state in track:
            timestamps.append(state.timestamp)
            # Get measurement from state
            if isinstance(state, Update) and state.hypothesis is not None:
                measurements.append(state.hypothesis.measurement)
            elif isinstance(state, Prediction):
                measurements.append(None)
            else:
                measurements.append(state)

        K = len(track)

        # Forward Pass
        # ============
        # Initialize first time step
        delta_0 = np.full(num_nodes, -np.inf)

        if measurements[0] is not None:
            # Find nearest node to first measurement
            nearest_node, distance = self._find_nearest_node(measurements[0])

            # Check if within snap distance
            off_graph = (self.max_snap_distance is not None and
                        distance > self.max_snap_distance)

            for idx, node_id in enumerate(node_list):
                log_obs = self._log_observation_likelihood(
                    node_id, measurements[0], ndim)

                # Apply penalty if detection is off-graph
                if off_graph and node_id != nearest_node:
                    log_obs += self.off_road_penalty

                delta_0[idx] = log_obs
        else:
            # Uniform prior if no measurement
            delta_0[:] = -np.log(num_nodes)

        delta.append(delta_0)
        psi.append(np.zeros(num_nodes, dtype=int))  # No previous state

        # Recursion for k = 1, ..., K-1
        for k in range(1, K):
            delta_k = np.full(num_nodes, -np.inf)
            psi_k = np.zeros(num_nodes, dtype=int)

            time_interval = timestamps[k] - timestamps[k-1]

            # Check if current measurement is off-graph
            off_graph = False
            nearest_node = None
            if measurements[k] is not None:
                nearest_node, distance = self._find_nearest_node(measurements[k])
                off_graph = (self.max_snap_distance is not None and
                            distance > self.max_snap_distance)

            for idx, node_id in enumerate(node_list):
                # Get neighbors of current node
                neighbors = self._get_neighbors(node_id)

                # If no neighbors (isolated node), allow transitions from all nodes
                if not neighbors:
                    neighbors = set(node_list)

                # Compute max over previous states (only neighbors)
                max_log_prob = -np.inf
                best_prev_idx = 0

                for neighbor_id in neighbors:
                    neighbor_idx = node_to_idx[neighbor_id]

                    # Transition probability (only for connected nodes)
                    log_trans = self._log_transition_probability(
                        neighbor_id, node_id, time_interval, ndim)

                    if not np.isfinite(log_trans):
                        continue

                    log_prob = delta[k-1][neighbor_idx] + log_trans

                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_idx = neighbor_idx

                psi_k[idx] = best_prev_idx

                # Add observation likelihood
                if measurements[k] is not None:
                    log_obs = self._log_observation_likelihood(
                        node_id, measurements[k], ndim)

                    # Apply penalty if detection is off-graph
                    if off_graph and node_id != nearest_node:
                        log_obs += self.off_road_penalty

                    delta_k[idx] = max_log_prob + log_obs
                else:
                    delta_k[idx] = max_log_prob

            delta.append(delta_k)
            psi.append(psi_k)

        # Termination
        # ===========
        # Find most likely final node
        optimal_path_idx = np.zeros(K, dtype=int)
        optimal_path_idx[K-1] = np.argmax(delta[K-1])

        # Backtracking
        # ============
        for k in range(K-2, -1, -1):
            optimal_path_idx[k] = psi[k+1][optimal_path_idx[k+1]]

        # Convert indices back to node IDs
        optimal_path = [node_list[idx] for idx in optimal_path_idx]

        # Create smoothed states
        smoothed_states = []
        for k in range(K):
            optimal_state_vec = self._get_node_position(optimal_path[k], ndim)

            # Create smoothed state with same type as original
            original_state = track[k]
            if isinstance(original_state, GaussianState):
                # Preserve covariance from original state
                smoothed_state = type(original_state).from_state(
                    original_state,
                    optimal_state_vec,
                    original_state.covar if hasattr(original_state, 'covar') else None
                )
            else:
                # For non-Gaussian states
                smoothed_state = type(original_state).from_state(
                    original_state,
                    optimal_state_vec
                )

            smoothed_states.append(smoothed_state)

        # Create smoothed track
        smoothed_track = copy.copy(track)
        smoothed_track.states = smoothed_states

        return smoothed_track
