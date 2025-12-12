"""Viterbi Smoother and related components.

This module implements the Viterbi algorithm for finding the most likely
sequence of hidden states in a Hidden Markov Model (HMM), applied to target
tracking and state estimation.
"""
import copy
from typing import Sequence

import numpy as np

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..types.prediction import Prediction
from ..types.state import State, GaussianState
from ..types.track import Track
from ..types.update import Update
from .base import Smoother


class ViterbiSmoother(Smoother):
    r"""Viterbi Algorithm Smoother

    Implements the Viterbi algorithm for finding the maximum a posteriori (MAP)
    state sequence in a Hidden Markov Model. This smoother finds the single most
    likely sequence of states given all observations, rather than computing the
    marginal distribution at each time step.

    The Viterbi algorithm operates in three stages:

    1. **Forward Pass (Initialization and Recursion)**:
       For each time step :math:`k`, compute the maximum probability of being in
       state :math:`s_k` given all observations up to time :math:`k`:

       .. math::

           \delta_k(s_k) = \max_{s_1,...,s_{k-1}} P(s_1,...,s_k, z_1,...,z_k)

       This is computed recursively as:

       .. math::

           \delta_k(s_k) = \max_{s_{k-1}} [\delta_{k-1}(s_{k-1}) \cdot
                           P(s_k|s_{k-1}) \cdot P(z_k|s_k)]

       where:
        - :math:`P(s_k|s_{k-1})` is the transition probability
        - :math:`P(z_k|s_k)` is the observation likelihood

       Store the most likely previous state:

       .. math::

           \psi_k(s_k) = \arg\max_{s_{k-1}} [\delta_{k-1}(s_{k-1}) \cdot P(s_k|s_{k-1})]

    2. **Termination**:
       Find the most likely final state:

       .. math::

           s_K^* = \arg\max_{s_K} \delta_K(s_K)

    3. **Backtracking**:
       Trace back through the stored pointers to recover the optimal sequence:

       .. math::

           s_k^* = \psi_{k+1}(s_{k+1}^*) \quad \text{for } k = K-1,...,1

    **Numerical Stability**: All computations are performed in log-space to avoid
    numerical underflow from multiplying many small probabilities:

    .. math::

        \log \delta_k(s_k) = \max_{s_{k-1}} [\log \delta_{k-1}(s_{k-1}) +
                             \log P(s_k|s_{k-1}) + \log P(z_k|s_k)]

    **State Discretization**: The algorithm requires a discrete state space. For
    continuous state spaces (e.g., Gaussian states), the state space must be
    discretized using a grid or quantization scheme.

    Notes
    -----
    - This implementation uses log-domain arithmetic throughout for numerical stability
    - Unlike filtering which estimates :math:`P(s_k|z_{1:k})`, Viterbi finds the
      complete sequence that maximizes :math:`P(s_{1:K}|z_{1:K})`
    - The algorithm has complexity :math:`O(K \cdot N^2)` where :math:`K` is the
      number of time steps and :math:`N` is the number of discrete states
    - For continuous state spaces, consider using grid-based discretization or
      particle-based approximations

    Example
    -------
    >>> from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel,\
    ...                                                 ConstantVelocity
    >>> from stonesoup.models.measurement.linear import LinearGaussian
    >>> from stonesoup.types.state import GaussianState
    >>> from stonesoup.types.track import Track
    >>> import numpy as np
    >>>
    >>> # Define models
    >>> transition_model = CombinedLinearGaussianTransitionModel(
    ...     [ConstantVelocity(0.1), ConstantVelocity(0.1)])
    >>> measurement_model = LinearGaussian(
    ...     ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2)*0.5)
    >>>
    >>> # Create smoother
    >>> smoother = ViterbiSmoother(
    ...     transition_model=transition_model,
    ...     measurement_model=measurement_model,
    ...     num_states=10)
    >>>
    >>> # Smooth a track
    >>> # track = ... (create a track from filtered states)
    >>> # smoothed_track = smoother.smooth(track)

    References
    ----------
    .. [1] Viterbi, A. J. (1967). "Error bounds for convolutional codes and an
           asymptotically optimum decoding algorithm". IEEE Transactions on
           Information Theory. 13 (2): 260–269.
    .. [2] Forney, G. D. (1973). "The Viterbi algorithm". Proceedings of the IEEE.
           61 (3): 268–278.
    .. [3] Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected
           applications in speech recognition". Proceedings of the IEEE. 77 (2): 257–286.

    """

    transition_model: TransitionModel = Property(
        doc="Transition model for state evolution. Used to compute transition "
            "probabilities :math:`P(s_k|s_{k-1})`."
    )
    measurement_model: MeasurementModel = Property(
        doc="Measurement model for observations. Used to compute observation "
            "likelihoods :math:`P(z_k|s_k)`."
    )
    num_states: int = Property(
        default=100,
        doc="Number of discrete states in the quantized state space. For continuous "
            "state spaces, this defines the granularity of the discretization grid. "
            "Higher values provide better resolution but increase computational cost "
            "quadratically. Default is 100."
    )
    state_bounds: Sequence = Property(
        default=None,
        doc="Bounds for state space discretization as list of (min, max) tuples for "
            "each state dimension. If None, bounds are estimated from the track data. "
            "Format: [(x_min, x_max), (y_min, y_max), ...] for each dimension."
    )

    def _discretize_state_space(self, track):
        """Discretize the continuous state space into a grid.

        Parameters
        ----------
        track : :class:`~.Track`
            Input track used to determine state space bounds if not provided

        Returns
        -------
        grid_points : :class:`numpy.ndarray`
            Array of shape (num_states, ndim) containing the discrete state values
        """
        # Get state dimension
        ndim = track[0].state_vector.shape[0]

        # Determine bounds for each dimension
        if self.state_bounds is None:
            # Estimate bounds from track data
            all_states = np.hstack([state.state_vector for state in track])
            mins = np.min(all_states, axis=1)
            maxs = np.max(all_states, axis=1)
            # Add 10% margin
            margins = 0.1 * (maxs - mins)
            bounds = [(mins[i] - margins[i], maxs[i] + margins[i])
                      for i in range(ndim)]
        else:
            bounds = self.state_bounds

        # Create grid for each dimension
        grid_1d = []
        for dim_min, dim_max in bounds:
            grid_1d.append(np.linspace(dim_min, dim_max, self.num_states))

        # For simplicity, use the same grid for all dimensions (can be extended)
        # In a full implementation, you might use different grids per dimension
        # or use a multi-dimensional grid
        grid_points = []
        for i in range(self.num_states):
            state_vec = np.zeros((ndim, 1))
            # Simple uniform quantization along primary dimension
            # This is a simplified approach - full implementation would use
            # multi-dimensional grid or more sophisticated quantization
            for dim in range(ndim):
                state_vec[dim, 0] = grid_1d[dim][i % len(grid_1d[dim])]
            grid_points.append(state_vec)

        return np.array(grid_points)

    def _log_transition_probability(self, state_from, state_to, time_interval):
        """Compute log transition probability.

        Parameters
        ----------
        state_from : :class:`numpy.ndarray`
            Starting state vector
        state_to : :class:`numpy.ndarray`
            Target state vector
        time_interval : :class:`datetime.timedelta`
            Time difference between states

        Returns
        -------
        log_prob : float
            Log probability of transition
        """
        # Compute predicted state
        predicted = self.transition_model.function(
            State(state_from), time_interval=time_interval)

        # Compute log-likelihood of reaching state_to from prediction
        if hasattr(self.transition_model, 'covar'):
            # Gaussian transition model
            covar = self.transition_model.covar(time_interval=time_interval)
            diff = state_to - predicted
            # Avoid numerical issues
            try:
                # Log of multivariate normal PDF
                ndim = len(state_to)
                log_det = np.linalg.slogdet(covar)[1]
                inv_covar = np.linalg.inv(covar)
                mahalanobis = float(diff.T @ inv_covar @ diff)
                log_prob = -0.5 * (ndim * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                log_prob = -np.inf
        else:
            # For non-Gaussian models, use simple Euclidean distance
            # (this is a simplification)
            distance = np.linalg.norm(state_to - predicted)
            log_prob = -distance  # Simple negative distance

        return log_prob

    def _log_observation_likelihood(self, state, measurement):
        """Compute log observation likelihood.

        Parameters
        ----------
        state : :class:`numpy.ndarray`
            State vector
        measurement : :class:`~.Detection` or :class:`~.State`
            Measurement/detection or state containing measurement

        Returns
        -------
        log_likelihood : float
            Log likelihood of observation given state
        """
        # Predicted measurement from state
        predicted_meas = self.measurement_model.function(State(state))

        # Extract measurement vector - handle both Detection and State objects
        if hasattr(measurement, 'state_vector'):
            # Map state to measurement space if needed
            if measurement.state_vector.shape[0] == predicted_meas.shape[0]:
                # Already in measurement space or dimensions match
                meas_vec = measurement.state_vector
            else:
                # Need to map from state space to measurement space
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
                ndim = len(diff)
                log_det = np.linalg.slogdet(covar)[1]
                inv_covar = np.linalg.inv(covar)
                mahalanobis = float(diff.T @ inv_covar @ diff)
                log_likelihood = -0.5 * (ndim * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                log_likelihood = -np.inf
        else:
            # For non-Gaussian models
            distance = np.linalg.norm(meas_vec - predicted_meas)
            log_likelihood = -distance

        return log_likelihood

    def smooth(self, track, **kwargs):
        r"""Perform Viterbi algorithm to find optimal state sequence.

        Implements the forward-backward Viterbi algorithm with backtracking to
        recover the maximum a posteriori (MAP) state sequence.

        Parameters
        ----------
        track : :class:`~.Track`
            Input track containing filtered states and associated measurements

        Returns
        -------
        smoothed_track : :class:`~.Track`
            Shallow copy of input track with states replaced by the optimal
            Viterbi sequence

        Raises
        ------
        ValueError
            If track is empty or contains invalid states
        TypeError
            If states don't have required attributes (state_vector, timestamp)

        Notes
        -----
        The algorithm complexity is :math:`O(K \cdot N^2)` where :math:`K` is the
        track length and :math:`N` is :attr:`num_states`.

        All computations use log-domain arithmetic to prevent numerical underflow.
        """
        if len(track) == 0:
            raise ValueError("Cannot smooth an empty track")

        # Discretize state space
        grid_points = self._discretize_state_space(track)

        # Initialize storage for forward pass
        # delta[k][s] = max log-probability of reaching state s at time k
        delta = []
        # psi[k][s] = argmax previous state leading to state s at time k
        psi = []

        # Extract measurements from track
        measurements = []
        timestamps = []
        for state in track:
            timestamps.append(state.timestamp)
            # Get measurement from state
            if isinstance(state, Update) and state.hypothesis is not None:
                measurements.append(state.hypothesis.measurement)
            elif isinstance(state, Prediction):
                # No measurement for predictions
                measurements.append(None)
            else:
                # Try to extract from state metadata or use state itself
                measurements.append(state)

        K = len(track)

        # Forward Pass
        # ==============
        # Initialize first time step
        delta_0 = np.zeros(self.num_states)
        for s in range(self.num_states):
            if measurements[0] is not None:
                # Log observation likelihood
                delta_0[s] = self._log_observation_likelihood(
                    grid_points[s], measurements[0])
            else:
                # Uniform prior if no measurement
                delta_0[s] = -np.log(self.num_states)

        delta.append(delta_0)
        psi.append(np.zeros(self.num_states, dtype=int))  # No previous state

        # Recursion for k = 1, ..., K-1
        for k in range(1, K):
            delta_k = np.zeros(self.num_states)
            psi_k = np.zeros(self.num_states, dtype=int)

            time_interval = timestamps[k] - timestamps[k-1]

            for s in range(self.num_states):
                # Compute max over previous states
                log_probs = np.zeros(self.num_states)
                for s_prev in range(self.num_states):
                    # Transition probability
                    log_trans = self._log_transition_probability(
                        grid_points[s_prev], grid_points[s], time_interval)
                    log_probs[s_prev] = delta[k-1][s_prev] + log_trans

                # Store max and argmax
                psi_k[s] = np.argmax(log_probs)
                max_log_prob = log_probs[psi_k[s]]

                # Add observation likelihood
                if measurements[k] is not None:
                    log_obs = self._log_observation_likelihood(
                        grid_points[s], measurements[k])
                    delta_k[s] = max_log_prob + log_obs
                else:
                    delta_k[s] = max_log_prob

            delta.append(delta_k)
            psi.append(psi_k)

        # Termination
        # ===========
        # Find most likely final state
        optimal_path = np.zeros(K, dtype=int)
        optimal_path[K-1] = np.argmax(delta[K-1])

        # Backtracking
        # ============
        for k in range(K-2, -1, -1):
            optimal_path[k] = psi[k+1][optimal_path[k+1]]

        # Create smoothed states
        smoothed_states = []
        for k in range(K):
            optimal_state_vec = grid_points[optimal_path[k]]

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
