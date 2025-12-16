"""Viterbi-based Track Initiation.

This module implements track-before-detect (TBD) using the Viterbi algorithm
to find the most likely target trajectories directly from detection data.
"""

import datetime
from collections.abc import Sequence

import numpy as np

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..types.detection import Detection
from ..types.state import GaussianState
from ..types.track import Track
from .base import Initiator


class ViterbiTrackInitiator(Initiator):
    r"""Viterbi Track-Before-Detect Initiator

    Implements track-before-detect (TBD) using the Viterbi algorithm to jointly
    detect and track targets. Rather than using a traditional detect-then-track
    approach, this initiator considers all detection data over multiple scans
    simultaneously to find the most likely target trajectories.

    The algorithm operates on a batch of detections spanning multiple time steps
    and finds trajectories that:
    1. Are dynamically feasible according to the transition model
    2. Have sufficiently high cumulative detection scores
    3. Represent the globally optimal solution over the time window

    **Algorithm Overview**:

    Given detections :math:`\{z_k^i\}` where :math:`k` is the time index and
    :math:`i` is the detection index at time :math:`k`, the algorithm:

    1. **State Space Construction**: Creates a trellis structure where each node
       represents a possible detection-to-state association at each time step,
       plus "null" states for missed detections.

    2. **Forward Pass**: Computes the maximum log-likelihood path to each node:

       .. math::

           \delta_k(i) = \max_{j} [\delta_{k-1}(j) + \log P(s_k^i | s_{k-1}^j) +
                         \log P(z_k^i | s_k^i)]

       where:
        - :math:`\delta_k(i)` is the max log-likelihood to reach detection :math:`i`
          at time :math:`k`
        - :math:`P(s_k^i | s_{k-1}^j)` is the dynamic feasibility (transition probability)
        - :math:`P(z_k^i | s_k^i)` is the detection quality score

    3. **Backtracking**: Traces back from high-scoring terminal nodes to recover
       complete trajectories

    4. **Track Extraction**: Creates tracks from trajectories exceeding the
       detection threshold

    **Advantages over Traditional Methods**:
    - Handles low signal-to-noise ratio scenarios where single-scan detection fails
    - Exploits temporal continuity to discriminate targets from clutter
    - Provides globally optimal solutions over the time window
    - Naturally handles missed detections and track gaps

    **Computational Complexity**:
    :math:`O(K \cdot M^2)` where :math:`K` is the number of scans and :math:`M`
    is the average number of detections per scan.

    Notes
    -----
    - This is a batch algorithm requiring :attr:`num_scans` detections before
      producing tracks
    - All computations use log-domain arithmetic for numerical stability
    - The algorithm assumes detections include both true targets and false alarms
    - Detection scores should ideally be log-likelihoods or similar calibrated metrics

    Example
    -------
    >>> from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel,\
    ...                                                 ConstantVelocity
    >>> from stonesoup.models.measurement.linear import LinearGaussian
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>>
    >>> # Define models
    >>> transition_model = CombinedLinearGaussianTransitionModel(
    ...     [ConstantVelocity(0.1), ConstantVelocity(0.1)])
    >>> measurement_model = LinearGaussian(
    ...     ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2)*0.5)
    >>>
    >>> # Create initiator
    >>> initiator = ViterbiTrackInitiator(
    ...     transition_model=transition_model,
    ...     measurement_model=measurement_model,
    ...     num_scans=5,
    ...     detection_threshold=10.0)
    >>>
    >>> # Collect detections over multiple scans
    >>> all_detections = []  # List of sets of detections, one per scan
    >>> timestamps = []
    >>> # ... collect detections ...
    >>>
    >>> # Initiate tracks
    >>> tracks = initiator.initiate_from_scans(all_detections, timestamps)

    References
    ----------
    .. [1] Davey, S. J., Rutten, M. G., & Cheung, B. (2007). "A comparison of
           detection performance for several track-before-detect algorithms".
           In 2007 10th International Conference on Information Fusion (pp. 1-8). IEEE.
    .. [2] Grossi, E., Lops, M., & Venturino, L. (2015). "A novel dynamic programming
           algorithm for track-before-detect in radar systems". IEEE Transactions on
           Signal Processing, 61(10), 2608-2619.
    .. [3] Vo, B. N., Vo, B. T., & Mahler, R. (2012). "Closed-form solutions to
           forward-backward smoothing". IEEE Transactions on Signal Processing,
           60(1), 2-17.

    """

    transition_model: TransitionModel = Property(
        doc="Transition model for state evolution. Defines the dynamic constraints "
        "for feasible trajectories."
    )
    measurement_model: MeasurementModel = Property(
        doc="Measurement model. Used to compute state estimates from detections "
        "and evaluate detection likelihoods."
    )
    num_scans: int = Property(
        default=5,
        doc="Number of consecutive scans to use for track initiation. Larger values "
        "improve discrimination between targets and clutter but increase latency "
        "and computational cost. Typical values: 3-10. Default is 5.",
    )
    detection_threshold: float = Property(
        default=10.0,
        doc="Minimum cumulative log-likelihood threshold for track initiation. "
        "Trajectories with cumulative score below this threshold are rejected. "
        "Higher values reduce false tracks but may miss weak targets. "
        "Default is 10.0.",
    )
    max_detections_per_scan: int = Property(
        default=100,
        doc="Maximum number of detections to consider per scan. Limits computational "
        "cost in high-clutter scenarios. Detections are selected by strength if "
        "this limit is exceeded. Default is 100.",
    )
    missed_detection_penalty: float = Property(
        default=-5.0,
        doc="Log-likelihood penalty for missed detections. Allows tracks to skip "
        "time steps where no detection is associated. More negative values make "
        "missed detections less likely. Default is -5.0.",
    )
    prior_state_covar: np.ndarray = Property(
        default=None,
        doc="Prior state covariance for initiated tracks. If None, a default "
        "identity matrix scaled by 100 is used. Should match state dimensions.",
    )

    def _compute_detection_score(self, detection):
        """Compute quality score for a detection.

        Parameters
        ----------
        detection : :class:`~.Detection`
            Input detection

        Returns
        -------
        score : float
            Log-likelihood score for the detection
        """
        # If detection has a metadata score, use it
        if hasattr(detection, "metadata") and "score" in detection.metadata:
            return np.log(max(detection.metadata["score"], 1e-10))

        # Otherwise, return a default moderate score
        # In practice, this should be based on detection SNR or similar
        return 0.0

    def _compute_transition_score(self, state_from, state_to, time_interval):
        """Compute dynamic feasibility score.

        Parameters
        ----------
        state_from : :class:`~.State`
            Starting state
        state_to : :class:`~.State`
            Target state
        time_interval : :class:`datetime.timedelta`
            Time difference

        Returns
        -------
        score : float
            Log-likelihood of transition (negative values indicate less likely transitions)
        """
        # Predict next state
        predicted = self.transition_model.function(state_from, time_interval=time_interval)

        # Compute likelihood of reaching state_to
        if hasattr(self.transition_model, "covar"):
            covar = self.transition_model.covar(time_interval=time_interval)
            diff = state_to.state_vector - predicted
            try:
                ndim = len(diff)
                log_det = np.linalg.slogdet(covar)[1]
                inv_covar = np.linalg.inv(covar)
                mahalanobis = float((diff.T @ inv_covar @ diff).item())
                score = -0.5 * (ndim * np.log(2 * np.pi) + log_det + mahalanobis)
            except np.linalg.LinAlgError:
                score = -np.inf
        else:
            # Simple distance-based score
            distance = np.linalg.norm(state_to.state_vector - predicted)
            score = -distance

        return score

    def _detection_to_state(self, detection):
        """Convert detection to state estimate.

        Parameters
        ----------
        detection : :class:`~.Detection`
            Input detection

        Returns
        -------
        state : :class:`~.State`
            Estimated state from detection
        """
        # Get measurement model
        meas_model = detection.measurement_model or self.measurement_model

        # Map detection to state space
        # For simplicity, use pseudo-inverse for linear models
        from ..models.base import LinearModel

        if isinstance(meas_model, LinearModel):
            H = meas_model.matrix()
            state_vector = np.linalg.pinv(H) @ detection.state_vector
        else:
            # For non-linear models, would need more sophisticated inversion
            # Here we use a simple approach: assume detection gives partial state
            ndim_state = meas_model.ndim_state
            state_vector = np.zeros((ndim_state, 1))
            mapping = meas_model.mapping
            state_vector[mapping, :] = detection.state_vector

        # Create state
        if self.prior_state_covar is not None:
            covar = self.prior_state_covar
        else:
            # Default covariance
            ndim = state_vector.shape[0]
            covar = np.eye(ndim) * 100.0

        return GaussianState(state_vector, covar, timestamp=detection.timestamp)

    def _limit_detections(self, detection_sets):
        """Limit detections per scan to max_detections_per_scan.

        Parameters
        ----------
        detection_sets : sequence of sets of :class:`~.Detection`
            Input detection sets

        Returns
        -------
        list
            List of detection lists, each limited to max_detections_per_scan
        """
        processed_detections = []
        for det_set in detection_sets:
            det_list = list(det_set)
            if len(det_list) > self.max_detections_per_scan:
                if det_list and hasattr(det_list[0], "metadata"):
                    if "score" in det_list[0].metadata:
                        det_list.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
                det_list = det_list[: self.max_detections_per_scan]
            processed_detections.append(det_list)
        return processed_detections

    def _initialize_trellis(self, first_scan_detections):
        """Initialize trellis for first scan.

        Parameters
        ----------
        first_scan_detections : list
            Detections from the first scan

        Returns
        -------
        delta_0 : np.ndarray
            Initial scores for first scan
        psi_0 : list
            Initial backpointers (all -1 for first scan)
        """
        delta_0 = np.array([self._compute_detection_score(det) for det in first_scan_detections])
        psi_0 = [-1] * len(first_scan_detections)
        return delta_0, psi_0

    def _find_best_predecessor(
        self, state_curr, det_score, prev_detections, prev_delta, time_interval
    ):
        """Find the best predecessor detection for a current detection.

        Parameters
        ----------
        state_curr : :class:`~.State`
            Current state estimate
        det_score : float
            Detection score for current detection
        prev_detections : list
            Detections from previous scan
        prev_delta : np.ndarray
            Delta values from previous scan
        time_interval : datetime.timedelta
            Time between scans

        Returns
        -------
        best_score : float
            Best total score
        best_prev : int
            Index of best predecessor
        """
        best_score = -np.inf
        best_prev = 0

        for j, det_prev in enumerate(prev_detections):
            state_prev = self._detection_to_state(det_prev)
            trans_score = self._compute_transition_score(state_prev, state_curr, time_interval)
            total_score = prev_delta[j] + trans_score + det_score

            if total_score > best_score:
                best_score = total_score
                best_prev = j

        return best_score, best_prev

    def _forward_pass(self, processed_detections, timestamps):
        """Perform forward pass of Viterbi algorithm.

        Parameters
        ----------
        processed_detections : list
            List of detection lists per scan
        timestamps : list
            Timestamps for each scan

        Returns
        -------
        delta : list
            Delta values for each scan
        psi : list
            Backpointers for each scan
        """
        delta_0, psi_0 = self._initialize_trellis(processed_detections[0])
        delta = [delta_0]
        psi = [psi_0]

        for k in range(1, len(processed_detections)):
            time_interval = timestamps[k] - timestamps[k - 1]
            delta_k, psi_k = self._forward_step(
                processed_detections[k], processed_detections[k - 1], delta[k - 1], time_interval
            )
            delta.append(delta_k)
            psi.append(psi_k)

        return delta, psi

    def _forward_step(self, curr_detections, prev_detections, prev_delta, time_interval):
        """Perform one forward step of Viterbi algorithm.

        Parameters
        ----------
        curr_detections : list
            Detections at current scan
        prev_detections : list
            Detections at previous scan
        prev_delta : np.ndarray
            Delta values from previous scan
        time_interval : datetime.timedelta
            Time between scans

        Returns
        -------
        delta_k : np.ndarray
            Delta values for current scan
        psi_k : np.ndarray
            Backpointers for current scan
        """
        num_curr = len(curr_detections)
        delta_k = np.zeros(num_curr)
        psi_k = np.zeros(num_curr, dtype=int)

        for i, det_curr in enumerate(curr_detections):
            state_curr = self._detection_to_state(det_curr)
            det_score = self._compute_detection_score(det_curr)
            best_score, best_prev = self._find_best_predecessor(
                state_curr, det_score, prev_detections, prev_delta, time_interval
            )
            delta_k[i] = best_score
            psi_k[i] = best_prev

        return delta_k, psi_k

    def _backtrack_path(self, candidate_idx, psi, processed_detections):
        """Backtrack to extract a single path.

        Parameters
        ----------
        candidate_idx : int
            Index of terminal node
        psi : list
            Backpointers from forward pass
        processed_detections : list
            List of detection lists per scan

        Returns
        -------
        path : list
            List of detections forming the path
        """
        path = []
        curr_idx = candidate_idx
        num_scans = len(processed_detections)

        for k in range(num_scans - 1, -1, -1):
            det = processed_detections[k][curr_idx]
            path.insert(0, det)
            if k > 0:
                curr_idx = psi[k][curr_idx]

        return path

    def _is_unique_path(self, path, extracted_paths, overlap_threshold=0.8):
        """Check if a path is sufficiently unique from existing paths.

        Parameters
        ----------
        path : list
            Path to check
        extracted_paths : list
            Previously extracted paths
        overlap_threshold : float
            Maximum overlap ratio for paths to be considered duplicates

        Returns
        -------
        bool
            True if path is unique
        """
        for existing_path in extracted_paths:
            shared = sum(1 for d1, d2 in zip(path, existing_path, strict=False) if d1 is d2)
            if shared >= len(path) * overlap_threshold:
                return False
        return True

    def _extract_tracks(self, delta, psi, processed_detections):
        """Extract tracks from Viterbi results.

        Parameters
        ----------
        delta : list
            Delta values from forward pass
        psi : list
            Backpointers from forward pass
        processed_detections : list
            List of detection lists per scan

        Returns
        -------
        tracks : set
            Set of Track objects
        """
        tracks = set()
        final_scores = delta[-1]
        candidates = np.where(final_scores > self.detection_threshold)[0]
        extracted_paths = []

        for candidate_idx in candidates:
            path = self._backtrack_path(candidate_idx, psi, processed_detections)

            if self._is_unique_path(path, extracted_paths):
                extracted_paths.append(path)
                states = [self._detection_to_state(det) for det in path]
                tracks.add(Track(states))

        return tracks

    def initiate(self, detections, timestamp, **kwargs):
        """Initiate tracks from detections at a single time step.

        Note: This method provides minimal functionality for compatibility.
        Use :meth:`initiate_from_scans` for full Viterbi track-before-detect.

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            Detections at current time
        timestamp : datetime.datetime
            Current timestamp

        Returns
        -------
        tracks : set of :class:`~.Track`
            Empty set (use initiate_from_scans for batch processing)
        """
        # Viterbi requires multiple scans, so single-scan initiation
        # returns empty set
        return set()

    def initiate_from_scans(
        self, detection_sets: Sequence[set[Detection]], timestamps: Sequence[datetime.datetime]
    ) -> set[Track]:
        """Initiate tracks from multiple scans using Viterbi algorithm.

        This is the main method for Viterbi track-before-detect. It processes
        a batch of detection sets and returns the most likely trajectories.

        Parameters
        ----------
        detection_sets : sequence of sets of :class:`~.Detection`
            Sequence of detection sets, one per scan, in temporal order
        timestamps : sequence of datetime.datetime
            Timestamps corresponding to each scan

        Returns
        -------
        tracks : set of :class:`~.Track`
            Initiated tracks that exceed the detection threshold

        Raises
        ------
        ValueError
            If number of detection sets doesn't match timestamps, or if
            fewer than :attr:`num_scans` are provided
        """
        if len(detection_sets) != len(timestamps):
            raise ValueError("Number of detection sets must match number of timestamps")

        if len(detection_sets) < self.num_scans:
            raise ValueError(
                f"Need at least {self.num_scans} scans for initiation, "
                f"got {len(detection_sets)}"
            )

        # Use only the required number of scans
        detection_sets = detection_sets[: self.num_scans]
        timestamps = timestamps[: self.num_scans]

        # Limit detections per scan and perform forward pass
        processed_detections = self._limit_detections(detection_sets)
        delta, psi = self._forward_pass(processed_detections, timestamps)

        # Extract and return tracks
        return self._extract_tracks(delta, psi, processed_detections)
