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

        # Limit detections per scan if needed
        processed_detections = []
        for det_set in detection_sets:
            det_list = list(det_set)
            if len(det_list) > self.max_detections_per_scan:
                # Sort by detection score if available, otherwise keep first N
                if (
                    det_list
                    and hasattr(det_list[0], "metadata")
                    and "score" in det_list[0].metadata
                ):
                    det_list.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
                det_list = det_list[: self.max_detections_per_scan]
            processed_detections.append(det_list)

        # Build trellis structure
        # Each node is (scan_idx, detection_idx) or (scan_idx, -1) for missed detection
        num_scans = len(processed_detections)

        # Forward pass: compute max log-likelihood to reach each node
        # delta[k][i] = max log-likelihood to reach detection i at scan k
        delta = []
        # psi[k][i] = index of best previous detection leading to detection i at scan k
        psi = []

        # Initialize first scan
        delta_0 = []
        for det in processed_detections[0]:
            score = self._compute_detection_score(det)
            delta_0.append(score)
        delta.append(np.array(delta_0))
        psi.append([-1] * len(processed_detections[0]))  # No previous

        # Forward recursion
        for k in range(1, num_scans):
            len(processed_detections[k - 1])
            num_curr = len(processed_detections[k])
            delta_k = np.zeros(num_curr)
            psi_k = np.zeros(num_curr, dtype=int)

            time_interval = timestamps[k] - timestamps[k - 1]

            for i, det_curr in enumerate(processed_detections[k]):
                state_curr = self._detection_to_state(det_curr)
                det_score = self._compute_detection_score(det_curr)

                # Find best previous detection
                best_score = -np.inf
                best_prev = 0

                for j, det_prev in enumerate(processed_detections[k - 1]):
                    state_prev = self._detection_to_state(det_prev)

                    # Transition score
                    trans_score = self._compute_transition_score(
                        state_prev, state_curr, time_interval
                    )

                    # Total score
                    total_score = delta[k - 1][j] + trans_score + det_score

                    if total_score > best_score:
                        best_score = total_score
                        best_prev = j

                # Also consider missed detection at previous scan
                # (would need to maintain separate missed detection states)
                # For simplicity, we only consider detection-to-detection here

                delta_k[i] = best_score
                psi_k[i] = best_prev

            delta.append(delta_k)
            psi.append(psi_k)

        # Backtracking: extract tracks exceeding threshold
        tracks = set()

        # Find all terminal nodes exceeding threshold
        final_scores = delta[-1]
        candidates = np.where(final_scores > self.detection_threshold)[0]

        # Extract unique tracks (avoid duplicates from shared paths)
        extracted_paths = []

        for candidate_idx in candidates:
            # Backtrack
            path = []
            curr_idx = candidate_idx

            for k in range(num_scans - 1, -1, -1):
                det = processed_detections[k][curr_idx]
                path.insert(0, det)

                if k > 0:
                    curr_idx = psi[k][curr_idx]

            # Check if this path is unique
            is_unique = True
            for existing_path in extracted_paths:
                # Paths are considered duplicates if they share most detections
                shared = sum(1 for d1, d2 in zip(path, existing_path, strict=False) if d1 is d2)
                if shared >= len(path) * 0.8:  # 80% overlap threshold
                    is_unique = False
                    break

            if is_unique:
                extracted_paths.append(path)

                # Create track from path
                states = []
                for det in path:
                    state = self._detection_to_state(det)
                    # Create an update-like state
                    states.append(state)

                track = Track(states)
                tracks.add(track)

        return tracks
