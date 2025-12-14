"""Voxel-based updater for volumetric tracking.

This module provides updaters for voxel-based state representations,
updating occupancy probabilities using Bayesian inference with measurements.
"""

import numpy as np

from ..base import Property
from ..types.update import VoxelUpdate
from ..types.voxel import VoxelState
from .base import Updater


class VoxelUpdater(Updater):
    """Voxel-based Bayesian updater.

    Updates voxel occupancy probabilities using Bayesian inference. For each
    voxel, the posterior occupancy probability is computed based on:

    1. **Detection case**: When a measurement is received, the likelihood
       of the measurement given occupancy is computed per voxel using the
       measurement model, and the occupancy is updated via Bayes' rule.

    2. **Missed detection case**: When no measurement is received in an
       observed region, the occupancy probability decreases based on the
       detection probability.

    The update follows the Bayesian occupancy grid framework where each
    voxel's occupancy :math:`p(m_i|z_{1:k})` is updated independently:

    .. math::

        p(m_i|z_{1:k}) = \\frac{p(z_k|m_i) p(m_i|z_{1:k-1})}{p(z_k)}

    where :math:`m_i` is the occupancy of voxel :math:`i`, :math:`z_k` is
    the measurement at time :math:`k`, and :math:`p(z_k|m_i)` is the
    measurement likelihood given occupancy.

    For missed detections (null hypothesis), the update is:

    .. math::

        p(m_i|\\neg z_k) = \\frac{(1 - P_D) p(m_i|z_{1:k-1})}{1 - P_D p(m_i|z_{1:k-1})}

    where :math:`P_D` is the detection probability.

    Parameters
    ----------
    measurement_model : :class:`~.MeasurementModel`
        Measurement model used to compute likelihood of measurements.
    detection_probability : float, optional
        Probability of detecting an occupied voxel. Default is 0.9.
        Must be in range (0, 1].
    clutter_intensity : float, optional
        Spatial density of clutter measurements (false alarms) per unit volume.
        Default is None (no clutter). Must be non-negative.

    Example
    -------
    >>> from stonesoup.models.measurement.linear import LinearGaussian
    >>> from stonesoup.types.array import CovarianceMatrix
    >>> import numpy as np
    >>>
    >>> # Create measurement model
    >>> measurement_model = LinearGaussian(
    ...     ndim_state=3,
    ...     mapping=[0, 1, 2],
    ...     noise_covar=CovarianceMatrix(np.eye(3) * 0.1)
    ... )
    >>>
    >>> # Create updater
    >>> updater = VoxelUpdater(
    ...     measurement_model=measurement_model,
    ...     detection_probability=0.9,
    ...     clutter_intensity=1e-6
    ... )
    """

    detection_probability: float = Property(
        default=0.9,
        doc="Probability of detection for an occupied voxel. Must be in range (0, 1]. "
        "Higher values mean occupied voxels are more likely to generate measurements.",
    )
    clutter_intensity: float = Property(
        default=None,
        doc="Clutter spatial density (false alarms per unit volume). Default None means no "
        "clutter. Must be non-negative. Used to model spurious measurements.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.detection_probability <= 1:
            raise ValueError(
                f"detection_probability must be in range (0, 1], "
                f"got {self.detection_probability}"
            )
        if self.clutter_intensity is not None and self.clutter_intensity < 0:
            raise ValueError(
                f"clutter_intensity must be non-negative, " f"got {self.clutter_intensity}"
            )

    def predict_measurement(
        self, predicted_state, measurement_model=None, measurement_noise=True, **kwargs
    ):
        """Predict measurement from voxel state.

        For voxel states, measurement prediction is not typically computed
        in closed form. This method is included for compatibility with the
        Updater interface but may not be used in practice.

        Parameters
        ----------
        predicted_state : :class:`~.VoxelPrediction`
            The predicted voxel state.
        measurement_model : :class:`~.MeasurementModel`, optional
            The measurement model. If None, uses the updater's measurement model.
        measurement_noise : bool, optional
            Whether to include measurement noise. Default True.
        **kwargs
            Additional arguments passed to the measurement model.

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            Measurement prediction. For voxel states, this is typically not
            computed in closed form and may return None or a placeholder.

        Raises
        ------
        NotImplementedError
            Measurement prediction is not implemented for voxel updaters as it
            requires integration over the voxel grid.
        """
        raise NotImplementedError(
            "Measurement prediction is not implemented for voxel updaters. "
            "Voxel updates are performed directly without explicit measurement "
            "prediction."
        )

    def update(self, hypothesis, **kwargs):
        """Update voxel occupancy using measurement.

        Performs Bayesian update of voxel occupancy probabilities based on
        the received measurement (or lack thereof).

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            Hypothesis with predicted voxel state (:class:`~.VoxelPrediction`)
            and associated detection. If the detection is a
            :class:`~.MissedDetection`, performs missed detection update.
        **kwargs
            Additional arguments passed to measurement model likelihood
            computation.

        Returns
        -------
        : :class:`~.VoxelUpdate`
            Updated voxel state with posterior occupancy probabilities.

        Notes
        -----
        For computational efficiency, the update can be implemented in log-space
        to avoid numerical underflow with small probabilities.

        The measurement model should provide a method to compute the likelihood
        of the measurement at each voxel location.
        """
        from ..types.detection import MissedDetection

        predicted_state = hypothesis.prediction
        measurement = hypothesis.measurement

        # Check that predicted state is a VoxelState
        if not isinstance(predicted_state, VoxelState):
            raise TypeError(f"predicted_state must be a VoxelState, got {type(predicted_state)}")

        # Get measurement model
        measurement_model = self._check_measurement_model(
            measurement.measurement_model if hasattr(measurement, "measurement_model") else None
        )

        # Initialize new occupancy (will be updated)
        if predicted_state.is_sparse:
            new_occupancy = predicted_state.occupancy.copy()
        else:
            new_occupancy = predicted_state.occupancy.copy()

        # Handle missed detection case
        if isinstance(measurement, MissedDetection):
            # Missed detection: decrease occupancy probability
            # p(m|no detection) = (1 - P_D * p(m)) / (1 - P_D * p(m))
            # Simplified: p(m|no detection) ∝ (1 - P_D) * p(m)
            if predicted_state.is_sparse:
                for idx in new_occupancy:
                    prior_prob = new_occupancy[idx]
                    # Bayesian update for missed detection
                    posterior = (
                        (1 - self.detection_probability)
                        * prior_prob
                        / (1 - self.detection_probability * prior_prob)
                    )
                    new_occupancy[idx] = posterior
            else:
                # Dense array update
                prior_prob = new_occupancy
                posterior = (
                    (1 - self.detection_probability)
                    * prior_prob
                    / (1 - self.detection_probability * prior_prob)
                )
                new_occupancy = posterior

        else:
            # Detection received: update with measurement likelihood
            # For each voxel, compute likelihood of measurement given occupancy
            measurement_vector = measurement.state_vector

            # Iterate over voxels and compute likelihood
            if predicted_state.is_sparse:
                # Sparse update: only update occupied voxels
                for idx in list(new_occupancy.keys()):
                    voxel_center = predicted_state.grid.voxel_center(idx)
                    prior_prob = new_occupancy[idx]

                    # Compute likelihood p(z|m_i)
                    # Measurement model likelihood at this voxel location
                    likelihood = self._compute_likelihood(
                        measurement_vector, voxel_center, measurement_model, **kwargs
                    )

                    # Bayesian update: p(m|z) ∝ p(z|m) * p(m)
                    # Detection probability weighted likelihood
                    detection_likelihood = self.detection_probability * likelihood

                    # Clutter term (false alarm)
                    if self.clutter_intensity is not None:
                        clutter_term = self.clutter_intensity
                    else:
                        clutter_term = 0.0

                    # Posterior: p(m|z) = p(z|m) * p(m) / p(z)
                    # where p(z) = p(z|m) * p(m) + p(z|not m) * p(not m)
                    numerator = detection_likelihood * prior_prob
                    denominator = detection_likelihood * prior_prob + clutter_term * (
                        1 - prior_prob
                    )

                    posterior = numerator / denominator if denominator > 0 else prior_prob

                    # Clamp to valid probability range
                    posterior = np.clip(posterior, 0.0, 1.0)
                    new_occupancy[idx] = posterior

            else:
                # Dense update: update all voxels
                shape = predicted_state.grid.shape
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            idx = (i, j, k)
                            voxel_center = predicted_state.grid.voxel_center(idx)
                            prior_prob = new_occupancy[idx]

                            # Compute likelihood
                            likelihood = self._compute_likelihood(
                                measurement_vector, voxel_center, measurement_model, **kwargs
                            )

                            # Detection likelihood
                            detection_likelihood = self.detection_probability * likelihood

                            # Clutter term
                            if self.clutter_intensity is not None:
                                clutter_term = self.clutter_intensity
                            else:
                                clutter_term = 0.0

                            # Bayesian update
                            numerator = detection_likelihood * prior_prob
                            denominator = detection_likelihood * prior_prob + clutter_term * (
                                1 - prior_prob
                            )

                            posterior = numerator / denominator if denominator > 0 else prior_prob

                            # Clamp to valid range
                            posterior = np.clip(posterior, 0.0, 1.0)
                            new_occupancy[idx] = posterior

        # Create updated voxel state
        return VoxelUpdate(
            grid=predicted_state.grid,
            occupancy=new_occupancy,
            timestamp=measurement.timestamp,
            hypothesis=hypothesis,
        )

    def _compute_likelihood(self, measurement_vector, voxel_center, measurement_model, **kwargs):
        """Compute measurement likelihood at a voxel location.

        Computes the probability density of the measurement given that the
        target is at the voxel center.

        Parameters
        ----------
        measurement_vector : :class:`~.StateVector`
            The measurement vector.
        voxel_center : np.ndarray
            The 3D coordinates of the voxel center.
        measurement_model : :class:`~.MeasurementModel`
            The measurement model.
        **kwargs
            Additional arguments for the measurement model.

        Returns
        -------
        float
            Likelihood value (probability density).
        """
        from ..types.array import StateVector
        from ..types.state import GaussianState

        # Create a hypothetical state at the voxel center
        # This is a simplification: we assume the target is exactly at the voxel center
        # For more accurate likelihood, integration over the voxel volume could be performed
        hypothetical_state = GaussianState(
            state_vector=StateVector(voxel_center),
            covar=np.eye(3) * 1e-6,  # Very small covariance (point target)
        )

        # Compute likelihood using measurement model
        try:
            likelihood = measurement_model.pdf(measurement_vector, hypothetical_state, **kwargs)
        except Exception:
            # If PDF computation fails, return small likelihood
            likelihood = 1e-10

        return float(likelihood)
