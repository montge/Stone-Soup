"""Voxel-based predictor implementations for volumetric tracking."""
import numpy as np
from scipy.ndimage import convolve

from .base import Predictor
from ._utils import predict_lru_cache
from ..base import Property
from ..models.transition import TransitionModel
from ..types.prediction import VoxelPrediction
from ..types.voxel import VoxelState


class VoxelPredictor(Predictor):
    """Voxel Predictor class

    A predictor for voxel-based state representations. Propagates voxel occupancy
    probabilities through a transition model while accounting for birth and death
    processes.

    This predictor is designed for volumetric tracking and 3D occupancy mapping,
    where the state is represented as a grid of voxels with associated occupancy
    probabilities. The prediction step involves:

    1. Applying the transition model to spread occupancy across adjacent voxels
    2. Modeling spontaneous birth of new occupied voxels
    3. Modeling death/disappearance of existing occupied voxels

    The transition model should define how occupancy probabilities propagate through
    the voxel grid based on expected motion or diffusion patterns.

    Parameters
    ----------
    transition_model : TransitionModel
        The transition model defining how voxel occupancy propagates. This model
        should be compatible with voxel grid structures and implement the standard
        :meth:`function` method to transform occupancy distributions.
    birth_probability : float, optional
        Probability of spontaneous birth in empty voxels per prediction step.
        Default is 0.01. Should be in range [0, 1].
    death_probability : float, optional
        Probability of death/disappearance in occupied voxels per prediction step.
        Default is 0.01. Should be in range [0, 1].

    Example
    -------
    >>> from datetime import datetime, timedelta
    >>> from stonesoup.types.voxel import VoxelGrid, VoxelState
    >>> # Create a voxel grid
    >>> grid = VoxelGrid(
    ...     bounds=np.array([0, 10, 0, 10, 0, 10]),
    ...     resolution=1.0
    ... )
    >>> # Create initial state with occupancy
    >>> occupancy = np.zeros((10, 10, 10))
    >>> occupancy[5, 5, 5] = 0.9  # High probability at center
    >>> prior = VoxelState(
    ...     grid=grid,
    ...     occupancy=occupancy,
    ...     timestamp=datetime.now()
    ... )
    >>> # Create predictor (transition_model should be defined)
    >>> predictor = VoxelPredictor(
    ...     transition_model=transition_model,
    ...     birth_probability=0.01,
    ...     death_probability=0.01
    ... )
    >>> # Predict forward in time
    >>> prediction = predictor.predict(
    ...     prior,
    ...     timestamp=datetime.now() + timedelta(seconds=1)
    ... )
    >>> # Prediction contains updated occupancy probabilities
    >>> isinstance(prediction, VoxelPrediction)
    True

    Notes
    -----
    The birth-death process modifies voxel occupancy probabilities as follows:

    .. math::

        p_{predicted}(v) = (1 - p_{death}) \\cdot p_{transition}(v) + p_{birth} \\cdot (1 - p_{transition}(v))

    where :math:`p_{transition}(v)` is the occupancy probability after applying
    the transition model to voxel :math:`v`.

    For sparse voxel representations, the predictor efficiently handles only
    non-zero occupancy voxels, improving computational performance for large grids.
    """

    transition_model: TransitionModel = Property(
        doc="Transition model defining how voxel occupancy propagates through the grid."
    )
    birth_probability: float = Property(
        default=0.01,
        doc="Probability of spontaneous birth in empty voxels. Must be in [0, 1]. Default 0.01."
    )
    death_probability: float = Property(
        default=0.01,
        doc="Probability of death in occupied voxels. Must be in [0, 1]. Default 0.01."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate probability bounds
        if not 0 <= self.birth_probability <= 1:
            raise ValueError(
                f"birth_probability must be in [0, 1], got {self.birth_probability}")
        if not 0 <= self.death_probability <= 1:
            raise ValueError(
                f"death_probability must be in [0, 1], got {self.death_probability}")

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """Voxel prediction step

        Propagates voxel occupancy probabilities forward in time using the transition
        model and birth-death process.

        Parameters
        ----------
        prior : VoxelState
            Prior voxel state with occupancy probabilities
        timestamp : datetime.datetime, optional
            Time at which the prediction is made. Used to compute time_interval
            for the transition model. Default None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the transition model's
            :meth:`function` method (e.g., control inputs, noise parameters)

        Returns
        -------
        VoxelPrediction
            Predicted voxel state with updated occupancy probabilities and the
            same grid structure as the prior

        Raises
        ------
        TypeError
            If prior is not a VoxelState instance

        Example
        -------
        >>> # After creating predictor and prior state
        >>> prediction = predictor.predict(
        ...     prior,
        ...     timestamp=datetime.now() + timedelta(seconds=1)
        ... )
        >>> # Access predicted occupancy
        >>> prob_at_center = prediction.probability_at(np.array([5, 5, 5]))

        Notes
        -----
        The prediction process:

        1. Computes time_interval from prior.timestamp and timestamp
        2. Applies transition model to propagate occupancy
        3. Applies death probability to reduce existing occupancy
        4. Applies birth probability to add new occupancy in empty voxels
        5. Clamps all probabilities to [0, 1] range
        """
        if not isinstance(prior, VoxelState):
            raise TypeError(
                f"prior must be a VoxelState, got {type(prior)}")

        # Compute time interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # timestamp or prior.timestamp is None
            time_interval = None

        # Apply transition model to propagate occupancy
        # The transition model function should handle VoxelState appropriately
        transitioned_state = self.transition_model.function(
            prior, time_interval=time_interval, **kwargs)

        # Get occupancy from transitioned state
        if isinstance(transitioned_state, VoxelState):
            predicted_occupancy = transitioned_state.occupancy
        else:
            # If transition model returns occupancy array directly
            predicted_occupancy = transitioned_state

        # Convert to dense if sparse for easier manipulation
        was_sparse = prior.is_sparse
        if isinstance(predicted_occupancy, dict):
            # Convert sparse to dense for processing
            dense_occupancy = np.zeros(prior.grid.shape)
            for idx, prob in predicted_occupancy.items():
                dense_occupancy[idx] = prob
            predicted_occupancy = dense_occupancy

        # Ensure we have a numpy array
        if not isinstance(predicted_occupancy, np.ndarray):
            predicted_occupancy = np.array(predicted_occupancy)

        # Apply birth-death process
        # Death: reduce occupancy in occupied voxels
        # Birth: add occupancy in empty voxels
        # p_new = (1 - p_death) * p_old + p_birth * (1 - p_old)
        predicted_occupancy = (
            (1 - self.death_probability) * predicted_occupancy +
            self.birth_probability * (1 - predicted_occupancy)
        )

        # Clamp probabilities to [0, 1]
        predicted_occupancy = np.clip(predicted_occupancy, 0.0, 1.0)

        # Convert back to sparse if original was sparse
        if was_sparse:
            # Convert to sparse representation
            sparse_occupancy = {}
            shape = prior.grid.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        prob = predicted_occupancy[i, j, k]
                        if prob > 1e-6:  # Threshold for sparsity
                            sparse_occupancy[(i, j, k)] = float(prob)
            predicted_occupancy = sparse_occupancy

        # Create prediction using VoxelPrediction
        return VoxelPrediction(
            grid=prior.grid,
            occupancy=predicted_occupancy,
            timestamp=timestamp,
            transition_model=self.transition_model,
            prior=prior
        )


class DiffusionVoxelPredictor(VoxelPredictor):
    """Diffusion-based Voxel Predictor

    A specialized voxel predictor that models occupancy propagation as a diffusion
    process, where occupancy spreads to neighboring voxels over time.

    This predictor uses a convolution-based approach to model diffusion, making it
    efficient for uniform grids. The diffusion is controlled by a diffusion coefficient
    that determines how quickly occupancy spreads to adjacent voxels.

    Parameters
    ----------
    diffusion_coefficient : float, optional
        Controls the rate of diffusion to neighboring voxels. Higher values result
        in faster spreading. Default is 0.1. Should be in range [0, 1].
    birth_probability : float, optional
        Probability of spontaneous birth in empty voxels. Default is 0.01.
    death_probability : float, optional
        Probability of death in occupied voxels. Default is 0.01.

    Example
    -------
    >>> from datetime import datetime, timedelta
    >>> from stonesoup.types.voxel import VoxelGrid, VoxelState
    >>> # Create voxel grid
    >>> grid = VoxelGrid(
    ...     bounds=np.array([0, 10, 0, 10, 0, 10]),
    ...     resolution=1.0
    ... )
    >>> # Create state with localized occupancy
    >>> occupancy = np.zeros((10, 10, 10))
    >>> occupancy[5, 5, 5] = 1.0
    >>> prior = VoxelState(grid=grid, occupancy=occupancy, timestamp=datetime.now())
    >>> # Create diffusion predictor
    >>> predictor = DiffusionVoxelPredictor(
    ...     diffusion_coefficient=0.2,
    ...     birth_probability=0.01,
    ...     death_probability=0.01
    ... )
    >>> # Predict - occupancy will diffuse to neighbors
    >>> prediction = predictor.predict(
    ...     prior,
    ...     timestamp=datetime.now() + timedelta(seconds=1)
    ... )

    Notes
    -----
    The diffusion is modeled using a 3D convolution with a kernel that spreads
    occupancy to the 6 face-adjacent neighbors (not including diagonal neighbors).

    The transition model property is not used in this predictor as the diffusion
    process is handled internally. To use this predictor, set transition_model=None
    or provide a dummy transition model.
    """

    diffusion_coefficient: float = Property(
        default=0.1,
        doc="Diffusion rate controlling spread to neighboring voxels. Must be in [0, 1]. "
            "Default 0.1."
    )

    # Override transition_model to be optional
    transition_model: TransitionModel = Property(
        default=None,
        doc="Transition model (not used for diffusion predictor, can be None)."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= self.diffusion_coefficient <= 1:
            raise ValueError(
                f"diffusion_coefficient must be in [0, 1], got {self.diffusion_coefficient}")

        # Create diffusion kernel: 3x3x3 with center and 6-connected neighbors
        # Center gets (1 - diffusion), each of 6 neighbors gets diffusion/6
        self.diffusion_kernel = np.zeros((3, 3, 3))
        # Center voxel
        self.diffusion_kernel[1, 1, 1] = 1 - self.diffusion_coefficient
        # 6-connected neighbors (face-adjacent)
        self.diffusion_kernel[0, 1, 1] = self.diffusion_coefficient / 6  # -x
        self.diffusion_kernel[2, 1, 1] = self.diffusion_coefficient / 6  # +x
        self.diffusion_kernel[1, 0, 1] = self.diffusion_coefficient / 6  # -y
        self.diffusion_kernel[1, 2, 1] = self.diffusion_coefficient / 6  # +y
        self.diffusion_kernel[1, 1, 0] = self.diffusion_coefficient / 6  # -z
        self.diffusion_kernel[1, 1, 2] = self.diffusion_coefficient / 6  # +z

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """Diffusion-based voxel prediction

        Propagates occupancy through diffusion to neighboring voxels, then applies
        birth-death process.

        Parameters
        ----------
        prior : VoxelState
            Prior voxel state with occupancy probabilities
        timestamp : datetime.datetime, optional
            Time at which the prediction is made. Default None.
        **kwargs : dict, optional
            Additional keyword arguments (unused in diffusion predictor)

        Returns
        -------
        VoxelPrediction
            Predicted voxel state after diffusion and birth-death process
        """
        if not isinstance(prior, VoxelState):
            raise TypeError(f"prior must be a VoxelState, got {type(prior)}")

        # Convert sparse to dense for diffusion
        was_sparse = prior.is_sparse
        if was_sparse:
            occupancy = np.zeros(prior.grid.shape)
            for idx, prob in prior.occupancy.items():
                occupancy[idx] = prob
        else:
            occupancy = prior.occupancy.copy()

        # Apply diffusion via convolution
        # mode='constant' treats boundary voxels as having 0 occupancy outside grid
        diffused_occupancy = convolve(
            occupancy,
            self.diffusion_kernel,
            mode='constant',
            cval=0.0
        )

        # Apply birth-death process
        predicted_occupancy = (
            (1 - self.death_probability) * diffused_occupancy +
            self.birth_probability * (1 - diffused_occupancy)
        )

        # Clamp to [0, 1]
        predicted_occupancy = np.clip(predicted_occupancy, 0.0, 1.0)

        # Convert back to sparse if needed
        if was_sparse:
            sparse_occupancy = {}
            shape = prior.grid.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        prob = predicted_occupancy[i, j, k]
                        if prob > 1e-6:
                            sparse_occupancy[(i, j, k)] = float(prob)
            predicted_occupancy = sparse_occupancy

        return VoxelPrediction(
            grid=prior.grid,
            occupancy=predicted_occupancy,
            timestamp=timestamp,
            transition_model=None,  # No explicit transition model used
            prior=prior
        )
