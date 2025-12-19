from datetime import datetime, timedelta

import numpy as np
import pytest

from ...base import Property
from ...dataassociator.neighbour import GNNWith2DAssignment
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...predictor.kalman import KalmanPredictor
from ...predictor.particle import ParticlePredictor
from ...sensor.radar import RadarRotatingBearingRange
from ...types.array import StateVector, StateVectors
from ...types.state import GaussianState, ParticleState
from ...types.track import Track
from ...updater.kalman import ExtendedKalmanUpdater
from ...updater.particle import ParticleUpdater
from ..reward import (
    AdditiveRewardFunction,
    ExpectedKLDivergence,
    MultiplicativeRewardFunction,
    MultiUpdateExpectedKLDivergence,
    RewardFunction,
    UncertaintyRewardFunction,
)


class DummyRewardFunction(RewardFunction):
    """Simple reward function for testing."""

    score: float = Property(default=1.0, doc="Score to return")

    def __call__(self, config, tracks, metric_time, *args, **kwargs):
        return self.score


def test_reward_function_base_class():
    """Test that RewardFunction base class raises NotImplementedError."""
    reward_func = RewardFunction()
    with pytest.raises(NotImplementedError):
        reward_func({}, set(), datetime.now())


@pytest.mark.parametrize(
    "reward_function, score_list, weights, expected_output",
    [
        (DummyRewardFunction, [1, 2], None, 3),
        (DummyRewardFunction, [0, -2], None, -2),
        (DummyRewardFunction, [0, -2], [0.5, 0.5], -1),
        (DummyRewardFunction, [3, 2], [0.4, 0.6], 2.4),
        (DummyRewardFunction, [3, 2, 1], [0.5, 0.4, 0.1], 2.4),
        (DummyRewardFunction, [5.5, 4.5], None, 10.0),
        (DummyRewardFunction, [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], 1.0),
    ],
    ids=[
        "simple_sum",
        "negative",
        "weighted_half",
        "weighted_custom",
        "three_values",
        "float_values",
        "equal_weights",
    ],
)
def test_additive_reward_function(reward_function, score_list, weights, expected_output):
    """Test AdditiveRewardFunction with various inputs."""
    additive = AdditiveRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list],
        weights=weights,
    )
    result = additive(config=None, tracks=None, metric_time=None)
    assert np.allclose(result, expected_output)


@pytest.mark.parametrize(
    "reward_function, score_list, weights, expected_output",
    [
        (DummyRewardFunction, [1, 2], None, 2),
        (DummyRewardFunction, [2, 3], [1, 1], 6),
        (DummyRewardFunction, [-2, 5], [0.5, 0.5], -2.5),
        (DummyRewardFunction, [3, 4], [2, 1], 24),
        (DummyRewardFunction, [2, 2, 2], None, 8),
    ],
    ids=["simple_product", "with_weights", "negative_values", "different_weights", "three_values"],
)
def test_multiplicative_reward_function(reward_function, score_list, weights, expected_output):
    """Test MultiplicativeRewardFunction with various inputs."""
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[reward_function(score=score) for score in score_list],
        weights=weights,
    )
    result = multiplicative(config=None, tracks=None, metric_time=None)
    assert np.allclose(result, expected_output)


def test_additive_reward_function_unequal_weights():
    """Test AdditiveRewardFunction raises error with unequal weights."""
    additive = AdditiveRewardFunction(
        reward_function_list=[DummyRewardFunction(score=1), DummyRewardFunction(score=2)],
        weights=[1, 2, 3],
    )
    with pytest.raises(IndexError):
        additive(config=None, tracks=None, metric_time=None)


def test_multiplicative_reward_function_unequal_weights():
    """Test MultiplicativeRewardFunction raises error with unequal weights."""
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[DummyRewardFunction(score=1), DummyRewardFunction(score=2)],
        weights=[1, 2, 3],
    )
    with pytest.raises(IndexError):
        multiplicative(config=None, tracks=None, metric_time=None)


def test_additive_reward_function_default_weights():
    """Test AdditiveRewardFunction uses default weights of 1."""
    additive = AdditiveRewardFunction(
        reward_function_list=[
            DummyRewardFunction(score=2),
            DummyRewardFunction(score=3),
            DummyRewardFunction(score=5),
        ]
    )
    result = additive(config=None, tracks=None, metric_time=None)
    assert np.allclose(result, 10.0)


def test_multiplicative_reward_function_default_weights():
    """Test MultiplicativeRewardFunction uses default weights of 1."""
    multiplicative = MultiplicativeRewardFunction(
        reward_function_list=[
            DummyRewardFunction(score=2),
            DummyRewardFunction(score=3),
            DummyRewardFunction(score=4),
        ]
    )
    result = multiplicative(config=None, tracks=None, metric_time=None)
    assert np.allclose(result, 24.0)


def test_uncertainty_reward_function_instantiation():
    """Test UncertaintyRewardFunction instantiation."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)

    # Test default parameters
    reward_func = UncertaintyRewardFunction(predictor, updater)
    assert reward_func.predictor == predictor
    assert reward_func.updater == updater
    assert reward_func.method_sum is True
    assert reward_func.return_tracks is False
    assert reward_func.measurement_noise is False

    # Test custom parameters
    reward_func = UncertaintyRewardFunction(
        predictor, updater, method_sum=False, return_tracks=True, measurement_noise=True
    )
    assert reward_func.method_sum is False
    assert reward_func.return_tracks is True
    assert reward_func.measurement_noise is True


def test_uncertainty_reward_function_call():
    """Test UncertaintyRewardFunction calculation."""
    time_start = datetime.now()

    # Create track
    track = Track(
        [
            GaussianState(
                [[1], [1], [1], [1]], np.diag([1.5, 0.25, 1.5, 0.25]), timestamp=time_start
            ),
            GaussianState(
                [[2], [1.5], [2], [1.5]],
                np.diag([3, 0.5, 3, 0.5]),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    # Create sensor
    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    # Create reward function
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_func = UncertaintyRewardFunction(predictor, updater)

    # Create config
    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    # Calculate reward
    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))

    assert isinstance(reward, (int, float))
    assert reward >= 0  # Uncertainty reduction should be non-negative


def test_uncertainty_reward_function_with_return_tracks():
    """Test UncertaintyRewardFunction returns tracks when requested."""
    time_start = datetime.now()

    track = Track(
        [
            GaussianState(
                [[1], [1], [1], [1]], np.diag([1.5, 0.25, 1.5, 0.25]), timestamp=time_start
            ),
            GaussianState(
                [[2], [1.5], [2], [1.5]],
                np.diag([3, 0.5, 3, 0.5]),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_func = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    result = reward_func(config, tracks, time_start + timedelta(seconds=2))

    assert isinstance(result, tuple)
    assert len(result) == 2
    reward, returned_tracks = result
    assert isinstance(reward, (int, float))
    assert isinstance(returned_tracks, set)


def test_uncertainty_reward_function_method_sum_false():
    """Test UncertaintyRewardFunction with method_sum=False (mean calculation)."""
    time_start = datetime.now()

    # Create multiple tracks
    track1 = Track(
        [
            GaussianState(
                [[1], [1], [1], [1]], np.diag([1.5, 0.25, 1.5, 0.25]), timestamp=time_start
            ),
            GaussianState(
                [[2], [1.5], [2], [1.5]],
                np.diag([3, 0.5, 3, 0.5]),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )

    track2 = Track(
        [
            GaussianState([[-1], [1], [-1], [1]], np.diag([3, 0.5, 3, 0.5]), timestamp=time_start),
            GaussianState(
                [[0], [1.5], [0], [1.5]],
                np.diag([1.5, 0.25, 1.5, 0.25]),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )

    tracks = {track1, track2}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_func = UncertaintyRewardFunction(predictor, updater, method_sum=False)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))


def test_uncertainty_reward_function_with_measurement_noise():
    """Test UncertaintyRewardFunction with measurement noise."""
    time_start = datetime.now()

    track = Track(
        [
            GaussianState(
                [[1], [1], [1], [1]], np.diag([1.5, 0.25, 1.5, 0.25]), timestamp=time_start
            ),
            GaussianState(
                [[2], [1.5], [2], [1.5]],
                np.diag([3, 0.5, 3, 0.5]),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_func = UncertaintyRewardFunction(predictor, updater, measurement_noise=True)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))


def test_expected_kld_instantiation():
    """Test ExpectedKLDivergence instantiation."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    # Test default parameters
    reward_func = ExpectedKLDivergence(predictor, updater)
    assert reward_func.predictor == predictor
    assert reward_func.updater == updater
    assert reward_func.method_sum is True
    assert reward_func.data_associator is None
    assert reward_func.return_tracks is False
    assert reward_func.measurement_noise is False

    # Test custom parameters
    reward_func = ExpectedKLDivergence(
        predictor, updater, method_sum=False, return_tracks=True, measurement_noise=True
    )
    assert reward_func.method_sum is False
    assert reward_func.return_tracks is True
    assert reward_func.measurement_noise is True


def test_expected_kld_with_none_predictor():
    """Test ExpectedKLDivergence with None predictor."""
    updater = ParticleUpdater(measurement_model=None)
    reward_func = ExpectedKLDivergence(predictor=None, updater=updater)
    assert reward_func.predictor is None


def test_expected_kld_call():
    """Test ExpectedKLDivergence calculation."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = ExpectedKLDivergence(predictor, updater)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))
    assert reward >= 0


def test_expected_kld_with_data_associator():
    """Test ExpectedKLDivergence with data associator."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=5
    )
    data_associator = GNNWith2DAssignment(hypothesiser)

    reward_func = ExpectedKLDivergence(predictor, updater, data_associator=data_associator)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))


def test_expected_kld_with_return_tracks():
    """Test ExpectedKLDivergence returns tracks when requested."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = ExpectedKLDivergence(predictor, updater, return_tracks=True)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    result = reward_func(config, tracks, time_start + timedelta(seconds=2))

    assert isinstance(result, tuple)
    assert len(result) == 2
    reward, returned_tracks = result
    assert isinstance(reward, (int, float))
    assert isinstance(returned_tracks, set)


def test_multi_update_expected_kld_instantiation():
    """Test MultiUpdateExpectedKLDivergence instantiation."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    # Test default parameters
    reward_func = MultiUpdateExpectedKLDivergence(predictor, updater)
    assert reward_func.predictor == predictor
    assert reward_func.updater == updater
    assert reward_func.updates_per_track == 2
    assert reward_func.measurement_noise is True

    # Test custom parameters
    reward_func = MultiUpdateExpectedKLDivergence(predictor, updater, updates_per_track=5)
    assert reward_func.updates_per_track == 5


def test_multi_update_expected_kld_raises_with_wrong_predictor():
    """Test MultiUpdateExpectedKLDivergence raises error with non-ParticlePredictor."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    with pytest.raises(
        NotImplementedError, match="Only ParticlePredictor types are currently compatible"
    ):
        MultiUpdateExpectedKLDivergence(predictor, updater)


def test_multi_update_expected_kld_raises_with_wrong_updater():
    """Test MultiUpdateExpectedKLDivergence raises error with non-ParticleUpdater."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)

    with pytest.raises(
        NotImplementedError, match="Only ParticleUpdater types are currently compatible"
    ):
        MultiUpdateExpectedKLDivergence(predictor, updater)


def test_multi_update_expected_kld_raises_with_low_updates():
    """Test MultiUpdateExpectedKLDivergence raises error with updates_per_track < 2."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    with pytest.raises(ValueError, match="updates_per_track = 1"):
        MultiUpdateExpectedKLDivergence(predictor, updater, updates_per_track=1)

    with pytest.raises(ValueError, match="updates_per_track = 0"):
        MultiUpdateExpectedKLDivergence(predictor, updater, updates_per_track=0)


def test_multi_update_expected_kld_call():
    """Test MultiUpdateExpectedKLDivergence calculation."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=100
                    ).T
                ),
                weight=np.array([1 / 100] * 100),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = MultiUpdateExpectedKLDivergence(predictor, updater, updates_per_track=3)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))
    assert reward >= 0


def test_multi_update_expected_kld_method_sum_false():
    """Test MultiUpdateExpectedKLDivergence with method_sum=False."""
    time_start = datetime.now()

    track1 = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )

    track2 = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([-1, 1, -1, 1]), cov=np.diag([3, 0.5, 3, 0.5]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([0, 1.5, 0, 1.5]),
                        cov=np.diag([1.5, 0.25, 1.5, 0.25]),
                        size=50,
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )

    tracks = {track1, track2}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = MultiUpdateExpectedKLDivergence(
        predictor, updater, method_sum=False, updates_per_track=2
    )

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))


def test_expected_kld_with_empty_tracks():
    """Test ExpectedKLDivergence with empty tracks."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = ExpectedKLDivergence(predictor, updater)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, set(), time_start + timedelta(seconds=2))
    assert reward == 0.0


def test_uncertainty_reward_with_empty_tracks():
    """Test UncertaintyRewardFunction with empty tracks."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_func = UncertaintyRewardFunction(predictor, updater)

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, set(), time_start + timedelta(seconds=2))
    assert reward == 0


def test_expected_kld_with_multiple_sensors():
    """Test ExpectedKLDivergence with multiple sensors."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor1 = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor1.timestamp = time_start

    sensor2 = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[10], [10]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([np.radians(180)]),
        max_range=np.inf,
    )
    sensor2.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = ExpectedKLDivergence(predictor, updater)

    # Get action generators and create a config with one action from each generator
    action_generators1 = sensor1.actions(time_start + timedelta(seconds=2))
    actions1 = tuple(next(iter(gen)) for gen in action_generators1)
    action_generators2 = sensor2.actions(time_start + timedelta(seconds=2))
    actions2 = tuple(next(iter(gen)) for gen in action_generators2)
    config = {sensor1: actions1, sensor2: actions2}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))


def test_combined_reward_functions():
    """Test combining multiple reward functions using Additive and Multiplicative."""
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    KalmanPredictor(transition_model)
    ExtendedKalmanUpdater(measurement_model=None)

    reward1 = DummyRewardFunction(score=5)
    reward2 = DummyRewardFunction(score=3)

    # Test combining with additive
    combined_additive = AdditiveRewardFunction(
        reward_function_list=[reward1, reward2], weights=[0.6, 0.4]
    )
    result = combined_additive(None, None, None)
    assert np.allclose(result, 4.2)  # 5*0.6 + 3*0.4 = 4.2

    # Test combining with multiplicative
    combined_mult = MultiplicativeRewardFunction(
        reward_function_list=[reward1, reward2], weights=[1, 1]
    )
    result = combined_mult(None, None, None)
    assert np.allclose(result, 15)  # 5*1 * 3*1 = 15


@pytest.mark.parametrize(
    "updates_per_track", [2, 3, 5, 10], ids=["2_updates", "3_updates", "5_updates", "10_updates"]
)
def test_multi_update_expected_kld_various_updates(updates_per_track):
    """Test MultiUpdateExpectedKLDivergence with various update counts."""
    time_start = datetime.now()

    track = Track(
        [
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([1, 1, 1, 1]), cov=np.diag([1.5, 0.25, 1.5, 0.25]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start,
            ),
            ParticleState(
                state_vector=StateVectors(
                    np.random.multivariate_normal(
                        mean=np.array([2, 1.5, 2, 1.5]), cov=np.diag([3, 0.5, 3, 0.5]), size=50
                    ).T
                ),
                weight=np.array([1 / 50] * 50),
                timestamp=time_start + timedelta(seconds=1),
            ),
        ]
    )
    tracks = {track}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0], [0, 0.75**2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_func = MultiUpdateExpectedKLDivergence(
        predictor, updater, updates_per_track=updates_per_track
    )

    # Get action generators and create a config with one action from each generator
    action_generators = sensor.actions(time_start + timedelta(seconds=2))
    actions = tuple(next(iter(gen)) for gen in action_generators)
    config = {sensor: actions}

    reward = reward_func(config, tracks, time_start + timedelta(seconds=2))
    assert isinstance(reward, (int, float))
    assert reward >= 0
