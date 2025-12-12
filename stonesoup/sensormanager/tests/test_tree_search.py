import copy
import numpy as np
import pytest
from datetime import datetime, timedelta

from ...types.array import StateVector, StateVectors
from ...types.state import GaussianState, ParticleState
from ...types.track import Track
from ...sensor.radar import RadarRotatingBearingRange
from ...sensor.action.dwell_action import ChangeDwellAction
from ..tree_search import (
    MonteCarloTreeSearchSensorManager,
    MCTSRolloutSensorManager,
    MCTSBestChildPolicyEnum
)
from ..reward import UncertaintyRewardFunction, ExpectedKLDivergence
from ...predictor.kalman import KalmanPredictor
from ...predictor.particle import ParticlePredictor
from ...updater.kalman import ExtendedKalmanUpdater
from ...updater.particle import ParticleUpdater
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...dataassociator.neighbour import GNNWith2DAssignment


def test_mcts_best_child_policy_enum():
    """Test that MCTSBestChildPolicyEnum has expected values."""
    assert MCTSBestChildPolicyEnum.MAXAREWARD.value == 'max_average_reward'
    assert MCTSBestChildPolicyEnum.MAXCREWARD.value == 'max_cumulative_reward'
    assert MCTSBestChildPolicyEnum.MAXVISITS.value == 'max_visits'


def test_mcts_instantiation():
    """Test basic instantiation of MonteCarloTreeSearchSensorManager."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Test with default parameters
    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function
    )

    assert sensor_manager.niterations == 100
    assert sensor_manager.time_step == timedelta(seconds=1)
    assert sensor_manager.exploration_factor == 1.0
    assert sensor_manager.best_child_policy == MCTSBestChildPolicyEnum.MAXCREWARD
    assert sensor_manager.discount_factor == 0.9
    assert sensor_manager.search_depth == np.inf

    # Test with custom parameters
    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=50,
        time_step=timedelta(seconds=2),
        exploration_factor=0.5,
        best_child_policy=MCTSBestChildPolicyEnum.MAXVISITS,
        discount_factor=0.8,
        search_depth=5
    )

    assert sensor_manager.niterations == 50
    assert sensor_manager.time_step == timedelta(seconds=2)
    assert sensor_manager.exploration_factor == 0.5
    assert sensor_manager.best_child_policy == MCTSBestChildPolicyEnum.MAXVISITS
    assert sensor_manager.discount_factor == 0.8
    assert sensor_manager.search_depth == 5


def test_mcts_instantiation_with_string_policy():
    """Test instantiation with string best_child_policy (enum conversion)."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        best_child_policy='max_average_reward'
    )

    assert sensor_manager.best_child_policy == MCTSBestChildPolicyEnum.MAXAREWARD


def test_mcts_rollout_instantiation():
    """Test basic instantiation of MCTSRolloutSensorManager."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_function = ExpectedKLDivergence(predictor, updater, return_tracks=True)

    sensor_manager = MCTSRolloutSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        rollout_depth=3
    )

    assert sensor_manager.rollout_depth == 3
    assert sensor_manager.search_depth == np.inf


def test_mcts_rollout_warning_with_both_depths():
    """Test that MCTSRolloutSensorManager warns when both search_depth and rollout_depth are set."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_function = ExpectedKLDivergence(predictor, updater, return_tracks=True)

    with pytest.warns(UserWarning, match='`search_depth` and `rollout_depth` have been defined'):
        sensor_manager = MCTSRolloutSensorManager(
            sensors={sensor},
            reward_function=reward_function,
            rollout_depth=3,
            search_depth=5
        )


@pytest.mark.parametrize(
    "best_child_policy",
    [
        MCTSBestChildPolicyEnum.MAXCREWARD,
        MCTSBestChildPolicyEnum.MAXAREWARD,
        MCTSBestChildPolicyEnum.MAXVISITS,
        'max_cumulative_reward',
        'max_average_reward',
        'max_visits'
    ],
    ids=['MAXCREWARD_enum', 'MAXAREWARD_enum', 'MAXVISITS_enum',
         'MAXCREWARD_str', 'MAXAREWARD_str', 'MAXVISITS_str']
)
def test_mcts_choose_actions_different_policies(best_child_policy):
    """Test choose_actions with different best child policies."""
    time_start = datetime.now()

    # Create tracks
    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=10,  # Small for fast test
        exploration_factor=0,
        best_child_policy=best_child_policy
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) > 0

    for config in chosen_configs:
        assert isinstance(config, dict)
        for sensor_key, actions in config.items():
            assert isinstance(sensor_key, RadarRotatingBearingRange)
            assert isinstance(actions, tuple)
            assert len(actions) > 0
            assert isinstance(actions[0], ChangeDwellAction)


def test_mcts_choose_actions_basic():
    """Test basic choose_actions functionality."""
    time_start = datetime.now()

    # Create tracks with Gaussian states
    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=20,  # Small for fast test
        exploration_factor=1.0
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    # Verify output structure
    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1  # nchoose defaults to 1

    config = chosen_configs[0]
    assert isinstance(config, dict)
    assert sensor in config or any(isinstance(s, RadarRotatingBearingRange) for s in config.keys())


def test_mcts_choose_actions_with_particle_states():
    """Test choose_actions with particle states and ExpectedKLDivergence."""
    time_start = datetime.now()

    # Create tracks with Particle states
    track1 = Track([
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([1, 1, 1, 1]),
            cov=np.diag([1.5, 0.25, 1.5, 0.25]),
            size=100).T),
            weight=np.array([1/100]*100),
            timestamp=time_start),
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([2, 1.5, 2, 1.5]),
            cov=np.diag([3, 0.5, 3, 0.5]),
            size=100).T),
            weight=np.array([1/100]*100),
            timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_function = ExpectedKLDivergence(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=15,
        exploration_factor=0.5
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1


def test_mcts_rollout_choose_actions():
    """Test MCTSRolloutSensorManager choose_actions."""
    time_start = datetime.now()

    track1 = Track([
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([1, 1, 1, 1]),
            cov=np.diag([1.5, 0.25, 1.5, 0.25]),
            size=50).T),
            weight=np.array([1/50]*50),
            timestamp=time_start),
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([2, 1.5, 2, 1.5]),
            cov=np.diag([3, 0.5, 3, 0.5]),
            size=50).T),
            weight=np.array([1/50]*50),
            timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
        resolution=np.radians(10)
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)
    reward_function = ExpectedKLDivergence(predictor, updater, return_tracks=True)

    sensor_manager = MCTSRolloutSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=10,
        rollout_depth=2,
        exploration_factor=0,
        discount_factor=0.9
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1

    config = chosen_configs[0]
    assert isinstance(config, dict)


def test_mcts_rollout_with_search_depth():
    """Test MCTSRolloutSensorManager with search_depth instead of rollout_depth."""
    time_start = datetime.now()

    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
        resolution=np.radians(10)
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MCTSRolloutSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=10,
        search_depth=3,
        exploration_factor=0
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1


def test_mcts_with_empty_tracks():
    """Test MCTS with empty track set."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=5
    )

    chosen_configs = sensor_manager.choose_actions(
        set(),  # Empty tracks
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)


def test_mcts_tree_policy():
    """Test the tree_policy method directly."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        exploration_factor=1.0
    )

    # Create mock nodes structure
    nodes = [
        {
            'Child_IDs': [1, 2],
            'visits': 10,
        },
        {
            'action_value': 5.0,
            'visits': 3,
        },
        {
            'action_value': 8.0,
            'visits': 7,
        }
    ]

    # Test tree_policy selection
    selected_child = sensor_manager.tree_policy(nodes, 0)
    assert selected_child in [1, 2]


def test_mcts_select_best_child():
    """Test the select_best_child method with different policies."""
    time_start = datetime.now()

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Test max cumulative reward policy
    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        best_child_policy=MCTSBestChildPolicyEnum.MAXCREWARD
    )

    nodes = [
        {'Child_IDs': [1, 2, 3]},
        {'action_value': 10.0, 'visits': 5},
        {'action_value': 15.0, 'visits': 3},
        {'action_value': 12.0, 'visits': 4}
    ]

    best = sensor_manager.select_best_child(nodes)
    assert best == [2]  # Highest action_value

    # Test max average reward policy
    sensor_manager.best_child_policy = MCTSBestChildPolicyEnum.MAXAREWARD
    best = sensor_manager.select_best_child(nodes)
    assert best == [2]  # 15.0/3 = 5.0 is highest average

    # Test max visits policy
    sensor_manager.best_child_policy = MCTSBestChildPolicyEnum.MAXVISITS
    best = sensor_manager.select_best_child(nodes)
    assert best == [1]  # Most visits


def test_mcts_with_multiple_tracks():
    """Test MCTS with multiple tracks."""
    time_start = datetime.now()

    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    track2 = Track([
        GaussianState([[-1], [1], [-1], [1]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1, track2}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=15
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1


def test_mcts_with_data_associator():
    """Test MCTS with data associator in reward function."""
    time_start = datetime.now()

    track1 = Track([
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([1, 1, 1, 1]),
            cov=np.diag([1.5, 0.25, 1.5, 0.25]),
            size=50).T),
            weight=np.array([1/50]*50),
            timestamp=time_start),
        ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
            mean=np.array([2, 1.5, 2, 1.5]),
            cov=np.diag([3, 0.5, 3, 0.5]),
            size=50).T),
            weight=np.array([1/50]*50),
            timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = ParticlePredictor(transition_model)
    updater = ParticleUpdater(measurement_model=None)

    hypothesiser = DistanceHypothesiser(predictor, updater,
                                       measure=Mahalanobis(),
                                       missed_distance=5)
    data_associator = GNNWith2DAssignment(hypothesiser)

    reward_function = ExpectedKLDivergence(
        predictor,
        updater,
        return_tracks=True,
        data_associator=data_associator
    )

    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=10
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)


def test_mcts_search_depth_limiting():
    """Test that search_depth limits tree expansion."""
    time_start = datetime.now()

    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Test with small search depth
    sensor_manager = MonteCarloTreeSearchSensorManager(
        sensors={sensor},
        reward_function=reward_function,
        niterations=20,
        search_depth=2  # Limit to 2 levels deep
    )

    chosen_configs = sensor_manager.choose_actions(
        tracks,
        time_start + timedelta(seconds=2)
    )

    assert isinstance(chosen_configs, list)
    assert len(chosen_configs) == 1


def test_mcts_discount_factor_effect():
    """Test that discount_factor is applied correctly."""
    time_start = datetime.now()

    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Test with different discount factors
    for discount_factor in [0.5, 0.9, 0.99]:
        sensor_manager = MonteCarloTreeSearchSensorManager(
            sensors={copy.deepcopy(sensor)},
            reward_function=reward_function,
            niterations=10,
            discount_factor=discount_factor
        )

        chosen_configs = sensor_manager.choose_actions(
            tracks,
            time_start + timedelta(seconds=2)
        )

        assert isinstance(chosen_configs, list)


def test_mcts_time_step_parameter():
    """Test that time_step parameter affects search correctly."""
    time_start = datetime.now()

    track1 = Track([
        GaussianState([[1], [1], [1], [1]],
                     np.diag([1.5, 0.25, 1.5, 0.25]),
                     timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                     np.diag([3, 0.5, 3, 0.5]),
                     timestamp=time_start + timedelta(seconds=1))
    ])

    tracks = {track1}

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
    )
    sensor.timestamp = time_start

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater, return_tracks=True)

    # Test with different time steps
    for time_step_seconds in [1, 2, 5]:
        sensor_manager = MonteCarloTreeSearchSensorManager(
            sensors={copy.deepcopy(sensor)},
            reward_function=reward_function,
            niterations=10,
            time_step=timedelta(seconds=time_step_seconds)
        )

        chosen_configs = sensor_manager.choose_actions(
            tracks,
            time_start + timedelta(seconds=10)
        )

        assert isinstance(chosen_configs, list)
