"""Tests for voxel-based predictors."""
import datetime
import numpy as np
import pytest

from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.predictor.voxel import VoxelPredictor, DiffusionVoxelPredictor
from stonesoup.types.prediction import VoxelPrediction
from stonesoup.types.voxel import VoxelGrid, VoxelState


# Fixtures for common test data
@pytest.fixture
def voxel_grid():
    """Create a simple 10x10x10 voxel grid for testing."""
    return VoxelGrid(
        bounds=np.array([0, 10, 0, 10, 0, 10]),
        resolution=1.0
    )


@pytest.fixture
def small_voxel_grid():
    """Create a smaller 5x5x5 voxel grid for faster testing."""
    return VoxelGrid(
        bounds=np.array([0, 5, 0, 5, 0, 5]),
        resolution=1.0
    )


@pytest.fixture
def dense_voxel_state(voxel_grid):
    """Create a voxel state with dense occupancy representation."""
    occupancy = np.zeros((10, 10, 10))
    # Set some voxels as occupied
    occupancy[5, 5, 5] = 0.9
    occupancy[5, 5, 6] = 0.7
    occupancy[5, 6, 5] = 0.6

    timestamp = datetime.datetime(2023, 1, 1, 12, 0, 0)
    return VoxelState(grid=voxel_grid, occupancy=occupancy, timestamp=timestamp)


@pytest.fixture
def sparse_voxel_state(voxel_grid):
    """Create a voxel state with sparse occupancy representation."""
    occupancy = {
        (5, 5, 5): 0.9,
        (5, 5, 6): 0.7,
        (5, 6, 5): 0.6,
        (6, 5, 5): 0.5
    }

    timestamp = datetime.datetime(2023, 1, 1, 12, 0, 0)
    return VoxelState(grid=voxel_grid, occupancy=occupancy, timestamp=timestamp)


@pytest.fixture
def simple_transition_model():
    """Create a simple transition model for voxel testing.

    Note: This is a placeholder - actual voxel transition models would need
    to handle VoxelState objects appropriately.
    """
    # For now, we create a basic transition model
    # In reality, voxel transition models need custom implementations
    return ConstantVelocity(noise_diff_coeff=0.1)


class MockVoxelTransitionModel:
    """Mock transition model that returns the input state unchanged.

    This is used for testing the VoxelPredictor's birth-death process
    without complex transition dynamics.
    """

    def function(self, state, time_interval=None, **kwargs):
        """Return the state's occupancy unchanged."""
        return state.occupancy


# Test VoxelPredictor instantiation
def test_voxel_predictor_instantiation(simple_transition_model):
    """Test creating a VoxelPredictor with valid parameters."""
    predictor = VoxelPredictor(
        transition_model=simple_transition_model,
        birth_probability=0.01,
        death_probability=0.02
    )

    assert predictor.transition_model == simple_transition_model
    assert predictor.birth_probability == 0.01
    assert predictor.death_probability == 0.02


def test_voxel_predictor_default_probabilities(simple_transition_model):
    """Test VoxelPredictor uses correct default values."""
    predictor = VoxelPredictor(transition_model=simple_transition_model)

    assert predictor.birth_probability == 0.01
    assert predictor.death_probability == 0.01


def test_voxel_predictor_invalid_birth_probability(simple_transition_model):
    """Test that invalid birth probability raises ValueError."""
    with pytest.raises(ValueError, match="birth_probability must be in \\[0, 1\\]"):
        VoxelPredictor(
            transition_model=simple_transition_model,
            birth_probability=-0.1
        )

    with pytest.raises(ValueError, match="birth_probability must be in \\[0, 1\\]"):
        VoxelPredictor(
            transition_model=simple_transition_model,
            birth_probability=1.5
        )


def test_voxel_predictor_invalid_death_probability(simple_transition_model):
    """Test that invalid death probability raises ValueError."""
    with pytest.raises(ValueError, match="death_probability must be in \\[0, 1\\]"):
        VoxelPredictor(
            transition_model=simple_transition_model,
            death_probability=-0.1
        )

    with pytest.raises(ValueError, match="death_probability must be in \\[0, 1\\]"):
        VoxelPredictor(
            transition_model=simple_transition_model,
            death_probability=1.2
        )


def test_voxel_predictor_extreme_probabilities(simple_transition_model):
    """Test VoxelPredictor with boundary probability values."""
    # Test with zero probabilities
    predictor_zero = VoxelPredictor(
        transition_model=simple_transition_model,
        birth_probability=0.0,
        death_probability=0.0
    )
    assert predictor_zero.birth_probability == 0.0
    assert predictor_zero.death_probability == 0.0

    # Test with maximum probabilities
    predictor_max = VoxelPredictor(
        transition_model=simple_transition_model,
        birth_probability=1.0,
        death_probability=1.0
    )
    assert predictor_max.birth_probability == 1.0
    assert predictor_max.death_probability == 1.0


# Test VoxelPredictor.predict() method
def test_voxel_predictor_predict_dense_state(dense_voxel_state):
    """Test prediction with dense voxel state."""
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )

    new_timestamp = dense_voxel_state.timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(dense_voxel_state, timestamp=new_timestamp)

    # Check prediction type
    assert isinstance(prediction, VoxelPrediction)
    assert prediction.timestamp == new_timestamp
    assert prediction.grid == dense_voxel_state.grid
    assert hasattr(prediction, 'transition_model')
    assert prediction.prior == dense_voxel_state


def test_voxel_predictor_predict_sparse_state(sparse_voxel_state):
    """Test prediction with sparse voxel state."""
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )

    new_timestamp = sparse_voxel_state.timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(sparse_voxel_state, timestamp=new_timestamp)

    # Check prediction type
    assert isinstance(prediction, VoxelPrediction)
    assert prediction.timestamp == new_timestamp
    assert prediction.is_sparse  # Should remain sparse
    assert prediction.grid == sparse_voxel_state.grid


def test_voxel_predictor_birth_death_process(small_voxel_grid):
    """Test that birth-death process is applied correctly."""
    # Create state with known occupancy
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0  # Fully occupied

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    # Create predictor with known birth/death probabilities
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.1,
        death_probability=0.2
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # For fully occupied voxel: p_new = (1 - 0.2) * 1.0 + 0.1 * (1 - 1.0) = 0.8
    expected_occupied = 0.8
    assert np.isclose(prediction.occupancy[2, 2, 2], expected_occupied, atol=1e-6)

    # For empty voxels: p_new = (1 - 0.2) * 0.0 + 0.1 * (1 - 0.0) = 0.1
    expected_empty = 0.1
    assert np.isclose(prediction.occupancy[0, 0, 0], expected_empty, atol=1e-6)


def test_voxel_predictor_probability_bounds(small_voxel_grid):
    """Test that predicted probabilities are clamped to [0, 1]."""
    # Create state with edge case occupancies
    occupancy = np.zeros((5, 5, 5))
    occupancy[0, 0, 0] = 0.0
    occupancy[2, 2, 2] = 1.0
    occupancy[4, 4, 4] = 0.5

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.05,
        death_probability=0.05
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # All probabilities should be in [0, 1]
    assert np.all(prediction.occupancy >= 0.0)
    assert np.all(prediction.occupancy <= 1.0)


def test_voxel_predictor_no_timestamp(dense_voxel_state):
    """Test prediction when timestamp is None."""
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )

    prediction = predictor.predict(dense_voxel_state, timestamp=None)

    assert isinstance(prediction, VoxelPrediction)
    assert prediction.timestamp is None


def test_voxel_predictor_invalid_prior_type(simple_transition_model):
    """Test that providing non-VoxelState prior raises TypeError."""
    from stonesoup.types.state import GaussianState

    predictor = VoxelPredictor(
        transition_model=simple_transition_model,
        birth_probability=0.01,
        death_probability=0.01
    )

    # Try to predict with wrong state type
    invalid_prior = GaussianState(
        state_vector=np.array([[0], [0]]),
        covar=np.eye(2),
        timestamp=datetime.datetime.now()
    )

    with pytest.raises(TypeError, match="prior must be a VoxelState"):
        predictor.predict(invalid_prior, timestamp=datetime.datetime.now())


def test_voxel_predictor_preserves_grid_structure(dense_voxel_state):
    """Test that prediction preserves the grid structure."""
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )

    new_timestamp = dense_voxel_state.timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(dense_voxel_state, timestamp=new_timestamp)

    # Grid should be the same object
    assert prediction.grid is dense_voxel_state.grid
    assert prediction.grid.shape == dense_voxel_state.grid.shape
    assert np.array_equal(prediction.grid.bounds, dense_voxel_state.grid.bounds)


# Test DiffusionVoxelPredictor
def test_diffusion_predictor_instantiation():
    """Test creating a DiffusionVoxelPredictor."""
    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.2,
        birth_probability=0.01,
        death_probability=0.02
    )

    assert predictor.diffusion_coefficient == 0.2
    assert predictor.birth_probability == 0.01
    assert predictor.death_probability == 0.02
    assert predictor.transition_model is None  # Not required for diffusion


def test_diffusion_predictor_default_values():
    """Test DiffusionVoxelPredictor uses correct default values."""
    predictor = DiffusionVoxelPredictor()

    assert predictor.diffusion_coefficient == 0.1
    assert predictor.birth_probability == 0.01
    assert predictor.death_probability == 0.01


def test_diffusion_predictor_invalid_coefficient():
    """Test that invalid diffusion coefficient raises ValueError."""
    with pytest.raises(ValueError, match="diffusion_coefficient must be in \\[0, 1\\]"):
        DiffusionVoxelPredictor(diffusion_coefficient=-0.1)

    with pytest.raises(ValueError, match="diffusion_coefficient must be in \\[0, 1\\]"):
        DiffusionVoxelPredictor(diffusion_coefficient=1.5)


def test_diffusion_predictor_kernel_creation():
    """Test that diffusion kernel is created correctly."""
    diffusion_coeff = 0.3
    predictor = DiffusionVoxelPredictor(diffusion_coefficient=diffusion_coeff)

    # Check kernel properties
    assert predictor.diffusion_kernel.shape == (3, 3, 3)

    # Center voxel should have (1 - diffusion_coefficient)
    assert np.isclose(predictor.diffusion_kernel[1, 1, 1], 1 - diffusion_coeff)

    # Each of 6 face-adjacent neighbors should have diffusion_coefficient / 6
    expected_neighbor_weight = diffusion_coeff / 6
    assert np.isclose(predictor.diffusion_kernel[0, 1, 1], expected_neighbor_weight)
    assert np.isclose(predictor.diffusion_kernel[2, 1, 1], expected_neighbor_weight)
    assert np.isclose(predictor.diffusion_kernel[1, 0, 1], expected_neighbor_weight)
    assert np.isclose(predictor.diffusion_kernel[1, 2, 1], expected_neighbor_weight)
    assert np.isclose(predictor.diffusion_kernel[1, 1, 0], expected_neighbor_weight)
    assert np.isclose(predictor.diffusion_kernel[1, 1, 2], expected_neighbor_weight)

    # Diagonal elements should be zero (no diagonal diffusion)
    assert predictor.diffusion_kernel[0, 0, 0] == 0.0
    assert predictor.diffusion_kernel[2, 2, 2] == 0.0


def test_diffusion_predictor_predict_dense(small_voxel_grid):
    """Test diffusion prediction with dense occupancy."""
    # Create state with single occupied voxel at center
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.6,  # High diffusion for visible effect
        birth_probability=0.0,  # No birth for clearer diffusion test
        death_probability=0.0   # No death for clearer diffusion test
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # Check that prediction is correct type
    assert isinstance(prediction, VoxelPrediction)
    assert prediction.timestamp == new_timestamp

    # Center voxel should have reduced occupancy due to diffusion
    assert prediction.occupancy[2, 2, 2] < 1.0

    # Adjacent voxels should have gained occupancy
    assert prediction.occupancy[1, 2, 2] > 0.0  # -x neighbor
    assert prediction.occupancy[3, 2, 2] > 0.0  # +x neighbor
    assert prediction.occupancy[2, 1, 2] > 0.0  # -y neighbor
    assert prediction.occupancy[2, 3, 2] > 0.0  # +y neighbor
    assert prediction.occupancy[2, 2, 1] > 0.0  # -z neighbor
    assert prediction.occupancy[2, 2, 3] > 0.0  # +z neighbor

    # Diagonal voxels should not gain occupancy (6-connected only)
    assert prediction.occupancy[1, 1, 1] == 0.0


def test_diffusion_predictor_predict_sparse(small_voxel_grid):
    """Test diffusion prediction with sparse occupancy."""
    occupancy = {
        (2, 2, 2): 0.9,
        (2, 2, 3): 0.5
    }

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.3,
        birth_probability=0.0,
        death_probability=0.0
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # Prediction should be sparse
    assert isinstance(prediction, VoxelPrediction)
    assert prediction.is_sparse

    # Occupancy should have spread to neighbors
    assert len(prediction.occupancy) > len(state.occupancy)


def test_diffusion_predictor_mass_conservation(small_voxel_grid):
    """Test that diffusion approximately conserves total probability mass.

    With no birth/death, total mass should be conserved by diffusion.
    """
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.5,
        birth_probability=0.0,
        death_probability=0.0
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # Total probability mass should be approximately conserved
    initial_mass = np.sum(state.occupancy)
    predicted_mass = np.sum(prediction.occupancy)

    # Allow small numerical error
    assert np.isclose(initial_mass, predicted_mass, rtol=1e-5)


def test_diffusion_predictor_with_birth_death(small_voxel_grid):
    """Test diffusion with birth-death process."""
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.2,
        birth_probability=0.05,
        death_probability=0.1
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # All probabilities should be in valid range
    assert np.all(prediction.occupancy >= 0.0)
    assert np.all(prediction.occupancy <= 1.0)

    # Birth should add probability to empty voxels
    assert prediction.occupancy[0, 0, 0] > 0.0


def test_diffusion_predictor_boundary_handling(small_voxel_grid):
    """Test that diffusion handles boundary voxels correctly."""
    # Place occupancy at corner
    occupancy = np.zeros((5, 5, 5))
    occupancy[0, 0, 0] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.6,
        birth_probability=0.0,
        death_probability=0.0
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # Boundary voxel should diffuse only to valid neighbors
    # Corner has only 3 neighbors, so more mass stays in center
    assert prediction.occupancy[0, 0, 0] > 0.0
    assert prediction.occupancy[1, 0, 0] > 0.0
    assert prediction.occupancy[0, 1, 0] > 0.0
    assert prediction.occupancy[0, 0, 1] > 0.0


def test_diffusion_predictor_invalid_prior_type():
    """Test that DiffusionVoxelPredictor rejects non-VoxelState priors."""
    from stonesoup.types.state import GaussianState

    predictor = DiffusionVoxelPredictor()

    invalid_prior = GaussianState(
        state_vector=np.array([[0], [0]]),
        covar=np.eye(2),
        timestamp=datetime.datetime.now()
    )

    with pytest.raises(TypeError, match="prior must be a VoxelState"):
        predictor.predict(invalid_prior, timestamp=datetime.datetime.now())


def test_diffusion_predictor_zero_diffusion(small_voxel_grid):
    """Test diffusion with zero coefficient (no spreading)."""
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.0,
        birth_probability=0.0,
        death_probability=0.0
    )

    new_timestamp = timestamp + datetime.timedelta(seconds=1)
    prediction = predictor.predict(state, timestamp=new_timestamp)

    # With zero diffusion, birth, and death, occupancy should be unchanged
    assert np.array_equal(prediction.occupancy, state.occupancy)


def test_diffusion_predictor_multiple_steps(small_voxel_grid):
    """Test that multiple prediction steps compound diffusion effect."""
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 1.0

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.3,
        birth_probability=0.0,
        death_probability=0.0
    )

    # First prediction
    timestamp1 = timestamp + datetime.timedelta(seconds=1)
    prediction1 = predictor.predict(state, timestamp=timestamp1)

    # Second prediction
    timestamp2 = timestamp1 + datetime.timedelta(seconds=1)
    prediction2 = predictor.predict(prediction1, timestamp=timestamp2)

    # Occupancy should spread further after second step
    # Voxels 2 steps away should have some occupancy
    assert prediction2.occupancy[0, 2, 2] > 0.0  # 2 steps in -x
    assert prediction2.occupancy[4, 2, 2] > 0.0  # 2 steps in +x


# Integration tests
def test_voxel_predictor_lru_cache(dense_voxel_state):
    """Test that LRU cache works for predictions."""
    predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )

    new_timestamp = dense_voxel_state.timestamp + datetime.timedelta(seconds=1)

    # First prediction
    prediction1 = predictor.predict(dense_voxel_state, timestamp=new_timestamp)

    # Second prediction with same inputs (should be cached)
    prediction2 = predictor.predict(dense_voxel_state, timestamp=new_timestamp)

    # Should return the same object from cache
    assert prediction1 is prediction2


def test_predictor_types_comparison(small_voxel_grid):
    """Compare VoxelPredictor and DiffusionVoxelPredictor behavior."""
    occupancy = np.zeros((5, 5, 5))
    occupancy[2, 2, 2] = 0.8

    timestamp = datetime.datetime.now()
    state = VoxelState(grid=small_voxel_grid, occupancy=occupancy, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1)

    # Generic predictor with mock model
    generic_predictor = VoxelPredictor(
        transition_model=MockVoxelTransitionModel(),
        birth_probability=0.01,
        death_probability=0.01
    )
    generic_prediction = generic_predictor.predict(state, timestamp=new_timestamp)

    # Diffusion predictor
    diffusion_predictor = DiffusionVoxelPredictor(
        diffusion_coefficient=0.2,
        birth_probability=0.01,
        death_probability=0.01
    )
    diffusion_prediction = diffusion_predictor.predict(state, timestamp=new_timestamp)

    # Both should return VoxelPrediction
    assert isinstance(generic_prediction, VoxelPrediction)
    assert isinstance(diffusion_prediction, VoxelPrediction)

    # But occupancy distributions will differ due to diffusion
    assert not np.array_equal(generic_prediction.occupancy, diffusion_prediction.occupancy)
