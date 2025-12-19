import datetime
import pickle

import numpy as np
import pytest

from ..array import CovarianceMatrix, StateVector
from ..state import GaussianState
from ..voxel import OctreeNode, VoxelGrid, VoxelState


def test_voxelgrid_creation():
    """Test VoxelGrid creation with valid and invalid bounds."""
    # Valid creation
    bounds = np.array([0, 10, 0, 10, 0, 10])
    grid = VoxelGrid(bounds=bounds, resolution=1.0)
    assert np.array_equal(grid.bounds, bounds)
    assert grid.resolution == 1.0
    assert grid.max_depth == 0

    # Valid creation with list input
    grid2 = VoxelGrid(bounds=[0, 10, 0, 10, 0, 10], resolution=0.5, max_depth=3)
    assert isinstance(grid2.bounds, np.ndarray)
    assert grid2.bounds.shape == (6,)
    assert grid2.resolution == 0.5
    assert grid2.max_depth == 3

    # Invalid: wrong shape
    with pytest.raises(ValueError, match="must be a 6-element array"):
        VoxelGrid(bounds=np.array([0, 10, 0, 10]), resolution=1.0)

    # Invalid: min >= max
    with pytest.raises(ValueError, match="x_min < x_max"):
        VoxelGrid(bounds=np.array([10, 0, 0, 10, 0, 10]), resolution=1.0)

    with pytest.raises(ValueError, match="y_min < y_max"):
        VoxelGrid(bounds=np.array([0, 10, 10, 0, 0, 10]), resolution=1.0)

    with pytest.raises(ValueError, match="z_min < z_max"):
        VoxelGrid(bounds=np.array([0, 10, 0, 10, 10, 0]), resolution=1.0)


def test_voxelgrid_properties():
    """Test VoxelGrid computed properties."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 20, 0, 30]), resolution=2.0)

    # Test dimensions
    assert np.array_equal(grid.dimensions, np.array([10, 20, 30]))

    # Test shape
    assert grid.shape == (5, 10, 15)

    # Test num_voxels
    assert grid.num_voxels == 5 * 10 * 15


def test_voxelgrid_contains():
    """Test VoxelGrid.contains() boundary tests."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Points inside
    assert grid.contains(np.array([5, 5, 5]))
    assert grid.contains(np.array([0, 0, 0]))  # Min boundary
    assert grid.contains(np.array([10, 10, 10]))  # Max boundary
    assert grid.contains(np.array([0.001, 0.001, 0.001]))
    assert grid.contains(np.array([9.999, 9.999, 9.999]))

    # Points outside
    assert not grid.contains(np.array([-0.001, 5, 5]))
    assert not grid.contains(np.array([5, -0.001, 5]))
    assert not grid.contains(np.array([5, 5, -0.001]))
    assert not grid.contains(np.array([10.001, 5, 5]))
    assert not grid.contains(np.array([5, 10.001, 5]))
    assert not grid.contains(np.array([5, 5, 10.001]))

    # Invalid shape
    with pytest.raises(ValueError, match="point shape should be"):
        grid.contains(np.array([5, 5]))

    with pytest.raises(ValueError, match="point shape should be"):
        grid.contains(np.array([5, 5, 5, 5]))


def test_voxelgrid_voxel_indices():
    """Test VoxelGrid.voxel_indices() for various points."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Test various points
    assert grid.voxel_indices(np.array([0.5, 0.5, 0.5])) == (0, 0, 0)
    assert grid.voxel_indices(np.array([5.5, 3.2, 8.7])) == (5, 3, 8)
    assert grid.voxel_indices(np.array([9.9, 9.9, 9.9])) == (9, 9, 9)
    assert grid.voxel_indices(np.array([0, 0, 0])) == (0, 0, 0)

    # Edge case: point exactly at max boundary should be clamped to last voxel
    assert grid.voxel_indices(np.array([10, 10, 10])) == (9, 9, 9)

    # Points outside bounds should return None
    assert grid.voxel_indices(np.array([-1, 5, 5])) is None
    assert grid.voxel_indices(np.array([5, -1, 5])) is None
    assert grid.voxel_indices(np.array([5, 5, -1])) is None
    assert grid.voxel_indices(np.array([11, 5, 5])) is None
    assert grid.voxel_indices(np.array([5, 11, 5])) is None
    assert grid.voxel_indices(np.array([5, 5, 11])) is None


def test_voxelgrid_voxel_indices_non_unit_resolution():
    """Test voxel_indices with non-unit resolution."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=2.0)

    # With 2.0 resolution, we have 5x5x5 voxels
    assert grid.shape == (5, 5, 5)
    assert grid.voxel_indices(np.array([1, 1, 1])) == (0, 0, 0)
    assert grid.voxel_indices(np.array([3, 3, 3])) == (1, 1, 1)
    assert grid.voxel_indices(np.array([9, 9, 9])) == (4, 4, 4)


def test_voxelgrid_voxel_indices_offset_bounds():
    """Test voxel_indices with offset bounds."""
    grid = VoxelGrid(bounds=np.array([-5, 5, -5, 5, 0, 10]), resolution=1.0)

    assert grid.shape == (10, 10, 10)
    assert grid.voxel_indices(np.array([-4.5, -4.5, 0.5])) == (0, 0, 0)
    assert grid.voxel_indices(np.array([0, 0, 5])) == (5, 5, 5)
    assert grid.voxel_indices(np.array([4.9, 4.9, 9.9])) == (9, 9, 9)


def test_voxelgrid_volume():
    """Test VoxelGrid.volume() calculation."""
    grid1 = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    assert grid1.volume() == 1000.0

    grid2 = VoxelGrid(bounds=np.array([0, 5, 0, 10, 0, 2]), resolution=0.5)
    assert grid2.volume() == 100.0

    grid3 = VoxelGrid(bounds=np.array([-5, 5, -10, 10, -2, 3]), resolution=1.0)
    assert grid3.volume() == 10 * 20 * 5


def test_voxelgrid_voxel_center():
    """Test VoxelGrid.voxel_center() method."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Test various voxel centers
    assert np.array_equal(grid.voxel_center((0, 0, 0)), np.array([0.5, 0.5, 0.5]))
    assert np.array_equal(grid.voxel_center((5, 3, 8)), np.array([5.5, 3.5, 8.5]))
    assert np.array_equal(grid.voxel_center((9, 9, 9)), np.array([9.5, 9.5, 9.5]))

    # Test with different resolution
    grid2 = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=2.0)
    assert np.array_equal(grid2.voxel_center((0, 0, 0)), np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(grid2.voxel_center((2, 2, 2)), np.array([5.0, 5.0, 5.0]))

    # Test with offset bounds
    grid3 = VoxelGrid(bounds=np.array([-5, 5, -5, 5, 0, 10]), resolution=1.0)
    assert np.array_equal(grid3.voxel_center((0, 0, 0)), np.array([-4.5, -4.5, 0.5]))
    assert np.array_equal(grid3.voxel_center((5, 5, 5)), np.array([0.5, 0.5, 5.5]))

    # Test invalid indices
    with pytest.raises(ValueError, match="Invalid voxel indices"):
        grid.voxel_center((10, 0, 0))

    with pytest.raises(ValueError, match="Invalid voxel indices"):
        grid.voxel_center((-1, 0, 0))


def test_octreenode_creation():
    """Test OctreeNode creation."""
    bounds = np.array([0, 10, 0, 10, 0, 10])
    node = OctreeNode(bounds=bounds, depth=0)

    assert np.array_equal(node.bounds, bounds)
    assert node.depth == 0
    assert node.data is None
    assert node.children is None
    assert node.is_leaf

    # With data
    node2 = OctreeNode(bounds=bounds, depth=1, data=0.5)
    assert node2.data == 0.5
    assert node2.is_leaf

    # Invalid bounds
    with pytest.raises(ValueError, match="must be a 6-element array"):
        OctreeNode(bounds=np.array([0, 10, 0, 10]), depth=0)


def test_octreenode_properties():
    """Test OctreeNode computed properties."""
    node = OctreeNode(bounds=np.array([0, 10, 0, 10, 0, 10]), depth=0, data=0.5)

    # Test center
    assert np.array_equal(node.center, np.array([5, 5, 5]))

    # Test volume
    assert node.volume == 1000.0

    # Test is_leaf
    assert node.is_leaf


def test_octreenode_subdivide():
    """Test OctreeNode subdivision."""
    node = OctreeNode(bounds=np.array([0, 10, 0, 10, 0, 10]), depth=0, data=0.5)

    # Subdivide
    children = node.subdivide()

    assert len(children) == 8
    assert all(isinstance(child, OctreeNode) for child in children)
    assert all(child.depth == 1 for child in children)
    assert all(child.data == 0.5 for child in children)  # Inherit parent data
    assert all(child.is_leaf for child in children)

    # Check octant bounds
    # Child 0: lower-x, lower-y, lower-z
    assert np.array_equal(children[0].bounds, np.array([0, 5, 0, 5, 0, 5]))
    # Child 7: upper-x, upper-y, upper-z
    assert np.array_equal(children[7].bounds, np.array([5, 10, 5, 10, 5, 10]))

    # Verify volumes sum correctly
    total_volume = sum(child.volume for child in children)
    assert np.isclose(total_volume, node.volume)

    # Cannot subdivide non-leaf
    node.children = children
    with pytest.raises(ValueError, match="Cannot subdivide non-leaf node"):
        node.subdivide()


def test_octreenode_contains():
    """Test OctreeNode.contains() method."""
    node = OctreeNode(bounds=np.array([0, 10, 0, 10, 0, 10]), depth=0)

    # Points inside
    assert node.contains(np.array([5, 5, 5]))
    assert node.contains(np.array([0, 0, 0]))
    assert node.contains(np.array([10, 10, 10]))

    # Points outside
    assert not node.contains(np.array([-0.1, 5, 5]))
    assert not node.contains(np.array([10.1, 5, 5]))

    # Invalid shape
    with pytest.raises(ValueError, match="point shape should be"):
        node.contains(np.array([5, 5]))


def test_voxelstate_creation_dense():
    """Test VoxelState creation with dense occupancy."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.random.random((10, 10, 10))
    timestamp = datetime.datetime.now()

    state = VoxelState(grid=grid, occupancy=occupancy, timestamp=timestamp)

    assert state.grid is grid
    assert np.array_equal(state.occupancy, occupancy)
    assert state.timestamp == timestamp
    assert not state.is_sparse


def test_voxelstate_creation_sparse():
    """Test VoxelState creation with sparse occupancy."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = {(5, 5, 5): 0.9, (6, 6, 6): 0.8}
    timestamp = datetime.datetime.now()

    state = VoxelState(grid=grid, occupancy=occupancy, timestamp=timestamp)

    assert state.grid is grid
    assert state.occupancy == occupancy
    assert state.timestamp == timestamp
    assert state.is_sparse


def test_voxelstate_creation_validation():
    """Test VoxelState creation validation."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Wrong dense shape
    with pytest.raises(ValueError, match="does not match grid shape"):
        VoxelState(grid=grid, occupancy=np.random.random((5, 5, 5)))

    # Invalid sparse key format
    with pytest.raises(ValueError, match="must be 3-tuples"):
        VoxelState(grid=grid, occupancy={(5, 5): 0.9})

    with pytest.raises(ValueError, match="must be 3-tuples"):
        VoxelState(grid=grid, occupancy={"key": 0.9})

    # Invalid sparse key types
    with pytest.raises(ValueError, match="must be integers"):
        VoxelState(grid=grid, occupancy={(5.5, 5, 5): 0.9})

    # Invalid occupancy type
    with pytest.raises(TypeError, match="must be numpy array or dict"):
        VoxelState(grid=grid, occupancy=[0.9, 0.8])


def test_voxelstate_probability_at_dense():
    """Test VoxelState.probability_at() with dense occupancy."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.ones((10, 10, 10)) * 0.5
    occupancy[5, 5, 5] = 0.9

    state = VoxelState(grid=grid, occupancy=occupancy)

    # Test probability retrieval
    assert state.probability_at(np.array([5.5, 5.5, 5.5])) == 0.9
    assert state.probability_at(np.array([3.2, 4.7, 8.1])) == 0.5

    # Test outside bounds returns 0
    assert state.probability_at(np.array([-1, 5, 5])) == 0.0
    assert state.probability_at(np.array([15, 5, 5])) == 0.0


def test_voxelstate_probability_at_sparse():
    """Test VoxelState.probability_at() with sparse occupancy."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = {(5, 5, 5): 0.9, (6, 6, 6): 0.8}

    state = VoxelState(grid=grid, occupancy=occupancy)

    # Test probability retrieval
    assert state.probability_at(np.array([5.5, 5.5, 5.5])) == 0.9
    assert state.probability_at(np.array([6.5, 6.5, 6.5])) == 0.8

    # Test empty voxels return 0
    assert state.probability_at(np.array([0.5, 0.5, 0.5])) == 0.0

    # Test outside bounds returns 0
    assert state.probability_at(np.array([-1, 5, 5])) == 0.0


def test_voxelstate_to_gaussian_single_voxel():
    """Test VoxelState.to_gaussian() with single occupied voxel."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.zeros((10, 10, 10))
    occupancy[5, 5, 5] = 1.0

    state = VoxelState(grid=grid, occupancy=occupancy)
    gaussian = state.to_gaussian()

    assert isinstance(gaussian, GaussianState)
    assert gaussian.state_vector.shape == (3, 1)

    # Mean should be at voxel center
    expected_mean = np.array([[5.5], [5.5], [5.5]])
    assert np.allclose(gaussian.state_vector, expected_mean)

    # Should have minimum covariance based on resolution
    min_var = (grid.resolution / 2) ** 2
    assert np.all(np.diag(gaussian.covar) >= min_var)


def test_voxelstate_to_gaussian_multiple_voxels():
    """Test VoxelState.to_gaussian() with multiple occupied voxels."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.zeros((10, 10, 10))
    occupancy[4, 5, 6] = 0.5
    occupancy[6, 5, 6] = 0.5

    state = VoxelState(grid=grid, occupancy=occupancy)
    gaussian = state.to_gaussian()

    assert isinstance(gaussian, GaussianState)

    # Mean should be weighted average of voxel centers
    # Centers: [4.5, 5.5, 6.5] and [6.5, 5.5, 6.5]
    # Equal weights, so mean should be [5.5, 5.5, 6.5]
    expected_mean = np.array([[5.5], [5.5], [6.5]])
    assert np.allclose(gaussian.state_vector, expected_mean)


def test_voxelstate_to_gaussian_sparse():
    """Test VoxelState.to_gaussian() with sparse occupancy."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = {(5, 5, 5): 0.9, (6, 6, 6): 0.1}

    state = VoxelState(grid=grid, occupancy=occupancy)
    gaussian = state.to_gaussian()

    assert isinstance(gaussian, GaussianState)

    # Mean should be weighted toward the higher probability voxel
    # Centers: [5.5, 5.5, 5.5] (weight 0.9) and [6.5, 6.5, 6.5] (weight 0.1)
    # After normalization: weights are 0.9 and 0.1
    expected_mean = np.array(
        [[5.5 * 0.9 + 6.5 * 0.1], [5.5 * 0.9 + 6.5 * 0.1], [5.5 * 0.9 + 6.5 * 0.1]]
    )
    assert np.allclose(gaussian.state_vector, expected_mean)


def test_voxelstate_to_gaussian_empty():
    """Test VoxelState.to_gaussian() with no occupied voxels."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.zeros((10, 10, 10))

    state = VoxelState(grid=grid, occupancy=occupancy)
    gaussian = state.to_gaussian()

    assert isinstance(gaussian, GaussianState)

    # Mean should be at grid center
    expected_mean = np.array([[5.0], [5.0], [5.0]])
    assert np.allclose(gaussian.state_vector, expected_mean)

    # Covariance should be large
    dims = grid.dimensions
    expected_var = (dims / 2) ** 2
    assert np.allclose(np.diag(gaussian.covar), expected_var)


def test_voxelstate_to_gaussian_timestamp():
    """Test VoxelState.to_gaussian() preserves timestamp."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = np.zeros((10, 10, 10))
    occupancy[5, 5, 5] = 1.0
    timestamp = datetime.datetime.now()

    state = VoxelState(grid=grid, occupancy=occupancy, timestamp=timestamp)
    gaussian = state.to_gaussian()

    assert gaussian.timestamp == timestamp


def test_voxelstate_from_gaussian():
    """Test VoxelState.from_gaussian() creates voxel representation."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    mean = StateVector([5.5, 5.5, 5.5])  # Use voxel center to avoid boundary issues
    covar = CovarianceMatrix(np.eye(3))
    timestamp = datetime.datetime.now()

    gaussian = GaussianState(state_vector=mean, covar=covar, timestamp=timestamp)
    voxel_state = VoxelState.from_gaussian(gaussian, grid, threshold=0.01)

    assert isinstance(voxel_state, VoxelState)
    assert voxel_state.grid is grid
    assert voxel_state.is_sparse
    assert voxel_state.timestamp == timestamp

    # Should have maximum probability at voxel containing mean
    max_prob_idx = max(voxel_state.occupancy, key=voxel_state.occupancy.get)

    # Mean [5.5, 5.5, 5.5] is at center of voxel (5, 5, 5)
    assert grid.voxel_indices(mean.flatten()) == (5, 5, 5)
    assert max_prob_idx == (5, 5, 5)

    # Probabilities should be normalized to [0, 1]
    assert all(0 <= prob <= 1 for prob in voxel_state.occupancy.values())
    assert max(voxel_state.occupancy.values()) == 1.0  # Normalized


def test_voxelstate_from_gaussian_invalid_dimension():
    """Test VoxelState.from_gaussian() rejects non-3D Gaussian."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    mean = StateVector([5, 5, 5, 5])  # 4D
    covar = CovarianceMatrix(np.eye(4))

    gaussian = GaussianState(state_vector=mean, covar=covar)

    with pytest.raises(ValueError, match="must be 3D"):
        VoxelState.from_gaussian(gaussian, grid)


def test_voxelstate_from_gaussian_threshold():
    """Test VoxelState.from_gaussian() threshold filtering."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    mean = StateVector([5, 5, 5])
    covar = CovarianceMatrix(np.eye(3) * 0.1)  # Tight covariance

    gaussian = GaussianState(state_vector=mean, covar=covar)

    # Low threshold should include more voxels
    voxel_state_low = VoxelState.from_gaussian(gaussian, grid, threshold=0.001)
    # High threshold should include fewer voxels
    voxel_state_high = VoxelState.from_gaussian(gaussian, grid, threshold=0.1)

    assert len(voxel_state_low.occupancy) >= len(voxel_state_high.occupancy)


def test_voxelstate_gaussian_roundtrip():
    """Test VoxelState.to_gaussian() and from_gaussian() round-trip."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    mean = StateVector([5, 5, 5])
    covar = CovarianceMatrix(np.eye(3) * 2.0)
    timestamp = datetime.datetime.now()

    # Start with Gaussian
    gaussian1 = GaussianState(state_vector=mean, covar=covar, timestamp=timestamp)

    # Convert to voxel and back
    voxel_state = VoxelState.from_gaussian(gaussian1, grid, threshold=0.01)
    gaussian2 = voxel_state.to_gaussian()

    # Means should be reasonably close
    assert np.allclose(gaussian1.state_vector, gaussian2.state_vector, atol=1.0)

    # Timestamps should be preserved
    assert gaussian2.timestamp == timestamp


def test_voxelstate_to_dense():
    """Test VoxelState.to_dense() conversion."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Sparse to dense
    sparse_occ = {(5, 5, 5): 0.9, (6, 6, 6): 0.8}
    sparse_state = VoxelState(grid=grid, occupancy=sparse_occ)
    dense_state = sparse_state.to_dense()

    assert not dense_state.is_sparse
    assert dense_state.occupancy.shape == (10, 10, 10)
    assert dense_state.occupancy[5, 5, 5] == 0.9
    assert dense_state.occupancy[6, 6, 6] == 0.8
    assert dense_state.occupancy[0, 0, 0] == 0.0

    # Dense to dense (should copy)
    dense_occ = np.random.random((10, 10, 10))
    dense_state2 = VoxelState(grid=grid, occupancy=dense_occ)
    dense_state3 = dense_state2.to_dense()

    assert not dense_state3.is_sparse
    assert np.array_equal(dense_state3.occupancy, dense_occ)
    assert dense_state3.occupancy is not dense_occ  # Should be a copy


def test_voxelstate_to_sparse():
    """Test VoxelState.to_sparse() conversion."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Dense to sparse
    dense_occ = np.zeros((10, 10, 10))
    dense_occ[5, 5, 5] = 0.9
    dense_occ[6, 6, 6] = 0.8
    dense_occ[7, 7, 7] = 0.000001  # Below default threshold

    dense_state = VoxelState(grid=grid, occupancy=dense_occ)
    sparse_state = dense_state.to_sparse()

    assert sparse_state.is_sparse
    assert len(sparse_state.occupancy) == 3  # Includes tiny value
    assert sparse_state.occupancy[(5, 5, 5)] == 0.9
    assert sparse_state.occupancy[(6, 6, 6)] == 0.8

    # With higher threshold
    sparse_state2 = dense_state.to_sparse(threshold=0.1)
    assert len(sparse_state2.occupancy) == 2  # Excludes values below 0.1

    # Sparse to sparse (should filter by threshold)
    sparse_occ = {(5, 5, 5): 0.9, (6, 6, 6): 0.05}
    sparse_state3 = VoxelState(grid=grid, occupancy=sparse_occ)
    sparse_state4 = sparse_state3.to_sparse(threshold=0.1)

    assert sparse_state4.is_sparse
    assert len(sparse_state4.occupancy) == 1
    assert (5, 5, 5) in sparse_state4.occupancy
    assert (6, 6, 6) not in sparse_state4.occupancy


def test_voxelstate_dense_sparse_roundtrip():
    """Test conversion between dense and sparse representations."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)

    # Start with sparse
    sparse_occ = {(5, 5, 5): 0.9, (6, 6, 6): 0.8}
    state1 = VoxelState(grid=grid, occupancy=sparse_occ)

    # Convert to dense and back
    state2 = state1.to_dense()
    state3 = state2.to_sparse()

    assert state1.is_sparse
    assert not state2.is_sparse
    assert state3.is_sparse
    assert state3.occupancy == sparse_occ


def test_voxelstate_timestamp_preservation():
    """Test that timestamps are preserved through conversions."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    timestamp = datetime.datetime.now()

    sparse_occ = {(5, 5, 5): 0.9}
    state1 = VoxelState(grid=grid, occupancy=sparse_occ, timestamp=timestamp)

    state2 = state1.to_dense()
    assert state2.timestamp == timestamp

    state3 = state2.to_sparse()
    assert state3.timestamp == timestamp


def test_voxelgrid_pickle():
    """Test VoxelGrid can be pickled."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0, max_depth=3)
    pickled = pickle.dumps(grid)
    unpickled = pickle.loads(pickled)  # nosec B301 - testing our own serialization

    assert np.array_equal(unpickled.bounds, grid.bounds)
    assert unpickled.resolution == grid.resolution
    assert unpickled.max_depth == grid.max_depth


def test_octreenode_pickle():
    """Test OctreeNode can be pickled."""
    node = OctreeNode(bounds=np.array([0, 10, 0, 10, 0, 10]), depth=0, data=0.5)
    pickled = pickle.dumps(node)
    unpickled = pickle.loads(pickled)  # nosec B301 - testing our own serialization

    assert np.array_equal(unpickled.bounds, node.bounds)
    assert unpickled.depth == node.depth
    assert unpickled.data == node.data


def test_voxelstate_pickle():
    """Test VoxelState can be pickled."""
    grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
    occupancy = {(5, 5, 5): 0.9}
    timestamp = datetime.datetime.now()

    state = VoxelState(grid=grid, occupancy=occupancy, timestamp=timestamp)
    pickled = pickle.dumps(state)
    unpickled = pickle.loads(pickled)  # nosec B301 - testing our own serialization

    assert unpickled.occupancy == occupancy
    assert unpickled.timestamp == timestamp
