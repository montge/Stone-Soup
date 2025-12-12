"""Voxel-based state representations for volumetric tracking.

This module provides voxel grid and octree-based state types for representing
spatial distributions in 3D space, useful for extended object tracking and
volumetric state estimation.
"""
import datetime
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal

from ..base import Property
from .array import StateVector, CovarianceMatrix
from .base import Type


class VoxelGrid(Type):
    """Voxel Grid definition.

    Defines a 3D voxel grid with uniform resolution and spatial bounds.
    Used as the basis for voxel-based state representations.

    Example
    -------
    >>> # Create a 10x10x10m grid with 0.5m resolution
    >>> grid = VoxelGrid(
    ...     bounds=np.array([-5, 5, -5, 5, 0, 10]),
    ...     resolution=0.5,
    ...     max_depth=3
    ... )
    >>> # Check if point is in grid
    >>> grid.contains(np.array([0, 0, 5]))
    True
    >>> # Get voxel indices for a point
    >>> indices = grid.voxel_indices(np.array([1.2, -2.3, 4.5]))
    >>> # Calculate grid volume
    >>> vol = grid.volume()
    """

    bounds: np.ndarray = Property(
        doc="Spatial bounds as [x_min, x_max, y_min, y_max, z_min, z_max]. "
            "Must be a 6-element array defining the axis-aligned bounding box."
    )
    resolution: float = Property(
        doc="Voxel resolution (edge length) in meters. All voxels are cubic "
            "with this edge length."
    )
    max_depth: int = Property(
        default=0,
        doc="Maximum octree subdivision depth for adaptive resolution. "
            "If 0 (default), uniform grid is used without octree subdivision."
    )

    def __init__(self, bounds, *args, **kwargs):
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        if bounds.shape != (6,):
            raise ValueError(
                f"bounds must be a 6-element array, got shape {bounds.shape}")
        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]):
            raise ValueError(
                "bounds must satisfy x_min < x_max, y_min < y_max, z_min < z_max")
        super().__init__(bounds, *args, **kwargs)

    @property
    def dimensions(self) -> np.ndarray:
        """Grid dimensions in each axis [dx, dy, dz]."""
        return np.array([
            self.bounds[1] - self.bounds[0],
            self.bounds[3] - self.bounds[2],
            self.bounds[5] - self.bounds[4]
        ])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Number of voxels along each axis (nx, ny, nz)."""
        dims = self.dimensions
        return tuple(int(np.ceil(d / self.resolution)) for d in dims)

    @property
    def num_voxels(self) -> int:
        """Total number of voxels in the grid."""
        return int(np.prod(self.shape))

    def voxel_indices(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert a 3D point to voxel grid indices.

        Parameters
        ----------
        point : np.ndarray
            3D point as [x, y, z] array.

        Returns
        -------
        Optional[Tuple[int, int, int]]
            Voxel indices (i, j, k) if point is within bounds, None otherwise.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> grid.voxel_indices(np.array([5.5, 3.2, 8.7]))
        (5, 3, 8)
        """
        if not self.contains(point):
            return None

        # Compute normalized position [0, 1] in each dimension
        normalized = np.array([
            (point[0] - self.bounds[0]) / (self.bounds[1] - self.bounds[0]),
            (point[1] - self.bounds[2]) / (self.bounds[3] - self.bounds[2]),
            (point[2] - self.bounds[4]) / (self.bounds[5] - self.bounds[4])
        ])

        # Convert to voxel indices
        shape = self.shape
        indices = np.floor(normalized * np.array(shape)).astype(int)

        # Clamp to valid range (handles edge case where point == max bound)
        indices = np.clip(indices, 0, np.array(shape) - 1)

        return tuple(indices)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a 3D point is within the grid bounds.

        Parameters
        ----------
        point : np.ndarray
            3D point as [x, y, z] array.

        Returns
        -------
        bool
            True if point is within grid bounds, False otherwise.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> grid.contains(np.array([5, 5, 5]))
        True
        >>> grid.contains(np.array([15, 5, 5]))
        False
        """
        if point.shape != (3,):
            raise ValueError(f"point must be 3D, got shape {point.shape}")

        return (self.bounds[0] <= point[0] <= self.bounds[1] and
                self.bounds[2] <= point[1] <= self.bounds[3] and
                self.bounds[4] <= point[2] <= self.bounds[5])

    def volume(self) -> float:
        """Calculate total volume of the grid.

        Returns
        -------
        float
            Total grid volume in cubic meters.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> grid.volume()
        1000.0
        """
        return float(np.prod(self.dimensions))

    def voxel_center(self, indices: Tuple[int, int, int]) -> np.ndarray:
        """Get the center point of a voxel given its indices.

        Parameters
        ----------
        indices : Tuple[int, int, int]
            Voxel indices (i, j, k).

        Returns
        -------
        np.ndarray
            3D point [x, y, z] at the voxel center.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> grid.voxel_center((5, 3, 8))
        array([5.5, 3.5, 8.5])
        """
        i, j, k = indices
        shape = self.shape

        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            raise ValueError(f"Invalid voxel indices {indices} for grid shape {shape}")

        # Center is at (index + 0.5) * resolution from min bound
        x = self.bounds[0] + (i + 0.5) * self.resolution
        y = self.bounds[2] + (j + 0.5) * self.resolution
        z = self.bounds[4] + (k + 0.5) * self.resolution

        return np.array([x, y, z])


class OctreeNode(Type):
    """Octree node for adaptive voxel resolution.

    Represents a node in an octree structure for hierarchical spatial
    subdivision. Each node can have up to 8 children representing octants.

    Example
    -------
    >>> # Create root node
    >>> root = OctreeNode(
    ...     bounds=np.array([-5, 5, -5, 5, 0, 10]),
    ...     depth=0,
    ...     data=0.5
    ... )
    >>> # Create children for subdivision
    >>> children = root.subdivide()
    >>> len(children)
    8
    """

    bounds: np.ndarray = Property(
        doc="Spatial bounds of this node as [x_min, x_max, y_min, y_max, z_min, z_max]."
    )
    depth: int = Property(
        doc="Depth level in the octree (0 = root)."
    )
    data: Optional[float] = Property(
        default=None,
        doc="Data value stored at this node (e.g., occupancy probability). "
            "None for non-leaf nodes."
    )
    children: Optional[Tuple['OctreeNode', ...]] = Property(
        default=None,
        doc="Tuple of 8 child nodes for subdivided octants, None for leaf nodes. "
            "Children are ordered by binary encoding: (x_low/high, y_low/high, z_low/high)."
    )

    def __init__(self, bounds, *args, **kwargs):
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        if bounds.shape != (6,):
            raise ValueError(
                f"bounds must be a 6-element array, got shape {bounds.shape}")
        super().__init__(bounds, *args, **kwargs)

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (has no children)."""
        return self.children is None

    @property
    def center(self) -> np.ndarray:
        """Center point of this node's bounds."""
        return np.array([
            (self.bounds[0] + self.bounds[1]) / 2,
            (self.bounds[2] + self.bounds[3]) / 2,
            (self.bounds[4] + self.bounds[5]) / 2
        ])

    @property
    def volume(self) -> float:
        """Volume of this node."""
        dx = self.bounds[1] - self.bounds[0]
        dy = self.bounds[3] - self.bounds[2]
        dz = self.bounds[5] - self.bounds[4]
        return dx * dy * dz

    def subdivide(self) -> Tuple['OctreeNode', ...]:
        """Subdivide this node into 8 octant children.

        Returns
        -------
        Tuple[OctreeNode, ...]
            Tuple of 8 child nodes, one for each octant.

        Example
        -------
        >>> node = OctreeNode(bounds=np.array([0, 10, 0, 10, 0, 10]), depth=0)
        >>> children = node.subdivide()
        >>> children[0].bounds  # Lower-left-bottom octant
        array([0., 5., 0., 5., 0., 5.])
        """
        if not self.is_leaf:
            raise ValueError("Cannot subdivide non-leaf node")

        center = self.center
        children = []

        # Create 8 children in binary order: (x, y, z) where each is low(0) or high(1)
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    child_bounds = np.array([
                        center[0] if ix else self.bounds[0],
                        self.bounds[1] if ix else center[0],
                        center[1] if iy else self.bounds[2],
                        self.bounds[3] if iy else center[1],
                        center[2] if iz else self.bounds[4],
                        self.bounds[5] if iz else center[2]
                    ])
                    child = OctreeNode(
                        bounds=child_bounds,
                        depth=self.depth + 1,
                        data=self.data  # Inherit parent's data
                    )
                    children.append(child)

        return tuple(children)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is within this node's bounds.

        Parameters
        ----------
        point : np.ndarray
            3D point as [x, y, z] array.

        Returns
        -------
        bool
            True if point is within bounds, False otherwise.
        """
        if point.shape != (3,):
            raise ValueError(f"point must be 3D, got shape {point.shape}")

        return (self.bounds[0] <= point[0] <= self.bounds[1] and
                self.bounds[2] <= point[1] <= self.bounds[3] and
                self.bounds[4] <= point[2] <= self.bounds[5])


class VoxelState(Type):
    """Voxel-based state representation.

    Represents a state as a voxel grid with occupancy probabilities.
    Useful for volumetric tracking, 3D occupancy mapping, and extended
    object tracking.

    The occupancy can be stored as either a dense numpy array or a sparse
    dictionary mapping voxel indices to probability values. Sparse storage
    is more efficient for grids with many empty voxels.

    Example
    -------
    >>> from datetime import datetime
    >>> # Create a voxel grid
    >>> grid = VoxelGrid(
    ...     bounds=np.array([0, 10, 0, 10, 0, 10]),
    ...     resolution=1.0
    ... )
    >>> # Create dense occupancy array
    >>> occupancy = np.random.random((10, 10, 10))
    >>> state = VoxelState(
    ...     grid=grid,
    ...     occupancy=occupancy,
    ...     timestamp=datetime.now()
    ... )
    >>> # Query probability at a point
    >>> prob = state.probability_at(np.array([5.5, 3.2, 8.7]))
    >>> # Convert to Gaussian approximation
    >>> gaussian_state = state.to_gaussian()
    """

    grid: VoxelGrid = Property(
        doc="VoxelGrid defining the spatial structure."
    )
    occupancy: Union[np.ndarray, dict] = Property(
        doc="Occupancy probabilities for voxels. Can be either:\n"
            "- Dense: numpy array with shape matching grid.shape\n"
            "- Sparse: dict mapping voxel indices (i,j,k) to probability values\n"
            "Values should be in [0, 1] representing probability of occupancy."
    )
    timestamp: datetime.datetime = Property(
        default=None,
        doc="Timestamp of the state. Default None."
    )

    def __init__(self, grid, occupancy, *args, **kwargs):
        # Validate occupancy format
        if isinstance(occupancy, np.ndarray):
            if occupancy.shape != grid.shape:
                raise ValueError(
                    f"Dense occupancy shape {occupancy.shape} does not match "
                    f"grid shape {grid.shape}")
        elif isinstance(occupancy, dict):
            # Validate sparse indices
            for idx in occupancy.keys():
                if not isinstance(idx, tuple) or len(idx) != 3:
                    raise ValueError(
                        f"Sparse occupancy keys must be 3-tuples, got {idx}")
                if not all(isinstance(i, (int, np.integer)) for i in idx):
                    raise ValueError(
                        f"Sparse occupancy indices must be integers, got {idx}")
        else:
            raise TypeError(
                f"occupancy must be numpy array or dict, got {type(occupancy)}")

        super().__init__(grid, occupancy, *args, **kwargs)

    @property
    def is_sparse(self) -> bool:
        """Whether occupancy is stored in sparse format."""
        return isinstance(self.occupancy, dict)

    def probability_at(self, point: np.ndarray) -> float:
        """Get occupancy probability at a 3D point.

        Parameters
        ----------
        point : np.ndarray
            3D point as [x, y, z] array.

        Returns
        -------
        float
            Occupancy probability at the point, or 0.0 if point is outside grid.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> occupancy = np.ones((10, 10, 10)) * 0.5
        >>> state = VoxelState(grid=grid, occupancy=occupancy)
        >>> state.probability_at(np.array([5.5, 5.5, 5.5]))
        0.5
        """
        indices = self.grid.voxel_indices(point)
        if indices is None:
            return 0.0

        if self.is_sparse:
            return self.occupancy.get(indices, 0.0)
        else:
            return float(self.occupancy[indices])

    def to_gaussian(self) -> 'GaussianState':
        """Convert voxel representation to Gaussian state approximation.

        Computes the mean and covariance of the occupied voxels weighted
        by their occupancy probabilities.

        Returns
        -------
        GaussianState
            Gaussian approximation with 3D state vector [x, y, z] and
            covariance matrix.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> occupancy = np.zeros((10, 10, 10))
        >>> occupancy[5, 5, 5] = 1.0  # Single occupied voxel
        >>> state = VoxelState(grid=grid, occupancy=occupancy)
        >>> gaussian = state.to_gaussian()
        >>> gaussian.state_vector
        array([[5.5],
               [5.5],
               [5.5]])
        """
        from .state import GaussianState

        # Collect occupied voxel centers and weights
        points = []
        weights = []

        if self.is_sparse:
            for indices, prob in self.occupancy.items():
                if prob > 0:
                    center = self.grid.voxel_center(indices)
                    points.append(center)
                    weights.append(prob)
        else:
            shape = self.grid.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        prob = self.occupancy[i, j, k]
                        if prob > 0:
                            center = self.grid.voxel_center((i, j, k))
                            points.append(center)
                            weights.append(prob)

        if not points:
            # No occupied voxels, return state at grid center with large covariance
            center = np.array([
                (self.grid.bounds[0] + self.grid.bounds[1]) / 2,
                (self.grid.bounds[2] + self.grid.bounds[3]) / 2,
                (self.grid.bounds[4] + self.grid.bounds[5]) / 2
            ])
            dims = self.grid.dimensions
            covar = np.diag((dims / 2) ** 2)  # Large uncertainty
            return GaussianState(
                state_vector=StateVector(center),
                covar=CovarianceMatrix(covar),
                timestamp=self.timestamp
            )

        points = np.array(points)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Compute weighted mean
        mean = np.sum(points * weights[:, np.newaxis], axis=0)

        # Compute weighted covariance
        centered = points - mean
        covar = (centered.T @ np.diag(weights) @ centered)

        # Ensure minimum covariance based on voxel resolution
        min_var = (self.grid.resolution / 2) ** 2
        covar += np.eye(3) * min_var

        return GaussianState(
            state_vector=StateVector(mean),
            covar=CovarianceMatrix(covar),
            timestamp=self.timestamp
        )

    @classmethod
    def from_gaussian(cls, gaussian_state: 'GaussianState', grid: VoxelGrid,
                      threshold: float = 0.01) -> 'VoxelState':
        """Create voxel state from Gaussian state.

        Evaluates the Gaussian PDF at each voxel center to create an
        occupancy distribution.

        Parameters
        ----------
        gaussian_state : GaussianState
            Gaussian state with 3D state vector and covariance.
        grid : VoxelGrid
            Grid structure for the voxel representation.
        threshold : float, optional
            Minimum probability threshold for sparse storage. Voxels with
            probability below this threshold are not stored. Default 0.01.

        Returns
        -------
        VoxelState
            Voxel state with occupancy computed from Gaussian PDF.

        Example
        -------
        >>> from stonesoup.types.state import GaussianState
        >>> mean = StateVector([5, 5, 5])
        >>> covar = CovarianceMatrix(np.eye(3))
        >>> gaussian = GaussianState(state_vector=mean, covar=covar)
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> voxel_state = VoxelState.from_gaussian(gaussian, grid, threshold=0.01)
        """
        if gaussian_state.state_vector.shape[0] != 3:
            raise ValueError(
                f"Gaussian state must be 3D, got {gaussian_state.state_vector.shape[0]}D")

        mean = gaussian_state.state_vector.flatten()
        covar = gaussian_state.covar

        # Create multivariate normal distribution
        rv = multivariate_normal(mean=mean, cov=covar, allow_singular=True)

        # Evaluate at voxel centers
        occupancy = {}
        max_prob = 0.0

        shape = grid.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    center = grid.voxel_center((i, j, k))
                    prob = rv.pdf(center)
                    if prob > threshold:
                        occupancy[(i, j, k)] = prob
                        max_prob = max(max_prob, prob)

        # Normalize to [0, 1] range
        if max_prob > 0:
            occupancy = {idx: prob / max_prob for idx, prob in occupancy.items()}

        return cls(
            grid=grid,
            occupancy=occupancy,
            timestamp=gaussian_state.timestamp
        )

    def to_dense(self) -> 'VoxelState':
        """Convert sparse occupancy to dense array representation.

        Returns
        -------
        VoxelState
            New voxel state with dense occupancy array.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> sparse_occ = {(5, 5, 5): 0.9, (6, 6, 6): 0.8}
        >>> state = VoxelState(grid=grid, occupancy=sparse_occ)
        >>> dense_state = state.to_dense()
        >>> dense_state.occupancy.shape
        (10, 10, 10)
        """
        if not self.is_sparse:
            return VoxelState(
                grid=self.grid,
                occupancy=self.occupancy.copy(),
                timestamp=self.timestamp
            )

        # Create dense array
        dense = np.zeros(self.grid.shape)
        for indices, prob in self.occupancy.items():
            dense[indices] = prob

        return VoxelState(
            grid=self.grid,
            occupancy=dense,
            timestamp=self.timestamp
        )

    def to_sparse(self, threshold: float = 1e-6) -> 'VoxelState':
        """Convert dense occupancy to sparse dictionary representation.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability to store in sparse representation. Default 1e-6.

        Returns
        -------
        VoxelState
            New voxel state with sparse occupancy dictionary.

        Example
        -------
        >>> grid = VoxelGrid(bounds=np.array([0, 10, 0, 10, 0, 10]), resolution=1.0)
        >>> dense_occ = np.zeros((10, 10, 10))
        >>> dense_occ[5, 5, 5] = 0.9
        >>> state = VoxelState(grid=grid, occupancy=dense_occ)
        >>> sparse_state = state.to_sparse()
        >>> sparse_state.occupancy
        {(5, 5, 5): 0.9}
        """
        if self.is_sparse:
            # Filter by threshold
            sparse = {idx: prob for idx, prob in self.occupancy.items()
                      if prob >= threshold}
            return VoxelState(
                grid=self.grid,
                occupancy=sparse,
                timestamp=self.timestamp
            )

        # Convert dense to sparse
        sparse = {}
        shape = self.grid.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    prob = self.occupancy[i, j, k]
                    if prob >= threshold:
                        sparse[(i, j, k)] = float(prob)

        return VoxelState(
            grid=self.grid,
            occupancy=sparse,
            timestamp=self.timestamp
        )
