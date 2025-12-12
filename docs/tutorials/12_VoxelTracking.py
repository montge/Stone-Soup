#!/usr/bin/env python

"""
=====================================
12 - Voxel-Based Volumetric Tracking
=====================================
"""

# %%
# This tutorial introduces Stone Soup's voxel-based tracking capabilities for
# volumetric state estimation and extended object tracking.
#
# Voxel tracking is useful for:
#
# - 3D occupancy mapping
# - Extended object tracking (objects larger than a point)
# - Track-before-detect scenarios
# - Volumetric sensor fusion
#
# Background
# ----------
#
# Voxel-based tracking represents spatial distributions using a 3D grid of cells
# (voxels), each storing an occupancy probability. This approach is particularly
# useful when:
#
# - Object extent is significant compared to sensor resolution
# - Multiple objects may occupy overlapping regions
# - Measurements provide volumetric information (e.g., LIDAR, radar returns)
#
# Creating a Voxel Grid
# ---------------------
#
# A voxel grid defines the spatial extent and resolution of the tracking volume:

# %%
import numpy as np
from stonesoup.types.voxel import VoxelGrid, VoxelState

# Create a 20x20x10 meter grid with 1m resolution
grid = VoxelGrid(
    bounds=np.array([-10, 10, -10, 10, 0, 10]),  # [x_min, x_max, y_min, y_max, z_min, z_max]
    resolution=1.0  # 1 meter voxels
)

print(f"Grid dimensions: {grid.dimensions} m")
print(f"Grid shape: {grid.shape} voxels")
print(f"Total voxels: {grid.num_voxels}")

# %%
# Voxel Indexing
# ^^^^^^^^^^^^^^
#
# The grid provides methods to convert between 3D coordinates and voxel indices:

# %%
# Check if a point is within the grid
point = np.array([5.5, -3.2, 4.8])
print(f"Point {point} in grid: {grid.contains(point)}")

# Get voxel indices for a point
indices = grid.voxel_indices(point)
print(f"Voxel indices: {indices}")

# Get voxel center coordinates
center = grid.voxel_center(indices)
print(f"Voxel center: {center}")

# %%
# Creating a Voxel State
# ----------------------
#
# A VoxelState combines a grid with occupancy probabilities:

# %%
from datetime import datetime

# Create occupancy array (same shape as grid)
occupancy = np.zeros(grid.shape)

# Set high occupancy at a few locations
occupancy[15, 10, 5] = 0.9  # Near (5, 0, 5)
occupancy[14, 10, 5] = 0.7
occupancy[15, 11, 5] = 0.6

# Create voxel state
voxel_state = VoxelState(
    grid=grid,
    occupancy=occupancy,
    timestamp=datetime.now()
)

print(f"State timestamp: {voxel_state.timestamp}")
print(f"Max occupancy: {voxel_state.occupancy.max():.2f}")
print(f"Occupied voxels (>0.5): {np.sum(voxel_state.occupancy > 0.5)}")

# %%
# Converting Between Voxel and Gaussian Representations
# -----------------------------------------------------
#
# VoxelState provides methods to convert to/from Gaussian state representations:

# %%
# Get Gaussian state approximation (weighted centroid and covariance)
gaussian_approx = voxel_state.to_gaussian()
if gaussian_approx is not None:
    print(f"Gaussian position estimate: {gaussian_approx.state_vector.flatten()}")
    print(f"Gaussian covariance diagonal: {np.diag(gaussian_approx.covar)}")

# %%
# Create voxel state from a Gaussian distribution
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState

# Define a Gaussian state
gaussian_state = GaussianState(
    state_vector=StateVector([2.0, -1.0, 3.0]),
    covar=CovarianceMatrix(np.diag([1.0, 1.0, 0.5])),
    timestamp=datetime.now()
)

# Create voxel state from Gaussian (returns sparse representation)
voxel_from_gaussian = VoxelState.from_gaussian(gaussian_state, grid, threshold=0.01)

# Convert to dense for easier manipulation
voxel_dense = voxel_from_gaussian.to_dense()
print(f"Voxels with occupancy > 0.1: {np.sum(voxel_dense.occupancy > 0.1)}")

# %%
# Octree-Based Adaptive Resolution
# --------------------------------
#
# For large volumes with localized objects, use OctreeNode for adaptive resolution:

# %%
from stonesoup.types.voxel import OctreeNode

# Create root node covering the full volume
root = OctreeNode(
    bounds=np.array([-10, 10, -10, 10, 0, 10]),
    depth=0  # Root is at depth 0
)

print(f"Root node depth: {root.depth}")
print(f"Root node volume: {root.volume:.1f} m^3")

# %%
# Subdivide the octree into 8 octant children
children = root.subdivide()
print(f"After subdivision: {len(children)} children")

# Check if a point is in a specific node
point = np.array([5.0, 5.0, 5.0])
print(f"Point {point} is in root: {root.contains(point)}")

# Find which child contains the point
for i, child in enumerate(children):
    if child.contains(point):
        print(f"Point is in child {i} at depth {child.depth}, volume {child.volume:.1f} m^3")

# %%
# Voxel Predictor
# ---------------
#
# The VoxelPredictor propagates occupancy probabilities through time:

# %%
from stonesoup.predictor.voxel import VoxelPredictor
from stonesoup.models.transition.linear import ConstantVelocity

# Note: VoxelPredictor requires a compatible transition model
# For demonstration, we'll show the concept

# Create a simple diffusion-based predictor
# predictor = VoxelPredictor(
#     transition_model=transition_model,
#     birth_probability=0.01,  # New objects appearing
#     death_probability=0.05   # Objects disappearing
# )

# %%
# The predictor applies:
#
# 1. **Motion model**: Spreads occupancy based on expected motion
# 2. **Birth process**: Small probability of new occupancy appearing
# 3. **Death process**: Occupied voxels may become empty
#
# Voxel Updater
# -------------
#
# The VoxelUpdater updates occupancy using Bayesian inference:

# %%
from stonesoup.updater.voxel import VoxelUpdater

# Note: Requires a measurement model
# updater = VoxelUpdater(
#     measurement_model=measurement_model,
#     detection_probability=0.9,
#     clutter_intensity=1e-6
# )

# %%
# The update follows Bayesian occupancy grid rules:
#
# - **Detection**: Increases occupancy where measurements are received
# - **Missed detection**: Decreases occupancy in observed empty regions
#
# Multi-Sensor Fusion
# -------------------
#
# Voxel states naturally support multi-sensor fusion by updating the same
# occupancy grid with measurements from different sensors:
#
# .. code-block:: python
#
#     # Fuse measurements from multiple sensors
#     for sensor in sensors:
#         detections = sensor.measure(voxel_state)
#         for detection in detections:
#             hypothesis = SingleHypothesis(prediction, detection)
#             voxel_state = updater.update(hypothesis)
#
# Visualization Example
# ---------------------
#
# Here's how to visualize voxel occupancy:

# %%
import matplotlib.pyplot as plt

# Find occupied voxels
occupied_mask = voxel_state.occupancy > 0.3
occupied_indices = np.argwhere(occupied_mask)

if len(occupied_indices) > 0:
    # Convert indices to coordinates
    x = (occupied_indices[:, 0] * grid.resolution + grid.bounds[0] + grid.resolution/2)
    y = (occupied_indices[:, 1] * grid.resolution + grid.bounds[2] + grid.resolution/2)
    z = (occupied_indices[:, 2] * grid.resolution + grid.bounds[4] + grid.resolution/2)

    # Get occupancy values for coloring
    colors = [voxel_state.occupancy[tuple(idx)] for idx in occupied_indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=100, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Voxel Occupancy')
    plt.colorbar(scatter, label='Occupancy Probability')
    plt.tight_layout()

# %%
# Summary
# -------
#
# Stone Soup's voxel tracking provides:
#
# - Flexible voxel grid definitions with configurable resolution
# - Octree-based adaptive resolution for efficient storage
# - Conversion between continuous and discrete state representations
# - Bayesian occupancy update framework
# - Support for birth/death processes
# - Multi-sensor fusion capabilities
#
# Voxel tracking is ideal for scenarios where object extent matters or
# traditional point-target assumptions don't hold.

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/VoxelTracking.png'
