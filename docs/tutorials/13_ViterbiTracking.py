#!/usr/bin/env python

"""
===================================
13 - Viterbi Algorithm for Tracking
===================================
"""

# %%
# This tutorial introduces Stone Soup's Viterbi algorithm implementations for
# target tracking. The Viterbi algorithm finds the most likely sequence of hidden
# states in a Hidden Markov Model (HMM), making it powerful for:
#
# - Track smoothing (finding optimal state sequence given all measurements)
# - Track-before-detect (TBD) scenarios
# - Graph-constrained tracking (vehicles on road/rail networks)
#
# Background
# ----------
#
# The Viterbi algorithm differs from standard Kalman filtering in a fundamental way:
#
# - **Kalman filter**: Estimates marginal distribution at each time step
# - **Viterbi**: Finds the single most likely complete state sequence
#
# This is particularly useful when:
#
# - The MAP (maximum a posteriori) sequence is needed, not just per-time estimates
# - State space can be discretized or is naturally discrete
# - Temporal consistency is important (e.g., vehicles staying on roads)
#
# Viterbi Smoothing
# -----------------
#
# The ViterbiSmoother takes an existing track and finds the optimal state sequence
# by considering all measurements simultaneously through forward-backward processing.

# %%
import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity
)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.smoother.viterbi import ViterbiSmoother

# %%
# First, let's define the motion and measurement models:

# 2D constant velocity model
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.5), ConstantVelocity(0.5)]
)

# Measure position only (not velocity)
measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=[0, 2],
    noise_covar=np.eye(2) * 2.0
)

# %%
# Create a simple track with some filtered states:

start_time = datetime(2024, 1, 1, 12, 0, 0)

# Simulate a simple track - object moving in positive x and y direction
states = []
for i in range(10):
    # True state: moving at roughly 5 m/s in x and 3 m/s in y
    x = 10 + i * 5 + np.random.randn() * 1.5  # Add some noise
    y = 20 + i * 3 + np.random.randn() * 1.5

    state = GaussianState(
        state_vector=StateVector([x, 5.0, y, 3.0]),  # [x, vx, y, vy]
        covar=CovarianceMatrix(np.diag([4.0, 1.0, 4.0, 1.0])),
        timestamp=start_time + timedelta(seconds=i)
    )
    states.append(state)

track = Track(states)
print(f"Track length: {len(track)} states")
print(f"First state: x={track[0].state_vector[0, 0]:.1f}, y={track[0].state_vector[2, 0]:.1f}")
print(f"Last state: x={track[-1].state_vector[0, 0]:.1f}, y={track[-1].state_vector[2, 0]:.1f}")

# %%
# Now apply Viterbi smoothing to find the optimal state sequence:

smoother = ViterbiSmoother(
    transition_model=transition_model,
    measurement_model=measurement_model,
    num_states=50  # Number of discrete states in the quantization grid
)

smoothed_track = smoother.smooth(track)
print(f"Smoothed track length: {len(smoothed_track)} states")

# %%
# Compare original and smoothed positions:

print("\nPosition comparison (x, y):")
print("-" * 50)
for i in range(0, len(track), 3):  # Every 3rd state
    orig_x = track[i].state_vector[0, 0]
    orig_y = track[i].state_vector[2, 0]
    smooth_x = smoothed_track[i].state_vector[0, 0]
    smooth_y = smoothed_track[i].state_vector[2, 0]
    print(f"t={i}: Original ({orig_x:.1f}, {orig_y:.1f}) -> Smoothed ({smooth_x:.1f}, {smooth_y:.1f})")

# %%
# The Viterbi algorithm discretizes the state space and finds the sequence of
# discrete states with maximum likelihood. The `num_states` parameter controls
# the resolution of this discretization.
#
# Track-Before-Detect with Viterbi
# --------------------------------
#
# The ViterbiTrackInitiator uses the Viterbi algorithm for track-before-detect
# (TBD), where we want to find tracks directly from raw detections without
# first applying a detection threshold.
#
# This is useful in low SNR scenarios where:
#
# - Single-scan detection is unreliable
# - Temporal continuity provides additional discrimination power
# - False alarm rates are high

# %%
from stonesoup.initiator.viterbi import ViterbiTrackInitiator
from stonesoup.types.detection import Detection

# Create the Viterbi initiator
initiator = ViterbiTrackInitiator(
    transition_model=transition_model,
    measurement_model=measurement_model,
    num_scans=3,  # Use 3 consecutive scans
    detection_threshold=-100.0,  # Low threshold (log-likelihood sum)
    max_detections_per_scan=50,
    missed_detection_penalty=-10.0
)

# %%
# Simulate a simple scenario with one target and random clutter:

detection_sets = []
tbd_timestamps = []

# True target: moving at ~1 m/s (consistent with transition model process noise)
for scan_idx in range(3):
    timestamp = start_time + timedelta(seconds=scan_idx)
    tbd_timestamps.append(timestamp)

    detections = set()

    # True target detection - consistent trajectory
    x_det = float(scan_idx)       # ~1 m/s in x
    y_det = float(scan_idx * 2)   # ~2 m/s in y
    detections.add(Detection(
        state_vector=StateVector([x_det, y_det]),
        timestamp=timestamp,
        measurement_model=measurement_model
    ))

    # One clutter detection far away (random location)
    detections.add(Detection(
        state_vector=StateVector([50.0 + scan_idx * 10, 50.0]),
        timestamp=timestamp,
        measurement_model=measurement_model
    ))

    detection_sets.append(detections)
    print(f"Scan {scan_idx}: {len(detections)} detections")

# %%
# Run the Viterbi track initiator:

tracks = initiator.initiate_from_scans(detection_sets, tbd_timestamps)
print(f"\nInitiated {len(tracks)} track(s)")

# %%
# The initiator finds trajectories that are dynamically plausible according
# to the motion model. The number of tracks depends on:
#
# - detection_threshold: cumulative log-likelihood threshold
# - transition model parameters: determine what motions are "feasible"
# - detection scores: if provided in metadata
#
# Examine any initiated tracks:

for i, track in enumerate(tracks):
    print(f"\nTrack {i+1}:")
    for state in track:
        x = state.state_vector[0, 0]
        y = state.state_vector[2, 0]
        print(f"  Position: ({x:.1f}, {y:.1f})")

if len(tracks) == 0:
    print("No tracks initiated (all trajectories below threshold)")

# %%
# The Viterbi TBD algorithm finds trajectories that are:
#
# 1. Dynamically consistent (following the motion model)
# 2. Have high cumulative detection scores
# 3. Represent globally optimal paths through the detection lattice
#
# Key tuning parameters:
#
# - Increase `detection_threshold` to reduce false tracks
# - Decrease it to find weaker targets
# - Adjust motion model noise to match expected target dynamics
#
# Graph-Constrained Viterbi
# -------------------------
#
# For tracking vehicles on road or rail networks, the GraphViterbiSmoother
# constrains the state space to a graph structure.

# %%
from stonesoup.smoother.graph_viterbi import GraphViterbiSmoother
import networkx as nx

# Create a simple road network graph
road_network = nx.DiGraph()

# Add nodes (intersections) with positions
nodes = {
    0: {'pos': np.array([0.0, 0.0])},
    1: {'pos': np.array([10.0, 0.0])},
    2: {'pos': np.array([20.0, 0.0])},
    3: {'pos': np.array([10.0, 10.0])},
    4: {'pos': np.array([20.0, 10.0])},
}

for node_id, attrs in nodes.items():
    road_network.add_node(node_id, **attrs)

# Add edges (roads) - bidirectional for simplicity
edges = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1),
         (2, 4), (4, 2), (3, 4), (4, 3)]

for edge in edges:
    road_network.add_edge(*edge)

print(f"Road network: {road_network.number_of_nodes()} nodes, {road_network.number_of_edges()} edges")

# %%
# Create the graph-constrained smoother:

graph_smoother = GraphViterbiSmoother(
    graph=road_network,
    transition_model=transition_model,
    measurement_model=measurement_model,
    off_road_penalty=-10.0  # Penalty for measurements far from graph
)

# %%
# Create a track with measurements near the road network:

graph_states = []
# Simulate vehicle moving along path: 0 -> 1 -> 2 -> 4
true_path = [
    np.array([1.0, 0.5]),    # Near node 0
    np.array([9.0, -0.5]),   # Near node 1
    np.array([19.0, 1.0]),   # Near node 2
    np.array([20.5, 9.0]),   # Near node 4
]

for i, true_pos in enumerate(true_path):
    # Add measurement noise
    meas_pos = true_pos + np.random.randn(2) * 1.5

    state = GaussianState(
        state_vector=StateVector([meas_pos[0], 0.0, meas_pos[1], 0.0]),
        covar=CovarianceMatrix(np.diag([4.0, 1.0, 4.0, 1.0])),
        timestamp=start_time + timedelta(seconds=i)
    )
    graph_states.append(state)

graph_track = Track(graph_states)

# %%
# Apply graph-constrained smoothing:

smoothed_graph_track = graph_smoother.smooth(graph_track)

print("Graph-constrained smoothing results:")
print("-" * 50)
for i, (orig, smoothed) in enumerate(zip(graph_track, smoothed_graph_track)):
    orig_x = orig.state_vector[0, 0]
    orig_y = orig.state_vector[2, 0]
    smooth_x = smoothed.state_vector[0, 0]
    smooth_y = smoothed.state_vector[2, 0]
    print(f"t={i}: Meas ({orig_x:.1f}, {orig_y:.1f}) -> On-graph ({smooth_x:.1f}, {smooth_y:.1f})")

# %%
# The graph-constrained smoother snaps the trajectory to valid paths on the
# road network, which can significantly improve tracking accuracy for vehicles.
#
# Choosing Parameters
# -------------------
#
# Key parameters for Viterbi algorithms:
#
# +------------------------+----------------------------------------+------------------+
# | Parameter              | Description                            | Typical Values   |
# +========================+========================================+==================+
# | num_states             | Grid resolution for state quantization | 50-200           |
# +------------------------+----------------------------------------+------------------+
# | state_bounds           | Bounds for state space discretization  | Data-dependent   |
# +------------------------+----------------------------------------+------------------+
# | detection_threshold    | Min cumulative score for TBD           | 5-20             |
# +------------------------+----------------------------------------+------------------+
# | num_scans              | Scans for TBD                          | 3-10             |
# +------------------------+----------------------------------------+------------------+
# | off_graph_penalty      | Penalty for off-network detections     | -5 to -20        |
# +------------------------+----------------------------------------+------------------+
#
# Summary
# -------
#
# Stone Soup's Viterbi implementations provide:
#
# - **ViterbiSmoother**: Find optimal state sequence for existing tracks
# - **ViterbiTrackInitiator**: Track-before-detect for low SNR scenarios
# - **GraphViterbiSmoother**: Network-constrained tracking for vehicles
#
# Key advantages of Viterbi over filtering approaches:
#
# - Finds globally optimal sequence over entire track
# - Can handle discrete or discretized state spaces
# - Naturally incorporates structural constraints (graphs)
# - Exploits temporal continuity for weak signals
#
# For most applications, standard Kalman filtering is sufficient. Use Viterbi when:
#
# - MAP sequence is needed (not just per-time estimates)
# - Operating in low SNR / track-before-detect scenarios
# - Tracking on constrained networks (roads, rails)

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/ViterbiTracking.png'
