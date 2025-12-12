State Estimation Requirements
=============================

This document defines requirements for state estimation algorithms
and coordinate systems in Stone Soup.

Coordinate Systems
------------------

.. req:: Geodetic Coordinates
   :id: REQ-COORD-001
   :status: implemented
   :tags: coordinates, geodetic

   The system shall support geodetic coordinates (latitude, longitude, altitude)
   with multiple reference ellipsoids (WGS84, GRS80, PZ-90, CGCS2000).

.. req:: ECEF Coordinates
   :id: REQ-COORD-002
   :status: implemented
   :tags: coordinates, ecef
   :satisfies: REQ-COORD-001

   The system shall support Earth-Centered Earth-Fixed (ECEF) Cartesian
   coordinates with transformations to/from geodetic coordinates.

.. req:: ECI Coordinates
   :id: REQ-COORD-003
   :status: implemented
   :tags: coordinates, eci

   The system shall support Earth-Centered Inertial (ECI) coordinates
   for space tracking applications.

.. req:: ECI-ECEF Transformation
   :id: REQ-COORD-004
   :status: implemented
   :tags: coordinates, transformation
   :satisfies: REQ-COORD-002, REQ-COORD-003

   The system shall support ECI to ECEF coordinate transformations
   at multiple precision levels:

   - Simple (ERA only): ~100 km accuracy
   - Standard (ERA + precession + nutation): ~1-10 m accuracy
   - High-precision (with EOP): ~1 cm accuracy

.. req:: Topocentric Frames
   :id: REQ-COORD-005
   :status: implemented
   :tags: coordinates, topocentric

   The system shall support topocentric reference frames including
   ENU (East-North-Up), NED (North-East-Down), and SEZ (South-East-Zenith).

Voxel Tracking
--------------

.. req:: Voxel Grid Support
   :id: REQ-VOXEL-001
   :status: implemented
   :tags: voxel, volumetric

   The system shall support voxel grid representations for
   volumetric state estimation with configurable resolution and bounds.

.. req:: Octree Adaptive Resolution
   :id: REQ-VOXEL-002
   :status: implemented
   :tags: voxel, octree
   :satisfies: REQ-VOXEL-001

   The system shall support octree-based adaptive resolution
   for efficient storage of sparse volumetric data.

.. req:: Voxel State Conversion
   :id: REQ-VOXEL-003
   :status: implemented
   :tags: voxel, conversion
   :satisfies: REQ-VOXEL-001, REQ-STATE-003

   The system shall support conversion between voxel states
   and Gaussian state representations.

.. req:: Voxel Prediction
   :id: REQ-VOXEL-004
   :status: implemented
   :tags: voxel, prediction
   :satisfies: REQ-VOXEL-001

   The system shall implement voxel-based prediction with
   motion model support and birth/death processes.

.. req:: Voxel Update
   :id: REQ-VOXEL-005
   :status: implemented
   :tags: voxel, update
   :satisfies: REQ-VOXEL-001

   The system shall implement Bayesian voxel occupancy updates
   based on sensor measurements.

Viterbi Tracking
----------------

.. req:: Viterbi Smoothing
   :id: REQ-VIT-001
   :status: implemented
   :tags: viterbi, smoothing

   The system shall implement Viterbi smoothing for finding
   the maximum a posteriori (MAP) state sequence.

.. req:: Track-Before-Detect
   :id: REQ-VIT-002
   :status: implemented
   :tags: viterbi, tbd
   :satisfies: REQ-VIT-001

   The system shall implement Viterbi-based track-before-detect
   for extracting tracks from raw detections in low-SNR scenarios.

.. req:: Graph-Constrained Tracking
   :id: REQ-VIT-003
   :status: implemented
   :tags: viterbi, graph
   :satisfies: REQ-VIT-001

   The system shall implement graph-constrained Viterbi smoothing
   for tracking on road and rail networks.
