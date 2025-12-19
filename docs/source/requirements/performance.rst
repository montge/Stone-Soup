Performance Requirements
========================

This document defines performance requirements for Stone Soup.

Computational Performance
-------------------------

.. perf:: Coordinate Transformation Speed
   :id: PERF-COORD-001
   :status: implemented
   :tags: performance, coordinates

   Geodetic to ECEF coordinate transformations shall complete
   in less than 20 microseconds per transformation on standard hardware.

.. perf:: Kalman Filter Performance
   :id: PERF-KF-001
   :status: implemented
   :tags: performance, kalman

   Kalman filter predict and update operations shall scale as O(n³)
   where n is the state dimension, with efficient implementations
   for small to medium state dimensions (n < 20).

.. perf:: Particle Filter Scalability
   :id: PERF-PF-001
   :status: implemented
   :tags: performance, particle

   Particle filter operations shall scale linearly O(N) with the
   number of particles, supporting efficient processing of
   10,000+ particles.

Numerical Accuracy
------------------

.. perf:: Coordinate Roundtrip Accuracy
   :id: PERF-ACC-001
   :status: implemented
   :tags: accuracy, coordinates

   Geodetic-ECEF-Geodetic coordinate roundtrip transformations
   shall maintain sub-millimeter accuracy.

.. perf:: ECI-ECEF Accuracy Levels
   :id: PERF-ACC-002
   :status: implemented
   :tags: accuracy, coordinates

   ECI-ECEF transformations shall achieve the following accuracy levels:

   - Simple mode: ~100 km (suitable for quick estimates)
   - Standard mode: ~1-10 m (suitable for LEO tracking)
   - High-precision mode: ~1 cm (suitable for GNSS applications)

.. perf:: Numerical Stability
   :id: PERF-NUM-001
   :status: implemented
   :tags: accuracy, numerical

   All matrix operations shall use numerically stable algorithms
   (e.g., Cholesky decomposition) to avoid accumulation of
   floating-point errors.

Memory Efficiency
-----------------

.. perf:: Sparse Voxel Storage
   :id: PERF-MEM-001
   :status: implemented
   :tags: memory, voxel

   Voxel states shall support sparse storage to reduce memory
   usage when most voxels are unoccupied.

.. perf:: Track Memory Management
   :id: PERF-MEM-002
   :status: implemented
   :tags: memory, tracks

   Track objects shall support configurable state history limits
   to bound memory usage in long-running applications.
