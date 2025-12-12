Test Traceability
=================

This document links automated tests to their corresponding requirements,
providing full requirements traceability for Stone Soup.

State Type Tests
----------------

.. test:: StateVector Tests
   :id: TEST-STATE-001
   :status: implemented
   :tests: REQ-STATE-001

   Tests for StateVector functionality in ``stonesoup/types/tests/test_array.py``
   and ``stonesoup/types/tests/test_state.py``.

   Verifies:

   - StateVector creation with arbitrary dimensions
   - Element access and modification
   - Vector arithmetic operations (add, subtract, scale)
   - Proper numpy array subclass behavior

.. test:: CovarianceMatrix Tests
   :id: TEST-STATE-002
   :status: implemented
   :tests: REQ-STATE-002

   Tests for CovarianceMatrix functionality in ``stonesoup/types/tests/test_array.py``.

   Verifies:

   - CovarianceMatrix creation
   - Symmetric matrix properties
   - Matrix operations (add, multiply, inverse)
   - Positive semi-definite validation

.. test:: GaussianState Tests
   :id: TEST-STATE-003
   :status: implemented
   :tests: REQ-STATE-003

   Tests for GaussianState in ``stonesoup/types/tests/test_state.py``.

   Verifies:

   - GaussianState creation with mean and covariance
   - Dimension consistency between state vector and covariance
   - Timestamp handling
   - State copying and equality

.. test:: ParticleState Tests
   :id: TEST-STATE-004
   :status: implemented
   :tests: REQ-STATE-004

   Tests for ParticleState in ``stonesoup/types/tests/test_state.py``.

   Verifies:

   - ParticleState creation with particles and weights
   - Weight normalization
   - Particle resampling interface
   - Mean and covariance computation from particles

Predictor Tests
---------------

.. test:: Kalman Predictor Tests
   :id: TEST-PRED-001
   :status: implemented
   :tests: REQ-PRED-001

   Tests for KalmanPredictor in ``stonesoup/predictor/tests/test_kalman.py``.

   Verifies:

   - Linear state prediction with transition matrix
   - Covariance propagation with process noise
   - Constant velocity model predictions
   - Time interval handling

.. test:: Extended Kalman Predictor Tests
   :id: TEST-PRED-002
   :status: implemented
   :tests: REQ-PRED-002

   Tests for ExtendedKalmanPredictor in ``stonesoup/predictor/tests/test_kalman.py``.

   Verifies:

   - Nonlinear state prediction using Jacobian
   - Linearization accuracy for various motion models
   - Covariance propagation through nonlinear functions

.. test:: Unscented Kalman Predictor Tests
   :id: TEST-PRED-003
   :status: implemented
   :tests: REQ-PRED-003

   Tests for UnscentedKalmanPredictor in ``stonesoup/predictor/tests/test_kalman.py``.

   Verifies:

   - Sigma point generation and propagation
   - Mean and covariance recovery from sigma points
   - Scaling parameter effects
   - Comparison with EKF for nonlinear models

.. test:: Particle Predictor Tests
   :id: TEST-PRED-004
   :status: implemented
   :tests: REQ-PRED-004

   Tests for ParticlePredictor in ``stonesoup/predictor/tests/test_particle.py``.

   Verifies:

   - Particle propagation through motion model
   - Process noise sampling
   - Weight preservation during prediction
   - Particle diversity maintenance

Updater Tests
-------------

.. test:: Kalman Updater Tests
   :id: TEST-UPD-001
   :status: implemented
   :tests: REQ-UPD-001

   Tests for KalmanUpdater in ``stonesoup/updater/tests/test_kalman.py``.

   Verifies:

   - Measurement incorporation
   - Kalman gain computation
   - Covariance reduction after update
   - Innovation (residual) calculation

.. test:: Extended Kalman Updater Tests
   :id: TEST-UPD-002
   :status: implemented
   :tests: REQ-UPD-002

   Tests for ExtendedKalmanUpdater in ``stonesoup/updater/tests/test_kalman.py``.

   Verifies:

   - Nonlinear measurement update using Jacobian
   - Measurement prediction accuracy
   - Update with various measurement models

.. test:: Particle Updater Tests
   :id: TEST-UPD-003
   :status: implemented
   :tests: REQ-UPD-003

   Tests for ParticleUpdater in ``stonesoup/updater/tests/test_particle.py``.

   Verifies:

   - Particle weight update based on likelihood
   - Weight normalization
   - Effective sample size computation
   - Resampling trigger conditions

Data Association Tests
----------------------

.. test:: Nearest Neighbor Tests
   :id: TEST-ASSOC-001
   :status: implemented
   :tests: REQ-ASSOC-001

   Tests for NearestNeighbourDataAssociator in
   ``stonesoup/dataassociator/tests/test_neighbour.py``.

   Verifies:

   - Single hypothesis assignment
   - Distance metric computation
   - Gating behavior
   - Association with multiple tracks

.. test:: Global Nearest Neighbor Tests
   :id: TEST-ASSOC-002
   :status: implemented
   :tests: REQ-ASSOC-002

   Tests for GNNDataAssociator in
   ``stonesoup/dataassociator/tests/test_neighbour.py``.

   Verifies:

   - Optimal assignment solving
   - Cost matrix construction
   - Multiple detection handling
   - Missed detection handling

.. test:: MHT Tests
   :id: TEST-ASSOC-003
   :status: implemented
   :tests: REQ-ASSOC-003

   Tests for multi-hypothesis tracking in
   ``stonesoup/dataassociator/tests/test_mht.py``.

   Verifies:

   - Hypothesis tree generation
   - Hypothesis pruning
   - N-scan back logic
   - Multiple hypothesis maintenance

Track Management Tests
----------------------

.. test:: Track Initiator Tests
   :id: TEST-TRACK-001
   :status: implemented
   :tests: REQ-TRACK-001

   Tests for initiators in ``stonesoup/initiator/tests/``.

   Verifies:

   - Single-point initiation
   - Multi-point initiation (M-of-N logic)
   - Initial state estimation
   - Initial covariance setting

.. test:: Track Deleter Tests
   :id: TEST-TRACK-002
   :status: implemented
   :tests: REQ-TRACK-002

   Tests for deleters in ``stonesoup/deleter/tests/``.

   Verifies:

   - Time-based deletion
   - Missed detection counting
   - Covariance-based deletion
   - Multi-criteria deletion

.. test:: Smoother Tests
   :id: TEST-TRACK-003
   :status: implemented
   :tests: REQ-TRACK-003

   Tests for smoothers in ``stonesoup/smoother/tests/``.

   Verifies:

   - Kalman smoothing (RTS smoother)
   - Backward pass computation
   - Smoothed state accuracy improvement
   - Particle smoothing

Traceability Matrix
-------------------

.. needtable::
   :columns: id;title;status;tests;satisfies
   :filter: type == 'test' or type == 'req'
   :style: table
   :sort: id

Requirements Coverage
---------------------

.. needflow::
   :filter: type in ['req', 'test']
   :show_link_names:
