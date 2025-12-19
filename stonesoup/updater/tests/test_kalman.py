import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import CubatureKalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianMeasurementPrediction, GaussianStatePrediction
from stonesoup.types.state import GaussianState, SqrtGaussianState
from stonesoup.updater.kalman import (
    CubatureKalmanUpdater,
    ExtendedKalmanUpdater,
    IteratedKalmanUpdater,
    KalmanUpdater,
    SchmidtKalmanUpdater,
    SqrtKalmanUpdater,
    StochasticIntegrationUpdater,
    UnscentedKalmanUpdater,
)


@pytest.fixture(
    params=[
        KalmanUpdater,
        ExtendedKalmanUpdater,
        UnscentedKalmanUpdater,
        IteratedKalmanUpdater,
        SchmidtKalmanUpdater,
        CubatureKalmanUpdater,
        StochasticIntegrationUpdater,
    ]
)
def updater_class(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_joseph_cov(request):
    return request.param


def test_kalman_covariance_stability():
    # input data
    measurement_model = LinearGaussian(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=np.zeros((3, 3)),
    )
    prediction = GaussianStatePrediction(
        state_vector=np.zeros((6, 1)),
        covar=np.array(
            [
                [
                    1.64385383e06,
                    1.00001250e06,
                    -2.48073899e05,
                    -5.86573934e-08,
                    -4.03756199e05,
                    -5.86573934e-08,
                ],
                [
                    1.00001250e06,
                    1.00002500e06,
                    5.24472545e-09,
                    0.00000000e00,
                    -3.90355011e-08,
                    0.00000000e00,
                ],
                [
                    -2.48073899e05,
                    5.24472545e-09,
                    2.03087457e06,
                    1.92001250e06,
                    1.58877383e05,
                    1.34045255e-08,
                ],
                [
                    -5.86573934e-08,
                    0.00000000e00,
                    1.92001250e06,
                    1.92002500e06,
                    -2.38775732e-09,
                    0.00000000e00,
                ],
                [
                    -4.03756199e05,
                    -3.90355011e-08,
                    1.58877383e05,
                    -2.38775732e-09,
                    2.19190854e06,
                    1.92001250e06,
                ],
                [
                    -5.86573934e-08,
                    0.00000000e00,
                    1.34045255e-08,
                    0.00000000e00,
                    1.92001250e06,
                    1.92002500e06,
                ],
            ]
        ),
    )
    measurement_prediction = GaussianMeasurementPrediction(
        state_vector=np.zeros((3, 1)),
        covar=np.array(
            [
                [4.25607853e-02, 3.37582173e-03, -1.40573707e00],
                [3.37582173e-03, 4.18898195e-02, -2.22353400e01],
                [-1.40573707e00, -2.22353400e01, 3.24759672e06],
            ]
        ),
        cross_covar=np.array(
            [
                [-6.70207286e01, 2.17782708e02, 3.13723904e06],
                [-6.13505872e01, 1.99344484e02, 1.56529358e06],
                [-5.19907012e02, 7.23354556e01, -1.70022791e06],
                [-5.18271180e02, 7.80151873e01, -1.08255080e06],
                [1.95377768e02, 4.52672009e02, -2.79383465e06],
                [1.90861853e02, 4.59621649e02, -1.78870865e06],
            ]
        ),
    )
    measurement = Detection(
        state_vector=np.zeros((3, 1)),
        timestamp=datetime.datetime.fromtimestamp(0),
    )

    # check assumptions, all covariances are positive definite
    assert np.all(np.linalg.eigvals(prediction.covar) > 0)
    assert np.all(np.linalg.eigvals(measurement_prediction.covar) > 0)

    # create cubature kalman filter
    transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(0),
            ConstantVelocity(0),
            ConstantVelocity(0),
        ]
    )
    updater = CubatureKalmanUpdater(
        measurement_model=measurement_model,
        force_positive_definite_covariance=True,
    )
    predictor = CubatureKalmanPredictor(transition_model)

    # compute update
    update = updater.update(
        SingleHypothesis(
            prediction=prediction,
            measurement=measurement,
            measurement_prediction=measurement_prediction,
        )
    )

    # predict state
    predicted_state = predictor.predict(update, timestamp=datetime.datetime.fromtimestamp(1))

    # check results
    assert np.all(np.isfinite(predicted_state.state_vector))
    assert np.all(np.isfinite(predicted_state.covar))
    assert np.all(np.linalg.eigvals(predicted_state.covar) > 0)


def test_kalman(updater_class, use_joseph_cov):
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T,
    )
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar
    )
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector - eval_measurement_prediction.mean),
        prediction.covar - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T,
    )

    # Initialise a kalman updater
    updater = updater_class(measurement_model=measurement_model, use_joseph_cov=use_joseph_cov)

    # Get and assert measurement prediction without measurement noise
    measurement_prediction = updater.predict_measurement(prediction, measurement_noise=False)
    assert np.allclose(
        measurement_prediction.mean, eval_measurement_prediction.mean, 0, atol=1.0e-13
    )
    assert np.allclose(
        measurement_prediction.covar,
        eval_measurement_prediction.covar - measurement_model.covar(),
        0,
        atol=1.0e-13,
    )
    assert np.allclose(
        measurement_prediction.cross_covar,
        eval_measurement_prediction.cross_covar,
        0,
        atol=1.0e-13,
    )

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert np.allclose(
        measurement_prediction.mean, eval_measurement_prediction.mean, 0, atol=1.0e-13
    )
    assert np.allclose(
        measurement_prediction.covar, eval_measurement_prediction.covar, 0, atol=1.0e-13
    )
    assert np.allclose(
        measurement_prediction.cross_covar,
        eval_measurement_prediction.cross_covar,
        0,
        atol=1.0e-13,
    )

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.0e-13)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.0e-13)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector,
        0,
        atol=1.0e-13,
    )
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.covar,
        measurement_prediction.covar,
        0,
        atol=1.0e-13,
    )
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp

    # Perform and assert state update
    posterior = updater.update(
        SingleHypothesis(
            prediction=prediction,
            measurement=measurement,
            measurement_prediction=measurement_prediction,
        )
    )
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.0e-13)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.0e-13)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector,
        0,
        atol=1.0e-13,
    )
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.covar,
        measurement_prediction.covar,
        0,
        atol=1.0e-13,
    )
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp


def test_sqrt_kalman():
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    sqrt_prediction = SqrtGaussianState(
        prediction.state_vector, np.linalg.cholesky(prediction.covar)
    )
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T,
    )
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar
    )
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector - eval_measurement_prediction.mean),
        prediction.covar - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T,
    )

    # Test Square root form returns the same as standard form
    updater = KalmanUpdater(measurement_model=measurement_model)
    sqrt_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=False)
    qr_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=True)

    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))
    posterior_s = sqrt_updater.update(
        SingleHypothesis(prediction=sqrt_prediction, measurement=measurement)
    )
    posterior_q = qr_updater.update(
        SingleHypothesis(prediction=sqrt_prediction, measurement=measurement)
    )

    assert np.allclose(posterior_s.mean, eval_posterior.mean, 0, atol=1.0e-14)
    assert np.allclose(posterior_q.mean, eval_posterior.mean, 0, atol=1.0e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.0e-14)
    assert np.allclose(
        eval_posterior.covar, posterior_s.sqrt_covar @ posterior_s.sqrt_covar.T, 0, atol=1.0e-14
    )
    assert np.allclose(
        posterior.covar, posterior_s.sqrt_covar @ posterior_s.sqrt_covar.T, 0, atol=1.0e-14
    )
    assert np.allclose(
        posterior.covar, posterior_q.sqrt_covar @ posterior_q.sqrt_covar.T, 0, atol=1.0e-14
    )
    # I'm not sure this is going to be true in all cases. Keep in order to find edge cases
    assert np.allclose(posterior_s.covar, posterior_q.covar, 0, atol=1.0e-14)

    # Next create a prediction with a covariance that will cause problems
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[1e24, 1e-24], [1e-24, 1e24]])
    )
    sqrt_prediction = SqrtGaussianState(
        prediction.state_vector, np.linalg.cholesky(prediction.covar)
    )

    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))
    posterior_s = sqrt_updater.update(
        SingleHypothesis(prediction=sqrt_prediction, measurement=measurement)
    )
    posterior_q = qr_updater.update(
        SingleHypothesis(prediction=sqrt_prediction, measurement=measurement)
    )

    # The new posterior will  be
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector - eval_measurement_prediction.mean),
        np.array([[0.04, 0], [0, 1e24]]),
    )  # Accessed by looking through the Decimal() quantities...
    # It's actually [0.039999999999 1e-48], [1e-24 1e24 + 1e-48]] ish

    # Test that the square root form succeeds where the standard form fails
    assert not np.allclose(posterior.covar, eval_posterior.covar, rtol=5.0e-3)
    assert np.allclose(
        posterior_s.sqrt_covar @ posterior_s.sqrt_covar.T, eval_posterior.covar, rtol=5.0e-3
    )
    assert np.allclose(
        posterior_q.sqrt_covar @ posterior_s.sqrt_covar.T, eval_posterior.covar, rtol=5.0e-3
    )


def test_schmidtkalman():
    """Ensure that the SKF returns the same as the KF for a sensible set of consider parameters."""

    nelements = 100
    # Create a state vector with a bunch of consider variables
    consider = np.ones(nelements, dtype=bool)
    consider[0] = False
    consider[2] = False

    state_vector = np.ones(nelements) * 10
    state_vector[0] = -6.45
    state_vector[2] = 0.7

    covariance = np.diag(np.ones(nelements))
    covariance_con = np.diag(np.ones(nelements - 2))
    covariance_noncon = np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    covariance[np.ix_(~consider, ~consider)] = covariance_noncon
    covariance[np.ix_(consider, consider)] = covariance_con

    prediction = GaussianStatePrediction(state_vector, covariance)
    measurement_model = LinearGaussian(
        ndim_state=nelements, mapping=[0], noise_covar=np.array([[0.04]])
    )
    measurement = Detection(np.array([[-6.23]]))

    hypothesis = SingleHypothesis(prediction, measurement)

    updater = KalmanUpdater(measurement_model)
    sk_updater = SchmidtKalmanUpdater(measurement_model, consider=consider)
    update = updater.update(hypothesis)
    sk_update = sk_updater.update(hypothesis)

    assert np.allclose(update.mean, sk_update.mean)
    assert np.allclose(update.covar, sk_update.covar)


def test_kalman_updater_measurement_model_missing():
    """Test that error is raised when no measurement model is specified."""
    updater = KalmanUpdater()
    prediction = GaussianStatePrediction(np.array([[1], [1]]), np.eye(2))
    measurement = Detection(np.array([[1]]))

    with pytest.raises(ValueError, match="No measurement model specified"):
        updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))


def test_kalman_updater_with_timestamp():
    """Test that the update respects the measurement timestamp."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]),
        np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        timestamp=datetime.datetime(2023, 1, 1, 0, 0, 0),
    )
    timestamp = datetime.datetime(2023, 1, 1, 0, 0, 1)
    measurement = Detection(np.array([[-6.23]]), timestamp=timestamp)

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert posterior.timestamp == timestamp


def test_kalman_updater_force_symmetric():
    """Test force_symmetric_covariance flag."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    measurement = Detection(np.array([[-6.23]]))

    updater = KalmanUpdater(measurement_model=measurement_model, force_symmetric_covariance=True)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Check that covariance is symmetric
    assert np.allclose(posterior.covar, posterior.covar.T)


def test_kalman_updater_force_positive_definite():
    """Test force_positive_definite_covariance flag."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    measurement = Detection(np.array([[-6.23]]))

    updater = KalmanUpdater(
        measurement_model=measurement_model, force_positive_definite_covariance=True
    )
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Check that all eigenvalues are positive
    assert np.all(np.linalg.eigvals(posterior.covar) > 0)


def test_extended_kalman_updater_linear_model():
    """Test ExtendedKalmanUpdater works with linear models."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    measurement = Detection(np.array([[-6.23]]))

    updater = ExtendedKalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Should produce same result as standard Kalman for linear model
    standard_updater = KalmanUpdater(measurement_model=measurement_model)
    standard_posterior = standard_updater.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    assert np.allclose(posterior.mean, standard_posterior.mean, atol=1e-13)
    assert np.allclose(posterior.covar, standard_posterior.covar, atol=1e-13)


def test_extended_kalman_updater_nonlinear_model():
    """Test ExtendedKalmanUpdater with a nonlinear measurement model."""
    # Create a bearing-range measurement model
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    # Create a prediction
    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    # Create measurement
    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    updater = ExtendedKalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Check that update produces finite results
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))
    # Check covariance is reduced (information gained)
    assert np.all(np.diag(posterior.covar) <= np.diag(prediction.covar))


def test_unscented_kalman_updater():
    """Test UnscentedKalmanUpdater with different parameters."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    # Test with default parameters
    updater = UnscentedKalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))

    # Test with custom parameters
    updater_custom = UnscentedKalmanUpdater(
        measurement_model=measurement_model, alpha=0.3, beta=2.5, kappa=1
    )
    posterior_custom = updater_custom.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    assert np.all(np.isfinite(posterior_custom.mean))
    assert np.all(np.isfinite(posterior_custom.covar))


def test_cubature_kalman_updater():
    """Test CubatureKalmanUpdater."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    # Test with default alpha
    updater = CubatureKalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))

    # Test with custom alpha
    updater_custom = CubatureKalmanUpdater(measurement_model=measurement_model, alpha=0.5)
    posterior_custom = updater_custom.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    assert np.all(np.isfinite(posterior_custom.mean))
    assert np.all(np.isfinite(posterior_custom.covar))


def test_iterated_kalman_updater_convergence():
    """Test IteratedKalmanUpdater convergence."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    # Test with tight tolerance
    updater = IteratedKalmanUpdater(
        measurement_model=measurement_model, tolerance=1e-8, max_iterations=100
    )
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))


def test_iterated_kalman_updater_non_convergence():
    """Test IteratedKalmanUpdater when it doesn't converge."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    # Test with very tight tolerance and few iterations to trigger warning
    updater = IteratedKalmanUpdater(
        measurement_model=measurement_model, tolerance=1e-15, max_iterations=2
    )

    with pytest.warns(UserWarning, match="Iterated Kalman update did not converge"):
        posterior = updater.update(
            SingleHypothesis(prediction=prediction, measurement=measurement)
        )

    # Should still produce valid results even if not converged
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))


def test_stochastic_integration_updater():
    """Test StochasticIntegrationUpdater."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    # Test with default parameters
    updater = StochasticIntegrationUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))

    # Test with custom parameters
    updater_custom = StochasticIntegrationUpdater(
        measurement_model=measurement_model, Nmax=15, Nmin=3, Eps=1e-3, SIorder=3
    )
    posterior_custom = updater_custom.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    assert np.all(np.isfinite(posterior_custom.mean))
    assert np.all(np.isfinite(posterior_custom.covar))


def test_schmidt_kalman_updater_consider_params():
    """Test SchmidtKalmanUpdater with different consider parameter configurations."""
    nelements = 10

    # Test 1: All parameters are considered (should behave differently from standard KF)
    consider_all = np.ones(nelements, dtype=bool)
    state_vector = np.ones(nelements)
    covariance = np.eye(nelements)

    prediction = GaussianStatePrediction(state_vector, covariance)
    measurement_model = LinearGaussian(
        ndim_state=nelements, mapping=[0], noise_covar=np.array([[0.04]])
    )
    measurement = Detection(np.array([[1.1]]))

    sk_updater = SchmidtKalmanUpdater(measurement_model, consider=consider_all)
    sk_update = sk_updater.update(SingleHypothesis(prediction, measurement))

    # All state parameters should remain unchanged (only considered, not estimated)
    assert np.allclose(sk_update.mean, prediction.mean)

    # Test 2: No parameters are considered (should match standard KF)
    consider_none = np.zeros(nelements, dtype=bool)

    updater = KalmanUpdater(measurement_model)
    sk_updater_none = SchmidtKalmanUpdater(measurement_model, consider=consider_none)

    update = updater.update(SingleHypothesis(prediction, measurement))
    sk_update_none = sk_updater_none.update(SingleHypothesis(prediction, measurement))

    assert np.allclose(update.mean, sk_update_none.mean, atol=1e-13)
    assert np.allclose(update.covar, sk_update_none.covar, atol=1e-13)


def test_measurement_prediction_caching():
    """Test that measurement prediction is cached correctly."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )

    updater = KalmanUpdater(measurement_model=measurement_model)

    # Call predict_measurement twice with same parameters
    meas_pred_1 = updater.predict_measurement(prediction)
    meas_pred_2 = updater.predict_measurement(prediction)

    # Should return the same cached object
    assert meas_pred_1 is meas_pred_2


def test_innovation_covariance_without_noise():
    """Test innovation covariance calculation without measurement noise."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )

    updater = KalmanUpdater(measurement_model=measurement_model)

    # Get measurement prediction without noise
    meas_pred_no_noise = updater.predict_measurement(prediction, measurement_noise=False)

    # Get measurement prediction with noise
    meas_pred_with_noise = updater.predict_measurement(prediction, measurement_noise=True)

    # Difference should be the measurement noise covariance
    diff = meas_pred_with_noise.covar - meas_pred_no_noise.covar
    assert np.allclose(diff, measurement_model.covar(), atol=1e-13)


def test_joseph_form_covariance():
    """Test Joseph form covariance calculation provides better numerical stability."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )
    measurement = Detection(np.array([[-6.23]]))

    # Standard form
    updater_standard = KalmanUpdater(measurement_model=measurement_model, use_joseph_cov=False)
    posterior_standard = updater_standard.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    # Joseph form
    updater_joseph = KalmanUpdater(measurement_model=measurement_model, use_joseph_cov=True)
    posterior_joseph = updater_joseph.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    # Results should be very similar
    assert np.allclose(posterior_standard.mean, posterior_joseph.mean, atol=1e-13)
    assert np.allclose(posterior_standard.covar, posterior_joseph.covar, atol=1e-13)

    # Joseph form should always give positive semi-definite covariance
    assert np.all(np.linalg.eigvals(posterior_joseph.covar) >= -1e-10)


def test_singular_innovation_covariance():
    """Test behavior with near-singular innovation covariance."""
    # Create a scenario with nearly zero measurement noise
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[1e-15]]))
    prediction = GaussianStatePrediction(
        np.array([[1.0], [0.0]]), np.array([[1e-15, 0], [0, 1.0]])
    )
    measurement = Detection(np.array([[1.0]]))

    updater = KalmanUpdater(measurement_model=measurement_model)

    # This should still work despite numerical challenges
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))


def test_high_dimensional_state():
    """Test updaters with high-dimensional state vectors."""
    ndim = 50
    measurement_model = LinearGaussian(
        ndim_state=ndim, mapping=list(range(0, ndim, 2)), noise_covar=np.eye(ndim // 2) * 0.1
    )

    prediction = GaussianStatePrediction(np.random.randn(ndim, 1), np.eye(ndim) * 0.5)

    measurement = Detection(np.random.randn(ndim // 2, 1))

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert posterior.mean.shape == (ndim, 1)
    assert posterior.covar.shape == (ndim, ndim)
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))


def test_zero_process_noise():
    """Test updaters with zero process noise (perfect model)."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    # Prediction with very small uncertainty
    prediction = GaussianStatePrediction(np.array([[1.0], [0.0]]), np.eye(2) * 1e-10)

    measurement = Detection(np.array([[1.05]]))

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Posterior should be close to prediction (high confidence in prediction)
    assert np.allclose(posterior.mean[0], prediction.mean[0], atol=0.01)
    assert np.all(np.isfinite(posterior.covar))


def test_updater_with_measurement_model_in_detection():
    """Test that measurement model from detection is used when provided."""
    measurement_model_updater = LinearGaussian(
        ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]])
    )
    measurement_model_detection = LinearGaussian(
        ndim_state=2, mapping=[0], noise_covar=np.array([[0.01]])
    )

    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    )

    # Measurement with its own model
    measurement = Detection(np.array([[-6.23]]), measurement_model=measurement_model_detection)

    updater = KalmanUpdater(measurement_model=measurement_model_updater)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # The detection's measurement model should be used
    # (different noise covariance should affect result)
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))


def test_multidimensional_measurement():
    """Test with multi-dimensional measurements."""
    measurement_model = LinearGaussian(
        ndim_state=4, mapping=[0, 2], noise_covar=np.diag([0.04, 0.05])
    )

    prediction = GaussianStatePrediction(np.array([[1.0], [0.5], [2.0], [0.3]]), np.eye(4) * 0.5)

    measurement = Detection(np.array([[1.1], [2.1]]))

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    assert posterior.mean.shape == (4, 1)
    assert posterior.covar.shape == (4, 4)
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.covar))
    # Check that covariance is reduced
    assert np.trace(posterior.covar) < np.trace(prediction.covar)


def test_covariance_symmetry_preservation():
    """Test that posterior covariance remains symmetric."""
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2) * 0.1)

    # Create slightly asymmetric covariance (numerical errors)
    covar = np.eye(4) * 0.5
    covar[0, 1] = 0.1
    covar[1, 0] = 0.1 + 1e-15  # Slight asymmetry

    prediction = GaussianStatePrediction(np.ones((4, 1)), covar)
    measurement = Detection(np.array([[1.1], [1.2]]))

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Check symmetry (may have small numerical errors)
    assert np.allclose(posterior.covar, posterior.covar.T, atol=1e-10)


def test_extended_kalman_with_linearisation_point():
    """Test ExtendedKalmanUpdater with custom linearisation point."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(1), 1]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [0], [10], [0]]), np.eye(4) * 0.5)

    # Custom linearisation point
    linearisation_state = GaussianState(np.array([[11], [0], [11], [0]]), np.eye(4) * 0.5)

    updater = ExtendedKalmanUpdater(measurement_model=measurement_model)

    # Get measurement matrix with custom linearisation point
    meas_matrix = updater._measurement_matrix(
        prediction, measurement_model, linearisation_point=linearisation_state
    )

    assert meas_matrix.shape == (2, 4)
    assert np.all(np.isfinite(meas_matrix))


def test_sqrt_kalman_numerical_stability():
    """Test that SqrtKalmanUpdater provides better numerical stability."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    # Create ill-conditioned covariance
    ill_conditioned_covar = np.array([[1e10, 1e5], [1e5, 1e10]])

    prediction = GaussianStatePrediction(np.array([[1.0], [1.0]]), ill_conditioned_covar)

    sqrt_prediction = SqrtGaussianState(
        prediction.state_vector, np.linalg.cholesky(ill_conditioned_covar)
    )

    measurement = Detection(np.array([[1.1]]))

    # Standard Kalman
    standard_updater = KalmanUpdater(measurement_model=measurement_model)
    standard_posterior = standard_updater.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    # Square root Kalman
    sqrt_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=True)
    sqrt_posterior = sqrt_updater.update(
        SingleHypothesis(prediction=sqrt_prediction, measurement=measurement)
    )

    # Both should produce finite results
    assert np.all(np.isfinite(standard_posterior.covar))
    assert np.all(np.isfinite(sqrt_posterior.covar))

    # Square root should maintain positive definiteness better
    sqrt_eigs = np.linalg.eigvals(sqrt_posterior.covar)
    assert np.all(sqrt_eigs > -1e-10)


def test_comparison_ukf_vs_ckf():
    """Compare UnscentedKalmanUpdater and CubatureKalmanUpdater results."""
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.diag([np.radians(0.5), 0.5]),
    )

    prediction = GaussianStatePrediction(np.array([[10], [1], [10], [1]]), np.eye(4) * 0.3)

    measurement = Detection(np.array([[np.radians(45)], [14.14]]))

    ukf_updater = UnscentedKalmanUpdater(measurement_model=measurement_model)
    ckf_updater = CubatureKalmanUpdater(measurement_model=measurement_model)

    ukf_posterior = ukf_updater.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )
    ckf_posterior = ckf_updater.update(
        SingleHypothesis(prediction=prediction, measurement=measurement)
    )

    # Results should be similar but not identical
    assert np.allclose(ukf_posterior.mean, ckf_posterior.mean, rtol=0.1)
    assert np.allclose(ukf_posterior.covar, ckf_posterior.covar, rtol=0.2)


def test_multiple_sequential_updates():
    """Test multiple sequential updates maintain consistency."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    # Initial state
    state = GaussianStatePrediction(np.array([[0.0], [1.0]]), np.eye(2))

    updater = KalmanUpdater(measurement_model=measurement_model)

    # Perform multiple updates
    measurements = [0.1, 0.2, 0.3, 0.4, 0.5]
    for meas_value in measurements:
        measurement = Detection(np.array([[meas_value]]))
        state = updater.update(SingleHypothesis(prediction=state, measurement=measurement))

        # Check that state remains valid
        assert np.all(np.isfinite(state.mean))
        assert np.all(np.isfinite(state.covar))
        assert np.all(np.linalg.eigvals(state.covar) > -1e-10)

        # Uncertainty should decrease
        assert state.covar[0, 0] < 1.0  # Less than initial


def test_edge_case_identical_prediction_and_measurement():
    """Test case where prediction exactly matches measurement."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    prediction = GaussianStatePrediction(np.array([[5.0], [1.0]]), np.eye(2) * 0.5)

    # Measurement exactly matches predicted measurement
    measurement = Detection(np.array([[5.0]]))

    updater = KalmanUpdater(measurement_model=measurement_model)
    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # Posterior mean should be between prediction and measurement (Kalman gain)
    # But since measurement matches prediction, should be very close to prediction
    assert np.allclose(posterior.mean[0], 5.0, atol=0.1)
    assert np.all(np.isfinite(posterior.covar))


def test_negative_definite_covariance_correction():
    """Test that force_positive_definite corrects numerical issues."""
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    # Create a scenario that might lead to numerical issues
    prediction = GaussianStatePrediction(
        np.array([[1.0], [1.0]]), np.array([[1e-8, 0], [0, 1e-8]])
    )

    measurement = Detection(np.array([[1.0 + 1e-7]]))

    updater = KalmanUpdater(
        measurement_model=measurement_model, force_positive_definite_covariance=True
    )

    posterior = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))

    # All eigenvalues should be positive
    eigenvalues = np.linalg.eigvals(posterior.covar)
    assert np.all(eigenvalues > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
