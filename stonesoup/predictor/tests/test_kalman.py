import datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ...models.control.linear import LinearControlModel
from ...models.transition.linear import (
    ConstantAcceleration,
    ConstantVelocity,
)
from ...models.transition.nonlinear import ConstantTurn
from ...predictor.kalman import (
    CubatureKalmanPredictor,
    ExtendedKalmanPredictor,
    KalmanPredictor,
    SqrtKalmanPredictor,
    StochasticIntegrationPredictor,
    UnscentedKalmanPredictor,
)
from ...types.array import CovarianceMatrix, StateVector
from ...types.prediction import GaussianStatePrediction, SqrtGaussianStatePrediction
from ...types.state import GaussianState, SqrtGaussianState, State
from ...types.track import Track


@pytest.mark.parametrize(
    "PredictorClass, transition_model, prior_mean, prior_covar",
    [
        (  # Standard Kalman
            KalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        ),
        (  # Extended Kalman
            ExtendedKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        ),
        (  # Unscented Kalman
            UnscentedKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        ),
        (  # cubature Kalman
            CubatureKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        ),
        (  # Stochastic Integration
            StochasticIntegrationPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        ),
    ],
    ids=["standard", "extended", "unscented", "cubature", "stochasticIntegration"],
)
def test_kalman(PredictorClass, transition_model, prior_mean, prior_covar):
    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)

    transition_model_matrix = transition_model.matrix(time_interval=time_interval)
    transition_model_covar = transition_model.covar(time_interval=time_interval)
    # Calculate evaluation variables
    eval_prediction = GaussianStatePrediction(
        transition_model_matrix @ prior.mean,
        transition_model_matrix @ prior.covar @ transition_model_matrix.T + transition_model_covar,
    )

    # Initialise a kalman predictor
    predictor = PredictorClass(transition_model=transition_model)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Assert presence of transition model
    assert hasattr(prediction, "transition_model")

    assert np.allclose(prediction.mean, eval_prediction.mean, 0, atol=1.0e-14)
    assert np.allclose(prediction.covar, eval_prediction.covar, 0, atol=1.0e-14)
    assert prediction.timestamp == new_timestamp

    # TODO: Test with Control Model


def test_lru_cache():
    predictor = KalmanPredictor(ConstantVelocity(noise_diff_coeff=0))

    timestamp = datetime.datetime.now()
    state = GaussianState([[0.0], [1.0]], np.diag([1.0, 1.0]), timestamp)
    track = Track([state])

    prediction_time = timestamp + datetime.timedelta(seconds=1)
    prediction1 = predictor.predict(track, prediction_time)
    assert np.array_equal(prediction1.state_vector, np.array([[1.0], [1.0]]))

    prediction2 = predictor.predict(track, prediction_time)
    assert prediction2 is prediction1

    track.append(GaussianState([[1.0], [1.0]], np.diag([1.0, 1.0]), prediction_time))
    prediction3 = predictor.predict(track, prediction_time)
    assert prediction3 is not prediction1


def test_sqrt_kalman():
    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define prior state
    prior_mean = np.array([[-6.45], [0.7]])
    prior_covar = np.array([[4.1123, 0.0013], [0.0013, 0.0365]])
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)
    sqrt_prior_covar = np.linalg.cholesky(prior_covar)
    sqrt_prior = SqrtGaussianState(prior_mean, sqrt_prior_covar, timestamp=timestamp)

    transition_model = ConstantVelocity(noise_diff_coeff=0.1)

    # Initialise a kalman predictor
    predictor = KalmanPredictor(transition_model=transition_model)
    sqrt_predictor = SqrtKalmanPredictor(transition_model=transition_model)
    # Can swap out this method
    sqrt_predictor = SqrtKalmanPredictor(transition_model=transition_model, qr_method=True)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)
    sqrt_prediction = sqrt_predictor.predict(prior=sqrt_prior, timestamp=new_timestamp)

    # Assert presence of transition model
    assert hasattr(prediction, "transition_model")

    assert np.allclose(prediction.mean, sqrt_prediction.mean, 0, atol=1.0e-14)
    assert np.allclose(
        prediction.covar,
        sqrt_prediction.sqrt_covar @ sqrt_prediction.sqrt_covar.T,
        0,
        atol=1.0e-14,
    )
    assert np.allclose(prediction.covar, sqrt_prediction.covar, 0, atol=1.0e-14)
    assert prediction.timestamp == sqrt_prediction.timestamp


# ============================================================================
# Additional Comprehensive Tests
# ============================================================================


# Comprehensive tests for KalmanPredictor class


def test_kalman_predictor_init_default_control_model():
    """Test that KalmanPredictor creates default zero-effect control model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    # Check control model was created
    assert predictor.control_model is not None
    assert isinstance(predictor.control_model, LinearControlModel)

    # Check control model has zero effect
    assert np.all(predictor.control_model.control_matrix == 0)
    assert np.all(predictor.control_model.control_noise == 0)


def test_kalman_predictor_init_custom_control_model():
    """Test KalmanPredictor initialization with custom control model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    control_matrix = np.array([[1.0], [0.5]])
    control_noise = np.array([[0.1]])
    control_model = LinearControlModel(control_matrix=control_matrix, control_noise=control_noise)

    predictor = KalmanPredictor(transition_model=transition_model, control_model=control_model)

    assert predictor.control_model is control_model
    assert np.array_equal(predictor.control_model.control_matrix, control_matrix)


def test_kalman_predictor_transition_matrix():
    """Test transition matrix calculation"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    time_interval = datetime.timedelta(seconds=1.0)
    trans_matrix = predictor._transition_matrix(time_interval=time_interval)

    # For ConstantVelocity with dt=1, should be [[1, 1], [0, 1]]
    expected = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert_allclose(trans_matrix, expected, rtol=1e-10)


def test_kalman_predictor_transition_function():
    """Test transition function application"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    prior_mean = StateVector([[10.0], [2.0]])
    prior = GaussianState(prior_mean, np.eye(2))

    time_interval = datetime.timedelta(seconds=1.0)
    result = predictor._transition_function(prior, time_interval=time_interval)

    # x_new = x_old + v*dt = 10 + 2*1 = 12
    # v_new = v_old = 2
    expected = StateVector([[12.0], [2.0]])
    assert_allclose(result, expected, rtol=1e-10)


def test_kalman_predictor_control_matrix():
    """Test control matrix retrieval"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    control_matrix = np.array([[0.5], [1.0]])
    control_model = LinearControlModel(
        control_matrix=control_matrix, control_noise=np.array([[0.01]])
    )
    predictor = KalmanPredictor(transition_model=transition_model, control_model=control_model)

    ctrl_mat = predictor._control_matrix()
    assert_allclose(ctrl_mat, control_matrix)


def test_kalman_predictor_predict_over_interval_with_timestamps():
    """Test prediction interval calculation with timestamps"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)
    new_timestamp = datetime.datetime(2024, 1, 1, 12, 0, 5)
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)

    interval = predictor._predict_over_interval(prior, new_timestamp)
    assert interval == datetime.timedelta(seconds=5)


def test_kalman_predictor_predict_over_interval_none_timestamps():
    """Test prediction interval when timestamps are None"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=None)

    interval = predictor._predict_over_interval(prior, None)
    assert interval is None


def test_kalman_predictor_predicted_covariance():
    """Test predicted covariance calculation"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    prior_covar = CovarianceMatrix([[1.0, 0.1], [0.1, 1.0]])
    prior = GaussianState(StateVector([[0.0], [1.0]]), prior_covar)

    time_interval = datetime.timedelta(seconds=1.0)
    predicted_cov = predictor._predicted_covariance(prior, time_interval)

    # Should be F @ P @ F.T + Q
    F = transition_model.matrix(time_interval=time_interval)
    Q = transition_model.covar(time_interval=time_interval)
    expected = F @ prior_covar @ F.T + Q

    assert_allclose(predicted_cov, expected, rtol=1e-10)


def test_kalman_predictor_predict_no_control_input():
    """Test prediction without control input"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=2.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Check prediction type
    assert isinstance(prediction, GaussianStatePrediction)

    # Check prediction has correct timestamp
    assert prediction.timestamp == new_timestamp

    # Check mean is approximately correct (position should increase by velocity * time)
    assert_allclose(prediction.mean[0], 2.0, atol=1e-10)
    assert_allclose(prediction.mean[1], 1.0, atol=1e-10)

    # Check covariance is positive definite
    assert np.all(np.linalg.eigvals(prediction.covar) > 0)


def test_kalman_predictor_predict_with_control_input():
    """Test prediction with control input"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.0)
    control_matrix = np.array([[0.5], [1.0]])
    control_model = LinearControlModel(
        control_matrix=control_matrix, control_noise=np.zeros((1, 1))
    )
    predictor = KalmanPredictor(transition_model=transition_model, control_model=control_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [0.0]]), np.eye(2), timestamp=timestamp)
    control_input = State(StateVector([[2.0]]), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(
        prior=prior, timestamp=new_timestamp, control_input=control_input
    )

    # Mean should be affected by control: B @ u = [[0.5], [1.0]] @ [[2.0]] = [[1.0], [2.0]]
    assert_allclose(prediction.mean, [[1.0], [2.0]], atol=1e-10)


def test_kalman_predictor_predict_returns_prior_reference():
    """Test that prediction contains reference to prior"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Check prediction has reference to prior
    assert hasattr(prediction, "prior")
    assert prediction.prior is prior


def test_kalman_predictor_predict_returns_transition_model():
    """Test that prediction contains transition model reference"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert hasattr(prediction, "transition_model")
    assert prediction.transition_model is transition_model


def test_kalman_predictor_covariance_growth():
    """Test that covariance grows with prediction"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.5)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Covariance should be larger due to process noise
    assert np.linalg.det(prediction.covar) > np.linalg.det(prior.covar)


# Comprehensive tests for ExtendedKalmanPredictor class


def test_ekf_predictor_with_linear_model():
    """Test EKF predictor with linear transition model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = ExtendedKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Should behave like standard Kalman for linear models
    assert isinstance(prediction, GaussianStatePrediction)
    assert_allclose(prediction.mean, [[1.0], [1.0]], atol=1e-10)


def test_ekf_predictor_with_nonlinear_model():
    """Test EKF predictor with nonlinear transition model (ConstantTurn)"""
    # ConstantTurn is a 5D state: [x, vx, y, vy, turn_rate]
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = ExtendedKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    # Initial state: position (0, 0), velocity (1, 0), turn rate 0.1 rad/s
    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior_covar = np.eye(5) * 0.1
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Prediction should be successful
    assert isinstance(prediction, GaussianStatePrediction)
    assert prediction.ndim == 5
    assert prediction.timestamp == new_timestamp


def test_ekf_transition_matrix_with_linear_model():
    """Test transition matrix for linear model returns matrix"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = ExtendedKalmanPredictor(transition_model=transition_model)

    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2))
    time_interval = datetime.timedelta(seconds=1.0)

    trans_mat = predictor._transition_matrix(prior, time_interval=time_interval)

    # Should return the matrix from linear model
    expected = transition_model.matrix(time_interval=time_interval)
    assert_allclose(trans_mat, expected)


def test_ekf_transition_matrix_with_nonlinear_model():
    """Test transition matrix for nonlinear model returns Jacobian"""
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = ExtendedKalmanPredictor(transition_model=transition_model)

    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior = GaussianState(prior_mean, np.eye(5))
    time_interval = datetime.timedelta(seconds=1.0)

    trans_mat = predictor._transition_matrix(prior, time_interval=time_interval)

    # Should return Jacobian for nonlinear model
    expected = transition_model.jacobian(prior, time_interval=time_interval)
    assert_allclose(trans_mat, expected)


def test_ekf_transition_function_nonlinear():
    """Test transition function for nonlinear model"""
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = ExtendedKalmanPredictor(transition_model=transition_model)

    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior = GaussianState(prior_mean, np.eye(5))
    time_interval = datetime.timedelta(seconds=1.0)

    result = predictor._transition_function(prior, time_interval=time_interval)

    # Should use nonlinear function
    expected = transition_model.function(prior, time_interval=time_interval)
    assert_allclose(result, expected)


def test_ekf_control_matrix_with_linear_control():
    """Test control matrix with linear control model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    control_matrix = np.array([[0.5], [1.0]])
    control_model = LinearControlModel(
        control_matrix=control_matrix, control_noise=np.array([[0.01]])
    )
    predictor = ExtendedKalmanPredictor(
        transition_model=transition_model, control_model=control_model
    )

    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2))
    control_input = State(StateVector([[1.0]]))

    ctrl_mat = predictor._control_matrix(control_input, prior)
    assert_allclose(ctrl_mat, control_matrix)


# Comprehensive tests for UnscentedKalmanPredictor class


def test_ukf_predictor_initialization():
    """Test UKF predictor initialization with default parameters"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = UnscentedKalmanPredictor(transition_model=transition_model)

    assert predictor.alpha == 0.5
    assert predictor.beta == 2
    assert predictor.kappa is None


def test_ukf_predictor_custom_parameters():
    """Test UKF predictor with custom alpha, beta, kappa"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = UnscentedKalmanPredictor(
        transition_model=transition_model, alpha=1.0, beta=0.0, kappa=1.0
    )

    assert predictor.alpha == 1.0
    assert predictor.beta == 0.0
    assert predictor.kappa == 1.0


def test_ukf_predictor_with_linear_model():
    """Test UKF predictor with linear model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = UnscentedKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    # For linear models, UKF should give similar results to KF
    assert_allclose(prediction.mean, [[1.0], [1.0]], atol=0.01)


def test_ukf_predictor_with_nonlinear_model():
    """Test UKF predictor with nonlinear model"""
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = UnscentedKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior_covar = np.eye(5) * 0.1
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    assert prediction.ndim == 5
    assert prediction.timestamp == new_timestamp


def test_ukf_transition_and_control_function():
    """Test combined transition and control function"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.0)
    control_matrix = np.array([[0.5], [1.0]])
    control_model = LinearControlModel(
        control_matrix=control_matrix, control_noise=np.zeros((1, 1))
    )
    predictor = UnscentedKalmanPredictor(
        transition_model=transition_model, control_model=control_model
    )

    prior_state = GaussianState(StateVector([[0.0], [0.0]]), np.eye(2))
    control_input = State(StateVector([[2.0]]))
    time_interval = datetime.timedelta(seconds=1.0)

    result = predictor._transition_and_control_function(
        prior_state, control_input=control_input, time_interval=time_interval
    )

    # Should apply both transition and control
    expected = StateVector([[1.0], [2.0]])  # Control effect
    assert_allclose(result, expected, atol=1e-10)


# Comprehensive tests for CubatureKalmanPredictor class


def test_ckf_predictor_initialization():
    """Test CKF predictor initialization"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = CubatureKalmanPredictor(transition_model=transition_model)

    assert predictor.alpha == 1.0


def test_ckf_predictor_custom_alpha():
    """Test CKF predictor with custom alpha"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = CubatureKalmanPredictor(transition_model=transition_model, alpha=0.5)

    assert predictor.alpha == 0.5


def test_ckf_predictor_with_linear_model():
    """Test CKF predictor with linear model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = CubatureKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    # Should give similar results to KF for linear models
    assert_allclose(prediction.mean, [[1.0], [1.0]], atol=0.01)


def test_ckf_predictor_with_nonlinear_model():
    """Test CKF predictor with nonlinear model"""
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = CubatureKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior_covar = np.eye(5) * 0.1
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    assert prediction.ndim == 5


# Comprehensive tests for SqrtKalmanPredictor class


def test_sqrt_predictor_initialization():
    """Test SqrtKalmanPredictor initialization"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = SqrtKalmanPredictor(transition_model=transition_model)

    assert predictor.qr_method is False


def test_sqrt_predictor_qr_method():
    """Test SqrtKalmanPredictor with QR decomposition method"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = SqrtKalmanPredictor(transition_model=transition_model, qr_method=True)

    assert predictor.qr_method is True

    timestamp = datetime.datetime.now()
    prior_mean = StateVector([[0.0], [1.0]])
    prior_covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    sqrt_prior_covar = np.linalg.cholesky(prior_covar)
    prior = SqrtGaussianState(prior_mean, sqrt_prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, SqrtGaussianStatePrediction)


def test_sqrt_predictor_predicted_covariance_cholesky():
    """Test predicted covariance using Cholesky method"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = SqrtKalmanPredictor(transition_model=transition_model, qr_method=False)

    prior_mean = StateVector([[0.0], [1.0]])
    prior_covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    sqrt_prior_covar = np.linalg.cholesky(prior_covar)
    prior = SqrtGaussianState(prior_mean, sqrt_prior_covar)

    time_interval = datetime.timedelta(seconds=1.0)
    sqrt_pred_cov = predictor._predicted_covariance(prior, time_interval)

    # Verify it's a valid square root (positive semi-definite)
    full_cov = sqrt_pred_cov @ sqrt_pred_cov.T
    eigenvalues = np.linalg.eigvals(full_cov)
    assert np.all(eigenvalues >= -1e-10)


def test_sqrt_predictor_predicted_covariance_qr():
    """Test predicted covariance using QR method"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = SqrtKalmanPredictor(transition_model=transition_model, qr_method=True)

    prior_mean = StateVector([[0.0], [1.0]])
    prior_covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    sqrt_prior_covar = np.linalg.cholesky(prior_covar)
    prior = SqrtGaussianState(prior_mean, sqrt_prior_covar)

    time_interval = datetime.timedelta(seconds=1.0)
    sqrt_pred_cov = predictor._predicted_covariance(prior, time_interval)

    # Verify it's a valid square root
    full_cov = sqrt_pred_cov @ sqrt_pred_cov.T
    eigenvalues = np.linalg.eigvals(full_cov)
    assert np.all(eigenvalues >= -1e-10)


def test_sqrt_predictor_returns_sqrt_prediction():
    """Test that SqrtKalmanPredictor returns SqrtGaussianStatePrediction"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = SqrtKalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior_mean = StateVector([[0.0], [1.0]])
    prior_covar = np.eye(2)
    sqrt_prior_covar = np.linalg.cholesky(prior_covar)
    prior = SqrtGaussianState(prior_mean, sqrt_prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, SqrtGaussianStatePrediction)
    assert hasattr(prediction, "sqrt_covar")


# Comprehensive tests for StochasticIntegrationPredictor class


def test_si_predictor_initialization():
    """Test StochasticIntegrationPredictor initialization"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = StochasticIntegrationPredictor(transition_model=transition_model)

    assert predictor.Nmax == 10
    assert predictor.Nmin == 5
    assert predictor.Eps == 5e-3
    assert predictor.SIorder == 5


def test_si_predictor_custom_parameters():
    """Test StochasticIntegrationPredictor with custom parameters"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = StochasticIntegrationPredictor(
        transition_model=transition_model,
        Nmax=15,
        Nmin=3,
        Eps=1e-3,
        SIorder=3,
    )

    assert predictor.Nmax == 15
    assert predictor.Nmin == 3
    assert predictor.Eps == 1e-3
    assert predictor.SIorder == 3


def test_si_predictor_with_linear_model():
    """Test SI predictor with linear model"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = StochasticIntegrationPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    # Should give reasonable results
    assert_allclose(prediction.mean, [[1.0], [1.0]], atol=0.1)


def test_si_predictor_with_nonlinear_model():
    """Test SI predictor with nonlinear model"""
    transition_model = ConstantTurn(
        linear_noise_coeffs=np.array([0.01, 0.01]), turn_noise_coeff=0.001
    )
    predictor = StochasticIntegrationPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior_mean = StateVector([[0.0], [1.0], [0.0], [0.0], [0.1]])
    prior_covar = np.eye(5) * 0.1
    prior = GaussianState(prior_mean, prior_covar, timestamp=timestamp)

    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert isinstance(prediction, GaussianStatePrediction)
    assert prediction.ndim == 5


# Test process noise integration across all predictors


@pytest.mark.parametrize(
    "PredictorClass",
    [
        KalmanPredictor,
        ExtendedKalmanPredictor,
        UnscentedKalmanPredictor,
        CubatureKalmanPredictor,
        StochasticIntegrationPredictor,
    ],
)
def test_process_noise_increases_uncertainty(PredictorClass):
    """Test that process noise increases uncertainty for all predictors"""
    # High noise coefficient
    transition_model = ConstantVelocity(noise_diff_coeff=1.0)
    predictor = PredictorClass(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2) * 0.1, timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Covariance should increase
    assert np.trace(prediction.covar) > np.trace(prior.covar)


@pytest.mark.parametrize(
    "PredictorClass",
    [
        KalmanPredictor,
        ExtendedKalmanPredictor,
        UnscentedKalmanPredictor,
        CubatureKalmanPredictor,
        StochasticIntegrationPredictor,
    ],
)
def test_zero_noise_preserves_linear_dynamics(PredictorClass):
    """Test that zero noise preserves linear dynamics exactly"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.0)
    predictor = PredictorClass(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior_covar = np.eye(2) * 0.5
    prior = GaussianState(StateVector([[0.0], [1.0]]), prior_covar, timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # For linear model with zero noise, covariance should only change via F @ P @ F.T
    F = transition_model.matrix(time_interval=datetime.timedelta(seconds=1.0))
    expected_cov = F @ prior_covar @ F.T

    assert_allclose(prediction.covar, expected_cov, atol=0.01)


# Test predictors with multi-dimensional models


def test_kalman_predictor_3d_constant_acceleration():
    """Test KalmanPredictor with 3D constant acceleration model"""
    # 3D: [x, v, a] - position, velocity, acceleration
    transition_model = ConstantAcceleration(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    # State: position=0, velocity=1, acceleration=0
    prior = GaussianState(
        StateVector([[0.0], [1.0], [0.0]]),
        np.eye(3),
        timestamp=timestamp,
    )
    new_timestamp = timestamp + datetime.timedelta(seconds=2.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    assert prediction.ndim == 3
    # With constant acceleration = 0 and v=1, position should be x = 0 + 1*2 = 2
    assert_allclose(prediction.mean[0], 2.0, atol=1e-9)
    assert_allclose(prediction.mean[1], 1.0, atol=1e-9)


# Test edge cases and boundary conditions


def test_kalman_predictor_zero_time_interval():
    """Test prediction with zero time interval"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)

    # Predict to same timestamp
    prediction = predictor.predict(prior=prior, timestamp=timestamp)

    # State should remain essentially unchanged (only process noise at dt=0)
    assert_allclose(prediction.mean, prior.mean, atol=1e-10)


def test_kalman_predictor_very_small_covariance():
    """Test prediction with very small initial covariance"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    # Very small covariance
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2) * 1e-10, timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Should still produce valid prediction
    assert np.all(np.linalg.eigvals(prediction.covar) >= 0)


def test_kalman_predictor_large_time_interval():
    """Test prediction with large time interval"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = KalmanPredictor(transition_model=transition_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    # Large time interval: 1 hour
    new_timestamp = timestamp + datetime.timedelta(hours=1.0)

    prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

    # Uncertainty should grow significantly
    assert np.trace(prediction.covar) > 100 * np.trace(prior.covar)


# Test predictors with various control models


def test_kalman_predictor_with_control_noise():
    """Test KalmanPredictor with control model that has noise"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.0)
    control_matrix = np.array([[1.0], [0.0]])
    control_noise = np.array([[0.5]])  # Non-zero control noise
    control_model = LinearControlModel(control_matrix=control_matrix, control_noise=control_noise)
    predictor = KalmanPredictor(transition_model=transition_model, control_model=control_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2) * 0.1, timestamp=timestamp)
    control_input = State(StateVector([[1.0]]), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    prediction = predictor.predict(
        prior=prior, timestamp=new_timestamp, control_input=control_input
    )

    # Control noise should contribute to predicted covariance
    # At minimum, predicted covariance should include control noise contribution
    assert np.trace(prediction.covar) > np.trace(prior.covar)


def test_kalman_predictor_none_control_input():
    """Test KalmanPredictor with None control input"""
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    control_matrix = np.array([[1.0], [0.5]])
    control_model = LinearControlModel(
        control_matrix=control_matrix, control_noise=np.zeros((1, 1))
    )
    predictor = KalmanPredictor(transition_model=transition_model, control_model=control_model)

    timestamp = datetime.datetime.now()
    prior = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp=timestamp)
    new_timestamp = timestamp + datetime.timedelta(seconds=1.0)

    # None control input should be handled gracefully
    prediction = predictor.predict(prior=prior, timestamp=new_timestamp, control_input=None)

    assert isinstance(prediction, GaussianStatePrediction)


# Test LRU cache behavior


def test_lru_cache_returns_same_object_for_same_input():
    """Test that cache returns same prediction object for identical inputs"""
    predictor = KalmanPredictor(ConstantVelocity(noise_diff_coeff=0.1))

    timestamp = datetime.datetime.now()
    state = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp)

    prediction_time = timestamp + datetime.timedelta(seconds=1)
    prediction1 = predictor.predict(state, prediction_time)
    prediction2 = predictor.predict(state, prediction_time)

    # Should return same object from cache
    assert prediction1 is prediction2


def test_lru_cache_invalidates_on_state_change():
    """Test that cache invalidates when state changes"""
    predictor = KalmanPredictor(ConstantVelocity(noise_diff_coeff=0.1))

    timestamp = datetime.datetime.now()
    state1 = GaussianState(StateVector([[0.0], [1.0]]), np.eye(2), timestamp)
    state2 = GaussianState(StateVector([[1.0], [1.0]]), np.eye(2), timestamp)

    prediction_time = timestamp + datetime.timedelta(seconds=1)
    prediction1 = predictor.predict(state1, prediction_time)
    prediction2 = predictor.predict(state2, prediction_time)

    # Should return different objects
    assert prediction1 is not prediction2
    assert not np.array_equal(prediction1.mean, prediction2.mean)
