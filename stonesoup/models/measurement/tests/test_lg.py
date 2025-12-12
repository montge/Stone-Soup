import pytest
from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ..linear import LinearGaussian
from ....types.state import State, ParticleState
from ....types.array import StateVector, StateVectors, CovarianceMatrix


@pytest.mark.parametrize(
    "H, R, ndim_state, mapping",
    [
        (       # 1D meas, 2D state
                np.array([[1, 0]]),
                np.array([[0.1]]),
                2,
                [0],
        ),
        (       # 2D meas, 4D state
                np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
                np.diag([0.1, 0.1]),
                4,
                [0, 2],
        ),
        (       # 4D meas, 2D state
                np.array([[1, 0], [0, 0], [0, 1], [0, 0]]),
                np.diag([0.1, 0.1, 0.1, 0.1]),
                2,
                [0, None, 1, None],
        ),
    ],
    ids=["1D_meas:2D_state", "2D_meas:4D_state", "4D_meas:2D_state"]
)
def test_lgmodel(H, R, ndim_state, mapping):
    """ LinearGaussian 1D Measurement Model test """

    # State related variables
    state_vec = np.array([[n] for n in range(ndim_state)])
    state = State(state_vec)

    # Create and a Constant Velocity model object
    lg = LinearGaussian(ndim_state=ndim_state,
                        noise_covar=R,
                        mapping=mapping)

    # Ensure ```lg.transfer_function()``` returns H
    assert np.array_equal(H, lg.matrix())

    # Ensure lg.jacobian() returns H
    assert np.array_equal(H, lg.jacobian(state=state))

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, lg.covar())

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = lg.function(state)
    assert np.array_equal(meas_pred_wo_noise, H@state_vec)

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = lg.pdf(State(meas_pred_wo_noise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = lg.function(state, noise=lg.rvs())
    assert not np.array_equal(meas_pred_w_inoise, H@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(State(meas_pred_w_inoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with external noise)
    noise = lg.rvs()
    meas_pred_w_enoise = lg.function(state,
                                     noise=noise)
    assert np.array_equal(meas_pred_w_enoise, H@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(State(meas_pred_w_enoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Test random seed give consistent results
    lg1 = LinearGaussian(ndim_state=ndim_state,
                         noise_covar=R,
                         mapping=mapping,
                         seed=1)
    lg2 = LinearGaussian(ndim_state=ndim_state,
                         noise_covar=R,
                         mapping=mapping,
                         seed=1)

    # Check first values produced by seed match
    for _ in range(3):
        assert all(lg1.rvs() == lg2.rvs())


# Additional comprehensive tests for LinearGaussian

def test_ndim_meas_derived_from_mapping():
    """Test that ndim_meas is derived from mapping length"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2))
    assert lg.ndim_meas == 2

    lg = LinearGaussian(ndim_state=6, mapping=[0, 1, 2, 3], noise_covar=np.eye(4))
    assert lg.ndim_meas == 4


def test_ndim_property_returns_ndim_meas():
    """Test that ndim returns ndim_meas"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 1, 2], noise_covar=np.eye(3))
    assert lg.ndim == lg.ndim_meas
    assert lg.ndim == 3


def test_covariance_matrix_conversion():
    """Test that noise_covar is converted to CovarianceMatrix"""
    R = np.array([[1.0, 0.5], [0.5, 2.0]])
    lg = LinearGaussian(ndim_state=4, mapping=[0, 1], noise_covar=R)
    assert isinstance(lg.noise_covar, CovarianceMatrix)
    assert np.array_equal(lg.noise_covar, R)


def test_matrix_simple_identity():
    """Test matrix for simple identity measurement"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1, 2], noise_covar=np.eye(3))
    H = lg.matrix()
    expected = np.eye(3)
    assert np.array_equal(H, expected)


def test_matrix_partial_observation():
    """Test matrix for partial state observation"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2))
    H = lg.matrix()
    expected = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])
    assert np.array_equal(H, expected)


def test_matrix_with_none_mapping():
    """Test matrix with None values in mapping"""
    lg = LinearGaussian(ndim_state=2, mapping=[0, None, 1], noise_covar=np.eye(3))
    H = lg.matrix()
    expected = np.array([[1, 0],
                        [0, 0],
                        [0, 1]])
    assert np.array_equal(H, expected)


def test_matrix_all_none_mapping():
    """Test matrix with all None mapping"""
    lg = LinearGaussian(ndim_state=3, mapping=[None, None], noise_covar=np.eye(2))
    H = lg.matrix()
    expected = np.zeros((2, 3))
    assert np.array_equal(H, expected)


def test_matrix_non_sequential_mapping():
    """Test matrix with non-sequential mapping"""
    lg = LinearGaussian(ndim_state=5, mapping=[4, 1, 0], noise_covar=np.eye(3))
    H = lg.matrix()
    expected = np.array([[0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0]])
    assert np.array_equal(H, expected)


def test_function_with_external_noise():
    """Test function with externally provided noise"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))
    noise = StateVector([0.1, 0.2])

    result = lg.function(state, noise=noise)
    expected = StateVector([1.1, 2.2])
    assert np.allclose(result, expected)


def test_function_with_none_noise():
    """Test function with noise=None (should be no noise)"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))
    result = lg.function(state, noise=None)
    expected = StateVector([1, 2])
    assert np.array_equal(result, expected)


def test_function_with_none_in_mapping():
    """Test function with None in mapping"""
    lg = LinearGaussian(ndim_state=2, mapping=[0, None, 1], noise_covar=np.eye(3))
    state = State(StateVector([5, 10]))
    result = lg.function(state, noise=False)
    expected = StateVector([5, 0, 10])
    assert np.array_equal(result, expected)


def test_rvs_zero_mean():
    """Test that rvs samples have approximately zero mean"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 1, 2], noise_covar=np.eye(3), seed=42)
    samples = lg.rvs(num_samples=10000)
    mean = np.mean(samples, axis=1)
    assert np.allclose(mean, 0, atol=0.1)


def test_rvs_correct_covariance():
    """Test that rvs samples have correct covariance"""
    R = np.array([[2.0, 0.5], [0.5, 1.0]])
    lg = LinearGaussian(ndim_state=4, mapping=[0, 1], noise_covar=R, seed=42)
    samples = lg.rvs(num_samples=10000)
    sample_cov = np.cov(samples)
    assert np.allclose(sample_cov, R, atol=0.1)


def test_pdf_exact_match():
    """Test PDF when measurement matches prediction exactly"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))
    measurement = State(StateVector([1, 2]))

    prob = lg.pdf(measurement, state)
    expected_prob = multivariate_normal.pdf([0, 0], cov=np.eye(2))
    assert approx(prob) == expected_prob


def test_pdf_with_offset():
    """Test PDF with offset between measurement and prediction"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))
    measurement = State(StateVector([1.5, 2.5]))

    prob = lg.pdf(measurement, state)
    expected_prob = multivariate_normal.pdf([0.5, 0.5], cov=np.eye(2))
    assert approx(prob) == expected_prob


def test_pdf_with_correlated_noise():
    """Test PDF with correlated noise covariance"""
    R = np.array([[2.0, 0.5], [0.5, 1.0]])
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=R)
    state = State(StateVector([1, 2, 3]))
    measurement = State(StateVector([1.5, 2.5]))

    prob = lg.pdf(measurement, state)
    expected_prob = multivariate_normal.pdf([0.5, 0.5], cov=R)
    assert approx(prob) == expected_prob


def test_logpdf_consistency():
    """Test log-PDF matches log of PDF"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))
    measurement = State(StateVector([1.5, 2.5]))

    logprob = lg.logpdf(measurement, state)
    prob = lg.pdf(measurement, state)
    assert approx(logprob) == np.log(prob)


def test_jacobian_equals_matrix():
    """Test that Jacobian equals measurement matrix for linear models"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3, 4]))

    H = lg.matrix()
    J = lg.jacobian(state)
    assert np.array_equal(H, J)


def test_jacobian_independent_of_state():
    """Test that Jacobian is independent of state value for linear model"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 1], noise_covar=np.eye(2))

    state1 = State(StateVector([1, 2, 3, 4]))
    state2 = State(StateVector([10, 20, 30, 40]))

    J1 = lg.jacobian(state1)
    J2 = lg.jacobian(state2)
    assert np.array_equal(J1, J2)


def test_single_dimension_measurement():
    """Test 1D measurement from multi-D state"""
    lg = LinearGaussian(ndim_state=5, mapping=[2], noise_covar=np.array([[0.5]]))
    state = State(StateVector([1, 2, 3, 4, 5]))
    result = lg.function(state, noise=False)
    assert result.shape == (1, 1)
    assert result[0, 0] == 3


def test_all_state_dimensions_measured():
    """Test measuring all state dimensions"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1, 2], noise_covar=np.eye(3))
    state = State(StateVector([1, 2, 3]))
    result = lg.function(state, noise=False)
    assert np.array_equal(result, state.state_vector)


def test_full_measurement_cycle():
    """Test complete measurement generation and evaluation cycle"""
    lg = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.eye(2), seed=42)
    state = State(StateVector([1, 2, 3, 4]))

    # Generate measurement with noise
    measurement_vec = lg.function(state, noise=True)
    measurement = State(measurement_vec)

    # Evaluate likelihood
    prob = lg.pdf(measurement, state)
    assert prob > 0
    assert prob <= 1


def test_measurement_update_scenario():
    """Test realistic measurement update scenario"""
    # Position-only measurement of position-velocity state
    lg = LinearGaussian(
        ndim_state=4,  # [x, vx, y, vy]
        mapping=[0, 2],  # Measure [x, y] only
        noise_covar=np.diag([0.5, 0.5]),
        seed=42
    )

    true_state = State(StateVector([10, 1, 20, 2]))  # position and velocity
    measurement = lg.function(true_state, noise=True)

    # Measurement should be close to true position
    assert measurement.shape == (2, 1)
    assert np.allclose(measurement[0, 0], 10, atol=3)  # Within ~3 sigma
    assert np.allclose(measurement[1, 0], 20, atol=3)


def test_comparison_with_scipy_multivariate_normal():
    """Test that PDF matches scipy's multivariate_normal exactly"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state = State(StateVector([1, 2, 3]))

    # Test several measurement values
    test_measurements = [
        StateVector([1, 2]),
        StateVector([1.5, 2.5]),
        StateVector([0, 1]),
        StateVector([2, 3]),
    ]

    for meas_vec in test_measurements:
        measurement = State(meas_vec)
        prob = lg.pdf(measurement, state)

        # Calculate expected using scipy directly
        predicted = lg.function(state, noise=False)
        residual = (meas_vec - predicted).ravel()
        expected_prob = multivariate_normal.pdf(residual, cov=lg.noise_covar)

        assert approx(prob) == expected_prob


def test_function_with_particle_state():
    """Test function with multiple state vectors (particles)"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2), seed=42)
    state_vectors = StateVectors([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    state = ParticleState(state_vectors, weight=[1/3, 1/3, 1/3])

    result = lg.function(state, noise=False)
    assert result.shape == (2, 3)
    assert np.array_equal(result[:, 0], [1, 4])
    assert np.array_equal(result[:, 1], [2, 5])
    assert np.array_equal(result[:, 2], [3, 6])


def test_pdf_with_particle_state():
    """Test PDF with particle state"""
    lg = LinearGaussian(ndim_state=3, mapping=[0, 1], noise_covar=np.eye(2))
    state_vectors = StateVectors([[1, 2], [2, 3], [3, 4]])
    state = ParticleState(state_vectors, weight=[1/2, 1/2])
    measurement = State(StateVector([1.5, 2.5]))

    prob = lg.pdf(measurement, state)
    assert isinstance(prob, np.ndarray)
    assert len(prob) == 2
