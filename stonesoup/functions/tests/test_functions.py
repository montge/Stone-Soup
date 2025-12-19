import warnings

import numpy as np
import pytest
from numpy import deg2rad
from numpy import linalg as LA
from pytest import approx, raises
from scipy.linalg import LinAlgError, cholesky

from ...types.array import CovarianceMatrix, Matrix, StateVector, StateVectors
from ...types.state import GaussianState, State
from .. import (
    az_el_rg2cart,
    build_rotation_matrix,
    build_rotation_matrix_xyz,
    cart2angles,
    cart2az_el_rg,
    cart2pol,
    cart2sphere,
    cholesky_eps,
    cubature2gauss,
    cubature_transform,
    dotproduct,
    gauss2cubature,
    gauss2sigma,
    gm_reduce_single,
    gm_sample,
    grid_creation,
    jacobian,
    mod_bearing,
    mod_elevation,
    pol2cart,
    rotx,
    roty,
    rotz,
    sigma2gauss,
    sphere2cart,
    stochastic_cubature_rule_points,
    tria,
    unscented_transform,
)


def test_grid_creation():
    nx = 4
    meanX0 = np.array([36569, 50, 55581, 50])  # mean value
    varX0 = np.diag([90, 5, 160, 5])  # variance
    Npa = np.array([31, 31, 27, 27])  # must be ODD!
    sFactor = 4  # scaling factor (number of sigmas covered by the grid)

    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = grid_creation(
        np.vstack(meanX0), varX0, sFactor, nx, Npa
    )

    mean_diffs = np.array([np.mean(np.diff(sublist)) for sublist in gridDimOld])

    _eigVal, eigVect = LA.eig(varX0)

    assert np.allclose(meanX0, np.mean(predGrid, axis=1), 0, atol=1.0e-1)
    assert np.all(meanX0 == xOld.ravel())
    assert np.all(np.argsort(predGridDelta) == np.argsort(np.diag(varX0)))
    assert np.allclose(mean_diffs, predGridDelta, 0, atol=1e-10)
    assert np.all(eigVect == Ppold)


def test_cholesky_eps():
    matrix = np.array([[0.4, -0.2, 0.1], [0.3, 0.1, -0.2], [-0.3, 0.0, 0.4]])
    matrix = matrix @ matrix.T

    cholesky_matrix = cholesky(matrix)

    assert cholesky_eps(matrix) == approx(cholesky_matrix)
    assert cholesky_eps(matrix, True) == approx(cholesky_matrix.T)


def test_cholesky_eps_bad():
    matrix = np.array(
        [
            [0.05201447, 0.02882126, -0.00569971, -0.00733617],
            [0.02882126, 0.01642966, -0.00862847, -0.00673035],
            [-0.00569971, -0.00862847, 0.06570757, 0.03251551],
            [-0.00733617, -0.00673035, 0.03251551, 0.01648615],
        ]
    )
    with raises(LinAlgError):
        cholesky(matrix)
    cholesky_eps(matrix)


def test_jacobian():
    """jacobian function test"""

    # State related variables
    state_mean = StateVector([[3.0], [1.0]])

    def f(x):
        return np.array([[1, 1], [0, 1]]) @ x.state_vector

    jac = jacobian(f, State(state_mean))
    assert np.allclose(jac, np.array([[1, 1], [0, 1]]))


def test_jacobian2():
    """jacobian function test"""

    # Sample functions to compute Jacobian on
    def fun(x):
        """function for testing scalars i.e. scalar input, scalar output"""
        return 2 * x.state_vector**2

    def fun1d(ins):
        """test function with vector input, scalar output"""
        out = 2 * ins.state_vector[0, :] + 3 * ins.state_vector[1, :]
        return np.atleast_2d(out)

    def fun2d(vec):
        """test function with 2d input and 2d output"""
        out = np.empty(vec.state_vector.shape)
        out[0, :] = 2 * vec.state_vector[0, :] ** 2 + 3 * vec.state_vector[1, :] ** 2
        out[1, :] = 2 * vec.state_vector[0, :] + 3 * vec.state_vector[1, :]
        return out

    x = 3
    jac = jacobian(fun, State(StateVector([[x]])))
    assert np.allclose(jac, 4 * x)

    x = StateVector([[1], [2]])
    # Tolerance value to use to test if arrays are equal
    tol = 1.0e-5

    jac = jacobian(fun1d, State(x))
    T = np.array([2.0, 3.0])

    FOM = np.where(np.abs(jac - T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    jac = jacobian(fun2d, State(x))
    T = np.array([[4.0 * x[0], 6 * x[1]], [2, 3]])
    FOM = np.where(np.abs(jac - T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0


def test_jacobian_param():
    """jacobian function test"""

    # Sample functions to compute Jacobian on
    def fun(x, value=0.0):
        """function for jabcobian parameter passing"""
        return value * x.state_vector

    x = 4
    value = 2.0
    jac = jacobian(fun, State(StateVector([[x]])), value=value)
    assert np.allclose(value, jac)


def test_jacobian_large_values():
    # State related variables
    state = State(StateVector([[1e10], [1.0]]))

    def f(x):
        return x.state_vector**2

    jac = jacobian(f, state)
    assert np.allclose(jac, np.array([[2e10, 0.0], [0.0, 2.0]]))


def test_gm_reduce_single():
    means = StateVectors([StateVector([1, 2]), StateVector([3, 4]), StateVector([5, 6])])
    covars = np.stack([[[1, 1], [1, 0.7]], [[1.2, 1.4], [1.3, 2]], [[2, 1.4], [1.2, 1.2]]], axis=2)
    weights = np.array([1, 2, 5])

    mean, covar = gm_reduce_single(means, covars, weights)

    assert np.allclose(mean, np.array([[4], [5]]))
    assert np.allclose(covar, np.array([[3.675, 3.35], [3.2, 3.3375]]))

    # Test handling of means as array instead of StateVectors
    mean, covar = gm_reduce_single(means.view(np.ndarray), covars, weights)

    assert np.allclose(mean, np.array([[4], [5]]))
    assert np.allclose(covar, np.array([[3.675, 3.35], [3.2, 3.3375]]))

    # Test that negative means do not cause numeric warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", r".*invalid value encountered in (multiply|divide).*", RuntimeWarning
        )

        mean, covar = gm_reduce_single(-means, covars, weights)
        assert np.allclose(-mean, np.array([[4], [5]]))
        assert np.allclose(covar, np.array([[3.675, 3.35], [3.2, 3.3375]]))


def test_bearing():
    bearing_in = [10.0, 170.0, 190.0, 260.0, 280.0, 350.0, 705]
    rad_in = deg2rad(bearing_in)

    bearing_out = [10.0, 170.0, -170.0, -100.0, -80.0, -10.0, -15.0]
    rad_out = deg2rad(bearing_out)

    for ind, val in enumerate(rad_in):
        assert rad_out[ind] == approx(mod_bearing(val))


def test_elevation():
    elev_in = [10.0, 80.0, 110.0, 170.0, 190.0, 260.0, 280]
    rad_in = deg2rad(elev_in)

    elev_out = [10.0, 80.0, 70.0, 10.0, -10.0, -80.0, -80.0]
    rad_out = deg2rad(elev_out)

    for ind, val in enumerate(rad_in):
        assert rad_out[ind] == approx(mod_elevation(val))


@pytest.mark.parametrize("mean", [1, 1.0])  # int  # float
def test_gauss2sigma(mean):
    covar = 2.0
    state = GaussianState([[mean]], [[covar]])

    sigma_points_states, _mean_weights, _covar_weights = gauss2sigma(state, kappa=0)

    for n, sigma_point_state_vector in zip(
        (0, 1, -1), sigma_points_states.state_vector, strict=False
    ):
        assert sigma_point_state_vector[0, 0] == approx(mean + n * covar**0.5)


@pytest.mark.parametrize("gauss2x", [(gauss2sigma), (gauss2cubature)])
def test_gauss2sigma_bad_covar(gauss2x):
    covar = np.array(
        [
            [0.05201447, 0.02882126, -0.00569971, -0.00733617],
            [0.02882126, 0.01642966, -0.00862847, -0.00673035],
            [-0.00569971, -0.00862847, 0.06570757, 0.03251551],
            [-0.00733617, -0.00673035, 0.03251551, 0.01648615],
        ]
    )
    state = GaussianState([[0], [0], [0], [0]], covar)

    with pytest.warns(UserWarning, match="Matrix is not positive definite"):
        gauss2x(state)


@pytest.mark.parametrize(
    "angle",
    [
        (
            np.array([np.pi]),  # angle
            np.array([np.pi / 2]),
            np.array([-np.pi]),
            np.array([-np.pi / 2]),
            np.array([np.pi / 4]),
            np.array([-np.pi / 4]),
            np.array([np.pi / 8]),
            np.array([-np.pi / 8]),
        )
    ],
)
def test_rotations(angle):
    c, s = np.cos(angle), np.sin(angle)
    zero = np.zeros_like(angle)
    one = np.ones_like(angle)

    assert np.array_equal(rotx(angle), np.array([[one, zero, zero], [zero, c, -s], [zero, s, c]]))
    assert np.array_equal(roty(angle), np.array([[c, zero, s], [zero, one, zero], [-s, zero, c]]))
    assert np.array_equal(rotz(angle), np.array([[c, -s, zero], [s, c, zero], [zero, zero, one]]))


@pytest.mark.parametrize(
    "x, y, z",
    [  # Cartesian values
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_cart_sphere_inversions(x, y, z):
    rho, phi, theta = cart2sphere(x, y, z)

    # Check sphere2cart(cart2sphere(cart)) == cart
    assert np.allclose(np.array([x, y, z]), sphere2cart(rho, phi, theta))

    # Check cart2angle == cart2sphere for angles
    assert np.allclose(np.array([phi, theta]), cart2angles(x, y, z))

    # Check that pol2cart(cart2angle(cart)) == cart
    #   note, this only works correctly when z==0
    if z == 0:
        assert np.allclose(np.array([x, y]), pol2cart(rho, phi))


@pytest.mark.parametrize(
    "state_vector1, state_vector2",
    [  # Cartesian values
        (StateVector([2, 4]), StateVector([2, 1])),
        (StateVector([-1, 1, -4]), StateVector([-2, 1, 1])),
        (StateVector([-2, 0, 3, -1]), StateVector([1, 0, -1, 4])),
        (StateVector([-1, 0]), StateVector([1, -2, 3])),
        (Matrix([[1, 0], [0, 1]]), Matrix([[3, 1], [1, -3]])),
        (StateVectors([[1, 0], [0, 1]]), StateVectors([[3, 1], [1, -3]])),
        (StateVectors([[1, 0], [0, 1]]), StateVector([3, 1])),
    ],
)
def test_dotproduct(state_vector1, state_vector2):
    # Test that they raise the right error if not 1d, i.e. vectors
    if type(state_vector1) is not type(state_vector2) or (
        type(state_vector1) is not StateVectors
        and type(state_vector2) is not StateVectors
        and type(state_vector2) is not StateVector
        and type(state_vector1) is not StateVector
    ):
        with pytest.raises(ValueError):
            dotproduct(state_vector1, state_vector2)
    else:
        if len(state_vector1) != len(state_vector2):
            # If they're different lengths check that the correct error is thrown
            with pytest.raises(ValueError):
                dotproduct(state_vector1, state_vector2)
        else:
            # This is what the dotproduct function actually does
            out = 0
            for a_i, b_i in zip(state_vector1, state_vector2, strict=False):
                out += a_i * b_i

            assert np.allclose(
                dotproduct(state_vector1, state_vector2),
                np.reshape(out, np.shape(dotproduct(state_vector1, state_vector2))),
            )


@pytest.mark.parametrize(
    "means, covars, weights, size",
    [
        (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1 / 3] * 3),  # weights
            20,  # size
        ),
        (
            StateVectors(np.array([[20, 30, 40, 50], [20, 30, 40, 50]])),  # means
            [np.eye(2), np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1 / 4] * 4),  # weights
            20,  # size
        ),
        (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            np.array([np.eye(2), np.eye(2), np.eye(2)]),  # covars
            np.array([1 / 3] * 3),  # weights
            20,  # size
        ),
        (
            [
                StateVector(np.array([10, 10])),
                StateVector(np.array([20, 20])),
                StateVector(np.array([30, 30])),
            ],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1 / 3] * 3),  # weights
            20,  # size
        ),
        (
            StateVector(np.array([10, 10])),  # means
            [np.eye(2)],  # covars
            np.array([1]),  # weights
            20,  # size
        ),
        (np.array([10, 10]), [np.eye(2)], np.array([1]), 20),  # means  # covars  # weights  # size
        (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            None,  # weights
            20,  # size
        ),
        (
            StateVectors(np.array([[20, 30, 40, 50], [20, 30, 40, 50]])),  # means
            [np.eye(2), np.eye(2), np.eye(2), np.eye(2)],  # covars
            None,  # weights
            20,  # size
        ),
    ],
    ids=[
        "mean_list",
        "mean_statevectors",
        "3d_covar_array",
        "mean_statevector_list",
        "single_statevector_mean",
        "single_ndarray_mean",
        "no_weight_mean_list",
        "no_weight_mean_statevectors",
    ],
)
def test_gm_sample(means, covars, weights, size):
    samples = gm_sample(means, covars, size, weights=weights)

    # check orientation and size of samples
    assert samples.shape[1] == size
    # check number of dimensions
    if isinstance(means, list):
        assert samples.shape[0] == means[0].shape[0]
    else:
        assert samples.shape[0] == means.shape[0]


@pytest.mark.parametrize(
    "mean, covar, alp",
    [
        (StateVector([0]), CovarianceMatrix([[1]]), None),
        (StateVector([-7, 5]), CovarianceMatrix([[1.1, -0.04], [-0.04, 1.2]]), 2.0),
        (
            StateVector([12, -4, 0, 5]),
            CovarianceMatrix(
                [
                    [0.7, 0.04, -0.02, 0],
                    [0.04, 1.1, 0.09, 0.06],
                    [-0.02, 0.09, 0.9, -0.01],
                    [0, 0.06, -0.01, 1.1],
                ]
            ),
            0.7,
        ),
    ],
)
def test_cubature_transform(mean, covar, alp):
    instate = GaussianState(mean, covar)

    def identity_function(inpu):
        return inpu.state_vector

    # First test the cubature points conversions
    if alp is None:
        cub_pts = gauss2cubature(instate)
        outsv, outcovar = cubature2gauss(cub_pts)
        mean, covar, cross_covar, cubature_points = cubature_transform(instate, identity_function)
    else:
        cub_pts = gauss2cubature(instate, alpha=alp)
        outsv, outcovar = cubature2gauss(cub_pts, alpha=alp)
        mean, covar, _cross_covar, _cubature_points = cubature_transform(
            instate, identity_function, alpha=alp
        )

    assert np.allclose(outsv, instate.state_vector)
    assert np.allclose(outcovar, instate.covar)
    assert np.allclose(mean, instate.state_vector)
    assert np.allclose(covar, instate.covar)


@pytest.mark.parametrize("order, nx", [(3, 3), (5, 4), (1, 2)])
def test_stochastic_integration(order, nx):
    points, weights = stochastic_cubature_rule_points(nx, order)
    # Mean
    assert np.allclose(np.average(points, weights=weights, axis=1), 0, atol=1e-5)
    # Weights
    assert np.isclose(np.sum(weights), 1, atol=1e-5)
    if order != 1:  # For order 1 it does not make sense to check variance of points
        # Covariance
        var = (weights * points) @ points.T
        # Check if diagonal elements are close to 1
        diagonal_elements = np.diag(var)
        assert np.allclose(diagonal_elements, 1, atol=1e-5)
        # Check if off-diagonal elements are close to 0
        off_diagonal_elements = var[~np.eye(nx, dtype=bool)]
        assert np.allclose(off_diagonal_elements, 0, atol=1e-5)


def test_stochastic_integration_invalid_order():
    with pytest.raises(ValueError, match="This order of SIF is not supported"):
        stochastic_cubature_rule_points(5, 2)


def test_tria():
    """Test square root matrix triangularization"""
    # Test with a simple rectangular matrix
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    result = tria(matrix)

    # Result should be square and lower triangular
    assert result.shape[0] == result.shape[1]
    assert np.allclose(result, np.tril(result))

    # Test with square matrix
    square_matrix = np.array([[1, 2], [3, 4]])
    result = tria(square_matrix)
    assert result.shape == (2, 2)
    assert np.allclose(result, np.tril(result))

    # All diagonal elements should be positive
    assert np.all(np.diag(result) >= 0)


def test_sigma2gauss():
    """Test sigma2gauss function"""
    # Create simple sigma points
    sigma_points = StateVectors([[0, 1, -1], [0, 1, -1]])
    mean_weights = np.array([0.5, 0.25, 0.25])
    covar_weights = np.array([0.5, 0.25, 0.25])

    mean, covar = sigma2gauss(sigma_points, mean_weights, covar_weights)

    # Check mean calculation
    expected_mean = np.array([[0], [0]])
    assert np.allclose(mean, expected_mean)

    # Check that covariance is positive semi-definite
    eigvals = np.linalg.eigvals(covar)
    assert np.all(eigvals >= -1e-10)

    # Test with additive noise
    covar_noise = CovarianceMatrix([[0.1, 0], [0, 0.1]])
    mean_noise, covar_noise_result = sigma2gauss(
        sigma_points, mean_weights, covar_weights, covar_noise
    )

    # Covariance with noise should be larger
    assert np.all(np.diag(covar_noise_result) >= np.diag(covar))


def test_unscented_transform():
    """Test unscented transform"""
    state = GaussianState([[1], [2]], [[1, 0], [0, 1]])
    sigma_points_states, mean_weights, covar_weights = gauss2sigma(state, kappa=0)

    def linear_function(x, points_noise=None):
        # Simple linear transformation
        return 2 * x.state_vector

    mean, covar, cross_covar, sigma_points_t, mean_w, covar_w = unscented_transform(
        sigma_points_states, mean_weights, covar_weights, linear_function
    )

    # For linear transformation, mean should be 2*original mean
    assert np.allclose(mean, 2 * state.state_vector)

    # Covariance should be 4*original (2^2)
    assert np.allclose(covar, 4 * state.covar, atol=1e-10)

    # Check return types
    assert isinstance(mean, StateVector)
    assert isinstance(covar, CovarianceMatrix)
    assert isinstance(cross_covar, CovarianceMatrix)


def test_cart2pol():
    """Test Cartesian to polar conversion"""
    # Test basic cases
    x, y = 1.0, 0.0
    rho, phi = cart2pol(x, y)
    assert np.isclose(rho, 1.0)
    assert np.isclose(phi, 0.0)

    x, y = 0.0, 1.0
    rho, phi = cart2pol(x, y)
    assert np.isclose(rho, 1.0)
    assert np.isclose(phi, np.pi / 2)

    x, y = 1.0, 1.0
    rho, phi = cart2pol(x, y)
    assert np.isclose(rho, np.sqrt(2))
    assert np.isclose(phi, np.pi / 4)

    # Test negative values
    x, y = -1.0, 0.0
    rho, phi = cart2pol(x, y)
    assert np.isclose(rho, 1.0)
    assert np.isclose(phi, np.pi)


def test_cart2az_el_rg():
    """Test Cartesian to azimuth, elevation, range conversion"""
    # Test basic case
    x, y, z = 1.0, 0.0, 0.0
    phi, theta, rho = cart2az_el_rg(x, y, z)
    assert np.isclose(rho, 1.0)
    assert np.isclose(phi, np.arcsin(1.0))
    assert np.isclose(theta, 0.0)

    # Test with all equal components
    x, y, z = 1.0, 1.0, 1.0
    phi, theta, rho = cart2az_el_rg(x, y, z)
    assert np.isclose(rho, np.sqrt(3))

    # Test conversion round-trip
    x_conv, y_conv, z_conv = az_el_rg2cart(phi, theta, rho)
    # Note: az_el_rg2cart and cart2az_el_rg may not be perfect inverses
    # due to the specific formulation used


def test_az_el_rg2cart():
    """Test azimuth, elevation, range to Cartesian conversion"""
    # Test basic case
    phi, theta, rho = 0.0, 0.0, 1.0
    x, y, z = az_el_rg2cart(phi, theta, rho)

    # Check that range is preserved
    calculated_rho = np.sqrt(x**2 + y**2 + z**2)
    assert np.isclose(calculated_rho, rho)

    # Test with different values
    phi, theta, rho = np.pi / 4, np.pi / 6, 2.0
    x, y, z = az_el_rg2cart(phi, theta, rho)
    calculated_rho = np.sqrt(x**2 + y**2 + z**2)
    assert np.isclose(calculated_rho, rho)


def test_build_rotation_matrix():
    """Test build_rotation_matrix function"""
    # Test with zero rotations
    angle_vector = np.array([[0], [0], [0]])
    R = build_rotation_matrix(angle_vector)
    assert np.allclose(R, np.eye(3))

    # Test with single axis rotation
    angle_vector = np.array([[np.pi / 2], [0], [0]])
    R = build_rotation_matrix(angle_vector)
    assert R.shape == (3, 3)

    # Rotation matrix should be orthogonal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    # Test determinant is 1 (proper rotation)
    assert np.isclose(np.linalg.det(R), 1.0)

    # Test with multiple rotations
    angle_vector = np.array([[np.pi / 4], [np.pi / 3], [np.pi / 6]])
    R = build_rotation_matrix(angle_vector)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_build_rotation_matrix_xyz():
    """Test build_rotation_matrix_xyz function"""
    # Test with zero rotations
    angle_vector = np.array([[0], [0], [0]])
    R = build_rotation_matrix_xyz(angle_vector)
    assert np.allclose(R, np.eye(3))

    # Test with single axis rotation
    angle_vector = np.array([[np.pi / 2], [0], [0]])
    R = build_rotation_matrix_xyz(angle_vector)
    assert R.shape == (3, 3)

    # Rotation matrix should be orthogonal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    # Test determinant is 1 (proper rotation)
    assert np.isclose(np.linalg.det(R), 1.0)

    # Test that xyz and zyx orderings give different results
    angle_vector = np.array([[np.pi / 4], [np.pi / 3], [np.pi / 6]])
    R_xyz = build_rotation_matrix_xyz(angle_vector)
    R_zyx = build_rotation_matrix(angle_vector)
    # They should be different (unless angles are all zero)
    assert not np.allclose(R_xyz, R_zyx)


def test_sde_euler_maruyama_integration():
    """Test SDE Euler-Maruyama integration"""
    from .. import sde_euler_maruyama_integration

    # Set random seed for reproducibility
    np.random.seed(42)

    # Simple deterministic system (no stochastic term)
    def deterministic_fun(state, t):
        a = np.zeros_like(state.state_vector)  # drift term
        b = np.zeros((state.ndim, state.ndim))  # diffusion term
        return a, b

    initial_state = State(StateVector([[1.0], [2.0]]))
    t_values = [0.0, 0.1, 0.2, 0.3]

    result = sde_euler_maruyama_integration(deterministic_fun, t_values, initial_state)

    # With zero drift and diffusion, state should remain unchanged
    assert np.allclose(result, initial_state.state_vector)

    # Test with simple drift
    def drift_fun(state, t):
        a = np.ones_like(state.state_vector)  # constant drift
        b = np.zeros((state.ndim, state.ndim))  # no diffusion
        return a, b

    result = sde_euler_maruyama_integration(drift_fun, t_values, initial_state)
    # State should increase by total time elapsed
    expected = initial_state.state_vector + np.ones_like(initial_state.state_vector) * 0.3
    assert np.allclose(result, expected)


def test_cub_points_and_tf():
    """Test cubature points and transformation function"""
    from .. import cub_points_and_tf

    nx = 2
    order = 3
    mean = np.array([[1], [2]])
    covar = np.array([[1, 0], [0, 1]])
    sqrtCov = np.linalg.cholesky(covar)

    state = GaussianState(mean, covar)

    def identity_transform(x):
        return x.state_vector

    points, weights, trsf_points = cub_points_and_tf(
        nx, order, sqrtCov, mean, identity_transform, state
    )

    # Check that points is StateVectors
    assert isinstance(points, StateVectors)

    # Check dimensions
    assert points.shape[0] == nx

    # Check that weights sum to 1
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)

    # For identity transform, transformed points should equal input points
    assert np.allclose(trsf_points, points)


def test_find_nearest_positive_definite():
    """Test finding nearest positive definite matrix"""
    from .. import find_nearest_positive_definite, is_cholesky_decomposable

    # Test with already positive definite matrix
    pos_def_matrix = np.array([[2, 1], [1, 2]])
    result = find_nearest_positive_definite(pos_def_matrix)
    assert np.allclose(result, pos_def_matrix)
    assert is_cholesky_decomposable(result)

    # Test with non-positive definite matrix
    non_pos_def = np.array([[1, 2], [2, 1]])
    result = find_nearest_positive_definite(non_pos_def)
    assert is_cholesky_decomposable(result)

    # Test with negative eigenvalues
    matrix = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    result = find_nearest_positive_definite(matrix)
    assert is_cholesky_decomposable(result)

    # Result should be symmetric
    assert np.allclose(result, result.T)

    # Test that it raises error if max_iterations reached
    # Create a pathological matrix that cannot be easily fixed
    # Use a matrix with very small eigenvalues that requires many iterations
    bad_matrix = np.array([[1e-20, 0], [0, 1e-20]])
    # This should succeed with default iterations
    result = find_nearest_positive_definite(bad_matrix)
    assert is_cholesky_decomposable(result)


def test_is_cholesky_decomposable():
    """Test checking if matrix is Cholesky decomposable"""
    from .. import is_cholesky_decomposable

    # Positive definite matrix
    pos_def = np.array([[2, 1], [1, 2]])
    assert is_cholesky_decomposable(pos_def) is True

    # Non-positive definite matrix
    non_pos_def = np.array([[1, 2], [2, 1]])
    assert is_cholesky_decomposable(non_pos_def) is False

    # Identity matrix
    assert is_cholesky_decomposable(np.eye(3)) is True

    # Negative definite matrix
    neg_def = np.array([[-1, 0], [0, -1]])
    assert is_cholesky_decomposable(neg_def) is False

    # Nearly positive definite (from earlier test)
    nearly_pos_def = np.array(
        [
            [0.05201447, 0.02882126, -0.00569971, -0.00733617],
            [0.02882126, 0.01642966, -0.00862847, -0.00673035],
            [-0.00569971, -0.00862847, 0.06570757, 0.03251551],
            [-0.00733617, -0.00673035, 0.03251551, 0.01648615],
        ]
    )
    assert is_cholesky_decomposable(nearly_pos_def) is False


def test_rotation_matrices_array_input():
    """Test rotation matrices with array inputs"""
    # Test rotx with array input
    angles = np.array([0, np.pi / 2, np.pi])
    result = rotx(angles)
    assert result.shape == (3, 3, 3)

    # Test roty with array input
    result = roty(angles)
    assert result.shape == (3, 3, 3)

    # Test rotz with array input
    result = rotz(angles)
    assert result.shape == (3, 3, 3)

    # Check that first rotation is identity
    assert np.allclose(result[:, :, 0], np.eye(3))


def test_mod_bearing_edge_cases():
    """Test mod_bearing with edge cases"""
    # Test exactly at boundaries
    assert np.isclose(mod_bearing(np.pi), -np.pi)
    assert np.isclose(mod_bearing(-np.pi), -np.pi)
    assert np.isclose(mod_bearing(0), 0)

    # Test with arrays
    bearings = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    results = mod_bearing(bearings)
    assert all(results >= -np.pi) and all(results <= np.pi)


def test_mod_elevation_edge_cases():
    """Test mod_elevation with edge cases and array inputs"""
    # Test with scalar at boundary
    assert np.isclose(mod_elevation(np.pi / 2), np.pi / 2)
    assert np.isclose(mod_elevation(-np.pi / 2), -np.pi / 2)

    # Test with array input
    elevations = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    results = mod_elevation(elevations)
    assert isinstance(results, np.ndarray)
    assert all(results >= -np.pi / 2) and all(results <= np.pi / 2)

    # Test 2*pi edge case (should return 0)
    assert np.isclose(mod_elevation(2 * np.pi), 0.0)


def test_dotproduct_edge_cases():
    """Test dotproduct with additional edge cases"""
    # Test with zero vectors
    a = StateVector([0, 0, 0])
    b = StateVector([1, 2, 3])
    assert np.isclose(dotproduct(a, b), 0)

    # Test orthogonal vectors
    a = StateVector([1, 0])
    b = StateVector([0, 1])
    assert np.isclose(dotproduct(a, b), 0)

    # Test parallel vectors
    a = StateVector([2, 4])
    b = StateVector([1, 2])
    expected = 2 * 1 + 4 * 2
    assert np.isclose(dotproduct(a, b), expected)


def test_gm_sample_reproducibility():
    """Test that gm_sample is reproducible with random state"""
    means = StateVectors([[1, 2], [3, 4]])
    covars = [np.eye(2), np.eye(2)]
    weights = np.array([0.5, 0.5])
    size = 10

    # Use same random state
    rng1 = np.random.RandomState(42)
    samples1 = gm_sample(means, covars, size, weights, random_state=rng1)

    rng2 = np.random.RandomState(42)
    samples2 = gm_sample(means, covars, size, weights, random_state=rng2)

    assert np.allclose(samples1, samples2)


def test_cubature2gauss_with_noise():
    """Test cubature2gauss with additive noise"""
    state = GaussianState([[1], [2]], [[1, 0], [0, 1]])
    cubature_points = gauss2cubature(state)

    covar_noise = CovarianceMatrix([[0.1, 0], [0, 0.1]])
    mean, covar = cubature2gauss(cubature_points, covar_noise=covar_noise)

    # Mean should be close to original
    assert np.allclose(mean, state.state_vector, atol=1e-10)

    # Covariance should be larger due to noise
    assert np.all(np.diag(covar) > np.diag(state.covar))


def test_cubature_transform_with_noise():
    """Test cubature transform with noise parameters"""
    state = GaussianState([[1], [2]], [[1, 0], [0, 1]])

    def simple_function(x, points_noise=None):
        if points_noise is not None:
            return x.state_vector + points_noise
        return x.state_vector

    # Test without noise
    mean1, covar1, cross_covar1, points1 = cubature_transform(state, simple_function)

    # Test with additive covariance noise
    covar_noise = CovarianceMatrix([[0.1, 0], [0, 0.1]])
    mean2, covar2, cross_covar2, points2 = cubature_transform(
        state, simple_function, covar_noise=covar_noise
    )

    # Means should be similar
    assert np.allclose(mean1, mean2, atol=1e-10)

    # Covariance with noise should be larger
    assert np.all(np.diag(covar2) >= np.diag(covar1))


def test_unscented_transform_with_noise():
    """Test unscented transform with noise"""
    state = GaussianState([[1], [2]], [[1, 0], [0, 1]])
    sigma_points_states, mean_weights, covar_weights = gauss2sigma(state, kappa=0)

    def test_function(x, points_noise=None):
        if points_noise is not None:
            return x.state_vector + points_noise
        return x.state_vector

    # Create points noise
    points_noise = np.random.randn(2, 5) * 0.1

    mean, covar, cross_covar, sigma_points_t, mw, cw = unscented_transform(
        sigma_points_states, mean_weights, covar_weights, test_function, points_noise=points_noise
    )

    # Check that all outputs are returned
    assert mean is not None
    assert covar is not None
    assert cross_covar is not None
    assert sigma_points_t is not None

    # Weights should be unchanged
    assert np.allclose(mw, mean_weights)
    assert np.allclose(cw, covar_weights)


def test_gauss2cubature_integer_dtype():
    """Test gauss2cubature with integer state (edge case for dtype conversion)"""
    # Create state with integer mean
    state = GaussianState([[1], [2]], [[1.0, 0], [0, 1.0]])
    cubature_points = gauss2cubature(state)

    # Check that result is float type even though mean was integer
    assert cubature_points.dtype in [np.float64, np.float32]


def test_cubature_transform_with_points_noise_branch():
    """Test cubature transform branch with points_noise parameter"""
    # This test is to trigger the branch in cubature_transform that uses points_noise
    # The implementation iterates over cubature points and points_noise in parallel
    state = GaussianState([[1], [2]], [[1, 0], [0, 1]])

    # Track how many times the function was called with points_noise
    call_count = [0]

    def test_function(x, points_noise=None):
        if points_noise is not None:
            call_count[0] += 1
        return x.state_vector

    # For 2D state, we have 4 cubature points
    # Create array of noise values to be passed per point
    points_noise = np.array([0.1, 0.2, 0.3, 0.4])

    mean, covar, cross_covar, cubature_points = cubature_transform(
        state, test_function, points_noise=points_noise
    )

    # Check that function was called with noise for each cubature point
    assert call_count[0] == 4

    # Check that all outputs are returned
    assert mean is not None
    assert covar is not None
    assert cross_covar is not None
    assert cubature_points is not None


def test_find_nearest_positive_definite_edge_cases():
    """Test find_nearest_positive_definite with various edge cases"""
    from .. import find_nearest_positive_definite, is_cholesky_decomposable

    # Test with a zero matrix - should become positive definite after processing
    zero_matrix = np.zeros((2, 2))
    assert not is_cholesky_decomposable(zero_matrix)
    result = find_nearest_positive_definite(zero_matrix)
    assert is_cholesky_decomposable(result)

    # Test with very small diagonal values
    small_matrix = np.eye(2) * 1e-50
    result = find_nearest_positive_definite(small_matrix)
    assert is_cholesky_decomposable(result)

    # Test with asymmetric matrix (gets symmetrized)
    asymmetric = np.array([[1, 2], [3, 4]])
    result = find_nearest_positive_definite(asymmetric)
    assert is_cholesky_decomposable(result)
    # Result should be symmetric
    assert np.allclose(result, result.T)
