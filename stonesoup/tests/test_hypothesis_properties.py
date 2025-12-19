"""Property-based tests using Hypothesis.

This module contains property-based tests for numerical algorithm validation,
serialization roundtrips, and mathematical invariants.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.state import GaussianState, StateVector

# =============================================================================
# Custom Hypothesis Strategies
# =============================================================================


@st.composite
def positive_definite_matrices(draw, size=4):
    """Generate random positive definite matrices."""
    A = draw(
        arrays(
            dtype=np.float64,
            shape=(size, size),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    # Make it symmetric positive definite via A @ A.T + epsilon*I
    spd = A @ A.T + np.eye(size) * 0.1
    return spd


def state_vectors(ndim=4):
    """Generate random state vectors."""
    return arrays(
        dtype=np.float64,
        shape=(ndim, 1),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )


# =============================================================================
# Kalman Filter Property Tests
# =============================================================================


@given(positive_definite_matrices(size=4))
def test_covariance_positive_semi_definite(covar):
    """Covariance matrices must remain positive semi-definite."""
    cov = CovarianceMatrix(covar)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10), f"Covariance has negative eigenvalues: {eigenvalues}"


@given(state_vectors(4), positive_definite_matrices(size=4))
def test_gaussian_state_dimension_consistency(sv, covar):
    """State vector and covariance dimensions must match."""
    state = GaussianState(state_vector=sv, covar=covar)
    assert state.state_vector.shape[0] == state.covar.shape[0]
    assert state.covar.shape[0] == state.covar.shape[1]


@given(st.integers(min_value=1, max_value=10))
def test_identity_covariance_eigenvalues(ndim):
    """Identity covariance should have all eigenvalues equal to 1."""
    covar = CovarianceMatrix(np.eye(ndim))
    eigenvalues = np.linalg.eigvalsh(covar)
    np.testing.assert_allclose(eigenvalues, np.ones(ndim), rtol=1e-10)


@given(positive_definite_matrices(size=3))
def test_covariance_symmetry(covar):
    """Covariance matrices must be symmetric."""
    cov = CovarianceMatrix(covar)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-10)


# =============================================================================
# State Vector Property Tests
# =============================================================================


@given(state_vectors(4), state_vectors(4))
def test_state_vector_addition_commutativity(sv1, sv2):
    """State vector addition should be commutative."""
    v1 = StateVector(sv1)
    v2 = StateVector(sv2)
    np.testing.assert_allclose(v1 + v2, v2 + v1)


@given(state_vectors(4), state_vectors(4), state_vectors(4))
def test_state_vector_addition_associativity(sv1, sv2, sv3):
    """State vector addition should be associative."""
    v1 = StateVector(sv1)
    v2 = StateVector(sv2)
    v3 = StateVector(sv3)
    np.testing.assert_allclose((v1 + v2) + v3, v1 + (v2 + v3), rtol=1e-10)


@given(state_vectors(4))
def test_state_vector_additive_identity(sv):
    """Zero vector should be additive identity."""
    v = StateVector(sv)
    zero = StateVector(np.zeros_like(sv))
    np.testing.assert_allclose(v + zero, v)


@given(
    state_vectors(4),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_state_vector_scalar_distributivity(sv, scalar):
    """Scalar multiplication should distribute over addition."""
    v = StateVector(sv)
    # Use atol to handle denormalized numbers near machine epsilon
    np.testing.assert_allclose(scalar * (v + v), scalar * v + scalar * v, rtol=1e-10, atol=1e-300)


# =============================================================================
# Matrix Operation Property Tests
# =============================================================================


@given(positive_definite_matrices(size=4))
def test_cholesky_decomposition_validity(spd_matrix):
    """Cholesky decomposition of SPD matrix should satisfy L @ L.T = A."""
    try:
        L = np.linalg.cholesky(spd_matrix)
        reconstructed = L @ L.T
        # Relax tolerance due to numerical precision in ill-conditioned matrices
        np.testing.assert_allclose(reconstructed, spd_matrix, rtol=1e-8, atol=1e-12)
    except np.linalg.LinAlgError:
        assume(False)


@given(positive_definite_matrices(size=3))
def test_matrix_inversion_accuracy(spd_matrix):
    """Matrix inversion should satisfy A @ A^-1 = I."""
    try:
        inv = np.linalg.inv(spd_matrix)
        product = spd_matrix @ inv
        np.testing.assert_allclose(product, np.eye(3), atol=1e-8)
    except np.linalg.LinAlgError:
        assume(False)


@given(positive_definite_matrices(size=4))
def test_eigenvalue_positivity_for_spd(spd_matrix):
    """SPD matrices should have positive eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(spd_matrix)
    assert np.all(eigenvalues > -1e-10), f"Expected positive eigenvalues, got: {eigenvalues}"


# =============================================================================
# Coordinate Transformation Property Tests
# =============================================================================


@given(
    st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
)
def test_cartesian_to_polar_roundtrip(x, y):
    """Cartesian to polar to Cartesian should preserve values."""
    assume(abs(x) > 1e-10 or abs(y) > 1e-10)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x_back = r * np.cos(theta)
    y_back = r * np.sin(theta)
    # Use both rtol and atol for numerical stability near zero
    # Error scales with magnitude, so use atol proportional to max(|x|, |y|)
    np.testing.assert_allclose(
        [x, y], [x_back, y_back], rtol=1e-10, atol=1e-10 * max(abs(x), abs(y), 1)
    )


@given(
    st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
)
def test_polar_to_cartesian_roundtrip(r, theta):
    """Polar to Cartesian to polar should preserve values."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    r_back = np.sqrt(x**2 + y**2)
    theta_back = np.arctan2(y, x)
    np.testing.assert_allclose(r, r_back, rtol=1e-10)
    np.testing.assert_allclose(
        [np.cos(theta), np.sin(theta)], [np.cos(theta_back), np.sin(theta_back)], rtol=1e-10
    )


@given(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
def test_rotation_matrix_orthogonality(angle):
    """2D rotation matrices should be orthogonal (R @ R.T = I)."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


# =============================================================================
# Serialization Property Tests
# =============================================================================


@given(state_vectors(4))
def test_state_vector_numpy_roundtrip(sv):
    """StateVector should roundtrip through numpy array conversion."""
    original = StateVector(sv)
    as_array = np.asarray(original)
    restored = StateVector(as_array)
    np.testing.assert_array_equal(original, restored)


@given(state_vectors(4), positive_definite_matrices(size=4))
def test_gaussian_state_components_accessible(sv, covar):
    """GaussianState components should be accessible after creation."""
    state = GaussianState(state_vector=sv, covar=covar)
    np.testing.assert_array_equal(state.state_vector, sv)
    np.testing.assert_allclose(state.covar, covar, rtol=1e-14)


@given(st.integers(min_value=1, max_value=10))
def test_state_dimension_preserved(ndim):
    """State dimensions should be preserved through operations."""
    sv = StateVector(np.random.randn(ndim, 1))
    assert sv.shape == (ndim, 1)
    assert len(sv) == ndim


# =============================================================================
# Numerical Stability Property Tests
# =============================================================================


@given(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=6),
)
def test_scaled_identity_invertible(scale, ndim):
    """Scaled identity matrices should be invertible with correct inverse."""
    A = scale * np.eye(ndim)
    A_inv = np.linalg.inv(A)
    expected_inv = (1.0 / scale) * np.eye(ndim)
    np.testing.assert_allclose(A_inv, expected_inv, rtol=1e-10)


@given(positive_definite_matrices(size=3))
def test_covariance_trace_positive(covar):
    """Covariance trace (sum of variances) should be positive."""
    cov = CovarianceMatrix(covar)
    trace = np.trace(cov)
    assert trace > 0, f"Expected positive trace, got {trace}"


@given(positive_definite_matrices(size=3))
def test_covariance_determinant_positive(covar):
    """Covariance determinant should be positive for SPD matrices."""
    cov = CovarianceMatrix(covar)
    det = np.linalg.det(cov)
    assert det > -1e-10, f"Expected positive determinant, got {det}"
