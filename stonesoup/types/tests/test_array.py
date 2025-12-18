import numpy as np
import pytest

from ..array import CovarianceMatrix, Matrix, PrecisionMatrix, StateVector, StateVectors


def test_statevector():
    with pytest.raises(ValueError):
        StateVector([[0, 1], [1, 2]])

    with pytest.raises(ValueError):
        StateVector([[[0, 1], [1, 2]]])

    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array)

    assert np.array_equal(state_vector, state_vector_array)
    assert np.array_equal(StateVector([1, 2, 3, 4]), state_vector_array)
    assert np.array_equal(StateVector([[1, 2, 3, 4]]), state_vector_array)
    assert np.array_equal(StateVector(state_vector_array), state_vector)


def test_statevectors():
    vec1 = np.array([[1.0], [2.0], [3.0]])
    vec2 = np.array([[2.0], [3.0], [4.0]])

    sv1 = StateVector(vec1)
    sv2 = StateVector(vec2)

    vecs1 = np.concatenate((vec1, vec2), axis=1)
    svs1 = StateVectors([sv1, sv2])
    svs2 = StateVectors(vecs1)
    svs3 = StateVectors([vec1, vec2])  # Creates 3dim array
    assert np.array_equal(svs1, vecs1)
    assert np.array_equal(svs2, vecs1)
    assert svs3.shape != vecs1.shape

    for sv in svs2:
        assert isinstance(sv, StateVector)


def test_statevectors_mean():
    svs = StateVectors([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mean = StateVector([[2.0, 5.0]])

    assert np.allclose(np.average(svs, axis=1), mean)
    assert np.allclose(np.mean(svs, axis=1, keepdims=True), mean)
    assert np.allclose(np.mean(svs.astype(object), axis=1, keepdims=True), mean)

    assert np.allclose(np.mean(svs, axis=1, where=np.array([True, True, False])), [[1.5, 4.5]])
    assert np.allclose(
        np.mean(svs, axis=1, dtype=int, where=np.array([True, True, False])), [[1, 4]]
    )

    with pytest.raises(TypeError):
        np.average(svs, axis=1, keepdims=False)


def test_standard_statevector_indexing():
    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array)

    # test standard indexing
    assert state_vector[2, 0] == 3
    assert not isinstance(state_vector[2, 0], StateVector)

    # test Slicing
    assert state_vector[1:2, 0] == 2
    assert isinstance(state_vector[1:2, 0], Matrix)  # (n,)
    assert isinstance(state_vector[1:2, :], StateVector)  # (n, 1)
    assert np.array_equal(state_vector[:], state_vector)
    assert isinstance(state_vector[:, 0], Matrix)  # (n,)
    assert isinstance(state_vector[:, :], StateVector)  # (n, 1)
    assert np.array_equal(state_vector[0:], state_vector)
    assert isinstance(state_vector[0:, 0], Matrix)  # (n,)
    assert isinstance(state_vector[0:, :], StateVector)  # (n, 1)

    # test list indices
    assert np.array_equal(state_vector[[1, 3]], StateVector([2, 4]))
    assert isinstance(state_vector[[1, 3], 0], Matrix)

    # test int indexing
    assert state_vector[2] == 3
    assert not isinstance(state_vector[2], StateVector)

    # test behaviour of ravel and flatten functions
    state_vector_ravel = state_vector.ravel()
    state_vector_flatten = state_vector.flatten()
    assert isinstance(state_vector_ravel, Matrix)
    assert isinstance(state_vector_flatten, Matrix)
    assert state_vector_flatten[0] == 1
    assert state_vector_ravel[0] == 1


def test_setting():
    state_vector_array = np.array([[1], [2], [3], [4]])
    state_vector = StateVector(state_vector_array.copy())

    state_vector[2, 0] = 4
    assert np.array_equal(state_vector, StateVector([1, 2, 4, 4]))

    state_vector[2] = 5
    assert np.array_equal(state_vector, StateVector([1, 2, 5, 4]))

    state_vector[:] = state_vector_array[:]
    assert np.array_equal(state_vector, StateVector([1, 2, 3, 4]))

    state_vector[1:3] = StateVector([5, 6])
    assert np.array_equal(state_vector, StateVector([1, 5, 6, 4]))


def test_covariancematrix():
    """CovarianceMatrix Type test"""

    with pytest.raises(ValueError):
        CovarianceMatrix(np.array([0]))

    covar_nparray = (
        np.array(
            [
                [2.2128, 0, 0, 0],
                [0.0002, 2.2130, 0, 0],
                [0.3897, -0.00004, 0.0128, 0],
                [0, 0.3897, 0.0013, 0.0135],
            ]
        )
        * 1e3
    )

    covar_matrix = CovarianceMatrix(covar_nparray)
    assert np.array_equal(covar_matrix, covar_nparray)


def test_precisionmatrix():
    """CovarianceMatrix Type test"""

    with pytest.raises(ValueError):
        PrecisionMatrix(np.array([0]))

    prec_nparray = np.array([[7, 1, 0.5, 0], [1, 4, 2, 0.4], [0.5, 2, 6, 0.3], [0, 0.4, 0.3, 5]])

    prec_matrix = PrecisionMatrix(prec_nparray)
    assert np.array_equal(prec_matrix, prec_nparray)


def test_matrix():
    """Matrix Type test"""

    covar_nparray = (
        np.array(
            [
                [2.2128, 0, 0, 0],
                [0.0002, 2.2130, 0, 0],
                [0.3897, -0.00004, 0.0128, 0],
                [0, 0.3897, 0.0013, 0.0135],
            ]
        )
        * 1e3
    )

    matrix = Matrix(covar_nparray)
    assert np.array_equal(matrix, covar_nparray)


def test_multiplication():
    vector = np.array([[1, 1, 1, 1]]).T
    state_vector = StateVector(vector)
    array = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    covar = CovarianceMatrix(array)
    Mtype = Matrix
    Vtype = StateVector

    assert np.array_equal(covar @ state_vector, array @ vector)
    assert np.array_equal(covar @ vector, array @ vector)
    assert np.array_equal(array @ state_vector, array @ vector)
    assert np.array_equal(state_vector.T @ covar.T, vector.T @ array.T)
    assert np.array_equal(vector.T @ covar.T, vector.T @ array.T)
    assert np.array_equal(state_vector.T @ array.T, vector.T @ array.T)

    assert type(array @ state_vector) == Vtype  # noqa: E721
    assert type(state_vector.T @ array.T) == Mtype  # noqa: E721
    assert type(covar @ vector) == Vtype  # noqa: E721
    assert type(vector.T @ covar.T) == Mtype  # noqa: E721


def test_array_ops():
    vector = np.array([[1, 1, 1, 1]]).T
    vector2 = vector + 2.0
    sv = StateVector(vector)
    array = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]).T
    covar = CovarianceMatrix(array)
    Mtype = Matrix
    Vtype = type(sv)

    assert np.array_equal(covar - vector, array - vector)
    assert type(covar - vector) == Mtype  # noqa: E721
    assert np.array_equal(covar + vector, array + vector)
    assert type(covar + vector) == Mtype  # noqa: E721
    assert np.array_equal(vector - covar, vector - array)
    assert type(vector - covar) == Mtype  # noqa: E721
    assert np.array_equal(vector + covar, vector + array)
    assert type(vector + covar) == Mtype  # noqa: E721

    assert np.array_equal(vector2 - sv, vector2 - vector)
    assert type(vector2 - sv) == Vtype  # noqa: E721
    assert np.array_equal(sv - vector2, vector - vector2)
    assert type(sv - vector2) == Vtype  # noqa: E721
    assert np.array_equal(vector2 + sv, vector2 + vector)
    assert type(vector2 + sv) == Vtype  # noqa: E721
    assert np.array_equal(sv + vector2, vector + vector2)
    assert type(sv + vector2) == Vtype  # noqa: E721
    assert type(sv + 2.0) == Vtype  # noqa: E721
    assert type(sv * 2.0) == Vtype  # noqa: E721

    assert np.array_equal(array - sv, array - vector)
    assert type(array - sv) == Mtype  # noqa: E721
    assert np.array_equal(sv - array, vector - array)
    assert type(sv - array) == Mtype  # noqa: E721
    assert np.array_equal(array + sv, array + vector)
    assert type(array + sv) == Mtype  # noqa: E721
    assert np.array_equal(sv + array, vector + array)
    assert type(sv + array) == Mtype  # noqa: E721
    assert type(covar + 2.0) == Mtype  # noqa: E721
    assert type(covar * 2.0) == Mtype  # noqa: E721


# ============================================================================
# GPU/CPU Interoperability Tests
# ============================================================================


def test_to_numpy_method():
    """Test to_numpy() method exists and works."""
    sv = StateVector([1, 2, 3])
    arr = sv.to_numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, sv)


def test_to_gpu_method_no_cupy():
    """Test to_gpu() raises ImportError when CuPy not available."""
    from stonesoup.backend import is_gpu_available

    sv = StateVector([1, 2, 3])
    if not is_gpu_available():
        with pytest.raises(ImportError):
            sv.to_gpu()


@pytest.fixture
def require_gpu():
    """Skip test if GPU is not available."""
    from stonesoup.backend import is_gpu_available

    if not is_gpu_available():
        pytest.skip("GPU not available")


def test_statevector_from_cupy(require_gpu):
    """Test creating StateVector from CuPy array."""
    import cupy as cp

    cupy_arr = cp.array([1.0, 2.0, 3.0])
    sv = StateVector(cupy_arr)

    assert isinstance(sv, StateVector)
    assert isinstance(sv, np.ndarray)
    np.testing.assert_array_equal(sv.flatten(), [1.0, 2.0, 3.0])


def test_covariancematrix_from_cupy(require_gpu):
    """Test creating CovarianceMatrix from CuPy array."""
    import cupy as cp

    cupy_arr = cp.eye(3)
    cov = CovarianceMatrix(cupy_arr)

    assert isinstance(cov, CovarianceMatrix)
    assert isinstance(cov, np.ndarray)
    np.testing.assert_array_equal(cov, np.eye(3))


def test_matrix_from_cupy(require_gpu):
    """Test creating Matrix from CuPy array."""
    import cupy as cp

    cupy_arr = cp.array([[1.0, 2.0], [3.0, 4.0]])
    mat = Matrix(cupy_arr)

    assert isinstance(mat, Matrix)
    assert isinstance(mat, np.ndarray)
    np.testing.assert_array_equal(mat, [[1.0, 2.0], [3.0, 4.0]])


def test_to_gpu_and_back(require_gpu):
    """Test roundtrip conversion to GPU and back."""
    sv = StateVector([1.0, 2.0, 3.0])
    gpu_arr = sv.to_gpu()
    cpu_arr = sv.to_numpy()

    np.testing.assert_array_equal(cpu_arr.flatten(), [1.0, 2.0, 3.0])

    # Create new StateVector from GPU array
    sv2 = StateVector(gpu_arr)
    np.testing.assert_array_equal(sv.flatten(), sv2.flatten())
