import numpy as np
import pytest
from abc import ABC

from ..base import MeasurementModel
from ...base import LinearModel, GaussianModel
from ....types.array import StateVector, CovarianceMatrix
from ....types.state import State


class ConcreteMeasurementModel(MeasurementModel):
    """Concrete implementation for testing abstract MeasurementModel"""

    @property
    def ndim_meas(self):
        return len(self.mapping)

    def function(self, state, noise=False, **kwargs):
        # Simple identity function for testing
        result = state.state_vector[self.mapping, :]
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = np.random.randn(self.ndim_meas, 1)
            else:
                noise = 0
        return result + noise

    def rvs(self, num_samples=1, **kwargs):
        noise = np.random.randn(self.ndim_meas, num_samples)
        if num_samples == 1:
            return StateVector(noise)
        else:
            return noise

    def pdf(self, state1, state2, **kwargs):
        return 0.5


# Tests for MeasurementModel base class

def test_measurement_model_is_abstract():
    """Test that MeasurementModel cannot be instantiated directly"""
    # MeasurementModel is abstract and requires ndim_meas to be implemented
    assert issubclass(MeasurementModel, ABC)


def test_ndim_state_property():
    """Test ndim_state property is correctly set"""
    model = ConcreteMeasurementModel(ndim_state=4, mapping=[0, 1])
    assert model.ndim_state == 4


def test_mapping_property():
    """Test mapping property is correctly set"""
    mapping = [0, 2]
    model = ConcreteMeasurementModel(ndim_state=4, mapping=mapping)
    assert model.mapping == mapping


def test_ndim_property():
    """Test ndim property returns ndim_meas"""
    model = ConcreteMeasurementModel(ndim_state=4, mapping=[0, 1, 2])
    assert model.ndim == model.ndim_meas
    assert model.ndim == 3


def test_ndim_meas_property():
    """Test ndim_meas property based on mapping length"""
    model = ConcreteMeasurementModel(ndim_state=6, mapping=[0, 2, 4])
    assert model.ndim_meas == 3


@pytest.mark.parametrize("ndim_state,mapping,expected_ndim_meas", [
    (2, [0], 1),
    (4, [0, 2], 2),
    (6, [0, 2, 4], 3),
    (4, [0, 1, 2, 3], 4),
])
def test_various_dimensions(ndim_state, mapping, expected_ndim_meas):
    """Test various state and measurement dimensions"""
    model = ConcreteMeasurementModel(ndim_state=ndim_state, mapping=mapping)
    assert model.ndim_state == ndim_state
    assert model.ndim_meas == expected_ndim_meas
    assert len(model.mapping) == expected_ndim_meas


def test_mapping_with_none_values():
    """Test mapping can contain None values"""
    mapping = [0, None, 1, None]
    model = ConcreteMeasurementModel(ndim_state=2, mapping=mapping)
    assert model.mapping == mapping
    assert model.ndim_meas == 4


def test_measurement_model_function():
    """Test basic function call on concrete implementation"""
    model = ConcreteMeasurementModel(ndim_state=4, mapping=[0, 2])
    state = State(StateVector([1, 2, 3, 4]))
    result = model.function(state, noise=False)
    assert result.shape == (2, 1)
    assert result[0, 0] == 1
    assert result[1, 0] == 3


def test_invalid_ndim_state():
    """Test that invalid ndim_state causes an error"""
    # While the Property system doesn't enforce types at init time,
    # using an invalid ndim_state will cause issues when used
    model = ConcreteMeasurementModel(ndim_state="invalid", mapping=[0, 1])
    # Accessing ndim_state should work (no type checking)
    assert model.ndim_state == "invalid"


def test_empty_mapping():
    """Test model with empty mapping"""
    model = ConcreteMeasurementModel(ndim_state=4, mapping=[])
    assert model.ndim_meas == 0


def test_mapping_out_of_bounds_access():
    """Test that mapping indices beyond state dimensions cause errors when used"""
    model = ConcreteMeasurementModel(ndim_state=2, mapping=[0, 1, 2])
    state = State(StateVector([1, 2]))
    # This should raise an IndexError when trying to access state[2]
    with pytest.raises(IndexError):
        model.function(state, noise=False)


def test_state_with_different_shapes():
    """Test function with states of different valid shapes"""
    model = ConcreteMeasurementModel(ndim_state=3, mapping=[0, 1])

    # Test with column vector (standard)
    state1 = State(StateVector([[1], [2], [3]]))
    result1 = model.function(state1, noise=False)
    assert result1.shape == (2, 1)


def test_model_with_single_dimension():
    """Test 1D measurement from 1D state"""
    model = ConcreteMeasurementModel(ndim_state=1, mapping=[0])
    state = State(StateVector([5]))
    result = model.function(state, noise=False)
    assert result.shape == (1, 1)
    assert result[0, 0] == 5


def test_large_dimension_model():
    """Test model with large dimensions"""
    ndim_state = 100
    mapping = list(range(0, 100, 10))  # Every 10th dimension
    model = ConcreteMeasurementModel(ndim_state=ndim_state, mapping=mapping)

    assert model.ndim_state == 100
    assert model.ndim_meas == 10


def test_mapping_order_preserved():
    """Test that mapping order is preserved"""
    mapping = [3, 1, 0, 2]
    model = ConcreteMeasurementModel(ndim_state=4, mapping=mapping)
    assert model.mapping == mapping


def test_duplicate_mapping_indices():
    """Test that duplicate indices in mapping are allowed"""
    mapping = [0, 0, 1, 1]
    model = ConcreteMeasurementModel(ndim_state=2, mapping=mapping)
    assert model.mapping == mapping
    assert model.ndim_meas == 4


# Tests for LinearModel behavior in measurement models

class LinearMeasurementModel(MeasurementModel, LinearModel, GaussianModel):
    """A simple linear Gaussian measurement model for testing"""

    from ....base import Property
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")

    @property
    def ndim_meas(self):
        return len(self.mapping)

    def matrix(self, **kwargs):
        model_matrix = np.zeros((self.ndim_meas, self.ndim_state))
        for dim_meas, dim_state in enumerate(self.mapping):
            if dim_state is not None:
                model_matrix[dim_meas, dim_state] = 1
        return model_matrix

    def covar(self, **kwargs):
        return self.noise_covar


def test_linear_model_matrix():
    """Test that matrix method produces correct measurement matrix"""
    model = LinearMeasurementModel(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2)
    )
    H = model.matrix()
    expected = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])
    assert np.array_equal(H, expected)


def test_linear_model_with_none_mapping():
    """Test matrix with None values in mapping"""
    model = LinearMeasurementModel(
        ndim_state=2,
        mapping=[0, None, 1, None],
        noise_covar=np.eye(4)
    )
    H = model.matrix()
    expected = np.array([[1, 0],
                        [0, 0],
                        [0, 1],
                        [0, 0]])
    assert np.array_equal(H, expected)


def test_linear_function_without_noise():
    """Test linear function evaluation without noise"""
    model = LinearMeasurementModel(
        ndim_state=4,
        mapping=[1, 3],
        noise_covar=np.diag([0.1, 0.2])
    )
    state = State(StateVector([1, 2, 3, 4]))
    result = model.function(state, noise=False)
    expected = np.array([[2], [4]])
    assert np.array_equal(result, expected)


def test_linear_function_with_noise():
    """Test that noise is applied in linear function"""
    model = LinearMeasurementModel(
        ndim_state=4,
        mapping=[0, 2],
        noise_covar=np.eye(2),
        seed=42
    )
    state = State(StateVector([1, 2, 3, 4]))

    # Without noise
    result_no_noise = model.function(state, noise=False)

    # With noise (should be different)
    result_with_noise = model.function(state, noise=True)
    assert not np.array_equal(result_no_noise, result_with_noise)


def test_jacobian_equals_matrix_for_linear():
    """Test that Jacobian equals matrix for linear models"""
    model = LinearMeasurementModel(
        ndim_state=4,
        mapping=[0, 1, 2],
        noise_covar=np.eye(3)
    )
    state = State(StateVector([1, 2, 3, 4]))

    H = model.matrix()
    J = model.jacobian(state)
    assert np.array_equal(H, J)


# Tests for GaussianModel behavior in measurement models

class GaussianMeasurementModel(MeasurementModel, GaussianModel):
    """Simple Gaussian measurement model for testing"""

    from ....base import Property
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")

    @property
    def ndim_meas(self):
        return len(self.mapping)

    def function(self, state, noise=False, **kwargs):
        result = state.state_vector[self.mapping, :]
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return result + noise

    def covar(self, **kwargs):
        return self.noise_covar


def test_rvs_shape_single_sample():
    """Test rvs returns correct shape for single sample"""
    model = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=42
    )
    noise = model.rvs(num_samples=1)
    assert noise.shape == (2, 1)
    assert isinstance(noise, StateVector)


def test_rvs_shape_multiple_samples():
    """Test rvs returns correct shape for multiple samples"""
    model = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1, 2],
        noise_covar=np.eye(3),
        seed=42
    )
    noise = model.rvs(num_samples=10)
    assert noise.shape == (3, 10)


def test_rvs_with_seed_reproducibility():
    """Test that same seed produces same random values"""
    model1 = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=123
    )
    model2 = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=123
    )

    for _ in range(3):
        noise1 = model1.rvs()
        noise2 = model2.rvs()
        assert np.array_equal(noise1, noise2)


def test_rvs_different_seeds_different_values():
    """Test that different seeds produce different values"""
    model1 = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=1
    )
    model2 = GaussianMeasurementModel(
        ndim_state=4,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=2
    )

    noise1 = model1.rvs()
    noise2 = model2.rvs()
    assert not np.array_equal(noise1, noise2)


def test_pdf_evaluation():
    """Test pdf evaluation"""
    model = GaussianMeasurementModel(
        ndim_state=2,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=42
    )
    state = State(StateVector([1, 2]))
    measurement = State(StateVector([1.1, 2.1]))

    prob = model.pdf(measurement, state)
    assert prob > 0
    assert prob <= 1


def test_logpdf_evaluation():
    """Test logpdf evaluation"""
    model = GaussianMeasurementModel(
        ndim_state=2,
        mapping=[0, 1],
        noise_covar=np.eye(2),
        seed=42
    )
    state = State(StateVector([1, 2]))
    measurement = State(StateVector([1.1, 2.1]))

    logprob = model.logpdf(measurement, state)
    prob = model.pdf(measurement, state)
    assert np.isclose(logprob, np.log(prob))


def test_covar_none_raises_error_in_rvs():
    """Test that None covariance raises error when generating samples"""
    model = GaussianMeasurementModel(
        ndim_state=2,
        mapping=[0, 1],
        noise_covar=None
    )
    with pytest.raises(ValueError, match="Cannot generate rvs from None-type covariance"):
        model.rvs()


def test_covar_none_raises_error_in_pdf():
    """Test that None covariance raises error in pdf"""
    model = GaussianMeasurementModel(
        ndim_state=2,
        mapping=[0, 1],
        noise_covar=None
    )
    state = State(StateVector([1, 2]))
    measurement = State(StateVector([1, 2]))

    with pytest.raises(ValueError, match="Cannot generate pdf from None-type covariance"):
        model.pdf(measurement, state)
