//! PyO3-based Python bindings for Stone Soup
//!
//! This module provides high-performance Python bindings to Stone Soup
//! using PyO3 and maturin.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// A Python module implemented in Rust using PyO3
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StateVector>()?;
    m.add_class::<GaussianState>()?;
    m.add_class::<Detection>()?;
    m.add_class::<Track>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;

    Ok(())
}

/// State vector representation in Python
#[pyclass]
struct StateVector {
    data: Vec<f64>,
}

#[pymethods]
impl StateVector {
    #[new]
    fn new(data: Vec<f64>) -> Self {
        StateVector { data }
    }

    fn __repr__(&self) -> String {
        format!("StateVector(dims={})", self.data.len())
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }
}

/// Gaussian state representation
#[pyclass]
struct GaussianState {
    state_vector: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

#[pymethods]
impl GaussianState {
    #[new]
    fn new(state_vector: Vec<f64>, covariance: Vec<Vec<f64>>) -> PyResult<Self> {
        // Validate dimensions
        let n = state_vector.len();
        if covariance.len() != n || covariance.iter().any(|row| row.len() != n) {
            return Err(PyRuntimeError::new_err(
                "Covariance matrix dimensions must match state vector"
            ));
        }

        Ok(GaussianState {
            state_vector,
            covariance,
        })
    }

    fn __repr__(&self) -> String {
        format!("GaussianState(dims={})", self.state_vector.len())
    }

    #[getter]
    fn state_vector(&self) -> Vec<f64> {
        self.state_vector.clone()
    }

    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        self.covariance.clone()
    }
}

/// Detection representation
#[pyclass]
struct Detection {
    measurement: Vec<f64>,
    timestamp: f64,
}

#[pymethods]
impl Detection {
    #[new]
    fn new(measurement: Vec<f64>, timestamp: f64) -> Self {
        Detection {
            measurement,
            timestamp,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Detection(dims={}, timestamp={})",
            self.measurement.len(),
            self.timestamp
        )
    }

    #[getter]
    fn measurement(&self) -> Vec<f64> {
        self.measurement.clone()
    }

    #[getter]
    fn timestamp(&self) -> f64 {
        self.timestamp
    }
}

/// Track representation
#[pyclass]
struct Track {
    states: Vec<GaussianState>,
    id: String,
}

#[pymethods]
impl Track {
    #[new]
    fn new(id: String) -> Self {
        Track {
            states: Vec::new(),
            id,
        }
    }

    fn __repr__(&self) -> String {
        format!("Track(id={}, length={})", self.id, self.states.len())
    }

    fn __len__(&self) -> usize {
        self.states.len()
    }

    #[getter]
    fn id(&self) -> &str {
        &self.id
    }
}

/// Initialize the Stone Soup core library
#[pyfunction]
fn initialize() -> PyResult<()> {
    // Placeholder for initialization logic
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_vector() {
        let sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.data.len(), 3);
    }

    #[test]
    fn test_gaussian_state() {
        let state = vec![1.0, 2.0];
        let covar = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let gs = GaussianState::new(state, covar);
        assert!(gs.is_ok());
    }
}
