//! PyO3-based Python bindings for Stone Soup
//!
//! This module provides high-performance Python bindings to Stone Soup
//! using PyO3 and maturin. It offers:
//!
//! - Type-safe StateVector and CovarianceMatrix classes
//! - NumPy array interoperability
//! - Kalman filter predict/update operations
//! - Detection and Track representations

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array1, Array2};

/// A Python module implemented in Rust using PyO3
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StateVector>()?;
    m.add_class::<CovarianceMatrix>()?;
    m.add_class::<GaussianState>()?;
    m.add_class::<Detection>()?;
    m.add_class::<Track>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_update, m)?)?;

    // Add GPU submodule
    let gpu = PyModule::new(m.py(), "gpu")?;
    gpu.add_function(wrap_pyfunction!(gpu_is_available, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(gpu_device_count, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(gpu_device_name, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(gpu_memory_info, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(gpu_matrix_multiply, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(gpu_batch_kalman_predict, &gpu)?)?;
    m.add_submodule(&gpu)?;

    Ok(())
}

/// State vector representation with NumPy integration
///
/// A state vector represents the estimated state of a target,
/// typically including position and velocity components.
#[pyclass]
#[derive(Clone)]
struct StateVector {
    data: Array1<f64>,
}

#[pymethods]
impl StateVector {
    /// Create a new state vector from a list or numpy array
    #[new]
    fn new(data: Vec<f64>) -> Self {
        StateVector {
            data: Array1::from_vec(data),
        }
    }

    /// Create a state vector from a numpy array
    #[staticmethod]
    fn from_numpy(_py: Python<'_>, arr: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let arr = arr.as_array().to_owned();
        Ok(StateVector { data: arr })
    }

    /// Create a zero state vector of given dimension
    #[staticmethod]
    fn zeros(dim: usize) -> Self {
        StateVector {
            data: Array1::zeros(dim),
        }
    }

    fn __repr__(&self) -> String {
        format!("StateVector(dims={}, data={:?})", self.data.len(), self.data.as_slice().unwrap_or(&[]))
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<f64> {
        self.data.get(idx).copied().ok_or_else(|| {
            PyValueError::new_err(format!("Index {} out of bounds", idx))
        })
    }

    fn __setitem__(&mut self, idx: usize, value: f64) -> PyResult<()> {
        if idx >= self.data.len() {
            return Err(PyValueError::new_err(format!("Index {} out of bounds", idx)));
        }
        self.data[idx] = value;
        Ok(())
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<f64> {
        self.data.to_vec()
    }

    /// Convert to numpy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.data.clone().into_pyarray(py)
    }

    /// Get the dimension of the state vector
    #[getter]
    fn dim(&self) -> usize {
        self.data.len()
    }

    /// Compute Euclidean norm
    fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Add two state vectors
    fn __add__(&self, other: &StateVector) -> PyResult<StateVector> {
        if self.data.len() != other.data.len() {
            return Err(PyValueError::new_err("Dimension mismatch"));
        }
        Ok(StateVector {
            data: &self.data + &other.data,
        })
    }

    /// Subtract two state vectors
    fn __sub__(&self, other: &StateVector) -> PyResult<StateVector> {
        if self.data.len() != other.data.len() {
            return Err(PyValueError::new_err("Dimension mismatch"));
        }
        Ok(StateVector {
            data: &self.data - &other.data,
        })
    }

    /// Scalar multiplication
    fn __mul__(&self, scalar: f64) -> StateVector {
        StateVector {
            data: &self.data * scalar,
        }
    }
}

/// Covariance matrix representation with NumPy integration
///
/// A covariance matrix represents the uncertainty in a state estimate.
/// It must be symmetric and positive semi-definite.
#[pyclass]
#[derive(Clone)]
struct CovarianceMatrix {
    data: Array2<f64>,
}

#[pymethods]
impl CovarianceMatrix {
    /// Create from 2D list
    #[new]
    fn new(data: Vec<Vec<f64>>) -> PyResult<Self> {
        let rows = data.len();
        if rows == 0 {
            return Err(PyValueError::new_err("Empty matrix"));
        }
        let cols = data[0].len();
        if rows != cols {
            return Err(PyValueError::new_err("Covariance matrix must be square"));
        }
        if data.iter().any(|row| row.len() != cols) {
            return Err(PyValueError::new_err("Inconsistent row lengths"));
        }

        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

        Ok(CovarianceMatrix { data: arr })
    }

    /// Create from numpy array
    #[staticmethod]
    fn from_numpy(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let arr = arr.as_array();
        if arr.nrows() != arr.ncols() {
            return Err(PyValueError::new_err("Covariance matrix must be square"));
        }
        Ok(CovarianceMatrix {
            data: arr.to_owned(),
        })
    }

    /// Create identity matrix
    #[staticmethod]
    fn identity(dim: usize) -> Self {
        CovarianceMatrix {
            data: Array2::eye(dim),
        }
    }

    /// Create diagonal matrix
    #[staticmethod]
    fn diagonal(diag: Vec<f64>) -> Self {
        let dim = diag.len();
        let mut arr = Array2::zeros((dim, dim));
        for (i, val) in diag.into_iter().enumerate() {
            arr[[i, i]] = val;
        }
        CovarianceMatrix { data: arr }
    }

    /// Create zero matrix
    #[staticmethod]
    fn zeros(dim: usize) -> Self {
        CovarianceMatrix {
            data: Array2::zeros((dim, dim)),
        }
    }

    fn __repr__(&self) -> String {
        format!("CovarianceMatrix(shape={}x{})", self.data.nrows(), self.data.ncols())
    }

    /// Get element at (row, col)
    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        self.data.get(idx).copied().ok_or_else(|| {
            PyValueError::new_err(format!("Index {:?} out of bounds", idx))
        })
    }

    /// Set element at (row, col)
    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        if idx.0 >= self.data.nrows() || idx.1 >= self.data.ncols() {
            return Err(PyValueError::new_err(format!("Index {:?} out of bounds", idx)));
        }
        self.data[idx] = value;
        Ok(())
    }

    /// Convert to numpy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.data.clone().into_pyarray(py)
    }

    /// Convert to 2D list
    fn to_list(&self) -> Vec<Vec<f64>> {
        self.data.rows().into_iter().map(|row| row.to_vec()).collect()
    }

    /// Get dimension
    #[getter]
    fn dim(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of rows
    #[getter]
    fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns
    #[getter]
    fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Compute trace
    fn trace(&self) -> f64 {
        self.data.diag().sum()
    }

    /// Compute determinant (simple implementation for small matrices)
    fn determinant(&self) -> PyResult<f64> {
        let n = self.data.nrows();
        match n {
            1 => Ok(self.data[[0, 0]]),
            2 => Ok(self.data[[0, 0]] * self.data[[1, 1]] - self.data[[0, 1]] * self.data[[1, 0]]),
            _ => Err(PyRuntimeError::new_err("Determinant only implemented for 1x1 and 2x2"))
        }
    }

    /// Add two matrices
    fn __add__(&self, other: &CovarianceMatrix) -> PyResult<CovarianceMatrix> {
        if self.data.dim() != other.data.dim() {
            return Err(PyValueError::new_err("Dimension mismatch"));
        }
        Ok(CovarianceMatrix {
            data: &self.data + &other.data,
        })
    }

    /// Subtract two matrices
    fn __sub__(&self, other: &CovarianceMatrix) -> PyResult<CovarianceMatrix> {
        if self.data.dim() != other.data.dim() {
            return Err(PyValueError::new_err("Dimension mismatch"));
        }
        Ok(CovarianceMatrix {
            data: &self.data - &other.data,
        })
    }

    /// Scalar multiplication
    fn __mul__(&self, scalar: f64) -> CovarianceMatrix {
        CovarianceMatrix {
            data: &self.data * scalar,
        }
    }
}

/// Gaussian state representation combining state vector and covariance
#[pyclass]
#[derive(Clone)]
struct GaussianState {
    #[pyo3(get)]
    state_vector: StateVector,
    #[pyo3(get)]
    covariance: CovarianceMatrix,
    #[pyo3(get, set)]
    timestamp: Option<f64>,
}

#[pymethods]
impl GaussianState {
    #[new]
    #[pyo3(signature = (state_vector, covariance, timestamp=None))]
    fn new(state_vector: StateVector, covariance: CovarianceMatrix, timestamp: Option<f64>) -> PyResult<Self> {
        if state_vector.dim() != covariance.dim() {
            return Err(PyValueError::new_err(
                format!("Dimension mismatch: state={}, covar={}",
                    state_vector.dim(), covariance.dim())
            ));
        }

        Ok(GaussianState {
            state_vector,
            covariance,
            timestamp,
        })
    }

    /// Create from numpy arrays
    #[staticmethod]
    #[pyo3(signature = (state, covar, timestamp=None))]
    fn from_numpy(
        py: Python<'_>,
        state: PyReadonlyArray1<'_, f64>,
        covar: PyReadonlyArray2<'_, f64>,
        timestamp: Option<f64>,
    ) -> PyResult<Self> {
        let sv = StateVector::from_numpy(py, state)?;
        let cm = CovarianceMatrix::from_numpy(covar)?;
        GaussianState::new(sv, cm, timestamp)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianState(dims={}, timestamp={:?})",
            self.state_vector.dim(),
            self.timestamp
        )
    }

    /// Get state dimension
    #[getter]
    fn dim(&self) -> usize {
        self.state_vector.dim()
    }

    /// Get state as numpy array
    fn state_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.state_vector.to_numpy(py)
    }

    /// Get covariance as numpy array
    fn covar_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.covariance.to_numpy(py)
    }
}

/// Detection representation
#[pyclass]
#[derive(Clone)]
struct Detection {
    measurement: StateVector,
    #[pyo3(get, set)]
    timestamp: f64,
}

#[pymethods]
impl Detection {
    #[new]
    fn new(measurement: Vec<f64>, timestamp: f64) -> Self {
        Detection {
            measurement: StateVector::new(measurement),
            timestamp,
        }
    }

    #[staticmethod]
    fn from_numpy(py: Python<'_>, arr: PyReadonlyArray1<'_, f64>, timestamp: f64) -> PyResult<Self> {
        let sv = StateVector::from_numpy(py, arr)?;
        Ok(Detection {
            measurement: sv,
            timestamp,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Detection(dims={}, timestamp={})",
            self.measurement.dim(),
            self.timestamp
        )
    }

    #[getter]
    fn measurement(&self) -> StateVector {
        self.measurement.clone()
    }

    fn measurement_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.measurement.to_numpy(py)
    }
}

/// Track representation as sequence of states
#[pyclass]
struct Track {
    states: Vec<GaussianState>,
    #[pyo3(get)]
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

    /// Append a state to the track
    fn append(&mut self, state: GaussianState) {
        self.states.push(state);
    }

    /// Get state at index
    fn __getitem__(&self, idx: usize) -> PyResult<GaussianState> {
        self.states.get(idx).cloned().ok_or_else(|| {
            PyValueError::new_err(format!("Index {} out of bounds", idx))
        })
    }

    /// Get the latest state
    fn latest(&self) -> PyResult<GaussianState> {
        self.states.last().cloned().ok_or_else(|| {
            PyRuntimeError::new_err("Track is empty")
        })
    }
}

/// Initialize the Stone Soup core library
#[pyfunction]
fn initialize() -> PyResult<()> {
    Ok(())
}

/// Perform Kalman filter prediction
///
/// Computes:
/// - x_pred = F @ x
/// - P_pred = F @ P @ F.T + Q
///
/// Args:
///     prior: Prior Gaussian state
///     transition_matrix: State transition matrix F (as 2D list or numpy array)
///     process_noise: Process noise covariance Q
///
/// Returns:
///     Predicted Gaussian state
#[pyfunction]
fn kalman_predict(
    prior: &GaussianState,
    transition_matrix: Vec<Vec<f64>>,
    process_noise: &CovarianceMatrix,
) -> PyResult<GaussianState> {
    let dim = prior.dim();

    // Convert transition matrix to ndarray
    let f_rows = transition_matrix.len();
    let f_cols = transition_matrix.first().map(|r| r.len()).unwrap_or(0);
    if f_rows != dim || f_cols != dim {
        return Err(PyValueError::new_err(format!(
            "Transition matrix dimensions {}x{} don't match state dimension {}",
            f_rows, f_cols, dim
        )));
    }

    let f_flat: Vec<f64> = transition_matrix.into_iter().flatten().collect();
    let f = Array2::from_shape_vec((dim, dim), f_flat)
        .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

    // x_pred = F @ x
    let x_pred_data: Vec<f64> = (0..dim)
        .map(|i| {
            (0..dim).map(|j| f[[i, j]] * prior.state_vector.data[j]).sum()
        })
        .collect();
    let x_pred = StateVector { data: Array1::from_vec(x_pred_data) };

    // P_pred = F @ P @ F.T + Q
    let p = &prior.covariance.data;
    let q = &process_noise.data;

    // F @ P
    let mut fp = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            fp[[i, j]] = (0..dim).map(|k| f[[i, k]] * p[[k, j]]).sum::<f64>();
        }
    }

    // (F @ P) @ F.T + Q
    let mut p_pred = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            p_pred[[i, j]] = (0..dim).map(|k| fp[[i, k]] * f[[j, k]]).sum::<f64>() + q[[i, j]];
        }
    }

    Ok(GaussianState {
        state_vector: x_pred,
        covariance: CovarianceMatrix { data: p_pred },
        timestamp: prior.timestamp,
    })
}

/// Perform Kalman filter update
///
/// Computes:
/// - y = z - H @ x_pred (innovation)
/// - S = H @ P_pred @ H.T + R (innovation covariance)
/// - K = P_pred @ H.T @ S^-1 (Kalman gain)
/// - x_post = x_pred + K @ y
/// - P_post = (I - K @ H) @ P_pred
///
/// Args:
///     predicted: Predicted Gaussian state
///     measurement: Measurement vector
///     measurement_matrix: Measurement matrix H
///     measurement_noise: Measurement noise covariance R
///
/// Returns:
///     Posterior Gaussian state
#[pyfunction]
fn kalman_update(
    predicted: &GaussianState,
    measurement: &StateVector,
    measurement_matrix: Vec<Vec<f64>>,
    measurement_noise: &CovarianceMatrix,
) -> PyResult<GaussianState> {
    let state_dim = predicted.dim();
    let meas_dim = measurement.dim();

    // Convert measurement matrix
    let h_rows = measurement_matrix.len();
    let h_cols = measurement_matrix.first().map(|r| r.len()).unwrap_or(0);
    if h_rows != meas_dim || h_cols != state_dim {
        return Err(PyValueError::new_err(format!(
            "Measurement matrix dimensions {}x{} don't match state={} meas={}",
            h_rows, h_cols, state_dim, meas_dim
        )));
    }

    let h_flat: Vec<f64> = measurement_matrix.into_iter().flatten().collect();
    let h = Array2::from_shape_vec((meas_dim, state_dim), h_flat)
        .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;

    let x = &predicted.state_vector.data;
    let p = &predicted.covariance.data;
    let z = &measurement.data;
    let r = &measurement_noise.data;

    // y = z - H @ x (innovation)
    let y: Array1<f64> = Array1::from_vec(
        (0..meas_dim)
            .map(|i| z[i] - (0..state_dim).map(|j| h[[i, j]] * x[j]).sum::<f64>())
            .collect()
    );

    // S = H @ P @ H.T + R
    let mut hp = Array2::zeros((meas_dim, state_dim));
    for i in 0..meas_dim {
        for j in 0..state_dim {
            hp[[i, j]] = (0..state_dim).map(|k| h[[i, k]] * p[[k, j]]).sum::<f64>();
        }
    }

    let mut s = Array2::zeros((meas_dim, meas_dim));
    for i in 0..meas_dim {
        for j in 0..meas_dim {
            s[[i, j]] = (0..state_dim).map(|k| hp[[i, k]] * h[[j, k]]).sum::<f64>() + r[[i, j]];
        }
    }

    // S^-1 (simple 1x1 or 2x2 inversion)
    let s_inv = match meas_dim {
        1 => {
            if s[[0, 0]].abs() < 1e-10 {
                return Err(PyRuntimeError::new_err("Singular innovation covariance"));
            }
            Array2::from_elem((1, 1), 1.0 / s[[0, 0]])
        }
        2 => {
            let det = s[[0, 0]] * s[[1, 1]] - s[[0, 1]] * s[[1, 0]];
            if det.abs() < 1e-10 {
                return Err(PyRuntimeError::new_err("Singular innovation covariance"));
            }
            Array2::from_shape_vec((2, 2), vec![
                s[[1, 1]] / det, -s[[0, 1]] / det,
                -s[[1, 0]] / det, s[[0, 0]] / det,
            ]).unwrap()
        }
        _ => return Err(PyRuntimeError::new_err(
            "Kalman update only supports 1D or 2D measurements currently"
        )),
    };

    // K = P @ H.T @ S^-1
    let mut pht = Array2::zeros((state_dim, meas_dim));
    for i in 0..state_dim {
        for j in 0..meas_dim {
            pht[[i, j]] = (0..state_dim).map(|k| p[[i, k]] * h[[j, k]]).sum::<f64>();
        }
    }

    let mut k = Array2::zeros((state_dim, meas_dim));
    for i in 0..state_dim {
        for j in 0..meas_dim {
            k[[i, j]] = (0..meas_dim).map(|l| pht[[i, l]] * s_inv[[l, j]]).sum::<f64>();
        }
    }

    // x_post = x + K @ y
    let x_post = Array1::from_vec(
        (0..state_dim)
            .map(|i| x[i] + (0..meas_dim).map(|j| k[[i, j]] * y[j]).sum::<f64>())
            .collect()
    );

    // P_post = (I - K @ H) @ P
    let mut kh = Array2::zeros((state_dim, state_dim));
    for i in 0..state_dim {
        for j in 0..state_dim {
            kh[[i, j]] = (0..meas_dim).map(|l| k[[i, l]] * h[[l, j]]).sum::<f64>();
        }
    }

    let mut p_post = Array2::zeros((state_dim, state_dim));
    for i in 0..state_dim {
        for j in 0..state_dim {
            let _i_minus_kh = if i == j { 1.0 } else { 0.0 } - kh[[i, j]];
            p_post[[i, j]] = (0..state_dim).map(|l| {
                let imkh_il = if i == l { 1.0 } else { 0.0 } - kh[[i, l]];
                imkh_il * p[[l, j]]
            }).sum::<f64>();
        }
    }

    Ok(GaussianState {
        state_vector: StateVector { data: x_post },
        covariance: CovarianceMatrix { data: p_post },
        timestamp: predicted.timestamp,
    })
}

// ============================================================================
// GPU Functions
// ============================================================================

/// Check if GPU acceleration is available
///
/// Returns True if CUDA GPU support is available in the underlying library.
/// This requires the library to be compiled with CUDA support and for a
/// CUDA-capable GPU to be present.
#[pyfunction]
fn gpu_is_available() -> bool {
    // Try to detect CUDA via environment or runtime
    // For now, this is a pure Rust implementation that doesn't require FFI
    #[cfg(feature = "cuda")]
    {
        // When CUDA feature is enabled, attempt to detect GPU
        // This would link to libstonesoup's CUDA detection
        false // Placeholder - requires FFI to libstonesoup
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of available GPU devices
///
/// Returns:
///     Number of CUDA-capable GPU devices, or 0 if none available
#[pyfunction]
fn gpu_device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        0 // Placeholder - requires FFI to libstonesoup
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Get the name of a GPU device
///
/// Args:
///     device: Device index (0-based)
///
/// Returns:
///     Device name string, or error if device not available
#[pyfunction]
fn gpu_device_name(device: i32) -> PyResult<String> {
    if !gpu_is_available() || device < 0 || device >= gpu_device_count() {
        return Err(PyRuntimeError::new_err(format!(
            "GPU device {} not available", device
        )));
    }
    Ok(format!("CUDA Device {}", device))
}

/// Get GPU memory information for a device
///
/// Args:
///     device: Device index (0-based)
///
/// Returns:
///     Dictionary with 'total' and 'free' memory in bytes
#[pyfunction]
fn gpu_memory_info(py: Python<'_>, device: i32) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    if !gpu_is_available() || device < 0 || device >= gpu_device_count() {
        return Err(PyRuntimeError::new_err(format!(
            "GPU device {} not available", device
        )));
    }

    let dict = PyDict::new(py);
    dict.set_item("total", 0_u64)?;
    dict.set_item("free", 0_u64)?;
    Ok(dict.into())
}

/// GPU-accelerated matrix multiplication
///
/// Computes C = A @ B using CUDA.
///
/// Args:
///     a: First matrix (m x k)
///     b: Second matrix (k x n)
///
/// Returns:
///     Result matrix C (m x n)
#[pyfunction]
fn gpu_matrix_multiply<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !gpu_is_available() {
        return Err(PyRuntimeError::new_err(
            "GPU not available. Install with CUDA support or use numpy for CPU computation."
        ));
    }

    let a_arr = a.as_array();
    let b_arr = b.as_array();

    let m = a_arr.nrows();
    let k = a_arr.ncols();
    let k2 = b_arr.nrows();
    let n = b_arr.ncols();

    if k != k2 {
        return Err(PyValueError::new_err(format!(
            "Matrix dimensions don't match for multiplication: {}x{} @ {}x{}",
            m, k, k2, n
        )));
    }

    // CPU fallback (GPU implementation would use FFI to libstonesoup)
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = (0..k).map(|l| a_arr[[i, l]] * b_arr[[l, j]]).sum::<f64>();
        }
    }

    Ok(c.into_pyarray(py))
}

/// GPU-accelerated batch Kalman predict
///
/// Performs Kalman prediction on multiple states in parallel using GPU.
///
/// Args:
///     states: Batch of state vectors (batch_size x state_dim)
///     covariances: Batch of covariance matrices (batch_size x state_dim x state_dim)
///     transition: Transition matrix F (state_dim x state_dim)
///     process_noise: Process noise Q (state_dim x state_dim)
///
/// Returns:
///     Tuple of (predicted_states, predicted_covariances)
#[pyfunction]
fn gpu_batch_kalman_predict<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    _covariances: &Bound<'py, PyArray2<f64>>,
    transition: PyReadonlyArray2<'py, f64>,
    _process_noise: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, PyObject)> {
    if !gpu_is_available() {
        return Err(PyRuntimeError::new_err(
            "GPU not available. Use stonesoup.backend with CuPy for GPU acceleration."
        ));
    }

    let x = states.as_array();
    let f = transition.as_array();

    let batch_size = x.nrows();
    let state_dim = x.ncols();

    if f.nrows() != state_dim || f.ncols() != state_dim {
        return Err(PyValueError::new_err("Transition matrix dimension mismatch"));
    }

    // CPU fallback (GPU implementation would use FFI to libstonesoup)
    let mut x_pred = Array2::zeros((batch_size, state_dim));

    for b in 0..batch_size {
        for i in 0..state_dim {
            x_pred[[b, i]] = (0..state_dim).map(|j| f[[i, j]] * x[[b, j]]).sum::<f64>();
        }
    }

    // Return predicted states (covariance update would be similar)
    Ok((x_pred.into_pyarray(py), py.None()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_vector() {
        let sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.data.len(), 3);
        assert_eq!(sv.norm(), (14.0_f64).sqrt());
    }

    #[test]
    fn test_covariance_matrix() {
        let cov = CovarianceMatrix::identity(3);
        assert_eq!(cov.trace(), 3.0);
    }

    #[test]
    fn test_gaussian_state() {
        let state = StateVector::new(vec![1.0, 2.0]);
        let covar = CovarianceMatrix::identity(2);
        let gs = GaussianState::new(state, covar, None);
        assert!(gs.is_ok());
    }

    #[test]
    fn test_gpu_not_available() {
        // GPU should not be available without CUDA feature
        #[cfg(not(feature = "cuda"))]
        assert!(!gpu_is_available());
    }
}
