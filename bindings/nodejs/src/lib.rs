//! Node.js bindings for Stone Soup using napi-rs
//!
//! This module provides JavaScript/TypeScript bindings to Stone Soup
//! using the napi-rs framework for native Node.js addons.
//!
//! # Example
//!
//! ```javascript
//! const { StateVector, GaussianState, kalmanPredict, kalmanUpdate } = require('@stonesoup/core');
//!
//! // Create initial state
//! const state = new StateVector([0.0, 1.0, 0.0, 1.0]);
//! const covar = [[1,0,0,0], [0,0.1,0,0], [0,0,1,0], [0,0,0,0.1]];
//! const prior = new GaussianState(state.toArray(), covar);
//!
//! // Predict
//! const F = [[1,1,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]];
//! const Q = [[0.01,0,0,0], [0,0.1,0,0], [0,0,0.01,0], [0,0,0,0.1]];
//! const predicted = kalmanPredict(prior, F, Q);
//! ```

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// State vector representation
#[napi]
pub struct StateVector {
    data: Vec<f64>,
}

#[napi]
impl StateVector {
    /// Create a new state vector
    #[napi(constructor)]
    pub fn new(data: Vec<f64>) -> Self {
        StateVector { data }
    }

    /// Create a zero state vector of given dimension
    #[napi(factory)]
    pub fn zeros(dim: u32) -> Self {
        StateVector {
            data: vec![0.0; dim as usize],
        }
    }

    /// Get the dimensionality of the state vector
    #[napi(getter)]
    pub fn dims(&self) -> u32 {
        self.data.len() as u32
    }

    /// Get the value at a specific index
    #[napi]
    pub fn get(&self, index: u32) -> Option<f64> {
        self.data.get(index as usize).copied()
    }

    /// Set the value at a specific index
    #[napi]
    pub fn set(&mut self, index: u32, value: f64) -> Result<()> {
        if let Some(elem) = self.data.get_mut(index as usize) {
            *elem = value;
            Ok(())
        } else {
            Err(Error::new(
                Status::InvalidArg,
                format!("Index {} out of bounds", index),
            ))
        }
    }

    /// Convert to JavaScript array
    #[napi]
    pub fn to_array(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Compute Euclidean norm
    #[napi]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Add two state vectors
    #[napi]
    pub fn add(&self, other: &StateVector) -> Result<StateVector> {
        if self.data.len() != other.data.len() {
            return Err(Error::new(Status::InvalidArg, "Dimension mismatch"));
        }
        Ok(StateVector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        })
    }

    /// Subtract two state vectors
    #[napi]
    pub fn sub(&self, other: &StateVector) -> Result<StateVector> {
        if self.data.len() != other.data.len() {
            return Err(Error::new(Status::InvalidArg, "Dimension mismatch"));
        }
        Ok(StateVector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        })
    }

    /// Scale by a factor
    #[napi]
    pub fn scale(&self, factor: f64) -> StateVector {
        StateVector {
            data: self.data.iter().map(|x| x * factor).collect(),
        }
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!("StateVector(dims={}, data={:?})", self.data.len(), self.data)
    }
}

/// Covariance matrix representation
#[napi]
pub struct CovarianceMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>, // row-major
}

#[napi]
impl CovarianceMatrix {
    /// Create from 2D array
    #[napi(constructor)]
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::new(Status::InvalidArg, "Empty matrix"));
        }
        let rows = data.len();
        let cols = data[0].len();
        if rows != cols {
            return Err(Error::new(
                Status::InvalidArg,
                "Covariance matrix must be square",
            ));
        }
        for row in &data {
            if row.len() != cols {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Inconsistent row lengths",
                ));
            }
        }

        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Ok(CovarianceMatrix {
            rows,
            cols,
            data: flat,
        })
    }

    /// Create an identity matrix
    #[napi(factory)]
    pub fn identity(dim: u32) -> Self {
        let dim = dim as usize;
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        CovarianceMatrix {
            rows: dim,
            cols: dim,
            data,
        }
    }

    /// Create a diagonal matrix
    #[napi(factory)]
    pub fn diagonal(diag: Vec<f64>) -> Self {
        let dim = diag.len();
        let mut data = vec![0.0; dim * dim];
        for (i, val) in diag.into_iter().enumerate() {
            data[i * dim + i] = val;
        }
        CovarianceMatrix {
            rows: dim,
            cols: dim,
            data,
        }
    }

    /// Get dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.rows as u32
    }

    /// Get element at (row, col)
    #[napi]
    pub fn get(&self, row: u32, col: u32) -> Option<f64> {
        let row = row as usize;
        let col = col as usize;
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set element at (row, col)
    #[napi]
    pub fn set(&mut self, row: u32, col: u32, value: f64) -> Result<()> {
        let row = row as usize;
        let col = col as usize;
        if row >= self.rows || col >= self.cols {
            return Err(Error::new(Status::InvalidArg, "Index out of bounds"));
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    /// Convert to 2D array
    #[napi]
    pub fn to_array(&self) -> Vec<Vec<f64>> {
        self.data
            .chunks(self.cols)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Compute trace
    #[napi]
    pub fn trace(&self) -> f64 {
        let min_dim = self.rows.min(self.cols);
        (0..min_dim).map(|i| self.data[i * self.cols + i]).sum()
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!("CovarianceMatrix({}x{})", self.rows, self.cols)
    }
}

/// Gaussian state with mean and covariance
#[napi]
pub struct GaussianState {
    state_vector: Vec<f64>,
    covariance: Vec<f64>, // row-major
    cov_dim: usize,
    timestamp: Option<f64>,
}

#[napi]
impl GaussianState {
    /// Create a new Gaussian state
    #[napi(constructor)]
    pub fn new(state_vector: Vec<f64>, covariance: Vec<Vec<f64>>) -> Result<Self> {
        let n = state_vector.len();

        if covariance.len() != n {
            return Err(Error::new(
                Status::InvalidArg,
                "Covariance matrix dimensions must match state vector",
            ));
        }

        for row in &covariance {
            if row.len() != n {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Covariance matrix must be square",
                ));
            }
        }

        let flat_cov: Vec<f64> = covariance.into_iter().flatten().collect();

        Ok(GaussianState {
            state_vector,
            covariance: flat_cov,
            cov_dim: n,
            timestamp: None,
        })
    }

    /// Create with timestamp
    #[napi(factory)]
    pub fn with_timestamp(
        state_vector: Vec<f64>,
        covariance: Vec<Vec<f64>>,
        timestamp: f64,
    ) -> Result<Self> {
        let mut state = Self::new(state_vector, covariance)?;
        state.timestamp = Some(timestamp);
        Ok(state)
    }

    /// Get the state vector
    #[napi(getter)]
    pub fn state_vector(&self) -> Vec<f64> {
        self.state_vector.clone()
    }

    /// Get the covariance matrix as 2D array
    #[napi(getter)]
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        self.covariance
            .chunks(self.cov_dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Get dimensionality
    #[napi(getter)]
    pub fn dims(&self) -> u32 {
        self.state_vector.len() as u32
    }

    /// Get timestamp
    #[napi(getter)]
    pub fn timestamp(&self) -> Option<f64> {
        self.timestamp
    }

    /// Set timestamp
    #[napi(setter)]
    pub fn set_timestamp(&mut self, timestamp: f64) {
        self.timestamp = Some(timestamp);
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!(
            "GaussianState(dims={}, timestamp={:?})",
            self.state_vector.len(),
            self.timestamp
        )
    }
}

/// Detection from a sensor
#[napi]
pub struct Detection {
    measurement: Vec<f64>,
    timestamp: f64,
}

#[napi]
impl Detection {
    /// Create a new detection
    #[napi(constructor)]
    pub fn new(measurement: Vec<f64>, timestamp: f64) -> Self {
        Detection {
            measurement,
            timestamp,
        }
    }

    /// Get the measurement
    #[napi(getter)]
    pub fn measurement(&self) -> Vec<f64> {
        self.measurement.clone()
    }

    /// Get the timestamp
    #[napi(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!(
            "Detection(dims={}, timestamp={})",
            self.measurement.len(),
            self.timestamp
        )
    }
}

/// Track representing a target over time
#[napi]
pub struct Track {
    id: String,
    state_vectors: Vec<Vec<f64>>,
    covariances: Vec<Vec<f64>>,
    cov_dims: Vec<usize>,
    timestamps: Vec<Option<f64>>,
}

#[napi]
impl Track {
    /// Create a new track
    #[napi(constructor)]
    pub fn new(id: String) -> Self {
        Track {
            id,
            state_vectors: Vec::new(),
            covariances: Vec::new(),
            cov_dims: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Get the track ID
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get the number of states
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.state_vectors.len() as u32
    }

    /// Add a state to the track
    #[napi]
    pub fn add_state(&mut self, state: &GaussianState) {
        self.state_vectors.push(state.state_vector.clone());
        self.covariances.push(state.covariance.clone());
        self.cov_dims.push(state.cov_dim);
        self.timestamps.push(state.timestamp);
    }

    /// String representation
    #[napi]
    pub fn to_string(&self) -> String {
        format!("Track(id={}, length={})", self.id, self.state_vectors.len())
    }
}

/// Perform Kalman filter prediction
///
/// Computes:
/// - x_pred = F * x
/// - P_pred = F * P * F^T + Q
#[napi]
pub fn kalman_predict(
    prior: &GaussianState,
    transition_matrix: Vec<Vec<f64>>,
    process_noise: Vec<Vec<f64>>,
) -> Result<GaussianState> {
    let dim = prior.state_vector.len();

    // Validate dimensions
    if transition_matrix.len() != dim || transition_matrix.iter().any(|r| r.len() != dim) {
        return Err(Error::new(
            Status::InvalidArg,
            "Transition matrix dimensions must match state dimension",
        ));
    }
    if process_noise.len() != dim || process_noise.iter().any(|r| r.len() != dim) {
        return Err(Error::new(
            Status::InvalidArg,
            "Process noise dimensions must match state dimension",
        ));
    }

    // x_pred = F * x
    let x_pred: Vec<f64> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| transition_matrix[i][j] * prior.state_vector[j])
                .sum()
        })
        .collect();

    // P_pred = F * P * F^T + Q
    // First: F * P
    let mut fp = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            fp[i][j] = (0..dim)
                .map(|k| transition_matrix[i][k] * prior.covariance[k * dim + j])
                .sum();
        }
    }

    // Then: (F * P) * F^T + Q
    let mut p_pred = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            let fpft: f64 = (0..dim).map(|k| fp[i][k] * transition_matrix[j][k]).sum();
            p_pred[i][j] = fpft + process_noise[i][j];
        }
    }

    let mut result = GaussianState::new(x_pred, p_pred)?;
    result.timestamp = prior.timestamp;
    Ok(result)
}

/// Perform Kalman filter update
///
/// Computes:
/// - y = z - H * x (innovation)
/// - S = H * P * H^T + R (innovation covariance)
/// - K = P * H^T * S^-1 (Kalman gain)
/// - x_post = x + K * y
/// - P_post = (I - K * H) * P
#[napi]
pub fn kalman_update(
    predicted: &GaussianState,
    measurement: Vec<f64>,
    measurement_matrix: Vec<Vec<f64>>,
    measurement_noise: Vec<Vec<f64>>,
) -> Result<GaussianState> {
    let state_dim = predicted.state_vector.len();
    let meas_dim = measurement.len();

    // Validate dimensions
    if measurement_matrix.len() != meas_dim
        || measurement_matrix.iter().any(|r| r.len() != state_dim)
    {
        return Err(Error::new(
            Status::InvalidArg,
            "Measurement matrix dimensions mismatch",
        ));
    }
    if measurement_noise.len() != meas_dim
        || measurement_noise.iter().any(|r| r.len() != meas_dim)
    {
        return Err(Error::new(
            Status::InvalidArg,
            "Measurement noise dimensions mismatch",
        ));
    }

    // y = z - H * x (innovation)
    let y: Vec<f64> = (0..meas_dim)
        .map(|i| {
            let hx: f64 = (0..state_dim)
                .map(|j| measurement_matrix[i][j] * predicted.state_vector[j])
                .sum();
            measurement[i] - hx
        })
        .collect();

    // S = H * P * H^T + R
    let mut hp = vec![vec![0.0; state_dim]; meas_dim];
    for i in 0..meas_dim {
        for j in 0..state_dim {
            hp[i][j] = (0..state_dim)
                .map(|k| measurement_matrix[i][k] * predicted.covariance[k * state_dim + j])
                .sum();
        }
    }

    let mut s = vec![vec![0.0; meas_dim]; meas_dim];
    for i in 0..meas_dim {
        for j in 0..meas_dim {
            let hpht: f64 = (0..state_dim)
                .map(|k| hp[i][k] * measurement_matrix[j][k])
                .sum();
            s[i][j] = hpht + measurement_noise[i][j];
        }
    }

    // S^-1 (only for 1x1 and 2x2)
    let s_inv = match meas_dim {
        1 => {
            if s[0][0].abs() < 1e-10 {
                return Err(Error::new(
                    Status::GenericFailure,
                    "Singular innovation covariance",
                ));
            }
            vec![vec![1.0 / s[0][0]]]
        }
        2 => {
            let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
            if det.abs() < 1e-10 {
                return Err(Error::new(
                    Status::GenericFailure,
                    "Singular innovation covariance",
                ));
            }
            vec![
                vec![s[1][1] / det, -s[0][1] / det],
                vec![-s[1][0] / det, s[0][0] / det],
            ]
        }
        _ => {
            return Err(Error::new(
                Status::GenericFailure,
                "Kalman update only supports 1D or 2D measurements",
            ))
        }
    };

    // K = P * H^T * S^-1
    let mut pht = vec![vec![0.0; meas_dim]; state_dim];
    for i in 0..state_dim {
        for j in 0..meas_dim {
            pht[i][j] = (0..state_dim)
                .map(|k| predicted.covariance[i * state_dim + k] * measurement_matrix[j][k])
                .sum();
        }
    }

    let mut k = vec![vec![0.0; meas_dim]; state_dim];
    for i in 0..state_dim {
        for j in 0..meas_dim {
            k[i][j] = (0..meas_dim).map(|l| pht[i][l] * s_inv[l][j]).sum();
        }
    }

    // x_post = x + K * y
    let x_post: Vec<f64> = (0..state_dim)
        .map(|i| {
            let ky: f64 = (0..meas_dim).map(|j| k[i][j] * y[j]).sum();
            predicted.state_vector[i] + ky
        })
        .collect();

    // P_post = (I - K * H) * P
    let mut kh = vec![vec![0.0; state_dim]; state_dim];
    for i in 0..state_dim {
        for j in 0..state_dim {
            kh[i][j] = (0..meas_dim).map(|l| k[i][l] * measurement_matrix[l][j]).sum();
        }
    }

    let mut p_post = vec![vec![0.0; state_dim]; state_dim];
    for i in 0..state_dim {
        for j in 0..state_dim {
            let imkh: f64 = (0..state_dim)
                .map(|l| {
                    let i_minus_kh = if i == l { 1.0 } else { 0.0 } - kh[i][l];
                    i_minus_kh * predicted.covariance[l * state_dim + j]
                })
                .sum();
            p_post[i][j] = imkh;
        }
    }

    let mut result = GaussianState::new(x_post, p_post)?;
    result.timestamp = predicted.timestamp;
    Ok(result)
}

/// Create a constant velocity transition matrix
#[napi]
pub fn constant_velocity_transition(ndim: u32, dt: f64) -> Vec<Vec<f64>> {
    let state_dim = (ndim * 2) as usize;
    let mut f = vec![vec![0.0; state_dim]; state_dim];

    // Identity on diagonal
    for i in 0..state_dim {
        f[i][i] = 1.0;
    }

    // dt for position-velocity coupling
    for i in 0..ndim as usize {
        f[i * 2][i * 2 + 1] = dt;
    }

    f
}

/// Create a position-only measurement matrix
#[napi]
pub fn position_measurement(ndim: u32) -> Vec<Vec<f64>> {
    let state_dim = (ndim * 2) as usize;
    let meas_dim = ndim as usize;
    let mut h = vec![vec![0.0; state_dim]; meas_dim];

    for i in 0..meas_dim {
        h[i][i * 2] = 1.0;
    }

    h
}

/// Initialize the Stone Soup library
#[napi]
pub fn initialize() -> Result<()> {
    Ok(())
}

/// Get version information
#[napi]
pub fn get_version() -> String {
    "0.1.0".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // StateVector tests
    #[test]
    fn test_state_vector_new() {
        let sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.dims(), 3);
        assert_eq!(sv.to_array(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_state_vector_zeros() {
        let sv = StateVector::zeros(4);
        assert_eq!(sv.dims(), 4);
        assert_eq!(sv.to_array(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_state_vector_get_set() {
        let mut sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.get(1), Some(2.0));
        assert_eq!(sv.get(5), None);

        sv.set(1, 5.0).unwrap();
        assert_eq!(sv.get(1), Some(5.0));

        assert!(sv.set(10, 1.0).is_err());
    }

    #[test]
    fn test_state_vector_norm() {
        let sv = StateVector::new(vec![3.0, 4.0]);
        assert!((sv.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_vector_add() {
        let sv1 = StateVector::new(vec![1.0, 2.0]);
        let sv2 = StateVector::new(vec![3.0, 4.0]);
        let result = sv1.add(&sv2).unwrap();
        assert_eq!(result.to_array(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_state_vector_add_dimension_mismatch() {
        let sv1 = StateVector::new(vec![1.0, 2.0]);
        let sv2 = StateVector::new(vec![3.0, 4.0, 5.0]);
        assert!(sv1.add(&sv2).is_err());
    }

    #[test]
    fn test_state_vector_sub() {
        let sv1 = StateVector::new(vec![5.0, 7.0]);
        let sv2 = StateVector::new(vec![3.0, 4.0]);
        let result = sv1.sub(&sv2).unwrap();
        assert_eq!(result.to_array(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_state_vector_scale() {
        let sv = StateVector::new(vec![1.0, 2.0, 3.0]);
        let result = sv.scale(2.0);
        assert_eq!(result.to_array(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_state_vector_to_string() {
        let sv = StateVector::new(vec![1.0, 2.0]);
        assert!(sv.to_string().contains("dims=2"));
    }

    // CovarianceMatrix tests
    #[test]
    fn test_covariance_matrix_new() {
        let cov = CovarianceMatrix::new(vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ]).unwrap();
        assert_eq!(cov.dim(), 2);
    }

    #[test]
    fn test_covariance_matrix_empty_error() {
        assert!(CovarianceMatrix::new(vec![]).is_err());
    }

    #[test]
    fn test_covariance_matrix_non_square_error() {
        assert!(CovarianceMatrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]).is_err());
    }

    #[test]
    fn test_covariance_matrix_inconsistent_rows_error() {
        assert!(CovarianceMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0],
        ]).is_err());
    }

    #[test]
    fn test_covariance_matrix_identity() {
        let cov = CovarianceMatrix::identity(3);
        assert_eq!(cov.dim(), 3);
        assert_eq!(cov.get(0, 0), Some(1.0));
        assert_eq!(cov.get(1, 1), Some(1.0));
        assert_eq!(cov.get(0, 1), Some(0.0));
    }

    #[test]
    fn test_covariance_matrix_diagonal() {
        let cov = CovarianceMatrix::diagonal(vec![1.0, 2.0, 3.0]);
        assert_eq!(cov.dim(), 3);
        assert_eq!(cov.get(0, 0), Some(1.0));
        assert_eq!(cov.get(1, 1), Some(2.0));
        assert_eq!(cov.get(2, 2), Some(3.0));
        assert_eq!(cov.get(0, 1), Some(0.0));
    }

    #[test]
    fn test_covariance_matrix_get_set() {
        let mut cov = CovarianceMatrix::identity(2);
        assert_eq!(cov.get(0, 1), Some(0.0));
        cov.set(0, 1, 0.5).unwrap();
        assert_eq!(cov.get(0, 1), Some(0.5));
        assert_eq!(cov.get(5, 5), None);
        assert!(cov.set(5, 5, 1.0).is_err());
    }

    #[test]
    fn test_covariance_matrix_to_array() {
        let cov = CovarianceMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        assert_eq!(cov.to_array(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_covariance_matrix_trace() {
        let cov = CovarianceMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        assert!((cov.trace() - 5.0).abs() < 1e-10);
    }

    // GaussianState tests
    #[test]
    fn test_gaussian_state_new() {
        let gs = GaussianState::new(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ).unwrap();
        assert_eq!(gs.dims(), 2);
        assert_eq!(gs.state_vector(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_gaussian_state_dimension_mismatch() {
        assert!(GaussianState::new(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        ).is_err());
    }

    #[test]
    fn test_gaussian_state_non_square_covariance() {
        assert!(GaussianState::new(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        ).is_err());
    }

    #[test]
    fn test_gaussian_state_with_timestamp() {
        let gs = GaussianState::with_timestamp(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            100.0,
        ).unwrap();
        assert_eq!(gs.timestamp(), Some(100.0));
    }

    #[test]
    fn test_gaussian_state_set_timestamp() {
        let mut gs = GaussianState::new(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ).unwrap();
        assert_eq!(gs.timestamp(), None);
        gs.set_timestamp(50.0);
        assert_eq!(gs.timestamp(), Some(50.0));
    }

    // Detection tests
    #[test]
    fn test_detection_new() {
        let det = Detection::new(vec![1.0, 2.0], 10.0);
        assert_eq!(det.measurement(), vec![1.0, 2.0]);
        assert_eq!(det.timestamp(), 10.0);
    }

    #[test]
    fn test_detection_to_string() {
        let det = Detection::new(vec![1.0, 2.0], 10.0);
        let s = det.to_string();
        assert!(s.contains("dims=2"));
        assert!(s.contains("timestamp=10"));
    }

    // Track tests
    #[test]
    fn test_track_new() {
        let track = Track::new("track-1".to_string());
        assert_eq!(track.id(), "track-1");
        assert_eq!(track.length(), 0);
    }

    #[test]
    fn test_track_add_state() {
        let mut track = Track::new("track-1".to_string());
        let gs = GaussianState::new(
            vec![1.0, 2.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ).unwrap();
        track.add_state(&gs);
        assert_eq!(track.length(), 1);
    }

    // Kalman filter tests
    #[test]
    fn test_kalman_predict() {
        let prior = GaussianState::new(
            vec![0.0, 1.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ).unwrap();

        let f = vec![
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        let q = vec![
            vec![0.1, 0.0],
            vec![0.0, 0.1],
        ];

        let predicted = kalman_predict(&prior, f, q).unwrap();
        assert_eq!(predicted.dims(), 2);
        // x_pred = F * x = [1*0 + 1*1, 0*0 + 1*1] = [1.0, 1.0]
        let state = predicted.state_vector();
        assert!((state[0] - 1.0).abs() < 1e-10);
        assert!((state[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_predict_dimension_error() {
        let prior = GaussianState::new(
            vec![0.0, 1.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ).unwrap();

        let f = vec![vec![1.0]]; // Wrong dimension
        let q = vec![vec![0.1, 0.0], vec![0.0, 0.1]];

        assert!(kalman_predict(&prior, f, q).is_err());
    }

    #[test]
    fn test_kalman_update_1d() {
        let predicted = GaussianState::new(
            vec![0.0, 0.0],
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
        ).unwrap();

        let z = vec![1.0]; // measurement
        let h = vec![vec![1.0, 0.0]]; // measure position only
        let r = vec![vec![1.0]]; // measurement noise

        let updated = kalman_update(&predicted, z, h, r).unwrap();
        assert_eq!(updated.dims(), 2);
        // Should move state toward measurement
        let state = updated.state_vector();
        assert!(state[0] > 0.0); // Moved toward measurement of 1.0
    }

    #[test]
    fn test_kalman_update_2d() {
        let predicted = GaussianState::new(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        ).unwrap();

        let z = vec![1.0, 2.0]; // measurement
        let h = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let r = vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ];

        let updated = kalman_update(&predicted, z, h, r).unwrap();
        assert_eq!(updated.dims(), 4);
    }

    #[test]
    fn test_kalman_update_unsupported_dimension() {
        let predicted = GaussianState::new(
            vec![0.0, 0.0, 0.0],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ).unwrap();

        let z = vec![1.0, 2.0, 3.0]; // 3D measurement not supported
        let h = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let r = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        assert!(kalman_update(&predicted, z, h, r).is_err());
    }

    // Helper function tests
    #[test]
    fn test_constant_velocity_transition() {
        let f = constant_velocity_transition(2, 0.1);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0].len(), 4);
        // Check diagonal
        assert_eq!(f[0][0], 1.0);
        assert_eq!(f[1][1], 1.0);
        // Check dt coupling
        assert_eq!(f[0][1], 0.1);
        assert_eq!(f[2][3], 0.1);
    }

    #[test]
    fn test_position_measurement() {
        let h = position_measurement(2);
        assert_eq!(h.len(), 2); // 2D measurement
        assert_eq!(h[0].len(), 4); // 4D state
        assert_eq!(h[0][0], 1.0);
        assert_eq!(h[0][1], 0.0);
        assert_eq!(h[1][2], 1.0);
        assert_eq!(h[1][3], 0.0);
    }

    #[test]
    fn test_initialize() {
        assert!(initialize().is_ok());
    }

    #[test]
    fn test_get_version() {
        assert_eq!(get_version(), "0.1.0");
    }
}
