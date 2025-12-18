//! Stone Soup Rust Bindings
//!
//! This crate provides Rust bindings to the Stone Soup tracking framework.
//! Stone Soup is a framework for target tracking and state estimation.
//!
//! # Features
//!
//! - Type-safe state vector and covariance matrix representations
//! - Kalman filter prediction and update operations
//! - Particle filter support
//! - FFI bindings to libstonesoup C library
//!
//! # Examples
//!
//! ```rust
//! use stonesoup::{StateVector, CovarianceMatrix, GaussianState};
//! use nalgebra::DVector;
//!
//! // Create a 4D state vector [x, vx, y, vy]
//! let state = StateVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
//!
//! // Create a diagonal covariance matrix
//! let covar = CovarianceMatrix::diagonal(&[1.0, 0.1, 1.0, 0.1]);
//!
//! // Create a Gaussian state
//! let gaussian = GaussianState::new(state, covar);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

use nalgebra::{DMatrix, DVector};
use std::ops::{Add, Mul, Sub};
use thiserror::Error;

/// Error types for Stone Soup operations
#[derive(Error, Debug)]
pub enum Error {
    /// Dimension mismatch between operands
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        got: usize,
    },

    /// Matrix is singular and cannot be inverted
    #[error("Singular matrix: cannot compute inverse")]
    SingularMatrix,

    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// FFI error from C library
    #[error("FFI error: {0}")]
    FfiError(String),
}

/// Result type for Stone Soup operations
pub type Result<T> = std::result::Result<T, Error>;

/// State vector representation
///
/// A state vector represents the estimated state of a target,
/// typically including position and velocity components.
#[derive(Debug, Clone, PartialEq)]
pub struct StateVector {
    data: DVector<f64>,
}

impl StateVector {
    /// Create a new state vector from a nalgebra DVector
    pub fn new(data: DVector<f64>) -> Self {
        Self { data }
    }

    /// Create a state vector from a Vec
    pub fn from_vec(data: Vec<f64>) -> Self {
        Self {
            data: DVector::from_vec(data),
        }
    }

    /// Create a state vector of zeros with given dimension
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: DVector::zeros(dim),
        }
    }

    /// Get the dimension of the state vector
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    /// Get mutable access to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: f64) -> Result<()> {
        if index >= self.dim() {
            return Err(Error::InvalidParameter(format!(
                "Index {} out of bounds for dimension {}",
                index,
                self.dim()
            )));
        }
        self.data[index] = value;
        Ok(())
    }

    /// Compute the Euclidean norm
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Get the underlying nalgebra vector
    pub fn inner(&self) -> &DVector<f64> {
        &self.data
    }
}

impl Add for StateVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
        }
    }
}

impl Sub for StateVector {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            data: self.data - other.data,
        }
    }
}

impl Mul<f64> for StateVector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            data: self.data * scalar,
        }
    }
}

/// Covariance matrix representation
///
/// A covariance matrix represents the uncertainty in a state estimate.
/// It must be symmetric and positive semi-definite.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceMatrix {
    data: DMatrix<f64>,
}

impl CovarianceMatrix {
    /// Create a new covariance matrix from a nalgebra DMatrix
    pub fn new(data: DMatrix<f64>) -> Result<Self> {
        if data.nrows() != data.ncols() {
            return Err(Error::DimensionMismatch {
                expected: data.nrows(),
                got: data.ncols(),
            });
        }
        Ok(Self { data })
    }

    /// Create a covariance matrix from a 2D Vec
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows != cols {
            return Err(Error::DimensionMismatch {
                expected: rows,
                got: cols,
            });
        }
        Ok(Self {
            data: DMatrix::from_vec(rows, cols, data),
        })
    }

    /// Create an identity covariance matrix
    pub fn identity(dim: usize) -> Self {
        Self {
            data: DMatrix::identity(dim, dim),
        }
    }

    /// Create a diagonal covariance matrix
    pub fn diagonal(diag: &[f64]) -> Self {
        let dim = diag.len();
        let mut data = DMatrix::zeros(dim, dim);
        for (i, &val) in diag.iter().enumerate() {
            data[(i, i)] = val;
        }
        Self { data }
    }

    /// Create a zero covariance matrix
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: DMatrix::zeros(dim, dim),
        }
    }

    /// Get the dimension (number of rows/cols)
    pub fn dim(&self) -> usize {
        self.data.nrows()
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.dim() && col < self.dim() {
            Some(self.data[(row, col)])
        } else {
            None
        }
    }

    /// Compute the matrix inverse
    pub fn inverse(&self) -> Result<Self> {
        self.data
            .clone()
            .try_inverse()
            .map(|inv| Self { data: inv })
            .ok_or(Error::SingularMatrix)
    }

    /// Compute the Cholesky decomposition (lower triangular)
    pub fn cholesky(&self) -> Result<DMatrix<f64>> {
        self.data
            .clone()
            .cholesky()
            .map(|chol| chol.l())
            .ok_or(Error::SingularMatrix)
    }

    /// Compute the determinant
    pub fn determinant(&self) -> f64 {
        self.data.determinant()
    }

    /// Compute the trace
    pub fn trace(&self) -> f64 {
        self.data.trace()
    }

    /// Get the underlying nalgebra matrix
    pub fn inner(&self) -> &DMatrix<f64> {
        &self.data
    }

    /// Transpose the matrix (for symmetric matrices, returns self)
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.transpose(),
        }
    }
}

impl Add for CovarianceMatrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
        }
    }
}

impl Sub for CovarianceMatrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            data: self.data - other.data,
        }
    }
}

impl Mul<f64> for CovarianceMatrix {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            data: self.data * scalar,
        }
    }
}

/// Gaussian state representation
///
/// A Gaussian state combines a state vector with a covariance matrix
/// to represent an uncertain state estimate.
#[derive(Debug, Clone)]
pub struct GaussianState {
    /// The state vector (mean)
    pub state_vector: StateVector,
    /// The covariance matrix
    pub covar: CovarianceMatrix,
    /// Optional timestamp
    pub timestamp: Option<f64>,
}

impl GaussianState {
    /// Create a new Gaussian state
    pub fn new(state_vector: StateVector, covar: CovarianceMatrix) -> Result<Self> {
        if state_vector.dim() != covar.dim() {
            return Err(Error::DimensionMismatch {
                expected: state_vector.dim(),
                got: covar.dim(),
            });
        }
        Ok(Self {
            state_vector,
            covar,
            timestamp: None,
        })
    }

    /// Create a new Gaussian state with timestamp
    pub fn with_timestamp(
        state_vector: StateVector,
        covar: CovarianceMatrix,
        timestamp: f64,
    ) -> Result<Self> {
        let mut state = Self::new(state_vector, covar)?;
        state.timestamp = Some(timestamp);
        Ok(state)
    }

    /// Get the state dimension
    pub fn dim(&self) -> usize {
        self.state_vector.dim()
    }
}

/// Kalman filter operations
pub mod kalman {
    use super::*;

    /// Perform Kalman filter prediction
    ///
    /// Computes:
    /// - x_pred = F * x
    /// - P_pred = F * P * F^T + Q
    ///
    /// # Arguments
    /// * `prior` - Prior Gaussian state
    /// * `transition_matrix` - State transition matrix F
    /// * `process_noise` - Process noise covariance Q
    ///
    /// # Returns
    /// Predicted Gaussian state
    pub fn predict(
        prior: &GaussianState,
        transition_matrix: &DMatrix<f64>,
        process_noise: &CovarianceMatrix,
    ) -> Result<GaussianState> {
        let dim = prior.dim();

        // Check dimensions
        if transition_matrix.nrows() != dim || transition_matrix.ncols() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: transition_matrix.nrows(),
            });
        }

        // x_pred = F * x
        let x_pred = StateVector::new(transition_matrix * prior.state_vector.inner());

        // P_pred = F * P * F^T + Q
        let p_pred_data =
            transition_matrix * prior.covar.inner() * transition_matrix.transpose()
                + process_noise.inner();

        let p_pred = CovarianceMatrix::new(p_pred_data)?;

        GaussianState::new(x_pred, p_pred)
    }

    /// Perform Kalman filter update
    ///
    /// Computes:
    /// - y = z - H * x_pred (innovation)
    /// - S = H * P_pred * H^T + R (innovation covariance)
    /// - K = P_pred * H^T * S^-1 (Kalman gain)
    /// - x_post = x_pred + K * y
    /// - P_post = (I - K * H) * P_pred
    ///
    /// # Arguments
    /// * `predicted` - Predicted Gaussian state
    /// * `measurement` - Measurement vector
    /// * `measurement_matrix` - Measurement matrix H
    /// * `measurement_noise` - Measurement noise covariance R
    ///
    /// # Returns
    /// Posterior Gaussian state
    pub fn update(
        predicted: &GaussianState,
        measurement: &StateVector,
        measurement_matrix: &DMatrix<f64>,
        measurement_noise: &CovarianceMatrix,
    ) -> Result<GaussianState> {
        let state_dim = predicted.dim();
        let meas_dim = measurement.dim();

        // Check dimensions
        if measurement_matrix.nrows() != meas_dim || measurement_matrix.ncols() != state_dim {
            return Err(Error::DimensionMismatch {
                expected: meas_dim,
                got: measurement_matrix.nrows(),
            });
        }

        // y = z - H * x_pred (innovation)
        let y = measurement.inner() - measurement_matrix * predicted.state_vector.inner();

        // S = H * P_pred * H^T + R (innovation covariance)
        let s = measurement_matrix * predicted.covar.inner() * measurement_matrix.transpose()
            + measurement_noise.inner();

        // K = P_pred * H^T * S^-1 (Kalman gain)
        let s_inv = s.clone().try_inverse().ok_or(Error::SingularMatrix)?;
        let k = predicted.covar.inner() * measurement_matrix.transpose() * s_inv;

        // x_post = x_pred + K * y
        let x_post = StateVector::new(predicted.state_vector.inner() + &k * y);

        // P_post = (I - K * H) * P_pred
        let identity = DMatrix::identity(state_dim, state_dim);
        let p_post_data = (identity - &k * measurement_matrix) * predicted.covar.inner();
        let p_post = CovarianceMatrix::new(p_post_data)?;

        GaussianState::new(x_post, p_post)
    }

    /// Compute innovation (measurement residual)
    ///
    /// y = z - H * x
    pub fn innovation(
        state: &GaussianState,
        measurement: &StateVector,
        measurement_matrix: &DMatrix<f64>,
    ) -> StateVector {
        StateVector::new(measurement.inner() - measurement_matrix * state.state_vector.inner())
    }

    /// Compute innovation covariance
    ///
    /// S = H * P * H^T + R
    pub fn innovation_covariance(
        state: &GaussianState,
        measurement_matrix: &DMatrix<f64>,
        measurement_noise: &CovarianceMatrix,
    ) -> Result<CovarianceMatrix> {
        let s = measurement_matrix * state.covar.inner() * measurement_matrix.transpose()
            + measurement_noise.inner();
        CovarianceMatrix::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== StateVector Tests ====================

    #[test]
    fn test_state_vector_creation() {
        let sv = StateVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sv.dim(), 4);
        assert_eq!(sv.get(0), Some(1.0));
        assert_eq!(sv.get(3), Some(4.0));
        assert_eq!(sv.get(4), None);
    }

    #[test]
    fn test_state_vector_zeros() {
        let sv = StateVector::zeros(5);
        assert_eq!(sv.dim(), 5);
        for i in 0..5 {
            assert_eq!(sv.get(i), Some(0.0));
        }
    }

    #[test]
    fn test_state_vector_new() {
        let data = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let sv = StateVector::new(data);
        assert_eq!(sv.dim(), 3);
        assert_eq!(sv.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_state_vector_set() {
        let mut sv = StateVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(sv.set(1, 5.0).is_ok());
        assert_eq!(sv.get(1), Some(5.0));

        // Test out of bounds
        assert!(sv.set(10, 1.0).is_err());
    }

    #[test]
    fn test_state_vector_as_mut_slice() {
        let mut sv = StateVector::from_vec(vec![1.0, 2.0, 3.0]);
        let slice = sv.as_mut_slice();
        slice[0] = 10.0;
        assert_eq!(sv.get(0), Some(10.0));
    }

    #[test]
    fn test_state_vector_norm() {
        let sv = StateVector::from_vec(vec![3.0, 4.0]);
        assert_relative_eq!(sv.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_state_vector_inner() {
        let sv = StateVector::from_vec(vec![1.0, 2.0]);
        let inner = sv.inner();
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0], 1.0);
    }

    #[test]
    fn test_state_vector_arithmetic() {
        let a = StateVector::from_vec(vec![1.0, 2.0]);
        let b = StateVector::from_vec(vec![3.0, 4.0]);

        let sum = a.clone() + b.clone();
        assert_eq!(sum.as_slice(), &[4.0, 6.0]);

        let diff = b - a.clone();
        assert_eq!(diff.as_slice(), &[2.0, 2.0]);

        let scaled = a * 2.0;
        assert_eq!(scaled.as_slice(), &[2.0, 4.0]);
    }

    // ==================== CovarianceMatrix Tests ====================

    #[test]
    fn test_covariance_matrix_diagonal() {
        let cov = CovarianceMatrix::diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(cov.dim(), 3);
        assert_eq!(cov.get(0, 0), Some(1.0));
        assert_eq!(cov.get(1, 1), Some(2.0));
        assert_eq!(cov.get(0, 1), Some(0.0));
    }

    #[test]
    fn test_covariance_matrix_zeros() {
        let cov = CovarianceMatrix::zeros(3);
        assert_eq!(cov.dim(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(cov.get(i, j), Some(0.0));
            }
        }
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
    fn test_covariance_matrix_new() {
        let data = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let cov = CovarianceMatrix::new(data).unwrap();
        assert_eq!(cov.dim(), 2);
        assert_eq!(cov.get(0, 1), Some(0.5));
    }

    #[test]
    fn test_covariance_matrix_new_non_square_fails() {
        let data = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(CovarianceMatrix::new(data).is_err());
    }

    #[test]
    fn test_covariance_matrix_from_vec() {
        let cov = CovarianceMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(cov.dim(), 2);
    }

    #[test]
    fn test_covariance_matrix_from_vec_non_square_fails() {
        assert!(CovarianceMatrix::from_vec(2, 3, vec![1.0; 6]).is_err());
    }

    #[test]
    fn test_covariance_matrix_get_out_of_bounds() {
        let cov = CovarianceMatrix::identity(2);
        assert_eq!(cov.get(5, 5), None);
    }

    #[test]
    fn test_covariance_matrix_inverse() {
        let cov = CovarianceMatrix::diagonal(&[2.0, 4.0]);
        let inv = cov.inverse().unwrap();
        assert_relative_eq!(inv.get(0, 0).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(inv.get(1, 1).unwrap(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_covariance_matrix_inverse_singular() {
        let cov = CovarianceMatrix::zeros(2);
        assert!(cov.inverse().is_err());
    }

    #[test]
    fn test_covariance_matrix_cholesky() {
        let cov = CovarianceMatrix::diagonal(&[4.0, 9.0]);
        let chol = cov.cholesky().unwrap();
        assert_relative_eq!(chol[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(chol[(1, 1)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_covariance_matrix_cholesky_fails_non_positive_definite() {
        // Not positive definite (negative diagonal)
        let data = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
        let cov = CovarianceMatrix::new(data).unwrap();
        assert!(cov.cholesky().is_err());
    }

    #[test]
    fn test_covariance_matrix_determinant() {
        let cov = CovarianceMatrix::diagonal(&[2.0, 3.0]);
        assert_relative_eq!(cov.determinant(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_covariance_matrix_trace() {
        let cov = CovarianceMatrix::diagonal(&[1.0, 2.0, 3.0]);
        assert_relative_eq!(cov.trace(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_covariance_matrix_transpose() {
        let data = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let cov = CovarianceMatrix::new(data).unwrap();
        let trans = cov.transpose();
        assert_eq!(trans.get(0, 1), Some(3.0));
        assert_eq!(trans.get(1, 0), Some(2.0));
    }

    #[test]
    fn test_covariance_matrix_inner() {
        let cov = CovarianceMatrix::identity(2);
        let inner = cov.inner();
        assert_eq!(inner.nrows(), 2);
        assert_eq!(inner.ncols(), 2);
    }

    #[test]
    fn test_covariance_matrix_arithmetic() {
        let a = CovarianceMatrix::identity(2);
        let b = CovarianceMatrix::identity(2);

        let sum = a.clone() + b.clone();
        assert_eq!(sum.get(0, 0), Some(2.0));

        let diff = a.clone() - b;
        assert_eq!(diff.get(0, 0), Some(0.0));

        let scaled = a * 3.0;
        assert_eq!(scaled.get(0, 0), Some(3.0));
    }

    // ==================== GaussianState Tests ====================

    #[test]
    fn test_gaussian_state() {
        let sv = StateVector::from_vec(vec![0.0, 1.0]);
        let cov = CovarianceMatrix::identity(2);
        let state = GaussianState::new(sv, cov).unwrap();
        assert_eq!(state.dim(), 2);
        assert!(state.timestamp.is_none());
    }

    #[test]
    fn test_gaussian_state_dimension_mismatch() {
        let sv = StateVector::from_vec(vec![0.0, 1.0, 2.0]);
        let cov = CovarianceMatrix::identity(2);
        assert!(GaussianState::new(sv, cov).is_err());
    }

    #[test]
    fn test_gaussian_state_with_timestamp() {
        let sv = StateVector::from_vec(vec![0.0, 1.0]);
        let cov = CovarianceMatrix::identity(2);
        let state = GaussianState::with_timestamp(sv, cov, 1.5).unwrap();
        assert_eq!(state.timestamp, Some(1.5));
    }

    // ==================== Kalman Filter Tests ====================

    #[test]
    fn test_kalman_predict() {
        // Simple 2D position-velocity state
        let prior = GaussianState::new(
            StateVector::from_vec(vec![0.0, 1.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        // Constant velocity transition: [1, dt; 0, 1]
        let dt = 1.0;
        let f = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);

        let q = CovarianceMatrix::diagonal(&[0.1, 0.1]);

        let predicted = kalman::predict(&prior, &f, &q).unwrap();

        // x_pred should be [0 + 1*dt, 1] = [1, 1]
        assert_relative_eq!(predicted.state_vector.get(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(predicted.state_vector.get(1).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kalman_predict_dimension_mismatch() {
        let prior = GaussianState::new(
            StateVector::from_vec(vec![0.0, 1.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        // Wrong dimension transition matrix
        let f = DMatrix::from_row_slice(3, 3, &[1.0; 9]);
        let q = CovarianceMatrix::diagonal(&[0.1, 0.1]);

        assert!(kalman::predict(&prior, &f, &q).is_err());
    }

    #[test]
    fn test_kalman_update() {
        let predicted = GaussianState::new(
            StateVector::from_vec(vec![1.0, 1.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        // Measure position only
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = CovarianceMatrix::diagonal(&[0.1]);
        let z = StateVector::from_vec(vec![1.1]);

        let posterior = kalman::update(&predicted, &z, &h, &r).unwrap();

        // Position should move towards measurement
        assert!(posterior.state_vector.get(0).unwrap() > 1.0);
        assert!(posterior.state_vector.get(0).unwrap() < 1.1);
    }

    #[test]
    fn test_kalman_update_dimension_mismatch() {
        let predicted = GaussianState::new(
            StateVector::from_vec(vec![1.0, 1.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        // Wrong dimension measurement matrix
        let h = DMatrix::from_row_slice(2, 3, &[1.0; 6]);
        let r = CovarianceMatrix::identity(2);
        let z = StateVector::from_vec(vec![1.0, 1.0]);

        assert!(kalman::update(&predicted, &z, &h, &r).is_err());
    }

    #[test]
    fn test_kalman_innovation() {
        let state = GaussianState::new(
            StateVector::from_vec(vec![1.0, 2.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        let z = StateVector::from_vec(vec![1.5]);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        let innov = kalman::innovation(&state, &z, &h);
        assert_relative_eq!(innov.get(0).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_kalman_innovation_covariance() {
        let state = GaussianState::new(
            StateVector::from_vec(vec![1.0, 2.0]),
            CovarianceMatrix::identity(2),
        )
        .unwrap();

        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = CovarianceMatrix::diagonal(&[0.5]);

        let s = kalman::innovation_covariance(&state, &h, &r).unwrap();
        // S = H * P * H^T + R = 1*1*1 + 0.5 = 1.5
        assert_relative_eq!(s.get(0, 0).unwrap(), 1.5, epsilon = 1e-10);
    }

    // ==================== Error Type Tests ====================

    #[test]
    fn test_error_display() {
        let e1 = Error::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert!(format!("{}", e1).contains("Dimension mismatch"));

        let e2 = Error::SingularMatrix;
        assert!(format!("{}", e2).contains("Singular matrix"));

        let e3 = Error::InvalidParameter("test".to_string());
        assert!(format!("{}", e3).contains("Invalid parameter"));

        let e4 = Error::FfiError("ffi error".to_string());
        assert!(format!("{}", e4).contains("FFI error"));
    }
}
