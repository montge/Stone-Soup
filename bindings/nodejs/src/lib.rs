//! Node.js bindings for Stone Soup using napi-rs
//!
//! This module provides JavaScript/TypeScript bindings to Stone Soup
//! using the napi-rs framework for native Node.js addons.

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

  /// String representation
  #[napi]
  pub fn to_string(&self) -> String {
    format!("StateVector(dims={})", self.data.len())
  }
}

/// Gaussian state with mean and covariance
#[napi]
pub struct GaussianState {
  state_vector: Vec<f64>,
  covariance: Vec<Vec<f64>>,
}

#[napi]
impl GaussianState {
  /// Create a new Gaussian state
  #[napi(constructor)]
  pub fn new(state_vector: Vec<f64>, covariance: Vec<Vec<f64>>) -> Result<Self> {
    let n = state_vector.len();

    // Validate covariance dimensions
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

    Ok(GaussianState {
      state_vector,
      covariance,
    })
  }

  /// Get the state vector
  #[napi(getter)]
  pub fn state_vector(&self) -> Vec<f64> {
    self.state_vector.clone()
  }

  /// Get the covariance matrix
  #[napi(getter)]
  pub fn covariance(&self) -> Vec<Vec<f64>> {
    self.covariance.clone()
  }

  /// Get dimensionality
  #[napi(getter)]
  pub fn dims(&self) -> u32 {
    self.state_vector.len() as u32
  }

  /// String representation
  #[napi]
  pub fn to_string(&self) -> String {
    format!("GaussianState(dims={})", self.state_vector.len())
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
  states: Vec<GaussianState>,
}

#[napi]
impl Track {
  /// Create a new track
  #[napi(constructor)]
  pub fn new(id: String) -> Self {
    Track {
      id,
      states: Vec::new(),
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
    self.states.len() as u32
  }

  /// String representation
  #[napi]
  pub fn to_string(&self) -> String {
    format!("Track(id={}, length={})", self.id, self.states.len())
  }
}

/// Initialize the Stone Soup library
#[napi]
pub fn initialize() -> Result<()> {
  // Placeholder for initialization logic
  Ok(())
}

/// Get version information
#[napi]
pub fn get_version() -> String {
  "0.1.0".to_string()
}
