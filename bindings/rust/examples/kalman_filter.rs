//! Kalman Filter Example
//!
//! This example demonstrates using the Stone Soup Rust bindings
//! to perform Kalman filtering on a simple constant-velocity target.

use nalgebra::DMatrix;
use stonesoup::{kalman, CovarianceMatrix, GaussianState, StateVector};

fn main() {
    println!("Stone Soup Rust Kalman Filter Example");
    println!("=====================================\n");

    // Initial state: [x, vx, y, vy] = [0, 1, 0, 1]
    // Target starts at origin moving diagonally
    let initial_state = StateVector::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

    // Initial covariance: moderate uncertainty in all dimensions
    let initial_covar = CovarianceMatrix::diagonal(&[1.0, 0.5, 1.0, 0.5]);

    let mut state = GaussianState::new(initial_state, initial_covar)
        .expect("Failed to create initial state");

    println!("Initial state: {:?}", state.state_vector.as_slice());

    // Time step
    let dt = 1.0;

    // Transition matrix for constant velocity model
    // [1  dt  0  0 ]    x'  = x + vx*dt
    // [0  1   0  0 ]    vx' = vx
    // [0  0   1  dt]    y'  = y + vy*dt
    // [0  0   0  1 ]    vy' = vy
    #[rustfmt::skip]
    let transition_matrix = DMatrix::from_row_slice(4, 4, &[
        1.0, dt,  0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, dt,
        0.0, 0.0, 0.0, 1.0,
    ]);

    // Process noise (small uncertainty in velocity)
    let process_noise = CovarianceMatrix::diagonal(&[0.01, 0.1, 0.01, 0.1]);

    // Measurement matrix: we can only observe position (x, y)
    #[rustfmt::skip]
    let measurement_matrix = DMatrix::from_row_slice(2, 4, &[
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]);

    // Measurement noise
    let measurement_noise = CovarianceMatrix::diagonal(&[0.5, 0.5]);

    // Simulated measurements (with noise)
    let measurements = [
        vec![1.1, 0.9],   // t=1: expected ~(1,1)
        vec![2.0, 2.1],   // t=2: expected ~(2,2)
        vec![3.05, 2.95], // t=3: expected ~(3,3)
        vec![4.0, 4.1],   // t=4: expected ~(4,4)
        vec![5.1, 4.9],   // t=5: expected ~(5,5)
    ];

    println!("\nRunning Kalman filter...\n");

    for (i, meas) in measurements.iter().enumerate() {
        let t = (i + 1) as f64;

        // Predict
        state = kalman::predict(&state, &transition_matrix, &process_noise)
            .expect("Prediction failed");

        println!(
            "t={}: Predicted position: ({:.2}, {:.2})",
            t,
            state.state_vector.get(0).unwrap_or(0.0),
            state.state_vector.get(2).unwrap_or(0.0)
        );

        // Update with measurement
        let z = StateVector::from_vec(meas.clone());
        state = kalman::update(&state, &z, &measurement_matrix, &measurement_noise)
            .expect("Update failed");

        println!(
            "t={}: Updated position:   ({:.2}, {:.2})",
            t,
            state.state_vector.get(0).unwrap_or(0.0),
            state.state_vector.get(2).unwrap_or(0.0)
        );
        println!(
            "t={}: Estimated velocity: ({:.2}, {:.2})",
            t,
            state.state_vector.get(1).unwrap_or(0.0),
            state.state_vector.get(3).unwrap_or(0.0)
        );
        println!();
    }

    println!("Final state: {:?}", state.state_vector.as_slice());
    println!(
        "Final covariance trace: {:.4}",
        state.covar.trace()
    );
}
