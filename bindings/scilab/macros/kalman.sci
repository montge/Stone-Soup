// kalman.sci - Kalman filter operations for Stone Soup
//
// This file provides high-level Scilab functions for Kalman filtering
// using the Stone Soup tracking framework.

function gs_pred = kalman_predict(gs_prior, F, Q)
    // Kalman filter prediction step
    //
    // Calling Sequence
    //   gs_pred = kalman_predict(gs_prior, F, Q)
    //
    // Parameters
    //   gs_prior : GaussianState, prior state
    //   F : matrix, state transition matrix
    //   Q : matrix, process noise covariance
    //   gs_pred : GaussianState, predicted state
    //
    // Description
    //   Performs the Kalman filter prediction step:
    //     x_pred = F * x
    //     P_pred = F * P * F' + Q
    //
    // Examples
    //   // 2D constant velocity model
    //   dt = 1.0;
    //   F = [1, dt; 0, 1];
    //   Q = 0.1 * eye(2, 2);
    //
    //   gs_prior = GaussianState([0; 1], eye(2, 2));
    //   gs_pred = kalman_predict(gs_prior, F, Q);

    // Validate inputs
    if typeof(gs_prior) ~= "GaussianState" then
        error("First argument must be a GaussianState");
    end

    sv = gs_prior.sv;
    P = gs_prior.covar;
    dim = length(sv);

    // Pure Scilab implementation
    // x_pred = F * x
    // P_pred = F * P * F' + Q
    x_pred = F * sv;
    P_pred = F * P * F' + Q;

    // Create output Gaussian state
    gs_pred = GaussianState(x_pred, P_pred, gs_prior.ts);
endfunction

function gs_post = kalman_update(gs_pred, measurement, H, R)
    // Kalman filter update step
    //
    // Calling Sequence
    //   gs_post = kalman_update(gs_pred, measurement, H, R)
    //
    // Parameters
    //   gs_pred : GaussianState, predicted state
    //   measurement : vector, measurement
    //   H : matrix, measurement matrix
    //   R : matrix, measurement noise covariance
    //   gs_post : GaussianState, posterior state
    //
    // Description
    //   Performs the Kalman filter update step:
    //     y = z - H * x_pred  (innovation)
    //     S = H * P_pred * H' + R  (innovation covariance)
    //     K = P_pred * H' * inv(S)  (Kalman gain)
    //     x_post = x_pred + K * y
    //     P_post = (I - K * H) * P_pred
    //
    // Examples
    //   // Update with position-only measurement
    //   H = [1, 0];
    //   R = 0.5;
    //
    //   measurement = [1.2];
    //   gs_post = kalman_update(gs_pred, measurement, H, R);

    // Validate inputs
    if typeof(gs_pred) ~= "GaussianState" then
        error("First argument must be a GaussianState");
    end

    sv = gs_pred.sv;
    P = gs_pred.covar;
    z = measurement(:);  // Ensure column vector

    // Pure Scilab implementation
    // y = z - H * x_pred  (innovation)
    // S = H * P_pred * H' + R  (innovation covariance)
    // K = P_pred * H' * inv(S)  (Kalman gain)
    // x_post = x_pred + K * y
    // P_post = (I - K * H) * P_pred
    state_dim = length(sv);
    y = z - H * sv;
    S = H * P * H' + R;
    K = P * H' / S;
    x_post = sv + K * y;
    P_post = (eye(state_dim, state_dim) - K * H) * P;

    // Create output Gaussian state
    gs_post = GaussianState(x_post, P_post, gs_pred.ts);
endfunction

function F = constant_velocity_transition(spatial_dims, dt)
    // Create constant velocity transition matrix
    //
    // Calling Sequence
    //   F = constant_velocity_transition(spatial_dims, dt)
    //
    // Parameters
    //   spatial_dims : number of spatial dimensions (1, 2, or 3)
    //   dt : time step
    //   F : transition matrix
    //
    // Description
    //   Creates a constant velocity transition matrix for tracking.
    //   State format is [x, vx, y, vy, z, vz] for 3D.
    //
    // Examples
    //   // 2D constant velocity, 1 second time step
    //   F = constant_velocity_transition(2, 1.0);
    //   // Returns:
    //   // [1, 1, 0, 0]
    //   // [0, 1, 0, 0]
    //   // [0, 0, 1, 1]
    //   // [0, 0, 0, 1]

    state_dim = spatial_dims * 2;
    F = eye(state_dim, state_dim);

    for i = 1:spatial_dims
        row = (i - 1) * 2 + 1;
        F(row, row + 1) = dt;
    end
endfunction

function H = position_measurement_matrix(spatial_dims)
    // Create position-only measurement matrix
    //
    // Calling Sequence
    //   H = position_measurement_matrix(spatial_dims)
    //
    // Parameters
    //   spatial_dims : number of spatial dimensions
    //   H : measurement matrix
    //
    // Description
    //   Creates a measurement matrix that extracts position from state.
    //   For state [x, vx, y, vy], H = [1, 0, 0, 0; 0, 0, 1, 0]
    //
    // Examples
    //   H = position_measurement_matrix(2);

    state_dim = spatial_dims * 2;
    H = zeros(spatial_dims, state_dim);

    for i = 1:spatial_dims
        col = (i - 1) * 2 + 1;
        H(i, col) = 1;
    end
endfunction

function Q = constant_velocity_process_noise(spatial_dims, q)
    // Create process noise for constant velocity model
    //
    // Calling Sequence
    //   Q = constant_velocity_process_noise(spatial_dims, q)
    //
    // Parameters
    //   spatial_dims : number of spatial dimensions
    //   q : process noise intensity
    //   Q : process noise covariance matrix
    //
    // Description
    //   Creates a discrete white noise acceleration process noise matrix.

    state_dim = spatial_dims * 2;
    Q = zeros(state_dim, state_dim);

    // For each spatial dimension, add velocity noise
    for i = 1:spatial_dims
        vel_idx = i * 2;  // Velocity index
        Q(vel_idx, vel_idx) = q;
    end
endfunction
