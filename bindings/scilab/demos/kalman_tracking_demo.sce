// Stone Soup Kalman Filter Tracking Demo
//
// This demo shows how to use the Stone Soup Scilab bindings to track
// a target using a Kalman filter with constant velocity motion model.

mode(-1);
lines(0);

// Load Stone Soup module
exec(get_absolute_file_path("kalman_tracking_demo.sce") + "../loader.sce", -1);

mprintf("Stone Soup Kalman Filter Tracking Demo\n");
mprintf("======================================\n\n");

// Simulation parameters
dt = 1.0;           // Time step (seconds)
n_steps = 20;       // Number of time steps
spatial_dims = 2;   // 2D tracking

// Process and measurement noise
process_noise_std = 0.5;    // m/s^2 (acceleration noise)
measurement_noise_std = 10; // m (position measurement noise)

// Create transition matrix (constant velocity model)
F = constant_velocity_transition(spatial_dims, dt);
mprintf("Transition matrix F:\n");
disp(F);

// Create measurement matrix (position only)
H = position_measurement_matrix(spatial_dims);
mprintf("Measurement matrix H:\n");
disp(H);

// Create process noise covariance
// Using discrete white noise acceleration model
Q = zeros(4, 4);
q = process_noise_std^2;
// For each dimension: [dt^2/2; dt] * q * [dt^2/2, dt]
block = [dt^4/4, dt^3/2; dt^3/2, dt^2] * q;
Q(1:2, 1:2) = block;
Q(3:4, 3:4) = block;
mprintf("Process noise Q:\n");
disp(Q);

// Create measurement noise covariance
R = measurement_noise_std^2 * eye(spatial_dims, spatial_dims);
mprintf("Measurement noise R:\n");
disp(R);

// True initial state: [x, vx, y, vy]
// Target starts at origin, moving diagonally
true_state = [0; 10; 0; 5];  // Position (0,0), velocity (10, 5) m/s

// Initialize filter
// Start with uncertain estimate
initial_state = [0; 0; 0; 0];
initial_covariance = 100 * eye(4, 4);
gs = GaussianState(initial_state, initial_covariance, 0.0);

mprintf("\nInitial state estimate:\n");
disp(gs_state_vector(gs)');

// Storage for results
true_positions = zeros(n_steps, 2);
measured_positions = zeros(n_steps, 2);
estimated_positions = zeros(n_steps, 2);
estimated_velocities = zeros(n_steps, 2);

// Simulation loop
mprintf("\nRunning simulation...\n");
for k = 1:n_steps
    // True state evolution
    true_state = F * true_state;

    // Generate noisy measurement
    true_position = [true_state(1); true_state(3)];
    noise = measurement_noise_std * rand(2, 1, "normal");
    measurement = true_position + noise;

    // Store true and measured positions
    true_positions(k, :) = true_position';
    measured_positions(k, :) = measurement';

    // Kalman filter prediction
    gs_pred = kalman_predict(gs, F, Q);

    // Kalman filter update
    gs = kalman_update(gs_pred, measurement, H, R);

    // Store estimates
    state_est = gs_state_vector(gs);
    estimated_positions(k, :) = [state_est(1), state_est(3)];
    estimated_velocities(k, :) = [state_est(2), state_est(4)];
end

// Display final results
mprintf("\nSimulation complete!\n\n");

mprintf("Final true state:     [%.2f, %.2f, %.2f, %.2f]\n", ..
        true_state(1), true_state(2), true_state(3), true_state(4));
mprintf("Final estimated state: [%.2f, %.2f, %.2f, %.2f]\n", ..
        state_est(1), state_est(2), state_est(3), state_est(4));

// Compute RMS errors
pos_errors = sqrt((true_positions(:,1) - estimated_positions(:,1)).^2 + ..
                  (true_positions(:,2) - estimated_positions(:,2)).^2);
meas_errors = sqrt((true_positions(:,1) - measured_positions(:,1)).^2 + ..
                   (true_positions(:,2) - measured_positions(:,2)).^2);

mprintf("\nRMS position error (measurement): %.2f m\n", sqrt(mean(meas_errors.^2)));
mprintf("RMS position error (filter):      %.2f m\n", sqrt(mean(pos_errors.^2)));
mprintf("Improvement factor: %.2fx\n", sqrt(mean(meas_errors.^2)) / sqrt(mean(pos_errors.^2)));

// Plot results if graphics available
try
    clf();
    subplot(2, 1, 1);
    plot(true_positions(:, 1), true_positions(:, 2), 'b-', 'LineWidth', 2);
    plot(measured_positions(:, 1), measured_positions(:, 2), 'rx', 'MarkerSize', 8);
    plot(estimated_positions(:, 1), estimated_positions(:, 2), 'g-o', 'LineWidth', 1.5);
    legend(['True'; 'Measured'; 'Estimated'], 'in_upper_left');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Kalman Filter Tracking Demo');
    xgrid();

    subplot(2, 1, 2);
    time = (1:n_steps)';
    plot(time, pos_errors, 'g-', 'LineWidth', 1.5);
    plot(time, meas_errors, 'r--', 'LineWidth', 1.5);
    legend(['Filter Error'; 'Measurement Error']);
    xlabel('Time Step');
    ylabel('Position Error (m)');
    title('Position Error Over Time');
    xgrid();
catch
    mprintf("\nGraphics not available, skipping plot.\n");
end

mprintf("\nDemo complete!\n");
