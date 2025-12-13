// Xcos Kalman Filter Demo
//
// This demo creates a complete Kalman filter tracking simulation
// using Stone Soup Xcos blocks.
//
// The scenario:
//   - 2D constant velocity target motion
//   - Position-only measurements with noise
//   - Kalman filter prediction and update
//
// Run this script to:
//   1. Load the Stone Soup Xcos palette
//   2. Create the tracking diagram
//   3. Run the simulation
//   4. Plot results

// Clear workspace
clear;
clc;

// Load Stone Soup toolbox
exec(get_absolute_file_path("xcos_kalman_demo.sce") + "../builder.sce", -1);

// Load Xcos palette
exec(get_absolute_file_path("xcos_kalman_demo.sce") + "../xcos/loader.sce");

// Simulation parameters
dt = 0.1;           // Time step
T = 10;             // Simulation duration
q = 0.1;            // Process noise intensity
sigma_meas = 1.0;   // Measurement noise std dev

// True initial state [x, vx, y, vy]
x0_true = [0; 5; 0; 3];

// Initial state estimate (with uncertainty)
x0_est = [0.5; 4.5; 0.5; 2.5];
P0 = diag([1, 1, 1, 1]);

// Generate ground truth trajectory
N = floor(T / dt);
times = (0:N-1) * dt;

// Transition matrix for constant velocity
F = [1, dt, 0, 0; ...
     0, 1, 0, 0; ...
     0, 0, 1, dt; ...
     0, 0, 0, 1];

// Process noise
Q = q * [dt^4/4, dt^3/2, 0, 0; ...
         dt^3/2, dt^2, 0, 0; ...
         0, 0, dt^4/4, dt^3/2; ...
         0, 0, dt^3/2, dt^2];

// Measurement matrix (position only)
H = [1, 0, 0, 0; ...
     0, 0, 1, 0];

// Measurement noise
R = sigma_meas^2 * eye(2, 2);

// Generate trajectory
x_true = zeros(4, N);
x_true(:, 1) = x0_true;

for k = 2:N
    // Add process noise
    w = sqrt(q) * [dt^2/2 * rand(1, "normal"); ...
                   dt * rand(1, "normal"); ...
                   dt^2/2 * rand(1, "normal"); ...
                   dt * rand(1, "normal")];
    x_true(:, k) = F * x_true(:, k-1) + w;
end

// Generate noisy measurements
z = H * x_true + sigma_meas * rand(2, N, "normal");

// Run Kalman filter
x_est = zeros(4, N);
P_trace = zeros(1, N);  // Trace of covariance
innovation = zeros(2, N);

x = x0_est;
P = P0;

for k = 1:N
    // Predict
    x_pred = F * x;
    P_pred = F * P * F' + Q;

    // Update
    y = z(:, k) - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' * inv(S);
    x = x_pred + K * y;
    I_KH = eye(4, 4) - K * H;
    P = I_KH * P_pred * I_KH' + K * R * K';

    x_est(:, k) = x;
    P_trace(k) = trace(P);
    innovation(:, k) = y;
end

// Plot results
figure(1);
clf();

// Trajectory plot
subplot(2, 2, 1);
plot(x_true(1, :), x_true(3, :), "b-", "LineWidth", 2);
plot(z(1, :), z(2, :), "r.", "MarkerSize", 4);
plot(x_est(1, :), x_est(3, :), "g--", "LineWidth", 2);
xlabel("X Position");
ylabel("Y Position");
title("2D Tracking");
legend(["True", "Measurements", "Estimate"]);
a = gca();
a.grid = [1 1];

// Position errors
subplot(2, 2, 2);
pos_err = sqrt((x_true(1, :) - x_est(1, :)).^2 + (x_true(3, :) - x_est(3, :)).^2);
plot(times, pos_err, "b-", "LineWidth", 1.5);
xlabel("Time (s)");
ylabel("Position Error");
title("Position RMSE");
a = gca();
a.grid = [1 1];

// Velocity estimates
subplot(2, 2, 3);
plot(times, x_true(2, :), "b-", "LineWidth", 2);
plot(times, x_est(2, :), "g--", "LineWidth", 1.5);
plot(times, x_true(4, :), "r-", "LineWidth", 2);
plot(times, x_est(4, :), "m--", "LineWidth", 1.5);
xlabel("Time (s)");
ylabel("Velocity");
title("Velocity Estimates");
legend(["True Vx", "Est Vx", "True Vy", "Est Vy"]);
a = gca();
a.grid = [1 1];

// Covariance trace
subplot(2, 2, 4);
plot(times, P_trace, "b-", "LineWidth", 1.5);
xlabel("Time (s)");
ylabel("trace(P)");
title("Covariance Trace");
a = gca();
a.grid = [1 1];

// Innovation statistics
figure(2);
clf();
subplot(2, 1, 1);
plot(times, innovation(1, :), "b-");
xlabel("Time (s)");
ylabel("Innovation X");
title("Innovation (should be zero-mean white noise)");
a = gca();
a.grid = [1 1];

subplot(2, 1, 2);
plot(times, innovation(2, :), "r-");
xlabel("Time (s)");
ylabel("Innovation Y");
a = gca();
a.grid = [1 1];

disp("Kalman Filter Demo Complete");
disp("Mean position error: " + string(mean(pos_err)));
