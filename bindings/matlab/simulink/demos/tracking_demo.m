% TRACKING_DEMO Multi-target tracking demonstration using Stone Soup Simulink blocks
%
%   This script demonstrates how to use the Stone Soup Simulink library
%   to perform Kalman filtering for target tracking.
%
%   The demo creates a simple tracking scenario with:
%   - A constant velocity target
%   - Noisy position measurements
%   - Kalman filter for state estimation
%
% SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
% SPDX-License-Identifier: MIT

%% Setup
clear; clc;

% Add path to Simulink library
libDir = fileparts(mfilename('fullpath'));
parentDir = fileparts(libDir);
addpath(parentDir);

%% Simulation Parameters
dt = 0.1;           % Time step (seconds)
T = 10;             % Total simulation time (seconds)
N = T / dt;         % Number of time steps

% State: [x, vx, y, vy]' - position and velocity in 2D
state_dim = 4;
meas_dim = 2;       % Measure position only: [x, y]'

%% Target Motion Model (Constant Velocity)
% State transition matrix
F = [1 dt 0  0;
     0  1 0  0;
     0  0 1 dt;
     0  0 0  1];

% Process noise (acceleration disturbance)
q = 0.1;  % Process noise intensity
Q_block = [dt^3/3, dt^2/2;
           dt^2/2, dt] * q;
Q = blkdiag(Q_block, Q_block);

%% Measurement Model
% Measurement matrix (observe position only)
H = [1 0 0 0;
     0 0 1 0];

% Measurement noise covariance
r = 1.0;  % Measurement noise standard deviation
R = r^2 * eye(meas_dim);

%% Initial State
x0_true = [0; 1; 0; 0.5];  % Start at origin, moving diagonally
P0 = diag([1, 0.1, 1, 0.1]);  % Initial uncertainty

%% Generate Ground Truth and Measurements
rng(42);  % For reproducibility

% Preallocate
x_true = zeros(state_dim, N);
z_meas = zeros(meas_dim, N);

x_true(:, 1) = x0_true;
z_meas(:, 1) = H * x_true(:, 1) + r * randn(meas_dim, 1);

for k = 2:N
    % Process noise
    w = chol(Q)' * randn(state_dim, 1);
    x_true(:, k) = F * x_true(:, k-1) + w;

    % Measurement noise
    v = r * randn(meas_dim, 1);
    z_meas(:, k) = H * x_true(:, k) + v;
end

%% Kalman Filter (MATLAB Implementation for Comparison)
x_est = zeros(state_dim, N);
P_est = zeros(state_dim, state_dim, N);

% Initialize
x_est(:, 1) = x0_true + chol(P0)' * randn(state_dim, 1);  % Noisy initial estimate
P_est(:, :, 1) = P0;

for k = 2:N
    % Predict
    x_pred = F * x_est(:, k-1);
    P_pred = F * P_est(:, :, k-1) * F' + Q;

    % Update
    y = z_meas(:, k) - H * x_pred;  % Innovation
    S = H * P_pred * H' + R;         % Innovation covariance
    K = P_pred * H' / S;             % Kalman gain

    x_est(:, k) = x_pred + K * y;
    I_KH = eye(state_dim) - K * H;
    P_est(:, :, k) = I_KH * P_pred * I_KH' + K * R * K';
end

%% Plotting
figure('Name', 'Multi-Target Tracking Demo', 'Position', [100 100 1200 800]);

% Subplot 1: 2D trajectory
subplot(2, 2, 1);
plot(x_true(1, :), x_true(3, :), 'g-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
hold on;
plot(z_meas(1, :), z_meas(2, :), 'r.', 'MarkerSize', 8, 'DisplayName', 'Measurements');
plot(x_est(1, :), x_est(3, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Kalman Estimate');
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('2D Target Trajectory');
legend('Location', 'best');
grid on;
axis equal;

% Subplot 2: X position over time
subplot(2, 2, 2);
t = (0:N-1) * dt;
plot(t, x_true(1, :), 'g-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
hold on;
plot(t, z_meas(1, :), 'r.', 'MarkerSize', 8, 'DisplayName', 'Measurements');
plot(t, x_est(1, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Kalman Estimate');
xlabel('Time (s)');
ylabel('X Position (m)');
title('X Position vs Time');
legend('Location', 'best');
grid on;

% Subplot 3: Y position over time
subplot(2, 2, 3);
plot(t, x_true(3, :), 'g-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
hold on;
plot(t, z_meas(2, :), 'r.', 'MarkerSize', 8, 'DisplayName', 'Measurements');
plot(t, x_est(3, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Kalman Estimate');
xlabel('Time (s)');
ylabel('Y Position (m)');
title('Y Position vs Time');
legend('Location', 'best');
grid on;

% Subplot 4: Position estimation error
subplot(2, 2, 4);
pos_error = sqrt((x_true(1, :) - x_est(1, :)).^2 + (x_true(3, :) - x_est(3, :)).^2);
plot(t, pos_error, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Position Error (m)');
title('Kalman Filter Position Error');
grid on;

% Calculate and display RMSE
rmse_pos = sqrt(mean(pos_error.^2));
text(0.5, 0.9, sprintf('RMSE: %.3f m', rmse_pos), 'Units', 'normalized', 'FontSize', 12);

sgtitle('Stone Soup Kalman Filter Tracking Demo');

%% Display Summary
fprintf('\n=== Tracking Demo Summary ===\n');
fprintf('Simulation time: %.1f seconds\n', T);
fprintf('Time step: %.2f seconds\n', dt);
fprintf('Number of steps: %d\n', N);
fprintf('State dimension: %d\n', state_dim);
fprintf('Measurement dimension: %d\n', meas_dim);
fprintf('Position RMSE: %.3f m\n', rmse_pos);
fprintf('\nTo use Stone Soup Simulink blocks:\n');
fprintf('1. Open stonesoup_lib.slx in Simulink\n');
fprintf('2. Drag blocks into your model\n');
fprintf('3. Configure mask parameters (F, Q, H, R)\n');
fprintf('4. Connect state and measurement signals\n');
