%% Kalman Filter Tracking Demo
%
% This demo demonstrates the Stone Soup MATLAB bindings for Kalman filtering.
% It creates a 2D constant velocity tracking scenario and compares:
%   - Pure MATLAB implementation
%   - Stone Soup MEX-based implementation
%
% Compatible with MATLAB R2016a+ and GNU Octave 5.0+

%% Clear workspace
clear; close all; clc;

%% Simulation parameters
dt = 0.1;           % Time step [s]
T = 10;             % Simulation duration [s]
q = 0.1;            % Process noise intensity
sigma_meas = 1.0;   % Measurement noise std dev

% True initial state [x, vx, y, vy]
x0_true = [0; 5; 0; 3];

% Initial estimate (with uncertainty)
x0_est = [0.5; 4.5; 0.5; 2.5];
P0 = eye(4);

%% System matrices
% State transition (constant velocity)
F = [1, dt, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, dt;
     0, 0, 0, 1];

% Process noise
Q = q * [dt^4/4, dt^3/2, 0, 0;
         dt^3/2, dt^2, 0, 0;
         0, 0, dt^4/4, dt^3/2;
         0, 0, dt^3/2, dt^2];

% Measurement matrix (position only)
H = [1, 0, 0, 0;
     0, 0, 1, 0];

% Measurement noise
R = sigma_meas^2 * eye(2);

%% Generate ground truth and measurements
N = floor(T / dt);
times = (0:N-1) * dt;

% Generate trajectory with process noise
rng(42);  % For reproducibility
x_true = zeros(4, N);
x_true(:, 1) = x0_true;

for k = 2:N
    w = sqrt(q) * [dt^2/2 * randn; dt * randn; dt^2/2 * randn; dt * randn];
    x_true(:, k) = F * x_true(:, k-1) + w;
end

% Generate noisy measurements
z = H * x_true + sigma_meas * randn(2, N);

%% Pure MATLAB Kalman Filter
fprintf('Running pure MATLAB Kalman filter...\n');
tic;

x_matlab = zeros(4, N);
P_trace_matlab = zeros(1, N);

x = x0_est;
P = P0;

for k = 1:N
    % Predict
    x_pred = F * x;
    P_pred = F * P * F' + Q;

    % Update
    y = z(:, k) - H * x_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x = x_pred + K * y;
    I_KH = eye(4) - K * H;
    P = I_KH * P_pred * I_KH' + K * R * K';  % Joseph form

    x_matlab(:, k) = x;
    P_trace_matlab(k) = trace(P);
end

time_matlab = toc;
fprintf('MATLAB time: %.4f s\n', time_matlab);

%% Stone Soup MEX Kalman Filter (if available)
x_stonesoup = zeros(4, N);
P_trace_stonesoup = zeros(1, N);
use_stonesoup = false;

try
    % Try to use Stone Soup MEX
    addpath('../mex');
    addpath('..');
    stonesoup_mex('version');
    use_stonesoup = true;

    fprintf('Running Stone Soup MEX Kalman filter...\n');
    tic;

    % Create initial state
    gs = stonesoup.GaussianState(x0_est, P0);

    for k = 1:N
        % Predict
        gs_pred = stonesoup.kalman_predict(gs, F, Q);

        % Update
        gs = stonesoup.kalman_update(gs_pred, z(:, k), H, R);

        x_stonesoup(:, k) = gs.state_vector;
        P_trace_stonesoup(k) = trace(gs.covariance);
    end

    time_stonesoup = toc;
    fprintf('Stone Soup time: %.4f s\n', time_stonesoup);

catch ME
    fprintf('Stone Soup MEX not available: %s\n', ME.message);
    fprintf('Run make.m in the mex/ directory to compile.\n');
end

%% Compute errors
pos_err_matlab = sqrt((x_true(1,:) - x_matlab(1,:)).^2 + ...
                      (x_true(3,:) - x_matlab(3,:)).^2);

if use_stonesoup
    pos_err_stonesoup = sqrt((x_true(1,:) - x_stonesoup(1,:)).^2 + ...
                             (x_true(3,:) - x_stonesoup(3,:)).^2);
    max_diff = max(abs(x_matlab(:) - x_stonesoup(:)));
    fprintf('Maximum difference between MATLAB and Stone Soup: %.2e\n', max_diff);
end

%% Plot results
figure('Name', 'Kalman Filter Tracking Demo', 'Position', [100, 100, 1000, 800]);

% Trajectory
subplot(2, 2, 1);
plot(x_true(1,:), x_true(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(z(1,:), z(2,:), 'r.', 'MarkerSize', 8, 'DisplayName', 'Measurements');
plot(x_matlab(1,:), x_matlab(3,:), 'g--', 'LineWidth', 1.5, 'DisplayName', 'MATLAB Est.');
if use_stonesoup
    plot(x_stonesoup(1,:), x_stonesoup(3,:), 'm:', 'LineWidth', 2, 'DisplayName', 'Stone Soup Est.');
end
xlabel('X Position [m]');
ylabel('Y Position [m]');
title('2D Tracking Trajectory');
legend('Location', 'best');
grid on;

% Position error
subplot(2, 2, 2);
plot(times, pos_err_matlab, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MATLAB');
hold on;
if use_stonesoup
    plot(times, pos_err_stonesoup, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Stone Soup');
end
xlabel('Time [s]');
ylabel('Position Error [m]');
title('Position RMSE');
legend('Location', 'best');
grid on;

% Velocity estimates
subplot(2, 2, 3);
plot(times, x_true(2,:), 'b-', 'LineWidth', 2, 'DisplayName', 'True V_x');
hold on;
plot(times, x_matlab(2,:), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Est V_x');
plot(times, x_true(4,:), 'r-', 'LineWidth', 2, 'DisplayName', 'True V_y');
plot(times, x_matlab(4,:), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Est V_y');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity Estimates');
legend('Location', 'best');
grid on;

% Covariance trace
subplot(2, 2, 4);
plot(times, P_trace_matlab, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MATLAB');
hold on;
if use_stonesoup
    plot(times, P_trace_stonesoup, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Stone Soup');
end
xlabel('Time [s]');
ylabel('trace(P)');
title('Covariance Trace (Filter Consistency)');
legend('Location', 'best');
grid on;

%% Summary statistics
fprintf('\n=== Summary ===\n');
fprintf('Mean position error: %.4f m\n', mean(pos_err_matlab));
fprintf('Final position error: %.4f m\n', pos_err_matlab(end));
fprintf('Final covariance trace: %.4f\n', P_trace_matlab(end));

if use_stonesoup
    fprintf('\nStone Soup vs MATLAB speedup: %.2fx\n', time_matlab / time_stonesoup);
end

fprintf('\nDemo complete!\n');
