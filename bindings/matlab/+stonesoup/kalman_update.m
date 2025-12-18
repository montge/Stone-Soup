function gs_post = kalman_update(gs_pred, measurement, H, R)
% KALMAN_UPDATE Kalman filter update step
%
%   gs_post = kalman_update(gs_pred, measurement, H, R)
%
% Inputs:
%   gs_pred     - Predicted GaussianState
%   measurement - Measurement vector (meas_dim x 1)
%   H           - Measurement matrix (meas_dim x state_dim)
%   R           - Measurement noise covariance (meas_dim x meas_dim)
%
% Outputs:
%   gs_post     - Posterior GaussianState
%
% Description:
%   Performs the Kalman filter update step:
%     y = z - H * x_pred  (innovation)
%     S = H * P_pred * H' + R  (innovation covariance)
%     K = P_pred * H' * inv(S)  (Kalman gain)
%     x_post = x_pred + K * y
%     P_post = (I - K * H) * P_pred
%
% Example:
%   % Position-only measurement
%   H = [1 0 0 0; 0 0 1 0];  % Extract x, y from [x, vx, y, vy]
%   R = 0.5 * eye(2);
%   measurement = [1.1; 1.2];
%
%   % Update
%   gs_post = stonesoup.kalman_update(gs_pred, measurement, H, R);
%
% See also: stonesoup.kalman_predict, stonesoup.GaussianState

% Validate inputs
if ~isa(gs_pred, 'stonesoup.GaussianState')
    error('stonesoup:invalidInput', 'gs_pred must be a GaussianState');
end

state_dim = gs_pred.dim;
measurement = measurement(:);  % Ensure column vector
meas_dim = length(measurement);

if size(H, 1) ~= meas_dim || size(H, 2) ~= state_dim
    error('stonesoup:dimensionMismatch', 'H must be %d x %d', meas_dim, state_dim);
end
if size(R, 1) ~= meas_dim || size(R, 2) ~= meas_dim
    error('stonesoup:dimensionMismatch', 'R must be %d x %d', meas_dim, meas_dim);
end

% Kalman update computation (pure MATLAB/Octave implementation)
% y = z - H * x_pred  (innovation)
% S = H * P_pred * H' + R  (innovation covariance)
% K = P_pred * H' * inv(S)  (Kalman gain)
% x_post = x_pred + K * y
% P_post = (I - K * H) * P_pred
x_pred = gs_pred.state_vector;
P_pred = gs_pred.covariance;

y = measurement - H * x_pred;            % Innovation
S = H * P_pred * H' + R;                 % Innovation covariance
K = P_pred * H' / S;                     % Kalman gain (using / instead of inv)
x_post = x_pred + K * y;                 % Posterior state
P_post = (eye(state_dim) - K * H) * P_pred;  % Posterior covariance

% Create output GaussianState
gs_post = stonesoup.GaussianState(x_post, P_post, gs_pred.timestamp);
end
