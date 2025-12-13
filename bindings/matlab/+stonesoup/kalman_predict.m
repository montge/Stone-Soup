function gs_pred = kalman_predict(gs_prior, F, Q)
% KALMAN_PREDICT Kalman filter prediction step
%
%   gs_pred = kalman_predict(gs_prior, F, Q)
%
% Inputs:
%   gs_prior - Prior GaussianState
%   F        - State transition matrix (state_dim x state_dim)
%   Q        - Process noise covariance (state_dim x state_dim)
%
% Outputs:
%   gs_pred  - Predicted GaussianState
%
% Description:
%   Performs the Kalman filter prediction step:
%     x_pred = F * x
%     P_pred = F * P * F' + Q
%
% Example:
%   % Create prior state
%   x = [0; 1; 0; 1];  % [x, vx, y, vy]
%   P = eye(4);
%   gs_prior = stonesoup.GaussianState(x, P);
%
%   % Constant velocity transition (dt = 1)
%   F = [1 1 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 1];
%   Q = 0.1 * eye(4);
%
%   % Predict
%   gs_pred = stonesoup.kalman_predict(gs_prior, F, Q);
%
% See also: stonesoup.kalman_update, stonesoup.GaussianState

% Validate inputs
if ~isa(gs_prior, 'stonesoup.GaussianState')
    error('stonesoup:invalidInput', 'gs_prior must be a GaussianState');
end

dim = gs_prior.dim;
if size(F, 1) ~= dim || size(F, 2) ~= dim
    error('stonesoup:dimensionMismatch', 'F must be %d x %d', dim, dim);
end
if size(Q, 1) ~= dim || size(Q, 2) ~= dim
    error('stonesoup:dimensionMismatch', 'Q must be %d x %d', dim, dim);
end

% Call MEX function
[x_pred, P_pred] = stonesoup_mex('kalman_predict', ...
    gs_prior.state_vector, gs_prior.covariance, F, Q);

% Create output GaussianState
gs_pred = stonesoup.GaussianState(x_pred, P_pred, gs_prior.timestamp);
end
