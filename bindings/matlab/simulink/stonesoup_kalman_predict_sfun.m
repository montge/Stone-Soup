function out = stonesoup_kalman_predict_sfun(u)
% STONESOUP_KALMAN_PREDICT_SFUN Simulink callback for Kalman prediction
%
%   This function is called by the Interpreted MATLAB Function block
%   in the Kalman Predictor Simulink block.
%
%   Input u is a muxed signal: [x; P_vec]
%   Output is a muxed signal: [x_pred; P_pred_vec]
%
% SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
% SPDX-License-Identifier: MIT

    % Get mask parameters from base workspace
    % These are set by the mask initialization
    F = evalin('caller', 'F');
    Q = evalin('caller', 'Q');

    state_dim = size(F, 1);

    % Split input
    x = u(1:state_dim);
    P_vec = u(state_dim+1:end);
    P = reshape(P_vec, state_dim, state_dim);

    % Kalman prediction
    x_pred = F * x;
    P_pred = F * P * F' + Q;

    % Combine output
    out = [x_pred; P_pred(:)];
end
