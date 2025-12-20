function out = stonesoup_kalman_update_sfun(u)
% STONESOUP_KALMAN_UPDATE_SFUN Simulink callback for Kalman update
%
%   This function is called by the Interpreted MATLAB Function block
%   in the Kalman Updater Simulink block.
%
%   Input u is a muxed signal: [x; P_vec; z]
%   Output is a muxed signal: [x_post; P_post_vec]
%
% SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
% SPDX-License-Identifier: MIT

    % Get mask parameters from base workspace
    % These are set by the mask initialization
    H = evalin('caller', 'H');
    R = evalin('caller', 'R');

    state_dim = size(H, 2);
    meas_dim = size(H, 1);

    % Split input
    x = u(1:state_dim);
    P_vec = u(state_dim+1:state_dim+state_dim^2);
    P = reshape(P_vec, state_dim, state_dim);
    z = u(state_dim+state_dim^2+1:end);

    % Kalman update equations
    % Innovation
    y = z - H * x;

    % Innovation covariance
    S = H * P * H' + R;

    % Kalman gain
    K = P * H' / S;

    % Updated state
    x_post = x + K * y;

    % Updated covariance (Joseph form for numerical stability)
    I_KH = eye(state_dim) - K * H;
    P_post = I_KH * P * I_KH' + K * R * K';

    % Combine output
    out = [x_post; P_post(:)];
end
