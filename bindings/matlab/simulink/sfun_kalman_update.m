function sfun_kalman_update(block)
% SFUN_KALMAN_UPDATE Simulink S-function for Kalman filter update
%
%   This Level-2 M-file S-function implements the Kalman filter update step.
%
% Block Parameters:
%   state_dim  - State vector dimension
%   meas_dim   - Measurement vector dimension
%   H          - Measurement matrix (meas_dim x state_dim)
%   R          - Measurement noise covariance (meas_dim x meas_dim)
%
% Inputs:
%   Port 1: Predicted state vector (state_dim x 1)
%   Port 2: Predicted covariance matrix (state_dim^2 x 1, vectorized)
%   Port 3: Measurement vector (meas_dim x 1)
%
% Outputs:
%   Port 1: Posterior state vector (state_dim x 1)
%   Port 2: Posterior covariance matrix (state_dim^2 x 1, vectorized)
%   Port 3: Innovation vector (meas_dim x 1)
%
% Usage in Simulink:
%   1. Add "Level-2 MATLAB S-Function" block
%   2. Set S-function name to "sfun_kalman_update"
%   3. Set parameters appropriately
%
% See also: sfun_kalman_predict

setup(block);
end

function setup(block)
    % Register number of ports
    block.NumInputPorts  = 3;
    block.NumOutputPorts = 3;

    % Get dimensions from parameters
    state_dim = block.DialogPrm(1).Data;
    meas_dim = block.DialogPrm(2).Data;

    % Setup input port 1: Predicted state vector
    block.InputPort(1).Dimensions = state_dim;
    block.InputPort(1).DatatypeID = 0;
    block.InputPort(1).Complexity = 'Real';
    block.InputPort(1).DirectFeedthrough = true;
    block.InputPort(1).SamplingMode = 'Sample';

    % Setup input port 2: Predicted covariance (vectorized)
    block.InputPort(2).Dimensions = state_dim * state_dim;
    block.InputPort(2).DatatypeID = 0;
    block.InputPort(2).Complexity = 'Real';
    block.InputPort(2).DirectFeedthrough = true;
    block.InputPort(2).SamplingMode = 'Sample';

    % Setup input port 3: Measurement
    block.InputPort(3).Dimensions = meas_dim;
    block.InputPort(3).DatatypeID = 0;
    block.InputPort(3).Complexity = 'Real';
    block.InputPort(3).DirectFeedthrough = true;
    block.InputPort(3).SamplingMode = 'Sample';

    % Setup output port 1: Posterior state
    block.OutputPort(1).Dimensions = state_dim;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';
    block.OutputPort(1).SamplingMode = 'Sample';

    % Setup output port 2: Posterior covariance (vectorized)
    block.OutputPort(2).Dimensions = state_dim * state_dim;
    block.OutputPort(2).DatatypeID = 0;
    block.OutputPort(2).Complexity = 'Real';
    block.OutputPort(2).SamplingMode = 'Sample';

    % Setup output port 3: Innovation
    block.OutputPort(3).Dimensions = meas_dim;
    block.OutputPort(3).DatatypeID = 0;
    block.OutputPort(3).Complexity = 'Real';
    block.OutputPort(3).SamplingMode = 'Sample';

    % Register parameters: state_dim, meas_dim, H, R
    block.NumDialogPrms = 4;
    block.DialogPrmsTunable = {'Nontunable', 'Nontunable', 'Tunable', 'Tunable'};

    % Set sample time
    block.SampleTimes = [-1 0];  % Inherited

    % Set block methods
    block.RegBlockMethod('Outputs', @Output);
    block.RegBlockMethod('SetInputPortSamplingMode', @SetInputPortSamplingMode);
end

function SetInputPortSamplingMode(block, idx, mode)
    block.InputPort(idx).SamplingMode = mode;
    for i = 1:3
        block.OutputPort(i).SamplingMode = mode;
    end
end

function Output(block)
    % Get parameters
    state_dim = block.DialogPrm(1).Data;
    meas_dim = block.DialogPrm(2).Data;
    H = block.DialogPrm(3).Data;
    R = block.DialogPrm(4).Data;

    % Get inputs
    x_pred = block.InputPort(1).Data;
    P_pred_vec = block.InputPort(2).Data;
    z = block.InputPort(3).Data;

    % Reshape covariance
    P_pred = reshape(P_pred_vec, state_dim, state_dim);

    % Kalman update equations
    % Innovation
    y = z - H * x_pred;

    % Innovation covariance
    S = H * P_pred * H' + R;

    % Kalman gain
    K = P_pred * H' / S;

    % Posterior state
    x_post = x_pred + K * y;

    % Posterior covariance (Joseph form for numerical stability)
    I_KH = eye(state_dim) - K * H;
    P_post = I_KH * P_pred * I_KH' + K * R * K';

    % Set outputs
    block.OutputPort(1).Data = x_post;
    block.OutputPort(2).Data = P_post(:);
    block.OutputPort(3).Data = y;
end
