function sfun_kalman_predict(block)
% SFUN_KALMAN_PREDICT Simulink S-function for Kalman filter prediction
%
%   This Level-2 M-file S-function implements the Kalman filter prediction step.
%
% Block Parameters:
%   state_dim  - State vector dimension
%   F          - State transition matrix (state_dim x state_dim)
%   Q          - Process noise covariance (state_dim x state_dim)
%
% Inputs:
%   Port 1: State vector (state_dim x 1)
%   Port 2: Covariance matrix (state_dim^2 x 1, vectorized column-major)
%
% Outputs:
%   Port 1: Predicted state vector (state_dim x 1)
%   Port 2: Predicted covariance matrix (state_dim^2 x 1, vectorized)
%
% Usage in Simulink:
%   1. Add "Level-2 MATLAB S-Function" block
%   2. Set S-function name to "sfun_kalman_predict"
%   3. Set parameters to "4, eye(4), 0.1*eye(4)" for 4D state
%
% See also: sfun_kalman_update

setup(block);
end

function setup(block)
    % Register number of ports
    block.NumInputPorts  = 2;
    block.NumOutputPorts = 2;

    % Get state dimension from parameter
    state_dim = block.DialogPrm(1).Data;

    % Setup input port 1: State vector
    block.InputPort(1).Dimensions = state_dim;
    block.InputPort(1).DatatypeID = 0;  % double
    block.InputPort(1).Complexity = 'Real';
    block.InputPort(1).DirectFeedthrough = true;
    block.InputPort(1).SamplingMode = 'Sample';

    % Setup input port 2: Covariance (vectorized)
    block.InputPort(2).Dimensions = state_dim * state_dim;
    block.InputPort(2).DatatypeID = 0;  % double
    block.InputPort(2).Complexity = 'Real';
    block.InputPort(2).DirectFeedthrough = true;
    block.InputPort(2).SamplingMode = 'Sample';

    % Setup output port 1: Predicted state
    block.OutputPort(1).Dimensions = state_dim;
    block.OutputPort(1).DatatypeID = 0;  % double
    block.OutputPort(1).Complexity = 'Real';
    block.OutputPort(1).SamplingMode = 'Sample';

    % Setup output port 2: Predicted covariance (vectorized)
    block.OutputPort(2).Dimensions = state_dim * state_dim;
    block.OutputPort(2).DatatypeID = 0;  % double
    block.OutputPort(2).Complexity = 'Real';
    block.OutputPort(2).SamplingMode = 'Sample';

    % Register parameters: state_dim, F, Q
    block.NumDialogPrms = 3;
    block.DialogPrmsTunable = {'Nontunable', 'Tunable', 'Tunable'};

    % Set sample time
    block.SampleTimes = [-1 0];  % Inherited sample time

    % Set block methods
    block.RegBlockMethod('Outputs', @Output);
    block.RegBlockMethod('SetInputPortSamplingMode', @SetInputPortSamplingMode);
end

function SetInputPortSamplingMode(block, idx, mode)
    block.InputPort(idx).SamplingMode = mode;
    block.OutputPort(1).SamplingMode = mode;
    block.OutputPort(2).SamplingMode = mode;
end

function Output(block)
    % Get parameters
    state_dim = block.DialogPrm(1).Data;
    F = block.DialogPrm(2).Data;
    Q = block.DialogPrm(3).Data;

    % Get inputs
    x = block.InputPort(1).Data;
    P_vec = block.InputPort(2).Data;

    % Reshape covariance from vector to matrix (column-major)
    P = reshape(P_vec, state_dim, state_dim);

    % Kalman prediction
    % x_pred = F * x
    % P_pred = F * P * F' + Q
    x_pred = F * x;
    P_pred = F * P * F' + Q;

    % Set outputs
    block.OutputPort(1).Data = x_pred;
    block.OutputPort(2).Data = P_pred(:);  % Vectorize column-major
end
