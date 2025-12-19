function sfun_constant_velocity(block)
% SFUN_CONSTANT_VELOCITY Simulink S-function for constant velocity model
%
%   This S-function generates transition matrix F and process noise Q
%   for a constant velocity motion model.
%
% Block Parameters:
%   spatial_dims - Number of spatial dimensions (1, 2, or 3)
%   q           - Process noise intensity (acceleration variance)
%
% Inputs:
%   Port 1: Time step dt (scalar)
%
% Outputs:
%   Port 1: Transition matrix F (state_dim^2 x 1, vectorized)
%   Port 2: Process noise Q (state_dim^2 x 1, vectorized)
%
% Description:
%   State vector format: [x, vx, y, vy, z, vz] for 3D
%   Transition: position += velocity * dt
%
% See also: sfun_kalman_predict

setup(block);
end

function setup(block)
    % Register number of ports
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 2;

    % Get parameters
    spatial_dims = block.DialogPrm(1).Data;
    state_dim = spatial_dims * 2;

    % Setup input port: dt
    block.InputPort(1).Dimensions = 1;
    block.InputPort(1).DatatypeID = 0;
    block.InputPort(1).Complexity = 'Real';
    block.InputPort(1).DirectFeedthrough = true;
    block.InputPort(1).SamplingMode = 'Sample';

    % Setup output port 1: F matrix (vectorized)
    block.OutputPort(1).Dimensions = state_dim * state_dim;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';
    block.OutputPort(1).SamplingMode = 'Sample';

    % Setup output port 2: Q matrix (vectorized)
    block.OutputPort(2).Dimensions = state_dim * state_dim;
    block.OutputPort(2).DatatypeID = 0;
    block.OutputPort(2).Complexity = 'Real';
    block.OutputPort(2).SamplingMode = 'Sample';

    % Register parameters: spatial_dims, q
    block.NumDialogPrms = 2;
    block.DialogPrmsTunable = {'Nontunable', 'Tunable'};

    % Set sample time
    block.SampleTimes = [-1 0];

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
    spatial_dims = block.DialogPrm(1).Data;
    q = block.DialogPrm(2).Data;
    state_dim = spatial_dims * 2;

    % Get input
    dt = block.InputPort(1).Data;

    % Build transition matrix F
    % For each spatial dimension: [pos; vel] -> [pos + vel*dt; vel]
    F = eye(state_dim);
    for i = 1:spatial_dims
        pos_idx = (i-1)*2 + 1;
        vel_idx = pos_idx + 1;
        F(pos_idx, vel_idx) = dt;
    end

    % Build process noise Q (discrete white noise acceleration)
    % For each dimension: Q_block = q * [dt^4/4, dt^3/2; dt^3/2, dt^2]
    Q = zeros(state_dim);
    Q_block = q * [dt^4/4, dt^3/2; dt^3/2, dt^2];
    for i = 1:spatial_dims
        idx = (i-1)*2 + (1:2);
        Q(idx, idx) = Q_block;
    end

    % Set outputs (vectorized column-major)
    block.OutputPort(1).Data = F(:);
    block.OutputPort(2).Data = Q(:);
end
