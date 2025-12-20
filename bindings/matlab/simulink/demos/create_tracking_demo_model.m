% CREATE_TRACKING_DEMO_MODEL Create a Simulink tracking demo model
%
%   This script creates a Simulink model that demonstrates the use of
%   Stone Soup Kalman filter blocks for target tracking.
%
% SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
% SPDX-License-Identifier: MIT

function create_tracking_demo_model()
    % Get the directory containing this script
    scriptDir = fileparts(mfilename('fullpath'));
    parentDir = fileparts(scriptDir);
    modelPath = fullfile(scriptDir, 'tracking_demo_model.slx');

    % Close existing model if open
    if bdIsLoaded('tracking_demo_model')
        close_system('tracking_demo_model', 0);
    end

    % Delete existing model file
    if exist(modelPath, 'file')
        delete(modelPath);
    end

    % Load the library
    libPath = fullfile(parentDir, 'stonesoup_lib.slx');
    if ~exist(libPath, 'file')
        error('Stone Soup library not found. Run create_stonesoup_lib.m first.');
    end
    load_system(libPath);

    % Create new model
    new_system('tracking_demo_model');
    open_system('tracking_demo_model');

    % Model parameters
    dt = 0.1;
    state_dim = 4;
    meas_dim = 2;

    % Create transition matrix F for constant velocity
    F = [1 dt 0  0;
         0  1 0  0;
         0  0 1 dt;
         0  0 0  1];

    % Process noise Q
    q = 0.1;
    Q_block = [dt^3/3, dt^2/2; dt^2/2, dt] * q;
    Q = blkdiag(Q_block, Q_block);

    % Measurement matrix H
    H = [1 0 0 0; 0 0 1 0];

    % Measurement noise R
    R = eye(meas_dim);

    % Initial state
    x0 = [0; 1; 0; 0.5];
    P0 = diag([1, 0.1, 1, 0.1]);

    % Add Gaussian State block (initial state source)
    add_block('stonesoup_lib/Gaussian State', 'tracking_demo_model/Initial State');
    set_param('tracking_demo_model/Initial State', 'Position', [50 100 150 180]);
    set_param('tracking_demo_model/Initial State', 'x0', mat2str(x0));
    set_param('tracking_demo_model/Initial State', 'P0', mat2str(P0));

    % Add Unit Delay for feedback loop (state)
    add_block('simulink/Discrete/Unit Delay', 'tracking_demo_model/State Delay');
    set_param('tracking_demo_model/State Delay', 'Position', [400 95 450 135]);
    set_param('tracking_demo_model/State Delay', 'InitialCondition', mat2str(x0));

    % Add Unit Delay for feedback loop (covariance)
    add_block('simulink/Discrete/Unit Delay', 'tracking_demo_model/Cov Delay');
    set_param('tracking_demo_model/Cov Delay', 'Position', [400 155 450 195]);
    set_param('tracking_demo_model/Cov Delay', 'InitialCondition', mat2str(P0(:)));

    % Add Kalman Predictor block
    add_block('stonesoup_lib/Kalman Predictor', 'tracking_demo_model/Predictor');
    set_param('tracking_demo_model/Predictor', 'Position', [200 100 320 180]);
    set_param('tracking_demo_model/Predictor', 'F', mat2str(F));
    set_param('tracking_demo_model/Predictor', 'Q', mat2str(Q));

    % Add Kalman Updater block
    add_block('stonesoup_lib/Kalman Updater', 'tracking_demo_model/Updater');
    set_param('tracking_demo_model/Updater', 'Position', [550 100 690 200]);
    set_param('tracking_demo_model/Updater', 'H', mat2str(H));
    set_param('tracking_demo_model/Updater', 'R', mat2str(R));

    % Add measurement input (from workspace)
    add_block('simulink/Sources/From Workspace', 'tracking_demo_model/Measurements');
    set_param('tracking_demo_model/Measurements', 'Position', [450 240 550 280]);
    set_param('tracking_demo_model/Measurements', 'VariableName', 'z_in');

    % Add state output scope
    add_block('simulink/Sinks/Scope', 'tracking_demo_model/State Scope');
    set_param('tracking_demo_model/State Scope', 'Position', [750 95 780 125]);

    % Add covariance output (to workspace)
    add_block('simulink/Sinks/To Workspace', 'tracking_demo_model/State Out');
    set_param('tracking_demo_model/State Out', 'Position', [750 155 830 185]);
    set_param('tracking_demo_model/State Out', 'VariableName', 'x_out');

    % Connect blocks
    % Initial state to predictor
    add_line('tracking_demo_model', 'Initial State/1', 'Predictor/1', 'autorouting', 'on');
    add_line('tracking_demo_model', 'Initial State/2', 'Predictor/2', 'autorouting', 'on');

    % Predictor to updater
    add_line('tracking_demo_model', 'Predictor/1', 'Updater/1', 'autorouting', 'on');
    add_line('tracking_demo_model', 'Predictor/2', 'Updater/2', 'autorouting', 'on');

    % Measurements to updater
    add_line('tracking_demo_model', 'Measurements/1', 'Updater/3', 'autorouting', 'on');

    % Updater outputs to scopes/workspaces
    add_line('tracking_demo_model', 'Updater/1', 'State Scope/1', 'autorouting', 'on');
    add_line('tracking_demo_model', 'Updater/1', 'State Out/1', 'autorouting', 'on');

    % Set simulation parameters
    set_param('tracking_demo_model', 'StopTime', '10');
    set_param('tracking_demo_model', 'Solver', 'FixedStepDiscrete');
    set_param('tracking_demo_model', 'FixedStep', num2str(dt));

    % Add annotations
    add_block('simulink/Model-Wide Utilities/DocBlock', 'tracking_demo_model/Info');
    set_param('tracking_demo_model/Info', 'Position', [50 250 250 310]);
    set_param('tracking_demo_model/Info', 'DocumentType', 'Text');

    % Save model
    save_system('tracking_demo_model', modelPath);
    fprintf('Created tracking demo model: %s\n', modelPath);
    fprintf('\nTo run the demo:\n');
    fprintf('1. Load measurement data: z_in = timeseries(z_data, t)\n');
    fprintf('2. Open the model: open_system(''tracking_demo_model'')\n');
    fprintf('3. Run simulation: sim(''tracking_demo_model'')\n');

    close_system('stonesoup_lib');
end
