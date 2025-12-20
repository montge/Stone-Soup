% CREATE_STONESOUP_LIB Create Stone Soup Simulink block library
%
%   This script creates the stonesoup_lib.slx Simulink library containing
%   blocks for Kalman filtering and state estimation.
%
%   Blocks created:
%   - Kalman Predictor: Kalman filter prediction step
%   - Kalman Updater: Kalman filter update step
%   - Constant Velocity Model: Generates F and Q matrices
%   - Gaussian State: Initial state and covariance source
%
% SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
% SPDX-License-Identifier: MIT

function create_stonesoup_lib()
    % Get the directory containing this script
    scriptDir = fileparts(mfilename('fullpath'));
    libPath = fullfile(scriptDir, 'stonesoup_lib.slx');

    % Close existing library if open
    if bdIsLoaded('stonesoup_lib')
        close_system('stonesoup_lib', 0);
    end

    % Delete existing library file
    if exist(libPath, 'file')
        delete(libPath);
    end

    % Create new library
    new_system('stonesoup_lib', 'Library');

    % Set library properties
    set_param('stonesoup_lib', 'Lock', 'off');
    set_param('stonesoup_lib', 'LibraryType', 'BlockLibrary');

    % Add blocks using simple subsystem approach
    add_kalman_predictor('stonesoup_lib');
    add_kalman_updater('stonesoup_lib');
    add_constant_velocity('stonesoup_lib');
    add_gaussian_state('stonesoup_lib');

    % Save the library
    save_system('stonesoup_lib', libPath);
    close_system('stonesoup_lib');

    fprintf('Created Stone Soup Simulink library: %s\n', libPath);
end

function add_kalman_predictor(lib)
    % Add Kalman Predictor block as masked subsystem
    blockPath = [lib '/Kalman Predictor'];

    % Create subsystem
    add_block('simulink/Ports & Subsystems/Subsystem', blockPath);

    % Clear default contents
    Simulink.SubSystem.deleteContents(blockPath);

    % Add Interpreted MATLAB Fcn for prediction
    add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', ...
        [blockPath '/Predict']);
    set_param([blockPath '/Predict'], 'MATLABFcn', 'stonesoup_kalman_predict_sfun');
    set_param([blockPath '/Predict'], 'Position', [140 45 240 105]);

    % Add Mux for inputs
    add_block('simulink/Signal Routing/Mux', [blockPath '/InputMux']);
    set_param([blockPath '/InputMux'], 'Inputs', '2');
    set_param([blockPath '/InputMux'], 'Position', [80 47 85 103]);

    % Add Demux for outputs
    add_block('simulink/Signal Routing/Demux', [blockPath '/OutputDemux']);
    set_param([blockPath '/OutputDemux'], 'Outputs', '2');
    set_param([blockPath '/OutputDemux'], 'Position', [300 47 305 103]);

    % Add input ports
    add_block('simulink/Sources/In1', [blockPath '/x']);
    set_param([blockPath '/x'], 'Position', [20 53 50 67]);

    add_block('simulink/Sources/In1', [blockPath '/P']);
    set_param([blockPath '/P'], 'Port', '2');
    set_param([blockPath '/P'], 'Position', [20 83 50 97]);

    % Add output ports
    add_block('simulink/Sinks/Out1', [blockPath '/x_pred']);
    set_param([blockPath '/x_pred'], 'Position', [350 53 380 67]);

    add_block('simulink/Sinks/Out1', [blockPath '/P_pred']);
    set_param([blockPath '/P_pred'], 'Port', '2');
    set_param([blockPath '/P_pred'], 'Position', [350 83 380 97]);

    % Connect blocks
    add_line(blockPath, 'x/1', 'InputMux/1');
    add_line(blockPath, 'P/1', 'InputMux/2');
    add_line(blockPath, 'InputMux/1', 'Predict/1');
    add_line(blockPath, 'Predict/1', 'OutputDemux/1');
    add_line(blockPath, 'OutputDemux/1', 'x_pred/1');
    add_line(blockPath, 'OutputDemux/2', 'P_pred/1');

    % Add mask
    maskObj = Simulink.Mask.create(blockPath);
    maskObj.Display = sprintf([...
        'color(''blue'');\n' ...
        'patch([0.1 0.9 0.9 0.1], [0.1 0.1 0.9 0.9], [0.9 0.9 1]);\n' ...
        'text(0.5, 0.65, ''Kalman'', ''HorizontalAlignment'', ''center'', ''FontWeight'', ''bold'');\n' ...
        'text(0.5, 0.35, ''Predictor'', ''HorizontalAlignment'', ''center'');\n' ...
        'port_label(''input'', 1, ''x'');\n' ...
        'port_label(''input'', 2, ''P'');\n' ...
        'port_label(''output'', 1, ''x_{pred}'');\n' ...
        'port_label(''output'', 2, ''P_{pred}'');']);

    maskObj.IconUnits = 'normalized';
    maskObj.Type = 'Kalman Predictor';
    maskObj.Description = 'Kalman filter prediction: x_pred = F*x, P_pred = F*P*F'' + Q';

    maskObj.addParameter('Name', 'F', 'Prompt', 'Transition matrix F:', ...
        'Type', 'edit', 'Value', 'eye(4)', 'Evaluate', 'on');
    maskObj.addParameter('Name', 'Q', 'Prompt', 'Process noise Q:', ...
        'Type', 'edit', 'Value', '0.01*eye(4)', 'Evaluate', 'on');

    set_param(blockPath, 'Position', [30 30 150 110]);
end

function add_kalman_updater(lib)
    % Add Kalman Updater block
    blockPath = [lib '/Kalman Updater'];

    % Create subsystem
    add_block('simulink/Ports & Subsystems/Subsystem', blockPath);
    Simulink.SubSystem.deleteContents(blockPath);

    % Add Interpreted MATLAB Fcn
    add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', ...
        [blockPath '/Update']);
    set_param([blockPath '/Update'], 'MATLABFcn', 'stonesoup_kalman_update_sfun');
    set_param([blockPath '/Update'], 'Position', [160 40 260 120]);

    % Add Mux for inputs
    add_block('simulink/Signal Routing/Mux', [blockPath '/InputMux']);
    set_param([blockPath '/InputMux'], 'Inputs', '3');
    set_param([blockPath '/InputMux'], 'Position', [100 42 105 118]);

    % Add Demux for outputs
    add_block('simulink/Signal Routing/Demux', [blockPath '/OutputDemux']);
    set_param([blockPath '/OutputDemux'], 'Outputs', '2');
    set_param([blockPath '/OutputDemux'], 'Position', [320 55 325 105]);

    % Add input ports
    add_block('simulink/Sources/In1', [blockPath '/x']);
    set_param([blockPath '/x'], 'Position', [20 47 50 63]);

    add_block('simulink/Sources/In1', [blockPath '/P']);
    set_param([blockPath '/P'], 'Port', '2');
    set_param([blockPath '/P'], 'Position', [20 77 50 93]);

    add_block('simulink/Sources/In1', [blockPath '/z']);
    set_param([blockPath '/z'], 'Port', '3');
    set_param([blockPath '/z'], 'Position', [20 107 50 123]);

    % Add output ports
    add_block('simulink/Sinks/Out1', [blockPath '/x_post']);
    set_param([blockPath '/x_post'], 'Position', [380 58 410 72]);

    add_block('simulink/Sinks/Out1', [blockPath '/P_post']);
    set_param([blockPath '/P_post'], 'Port', '2');
    set_param([blockPath '/P_post'], 'Position', [380 88 410 102]);

    % Connect blocks
    add_line(blockPath, 'x/1', 'InputMux/1');
    add_line(blockPath, 'P/1', 'InputMux/2');
    add_line(blockPath, 'z/1', 'InputMux/3');
    add_line(blockPath, 'InputMux/1', 'Update/1');
    add_line(blockPath, 'Update/1', 'OutputDemux/1');
    add_line(blockPath, 'OutputDemux/1', 'x_post/1');
    add_line(blockPath, 'OutputDemux/2', 'P_post/1');

    % Add mask
    maskObj = Simulink.Mask.create(blockPath);
    maskObj.Display = sprintf([...
        'color(''green'');\n' ...
        'patch([0.1 0.9 0.9 0.1], [0.1 0.1 0.9 0.9], [0.9 1 0.9]);\n' ...
        'text(0.5, 0.65, ''Kalman'', ''HorizontalAlignment'', ''center'', ''FontWeight'', ''bold'');\n' ...
        'text(0.5, 0.35, ''Updater'', ''HorizontalAlignment'', ''center'');\n' ...
        'port_label(''input'', 1, ''x'');\n' ...
        'port_label(''input'', 2, ''P'');\n' ...
        'port_label(''input'', 3, ''z'');\n' ...
        'port_label(''output'', 1, ''x_{post}'');\n' ...
        'port_label(''output'', 2, ''P_{post}'');']);

    maskObj.IconUnits = 'normalized';
    maskObj.Type = 'Kalman Updater';
    maskObj.Description = 'Kalman filter update step with measurement';

    maskObj.addParameter('Name', 'H', 'Prompt', 'Measurement matrix H:', ...
        'Type', 'edit', 'Value', '[1 0 0 0; 0 0 1 0]', 'Evaluate', 'on');
    maskObj.addParameter('Name', 'R', 'Prompt', 'Measurement noise R:', ...
        'Type', 'edit', 'Value', '0.1*eye(2)', 'Evaluate', 'on');

    set_param(blockPath, 'Position', [180 30 320 130]);
end

function add_constant_velocity(lib)
    % Add Constant Velocity Model block
    blockPath = [lib '/Constant Velocity Model'];

    % Create subsystem
    add_block('simulink/Ports & Subsystems/Subsystem', blockPath);
    Simulink.SubSystem.deleteContents(blockPath);

    % Add constant blocks
    add_block('simulink/Sources/Constant', [blockPath '/F']);
    set_param([blockPath '/F'], 'Value', 'F_cv(:)');
    set_param([blockPath '/F'], 'Position', [30 30 80 60]);

    add_block('simulink/Sources/Constant', [blockPath '/Q']);
    set_param([blockPath '/Q'], 'Value', 'Q_cv(:)');
    set_param([blockPath '/Q'], 'Position', [30 90 80 120]);

    % Add output ports
    add_block('simulink/Sinks/Out1', [blockPath '/F_out']);
    set_param([blockPath '/F_out'], 'Position', [150 35 180 55]);

    add_block('simulink/Sinks/Out1', [blockPath '/Q_out']);
    set_param([blockPath '/Q_out'], 'Port', '2');
    set_param([blockPath '/Q_out'], 'Position', [150 95 180 115]);

    % Connect
    add_line(blockPath, 'F/1', 'F_out/1');
    add_line(blockPath, 'Q/1', 'Q_out/1');

    % Add mask
    maskObj = Simulink.Mask.create(blockPath);
    maskObj.Display = sprintf([...
        'color(''red'');\n' ...
        'patch([0.1 0.9 0.9 0.1], [0.1 0.1 0.9 0.9], [1 0.9 0.9]);\n' ...
        'text(0.5, 0.65, ''Constant'', ''HorizontalAlignment'', ''center'', ''FontWeight'', ''bold'');\n' ...
        'text(0.5, 0.35, ''Velocity'', ''HorizontalAlignment'', ''center'');\n' ...
        'port_label(''output'', 1, ''F'');\n' ...
        'port_label(''output'', 2, ''Q'');']);

    maskObj.IconUnits = 'normalized';
    maskObj.Type = 'Constant Velocity Model';
    maskObj.Description = 'Generates F and Q for constant velocity motion';

    maskObj.addParameter('Name', 'ndim', 'Prompt', 'Spatial dimensions:', ...
        'Type', 'edit', 'Value', '2', 'Evaluate', 'on');
    maskObj.addParameter('Name', 'dt', 'Prompt', 'Time step (s):', ...
        'Type', 'edit', 'Value', '0.1', 'Evaluate', 'on');
    maskObj.addParameter('Name', 'q', 'Prompt', 'Process noise:', ...
        'Type', 'edit', 'Value', '0.01', 'Evaluate', 'on');

    maskObj.Initialization = sprintf([...
        'state_dim = 2 * ndim;\n' ...
        'F_cv = eye(state_dim);\n' ...
        'for i = 1:ndim, F_cv(2*i-1, 2*i) = dt; end\n' ...
        'Q_block = [dt^3/3, dt^2/2; dt^2/2, dt] * q;\n' ...
        'Q_cv = kron(eye(ndim), Q_block);']);

    set_param(blockPath, 'Position', [30 160 150 240]);
end

function add_gaussian_state(lib)
    % Add Gaussian State block
    blockPath = [lib '/Gaussian State'];

    % Create subsystem
    add_block('simulink/Ports & Subsystems/Subsystem', blockPath);
    Simulink.SubSystem.deleteContents(blockPath);

    % Add constant blocks
    add_block('simulink/Sources/Constant', [blockPath '/x0']);
    set_param([blockPath '/x0'], 'Value', 'x0_init');
    set_param([blockPath '/x0'], 'Position', [30 30 80 60]);

    add_block('simulink/Sources/Constant', [blockPath '/P0']);
    set_param([blockPath '/P0'], 'Value', 'P0_init(:)');
    set_param([blockPath '/P0'], 'Position', [30 90 80 120]);

    % Add output ports
    add_block('simulink/Sinks/Out1', [blockPath '/x0_out']);
    set_param([blockPath '/x0_out'], 'Position', [150 35 180 55]);

    add_block('simulink/Sinks/Out1', [blockPath '/P0_out']);
    set_param([blockPath '/P0_out'], 'Port', '2');
    set_param([blockPath '/P0_out'], 'Position', [150 95 180 115]);

    % Connect
    add_line(blockPath, 'x0/1', 'x0_out/1');
    add_line(blockPath, 'P0/1', 'P0_out/1');

    % Add mask
    maskObj = Simulink.Mask.create(blockPath);
    maskObj.Display = sprintf([...
        'color(''magenta'');\n' ...
        'patch([0.1 0.9 0.9 0.1], [0.1 0.1 0.9 0.9], [1 0.9 1]);\n' ...
        'text(0.5, 0.65, ''Gaussian'', ''HorizontalAlignment'', ''center'', ''FontWeight'', ''bold'');\n' ...
        'text(0.5, 0.35, ''State'', ''HorizontalAlignment'', ''center'');\n' ...
        'port_label(''output'', 1, ''x_0'');\n' ...
        'port_label(''output'', 2, ''P_0'');']);

    maskObj.IconUnits = 'normalized';
    maskObj.Type = 'Gaussian State Source';
    maskObj.Description = 'Initial state and covariance for Kalman filter';

    maskObj.addParameter('Name', 'x0', 'Prompt', 'Initial state x0:', ...
        'Type', 'edit', 'Value', '[0; 0; 0; 0]', 'Evaluate', 'on');
    maskObj.addParameter('Name', 'P0', 'Prompt', 'Initial covariance P0:', ...
        'Type', 'edit', 'Value', 'eye(4)', 'Evaluate', 'on');

    maskObj.Initialization = sprintf('x0_init = x0;\nP0_init = P0;');

    set_param(blockPath, 'Position', [180 160 320 240]);
end
