% STONESOUP - MATLAB bindings for the Stone Soup tracking framework
%
% Stone Soup is a framework for target tracking and state estimation.
% This package provides MATLAB interfaces to Stone Soup functionality.
%
% Version 0.1.0
%
% Getting Started:
%   - Initialize the library with stonesoup.initialize()
%   - Clean up with stonesoup.cleanup()
%
% Core Classes:
%   StateVector      - Represents a state vector in n-dimensional space
%   GaussianState    - Gaussian state with mean and covariance
%   Detection        - Sensor detection/measurement
%   Track            - Target track over time
%
% Tracking Components:
%   TransitionModel  - State transition models (e.g., constant velocity)
%   MeasurementModel - Measurement/observation models
%   Predictor        - Predict future states
%   Updater          - Update predictions with measurements
%   Tracker          - Complete tracking system
%
% Examples:
%   % Initialize the library
%   stonesoup.initialize();
%
%   % Create a state vector
%   sv = stonesoup.StateVector([1.0; 2.0; 3.0; 4.0]);
%
%   % Create a Gaussian state
%   covar = eye(4);
%   gs = stonesoup.GaussianState(sv, covar);
%
%   % Clean up when done
%   stonesoup.cleanup();
%
% Requirements:
%   - MATLAB R2021b or later
%   - Stone Soup native library (libstonesoup)
%
% For more information, see:
%   https://github.com/dstl/Stone-Soup
%
% Copyright (c) Stone Soup Contributors
% Licensed under MIT License

% Version information
function ver = version()
    ver = '0.1.0';
end
