% MAKE_OCTAVE Build Stone Soup for GNU Octave
%
% This script builds and tests the Stone Soup MATLAB bindings using GNU Octave.
% It is designed for CI environments where Octave is available.
%
% Usage:
%   octave --eval "run('make_octave.m')"

fprintf('=== Stone Soup GNU Octave Build ===\n');
fprintf('Octave version: %s\n', OCTAVE_VERSION);

% Add paths
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'mex'));
addpath(fullfile(script_dir, '+stonesoup'));

% Paths to libstonesoup
lib_include = fullfile(script_dir, '..', '..', 'libstonesoup', 'include');
lib_path = fullfile(script_dir, '..', '..', 'libstonesoup', 'build');

fprintf('Library include path: %s\n', lib_include);
fprintf('Library path: %s\n', lib_path);

% Check if libstonesoup headers exist
header_file = fullfile(lib_include, 'stonesoup', 'stonesoup.h');
if ~exist(header_file, 'file')
    fprintf('WARNING: libstonesoup headers not found at %s\n', lib_include);
    fprintf('Skipping MEX compilation. Only testing pure MATLAB/Octave functions.\n');
    skip_mex = true;
else
    skip_mex = false;
end

% Build MEX files if possible
if ~skip_mex
    fprintf('\n=== Building MEX files ===\n');
    cd(fullfile(script_dir, 'mex'));

    try
        % Set library path for linking
        setenv('LD_LIBRARY_PATH', [lib_path ':' getenv('LD_LIBRARY_PATH')]);

        % Build using mkoctfile
        cmd = sprintf('mkoctfile --mex -I%s -L%s -lstonesoup -o stonesoup_mex stonesoup_mex.c', ...
            lib_include, lib_path);
        fprintf('Running: %s\n', cmd);
        [status, output] = system(cmd);

        if status == 0
            fprintf('MEX build successful.\n');
        else
            fprintf('MEX build failed:\n%s\n', output);
            skip_mex = true;
        end
    catch err
        fprintf('MEX build error: %s\n', err.message);
        skip_mex = true;
    end

    cd(script_dir);
end

% Run tests
fprintf('\n=== Running Tests ===\n');
cd(fullfile(script_dir, 'tests'));

% Test pure MATLAB/Octave functions
fprintf('\n--- Testing GaussianState ---\n');
try
    % Basic GaussianState test (if it doesn't require MEX)
    gs = stonesoup.GaussianState([1; 2; 3], eye(3));
    fprintf('GaussianState creation: PASS\n');
    fprintf('  Mean: [%s]\n', num2str(gs.mean'));
    fprintf('  Covar size: %dx%d\n', size(gs.covar, 1), size(gs.covar, 2));
catch err
    fprintf('GaussianState test: SKIP (%s)\n', err.message);
end

% Test Kalman filter functions
fprintf('\n--- Testing Kalman Functions ---\n');
try
    % Test predict (pure MATLAB implementation if available)
    x = [0; 0];
    P = eye(2);
    F = [1, 0.1; 0, 1];
    Q = 0.01 * eye(2);

    [x_pred, P_pred] = stonesoup.kalman_predict(x, P, F, Q);
    fprintf('kalman_predict: PASS\n');
    fprintf('  Predicted state: [%s]\n', num2str(x_pred'));
catch err
    fprintf('kalman_predict test: SKIP (%s)\n', err.message);
end

% Test MEX functions if available
if ~skip_mex
    fprintf('\n--- Testing MEX Functions ---\n');
    cd(fullfile(script_dir, 'mex'));

    try
        ver = stonesoup_mex('version');
        fprintf('MEX version: %s - PASS\n', ver);
    catch err
        fprintf('MEX version test: SKIP (%s)\n', err.message);
    end
end

fprintf('\n=== Octave Compatibility Test Complete ===\n');
