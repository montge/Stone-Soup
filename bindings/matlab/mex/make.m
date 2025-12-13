function make(target)
% MAKE Build Stone Soup MEX files
%
%   MAKE or MAKE('all')    - Build all MEX files
%   MAKE('clean')          - Remove compiled MEX files
%   MAKE('test')           - Build and run tests
%
% Prerequisites:
%   - libstonesoup compiled and available
%   - MATLAB or Octave with MEX compiler configured
%
% Example:
%   cd bindings/matlab/mex
%   make

if nargin < 1
    target = 'all';
end

% Paths
mex_dir = fileparts(mfilename('fullpath'));
lib_include = fullfile(mex_dir, '..', '..', '..', 'libstonesoup', 'include');
lib_path = fullfile(mex_dir, '..', '..', '..', 'libstonesoup', 'build');

% Check if libstonesoup exists
if ~exist(fullfile(lib_include, 'stonesoup', 'stonesoup.h'), 'file')
    error('stonesoup:notFound', ...
        'libstonesoup headers not found at %s\nBuild libstonesoup first.', lib_include);
end

switch lower(target)
    case 'all'
        fprintf('Building Stone Soup MEX files...\n');

        % Detect if running in Octave
        is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

        if is_octave
            % Build with mkoctfile
            fprintf('Using Octave mkoctfile...\n');
            cmd = sprintf('mkoctfile --mex -I%s -L%s -lstonesoup -o stonesoup_mex stonesoup_mex.c', ...
                lib_include, lib_path);
            system(cmd);
        else
            % Build with MATLAB mex
            fprintf('Using MATLAB mex...\n');
            mex('-v', ...
                ['-I' lib_include], ...
                ['-L' lib_path], ...
                '-lstonesoup', ...
                'stonesoup_mex.c');
        end

        fprintf('Build complete.\n');
        fprintf('Add %s to your path, then run:\n', mex_dir);
        fprintf('  stonesoup_mex(''version'')\n');

    case 'clean'
        fprintf('Cleaning MEX files...\n');
        delete(fullfile(mex_dir, ['stonesoup_mex.' mexext]));
        delete(fullfile(mex_dir, 'stonesoup_mex.mex'));
        delete(fullfile(mex_dir, 'stonesoup_mex.o'));
        fprintf('Clean complete.\n');

    case 'test'
        fprintf('Running tests...\n');
        make('all');

        % Basic functionality test
        fprintf('\nTesting version...\n');
        ver = stonesoup_mex('version');
        fprintf('Version: %s\n', ver);

        fprintf('\nTesting kalman_predict...\n');
        x = [0; 1];
        P = eye(2);
        F = [1, 1; 0, 1];
        Q = 0.1 * eye(2);
        [x_pred, P_pred] = stonesoup_mex('kalman_predict', x, P, F, Q);
        fprintf('x_pred = [%.2f; %.2f]\n', x_pred(1), x_pred(2));

        fprintf('\nTests complete.\n');

    otherwise
        error('stonesoup:invalidTarget', ...
            'Unknown target: %s\nValid targets: all, clean, test', target);
end
end
