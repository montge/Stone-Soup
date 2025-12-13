classdef GaussianState < handle
    % GAUSSIANSTATE Gaussian state with mean and covariance
    %
    %   A GaussianState represents a state distribution characterized by
    %   a mean state vector and covariance matrix.
    %
    % Properties:
    %   state_vector - Mean state vector (column vector)
    %   covariance   - Covariance matrix (symmetric positive definite)
    %   timestamp    - Time stamp (optional)
    %
    % Example:
    %   % Create a 4D Gaussian state
    %   x = [0; 1; 0; 1];  % [x, vx, y, vy]
    %   P = eye(4);
    %   gs = stonesoup.GaussianState(x, P);
    %
    %   % Access state elements
    %   position_x = gs.state_vector(1);
    %   velocity_x = gs.state_vector(2);
    %
    % See also: stonesoup.kalman_predict, stonesoup.kalman_update

    properties
        state_vector  % Mean state vector
        covariance    % Covariance matrix
        timestamp     % Time stamp
    end

    properties (Dependent)
        dim  % State dimension
    end

    methods
        function obj = GaussianState(state_vector, covariance, timestamp)
            % GAUSSIANSTATE Create a Gaussian state
            %
            %   gs = GaussianState(state_vector, covariance)
            %   gs = GaussianState(state_vector, covariance, timestamp)

            if nargin < 2
                error('stonesoup:invalidInput', ...
                    'GaussianState requires state_vector and covariance');
            end

            % Ensure column vector
            obj.state_vector = state_vector(:);

            % Validate covariance dimensions
            dim = length(obj.state_vector);
            if size(covariance, 1) ~= dim || size(covariance, 2) ~= dim
                error('stonesoup:dimensionMismatch', ...
                    'Covariance must be %d x %d', dim, dim);
            end
            obj.covariance = covariance;

            % Set timestamp
            if nargin >= 3
                obj.timestamp = timestamp;
            else
                obj.timestamp = 0;
            end
        end

        function d = get.dim(obj)
            % Get state dimension
            d = length(obj.state_vector);
        end

        function disp(obj)
            % Custom display
            fprintf('GaussianState with properties:\n');
            fprintf('  dim: %d\n', obj.dim);
            fprintf('  state_vector: [%s]''\n', num2str(obj.state_vector', '%.3f '));
            fprintf('  covariance: [%d x %d matrix]\n', obj.dim, obj.dim);
            fprintf('  timestamp: %.3f\n', obj.timestamp);
        end
    end
end
