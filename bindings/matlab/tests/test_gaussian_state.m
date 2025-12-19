%% Test GaussianState class
%
% Unit tests for stonesoup.GaussianState
%
% Run with: runtests('test_gaussian_state')

function tests = test_gaussian_state
    tests = functiontests(localfunctions);
end

function test_construction_basic(testCase)
    % Test basic construction
    x = [1; 2; 3; 4];
    P = eye(4);

    gs = stonesoup.GaussianState(x, P);

    verifyEqual(testCase, gs.state_vector, x);
    verifyEqual(testCase, gs.covariance, P);
    verifyEqual(testCase, gs.dim, 4);
    verifyTrue(testCase, isempty(gs.timestamp));
end

function test_construction_with_timestamp(testCase)
    % Test construction with timestamp
    x = [1; 2];
    P = [1, 0.5; 0.5, 1];
    t = 1.5;

    gs = stonesoup.GaussianState(x, P, t);

    verifyEqual(testCase, gs.state_vector, x);
    verifyEqual(testCase, gs.covariance, P);
    verifyEqual(testCase, gs.timestamp, t);
end

function test_construction_row_vector(testCase)
    % Test that row vectors are converted to column
    x = [1, 2, 3];
    P = eye(3);

    gs = stonesoup.GaussianState(x, P);

    verifyEqual(testCase, size(gs.state_vector), [3, 1]);
    verifyEqual(testCase, gs.state_vector, [1; 2; 3]);
end

function test_dimension_mismatch_error(testCase)
    % Test that dimension mismatch throws error
    x = [1; 2; 3];
    P = eye(4);  % Wrong dimension

    verifyError(testCase, @() stonesoup.GaussianState(x, P), 'stonesoup:dimensionMismatch');
end

function test_non_square_covariance_error(testCase)
    % Test that non-square covariance throws error
    x = [1; 2];
    P = [1, 2, 3; 4, 5, 6];  % Not square

    verifyError(testCase, @() stonesoup.GaussianState(x, P), 'stonesoup:invalidCovariance');
end

function test_mean_property(testCase)
    % Test mean alias for state_vector
    x = [1; 2; 3];
    P = eye(3);

    gs = stonesoup.GaussianState(x, P);

    verifyEqual(testCase, gs.mean, x);
end
