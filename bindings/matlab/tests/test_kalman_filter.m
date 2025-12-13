%% Test Kalman Filter functions
%
% Unit tests for stonesoup.kalman_predict and stonesoup.kalman_update
%
% Run with: runtests('test_kalman_filter')

function tests = test_kalman_filter
    tests = functiontests(localfunctions);
end

function test_kalman_predict_identity(testCase)
    % Test prediction with identity transition
    x = [1; 2; 3; 4];
    P = eye(4);
    F = eye(4);
    Q = zeros(4);

    gs_prior = stonesoup.GaussianState(x, P);
    gs_pred = stonesoup.kalman_predict(gs_prior, F, Q);

    verifyEqual(testCase, gs_pred.state_vector, x, 'AbsTol', 1e-10);
    verifyEqual(testCase, gs_pred.covariance, P, 'AbsTol', 1e-10);
end

function test_kalman_predict_constant_velocity(testCase)
    % Test prediction with constant velocity model
    dt = 0.1;
    x = [0; 1; 0; 1];  % [x, vx, y, vy]
    P = eye(4);

    F = [1, dt, 0, 0;
         0, 1, 0, 0;
         0, 0, 1, dt;
         0, 0, 0, 1];
    Q = 0.01 * eye(4);

    gs_prior = stonesoup.GaussianState(x, P);
    gs_pred = stonesoup.kalman_predict(gs_prior, F, Q);

    % Expected: x moves by vx*dt, y moves by vy*dt
    x_expected = [dt; 1; dt; 1];
    verifyEqual(testCase, gs_pred.state_vector, x_expected, 'AbsTol', 1e-10);

    % Covariance should increase
    verifyTrue(testCase, trace(gs_pred.covariance) > trace(P));
end

function test_kalman_update_perfect_measurement(testCase)
    % Test update with perfect measurement (R = 0)
    x = [0; 1];
    P = eye(2);
    z = [1];
    H = [1, 0];
    R = 1e-10;  % Near-zero for numerical stability

    gs_pred = stonesoup.GaussianState(x, P);
    gs_post = stonesoup.kalman_update(gs_pred, z, H, R);

    % Position should match measurement
    verifyEqual(testCase, gs_post.state_vector(1), 1, 'AbsTol', 1e-6);
end

function test_kalman_update_high_noise(testCase)
    % Test update with very high measurement noise
    x = [0; 1];
    P = eye(2);
    z = [10];  % Far from prediction
    H = [1, 0];
    R = 1e6;   % Very high noise

    gs_pred = stonesoup.GaussianState(x, P);
    gs_post = stonesoup.kalman_update(gs_pred, z, H, R);

    % State should barely change (measurement is distrusted)
    verifyEqual(testCase, gs_post.state_vector(1), x(1), 'AbsTol', 0.01);
end

function test_kalman_cycle(testCase)
    % Test complete predict-update cycle
    dt = 1;
    F = [1, dt; 0, 1];
    Q = 0.1 * [dt^4/4, dt^3/2; dt^3/2, dt^2];
    H = [1, 0];
    R = 0.5;

    x0 = [0; 1];
    P0 = eye(2);

    gs = stonesoup.GaussianState(x0, P0);

    % Predict
    gs_pred = stonesoup.kalman_predict(gs, F, Q);
    verifyEqual(testCase, gs_pred.state_vector(1), 1, 'AbsTol', 1e-10);  % x = x + v*dt

    % Update with measurement close to prediction
    z = 1.1;
    gs_post = stonesoup.kalman_update(gs_pred, z, H, R);

    % State should be between prediction and measurement
    verifyTrue(testCase, gs_post.state_vector(1) > gs_pred.state_vector(1));
    verifyTrue(testCase, gs_post.state_vector(1) < z);

    % Covariance should decrease after update
    verifyTrue(testCase, gs_post.covariance(1,1) < gs_pred.covariance(1,1));
end

function test_dimension_validation(testCase)
    % Test that dimension mismatches throw errors
    x = [1; 2; 3; 4];
    P = eye(4);
    F_wrong = eye(3);  % Wrong dimension
    Q = eye(4);

    gs = stonesoup.GaussianState(x, P);

    verifyError(testCase, @() stonesoup.kalman_predict(gs, F_wrong, Q), ...
        'stonesoup:dimensionMismatch');
end

function test_joseph_form_stability(testCase)
    % Test that covariance remains positive definite
    % Joseph form should maintain numerical stability

    x = [0; 1; 0; 1];
    P = diag([1e-6, 1, 1e-6, 1]);  % Near-singular

    F = eye(4);
    Q = 1e-8 * eye(4);
    H = [1, 0, 0, 0; 0, 0, 1, 0];
    R = 1e-6 * eye(2);

    gs = stonesoup.GaussianState(x, P);

    % Run many cycles
    for k = 1:100
        gs = stonesoup.kalman_predict(gs, F, Q);
        z = H * gs.state_vector + 0.001 * randn(2, 1);
        gs = stonesoup.kalman_update(gs, z, H, R);

        % Verify positive definiteness
        eigvals = eig(gs.covariance);
        verifyTrue(testCase, all(eigvals > 0), 'Covariance must remain positive definite');
    end
end
