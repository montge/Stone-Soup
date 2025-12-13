// Stone Soup Kalman Filter Unit Tests

// Load Stone Soup module
exec(get_absolute_file_path("test_kalman.tst") + "../../loader.sce", -1);

// Test: Create Gaussian state
sv = [0; 0; 0; 0];
P = eye(4, 4);
gs = GaussianState(sv, P);

assert_checkequal(gs_state_vector(gs), sv);
assert_checkequal(gs_covariance(gs), P);
assert_checkequal(gs_dim(gs), 4);

// Test: Constant velocity transition matrix
dt = 1.0;
F = constant_velocity_transition(2, dt);
expected_F = [1, 1, 0, 0;
              0, 1, 0, 0;
              0, 0, 1, 1;
              0, 0, 0, 1];
assert_checkequal(F, expected_F);

// Test: Position measurement matrix
H = position_measurement_matrix(2);
expected_H = [1, 0, 0, 0;
              0, 0, 1, 0];
assert_checkequal(H, expected_H);

// Test: Kalman prediction (pure Scilab implementation for testing)
// x_pred = F * x
// P_pred = F * P * F' + Q
Q = 0.1 * eye(4, 4);
x = [0; 1; 0; 1];  // Position 0, velocity 1 in both dims
P = eye(4, 4);

x_pred_expected = F * x;
P_pred_expected = F * P * F' + Q;

assert_checkalmostequal(x_pred_expected, [1; 1; 1; 1], 1e-10);

// Test: Kalman update (pure Scilab for testing)
H = position_measurement_matrix(2);
R = 0.1 * eye(2, 2);
z = [1.1; 1.2];  // Measurement

// Innovation
y = z - H * x_pred_expected;

// Innovation covariance
S = H * P_pred_expected * H' + R;

// Kalman gain
K = P_pred_expected * H' * inv(S);

// Posterior
x_post = x_pred_expected + K * y;
P_post = (eye(4, 4) - K * H) * P_pred_expected;

// Verify posterior state is between prediction and measurement
assert_checktrue(abs(x_post(1) - 1.0) < abs(z(1) - x_pred_expected(1)));
assert_checktrue(abs(x_post(3) - 1.0) < abs(z(2) - x_pred_expected(3)));

disp("All Kalman filter tests passed!");
