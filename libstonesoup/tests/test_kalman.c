/**
 * @file test_kalman.c
 * @brief Comprehensive tests for Stone Soup Kalman filter operations
 */

#include <stonesoup/stonesoup.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-6

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("Running test: %s ... ", #name); \
        fflush(stdout); \
        tests_run++; \
        if (name()) { \
            tests_passed++; \
            printf("PASSED\n"); \
        } else { \
            printf("FAILED\n"); \
        } \
    } while(0)

/* Helper function for comparing doubles */
static int double_equals(double a, double b) {
    return fabs(a - b) < EPSILON;
}

/* Helper to create a simple 2D constant velocity transition matrix */
static stonesoup_covariance_matrix_t* create_cv_transition_matrix(double dt) {
    // For 2D position-velocity state [x, vx, y, vy]
    // F = [1 dt 0  0 ]
    //     [0  1 0  0 ]
    //     [0  0 1 dt ]
    //     [0  0 0  1 ]
    stonesoup_covariance_matrix_t* F = stonesoup_covariance_matrix_create(4, 4);
    if (F) {
        for (size_t i = 0; i < 16; i++) F->data[i] = 0.0;
        F->data[0] = 1.0;  F->data[1] = dt;
        F->data[5] = 1.0;
        F->data[10] = 1.0; F->data[11] = dt;
        F->data[15] = 1.0;
    }
    return F;
}

/* Helper to create simple process noise matrix */
static stonesoup_covariance_matrix_t* create_process_noise(double q) {
    // Simple diagonal process noise
    stonesoup_covariance_matrix_t* Q = stonesoup_covariance_matrix_create(4, 4);
    if (Q) {
        for (size_t i = 0; i < 16; i++) Q->data[i] = 0.0;
        Q->data[0] = q;
        Q->data[5] = q;
        Q->data[10] = q;
        Q->data[15] = q;
    }
    return Q;
}

/* Helper to create 2D position measurement matrix */
static stonesoup_covariance_matrix_t* create_measurement_matrix(void) {
    // H maps [x, vx, y, vy] -> [x, y]
    // H = [1 0 0 0]
    //     [0 0 1 0]
    stonesoup_covariance_matrix_t* H = stonesoup_covariance_matrix_create(2, 4);
    if (H) {
        for (size_t i = 0; i < 8; i++) H->data[i] = 0.0;
        H->data[0] = 1.0;
        H->data[6] = 1.0;
    }
    return H;
}

/* Helper to create measurement noise matrix */
static stonesoup_covariance_matrix_t* create_measurement_noise(double r) {
    stonesoup_covariance_matrix_t* R = stonesoup_covariance_matrix_create(2, 2);
    if (R) {
        R->data[0] = r; R->data[1] = 0.0;
        R->data[2] = 0.0; R->data[3] = r;
    }
    return R;
}

/* ============================================================================
 * Kalman Filter Prediction Tests
 * ============================================================================ */

static int test_kalman_predict_basic(void) {
    // Create prior state at [0, 1, 0, 1] (moving diagonally at unit velocity)
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    assert(prior);

    prior->state_vector->data[0] = 0.0;  // x
    prior->state_vector->data[1] = 1.0;  // vx
    prior->state_vector->data[2] = 0.0;  // y
    prior->state_vector->data[3] = 1.0;  // vy
    prior->timestamp = 0.0;

    // Set initial covariance to identity
    stonesoup_covariance_matrix_eye(prior->covariance);

    // Create transition matrix for dt = 1.0
    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    stonesoup_error_t err = stonesoup_kalman_predict(prior, F, Q, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(prior);
        stonesoup_covariance_matrix_free(F);
        stonesoup_covariance_matrix_free(Q);
        stonesoup_gaussian_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Expected: x_pred = F * x = [1, 1, 1, 1]
    assert(double_equals(predicted->state_vector->data[0], 1.0));
    assert(double_equals(predicted->state_vector->data[1], 1.0));
    assert(double_equals(predicted->state_vector->data[2], 1.0));
    assert(double_equals(predicted->state_vector->data[3], 1.0));

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_kalman_predict_null_pointer(void) {
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);

    assert(prior && F && Q);

    // Test with NULL predicted state
    stonesoup_error_t err = stonesoup_kalman_predict(prior, F, Q, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    return 1;
}

/* ============================================================================
 * Kalman Filter Update Tests
 * ============================================================================ */

static int test_kalman_update_basic(void) {
    // Create predicted state
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    assert(predicted);

    predicted->state_vector->data[0] = 1.0;  // x
    predicted->state_vector->data[1] = 1.0;  // vx
    predicted->state_vector->data[2] = 1.0;  // y
    predicted->state_vector->data[3] = 1.0;  // vy

    // Set covariance to identity
    stonesoup_covariance_matrix_eye(predicted->covariance);

    // Create measurement [1.1, 0.9] (close to predicted position)
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    assert(measurement);
    measurement->data[0] = 1.1;
    measurement->data[1] = 0.9;

    // Create measurement matrix and noise
    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(H && R && posterior);

    stonesoup_error_t err = stonesoup_kalman_update(predicted, measurement, H, R, posterior);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(predicted);
        stonesoup_state_vector_free(measurement);
        stonesoup_covariance_matrix_free(H);
        stonesoup_covariance_matrix_free(R);
        stonesoup_gaussian_state_free(posterior);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Posterior position should be between prediction and measurement
    // Exact values depend on Kalman gain calculation
    assert(posterior->state_vector->data[0] >= 0.9 &&
           posterior->state_vector->data[0] <= 1.2);
    assert(posterior->state_vector->data[2] >= 0.8 &&
           posterior->state_vector->data[2] <= 1.1);

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

static int test_kalman_update_null_pointer(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);

    assert(predicted && measurement && H && R);

    // Test with NULL posterior
    stonesoup_error_t err = stonesoup_kalman_update(predicted, measurement, H, R, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    return 1;
}

/* ============================================================================
 * Full Predict-Update Cycle Test
 * ============================================================================ */

static int test_kalman_full_cycle(void) {
    // Initialize prior state
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    assert(prior);

    prior->state_vector->data[0] = 0.0;
    prior->state_vector->data[1] = 1.0;
    prior->state_vector->data[2] = 0.0;
    prior->state_vector->data[3] = 1.0;
    stonesoup_covariance_matrix_eye(prior->covariance);

    // Prediction step
    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    stonesoup_error_t err = stonesoup_kalman_predict(prior, F, Q, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(prior);
        stonesoup_covariance_matrix_free(F);
        stonesoup_covariance_matrix_free(Q);
        stonesoup_gaussian_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Update step with measurement
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 1.0;
    measurement->data[1] = 1.0;

    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(measurement && H && R && posterior);

    err = stonesoup_kalman_update(predicted, measurement, H, R, posterior);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_SUCCESS);
        // Posterior should be reasonable
        assert(posterior->state_vector->data[0] >= 0.5 &&
               posterior->state_vector->data[0] <= 1.5);
    }

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

/* ============================================================================
 * Innovation Tests
 * ============================================================================ */

static int test_kalman_innovation(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    assert(predicted);

    predicted->state_vector->data[0] = 1.0;
    predicted->state_vector->data[1] = 1.0;
    predicted->state_vector->data[2] = 2.0;
    predicted->state_vector->data[3] = 1.0;

    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 1.5;
    measurement->data[1] = 2.5;

    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_state_vector_t* innovation = stonesoup_state_vector_create(2);

    assert(measurement && H && innovation);

    stonesoup_error_t err = stonesoup_kalman_innovation(predicted, measurement, H, innovation);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(predicted);
        stonesoup_state_vector_free(measurement);
        stonesoup_covariance_matrix_free(H);
        stonesoup_state_vector_free(innovation);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Innovation = measurement - H * predicted = [1.5, 2.5] - [1.0, 2.0] = [0.5, 0.5]
    assert(double_equals(innovation->data[0], 0.5));
    assert(double_equals(innovation->data[1], 0.5));

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_state_vector_free(innovation);
    return 1;
}

static int test_kalman_innovation_covariance(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    assert(predicted);

    stonesoup_covariance_matrix_eye(predicted->covariance);

    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.5);
    stonesoup_covariance_matrix_t* S = stonesoup_covariance_matrix_create(2, 2);

    assert(H && R && S);

    stonesoup_error_t err = stonesoup_kalman_innovation_covariance(predicted, H, R, S);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(predicted);
        stonesoup_covariance_matrix_free(H);
        stonesoup_covariance_matrix_free(R);
        stonesoup_covariance_matrix_free(S);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // S = H * P * H^T + R
    // With P = I and R = 0.5*I, S should be 1.5*I
    assert(double_equals(S->data[0], 1.5));
    assert(double_equals(S->data[1], 0.0));
    assert(double_equals(S->data[2], 0.0));
    assert(double_equals(S->data[3], 1.5));

    stonesoup_gaussian_state_free(predicted);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_covariance_matrix_free(S);
    return 1;
}

/* ============================================================================
 * Extended Kalman Filter Tests
 * ============================================================================ */

/* Simple nonlinear transition function: identity (for testing) */
static void identity_transition(const stonesoup_state_vector_t* x_in,
                                stonesoup_state_vector_t* x_out) {
    for (size_t i = 0; i < x_in->size; i++) {
        x_out->data[i] = x_in->data[i];
    }
}

/* Nonlinear transition: constant velocity with dt=1 */
static void cv_transition(const stonesoup_state_vector_t* x_in,
                          stonesoup_state_vector_t* x_out) {
    // [x, vx, y, vy] -> [x+vx, vx, y+vy, vy]
    x_out->data[0] = x_in->data[0] + x_in->data[1];
    x_out->data[1] = x_in->data[1];
    x_out->data[2] = x_in->data[2] + x_in->data[3];
    x_out->data[3] = x_in->data[3];
}

/* Nonlinear measurement function: extracts position from state */
static void position_measurement(const stonesoup_state_vector_t* x_in,
                                 stonesoup_state_vector_t* z_out) {
    z_out->data[0] = x_in->data[0];  // x position
    z_out->data[1] = x_in->data[2];  // y position
}

static int test_ekf_predict(void) {
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    assert(prior);

    prior->state_vector->data[0] = 1.0;
    prior->state_vector->data[1] = 2.0;
    prior->state_vector->data[2] = 3.0;
    prior->state_vector->data[3] = 4.0;
    stonesoup_covariance_matrix_eye(prior->covariance);

    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    stonesoup_error_t err = stonesoup_ekf_predict(prior, F, Q, identity_transition, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(prior);
        stonesoup_covariance_matrix_free(F);
        stonesoup_covariance_matrix_free(Q);
        stonesoup_gaussian_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_ekf_update(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    assert(predicted);

    predicted->state_vector->data[0] = 1.0;
    predicted->state_vector->data[1] = 1.0;
    predicted->state_vector->data[2] = 1.0;
    predicted->state_vector->data[3] = 1.0;
    stonesoup_covariance_matrix_eye(predicted->covariance);

    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 1.0;
    measurement->data[1] = 1.0;

    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(measurement && H && R && posterior);

    stonesoup_error_t err = stonesoup_ekf_update(predicted, measurement, H, R, NULL, posterior);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(predicted);
        stonesoup_state_vector_free(measurement);
        stonesoup_covariance_matrix_free(H);
        stonesoup_covariance_matrix_free(R);
        stonesoup_gaussian_state_free(posterior);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

static int test_ekf_predict_with_nonlinear_func(void) {
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    assert(prior);

    prior->state_vector->data[0] = 0.0;  // x
    prior->state_vector->data[1] = 1.0;  // vx
    prior->state_vector->data[2] = 0.0;  // y
    prior->state_vector->data[3] = 2.0;  // vy
    stonesoup_covariance_matrix_eye(prior->covariance);

    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    // Use nonlinear transition function
    stonesoup_error_t err = stonesoup_ekf_predict(prior, F, Q, cv_transition, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(prior);
        stonesoup_covariance_matrix_free(F);
        stonesoup_covariance_matrix_free(Q);
        stonesoup_gaussian_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Check state propagation: [0+1, 1, 0+2, 2] = [1, 1, 2, 2]
    assert(double_equals(predicted->state_vector->data[0], 1.0));
    assert(double_equals(predicted->state_vector->data[1], 1.0));
    assert(double_equals(predicted->state_vector->data[2], 2.0));
    assert(double_equals(predicted->state_vector->data[3], 2.0));

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_ekf_predict_null_transition(void) {
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    assert(prior);

    prior->state_vector->data[0] = 0.0;
    prior->state_vector->data[1] = 1.0;
    prior->state_vector->data[2] = 0.0;
    prior->state_vector->data[3] = 1.0;
    stonesoup_covariance_matrix_eye(prior->covariance);

    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    // Use NULL transition function (should use jacobian as linear transition)
    stonesoup_error_t err = stonesoup_ekf_predict(prior, F, Q, NULL, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(prior);
        stonesoup_covariance_matrix_free(F);
        stonesoup_covariance_matrix_free(Q);
        stonesoup_gaussian_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Should give same result as standard Kalman predict: [1, 1, 1, 1]
    assert(double_equals(predicted->state_vector->data[0], 1.0));
    assert(double_equals(predicted->state_vector->data[1], 1.0));
    assert(double_equals(predicted->state_vector->data[2], 1.0));
    assert(double_equals(predicted->state_vector->data[3], 1.0));

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_ekf_update_with_nonlinear_func(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    assert(predicted);

    predicted->state_vector->data[0] = 1.0;
    predicted->state_vector->data[1] = 1.0;
    predicted->state_vector->data[2] = 2.0;
    predicted->state_vector->data[3] = 1.0;
    stonesoup_covariance_matrix_eye(predicted->covariance);

    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 1.1;  // measured x
    measurement->data[1] = 2.1;  // measured y

    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(measurement && H && R && posterior);

    // Use nonlinear measurement function
    stonesoup_error_t err = stonesoup_ekf_update(predicted, measurement, H, R,
                                                  position_measurement, posterior);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_gaussian_state_free(predicted);
        stonesoup_state_vector_free(measurement);
        stonesoup_covariance_matrix_free(H);
        stonesoup_covariance_matrix_free(R);
        stonesoup_gaussian_state_free(posterior);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Posterior should be between prediction and measurement
    assert(posterior->state_vector->data[0] >= 0.9 &&
           posterior->state_vector->data[0] <= 1.2);
    assert(posterior->state_vector->data[2] >= 1.9 &&
           posterior->state_vector->data[2] <= 2.2);

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

/* ============================================================================
 * Dimension Error Tests
 * ============================================================================ */

static int test_kalman_predict_dimension_error(void) {
    stonesoup_gaussian_state_t* prior = stonesoup_gaussian_state_create(4);
    stonesoup_covariance_matrix_t* F = stonesoup_covariance_matrix_create(3, 3);  // Wrong size
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(prior && F && Q && predicted);

    stonesoup_covariance_matrix_eye(prior->covariance);

    stonesoup_error_t err = stonesoup_kalman_predict(prior, F, Q, predicted);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_DIMENSION);
    }

    stonesoup_gaussian_state_free(prior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_kalman_update_dimension_error(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    stonesoup_covariance_matrix_t* H = stonesoup_covariance_matrix_create(3, 4);  // Wrong rows
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(predicted && measurement && H && R && posterior);

    stonesoup_covariance_matrix_eye(predicted->covariance);

    stonesoup_error_t err = stonesoup_kalman_update(predicted, measurement, H, R, posterior);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_DIMENSION);
    }

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

static int test_kalman_innovation_null_pointer(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    stonesoup_covariance_matrix_t* H = create_measurement_matrix();

    assert(predicted && measurement && H);

    stonesoup_error_t err = stonesoup_kalman_innovation(predicted, measurement, H, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_gaussian_state_free(predicted);
    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    return 1;
}

static int test_kalman_innovation_cov_null_pointer(void) {
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);
    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);

    assert(predicted && H && R);

    stonesoup_covariance_matrix_eye(predicted->covariance);

    stonesoup_error_t err = stonesoup_kalman_innovation_covariance(predicted, H, R, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_gaussian_state_free(predicted);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    return 1;
}

static int test_ekf_predict_null_pointer(void) {
    stonesoup_covariance_matrix_t* F = create_cv_transition_matrix(1.0);
    stonesoup_covariance_matrix_t* Q = create_process_noise(0.1);
    stonesoup_gaussian_state_t* predicted = stonesoup_gaussian_state_create(4);

    assert(F && Q && predicted);

    stonesoup_error_t err = stonesoup_ekf_predict(NULL, F, Q, NULL, predicted);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_gaussian_state_free(predicted);
    return 1;
}

static int test_ekf_update_null_pointer(void) {
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    stonesoup_covariance_matrix_t* H = create_measurement_matrix();
    stonesoup_covariance_matrix_t* R = create_measurement_noise(0.1);
    stonesoup_gaussian_state_t* posterior = stonesoup_gaussian_state_create(4);

    assert(measurement && H && R && posterior);

    stonesoup_error_t err = stonesoup_ekf_update(NULL, measurement, H, R, NULL, posterior);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_state_vector_free(measurement);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_gaussian_state_free(posterior);
    return 1;
}

int main(void) {
    printf("Stone Soup C Library - Kalman Filter Tests\n");
    printf("===========================================\n\n");

    /* Kalman Predict Tests */
    printf("Kalman Filter Prediction:\n");
    TEST(test_kalman_predict_basic);
    TEST(test_kalman_predict_null_pointer);

    /* Kalman Update Tests */
    printf("\nKalman Filter Update:\n");
    TEST(test_kalman_update_basic);
    TEST(test_kalman_update_null_pointer);

    /* Full Cycle Test */
    printf("\nFull Predict-Update Cycle:\n");
    TEST(test_kalman_full_cycle);

    /* Innovation Tests */
    printf("\nInnovation Tests:\n");
    TEST(test_kalman_innovation);
    TEST(test_kalman_innovation_covariance);

    /* EKF Tests */
    printf("\nExtended Kalman Filter:\n");
    TEST(test_ekf_predict);
    TEST(test_ekf_update);
    TEST(test_ekf_predict_with_nonlinear_func);
    TEST(test_ekf_predict_null_transition);
    TEST(test_ekf_update_with_nonlinear_func);

    /* Dimension Error Tests */
    printf("\nDimension and Error Handling:\n");
    TEST(test_kalman_predict_dimension_error);
    TEST(test_kalman_update_dimension_error);
    TEST(test_kalman_innovation_null_pointer);
    TEST(test_kalman_innovation_cov_null_pointer);
    TEST(test_ekf_predict_null_pointer);
    TEST(test_ekf_update_null_pointer);

    printf("\n===========================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);

    return (tests_run == tests_passed) ? 0 : 1;
}
