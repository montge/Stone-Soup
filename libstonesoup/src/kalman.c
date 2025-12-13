/**
 * @file kalman.c
 * @brief Implementation of Kalman filter operations
 */

#include "stonesoup/kalman.h"
#include "stonesoup/matrix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

stonesoup_error_t stonesoup_kalman_predict(
    const stonesoup_gaussian_state_t* prior,
    const stonesoup_covariance_matrix_t* transition_matrix,
    const stonesoup_covariance_matrix_t* process_noise,
    stonesoup_gaussian_state_t* predicted) {

    if (!prior || !transition_matrix || !process_noise || !predicted) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!prior->state_vector || !prior->covariance ||
        !predicted->state_vector || !predicted->covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = prior->state_vector->size;

    // Check dimensions
    if (transition_matrix->rows != state_dim || transition_matrix->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (process_noise->rows != state_dim || process_noise->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (predicted->state_vector->size != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (predicted->covariance->rows != state_dim || predicted->covariance->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // x_pred = F * x
    err = stonesoup_matrix_vector_multiply(transition_matrix, prior->state_vector,
                                           predicted->state_vector);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    // P_pred = F * P * F^T + Q
    // First compute F * P
    stonesoup_covariance_matrix_t* FP = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!FP) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply(transition_matrix, prior->covariance, FP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(FP);
        return err;
    }

    // Now compute F * P * F^T
    stonesoup_covariance_matrix_t* FPFt = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!FPFt) {
        stonesoup_covariance_matrix_free(FP);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply_A_Bt(FP, transition_matrix, FPFt);
    stonesoup_covariance_matrix_free(FP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(FPFt);
        return err;
    }

    // Finally add Q: P_pred = F * P * F^T + Q
    err = stonesoup_matrix_add(FPFt, process_noise, predicted->covariance);
    stonesoup_covariance_matrix_free(FPFt);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    // Copy timestamp
    predicted->timestamp = prior->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_kalman_update(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    const stonesoup_covariance_matrix_t* measurement_noise,
    stonesoup_gaussian_state_t* posterior) {

    if (!predicted || !measurement || !measurement_matrix ||
        !measurement_noise || !posterior) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!predicted->state_vector || !predicted->covariance ||
        !posterior->state_vector || !posterior->covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = predicted->state_vector->size;
    size_t meas_dim = measurement->size;

    // Check dimensions
    if (measurement_matrix->rows != meas_dim || measurement_matrix->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (measurement_noise->rows != meas_dim || measurement_noise->cols != meas_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (posterior->state_vector->size != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (posterior->covariance->rows != state_dim || posterior->covariance->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // y = z - H * x_pred (innovation)
    stonesoup_state_vector_t* innovation = stonesoup_state_vector_create(meas_dim);
    if (!innovation) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_kalman_innovation(predicted, measurement, measurement_matrix, innovation);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        return err;
    }

    // S = H * P_pred * H^T + R (innovation covariance)
    stonesoup_covariance_matrix_t* S = stonesoup_covariance_matrix_create(meas_dim, meas_dim);
    if (!S) {
        stonesoup_state_vector_free(innovation);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_kalman_innovation_covariance(predicted, measurement_matrix,
                                                  measurement_noise, S);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        return err;
    }

    // K = P_pred * H^T * S^-1 (Kalman gain)
    // Since K = PHt * S^-1, we solve S * K^T = PHt^T for K^T, then transpose
    // First compute P_pred * H^T
    stonesoup_covariance_matrix_t* PHt = stonesoup_covariance_matrix_create(state_dim, meas_dim);
    if (!PHt) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply_A_Bt(predicted->covariance, measurement_matrix, PHt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt);
        return err;
    }

    // Transpose PHt to get PHt^T (meas_dim x state_dim)
    stonesoup_covariance_matrix_t* PHt_T = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    if (!PHt_T) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_transpose(PHt, PHt_T);
    stonesoup_covariance_matrix_free(PHt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt_T);
        return err;
    }

    // Solve S * Kt = PHt^T for Kt (K transposed)
    // Kt is (meas_dim x state_dim)
    stonesoup_covariance_matrix_t* Kt = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    if (!Kt) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt_T);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_cholesky_solve(S, PHt_T, Kt);
    stonesoup_covariance_matrix_free(S);
    stonesoup_covariance_matrix_free(PHt_T);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(Kt);
        return err;
    }

    // Transpose Kt to get K (state_dim x meas_dim)
    stonesoup_covariance_matrix_t* K = stonesoup_covariance_matrix_create(state_dim, meas_dim);
    if (!K) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(Kt);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_transpose(Kt, K);
    stonesoup_covariance_matrix_free(Kt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(K);
        return err;
    }

    // x_post = x_pred + K * y
    stonesoup_state_vector_t* Ky = stonesoup_state_vector_create(state_dim);
    if (!Ky) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(K);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_vector_multiply(K, innovation, Ky);
    stonesoup_state_vector_free(innovation);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(K);
        stonesoup_state_vector_free(Ky);
        return err;
    }

    err = stonesoup_vector_add(predicted->state_vector, Ky, posterior->state_vector);
    stonesoup_state_vector_free(Ky);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(K);
        return err;
    }

    // P_post = (I - K * H) * P_pred
    // First compute K * H
    stonesoup_covariance_matrix_t* KH = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!KH) {
        stonesoup_covariance_matrix_free(K);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply(K, measurement_matrix, KH);
    stonesoup_covariance_matrix_free(K);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(KH);
        return err;
    }

    // Create identity matrix I
    stonesoup_covariance_matrix_t* I = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!I) {
        stonesoup_covariance_matrix_free(KH);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_covariance_matrix_eye(I);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(KH);
        stonesoup_covariance_matrix_free(I);
        return err;
    }

    // Compute I - K^T * H
    stonesoup_covariance_matrix_t* I_KH = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!I_KH) {
        stonesoup_covariance_matrix_free(KH);
        stonesoup_covariance_matrix_free(I);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_subtract(I, KH, I_KH);
    stonesoup_covariance_matrix_free(I);
    stonesoup_covariance_matrix_free(KH);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(I_KH);
        return err;
    }

    // Finally compute P_post = (I - K^T * H) * P_pred
    err = stonesoup_matrix_multiply(I_KH, predicted->covariance, posterior->covariance);
    stonesoup_covariance_matrix_free(I_KH);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    // Copy timestamp
    posterior->timestamp = predicted->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_ekf_predict(
    const stonesoup_gaussian_state_t* prior,
    const stonesoup_covariance_matrix_t* jacobian,
    const stonesoup_covariance_matrix_t* process_noise,
    void (*transition_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    stonesoup_gaussian_state_t* predicted) {

    if (!prior || !jacobian || !process_noise || !predicted) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!prior->state_vector || !prior->covariance ||
        !predicted->state_vector || !predicted->covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = prior->state_vector->size;

    // Check dimensions
    if (jacobian->rows != state_dim || jacobian->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (process_noise->rows != state_dim || process_noise->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (predicted->state_vector->size != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (predicted->covariance->rows != state_dim || predicted->covariance->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // State propagation: x_pred = f(x) or F * x if no transition function
    if (transition_func != NULL) {
        // Use nonlinear transition function
        transition_func(prior->state_vector, predicted->state_vector);
    } else {
        // Use jacobian as linear transition matrix
        err = stonesoup_matrix_vector_multiply(jacobian, prior->state_vector,
                                               predicted->state_vector);
        if (err != STONESOUP_SUCCESS) {
            return err;
        }
    }

    // Covariance propagation: P_pred = F * P * F^T + Q (F is jacobian)
    // This is identical to standard Kalman prediction with F = jacobian

    // First compute F * P
    stonesoup_covariance_matrix_t* FP = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!FP) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply(jacobian, prior->covariance, FP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(FP);
        return err;
    }

    // Now compute F * P * F^T
    stonesoup_covariance_matrix_t* FPFt = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!FPFt) {
        stonesoup_covariance_matrix_free(FP);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply_A_Bt(FP, jacobian, FPFt);
    stonesoup_covariance_matrix_free(FP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(FPFt);
        return err;
    }

    // Finally add Q: P_pred = F * P * F^T + Q
    err = stonesoup_matrix_add(FPFt, process_noise, predicted->covariance);
    stonesoup_covariance_matrix_free(FPFt);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    // Copy timestamp
    predicted->timestamp = prior->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_ekf_update(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* jacobian,
    const stonesoup_covariance_matrix_t* measurement_noise,
    void (*measurement_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    stonesoup_gaussian_state_t* posterior) {

    if (!predicted || !measurement || !jacobian ||
        !measurement_noise || !posterior) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!predicted->state_vector || !predicted->covariance ||
        !posterior->state_vector || !posterior->covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = predicted->state_vector->size;
    size_t meas_dim = measurement->size;

    // Check dimensions
    if (jacobian->rows != meas_dim || jacobian->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (measurement_noise->rows != meas_dim || measurement_noise->cols != meas_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (posterior->state_vector->size != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (posterior->covariance->rows != state_dim || posterior->covariance->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // Compute innovation: y = z - h(x_pred) or y = z - H * x_pred
    stonesoup_state_vector_t* innovation = stonesoup_state_vector_create(meas_dim);
    if (!innovation) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    if (measurement_func != NULL) {
        // Use nonlinear measurement function: y = z - h(x_pred)
        stonesoup_state_vector_t* predicted_meas = stonesoup_state_vector_create(meas_dim);
        if (!predicted_meas) {
            stonesoup_state_vector_free(innovation);
            return STONESOUP_ERROR_ALLOCATION;
        }

        measurement_func(predicted->state_vector, predicted_meas);
        err = stonesoup_vector_subtract(measurement, predicted_meas, innovation);
        stonesoup_state_vector_free(predicted_meas);
        if (err != STONESOUP_SUCCESS) {
            stonesoup_state_vector_free(innovation);
            return err;
        }
    } else {
        // Use jacobian as linear measurement matrix
        err = stonesoup_kalman_innovation(predicted, measurement, jacobian, innovation);
        if (err != STONESOUP_SUCCESS) {
            stonesoup_state_vector_free(innovation);
            return err;
        }
    }

    // S = H * P_pred * H^T + R (innovation covariance, H is jacobian)
    stonesoup_covariance_matrix_t* S = stonesoup_covariance_matrix_create(meas_dim, meas_dim);
    if (!S) {
        stonesoup_state_vector_free(innovation);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_kalman_innovation_covariance(predicted, jacobian,
                                                  measurement_noise, S);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        return err;
    }

    // K = P_pred * H^T * S^-1 (Kalman gain)
    // Since K = PHt * S^-1, we solve S * K^T = PHt^T for K^T, then transpose
    // First compute P_pred * H^T
    stonesoup_covariance_matrix_t* PHt = stonesoup_covariance_matrix_create(state_dim, meas_dim);
    if (!PHt) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply_A_Bt(predicted->covariance, jacobian, PHt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt);
        return err;
    }

    // Transpose PHt to get PHt^T (meas_dim x state_dim)
    stonesoup_covariance_matrix_t* PHt_T = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    if (!PHt_T) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_transpose(PHt, PHt_T);
    stonesoup_covariance_matrix_free(PHt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt_T);
        return err;
    }

    // Solve S * Kt = PHt^T for Kt (K transposed)
    // Kt is (meas_dim x state_dim)
    stonesoup_covariance_matrix_t* Kt = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    if (!Kt) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(S);
        stonesoup_covariance_matrix_free(PHt_T);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_cholesky_solve(S, PHt_T, Kt);
    stonesoup_covariance_matrix_free(S);
    stonesoup_covariance_matrix_free(PHt_T);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(Kt);
        return err;
    }

    // Transpose Kt to get K (state_dim x meas_dim)
    stonesoup_covariance_matrix_t* K = stonesoup_covariance_matrix_create(state_dim, meas_dim);
    if (!K) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(Kt);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_transpose(Kt, K);
    stonesoup_covariance_matrix_free(Kt);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(K);
        return err;
    }

    // x_post = x_pred + K * y
    stonesoup_state_vector_t* Ky = stonesoup_state_vector_create(state_dim);
    if (!Ky) {
        stonesoup_state_vector_free(innovation);
        stonesoup_covariance_matrix_free(K);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_vector_multiply(K, innovation, Ky);
    stonesoup_state_vector_free(innovation);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(K);
        stonesoup_state_vector_free(Ky);
        return err;
    }

    err = stonesoup_vector_add(predicted->state_vector, Ky, posterior->state_vector);
    stonesoup_state_vector_free(Ky);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(K);
        return err;
    }

    // P_post = (I - K * H) * P_pred
    // First compute K * H
    stonesoup_covariance_matrix_t* KH = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!KH) {
        stonesoup_covariance_matrix_free(K);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply(K, jacobian, KH);
    stonesoup_covariance_matrix_free(K);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(KH);
        return err;
    }

    // Create identity matrix I
    stonesoup_covariance_matrix_t* I = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!I) {
        stonesoup_covariance_matrix_free(KH);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_covariance_matrix_eye(I);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(KH);
        stonesoup_covariance_matrix_free(I);
        return err;
    }

    // Compute I - K * H
    stonesoup_covariance_matrix_t* I_KH = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!I_KH) {
        stonesoup_covariance_matrix_free(KH);
        stonesoup_covariance_matrix_free(I);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_subtract(I, KH, I_KH);
    stonesoup_covariance_matrix_free(I);
    stonesoup_covariance_matrix_free(KH);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(I_KH);
        return err;
    }

    // Finally compute P_post = (I - K * H) * P_pred
    err = stonesoup_matrix_multiply(I_KH, predicted->covariance, posterior->covariance);
    stonesoup_covariance_matrix_free(I_KH);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    // Copy timestamp
    posterior->timestamp = predicted->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_kalman_innovation(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    stonesoup_state_vector_t* innovation) {

    if (!predicted || !measurement || !measurement_matrix || !innovation) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!predicted->state_vector) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = predicted->state_vector->size;
    size_t meas_dim = measurement->size;

    // Check dimensions
    if (measurement_matrix->rows != meas_dim || measurement_matrix->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (innovation->size != meas_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // Compute H * x
    stonesoup_state_vector_t* Hx = stonesoup_state_vector_create(meas_dim);
    if (!Hx) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_vector_multiply(measurement_matrix, predicted->state_vector, Hx);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_state_vector_free(Hx);
        return err;
    }

    // Compute y = z - H * x
    err = stonesoup_vector_subtract(measurement, Hx, innovation);
    stonesoup_state_vector_free(Hx);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_kalman_innovation_covariance(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    const stonesoup_covariance_matrix_t* measurement_noise,
    stonesoup_covariance_matrix_t* innovation_cov) {

    if (!predicted || !measurement_matrix ||
        !measurement_noise || !innovation_cov) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (!predicted->covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t state_dim = predicted->covariance->rows;
    size_t meas_dim = measurement_matrix->rows;

    // Check dimensions
    if (measurement_matrix->cols != state_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (measurement_noise->rows != meas_dim || measurement_noise->cols != meas_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (innovation_cov->rows != meas_dim || innovation_cov->cols != meas_dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    stonesoup_error_t err;

    // S = H * P * H^T + R
    // First compute H * P
    stonesoup_covariance_matrix_t* HP = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    if (!HP) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply(measurement_matrix, predicted->covariance, HP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(HP);
        return err;
    }

    // Now compute H * P * H^T
    stonesoup_covariance_matrix_t* HPHt = stonesoup_covariance_matrix_create(meas_dim, meas_dim);
    if (!HPHt) {
        stonesoup_covariance_matrix_free(HP);
        return STONESOUP_ERROR_ALLOCATION;
    }

    err = stonesoup_matrix_multiply_A_Bt(HP, measurement_matrix, HPHt);
    stonesoup_covariance_matrix_free(HP);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(HPHt);
        return err;
    }

    // Finally add R: S = H * P * H^T + R
    err = stonesoup_matrix_add(HPHt, measurement_noise, innovation_cov);
    stonesoup_covariance_matrix_free(HPHt);
    if (err != STONESOUP_SUCCESS) {
        return err;
    }

    return STONESOUP_SUCCESS;
}
