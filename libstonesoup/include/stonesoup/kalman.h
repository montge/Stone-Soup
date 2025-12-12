/**
 * @file kalman.h
 * @brief Kalman filter operations
 */

#ifndef STONESOUP_KALMAN_H
#define STONESOUP_KALMAN_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup kalman Kalman Filtering
 * @brief Kalman filter prediction and update operations
 * @{
 */

/**
 * @brief Kalman filter prediction step
 *
 * Performs the prediction step of a Kalman filter:
 * x_pred = F * x
 * P_pred = F * P * F^T + Q
 *
 * @param prior Prior Gaussian state
 * @param transition_matrix State transition matrix F (state_dim × state_dim)
 * @param process_noise Process noise covariance Q (state_dim × state_dim)
 * @param predicted Output predicted state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_kalman_predict(
    const stonesoup_gaussian_state_t* prior,
    const stonesoup_covariance_matrix_t* transition_matrix,
    const stonesoup_covariance_matrix_t* process_noise,
    stonesoup_gaussian_state_t* predicted
);

/**
 * @brief Kalman filter update step
 *
 * Performs the update step of a Kalman filter given a measurement:
 * y = z - H * x_pred  (innovation)
 * S = H * P_pred * H^T + R  (innovation covariance)
 * K = P_pred * H^T * S^-1  (Kalman gain)
 * x_post = x_pred + K * y
 * P_post = (I - K * H) * P_pred
 *
 * @param predicted Predicted Gaussian state
 * @param measurement Measurement vector
 * @param measurement_matrix Measurement matrix H (meas_dim × state_dim)
 * @param measurement_noise Measurement noise covariance R (meas_dim × meas_dim)
 * @param posterior Output posterior state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_kalman_update(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    const stonesoup_covariance_matrix_t* measurement_noise,
    stonesoup_gaussian_state_t* posterior
);

/**
 * @brief Extended Kalman Filter (EKF) prediction step
 *
 * Performs the prediction step of an EKF using a linearized transition function.
 * The function pointer allows for nonlinear state transition functions.
 *
 * @param prior Prior Gaussian state
 * @param jacobian Jacobian of transition function (state_dim × state_dim)
 * @param process_noise Process noise covariance Q (state_dim × state_dim)
 * @param transition_func Nonlinear state transition function (optional, NULL for linear)
 * @param predicted Output predicted state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_ekf_predict(
    const stonesoup_gaussian_state_t* prior,
    const stonesoup_covariance_matrix_t* jacobian,
    const stonesoup_covariance_matrix_t* process_noise,
    void (*transition_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    stonesoup_gaussian_state_t* predicted
);

/**
 * @brief Extended Kalman Filter (EKF) update step
 *
 * Performs the update step of an EKF using a linearized measurement function.
 *
 * @param predicted Predicted Gaussian state
 * @param measurement Measurement vector
 * @param jacobian Jacobian of measurement function (meas_dim × state_dim)
 * @param measurement_noise Measurement noise covariance R (meas_dim × meas_dim)
 * @param measurement_func Nonlinear measurement function (optional, NULL for linear)
 * @param posterior Output posterior state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_ekf_update(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* jacobian,
    const stonesoup_covariance_matrix_t* measurement_noise,
    void (*measurement_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    stonesoup_gaussian_state_t* posterior
);

/**
 * @brief Compute innovation (measurement residual)
 *
 * Computes y = z - H * x where:
 * y is the innovation
 * z is the measurement
 * H is the measurement matrix
 * x is the predicted state
 *
 * @param predicted Predicted state
 * @param measurement Measurement vector
 * @param measurement_matrix Measurement matrix H
 * @param innovation Output innovation vector (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_kalman_innovation(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    stonesoup_state_vector_t* innovation
);

/**
 * @brief Compute innovation covariance
 *
 * Computes S = H * P * H^T + R
 *
 * @param predicted Predicted state with covariance P
 * @param measurement_matrix Measurement matrix H
 * @param measurement_noise Measurement noise covariance R
 * @param innovation_cov Output innovation covariance (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_kalman_innovation_covariance(
    const stonesoup_gaussian_state_t* predicted,
    const stonesoup_covariance_matrix_t* measurement_matrix,
    const stonesoup_covariance_matrix_t* measurement_noise,
    stonesoup_covariance_matrix_t* innovation_cov
);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_KALMAN_H */
