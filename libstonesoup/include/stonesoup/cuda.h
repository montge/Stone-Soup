/* SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors */
/* SPDX-License-Identifier: MIT */

/**
 * @file cuda.h
 * @brief CUDA GPU acceleration support for Stone Soup
 *
 * This header provides CUDA-accelerated versions of Stone Soup operations.
 * Functions are only available when compiled with ENABLE_CUDA=ON.
 */

#ifndef STONESOUP_CUDA_H
#define STONESOUP_CUDA_H

#include "stonesoup/types.h"
#include "stonesoup/matrix.h"
#include "stonesoup/kalman.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup cuda CUDA GPU Acceleration
 * @brief GPU-accelerated operations using CUDA
 * @{
 */

/**
 * @brief Check if CUDA is available at runtime
 * @return 1 if CUDA is available, 0 otherwise
 */
int stonesoup_cuda_available(void);

/**
 * @brief Get number of CUDA devices
 * @return Number of CUDA-capable devices
 */
int stonesoup_cuda_device_count(void);

/**
 * @brief Get CUDA device name
 * @param device Device index
 * @param name Output buffer for device name
 * @param max_len Maximum length of name buffer
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_device_name(int device, char *name, size_t max_len);

/**
 * @brief Get CUDA device memory info
 * @param device Device index
 * @param total_bytes Output: total device memory in bytes
 * @param free_bytes Output: free device memory in bytes
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_memory_info(int device, size_t *total_bytes, size_t *free_bytes);

/**
 * @brief Set current CUDA device
 * @param device Device index
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_set_device(int device);

/* Matrix operations */

/**
 * @brief GPU matrix multiplication: C = A * B
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @param C Output matrix (m x n)
 * @param m Rows of A and C
 * @param k Columns of A, rows of B
 * @param n Columns of B and C
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_matrix_multiply(
    const ss_float_t *A, const ss_float_t *B, ss_float_t *C,
    int m, int k, int n);

/**
 * @brief GPU matrix-vector multiplication: y = A * x
 * @param A Matrix (m x n)
 * @param x Vector (n)
 * @param y Output vector (m)
 * @param m Rows of A
 * @param n Columns of A
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_matrix_vector_multiply(
    const ss_float_t *A, const ss_float_t *x, ss_float_t *y,
    int m, int n);

/**
 * @brief GPU matrix transpose: B = A^T
 * @param A Input matrix (m x n)
 * @param B Output matrix (n x m)
 * @param m Rows of A
 * @param n Columns of A
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_matrix_transpose(
    const ss_float_t *A, ss_float_t *B,
    int m, int n);

/* Kalman filter operations */

/**
 * @brief GPU batch Kalman predict for multiple states
 * @param x_batch Input/output state vectors (batch_size x state_dim)
 * @param P_batch Input/output covariances (batch_size x state_dim x state_dim)
 * @param F Transition matrix (state_dim x state_dim)
 * @param Q Process noise (state_dim x state_dim)
 * @param batch_size Number of states
 * @param state_dim State dimension
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_batch_kalman_predict(
    ss_float_t *x_batch, ss_float_t *P_batch,
    const ss_float_t *F, const ss_float_t *Q,
    int batch_size, int state_dim);

/**
 * @brief GPU Kalman predict
 * @param x State vector (state_dim)
 * @param P Covariance matrix (state_dim x state_dim)
 * @param F Transition matrix (state_dim x state_dim)
 * @param Q Process noise (state_dim x state_dim)
 * @param state_dim State dimension
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_kalman_predict(
    ss_float_t *x, ss_float_t *P,
    const ss_float_t *F, const ss_float_t *Q,
    int state_dim);

/**
 * @brief GPU Kalman update
 * @param x State vector (state_dim)
 * @param P Covariance matrix (state_dim x state_dim)
 * @param z Measurement (meas_dim)
 * @param H Measurement matrix (meas_dim x state_dim)
 * @param R Measurement noise (meas_dim x meas_dim)
 * @param state_dim State dimension
 * @param meas_dim Measurement dimension
 * @return 0 on success, -1 on error
 */
int stonesoup_cuda_kalman_update(
    ss_float_t *x, ss_float_t *P,
    const ss_float_t *z, const ss_float_t *H, const ss_float_t *R,
    int state_dim, int meas_dim);

/** @} */ /* end of cuda group */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_CUDA_H */
