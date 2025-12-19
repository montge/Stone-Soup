/**
 * @file matrix.h
 * @brief Matrix and linear algebra operations for Stone Soup
 *
 * This module provides fundamental matrix operations needed for
 * state estimation algorithms like Kalman filtering.
 */

#ifndef STONESOUP_MATRIX_H
#define STONESOUP_MATRIX_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup matrix Matrix Operations
 * @brief Linear algebra operations for state estimation
 * @{
 */

/* ============================================================================
 * Matrix-Matrix Operations
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiplication C = A * B
 *
 * @param A Left matrix (m × k)
 * @param B Right matrix (k × n)
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_multiply(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
);

/**
 * @brief Matrix-matrix multiplication with transpose C = A^T * B
 *
 * @param A Left matrix (k × m), will be transposed
 * @param B Right matrix (k × n)
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_multiply_At_B(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
);

/**
 * @brief Matrix-matrix multiplication with transpose C = A * B^T
 *
 * @param A Left matrix (m × k)
 * @param B Right matrix (n × k), will be transposed
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_multiply_A_Bt(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
);

/**
 * @brief Matrix addition C = A + B
 *
 * @param A First matrix (m × n)
 * @param B Second matrix (m × n)
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_add(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
);

/**
 * @brief Matrix subtraction C = A - B
 *
 * @param A First matrix (m × n)
 * @param B Second matrix (m × n)
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_subtract(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
);

/**
 * @brief Matrix transpose B = A^T
 *
 * @param A Input matrix (m × n)
 * @param B Output transposed matrix (n × m), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_transpose(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* B
);

/**
 * @brief Scale matrix B = alpha * A
 *
 * @param A Input matrix
 * @param alpha Scalar multiplier
 * @param B Output matrix, must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_scale(
    const stonesoup_covariance_matrix_t* A,
    double alpha,
    stonesoup_covariance_matrix_t* B
);

/* ============================================================================
 * Matrix-Vector Operations
 * ============================================================================ */

/**
 * @brief Matrix-vector multiplication y = A * x
 *
 * @param A Matrix (m × n)
 * @param x Input vector (n)
 * @param y Output vector (m), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_vector_multiply(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* x,
    stonesoup_state_vector_t* y
);

/**
 * @brief Transposed matrix-vector multiplication y = A^T * x
 *
 * @param A Matrix (m × n), will be transposed
 * @param x Input vector (m)
 * @param y Output vector (n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_vector_multiply_At(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* x,
    stonesoup_state_vector_t* y
);

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

/**
 * @brief Vector addition z = x + y
 *
 * @param x First vector
 * @param y Second vector
 * @param z Output vector, must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_vector_add(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_state_vector_t* z
);

/**
 * @brief Vector subtraction z = x - y
 *
 * @param x First vector
 * @param y Second vector
 * @param z Output vector, must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_vector_subtract(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_state_vector_t* z
);

/**
 * @brief Scale vector y = alpha * x
 *
 * @param x Input vector
 * @param alpha Scalar multiplier
 * @param y Output vector, must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_vector_scale(
    const stonesoup_state_vector_t* x,
    double alpha,
    stonesoup_state_vector_t* y
);

/**
 * @brief Vector dot product
 *
 * @param x First vector
 * @param y Second vector
 * @param result Output scalar result
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_vector_dot(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    double* result
);

/**
 * @brief Outer product C = x * y^T
 *
 * @param x First vector (m)
 * @param y Second vector (n)
 * @param C Output matrix (m × n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_vector_outer_product(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_covariance_matrix_t* C
);

/* ============================================================================
 * Matrix Decomposition and Inversion
 * ============================================================================ */

/**
 * @brief Compute matrix inverse B = A^(-1)
 *
 * Uses LU decomposition for general matrices.
 * For positive definite matrices, use stonesoup_matrix_cholesky_solve.
 *
 * @param A Input square matrix (n × n)
 * @param B Output inverse matrix (n × n), must be pre-allocated
 * @return Error code (STONESOUP_ERROR_SINGULAR if matrix is singular)
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_inverse(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* B
);

/**
 * @brief Cholesky decomposition A = L * L^T
 *
 * Computes lower triangular Cholesky factor L for positive definite matrix A.
 *
 * @param A Input positive definite matrix (n × n)
 * @param L Output lower triangular Cholesky factor (n × n)
 * @return Error code (STONESOUP_ERROR_SINGULAR if not positive definite)
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_cholesky(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* L
);

/**
 * @brief Solve linear system A * X = B for X using Cholesky decomposition
 *
 * Efficient solver for positive definite A (e.g., covariance matrices).
 *
 * @param A Positive definite matrix (n × n)
 * @param B Right-hand side matrix (n × m)
 * @param X Output solution matrix (n × m), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_cholesky_solve(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* X
);

/**
 * @brief Solve linear system A * x = b for x using Cholesky decomposition
 *
 * Efficient solver for positive definite A (e.g., covariance matrices).
 *
 * @param A Positive definite matrix (n × n)
 * @param b Right-hand side vector (n)
 * @param x Output solution vector (n), must be pre-allocated
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_cholesky_solve_vector(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* b,
    stonesoup_state_vector_t* x
);

/**
 * @brief Compute matrix determinant
 *
 * @param A Input square matrix
 * @param det Output determinant value
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_determinant(
    const stonesoup_covariance_matrix_t* A,
    double* det
);

/**
 * @brief Compute log determinant (more numerically stable)
 *
 * @param A Input positive definite matrix
 * @param log_det Output log determinant value
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_matrix_log_determinant(
    const stonesoup_covariance_matrix_t* A,
    double* log_det
);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_MATRIX_H */
