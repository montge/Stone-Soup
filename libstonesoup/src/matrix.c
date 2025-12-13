/**
 * @file matrix.c
 * @brief Implementation of matrix and linear algebra operations for Stone Soup
 */

#include "stonesoup/matrix.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * Helper macros for matrix indexing (row-major storage)
 * ============================================================================ */

#define MAT_IDX(mat, i, j) ((mat)->data[(i) * (mat)->cols + (j)])
#define MAT_AT(mat, i, j) ((mat)->data[(i) * (mat)->cols + (j)])

/* ============================================================================
 * Matrix-Matrix Operations
 * ============================================================================ */

stonesoup_error_t
stonesoup_matrix_multiply(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: A (m × k), B (k × n), C (m × n) */
    if (A->cols != B->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (C->rows != A->rows || C->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->rows;
    size_t k = A->cols;
    size_t n = B->cols;

    /* C = A * B */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                sum += MAT_AT(A, i, p) * MAT_AT(B, p, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_multiply_At_B(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: A^T (m × k) where A is (k × m), B (k × n), C (m × n) */
    if (A->rows != B->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (C->rows != A->cols || C->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->cols;  /* rows of A^T */
    size_t k = A->rows;  /* cols of A^T = rows of B */
    size_t n = B->cols;

    /* C = A^T * B */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                sum += MAT_AT(A, p, i) * MAT_AT(B, p, j);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_multiply_A_Bt(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: A (m × k), B^T (k × n) where B is (n × k), C (m × n) */
    if (A->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (C->rows != A->rows || C->cols != B->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->rows;
    size_t k = A->cols;  /* = cols of B */
    size_t n = B->rows;  /* rows of B^T */

    /* C = A * B^T */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                sum += MAT_AT(A, i, p) * MAT_AT(B, j, p);
            }
            MAT_AT(C, i, j) = sum;
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_add(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: all matrices must have same dimensions */
    if (A->rows != B->rows || A->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (C->rows != A->rows || C->cols != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t size = A->rows * A->cols;
    for (size_t i = 0; i < size; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_subtract(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: all matrices must have same dimensions */
    if (A->rows != B->rows || A->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (C->rows != A->rows || C->cols != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t size = A->rows * A->cols;
    for (size_t i = 0; i < size; i++) {
        C->data[i] = A->data[i] - B->data[i];
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_transpose(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* B
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: B must be (n × m) if A is (m × n) */
    if (B->rows != A->cols || B->cols != A->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->rows;
    size_t n = A->cols;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            MAT_AT(B, j, i) = MAT_AT(A, i, j);
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_scale(
    const stonesoup_covariance_matrix_t* A,
    double alpha,
    stonesoup_covariance_matrix_t* B
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (B->rows != A->rows || B->cols != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t size = A->rows * A->cols;
    for (size_t i = 0; i < size; i++) {
        B->data[i] = alpha * A->data[i];
    }

    return STONESOUP_SUCCESS;
}

/* ============================================================================
 * Matrix-Vector Operations
 * ============================================================================ */

stonesoup_error_t
stonesoup_matrix_vector_multiply(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* x,
    stonesoup_state_vector_t* y
) {
    /* Null pointer checks */
    if (A == NULL || x == NULL || y == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || x->data == NULL || y->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: A (m × n), x (n), y (m) */
    if (A->cols != x->size) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (y->size != A->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->rows;
    size_t n = A->cols;

    /* y = A * x */
    for (size_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += MAT_AT(A, i, j) * x->data[j];
        }
        y->data[i] = sum;
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_vector_multiply_At(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* x,
    stonesoup_state_vector_t* y
) {
    /* Null pointer checks */
    if (A == NULL || x == NULL || y == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || x->data == NULL || y->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: A^T (n × m) where A is (m × n), x (m), y (n) */
    if (A->rows != x->size) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (y->size != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = A->rows;
    size_t n = A->cols;

    /* y = A^T * x */
    for (size_t j = 0; j < n; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < m; i++) {
            sum += MAT_AT(A, i, j) * x->data[i];
        }
        y->data[j] = sum;
    }

    return STONESOUP_SUCCESS;
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

stonesoup_error_t
stonesoup_vector_add(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_state_vector_t* z
) {
    /* Null pointer checks */
    if (x == NULL || y == NULL || z == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (x->data == NULL || y->data == NULL || z->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (x->size != y->size || z->size != x->size) {
        return STONESOUP_ERROR_DIMENSION;
    }

    for (size_t i = 0; i < x->size; i++) {
        z->data[i] = x->data[i] + y->data[i];
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_vector_subtract(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_state_vector_t* z
) {
    /* Null pointer checks */
    if (x == NULL || y == NULL || z == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (x->data == NULL || y->data == NULL || z->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (x->size != y->size || z->size != x->size) {
        return STONESOUP_ERROR_DIMENSION;
    }

    for (size_t i = 0; i < x->size; i++) {
        z->data[i] = x->data[i] - y->data[i];
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_vector_scale(
    const stonesoup_state_vector_t* x,
    double alpha,
    stonesoup_state_vector_t* y
) {
    /* Null pointer checks */
    if (x == NULL || y == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (x->data == NULL || y->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (y->size != x->size) {
        return STONESOUP_ERROR_DIMENSION;
    }

    for (size_t i = 0; i < x->size; i++) {
        y->data[i] = alpha * x->data[i];
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_vector_dot(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    double* result
) {
    /* Null pointer checks */
    if (x == NULL || y == NULL || result == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (x->data == NULL || y->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (x->size != y->size) {
        return STONESOUP_ERROR_DIMENSION;
    }

    double sum = 0.0;
    for (size_t i = 0; i < x->size; i++) {
        sum += x->data[i] * y->data[i];
    }
    *result = sum;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_vector_outer_product(
    const stonesoup_state_vector_t* x,
    const stonesoup_state_vector_t* y,
    stonesoup_covariance_matrix_t* C
) {
    /* Null pointer checks */
    if (x == NULL || y == NULL || C == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (x->data == NULL || y->data == NULL || C->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks: C must be (m × n) where x has m elements, y has n */
    if (C->rows != x->size || C->cols != y->size) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t m = x->size;
    size_t n = y->size;

    /* C = x * y^T */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            MAT_AT(C, i, j) = x->data[i] * y->data[j];
        }
    }

    return STONESOUP_SUCCESS;
}

/* ============================================================================
 * Matrix Decomposition and Inversion
 * ============================================================================ */

/**
 * @brief LU decomposition with partial pivoting
 *
 * Decomposes A into P * A = L * U where P is a permutation matrix
 * represented by the perm array.
 */
static stonesoup_error_t
lu_decomposition(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* LU,
    size_t* perm
) {
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Copy A to LU */
    memcpy(LU->data, A->data, n * n * sizeof(double));

    /* Initialize permutation to identity */
    for (size_t i = 0; i < n; i++) {
        perm[i] = i;
    }

    /* Perform LU decomposition with partial pivoting */
    for (size_t k = 0; k < n; k++) {
        /* Find pivot */
        size_t pivot = k;
        double max_val = fabs(MAT_AT(LU, k, k));
        for (size_t i = k + 1; i < n; i++) {
            double val = fabs(MAT_AT(LU, i, k));
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }

        /* Check for singularity */
        if (max_val < DBL_EPSILON) {
            return STONESOUP_ERROR_SINGULAR;
        }

        /* Swap rows if needed */
        if (pivot != k) {
            /* Swap in permutation */
            size_t temp_perm = perm[k];
            perm[k] = perm[pivot];
            perm[pivot] = temp_perm;

            /* Swap rows in LU */
            for (size_t j = 0; j < n; j++) {
                double temp = MAT_AT(LU, k, j);
                MAT_AT(LU, k, j) = MAT_AT(LU, pivot, j);
                MAT_AT(LU, pivot, j) = temp;
            }
        }

        /* Compute multipliers and eliminate */
        for (size_t i = k + 1; i < n; i++) {
            MAT_AT(LU, i, k) /= MAT_AT(LU, k, k);
            for (size_t j = k + 1; j < n; j++) {
                MAT_AT(LU, i, j) -= MAT_AT(LU, i, k) * MAT_AT(LU, k, j);
            }
        }
    }

    return STONESOUP_SUCCESS;
}

/**
 * @brief Solve L * x = b where L is lower triangular (forward substitution)
 *
 * This version assumes L has real values on the diagonal (e.g., from Cholesky).
 */
static void
forward_substitution(
    const stonesoup_covariance_matrix_t* L,
    const double* b,
    double* x,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        double sum = b[i];
        for (size_t j = 0; j < i; j++) {
            sum -= MAT_AT(L, i, j) * x[j];
        }
        x[i] = sum / MAT_AT(L, i, i);
    }
}

/**
 * @brief Solve L * x = b where L has implicit 1's on diagonal (LU format)
 *
 * This version is for packed LU decomposition where L has implicit 1's.
 */
static void
lu_forward_substitution(
    const stonesoup_covariance_matrix_t* LU,
    const double* b,
    double* x,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        double sum = b[i];
        for (size_t j = 0; j < i; j++) {
            sum -= MAT_AT(LU, i, j) * x[j];
        }
        x[i] = sum;  /* L has implicit 1's on diagonal */
    }
}

/**
 * @brief Solve U * x = b where U is upper triangular (backward substitution)
 */
static void
backward_substitution(
    const stonesoup_covariance_matrix_t* U,
    const double* b,
    double* x,
    size_t n
) {
    for (size_t i = n; i-- > 0; ) {
        double sum = b[i];
        for (size_t j = i + 1; j < n; j++) {
            sum -= MAT_AT(U, i, j) * x[j];
        }
        x[i] = sum / MAT_AT(U, i, i);
    }
}

/**
 * @brief Solve L^T * x = b where L is lower triangular (transpose backward substitution)
 *
 * This solves the system with L's transpose without explicitly forming L^T.
 * L^T[i][j] = L[j][i], so we access L[j][i] instead of L[i][j].
 */
static void
cholesky_backward_substitution(
    const stonesoup_covariance_matrix_t* L,
    const double* b,
    double* x,
    size_t n
) {
    for (size_t i = n; i-- > 0; ) {
        double sum = b[i];
        for (size_t j = i + 1; j < n; j++) {
            sum -= MAT_AT(L, j, i) * x[j];  /* Access L^T[i][j] = L[j][i] */
        }
        x[i] = sum / MAT_AT(L, i, i);  /* L^T diagonal = L diagonal */
    }
}

stonesoup_error_t
stonesoup_matrix_inverse(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* B
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (B->rows != A->rows || B->cols != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Allocate workspace for LU decomposition */
    stonesoup_covariance_matrix_t* LU = stonesoup_covariance_matrix_create(n, n);
    if (LU == NULL) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    size_t* perm = (size_t*)malloc(n * sizeof(size_t));
    if (perm == NULL) {
        stonesoup_covariance_matrix_free(LU);
        return STONESOUP_ERROR_ALLOCATION;
    }

    double* work = (double*)malloc(n * sizeof(double));
    if (work == NULL) {
        stonesoup_covariance_matrix_free(LU);
        free(perm);
        return STONESOUP_ERROR_ALLOCATION;
    }

    /* Perform LU decomposition */
    stonesoup_error_t err = lu_decomposition(A, LU, perm);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(LU);
        free(perm);
        free(work);
        return err;
    }

    /* Solve for each column of the inverse */
    for (size_t j = 0; j < n; j++) {
        /* Create unit vector e_j with permutation applied */
        for (size_t i = 0; i < n; i++) {
            work[i] = (perm[i] == j) ? 1.0 : 0.0;
        }

        /* Solve L * y = P * e_j */
        double* y = (double*)malloc(n * sizeof(double));
        if (y == NULL) {
            stonesoup_covariance_matrix_free(LU);
            free(perm);
            free(work);
            return STONESOUP_ERROR_ALLOCATION;
        }

        lu_forward_substitution(LU, work, y, n);

        /* Solve U * x = y */
        for (size_t i = 0; i < n; i++) {
            work[i] = y[i];
        }
        free(y);

        backward_substitution(LU, work, work, n);

        /* Store column in B */
        for (size_t i = 0; i < n; i++) {
            MAT_AT(B, i, j) = work[i];
        }
    }

    /* Clean up */
    stonesoup_covariance_matrix_free(LU);
    free(perm);
    free(work);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_cholesky(
    const stonesoup_covariance_matrix_t* A,
    stonesoup_covariance_matrix_t* L
) {
    /* Null pointer checks */
    if (A == NULL || L == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || L->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (L->rows != A->rows || L->cols != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Initialize L to zero */
    memset(L->data, 0, n * n * sizeof(double));

    /* Cholesky decomposition: A = L * L^T */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double sum = MAT_AT(A, i, j);
            for (size_t k = 0; k < j; k++) {
                sum -= MAT_AT(L, i, k) * MAT_AT(L, j, k);
            }

            if (i == j) {
                /* Diagonal element */
                if (sum <= 0.0) {
                    /* Matrix is not positive definite */
                    return STONESOUP_ERROR_SINGULAR;
                }
                MAT_AT(L, i, j) = sqrt(sum);
            } else {
                /* Off-diagonal element */
                MAT_AT(L, i, j) = sum / MAT_AT(L, j, j);
            }
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_cholesky_solve(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_covariance_matrix_t* B,
    stonesoup_covariance_matrix_t* X
) {
    /* Null pointer checks */
    if (A == NULL || B == NULL || X == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || B->data == NULL || X->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (B->rows != A->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (X->rows != A->rows || X->cols != B->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;
    size_t m = B->cols;

    /* Compute Cholesky decomposition */
    stonesoup_covariance_matrix_t* L = stonesoup_covariance_matrix_create(n, n);
    if (L == NULL) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    stonesoup_error_t err = stonesoup_matrix_cholesky(A, L);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(L);
        return err;
    }

    /* Allocate workspace */
    double* y = (double*)malloc(n * sizeof(double));
    if (y == NULL) {
        stonesoup_covariance_matrix_free(L);
        return STONESOUP_ERROR_ALLOCATION;
    }

    /* Solve for each column of X */
    for (size_t j = 0; j < m; j++) {
        /* Extract column j of B */
        double* b_col = (double*)malloc(n * sizeof(double));
        if (b_col == NULL) {
            stonesoup_covariance_matrix_free(L);
            free(y);
            return STONESOUP_ERROR_ALLOCATION;
        }

        for (size_t i = 0; i < n; i++) {
            b_col[i] = MAT_AT(B, i, j);
        }

        /* Solve L * y = b */
        forward_substitution(L, b_col, y, n);

        /* Solve L^T * x = y */
        cholesky_backward_substitution(L, y, b_col, n);

        /* Store result in X */
        for (size_t i = 0; i < n; i++) {
            MAT_AT(X, i, j) = b_col[i];
        }

        free(b_col);
    }

    /* Clean up */
    stonesoup_covariance_matrix_free(L);
    free(y);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_cholesky_solve_vector(
    const stonesoup_covariance_matrix_t* A,
    const stonesoup_state_vector_t* b,
    stonesoup_state_vector_t* x
) {
    /* Null pointer checks */
    if (A == NULL || b == NULL || x == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL || b->data == NULL || x->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }
    if (b->size != A->rows || x->size != A->rows) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Compute Cholesky decomposition */
    stonesoup_covariance_matrix_t* L = stonesoup_covariance_matrix_create(n, n);
    if (L == NULL) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    stonesoup_error_t err = stonesoup_matrix_cholesky(A, L);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(L);
        return err;
    }

    /* Allocate workspace */
    double* y = (double*)malloc(n * sizeof(double));
    if (y == NULL) {
        stonesoup_covariance_matrix_free(L);
        return STONESOUP_ERROR_ALLOCATION;
    }

    /* Solve L * y = b */
    forward_substitution(L, b->data, y, n);

    /* Solve L^T * x = y */
    cholesky_backward_substitution(L, y, x->data, n);

    /* Clean up */
    stonesoup_covariance_matrix_free(L);
    free(y);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_determinant(
    const stonesoup_covariance_matrix_t* A,
    double* det
) {
    /* Null pointer checks */
    if (A == NULL || det == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Use LU decomposition to compute determinant */
    stonesoup_covariance_matrix_t* LU = stonesoup_covariance_matrix_create(n, n);
    if (LU == NULL) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    size_t* perm = (size_t*)malloc(n * sizeof(size_t));
    if (perm == NULL) {
        stonesoup_covariance_matrix_free(LU);
        return STONESOUP_ERROR_ALLOCATION;
    }

    stonesoup_error_t err = lu_decomposition(A, LU, perm);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(LU);
        free(perm);
        *det = 0.0;
        return err;
    }

    /* Determinant is product of diagonal elements of U */
    /* times sign of permutation */
    double result = 1.0;
    for (size_t i = 0; i < n; i++) {
        result *= MAT_AT(LU, i, i);
    }

    /* Count permutation sign */
    int parity = 0;
    for (size_t i = 0; i < n; i++) {
        if (perm[i] != i) {
            /* Find where i is in the permutation */
            for (size_t j = i + 1; j < n; j++) {
                if (perm[j] == i) {
                    parity++;
                    break;
                }
            }
        }
    }

    if (parity % 2 == 1) {
        result = -result;
    }

    *det = result;

    stonesoup_covariance_matrix_free(LU);
    free(perm);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t
stonesoup_matrix_log_determinant(
    const stonesoup_covariance_matrix_t* A,
    double* log_det
) {
    /* Null pointer checks */
    if (A == NULL || log_det == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }
    if (A->data == NULL) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    /* Dimension checks */
    if (A->rows != A->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    size_t n = A->rows;

    /* Use Cholesky decomposition for positive definite matrices */
    /* log(det(A)) = 2 * sum(log(L_ii)) where A = L * L^T */
    stonesoup_covariance_matrix_t* L = stonesoup_covariance_matrix_create(n, n);
    if (L == NULL) {
        return STONESOUP_ERROR_ALLOCATION;
    }

    stonesoup_error_t err = stonesoup_matrix_cholesky(A, L);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_covariance_matrix_free(L);
        return err;
    }

    /* Compute log determinant */
    double result = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diag = MAT_AT(L, i, i);
        if (diag <= 0.0) {
            stonesoup_covariance_matrix_free(L);
            return STONESOUP_ERROR_SINGULAR;
        }
        result += log(diag);
    }
    result *= 2.0;  /* Because det(A) = det(L)^2 */

    *log_det = result;

    stonesoup_covariance_matrix_free(L);

    return STONESOUP_SUCCESS;
}
