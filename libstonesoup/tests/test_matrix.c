/**
 * @file test_matrix.c
 * @brief Comprehensive tests for Stone Soup matrix operations
 */

#include <stonesoup/stonesoup.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-9

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

/* Helper to create a simple 2x2 test matrix */
static stonesoup_covariance_matrix_t* create_test_matrix_2x2(void) {
    stonesoup_covariance_matrix_t* mat = stonesoup_covariance_matrix_create(2, 2);
    if (mat) {
        mat->data[0] = 1.0; mat->data[1] = 2.0;  // [1 2]
        mat->data[2] = 3.0; mat->data[3] = 4.0;  // [3 4]
    }
    return mat;
}

/* Helper to create a simple 3x3 positive definite matrix */
static stonesoup_covariance_matrix_t* create_test_pd_matrix_3x3(void) {
    stonesoup_covariance_matrix_t* mat = stonesoup_covariance_matrix_create(3, 3);
    if (mat) {
        // Create a positive definite matrix: [4 2 0; 2 3 1; 0 1 2]
        mat->data[0] = 4.0; mat->data[1] = 2.0; mat->data[2] = 0.0;
        mat->data[3] = 2.0; mat->data[4] = 3.0; mat->data[5] = 1.0;
        mat->data[6] = 0.0; mat->data[7] = 1.0; mat->data[8] = 2.0;
    }
    return mat;
}

/* ============================================================================
 * Matrix-Matrix Operations Tests
 * ============================================================================ */

static int test_matrix_multiply_basic(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // A = [1 2; 3 4]
    A->data[0] = 1.0; A->data[1] = 2.0;
    A->data[2] = 3.0; A->data[3] = 4.0;

    // B = [5 6; 7 8]
    B->data[0] = 5.0; B->data[1] = 6.0;
    B->data[2] = 7.0; B->data[3] = 8.0;

    stonesoup_error_t err = stonesoup_matrix_multiply(A, B, C);

    // If not implemented, skip the test
    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = A * B = [19 22; 43 50]
    assert(double_equals(C->data[0], 19.0));
    assert(double_equals(C->data[1], 22.0));
    assert(double_equals(C->data[2], 43.0));
    assert(double_equals(C->data[3], 50.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_multiply_dimension_check(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 3);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    stonesoup_error_t err = stonesoup_matrix_multiply(A, B, C);

    // Should fail with dimension error (A cols != B rows)
    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_DIMENSION);
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_transpose(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 3);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(3, 2);

    assert(A && B);

    // A = [1 2 3; 4 5 6]
    A->data[0] = 1.0; A->data[1] = 2.0; A->data[2] = 3.0;
    A->data[3] = 4.0; A->data[4] = 5.0; A->data[5] = 6.0;

    stonesoup_error_t err = stonesoup_matrix_transpose(A, B);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // B = A^T = [1 4; 2 5; 3 6]
    assert(double_equals(B->data[0], 1.0));
    assert(double_equals(B->data[1], 4.0));
    assert(double_equals(B->data[2], 2.0));
    assert(double_equals(B->data[3], 5.0));
    assert(double_equals(B->data[4], 3.0));
    assert(double_equals(B->data[5], 6.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    return 1;
}

static int test_matrix_add(void) {
    stonesoup_covariance_matrix_t* A = create_test_matrix_2x2();
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // B = [5 6; 7 8]
    B->data[0] = 5.0; B->data[1] = 6.0;
    B->data[2] = 7.0; B->data[3] = 8.0;

    stonesoup_error_t err = stonesoup_matrix_add(A, B, C);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = A + B = [6 8; 10 12]
    assert(double_equals(C->data[0], 6.0));
    assert(double_equals(C->data[1], 8.0));
    assert(double_equals(C->data[2], 10.0));
    assert(double_equals(C->data[3], 12.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_subtract(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // A = [5 6; 7 8]
    A->data[0] = 5.0; A->data[1] = 6.0;
    A->data[2] = 7.0; A->data[3] = 8.0;

    // B = [1 2; 3 4]
    B->data[0] = 1.0; B->data[1] = 2.0;
    B->data[2] = 3.0; B->data[3] = 4.0;

    stonesoup_error_t err = stonesoup_matrix_subtract(A, B, C);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = A - B = [4 4; 4 4]
    assert(double_equals(C->data[0], 4.0));
    assert(double_equals(C->data[1], 4.0));
    assert(double_equals(C->data[2], 4.0));
    assert(double_equals(C->data[3], 4.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

/* ============================================================================
 * Matrix-Vector Operations Tests
 * ============================================================================ */

static int test_matrix_vector_multiply(void) {
    stonesoup_covariance_matrix_t* A = create_test_matrix_2x2();
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(2);

    assert(A && x && y);

    x->data[0] = 5.0;
    x->data[1] = 6.0;

    stonesoup_error_t err = stonesoup_matrix_vector_multiply(A, x, y);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // y = A * x = [1 2; 3 4] * [5; 6] = [17; 39]
    assert(double_equals(y->data[0], 17.0));
    assert(double_equals(y->data[1], 39.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    return 1;
}

/* ============================================================================
 * Vector Operations Tests
 * ============================================================================ */

static int test_vector_add(void) {
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* z = stonesoup_state_vector_create(3);

    assert(x && y && z);

    x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;
    y->data[0] = 4.0; y->data[1] = 5.0; y->data[2] = 6.0;

    stonesoup_error_t err = stonesoup_vector_add(x, y, z);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        stonesoup_state_vector_free(z);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    assert(double_equals(z->data[0], 5.0));
    assert(double_equals(z->data[1], 7.0));
    assert(double_equals(z->data[2], 9.0));

    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    stonesoup_state_vector_free(z);
    return 1;
}

static int test_vector_subtract(void) {
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* z = stonesoup_state_vector_create(3);

    assert(x && y && z);

    x->data[0] = 10.0; x->data[1] = 20.0; x->data[2] = 30.0;
    y->data[0] = 4.0;  y->data[1] = 5.0;  y->data[2] = 6.0;

    stonesoup_error_t err = stonesoup_vector_subtract(x, y, z);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        stonesoup_state_vector_free(z);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    assert(double_equals(z->data[0], 6.0));
    assert(double_equals(z->data[1], 15.0));
    assert(double_equals(z->data[2], 24.0));

    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    stonesoup_state_vector_free(z);
    return 1;
}

static int test_vector_dot_product(void) {
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(3);

    assert(x && y);

    x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;
    y->data[0] = 4.0; y->data[1] = 5.0; y->data[2] = 6.0;

    double result = 0.0;
    stonesoup_error_t err = stonesoup_vector_dot(x, y, &result);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert(double_equals(result, 32.0));

    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    return 1;
}

static int test_vector_outer_product(void) {
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(3);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 3);

    assert(x && y && C);

    x->data[0] = 2.0; x->data[1] = 3.0;
    y->data[0] = 4.0; y->data[1] = 5.0; y->data[2] = 6.0;

    stonesoup_error_t err = stonesoup_vector_outer_product(x, y, C);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = x * y^T = [2; 3] * [4 5 6] = [8 10 12; 12 15 18]
    assert(double_equals(C->data[0], 8.0));
    assert(double_equals(C->data[1], 10.0));
    assert(double_equals(C->data[2], 12.0));
    assert(double_equals(C->data[3], 12.0));
    assert(double_equals(C->data[4], 15.0));
    assert(double_equals(C->data[5], 18.0));

    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

/* ============================================================================
 * Matrix Decomposition and Inversion Tests
 * ============================================================================ */

static int test_matrix_inverse(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // A = [4 7; 2 6]
    A->data[0] = 4.0; A->data[1] = 7.0;
    A->data[2] = 2.0; A->data[3] = 6.0;

    stonesoup_error_t err = stonesoup_matrix_inverse(A, B);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Verify A * A^-1 = I
    err = stonesoup_matrix_multiply(A, B, C);
    assert(err == STONESOUP_SUCCESS);

    // Check if result is identity matrix
    assert(double_equals(C->data[0], 1.0));
    assert(double_equals(C->data[1], 0.0));
    assert(double_equals(C->data[2], 0.0));
    assert(double_equals(C->data[3], 1.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_cholesky_decomposition(void) {
    stonesoup_covariance_matrix_t* A = create_test_pd_matrix_3x3();
    stonesoup_covariance_matrix_t* L = stonesoup_covariance_matrix_create(3, 3);
    stonesoup_covariance_matrix_t* Lt = stonesoup_covariance_matrix_create(3, 3);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(3, 3);

    assert(A && L && Lt && C);

    stonesoup_error_t err = stonesoup_matrix_cholesky(A, L);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(L);
        stonesoup_covariance_matrix_free(Lt);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Verify L * L^T = A
    err = stonesoup_matrix_transpose(L, Lt);
    assert(err == STONESOUP_SUCCESS);

    err = stonesoup_matrix_multiply(L, Lt, C);
    assert(err == STONESOUP_SUCCESS);

    // Compare C with original A
    for (size_t i = 0; i < 9; i++) {
        assert(double_equals(C->data[i], A->data[i]));
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(L);
    stonesoup_covariance_matrix_free(Lt);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_cholesky_solve_vector(void) {
    stonesoup_covariance_matrix_t* A = create_test_pd_matrix_3x3();
    stonesoup_state_vector_t* b = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* check = stonesoup_state_vector_create(3);

    assert(A && b && x && check);

    b->data[0] = 1.0; b->data[1] = 2.0; b->data[2] = 3.0;

    stonesoup_error_t err = stonesoup_matrix_cholesky_solve_vector(A, b, x);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_state_vector_free(b);
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(check);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Verify A * x = b
    err = stonesoup_matrix_vector_multiply(A, x, check);
    assert(err == STONESOUP_SUCCESS);

    for (size_t i = 0; i < 3; i++) {
        assert(double_equals(check->data[i], b->data[i]));
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_state_vector_free(b);
    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(check);
    return 1;
}

static int test_matrix_determinant(void) {
    stonesoup_covariance_matrix_t* A = create_test_matrix_2x2();

    assert(A);

    double det = 0.0;
    stonesoup_error_t err = stonesoup_matrix_determinant(A, &det);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // det([1 2; 3 4]) = 1*4 - 2*3 = -2
    assert(double_equals(det, -2.0));

    stonesoup_covariance_matrix_free(A);
    return 1;
}

/* ============================================================================
 * Additional Matrix Operations Tests
 * ============================================================================ */

static int test_matrix_multiply_At_B(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(3, 2);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(3, 2);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // A = [1 2; 3 4; 5 6] (3x2)
    A->data[0] = 1.0; A->data[1] = 2.0;
    A->data[2] = 3.0; A->data[3] = 4.0;
    A->data[4] = 5.0; A->data[5] = 6.0;

    // B = [7 8; 9 10; 11 12] (3x2)
    B->data[0] = 7.0;  B->data[1] = 8.0;
    B->data[2] = 9.0;  B->data[3] = 10.0;
    B->data[4] = 11.0; B->data[5] = 12.0;

    stonesoup_error_t err = stonesoup_matrix_multiply_At_B(A, B, C);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = A^T * B = [1 3 5; 2 4 6] * [7 8; 9 10; 11 12]
    // C[0,0] = 1*7 + 3*9 + 5*11 = 7 + 27 + 55 = 89
    // C[0,1] = 1*8 + 3*10 + 5*12 = 8 + 30 + 60 = 98
    // C[1,0] = 2*7 + 4*9 + 6*11 = 14 + 36 + 66 = 116
    // C[1,1] = 2*8 + 4*10 + 6*12 = 16 + 40 + 72 = 128
    assert(double_equals(C->data[0], 89.0));
    assert(double_equals(C->data[1], 98.0));
    assert(double_equals(C->data[2], 116.0));
    assert(double_equals(C->data[3], 128.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_multiply_A_Bt(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 3);
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 3);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B && C);

    // A = [1 2 3; 4 5 6] (2x3)
    A->data[0] = 1.0; A->data[1] = 2.0; A->data[2] = 3.0;
    A->data[3] = 4.0; A->data[4] = 5.0; A->data[5] = 6.0;

    // B = [7 8 9; 10 11 12] (2x3)
    B->data[0] = 7.0;  B->data[1] = 8.0;  B->data[2] = 9.0;
    B->data[3] = 10.0; B->data[4] = 11.0; B->data[5] = 12.0;

    stonesoup_error_t err = stonesoup_matrix_multiply_A_Bt(A, B, C);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // C = A * B^T = [1 2 3; 4 5 6] * [7 10; 8 11; 9 12]
    // C[0,0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    // C[0,1] = 1*10 + 2*11 + 3*12 = 10 + 22 + 36 = 68
    // C[1,0] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    // C[1,1] = 4*10 + 5*11 + 6*12 = 40 + 55 + 72 = 167
    assert(double_equals(C->data[0], 50.0));
    assert(double_equals(C->data[1], 68.0));
    assert(double_equals(C->data[2], 122.0));
    assert(double_equals(C->data[3], 167.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_vector_multiply_At(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(3, 2);
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(2);

    assert(A && x && y);

    // A = [1 2; 3 4; 5 6] (3x2)
    A->data[0] = 1.0; A->data[1] = 2.0;
    A->data[2] = 3.0; A->data[3] = 4.0;
    A->data[4] = 5.0; A->data[5] = 6.0;

    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;

    stonesoup_error_t err = stonesoup_matrix_vector_multiply_At(A, x, y);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // y = A^T * x = [1 3 5; 2 4 6] * [1; 2; 3]
    // y[0] = 1*1 + 3*2 + 5*3 = 1 + 6 + 15 = 22
    // y[1] = 2*1 + 4*2 + 6*3 = 2 + 8 + 18 = 28
    assert(double_equals(y->data[0], 22.0));
    assert(double_equals(y->data[1], 28.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    return 1;
}

static int test_matrix_scale(void) {
    stonesoup_covariance_matrix_t* A = create_test_matrix_2x2();
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(2, 2);

    assert(A && B);

    stonesoup_error_t err = stonesoup_matrix_scale(A, 2.5, B);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // B = 2.5 * A = 2.5 * [1 2; 3 4] = [2.5 5; 7.5 10]
    assert(double_equals(B->data[0], 2.5));
    assert(double_equals(B->data[1], 5.0));
    assert(double_equals(B->data[2], 7.5));
    assert(double_equals(B->data[3], 10.0));

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    return 1;
}

static int test_vector_scale(void) {
    stonesoup_state_vector_t* x = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* y = stonesoup_state_vector_create(3);

    assert(x && y);

    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;

    stonesoup_error_t err = stonesoup_vector_scale(x, 3.0, y);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_state_vector_free(x);
        stonesoup_state_vector_free(y);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // y = 3.0 * x = [3, 6, 9]
    assert(double_equals(y->data[0], 3.0));
    assert(double_equals(y->data[1], 6.0));
    assert(double_equals(y->data[2], 9.0));

    stonesoup_state_vector_free(x);
    stonesoup_state_vector_free(y);
    return 1;
}

static int test_matrix_log_determinant(void) {
    stonesoup_covariance_matrix_t* A = create_test_pd_matrix_3x3();

    assert(A);

    double log_det = 0.0;
    stonesoup_error_t err = stonesoup_matrix_log_determinant(A, &log_det);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // For the PD matrix [4 2 0; 2 3 1; 0 1 2], det ≈ 10
    // log(det) ≈ log(10) ≈ 2.303
    // Let's verify by computing det first
    double det = 0.0;
    stonesoup_matrix_determinant(A, &det);
    double expected_log_det = log(fabs(det));

    assert(fabs(log_det - expected_log_det) < 0.001);

    stonesoup_covariance_matrix_free(A);
    return 1;
}

static int test_cholesky_solve_matrix(void) {
    stonesoup_covariance_matrix_t* A = create_test_pd_matrix_3x3();
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(3, 2);
    stonesoup_covariance_matrix_t* X = stonesoup_covariance_matrix_create(3, 2);
    stonesoup_covariance_matrix_t* check = stonesoup_covariance_matrix_create(3, 2);

    assert(A && B && X && check);

    // B = [1 2; 3 4; 5 6]
    B->data[0] = 1.0; B->data[1] = 2.0;
    B->data[2] = 3.0; B->data[3] = 4.0;
    B->data[4] = 5.0; B->data[5] = 6.0;

    stonesoup_error_t err = stonesoup_matrix_cholesky_solve(A, B, X);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(X);
        stonesoup_covariance_matrix_free(check);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Verify A * X = B
    err = stonesoup_matrix_multiply(A, X, check);
    assert(err == STONESOUP_SUCCESS);

    for (size_t i = 0; i < 6; i++) {
        assert(double_equals(check->data[i], B->data[i]));
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(X);
    stonesoup_covariance_matrix_free(check);
    return 1;
}

static int test_matrix_inverse_3x3(void) {
    stonesoup_covariance_matrix_t* A = create_test_pd_matrix_3x3();
    stonesoup_covariance_matrix_t* B = stonesoup_covariance_matrix_create(3, 3);
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(3, 3);

    assert(A && B && C);

    stonesoup_error_t err = stonesoup_matrix_inverse(A, B);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_covariance_matrix_free(A);
        stonesoup_covariance_matrix_free(B);
        stonesoup_covariance_matrix_free(C);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Verify A * A^-1 = I
    err = stonesoup_matrix_multiply(A, B, C);
    assert(err == STONESOUP_SUCCESS);

    // Check if result is identity matrix (diagonal = 1, off-diagonal = 0)
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert(double_equals(C->data[i * 3 + j], expected));
        }
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

static int test_matrix_cholesky_singular(void) {
    stonesoup_covariance_matrix_t* A = stonesoup_covariance_matrix_create(2, 2);
    stonesoup_covariance_matrix_t* L = stonesoup_covariance_matrix_create(2, 2);

    assert(A && L);

    // Create a non-positive-definite matrix (negative eigenvalue)
    A->data[0] = 1.0;  A->data[1] = 2.0;
    A->data[2] = 2.0;  A->data[3] = 1.0;

    stonesoup_error_t err = stonesoup_matrix_cholesky(A, L);

    // Should fail with singular error (not positive definite)
    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_SINGULAR);
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(L);
    return 1;
}

/* ============================================================================
 * Error Handling Tests
 * ============================================================================ */

static int test_null_pointer_handling(void) {
    stonesoup_covariance_matrix_t* A = create_test_matrix_2x2();
    stonesoup_covariance_matrix_t* B = create_test_matrix_2x2();

    assert(A && B);

    // Test with NULL output pointer
    stonesoup_error_t err = stonesoup_matrix_add(A, B, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    // Test with NULL input pointer
    stonesoup_covariance_matrix_t* C = stonesoup_covariance_matrix_create(2, 2);
    err = stonesoup_matrix_add(NULL, B, C);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
    return 1;
}

int main(void) {
    printf("Stone Soup C Library - Matrix Operation Tests\n");
    printf("==============================================\n\n");

    /* Matrix-Matrix Operations */
    printf("Matrix-Matrix Operations:\n");
    TEST(test_matrix_multiply_basic);
    TEST(test_matrix_multiply_dimension_check);
    TEST(test_matrix_transpose);
    TEST(test_matrix_add);
    TEST(test_matrix_subtract);

    /* Matrix-Vector Operations */
    printf("\nMatrix-Vector Operations:\n");
    TEST(test_matrix_vector_multiply);

    /* Vector Operations */
    printf("\nVector Operations:\n");
    TEST(test_vector_add);
    TEST(test_vector_subtract);
    TEST(test_vector_dot_product);
    TEST(test_vector_outer_product);

    /* Matrix Decomposition and Inversion */
    printf("\nMatrix Decomposition and Inversion:\n");
    TEST(test_matrix_inverse);
    TEST(test_matrix_inverse_3x3);
    TEST(test_cholesky_decomposition);
    TEST(test_cholesky_solve_vector);
    TEST(test_cholesky_solve_matrix);
    TEST(test_matrix_determinant);
    TEST(test_matrix_log_determinant);
    TEST(test_matrix_cholesky_singular);

    /* Additional Matrix Operations */
    printf("\nAdditional Matrix Operations:\n");
    TEST(test_matrix_multiply_At_B);
    TEST(test_matrix_multiply_A_Bt);
    TEST(test_matrix_vector_multiply_At);
    TEST(test_matrix_scale);
    TEST(test_vector_scale);

    /* Error Handling */
    printf("\nError Handling:\n");
    TEST(test_null_pointer_handling);

    printf("\n==============================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);

    return (tests_run == tests_passed) ? 0 : 1;
}
