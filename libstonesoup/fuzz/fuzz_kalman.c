/**
 * @file fuzz_kalman.c
 * @brief Fuzzing harness for Kalman filter API boundaries
 *
 * This file provides fuzzing targets for libFuzzer/AFL to test
 * the robustness of the Stone Soup C library API boundaries.
 *
 * Build with libFuzzer:
 *   clang -g -O1 -fsanitize=fuzzer,address,undefined \
 *         -I../include fuzz_kalman.c ../src/*.c -lm -o fuzz_kalman
 *
 * Build with AFL:
 *   afl-gcc -g -O1 -fsanitize=address,undefined \
 *           -I../include fuzz_kalman.c ../src/*.c -lm -o fuzz_kalman_afl
 *
 * Run with libFuzzer:
 *   ./fuzz_kalman -max_len=1024 corpus/
 *
 * Run with AFL:
 *   afl-fuzz -i corpus/ -o findings/ ./fuzz_kalman_afl @@
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stonesoup/stonesoup.h"

/* Maximum dimensions to fuzz (prevent OOM) */
#define MAX_FUZZ_DIM 16
#define MIN_FUZZ_DIM 1

/**
 * @brief Extract a bounded integer from fuzzer data
 */
static size_t extract_bounded_size(const uint8_t **data, size_t *size,
                                   size_t min_val, size_t max_val) {
    if (*size < sizeof(uint8_t)) {
        return min_val;
    }
    uint8_t raw = **data;
    (*data)++;
    (*size)--;

    size_t range = max_val - min_val + 1;
    return min_val + (raw % range);
}

/**
 * @brief Extract a double from fuzzer data
 */
static double extract_double(const uint8_t **data, size_t *size) {
    if (*size < sizeof(double)) {
        return 0.0;
    }
    double val;
    memcpy(&val, *data, sizeof(double));
    *data += sizeof(double);
    *size -= sizeof(double);

    /* Sanitize NaN and Inf */
    if (!isfinite(val)) {
        return 0.0;
    }
    /* Clamp to reasonable range */
    if (val > 1e10) val = 1e10;
    if (val < -1e10) val = -1e10;
    return val;
}

/**
 * @brief Fill a state vector from fuzzer data
 */
static int fill_state_vector(stonesoup_state_vector_t *vec,
                             const uint8_t **data, size_t *size) {
    if (!vec || !vec->data) return -1;

    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = extract_double(data, size);
    }
    return 0;
}

/**
 * @brief Fill a covariance matrix from fuzzer data (ensure positive semi-definite)
 */
static int fill_covariance_matrix(stonesoup_covariance_matrix_t *mat,
                                  const uint8_t **data, size_t *size) {
    if (!mat || !mat->data) return -1;

    /* Fill with symmetric values */
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = i; j < mat->cols; j++) {
            double val = extract_double(data, size);
            mat->data[i * mat->cols + j] = val;
            mat->data[j * mat->cols + i] = val;
        }
    }

    /* Make positive semi-definite by adding to diagonal */
    for (size_t i = 0; i < mat->rows; i++) {
        mat->data[i * mat->cols + i] += 10.0; /* Ensure positive eigenvalues */
    }

    return 0;
}

/**
 * @brief Fuzz target for state vector operations
 */
static void fuzz_state_vector(const uint8_t *data, size_t size) {
    size_t dim = extract_bounded_size(&data, &size, MIN_FUZZ_DIM, MAX_FUZZ_DIM);

    stonesoup_state_vector_t *vec1 = stonesoup_state_vector_create(dim);
    stonesoup_state_vector_t *vec2 = stonesoup_state_vector_create(dim);
    stonesoup_state_vector_t *result = stonesoup_state_vector_create(dim);

    if (!vec1 || !vec2 || !result) goto cleanup;

    fill_state_vector(vec1, &data, &size);
    fill_state_vector(vec2, &data, &size);

    /* Test vector operations */
    stonesoup_vector_add(vec1, vec2, result);
    stonesoup_vector_subtract(vec1, vec2, result);
    stonesoup_vector_scale(vec1, extract_double(&data, &size), result);

    double dot_result;
    stonesoup_vector_dot(vec1, vec2, &dot_result);

cleanup:
    stonesoup_state_vector_free(vec1);
    stonesoup_state_vector_free(vec2);
    stonesoup_state_vector_free(result);
}

/**
 * @brief Fuzz target for matrix operations
 */
static void fuzz_matrix_ops(const uint8_t *data, size_t size) {
    size_t dim = extract_bounded_size(&data, &size, MIN_FUZZ_DIM, MAX_FUZZ_DIM);

    stonesoup_covariance_matrix_t *A = stonesoup_covariance_matrix_create(dim, dim);
    stonesoup_covariance_matrix_t *B = stonesoup_covariance_matrix_create(dim, dim);
    stonesoup_covariance_matrix_t *C = stonesoup_covariance_matrix_create(dim, dim);

    if (!A || !B || !C) goto cleanup;

    fill_covariance_matrix(A, &data, &size);
    fill_covariance_matrix(B, &data, &size);

    /* Test matrix operations */
    stonesoup_matrix_add(A, B, C);
    stonesoup_matrix_subtract(A, B, C);
    stonesoup_matrix_multiply(A, B, C);
    stonesoup_matrix_transpose(A, C);
    stonesoup_matrix_scale(A, extract_double(&data, &size), C);

    /* Test Cholesky (may fail for some inputs, that's expected) */
    stonesoup_matrix_cholesky(A, C);

    /* Test inverse (may fail for singular matrices) */
    stonesoup_matrix_inverse(A, C);

cleanup:
    stonesoup_covariance_matrix_free(A);
    stonesoup_covariance_matrix_free(B);
    stonesoup_covariance_matrix_free(C);
}

/**
 * @brief Fuzz target for Kalman filter operations
 */
static void fuzz_kalman_filter(const uint8_t *data, size_t size) {
    size_t state_dim = extract_bounded_size(&data, &size, MIN_FUZZ_DIM, MAX_FUZZ_DIM);
    size_t meas_dim = extract_bounded_size(&data, &size, MIN_FUZZ_DIM, state_dim);

    /* Create states */
    stonesoup_gaussian_state_t *prior = stonesoup_gaussian_state_create(state_dim);
    stonesoup_gaussian_state_t *predicted = stonesoup_gaussian_state_create(state_dim);
    stonesoup_gaussian_state_t *posterior = stonesoup_gaussian_state_create(state_dim);

    /* Create matrices */
    stonesoup_covariance_matrix_t *F = stonesoup_covariance_matrix_create(state_dim, state_dim);
    stonesoup_covariance_matrix_t *Q = stonesoup_covariance_matrix_create(state_dim, state_dim);
    stonesoup_covariance_matrix_t *H = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    stonesoup_covariance_matrix_t *R = stonesoup_covariance_matrix_create(meas_dim, meas_dim);

    /* Create measurement */
    stonesoup_state_vector_t *z = stonesoup_state_vector_create(meas_dim);

    if (!prior || !predicted || !posterior || !F || !Q || !H || !R || !z) {
        goto cleanup;
    }

    /* Fill with fuzz data */
    fill_state_vector(prior->state_vector, &data, &size);
    fill_covariance_matrix(prior->covariance, &data, &size);
    fill_covariance_matrix(F, &data, &size);
    fill_covariance_matrix(Q, &data, &size);
    fill_covariance_matrix(R, &data, &size);
    fill_state_vector(z, &data, &size);

    /* Fill H as a simple selection matrix */
    for (size_t i = 0; i < meas_dim && i < state_dim; i++) {
        H->data[i * state_dim + i] = 1.0;
    }

    /* Test Kalman predict */
    stonesoup_error_t err = stonesoup_kalman_predict(prior, F, Q, predicted);
    (void)err; /* Expected to succeed or fail gracefully */

    /* Test Kalman update */
    if (err == STONESOUP_SUCCESS) {
        stonesoup_kalman_update(predicted, z, H, R, posterior);
    }

cleanup:
    stonesoup_gaussian_state_free(prior);
    stonesoup_gaussian_state_free(predicted);
    stonesoup_gaussian_state_free(posterior);
    stonesoup_covariance_matrix_free(F);
    stonesoup_covariance_matrix_free(Q);
    stonesoup_covariance_matrix_free(H);
    stonesoup_covariance_matrix_free(R);
    stonesoup_state_vector_free(z);
}

/**
 * @brief Fuzz target for null pointer handling
 */
static void fuzz_null_handling(const uint8_t *data, size_t size) {
    /* Test that null pointers are handled gracefully */
    stonesoup_state_vector_free(NULL);
    stonesoup_covariance_matrix_free(NULL);
    stonesoup_gaussian_state_free(NULL);

    stonesoup_state_vector_t *vec = NULL;
    stonesoup_covariance_matrix_t *mat = NULL;

    /* These should return error codes, not crash */
    stonesoup_state_vector_fill(vec, 0.0);
    stonesoup_covariance_matrix_eye(mat);

    /* Test zero-size creation (should fail gracefully) */
    vec = stonesoup_state_vector_create(0);
    stonesoup_state_vector_free(vec);

    mat = stonesoup_covariance_matrix_create(0, 0);
    stonesoup_covariance_matrix_free(mat);
}

/**
 * @brief Main libFuzzer entry point
 */
#ifdef __cplusplus
extern "C"
#endif
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 1) return 0;

    /* Use first byte to select which function to fuzz */
    uint8_t selector = data[0] % 4;
    data++;
    size--;

    switch (selector) {
        case 0:
            fuzz_state_vector(data, size);
            break;
        case 1:
            fuzz_matrix_ops(data, size);
            break;
        case 2:
            fuzz_kalman_filter(data, size);
            break;
        case 3:
            fuzz_null_handling(data, size);
            break;
    }

    return 0;
}

/**
 * @brief Main entry point for AFL
 */
#ifndef __AFL_FUZZ_TESTCASE_LEN
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(size);
    if (!data) {
        fclose(f);
        return 1;
    }

    if (fread(data, 1, size, f) != size) {
        free(data);
        fclose(f);
        return 1;
    }

    fclose(f);

    LLVMFuzzerTestOneInput(data, size);

    free(data);
    return 0;
}
#endif
