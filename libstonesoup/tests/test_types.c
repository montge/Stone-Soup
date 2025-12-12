/**
 * @file test_types.c
 * @brief Basic tests for Stone Soup type system
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

/* Test version functions */
static int test_version(void) {
    const char* version = stonesoup_version();
    assert(version != NULL);
    assert(stonesoup_version_major() == 0);
    assert(stonesoup_version_minor() == 1);
    assert(stonesoup_version_patch() == 0);
    return 1;
}

/* Test error string function */
static int test_error_strings(void) {
    const char* msg = stonesoup_error_string(STONESOUP_SUCCESS);
    assert(msg != NULL);

    msg = stonesoup_error_string(STONESOUP_ERROR_NULL_POINTER);
    assert(msg != NULL);

    msg = stonesoup_error_string(STONESOUP_ERROR_ALLOCATION);
    assert(msg != NULL);

    return 1;
}

/* Test state vector creation and destruction */
static int test_state_vector_create_free(void) {
    stonesoup_state_vector_t* vec = stonesoup_state_vector_create(4);
    assert(vec != NULL);
    assert(vec->size == 4);
    assert(vec->data != NULL);

    stonesoup_state_vector_free(vec);
    return 1;
}

/* Test state vector fill */
static int test_state_vector_fill(void) {
    stonesoup_state_vector_t* vec = stonesoup_state_vector_create(3);
    assert(vec != NULL);

    stonesoup_error_t err = stonesoup_state_vector_fill(vec, 5.0);
    assert(err == STONESOUP_SUCCESS);

    for (size_t i = 0; i < vec->size; i++) {
        assert(double_equals(vec->data[i], 5.0));
    }

    stonesoup_state_vector_free(vec);
    return 1;
}

/* Test state vector copy */
static int test_state_vector_copy(void) {
    stonesoup_state_vector_t* vec1 = stonesoup_state_vector_create(3);
    assert(vec1 != NULL);

    vec1->data[0] = 1.0;
    vec1->data[1] = 2.0;
    vec1->data[2] = 3.0;

    stonesoup_state_vector_t* vec2 = stonesoup_state_vector_copy(vec1);
    assert(vec2 != NULL);
    assert(vec2->size == vec1->size);

    for (size_t i = 0; i < vec1->size; i++) {
        assert(double_equals(vec2->data[i], vec1->data[i]));
    }

    stonesoup_state_vector_free(vec1);
    stonesoup_state_vector_free(vec2);
    return 1;
}

/* Test covariance matrix creation */
static int test_covariance_matrix_create_free(void) {
    stonesoup_covariance_matrix_t* mat = stonesoup_covariance_matrix_create(3, 3);
    assert(mat != NULL);
    assert(mat->rows == 3);
    assert(mat->cols == 3);
    assert(mat->data != NULL);

    stonesoup_covariance_matrix_free(mat);
    return 1;
}

/* Test identity matrix */
static int test_covariance_matrix_eye(void) {
    stonesoup_covariance_matrix_t* mat = stonesoup_covariance_matrix_create(3, 3);
    assert(mat != NULL);

    stonesoup_error_t err = stonesoup_covariance_matrix_eye(mat);
    assert(err == STONESOUP_SUCCESS);

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert(double_equals(mat->data[i * mat->cols + j], expected));
        }
    }

    stonesoup_covariance_matrix_free(mat);
    return 1;
}

/* Test Gaussian state creation */
static int test_gaussian_state_create_free(void) {
    stonesoup_gaussian_state_t* state = stonesoup_gaussian_state_create(4);
    assert(state != NULL);
    assert(state->state_vector != NULL);
    assert(state->state_vector->size == 4);
    assert(state->covariance != NULL);
    assert(state->covariance->rows == 4);
    assert(state->covariance->cols == 4);

    stonesoup_gaussian_state_free(state);
    return 1;
}

/* Test Gaussian state copy */
static int test_gaussian_state_copy(void) {
    stonesoup_gaussian_state_t* state1 = stonesoup_gaussian_state_create(2);
    assert(state1 != NULL);

    state1->state_vector->data[0] = 1.0;
    state1->state_vector->data[1] = 2.0;
    state1->timestamp = 123.456;

    stonesoup_gaussian_state_t* state2 = stonesoup_gaussian_state_copy(state1);
    assert(state2 != NULL);
    assert(double_equals(state2->state_vector->data[0], 1.0));
    assert(double_equals(state2->state_vector->data[1], 2.0));
    assert(double_equals(state2->timestamp, 123.456));

    stonesoup_gaussian_state_free(state1);
    stonesoup_gaussian_state_free(state2);
    return 1;
}

/* Test particle state creation */
static int test_particle_state_create_free(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(100, 4);
    assert(state != NULL);
    assert(state->num_particles == 100);
    assert(state->particles != NULL);

    for (size_t i = 0; i < state->num_particles; i++) {
        assert(state->particles[i].state_vector != NULL);
        assert(state->particles[i].state_vector->size == 4);
        assert(double_equals(state->particles[i].weight, 1.0 / 100.0));
    }

    stonesoup_particle_state_free(state);
    return 1;
}

/* Test particle weight normalization */
static int test_particle_normalize_weights(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state != NULL);

    // Set arbitrary weights
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].weight = (double)(i + 1);
    }

    stonesoup_error_t err = stonesoup_particle_state_normalize_weights(state);
    assert(err == STONESOUP_SUCCESS);

    // Check that weights sum to 1
    double sum = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        sum += state->particles[i].weight;
    }
    assert(double_equals(sum, 1.0));

    stonesoup_particle_state_free(state);
    return 1;
}

/* Test library initialization */
static int test_init_cleanup(void) {
    stonesoup_error_t err = stonesoup_init();
    assert(err == STONESOUP_SUCCESS);

    stonesoup_cleanup();
    return 1;
}

int main(void) {
    printf("Stone Soup C Library - Type Tests\n");
    printf("==================================\n\n");

    TEST(test_version);
    TEST(test_error_strings);
    TEST(test_state_vector_create_free);
    TEST(test_state_vector_fill);
    TEST(test_state_vector_copy);
    TEST(test_covariance_matrix_create_free);
    TEST(test_covariance_matrix_eye);
    TEST(test_gaussian_state_create_free);
    TEST(test_gaussian_state_copy);
    TEST(test_particle_state_create_free);
    TEST(test_particle_normalize_weights);
    TEST(test_init_cleanup);

    printf("\n==================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);

    return (tests_run == tests_passed) ? 0 : 1;
}
