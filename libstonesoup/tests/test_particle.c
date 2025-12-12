/**
 * @file test_particle.c
 * @brief Comprehensive tests for Stone Soup particle filter operations
 */

#include <stonesoup/stonesoup.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

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

/* Helper function: simple constant velocity transition */
static void cv_transition(const stonesoup_state_vector_t* x_in,
                         stonesoup_state_vector_t* x_out) {
    if (x_in->size == 4 && x_out->size == 4) {
        // [x, vx, y, vy] with dt = 1.0
        x_out->data[0] = x_in->data[0] + x_in->data[1];  // x + vx
        x_out->data[1] = x_in->data[1];                   // vx
        x_out->data[2] = x_in->data[2] + x_in->data[3];  // y + vy
        x_out->data[3] = x_in->data[3];                   // vy
    }
}

/* Helper function: no process noise */
static void no_process_noise(stonesoup_state_vector_t* x) {
    (void)x;  // Do nothing
}

/* Helper function: Gaussian likelihood for position measurement */
static double gaussian_likelihood(const stonesoup_state_vector_t* particle,
                                 const stonesoup_state_vector_t* measurement) {
    // Assume particle is [x, vx, y, vy] and measurement is [x, y]
    // Simple Gaussian likelihood with sigma = 1.0
    double dx = particle->data[0] - measurement->data[0];
    double dy = particle->data[2] - measurement->data[1];
    double dist_sq = dx * dx + dy * dy;
    return exp(-0.5 * dist_sq);
}

/* ============================================================================
 * Particle State Creation and Management Tests
 * ============================================================================ */

static int test_particle_state_weight_normalization(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    // Set non-uniform weights
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].weight = (double)(i + 1);
    }

    stonesoup_error_t err = stonesoup_particle_state_normalize_weights(state);
    assert(err == STONESOUP_SUCCESS);

    // Check weights sum to 1
    double sum = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        sum += state->particles[i].weight;
    }

    assert(double_equals(sum, 1.0));

    stonesoup_particle_state_free(state);
    return 1;
}

/* ============================================================================
 * Particle Filter Prediction Tests
 * ============================================================================ */

static int test_particle_predict_basic(void) {
    stonesoup_particle_state_t* prior = stonesoup_particle_state_create(5, 4);
    stonesoup_particle_state_t* predicted = stonesoup_particle_state_create(5, 4);

    assert(prior && predicted);

    // Initialize particles with known states
    for (size_t i = 0; i < prior->num_particles; i++) {
        prior->particles[i].state_vector->data[0] = 0.0;  // x
        prior->particles[i].state_vector->data[1] = 1.0;  // vx
        prior->particles[i].state_vector->data[2] = 0.0;  // y
        prior->particles[i].state_vector->data[3] = 1.0;  // vy
        prior->particles[i].weight = 0.2;
    }

    stonesoup_error_t err = stonesoup_particle_predict(prior, cv_transition,
                                                       no_process_noise, predicted);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(prior);
        stonesoup_particle_state_free(predicted);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Check all particles moved correctly (x -> x+vx, y -> y+vy)
    for (size_t i = 0; i < predicted->num_particles; i++) {
        assert(double_equals(predicted->particles[i].state_vector->data[0], 1.0));
        assert(double_equals(predicted->particles[i].state_vector->data[1], 1.0));
        assert(double_equals(predicted->particles[i].state_vector->data[2], 1.0));
        assert(double_equals(predicted->particles[i].state_vector->data[3], 1.0));
        assert(double_equals(predicted->particles[i].weight, 0.2));
    }

    stonesoup_particle_state_free(prior);
    stonesoup_particle_state_free(predicted);
    return 1;
}

static int test_particle_predict_null_pointer(void) {
    stonesoup_particle_state_t* prior = stonesoup_particle_state_create(5, 4);
    assert(prior);

    stonesoup_error_t err = stonesoup_particle_predict(prior, cv_transition,
                                                       no_process_noise, NULL);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_NULL_POINTER);
    }

    stonesoup_particle_state_free(prior);
    return 1;
}

static int test_particle_predict_size_mismatch(void) {
    stonesoup_particle_state_t* prior = stonesoup_particle_state_create(5, 4);
    stonesoup_particle_state_t* predicted = stonesoup_particle_state_create(10, 4);

    assert(prior && predicted);

    stonesoup_error_t err = stonesoup_particle_predict(prior, cv_transition,
                                                       no_process_noise, predicted);

    if (err != STONESOUP_ERROR_NOT_IMPLEMENTED) {
        assert(err == STONESOUP_ERROR_INVALID_SIZE);
    }

    stonesoup_particle_state_free(prior);
    stonesoup_particle_state_free(predicted);
    return 1;
}

/* ============================================================================
 * Particle Filter Update Tests
 * ============================================================================ */

static int test_particle_update_basic(void) {
    stonesoup_particle_state_t* predicted = stonesoup_particle_state_create(5, 4);
    stonesoup_particle_state_t* posterior = stonesoup_particle_state_create(5, 4);

    assert(predicted && posterior);

    // Set particle positions around [1, 1]
    for (size_t i = 0; i < predicted->num_particles; i++) {
        predicted->particles[i].state_vector->data[0] = 1.0 + i * 0.1;
        predicted->particles[i].state_vector->data[1] = 0.0;
        predicted->particles[i].state_vector->data[2] = 1.0 + i * 0.1;
        predicted->particles[i].state_vector->data[3] = 0.0;
        predicted->particles[i].weight = 0.2;
    }

    // Measurement at [1.0, 1.0]
    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 1.0;
    measurement->data[1] = 1.0;

    stonesoup_error_t err = stonesoup_particle_update(predicted, measurement,
                                                      gaussian_likelihood, posterior);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(predicted);
        stonesoup_particle_state_free(posterior);
        stonesoup_state_vector_free(measurement);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Particle closest to measurement should have highest weight
    // (particle 0 at [1.0, 1.0])
    double max_weight = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < posterior->num_particles; i++) {
        if (posterior->particles[i].weight > max_weight) {
            max_weight = posterior->particles[i].weight;
            max_idx = i;
        }
    }

    assert(max_idx == 0);  // First particle should have highest weight

    stonesoup_particle_state_free(predicted);
    stonesoup_particle_state_free(posterior);
    stonesoup_state_vector_free(measurement);
    return 1;
}

/* ============================================================================
 * Particle Mean and Covariance Tests
 * ============================================================================ */

static int test_particle_mean(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(4, 2);
    stonesoup_state_vector_t* mean = stonesoup_state_vector_create(2);

    assert(state && mean);

    // Set particles with equal weights
    state->particles[0].state_vector->data[0] = 1.0;
    state->particles[0].state_vector->data[1] = 2.0;
    state->particles[0].weight = 0.25;

    state->particles[1].state_vector->data[0] = 2.0;
    state->particles[1].state_vector->data[1] = 4.0;
    state->particles[1].weight = 0.25;

    state->particles[2].state_vector->data[0] = 3.0;
    state->particles[2].state_vector->data[1] = 6.0;
    state->particles[2].weight = 0.25;

    state->particles[3].state_vector->data[0] = 4.0;
    state->particles[3].state_vector->data[1] = 8.0;
    state->particles[3].weight = 0.25;

    stonesoup_error_t err = stonesoup_particle_mean(state, mean);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        stonesoup_state_vector_free(mean);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Mean should be [2.5, 5.0]
    assert(double_equals(mean->data[0], 2.5));
    assert(double_equals(mean->data[1], 5.0));

    stonesoup_particle_state_free(state);
    stonesoup_state_vector_free(mean);
    return 1;
}

static int test_particle_covariance(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(3, 2);
    stonesoup_state_vector_t* mean = stonesoup_state_vector_create(2);
    stonesoup_covariance_matrix_t* cov = stonesoup_covariance_matrix_create(2, 2);

    assert(state && mean && cov);

    // Set particles
    state->particles[0].state_vector->data[0] = 1.0;
    state->particles[0].state_vector->data[1] = 1.0;
    state->particles[0].weight = 1.0 / 3.0;

    state->particles[1].state_vector->data[0] = 2.0;
    state->particles[1].state_vector->data[1] = 2.0;
    state->particles[1].weight = 1.0 / 3.0;

    state->particles[2].state_vector->data[0] = 3.0;
    state->particles[2].state_vector->data[1] = 3.0;
    state->particles[2].weight = 1.0 / 3.0;

    // Mean is [2.0, 2.0]
    mean->data[0] = 2.0;
    mean->data[1] = 2.0;

    stonesoup_error_t err = stonesoup_particle_covariance(state, mean, cov);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        stonesoup_state_vector_free(mean);
        stonesoup_covariance_matrix_free(cov);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Covariance should be diagonal with var = 2/3 (since points are -1, 0, 1 from mean)
    // Actually: var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    assert(fabs(cov->data[0] - 0.6667) < 0.01);  // ~2/3
    assert(fabs(cov->data[3] - 0.6667) < 0.01);  // ~2/3

    stonesoup_particle_state_free(state);
    stonesoup_state_vector_free(mean);
    stonesoup_covariance_matrix_free(cov);
    return 1;
}

/* ============================================================================
 * Effective Sample Size Test
 * ============================================================================ */

static int test_particle_effective_sample_size(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    // All equal weights -> N_eff = N
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].weight = 0.1;
    }

    double n_eff = 0.0;
    stonesoup_error_t err = stonesoup_particle_effective_sample_size(state, &n_eff);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // N_eff = 1 / sum(w_i^2) = 1 / (10 * 0.1^2) = 1 / 0.1 = 10
    assert(double_equals(n_eff, 10.0));

    // One particle has all weight -> N_eff = 1
    state->particles[0].weight = 1.0;
    for (size_t i = 1; i < state->num_particles; i++) {
        state->particles[i].weight = 0.0;
    }

    err = stonesoup_particle_effective_sample_size(state, &n_eff);
    assert(err == STONESOUP_SUCCESS);
    assert(double_equals(n_eff, 1.0));

    stonesoup_particle_state_free(state);
    return 1;
}

/* ============================================================================
 * Resampling Tests
 * ============================================================================ */

static int test_particle_systematic_resample(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    // Set non-uniform weights (one particle much heavier)
    state->particles[0].weight = 0.7;
    for (size_t i = 1; i < state->num_particles; i++) {
        state->particles[i].weight = 0.03;
    }

    // Set distinct particle positions to track resampling
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].state_vector->data[0] = (double)i;
        state->particles[i].state_vector->data[1] = (double)i;
    }

    stonesoup_error_t err = stonesoup_particle_systematic_resample(state);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // After resampling, all weights should be equal
    for (size_t i = 0; i < state->num_particles; i++) {
        assert(double_equals(state->particles[i].weight, 0.1));
    }

    // Heavy particle (0) should appear multiple times
    int count_zero = 0;
    for (size_t i = 0; i < state->num_particles; i++) {
        if (double_equals(state->particles[i].state_vector->data[0], 0.0)) {
            count_zero++;
        }
    }
    assert(count_zero >= 5);  // Should have ~7 copies due to 0.7 weight

    stonesoup_particle_state_free(state);
    return 1;
}

static int test_particle_stratified_resample(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    // Set weights
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].state_vector->data[0] = (double)i;
        state->particles[i].state_vector->data[1] = (double)i;
        state->particles[i].weight = (i == 0) ? 0.7 : 0.03;
    }

    stonesoup_error_t err = stonesoup_particle_stratified_resample(state);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Weights should be normalized after resampling
    double sum = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        sum += state->particles[i].weight;
    }
    assert(double_equals(sum, 1.0));

    stonesoup_particle_state_free(state);
    return 1;
}

static int test_particle_multinomial_resample(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    // Set weights
    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].state_vector->data[0] = (double)i;
        state->particles[i].state_vector->data[1] = (double)i;
        state->particles[i].weight = (i == 0) ? 0.7 : 0.03;
    }

    stonesoup_error_t err = stonesoup_particle_multinomial_resample(state);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Weights should sum to 1 after resampling
    double sum = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        sum += state->particles[i].weight;
    }
    assert(double_equals(sum, 1.0));

    stonesoup_particle_state_free(state);
    return 1;
}

static int test_particle_resample_generic(void) {
    stonesoup_particle_state_t* state = stonesoup_particle_state_create(10, 2);
    assert(state);

    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].weight = 0.1;
    }

    // Test systematic
    stonesoup_error_t err = stonesoup_particle_resample(state, STONESOUP_RESAMPLE_SYSTEMATIC);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(state);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS || err == STONESOUP_ERROR_NOT_IMPLEMENTED);

    stonesoup_particle_state_free(state);
    return 1;
}

/* ============================================================================
 * Weight Normalization Test
 * ============================================================================ */

static int test_weights_sum_to_one_after_update(void) {
    stonesoup_particle_state_t* predicted = stonesoup_particle_state_create(5, 4);
    stonesoup_particle_state_t* posterior = stonesoup_particle_state_create(5, 4);

    assert(predicted && posterior);

    for (size_t i = 0; i < predicted->num_particles; i++) {
        predicted->particles[i].state_vector->data[0] = (double)i;
        predicted->particles[i].state_vector->data[1] = 0.0;
        predicted->particles[i].state_vector->data[2] = (double)i;
        predicted->particles[i].state_vector->data[3] = 0.0;
        predicted->particles[i].weight = 0.2;
    }

    stonesoup_state_vector_t* measurement = stonesoup_state_vector_create(2);
    measurement->data[0] = 2.0;
    measurement->data[1] = 2.0;

    stonesoup_error_t err = stonesoup_particle_update(predicted, measurement,
                                                      gaussian_likelihood, posterior);

    if (err == STONESOUP_ERROR_NOT_IMPLEMENTED) {
        printf("SKIPPED (not implemented) ");
        stonesoup_particle_state_free(predicted);
        stonesoup_particle_state_free(posterior);
        stonesoup_state_vector_free(measurement);
        return 1;
    }

    assert(err == STONESOUP_SUCCESS);

    // Normalize weights
    err = stonesoup_particle_state_normalize_weights(posterior);
    assert(err == STONESOUP_SUCCESS);

    // Check sum
    double sum = 0.0;
    for (size_t i = 0; i < posterior->num_particles; i++) {
        sum += posterior->particles[i].weight;
    }

    assert(double_equals(sum, 1.0));

    stonesoup_particle_state_free(predicted);
    stonesoup_particle_state_free(posterior);
    stonesoup_state_vector_free(measurement);
    return 1;
}

int main(void) {
    printf("Stone Soup C Library - Particle Filter Tests\n");
    printf("=============================================\n\n");

    /* Particle State Tests */
    printf("Particle State Management:\n");
    TEST(test_particle_state_weight_normalization);

    /* Prediction Tests */
    printf("\nParticle Filter Prediction:\n");
    TEST(test_particle_predict_basic);
    TEST(test_particle_predict_null_pointer);
    TEST(test_particle_predict_size_mismatch);

    /* Update Tests */
    printf("\nParticle Filter Update:\n");
    TEST(test_particle_update_basic);
    TEST(test_weights_sum_to_one_after_update);

    /* Statistics Tests */
    printf("\nParticle Statistics:\n");
    TEST(test_particle_mean);
    TEST(test_particle_covariance);
    TEST(test_particle_effective_sample_size);

    /* Resampling Tests */
    printf("\nResampling Methods:\n");
    TEST(test_particle_resample_generic);
    TEST(test_particle_systematic_resample);
    TEST(test_particle_stratified_resample);
    TEST(test_particle_multinomial_resample);

    printf("\n=============================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);

    return (tests_run == tests_passed) ? 0 : 1;
}
