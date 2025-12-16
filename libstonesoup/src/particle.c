/**
 * @file particle.c
 * @brief Implementation of particle filter operations
 */

#include "stonesoup/particle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

stonesoup_error_t stonesoup_particle_predict(
    const stonesoup_particle_state_t* prior,
    void (*transition_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    void (*process_noise_func)(stonesoup_state_vector_t*),
    stonesoup_particle_state_t* predicted) {

    if (!prior || !transition_func || !predicted) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (prior->num_particles != predicted->num_particles) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // For each particle, apply transition function and add process noise
    for (size_t i = 0; i < prior->num_particles; i++) {
        // Apply transition function
        transition_func(prior->particles[i].state_vector,
                       predicted->particles[i].state_vector);

        // Add process noise if function provided
        if (process_noise_func) {
            process_noise_func(predicted->particles[i].state_vector);
        }

        // Copy weight to predicted state
        predicted->particles[i].weight = prior->particles[i].weight;
    }

    // Copy timestamp
    predicted->timestamp = prior->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_update(
    const stonesoup_particle_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    double (*likelihood_func)(const stonesoup_state_vector_t*, const stonesoup_state_vector_t*),
    stonesoup_particle_state_t* posterior) {

    if (!predicted || !measurement || !likelihood_func || !posterior) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (predicted->num_particles != posterior->num_particles) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // For each particle, compute likelihood and update weight
    for (size_t i = 0; i < predicted->num_particles; i++) {
        // Compute likelihood of measurement given particle state
        double likelihood = likelihood_func(predicted->particles[i].state_vector, measurement);

        // Update weight: w_new = w_old * likelihood
        posterior->particles[i].weight = predicted->particles[i].weight * likelihood;

        // Copy particle state to posterior
        size_t dim = predicted->particles[i].state_vector->size;
        memcpy(posterior->particles[i].state_vector->data,
               predicted->particles[i].state_vector->data,
               dim * sizeof(double));
    }

    // Copy timestamp
    posterior->timestamp = predicted->timestamp;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_resample(
    stonesoup_particle_state_t* state,
    stonesoup_resample_method_t method) {

    if (!state) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    switch (method) {
        case STONESOUP_RESAMPLE_SYSTEMATIC:
            return stonesoup_particle_systematic_resample(state);
        case STONESOUP_RESAMPLE_STRATIFIED:
            return stonesoup_particle_stratified_resample(state);
        case STONESOUP_RESAMPLE_MULTINOMIAL:
            return stonesoup_particle_multinomial_resample(state);
        case STONESOUP_RESAMPLE_RESIDUAL:
            // TODO: Implement residual resampling
            return STONESOUP_ERROR_NOT_IMPLEMENTED;
        default:
            return STONESOUP_ERROR_INVALID_ARG;
    }
}

stonesoup_error_t stonesoup_particle_effective_sample_size(
    const stonesoup_particle_state_t* state,
    double* n_eff) {

    if (!state || !n_eff) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        double w = state->particles[i].weight;
        sum_sq += w * w;
    }

    if (sum_sq <= 0.0) {
        return STONESOUP_ERROR_INVALID_ARG;
    }

    *n_eff = 1.0 / sum_sq;
    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_mean(
    const stonesoup_particle_state_t* state,
    stonesoup_state_vector_t* mean) {

    if (!state || !mean) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (state->num_particles == 0) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // Get state dimension from first particle
    size_t dim = state->particles[0].state_vector->size;

    if (mean->size != dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    // Initialize mean to zero
    memset(mean->data, 0, dim * sizeof(double));

    // Compute weighted mean
    for (size_t i = 0; i < state->num_particles; i++) {
        double w = state->particles[i].weight;
        for (size_t j = 0; j < dim; j++) {
            mean->data[j] += w * state->particles[i].state_vector->data[j];
        }
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_covariance(
    const stonesoup_particle_state_t* state,
    const stonesoup_state_vector_t* mean,
    stonesoup_covariance_matrix_t* covariance) {

    if (!state || !covariance) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (state->num_particles == 0) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // Get state dimension from first particle
    size_t dim = state->particles[0].state_vector->size;

    if (covariance->rows != dim || covariance->cols != dim) {
        return STONESOUP_ERROR_DIMENSION;
    }

    // Compute or use provided mean
    stonesoup_state_vector_t* mean_temp = NULL;
    const stonesoup_state_vector_t* mean_to_use = mean;

    if (!mean) {
        // Compute mean if not provided
        mean_temp = stonesoup_state_vector_create(dim);
        if (!mean_temp) {
            return STONESOUP_ERROR_ALLOCATION;
        }

        stonesoup_error_t err = stonesoup_particle_mean(state, mean_temp);
        if (err != STONESOUP_SUCCESS) {
            stonesoup_state_vector_free(mean_temp);
            return err;
        }
        mean_to_use = mean_temp;
    }

    // Initialize covariance matrix to zero
    memset(covariance->data, 0, dim * dim * sizeof(double));

    // Compute weighted covariance: Cov = sum_i(w_i * (x_i - mean) * (x_i - mean)^T)
    for (size_t i = 0; i < state->num_particles; i++) {
        double w = state->particles[i].weight;
        const double* x = state->particles[i].state_vector->data;

        for (size_t row = 0; row < dim; row++) {
            double diff_row = x[row] - mean_to_use->data[row];

            for (size_t col = 0; col < dim; col++) {
                double diff_col = x[col] - mean_to_use->data[col];
                // Row-major indexing: covariance->data[row * dim + col]
                covariance->data[row * dim + col] += w * diff_row * diff_col;
            }
        }
    }

    // Free temporary mean if we allocated it
    if (mean_temp) {
        stonesoup_state_vector_free(mean_temp);
    }

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_systematic_resample(
    stonesoup_particle_state_t* state) {

    if (!state) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t N = state->num_particles;
    if (N == 0) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // Get state dimension from first particle
    size_t dim = state->particles[0].state_vector->size;

    // Allocate temporary arrays
    double* cumsum = (double*)malloc(N * sizeof(double));
    size_t* indices = (size_t*)malloc(N * sizeof(size_t));

    if (!cumsum || !indices) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    // Compute cumulative sum of weights
    cumsum[0] = state->particles[0].weight;
    for (size_t i = 1; i < N; i++) {
        cumsum[i] = cumsum[i-1] + state->particles[i].weight;
    }

    // Normalize cumulative sum
    double total_weight = cumsum[N-1];
    if (total_weight <= 0.0) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < N; i++) {
        cumsum[i] /= total_weight;
    }

    // Generate single random starting point u ~ U(0, 1/N)
    double u = ((double)rand() / RAND_MAX) / N;  // NOSONAR c:S2245 - PRNG for simulation

    // Select particle indices using systematic resampling
    size_t j = 0;
    for (size_t i = 0; i < N; i++) {
        double u_i = u + (double)i / N;

        // Find particle index for this position
        while (j < N-1 && cumsum[j] < u_i) {
            j++;
        }

        indices[i] = j;
    }

    free(cumsum);

    // Allocate temporary storage for copying state vectors
    double** temp_data = (double**)malloc(N * sizeof(double*));
    if (!temp_data) {
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    for (size_t i = 0; i < N; i++) {
        temp_data[i] = (double*)malloc(dim * sizeof(double));
        if (!temp_data[i]) {
            for (size_t k = 0; k < i; k++) {
                free(temp_data[k]);
            }
            free(temp_data);
            free(indices);
            return STONESOUP_ERROR_ALLOCATION;
        }
        // Copy the selected particle's data
        memcpy(temp_data[i], state->particles[indices[i]].state_vector->data,
               dim * sizeof(double));
    }

    free(indices);

    // Copy data back to particles and reset weights
    for (size_t i = 0; i < N; i++) {
        memcpy(state->particles[i].state_vector->data, temp_data[i],
               dim * sizeof(double));
        state->particles[i].weight = 1.0 / N;
        free(temp_data[i]);
    }

    free(temp_data);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_stratified_resample(
    stonesoup_particle_state_t* state) {

    if (!state) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t N = state->num_particles;
    if (N == 0) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // Get state dimension from first particle
    size_t dim = state->particles[0].state_vector->size;

    // Allocate temporary arrays
    double* cumsum = (double*)malloc(N * sizeof(double));
    size_t* indices = (size_t*)malloc(N * sizeof(size_t));

    if (!cumsum || !indices) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    // Compute cumulative sum of weights
    cumsum[0] = state->particles[0].weight;
    for (size_t i = 1; i < N; i++) {
        cumsum[i] = cumsum[i-1] + state->particles[i].weight;
    }

    // Normalize cumulative sum
    double total_weight = cumsum[N-1];
    if (total_weight <= 0.0) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < N; i++) {
        cumsum[i] /= total_weight;
    }

    // Divide [0,1] into N strata and generate random point within each stratum
    size_t j = 0;
    for (size_t i = 0; i < N; i++) {
        // Generate random point in stratum [i/N, (i+1)/N]
        double u_i = ((double)i + (double)rand() / RAND_MAX) / N;  // NOSONAR c:S2245 - PRNG for simulation

        // Find particle index for this position
        while (j < N-1 && cumsum[j] < u_i) {
            j++;
        }

        indices[i] = j;
    }

    free(cumsum);

    // Allocate temporary storage for copying state vectors
    double** temp_data = (double**)malloc(N * sizeof(double*));
    if (!temp_data) {
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    for (size_t i = 0; i < N; i++) {
        temp_data[i] = (double*)malloc(dim * sizeof(double));
        if (!temp_data[i]) {
            for (size_t k = 0; k < i; k++) {
                free(temp_data[k]);
            }
            free(temp_data);
            free(indices);
            return STONESOUP_ERROR_ALLOCATION;
        }
        // Copy the selected particle's data
        memcpy(temp_data[i], state->particles[indices[i]].state_vector->data,
               dim * sizeof(double));
    }

    free(indices);

    // Copy data back to particles and reset weights
    for (size_t i = 0; i < N; i++) {
        memcpy(state->particles[i].state_vector->data, temp_data[i],
               dim * sizeof(double));
        state->particles[i].weight = 1.0 / N;
        free(temp_data[i]);
    }

    free(temp_data);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_particle_multinomial_resample(
    stonesoup_particle_state_t* state) {

    if (!state) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    size_t N = state->num_particles;
    if (N == 0) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // Get state dimension from first particle
    size_t dim = state->particles[0].state_vector->size;

    // Allocate temporary arrays
    double* cumsum = (double*)malloc(N * sizeof(double));
    size_t* indices = (size_t*)malloc(N * sizeof(size_t));

    if (!cumsum || !indices) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    // Compute cumulative sum of weights
    cumsum[0] = state->particles[0].weight;
    for (size_t i = 1; i < N; i++) {
        cumsum[i] = cumsum[i-1] + state->particles[i].weight;
    }

    // Normalize cumulative sum
    double total_weight = cumsum[N-1];
    if (total_weight <= 0.0) {
        free(cumsum);
        free(indices);
        return STONESOUP_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < N; i++) {
        cumsum[i] /= total_weight;
    }

    // Generate N random numbers and select particle indices
    for (size_t i = 0; i < N; i++) {
        // Generate random number u ~ U(0, 1)
        double u = (double)rand() / RAND_MAX;  // NOSONAR c:S2245 - PRNG for simulation

        // Find particle index using cumulative distribution
        size_t j = 0;
        while (j < N-1 && cumsum[j] < u) {
            j++;
        }

        indices[i] = j;
    }

    free(cumsum);

    // Allocate temporary storage for copying state vectors
    double** temp_data = (double**)malloc(N * sizeof(double*));
    if (!temp_data) {
        free(indices);
        return STONESOUP_ERROR_ALLOCATION;
    }

    for (size_t i = 0; i < N; i++) {
        temp_data[i] = (double*)malloc(dim * sizeof(double));
        if (!temp_data[i]) {
            for (size_t k = 0; k < i; k++) {
                free(temp_data[k]);
            }
            free(temp_data);
            free(indices);
            return STONESOUP_ERROR_ALLOCATION;
        }
        // Copy the selected particle's data
        memcpy(temp_data[i], state->particles[indices[i]].state_vector->data,
               dim * sizeof(double));
    }

    free(indices);

    // Copy data back to particles and reset weights
    for (size_t i = 0; i < N; i++) {
        memcpy(state->particles[i].state_vector->data, temp_data[i],
               dim * sizeof(double));
        state->particles[i].weight = 1.0 / N;
        free(temp_data[i]);
    }

    free(temp_data);

    return STONESOUP_SUCCESS;
}
