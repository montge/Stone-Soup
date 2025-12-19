/**
 * @file types.c
 * @brief Implementation of core type operations
 */

#include "stonesoup/types.h"
#include "stonesoup/version.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Version Functions */

const char* stonesoup_version(void) {
    return STONESOUP_VERSION_STRING;
}

int stonesoup_version_major(void) {
    return STONESOUP_VERSION_MAJOR;
}

int stonesoup_version_minor(void) {
    return STONESOUP_VERSION_MINOR;
}

int stonesoup_version_patch(void) {
    return STONESOUP_VERSION_PATCH;
}

/* Error Handling */

const char* stonesoup_error_string(stonesoup_error_t error) {
    switch (error) {
        case STONESOUP_SUCCESS:
            return "Success";
        case STONESOUP_ERROR_NULL_POINTER:
            return "Null pointer argument";
        case STONESOUP_ERROR_INVALID_SIZE:
            return "Invalid size parameter";
        case STONESOUP_ERROR_ALLOCATION:
            return "Memory allocation failed";
        case STONESOUP_ERROR_DIMENSION:
            return "Dimension mismatch";
        case STONESOUP_ERROR_SINGULAR:
            return "Singular matrix";
        case STONESOUP_ERROR_INVALID_ARG:
            return "Invalid argument";
        case STONESOUP_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        default:
            return "Unknown error";
    }
}

/* State Vector Functions */

stonesoup_state_vector_t* stonesoup_state_vector_create(size_t size) {
    if (size == 0) {
        return NULL;
    }

    stonesoup_state_vector_t* vec = malloc(sizeof(stonesoup_state_vector_t));
    if (!vec) {
        return NULL;
    }

    vec->data = calloc(size, sizeof(double));
    if (!vec->data) {
        free(vec);
        return NULL;
    }

    vec->size = size;
    return vec;
}

void stonesoup_state_vector_free(stonesoup_state_vector_t* vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

stonesoup_state_vector_t* stonesoup_state_vector_copy(const stonesoup_state_vector_t* src) {
    if (!src) {
        return NULL;
    }

    stonesoup_state_vector_t* dst = stonesoup_state_vector_create(src->size);
    if (!dst) {
        return NULL;
    }

    memcpy(dst->data, src->data, src->size * sizeof(double));
    return dst;
}

stonesoup_error_t stonesoup_state_vector_fill(stonesoup_state_vector_t* vec, double value) {
    if (!vec || !vec->data) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = value;
    }

    return STONESOUP_SUCCESS;
}

/* Covariance Matrix Functions */

stonesoup_covariance_matrix_t* stonesoup_covariance_matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }

    stonesoup_covariance_matrix_t* mat = malloc(sizeof(stonesoup_covariance_matrix_t));
    if (!mat) {
        return NULL;
    }

    mat->data = calloc(rows * cols, sizeof(double));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

void stonesoup_covariance_matrix_free(stonesoup_covariance_matrix_t* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

stonesoup_covariance_matrix_t* stonesoup_covariance_matrix_copy(
    const stonesoup_covariance_matrix_t* src) {
    if (!src) {
        return NULL;
    }

    stonesoup_covariance_matrix_t* dst = stonesoup_covariance_matrix_create(src->rows, src->cols);
    if (!dst) {
        return NULL;
    }

    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
    return dst;
}

stonesoup_error_t stonesoup_covariance_matrix_eye(stonesoup_covariance_matrix_t* mat) {
    if (!mat || !mat->data) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (mat->rows != mat->cols) {
        return STONESOUP_ERROR_DIMENSION;
    }

    // Zero all elements
    memset(mat->data, 0, mat->rows * mat->cols * sizeof(double));

    // Set diagonal to 1
    for (size_t i = 0; i < mat->rows; i++) {
        mat->data[i * mat->cols + i] = 1.0;
    }

    return STONESOUP_SUCCESS;
}

/* Gaussian State Functions */

stonesoup_gaussian_state_t* stonesoup_gaussian_state_create(size_t state_dim) {
    if (state_dim == 0) {
        return NULL;
    }

    stonesoup_gaussian_state_t* state = malloc(sizeof(stonesoup_gaussian_state_t));
    if (!state) {
        return NULL;
    }

    state->state_vector = stonesoup_state_vector_create(state_dim);
    if (!state->state_vector) {
        free(state);
        return NULL;
    }

    state->covariance = stonesoup_covariance_matrix_create(state_dim, state_dim);
    if (!state->covariance) {
        stonesoup_state_vector_free(state->state_vector);
        free(state);
        return NULL;
    }

    state->timestamp = 0.0;
    return state;
}

void stonesoup_gaussian_state_free(stonesoup_gaussian_state_t* state) {
    if (state) {
        stonesoup_state_vector_free(state->state_vector);
        stonesoup_covariance_matrix_free(state->covariance);
        free(state);
    }
}

stonesoup_gaussian_state_t* stonesoup_gaussian_state_copy(const stonesoup_gaussian_state_t* src) {
    if (!src) {
        return NULL;
    }

    stonesoup_gaussian_state_t* dst = malloc(sizeof(stonesoup_gaussian_state_t));
    if (!dst) {
        return NULL;
    }

    dst->state_vector = stonesoup_state_vector_copy(src->state_vector);
    if (!dst->state_vector) {
        free(dst);
        return NULL;
    }

    dst->covariance = stonesoup_covariance_matrix_copy(src->covariance);
    if (!dst->covariance) {
        stonesoup_state_vector_free(dst->state_vector);
        free(dst);
        return NULL;
    }

    dst->timestamp = src->timestamp;
    return dst;
}

/* Particle State Functions */

stonesoup_particle_state_t* stonesoup_particle_state_create(
    size_t num_particles, size_t state_dim) {
    if (num_particles == 0 || state_dim == 0) {
        return NULL;
    }

    stonesoup_particle_state_t* state = malloc(sizeof(stonesoup_particle_state_t));
    if (!state) {
        return NULL;
    }

    state->particles = calloc(num_particles, sizeof(stonesoup_particle_t));
    if (!state->particles) {
        free(state);
        return NULL;
    }

    // Initialize each particle
    for (size_t i = 0; i < num_particles; i++) {
        state->particles[i].state_vector = stonesoup_state_vector_create(state_dim);
        if (!state->particles[i].state_vector) {
            // Clean up already allocated particles
            for (size_t j = 0; j < i; j++) {
                stonesoup_state_vector_free(state->particles[j].state_vector);
            }
            free(state->particles);
            free(state);
            return NULL;
        }
        state->particles[i].weight = 1.0 / num_particles;
    }

    state->num_particles = num_particles;
    state->timestamp = 0.0;
    return state;
}

void stonesoup_particle_state_free(stonesoup_particle_state_t* state) {
    if (state) {
        if (state->particles) {
            for (size_t i = 0; i < state->num_particles; i++) {
                stonesoup_state_vector_free(state->particles[i].state_vector);
            }
            free(state->particles);
        }
        free(state);
    }
}

stonesoup_error_t stonesoup_particle_state_normalize_weights(
    stonesoup_particle_state_t* state) {
    if (!state || !state->particles) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    double sum = 0.0;
    for (size_t i = 0; i < state->num_particles; i++) {
        sum += state->particles[i].weight;
    }

    if (sum <= 0.0) {
        return STONESOUP_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < state->num_particles; i++) {
        state->particles[i].weight /= sum;
    }

    return STONESOUP_SUCCESS;
}

/* Library Initialization */

stonesoup_error_t stonesoup_init(void) {
    // Currently no initialization needed
    return STONESOUP_SUCCESS;
}

void stonesoup_cleanup(void) {
    // Currently no cleanup needed
}
