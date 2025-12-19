/**
 * @file types.h
 * @brief Core type definitions for Stone Soup C library
 */

#ifndef STONESOUP_TYPES_H
#define STONESOUP_TYPES_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup types Core Types
 * @brief Fundamental data structures for state estimation
 * @{
 */

/* DLL export/import macros */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef STONESOUP_BUILD_SHARED
    #define STONESOUP_EXPORT __declspec(dllexport)
  #else
    #define STONESOUP_EXPORT __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define STONESOUP_EXPORT __attribute__((visibility("default")))
  #else
    #define STONESOUP_EXPORT
  #endif
#endif

/**
 * @brief Error codes returned by library functions
 */
typedef enum {
    STONESOUP_SUCCESS = 0,          /**< Operation successful */
    STONESOUP_ERROR_NULL_POINTER,   /**< Null pointer argument */
    STONESOUP_ERROR_INVALID_SIZE,   /**< Invalid size parameter */
    STONESOUP_ERROR_ALLOCATION,     /**< Memory allocation failed */
    STONESOUP_ERROR_DIMENSION,      /**< Dimension mismatch */
    STONESOUP_ERROR_SINGULAR,       /**< Singular matrix */
    STONESOUP_ERROR_INVALID_ARG,    /**< Invalid argument */
    STONESOUP_ERROR_NOT_IMPLEMENTED /**< Feature not implemented */
} stonesoup_error_t;

/**
 * @brief State vector representation
 *
 * Represents an n-dimensional state vector. Data is stored in column-major
 * format compatible with common linear algebra libraries.
 */
typedef struct {
    double* data;     /**< Pointer to vector data */
    size_t size;      /**< Number of elements */
} stonesoup_state_vector_t;

/**
 * @brief Covariance matrix representation
 *
 * Represents an n×n covariance matrix. Data is stored in row-major format.
 * The matrix is assumed to be symmetric and positive semi-definite.
 */
typedef struct {
    double* data;     /**< Pointer to matrix data (row-major) */
    size_t rows;      /**< Number of rows */
    size_t cols;      /**< Number of columns */
} stonesoup_covariance_matrix_t;

/**
 * @brief Gaussian state representation
 *
 * Represents a state with Gaussian uncertainty characterized by a mean
 * state vector and covariance matrix.
 */
typedef struct {
    stonesoup_state_vector_t* state_vector;        /**< Mean state vector */
    stonesoup_covariance_matrix_t* covariance;     /**< Covariance matrix */
    double timestamp;                               /**< Time stamp (optional) */
} stonesoup_gaussian_state_t;

/**
 * @brief Particle representation
 *
 * Represents a single particle with state vector and weight for
 * particle filtering applications.
 */
typedef struct {
    stonesoup_state_vector_t* state_vector;  /**< Particle state */
    double weight;                            /**< Particle weight */
} stonesoup_particle_t;

/**
 * @brief Particle state representation
 *
 * Represents a state distribution as a collection of weighted particles.
 */
typedef struct {
    stonesoup_particle_t* particles;  /**< Array of particles */
    size_t num_particles;             /**< Number of particles */
    double timestamp;                 /**< Time stamp (optional) */
} stonesoup_particle_state_t;

/* State Vector Functions */

/**
 * @brief Create a new state vector
 * @param size Number of elements in the vector
 * @return Pointer to newly allocated state vector, or NULL on error
 */
STONESOUP_EXPORT stonesoup_state_vector_t*
stonesoup_state_vector_create(size_t size);

/**
 * @brief Free a state vector
 * @param vec Pointer to state vector to free
 */
STONESOUP_EXPORT void
stonesoup_state_vector_free(stonesoup_state_vector_t* vec);

/**
 * @brief Copy a state vector
 * @param src Source state vector
 * @return Pointer to newly allocated copy, or NULL on error
 */
STONESOUP_EXPORT stonesoup_state_vector_t*
stonesoup_state_vector_copy(const stonesoup_state_vector_t* src);

/**
 * @brief Set all elements of a state vector to a value
 * @param vec State vector
 * @param value Value to set
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_state_vector_fill(stonesoup_state_vector_t* vec, double value);

/* Covariance Matrix Functions */

/**
 * @brief Create a new covariance matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to newly allocated covariance matrix, or NULL on error
 */
STONESOUP_EXPORT stonesoup_covariance_matrix_t*
stonesoup_covariance_matrix_create(size_t rows, size_t cols);

/**
 * @brief Free a covariance matrix
 * @param mat Pointer to covariance matrix to free
 */
STONESOUP_EXPORT void
stonesoup_covariance_matrix_free(stonesoup_covariance_matrix_t* mat);

/**
 * @brief Copy a covariance matrix
 * @param src Source covariance matrix
 * @return Pointer to newly allocated copy, or NULL on error
 */
STONESOUP_EXPORT stonesoup_covariance_matrix_t*
stonesoup_covariance_matrix_copy(const stonesoup_covariance_matrix_t* src);

/**
 * @brief Set covariance matrix to identity
 * @param mat Covariance matrix
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_covariance_matrix_eye(stonesoup_covariance_matrix_t* mat);

/* Gaussian State Functions */

/**
 * @brief Create a new Gaussian state
 * @param state_dim Dimension of state vector
 * @return Pointer to newly allocated Gaussian state, or NULL on error
 */
STONESOUP_EXPORT stonesoup_gaussian_state_t*
stonesoup_gaussian_state_create(size_t state_dim);

/**
 * @brief Free a Gaussian state
 * @param state Pointer to Gaussian state to free
 */
STONESOUP_EXPORT void
stonesoup_gaussian_state_free(stonesoup_gaussian_state_t* state);

/**
 * @brief Copy a Gaussian state
 * @param src Source Gaussian state
 * @return Pointer to newly allocated copy, or NULL on error
 */
STONESOUP_EXPORT stonesoup_gaussian_state_t*
stonesoup_gaussian_state_copy(const stonesoup_gaussian_state_t* src);

/* Particle State Functions */

/**
 * @brief Create a new particle state
 * @param num_particles Number of particles
 * @param state_dim Dimension of each particle's state vector
 * @return Pointer to newly allocated particle state, or NULL on error
 */
STONESOUP_EXPORT stonesoup_particle_state_t*
stonesoup_particle_state_create(size_t num_particles, size_t state_dim);

/**
 * @brief Free a particle state
 * @param state Pointer to particle state to free
 */
STONESOUP_EXPORT void
stonesoup_particle_state_free(stonesoup_particle_state_t* state);

/**
 * @brief Normalize particle weights to sum to 1.0
 * @param state Particle state
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_state_normalize_weights(stonesoup_particle_state_t* state);

/**
 * @brief Get error message string for error code
 * @param error Error code
 * @return Human-readable error message
 */
STONESOUP_EXPORT const char*
stonesoup_error_string(stonesoup_error_t error);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_TYPES_H */
