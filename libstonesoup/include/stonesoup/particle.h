/**
 * @file particle.h
 * @brief Particle filter operations
 */

#ifndef STONESOUP_PARTICLE_H
#define STONESOUP_PARTICLE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup particle Particle Filtering
 * @brief Particle filter prediction, update, and resampling operations
 * @{
 */

/**
 * @brief Resampling algorithm types
 */
typedef enum {
    STONESOUP_RESAMPLE_MULTINOMIAL,   /**< Multinomial resampling */
    STONESOUP_RESAMPLE_SYSTEMATIC,    /**< Systematic resampling */
    STONESOUP_RESAMPLE_STRATIFIED,    /**< Stratified resampling */
    STONESOUP_RESAMPLE_RESIDUAL       /**< Residual resampling */
} stonesoup_resample_method_t;

/**
 * @brief Particle filter prediction step
 *
 * Propagates each particle through a transition function with process noise.
 *
 * @param prior Prior particle state
 * @param transition_func State transition function
 * @param process_noise_func Function to add process noise to state
 * @param predicted Output predicted particle state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_predict(
    const stonesoup_particle_state_t* prior,
    void (*transition_func)(const stonesoup_state_vector_t*, stonesoup_state_vector_t*),
    void (*process_noise_func)(stonesoup_state_vector_t*),
    stonesoup_particle_state_t* predicted
);

/**
 * @brief Particle filter update step
 *
 * Updates particle weights based on measurement likelihood.
 *
 * @param predicted Predicted particle state
 * @param measurement Measurement vector
 * @param likelihood_func Measurement likelihood function
 * @param posterior Output posterior particle state (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_update(
    const stonesoup_particle_state_t* predicted,
    const stonesoup_state_vector_t* measurement,
    double (*likelihood_func)(const stonesoup_state_vector_t*, const stonesoup_state_vector_t*),
    stonesoup_particle_state_t* posterior
);

/**
 * @brief Resample particles
 *
 * Resamples particles according to their weights to combat particle degeneracy.
 * After resampling, all particles have equal weights.
 *
 * @param state Particle state to resample (modified in place)
 * @param method Resampling method
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_resample(
    stonesoup_particle_state_t* state,
    stonesoup_resample_method_t method
);

/**
 * @brief Compute effective sample size
 *
 * Computes N_eff = 1 / sum(w_i^2) where w_i are normalized weights.
 * Used to determine when resampling is needed.
 *
 * @param state Particle state
 * @param n_eff Output effective sample size
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_effective_sample_size(
    const stonesoup_particle_state_t* state,
    double* n_eff
);

/**
 * @brief Compute mean state from particles
 *
 * Computes weighted mean state vector from particle distribution.
 *
 * @param state Particle state
 * @param mean Output mean state vector (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_mean(
    const stonesoup_particle_state_t* state,
    stonesoup_state_vector_t* mean
);

/**
 * @brief Compute covariance from particles
 *
 * Computes weighted covariance matrix from particle distribution.
 *
 * @param state Particle state
 * @param mean Mean state vector (if NULL, will be computed)
 * @param covariance Output covariance matrix (allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_covariance(
    const stonesoup_particle_state_t* state,
    const stonesoup_state_vector_t* mean,
    stonesoup_covariance_matrix_t* covariance
);

/**
 * @brief Systematic resampling
 *
 * Low-variance systematic resampling algorithm.
 *
 * @param state Particle state to resample (modified in place)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_systematic_resample(stonesoup_particle_state_t* state);

/**
 * @brief Stratified resampling
 *
 * Stratified resampling algorithm with reduced variance.
 *
 * @param state Particle state to resample (modified in place)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_stratified_resample(stonesoup_particle_state_t* state);

/**
 * @brief Multinomial resampling
 *
 * Simple multinomial resampling (higher variance than systematic/stratified).
 *
 * @param state Particle state to resample (modified in place)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_particle_multinomial_resample(stonesoup_particle_state_t* state);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_PARTICLE_H */
