/**
 * @file stonesoup.h
 * @brief Main public header for Stone Soup C library
 *
 * This is the primary header file that should be included by users of the
 * Stone Soup C library. It includes all other public headers.
 *
 * @mainpage Stone Soup C Library
 *
 * @section intro_sec Introduction
 *
 * Stone Soup is a software project to provide the target tracking and state
 * estimation community with a framework for development and testing of
 * tracking algorithms.
 *
 * This C library provides high-performance implementations of core tracking
 * algorithms including:
 * - Kalman filtering (linear and extended)
 * - Particle filtering with various resampling schemes
 * - Coordinate transformations
 *
 * @section usage_sec Usage
 *
 * Include the main header:
 * @code
 * #include <stonesoup/stonesoup.h>
 * @endcode
 *
 * Link against the library:
 * @code
 * gcc myapp.c -lstonesoup -lm
 * @endcode
 *
 * @section example_sec Example
 *
 * @code
 * // Create a Gaussian state
 * stonesoup_gaussian_state_t* state = stonesoup_gaussian_state_create(4);
 *
 * // Set initial state
 * state->state_vector->data[0] = 0.0;  // x position
 * state->state_vector->data[1] = 0.0;  // y position
 * state->state_vector->data[2] = 1.0;  // x velocity
 * state->state_vector->data[3] = 1.0;  // y velocity
 *
 * // Set initial covariance to identity
 * stonesoup_covariance_matrix_eye(state->covariance);
 *
 * // Clean up
 * stonesoup_gaussian_state_free(state);
 * @endcode
 *
 * @section license_sec License
 *
 * MIT License - See LICENSE file for details
 */

#ifndef STONESOUP_H
#define STONESOUP_H

#include "stonesoup/version.h"
#include "stonesoup/types.h"
#include "stonesoup/kalman.h"
#include "stonesoup/particle.h"
#include "stonesoup/coordinates.h"

/**
 * @defgroup core Core Library
 * @brief Core library initialization and utilities
 * @{
 */

/**
 * @brief Initialize the Stone Soup library
 *
 * This function should be called once at the start of your program
 * before using any other library functions. Currently this is a no-op
 * but may be used for initialization in future versions.
 *
 * @return Error code (currently always STONESOUP_SUCCESS)
 */
STONESOUP_EXPORT stonesoup_error_t stonesoup_init(void);

/**
 * @brief Clean up library resources
 *
 * This function should be called once at the end of your program
 * to clean up any library resources. Currently this is a no-op
 * but may be used for cleanup in future versions.
 */
STONESOUP_EXPORT void stonesoup_cleanup(void);

/** @} */

#endif /* STONESOUP_H */
