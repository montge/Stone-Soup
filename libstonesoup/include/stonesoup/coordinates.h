/**
 * @file coordinates.h
 * @brief Coordinate transformation operations
 */

#ifndef STONESOUP_COORDINATES_H
#define STONESOUP_COORDINATES_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup coordinates Coordinate Transformations
 * @brief Transformations between different coordinate systems
 * @{
 */

/**
 * @brief Convert Cartesian to spherical coordinates
 *
 * Converts [x, y, z] to [range, azimuth, elevation].
 * Azimuth is measured from the x-axis.
 * Elevation is measured from the xy-plane.
 *
 * @param cartesian Input Cartesian coordinates (size 3)
 * @param spherical Output spherical coordinates (size 3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_cart2sphere(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_state_vector_t* spherical
);

/**
 * @brief Convert spherical to Cartesian coordinates
 *
 * Converts [range, azimuth, elevation] to [x, y, z].
 *
 * @param spherical Input spherical coordinates (size 3)
 * @param cartesian Output Cartesian coordinates (size 3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_sphere2cart(
    const stonesoup_state_vector_t* spherical,
    stonesoup_state_vector_t* cartesian
);

/**
 * @brief Convert Cartesian to polar coordinates (2D)
 *
 * Converts [x, y] to [range, bearing].
 *
 * @param cartesian Input Cartesian coordinates (size 2)
 * @param polar Output polar coordinates (size 2, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_cart2polar(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_state_vector_t* polar
);

/**
 * @brief Convert polar to Cartesian coordinates (2D)
 *
 * Converts [range, bearing] to [x, y].
 *
 * @param polar Input polar coordinates (size 2)
 * @param cartesian Output Cartesian coordinates (size 2, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_polar2cart(
    const stonesoup_state_vector_t* polar,
    stonesoup_state_vector_t* cartesian
);

/**
 * @brief Compute Jacobian of Cartesian to spherical transformation
 *
 * Computes the Jacobian matrix for linearizing the coordinate transformation
 * at a given point. Useful for Extended Kalman Filtering.
 *
 * @param cartesian Point at which to compute Jacobian (size 3)
 * @param jacobian Output Jacobian matrix (3×3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_cart2sphere_jacobian(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_covariance_matrix_t* jacobian
);

/**
 * @brief Compute Jacobian of spherical to Cartesian transformation
 *
 * @param spherical Point at which to compute Jacobian (size 3)
 * @param jacobian Output Jacobian matrix (3×3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_sphere2cart_jacobian(
    const stonesoup_state_vector_t* spherical,
    stonesoup_covariance_matrix_t* jacobian
);

/**
 * @brief Compute Jacobian of Cartesian to polar transformation (2D)
 *
 * @param cartesian Point at which to compute Jacobian (size 2)
 * @param jacobian Output Jacobian matrix (2×2, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_cart2polar_jacobian(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_covariance_matrix_t* jacobian
);

/**
 * @brief Compute Jacobian of polar to Cartesian transformation (2D)
 *
 * @param polar Point at which to compute Jacobian (size 2)
 * @param jacobian Output Jacobian matrix (2×2, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_polar2cart_jacobian(
    const stonesoup_state_vector_t* polar,
    stonesoup_covariance_matrix_t* jacobian
);

/**
 * @brief Convert geodetic to ECEF coordinates
 *
 * Converts [latitude, longitude, altitude] (in radians and meters)
 * to Earth-Centered Earth-Fixed (ECEF) [x, y, z] coordinates.
 * Uses WGS84 ellipsoid.
 *
 * @param geodetic Input geodetic coordinates [lat, lon, alt] (size 3)
 * @param ecef Output ECEF coordinates (size 3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_geodetic2ecef(
    const stonesoup_state_vector_t* geodetic,
    stonesoup_state_vector_t* ecef
);

/**
 * @brief Convert ECEF to geodetic coordinates
 *
 * Converts ECEF [x, y, z] to [latitude, longitude, altitude].
 * Uses WGS84 ellipsoid.
 *
 * @param ecef Input ECEF coordinates (size 3)
 * @param geodetic Output geodetic coordinates [lat, lon, alt] (size 3, allocated by caller)
 * @return Error code
 */
STONESOUP_EXPORT stonesoup_error_t
stonesoup_ecef2geodetic(
    const stonesoup_state_vector_t* ecef,
    stonesoup_state_vector_t* geodetic
);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_COORDINATES_H */
