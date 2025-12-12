/**
 * @file coordinates.c
 * @brief Implementation of coordinate transformation operations
 */

#include "stonesoup/coordinates.h"
#include <math.h>
#include <string.h>

/* WGS84 ellipsoid constants */
#define WGS84_A 6378137.0               /* Semi-major axis (meters) */
#define WGS84_F (1.0 / 298.257223563)   /* Flattening */
#define WGS84_B (WGS84_A * (1.0 - WGS84_F))  /* Semi-minor axis */
#define WGS84_E2 (2.0 * WGS84_F - WGS84_F * WGS84_F)  /* First eccentricity squared */

stonesoup_error_t stonesoup_cart2sphere(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_state_vector_t* spherical) {

    if (!cartesian || !spherical) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (cartesian->size != 3 || spherical->size != 3) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    double x = cartesian->data[0];
    double y = cartesian->data[1];
    double z = cartesian->data[2];

    // Range
    double range = sqrt(x*x + y*y + z*z);

    // Azimuth (from x-axis)
    double azimuth = atan2(y, x);

    // Elevation (from xy-plane)
    double elevation = atan2(z, sqrt(x*x + y*y));

    spherical->data[0] = range;
    spherical->data[1] = azimuth;
    spherical->data[2] = elevation;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_sphere2cart(
    const stonesoup_state_vector_t* spherical,
    stonesoup_state_vector_t* cartesian) {

    if (!spherical || !cartesian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (spherical->size != 3 || cartesian->size != 3) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    double range = spherical->data[0];
    double azimuth = spherical->data[1];
    double elevation = spherical->data[2];

    double cos_el = cos(elevation);

    cartesian->data[0] = range * cos_el * cos(azimuth);
    cartesian->data[1] = range * cos_el * sin(azimuth);
    cartesian->data[2] = range * sin(elevation);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_cart2polar(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_state_vector_t* polar) {

    if (!cartesian || !polar) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (cartesian->size != 2 || polar->size != 2) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    double x = cartesian->data[0];
    double y = cartesian->data[1];

    // Range
    double range = sqrt(x*x + y*y);

    // Bearing
    double bearing = atan2(y, x);

    polar->data[0] = range;
    polar->data[1] = bearing;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_polar2cart(
    const stonesoup_state_vector_t* polar,
    stonesoup_state_vector_t* cartesian) {

    if (!polar || !cartesian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (polar->size != 2 || cartesian->size != 2) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    double range = polar->data[0];
    double bearing = polar->data[1];

    cartesian->data[0] = range * cos(bearing);
    cartesian->data[1] = range * sin(bearing);

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_cart2sphere_jacobian(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_covariance_matrix_t* jacobian) {

    if (!cartesian || !jacobian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (cartesian->size != 3 || jacobian->rows != 3 || jacobian->cols != 3) {
        return STONESOUP_ERROR_DIMENSION;
    }

    // TODO: Implement Jacobian computation for cart2sphere

    return STONESOUP_ERROR_NOT_IMPLEMENTED;
}

stonesoup_error_t stonesoup_sphere2cart_jacobian(
    const stonesoup_state_vector_t* spherical,
    stonesoup_covariance_matrix_t* jacobian) {

    if (!spherical || !jacobian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (spherical->size != 3 || jacobian->rows != 3 || jacobian->cols != 3) {
        return STONESOUP_ERROR_DIMENSION;
    }

    // TODO: Implement Jacobian computation for sphere2cart

    return STONESOUP_ERROR_NOT_IMPLEMENTED;
}

stonesoup_error_t stonesoup_cart2polar_jacobian(
    const stonesoup_state_vector_t* cartesian,
    stonesoup_covariance_matrix_t* jacobian) {

    if (!cartesian || !jacobian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (cartesian->size != 2 || jacobian->rows != 2 || jacobian->cols != 2) {
        return STONESOUP_ERROR_DIMENSION;
    }

    double x = cartesian->data[0];
    double y = cartesian->data[1];
    double r_sq = x*x + y*y;
    double r = sqrt(r_sq);

    if (r < 1e-10) {
        return STONESOUP_ERROR_SINGULAR;
    }

    // Jacobian matrix:
    // [ x/r,   y/r   ]
    // [-y/r^2, x/r^2 ]

    jacobian->data[0] = x / r;       // drange/dx
    jacobian->data[1] = y / r;       // drange/dy
    jacobian->data[2] = -y / r_sq;   // dbearing/dx
    jacobian->data[3] = x / r_sq;    // dbearing/dy

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_polar2cart_jacobian(
    const stonesoup_state_vector_t* polar,
    stonesoup_covariance_matrix_t* jacobian) {

    if (!polar || !jacobian) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (polar->size != 2 || jacobian->rows != 2 || jacobian->cols != 2) {
        return STONESOUP_ERROR_DIMENSION;
    }

    double range = polar->data[0];
    double bearing = polar->data[1];
    double cos_b = cos(bearing);
    double sin_b = sin(bearing);

    // Jacobian matrix:
    // [ cos(bearing), -range*sin(bearing) ]
    // [ sin(bearing),  range*cos(bearing) ]

    jacobian->data[0] = cos_b;           // dx/drange
    jacobian->data[1] = -range * sin_b;  // dx/dbearing
    jacobian->data[2] = sin_b;           // dy/drange
    jacobian->data[3] = range * cos_b;   // dy/dbearing

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_geodetic2ecef(
    const stonesoup_state_vector_t* geodetic,
    stonesoup_state_vector_t* ecef) {

    if (!geodetic || !ecef) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (geodetic->size != 3 || ecef->size != 3) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    double lat = geodetic->data[0];
    double lon = geodetic->data[1];
    double alt = geodetic->data[2];

    double sin_lat = sin(lat);
    double cos_lat = cos(lat);
    double sin_lon = sin(lon);
    double cos_lon = cos(lon);

    // Prime vertical radius of curvature
    double N = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

    ecef->data[0] = (N + alt) * cos_lat * cos_lon;
    ecef->data[1] = (N + alt) * cos_lat * sin_lon;
    ecef->data[2] = (N * (1.0 - WGS84_E2) + alt) * sin_lat;

    return STONESOUP_SUCCESS;
}

stonesoup_error_t stonesoup_ecef2geodetic(
    const stonesoup_state_vector_t* ecef,
    stonesoup_state_vector_t* geodetic) {

    if (!ecef || !geodetic) {
        return STONESOUP_ERROR_NULL_POINTER;
    }

    if (ecef->size != 3 || geodetic->size != 3) {
        return STONESOUP_ERROR_INVALID_SIZE;
    }

    // TODO: Implement iterative ECEF to geodetic conversion
    // This is more complex than geodetic to ECEF

    return STONESOUP_ERROR_NOT_IMPLEMENTED;
}
