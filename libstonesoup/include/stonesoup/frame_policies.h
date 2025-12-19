/**
 * @file frame_policies.h
 * @brief Coordinate frame-specific numeric policies for Stone Soup C library
 *
 * This header provides frame-specific numeric policies that define precision
 * requirements, valid range limits, and recommended scale factors for different
 * reference frames used in target tracking and state estimation.
 *
 * Different coordinate frames have different characteristics based on their
 * intended use and the scale of values they represent. This module provides
 * compile-time and runtime access to these policies to enable:
 *
 * - Validation of coordinate values against frame-specific limits
 * - Selection of appropriate numeric precision for each frame
 * - Detection of out-of-range values that may indicate errors
 * - Documentation of numeric assumptions for safety-critical systems
 *
 * @section frame_types Supported Reference Frames
 *
 * **Earth-Centered Inertial (ECI):**
 * - Non-rotating inertial frame with origin at Earth's center
 * - Used for orbital mechanics and satellite tracking
 * - Coordinates typically in meters (6-50 million meter range)
 * - Requires high precision for orbital accuracy
 *
 * **Earth-Centered Earth-Fixed (ECEF):**
 * - Rotating frame fixed to Earth's surface
 * - Used for ground-based and LEO tracking
 * - Coordinates in meters (6-7 million meter range)
 * - Suitable for GPS and terrestrial applications
 *
 * **Latitude-Longitude-Altitude (LLA):**
 * - Geodetic coordinates on Earth ellipsoid
 * - Latitude/longitude in radians, altitude in meters
 * - Special range constraints: lat ∈ [-π/2, π/2], lon ∈ [-π, π]
 * - High angular precision needed for accuracy
 *
 * **East-North-Up (ENU):**
 * - Local tangent plane centered at observer
 * - Used for local tracking (radar, sonar)
 * - Coordinates in meters (typically 0-500 km)
 * - Good numerical conditioning for local operations
 *
 * **Lunar-Centered Inertial:**
 * - Non-rotating frame with origin at Moon's center
 * - Used for cislunar and lunar surface operations
 * - Coordinates in meters (0-10 million meter range)
 * - Critical for lunar orbit determination
 *
 * **Voxel Grid:**
 * - Discrete grid coordinates for volume representation
 * - Integer indices into 3D array
 * - Used for occupancy mapping and spatial hashing
 * - Requires integer range validation
 *
 * **Cartesian (Generic):**
 * - General-purpose Cartesian coordinates
 * - No specific physical interpretation
 * - Wide range support for flexibility
 *
 * **Polar:**
 * - 2D polar coordinates (range, bearing)
 * - Used for 2D tracking problems
 * - Range ≥ 0, bearing ∈ [0, 2π]
 *
 * **Spherical:**
 * - 3D spherical coordinates (range, azimuth, elevation)
 * - Used for radar and sensor modeling
 * - Range ≥ 0, angles in radians
 *
 * @section usage_examples Usage Examples
 *
 * @code
 * // Get policy for ECI frame
 * const ss_frame_policy_t* eci_policy = ss_get_frame_policy(SS_FRAME_ECI);
 *
 * // Validate a position value
 * ss_real_t x = 7000000.0;  // 7000 km
 * if (!ss_validate_position(x, eci_policy)) {
 *     fprintf(stderr, "Position out of range for ECI frame\n");
 * }
 *
 * // Check if value is within frame limits
 * SS_VALIDATE_FRAME_POSITION(SS_FRAME_ECEF, latitude);
 *
 * // Get recommended scale factor for better conditioning
 * ss_real_t scale = eci_policy->position_scale_factor;  // 1000.0 (use km)
 * ss_real_t x_scaled = x / scale;
 * @endcode
 *
 * @section safety_considerations Safety-Critical Considerations
 *
 * For DO-178C/DO-254 and safety-critical applications:
 *
 * - Always validate coordinates against frame policies before use
 * - Document which reference frame is used in each module
 * - Use SS_VALIDATE_FRAME_* macros in debug builds to catch violations
 * - Consider the impact of coordinate transformations on precision
 * - Test boundary conditions at frame limits
 * - Verify that precision requirements are met for operational scenarios
 */

#ifndef STONESOUP_FRAME_POLICIES_H
#define STONESOUP_FRAME_POLICIES_H

#include "precision.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup frame_policies Frame Policies
 * @brief Numeric policies for coordinate reference frames
 * @{
 */

/**
 * @brief Supported reference frame types
 */
typedef enum {
    SS_FRAME_CARTESIAN,      /**< Generic Cartesian coordinates */
    SS_FRAME_POLAR,          /**< 2D polar coordinates (range, bearing) */
    SS_FRAME_SPHERICAL,      /**< 3D spherical coordinates (range, az, el) */
    SS_FRAME_ECI,            /**< Earth-Centered Inertial */
    SS_FRAME_ECEF,           /**< Earth-Centered Earth-Fixed */
    SS_FRAME_LLA,            /**< Latitude-Longitude-Altitude (geodetic) */
    SS_FRAME_ENU,            /**< East-North-Up (local tangent plane) */
    SS_FRAME_LUNAR,          /**< Lunar-Centered Inertial */
    SS_FRAME_VOXEL,          /**< Voxel grid (discrete integer coordinates) */
    SS_FRAME_UNKNOWN         /**< Unknown or undefined frame */
} ss_reference_frame_t;

/**
 * @brief Numeric policy for a coordinate reference frame
 *
 * Defines precision requirements, valid range limits, and recommended
 * scale factors for a specific reference frame. All linear measurements
 * are in meters, angular measurements in radians, velocities in m/s.
 */
typedef struct {
    /** Reference frame identifier */
    ss_reference_frame_t frame;

    /** Human-readable frame name */
    const char* name;

    /** Brief description of frame */
    const char* description;

    /* Position Constraints */

    /** Required position precision in meters (e.g., 0.01 = 1 cm) */
    ss_real_t position_precision;

    /** Minimum valid position magnitude in meters */
    ss_real_t min_position;

    /** Maximum valid position magnitude in meters */
    ss_real_t max_position;

    /** Recommended scale factor for position (e.g., 1000.0 to use km) */
    ss_real_t position_scale_factor;

    /* Velocity Constraints */

    /** Required velocity precision in m/s (e.g., 0.001 = 1 mm/s) */
    ss_real_t velocity_precision;

    /** Minimum valid velocity magnitude in m/s */
    ss_real_t min_velocity;

    /** Maximum valid velocity magnitude in m/s */
    ss_real_t max_velocity;

    /** Recommended scale factor for velocity */
    ss_real_t velocity_scale_factor;

    /* Time Constraints */

    /** Required time precision in seconds */
    ss_real_t time_precision;

    /** Maximum time interval supported in seconds */
    ss_real_t max_time_interval;

    /* Angular Constraints (for frames with angular components) */

    /** Required angular precision in radians (e.g., 1e-9 rad) */
    ss_real_t angular_precision;

    /** Minimum valid angle in radians (or 0.0 if not applicable) */
    ss_real_t min_angle;

    /** Maximum valid angle in radians (or 0.0 if not applicable) */
    ss_real_t max_angle;

    /* Frame Properties */

    /** True if frame is rotating (like ECEF), false if inertial (like ECI) */
    bool is_rotating;

    /** True if frame uses angular coordinates */
    bool has_angular_coords;

    /** True if frame has special singularities (e.g., poles in LLA) */
    bool has_singularities;

    /** Number of spatial dimensions (2 or 3) */
    int spatial_dims;

} ss_frame_policy_t;

/*===========================================================================*/
/* Frame Policy Access Functions                                             */
/*===========================================================================*/

/**
 * @brief Get the numeric policy for a specific reference frame
 *
 * Returns a pointer to a static policy structure containing precision
 * requirements, range limits, and scale factors for the specified frame.
 *
 * @param frame Reference frame identifier
 * @return Pointer to frame policy (never NULL; returns UNKNOWN policy for invalid frames)
 */
const ss_frame_policy_t* ss_get_frame_policy(ss_reference_frame_t frame);

/**
 * @brief Get frame policy by name string
 *
 * Looks up a frame policy by its name (case-insensitive).
 * Useful for configuration files or user input.
 *
 * @param name Frame name (e.g., "ECI", "ECEF", "LLA")
 * @return Pointer to frame policy, or NULL if name not recognized
 */
const ss_frame_policy_t* ss_get_frame_policy_by_name(const char* name);

/**
 * @brief Get human-readable name of reference frame
 *
 * @param frame Reference frame identifier
 * @return Frame name string (never NULL)
 */
const char* ss_frame_name(ss_reference_frame_t frame);

/**
 * @brief Get description of reference frame
 *
 * @param frame Reference frame identifier
 * @return Frame description string (never NULL)
 */
const char* ss_frame_description(ss_reference_frame_t frame);

/*===========================================================================*/
/* Validation Functions                                                       */
/*===========================================================================*/

/**
 * @brief Validate a position value against frame policy
 *
 * Checks if a position value is within the valid range for the frame.
 *
 * @param position Position value in meters
 * @param policy Frame policy to validate against
 * @return true if valid, false if out of range
 */
static inline bool ss_validate_position(ss_real_t position,
                                         const ss_frame_policy_t* policy) {
    ss_real_t abs_pos = ss_fabs(position);
    return (abs_pos >= policy->min_position && abs_pos <= policy->max_position);
}

/**
 * @brief Validate a velocity value against frame policy
 *
 * @param velocity Velocity value in m/s
 * @param policy Frame policy to validate against
 * @return true if valid, false if out of range
 */
static inline bool ss_validate_velocity(ss_real_t velocity,
                                         const ss_frame_policy_t* policy) {
    ss_real_t abs_vel = ss_fabs(velocity);
    return (abs_vel >= policy->min_velocity && abs_vel <= policy->max_velocity);
}

/**
 * @brief Validate an angular value against frame policy
 *
 * @param angle Angle value in radians
 * @param policy Frame policy to validate against
 * @return true if valid, false if out of range or frame has no angles
 */
static inline bool ss_validate_angle(ss_real_t angle,
                                      const ss_frame_policy_t* policy) {
    if (!policy->has_angular_coords) {
        return false;
    }
    return (angle >= policy->min_angle && angle <= policy->max_angle);
}

/**
 * @brief Check if position meets required precision for frame
 *
 * Determines if the numeric precision of ss_real_t is sufficient to
 * represent positions in this frame with required accuracy.
 *
 * @param policy Frame policy to check
 * @return true if precision is sufficient, false otherwise
 */
bool ss_check_position_precision(const ss_frame_policy_t* policy);

/**
 * @brief Check if velocity meets required precision for frame
 *
 * @param policy Frame policy to check
 * @return true if precision is sufficient, false otherwise
 */
bool ss_check_velocity_precision(const ss_frame_policy_t* policy);

/**
 * @brief Check if angular precision is sufficient for frame
 *
 * @param policy Frame policy to check
 * @return true if precision is sufficient, false otherwise
 */
bool ss_check_angular_precision(const ss_frame_policy_t* policy);

/*===========================================================================*/
/* Validation Macros                                                          */
/*===========================================================================*/

#if defined(STONESOUP_DEBUG) || defined(STONESOUP_RANGE_CHECK)

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Validate position against frame policy (debug builds only)
 *
 * In debug builds, validates that a position value is within the valid
 * range for the specified frame. Aborts with diagnostic message on failure.
 *
 * @param frame Reference frame identifier
 * @param position Position value to validate
 */
#define SS_VALIDATE_FRAME_POSITION(frame, position) \
    do { \
        const ss_frame_policy_t* _policy = ss_get_frame_policy(frame); \
        ss_real_t _pos = (ss_real_t)(position); \
        if (!ss_validate_position(_pos, _policy)) { \
            fprintf(stderr, \
                "Stone Soup frame position violation at %s:%d\n" \
                "  Frame: %s\n" \
                "  Position: %g m (valid range: [%g, %g] m)\n", \
                __FILE__, __LINE__, \
                _policy->name, \
                (double)_pos, \
                (double)_policy->min_position, \
                (double)_policy->max_position); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Validate velocity against frame policy (debug builds only)
 *
 * @param frame Reference frame identifier
 * @param velocity Velocity value to validate
 */
#define SS_VALIDATE_FRAME_VELOCITY(frame, velocity) \
    do { \
        const ss_frame_policy_t* _policy = ss_get_frame_policy(frame); \
        ss_real_t _vel = (ss_real_t)(velocity); \
        if (!ss_validate_velocity(_vel, _policy)) { \
            fprintf(stderr, \
                "Stone Soup frame velocity violation at %s:%d\n" \
                "  Frame: %s\n" \
                "  Velocity: %g m/s (valid range: [%g, %g] m/s)\n", \
                __FILE__, __LINE__, \
                _policy->name, \
                (double)_vel, \
                (double)_policy->min_velocity, \
                (double)_policy->max_velocity); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Validate angle against frame policy (debug builds only)
 *
 * @param frame Reference frame identifier
 * @param angle Angle value to validate (radians)
 */
#define SS_VALIDATE_FRAME_ANGLE(frame, angle) \
    do { \
        const ss_frame_policy_t* _policy = ss_get_frame_policy(frame); \
        ss_real_t _ang = (ss_real_t)(angle); \
        if (!ss_validate_angle(_ang, _policy)) { \
            fprintf(stderr, \
                "Stone Soup frame angle violation at %s:%d\n" \
                "  Frame: %s\n" \
                "  Angle: %g rad (valid range: [%g, %g] rad)\n", \
                __FILE__, __LINE__, \
                _policy->name, \
                (double)_ang, \
                (double)_policy->min_angle, \
                (double)_policy->max_angle); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Check that frame precision is sufficient (debug builds only)
 *
 * Validates that the current ss_real_t precision mode is sufficient
 * for the required precision in this frame. Aborts if insufficient.
 *
 * @param frame Reference frame identifier
 */
#define SS_CHECK_FRAME_PRECISION(frame) \
    do { \
        const ss_frame_policy_t* _policy = ss_get_frame_policy(frame); \
        if (!ss_check_position_precision(_policy) || \
            !ss_check_velocity_precision(_policy) || \
            (_policy->has_angular_coords && !ss_check_angular_precision(_policy))) { \
            fprintf(stderr, \
                "Stone Soup frame precision insufficient at %s:%d\n" \
                "  Frame: %s\n" \
                "  Current precision: %s (%zu bytes, %d digits)\n" \
                "  Required position precision: %g m\n" \
                "  Required velocity precision: %g m/s\n", \
                __FILE__, __LINE__, \
                _policy->name, \
                ss_precision_mode(), \
                ss_real_size(), \
                ss_real_digits(), \
                (double)_policy->position_precision, \
                (double)_policy->velocity_precision); \
            abort(); \
        } \
    } while(0)

#else

/* In release builds, validation macros are no-ops */
#define SS_VALIDATE_FRAME_POSITION(frame, position) ((void)0)
#define SS_VALIDATE_FRAME_VELOCITY(frame, velocity) ((void)0)
#define SS_VALIDATE_FRAME_ANGLE(frame, angle) ((void)0)
#define SS_CHECK_FRAME_PRECISION(frame) ((void)0)

#endif /* STONESOUP_DEBUG || STONESOUP_RANGE_CHECK */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_FRAME_POLICIES_H */
