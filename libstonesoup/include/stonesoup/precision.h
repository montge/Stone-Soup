/**
 * @file precision.h
 * @brief Configurable numeric precision for Stone Soup C library
 *
 * This header provides compile-time configuration for numeric precision
 * and domain-specific range limits. It supports three precision modes:
 *
 * - STONESOUP_PRECISION_SINGLE: 32-bit float (embedded systems)
 * - STONESOUP_PRECISION_DOUBLE: 64-bit double (default)
 * - STONESOUP_PRECISION_EXTENDED: long double (scientific computing)
 *
 * It also provides domain-specific range constants and optional
 * overflow checking macros for debug builds.
 */

#ifndef STONESOUP_PRECISION_H
#define STONESOUP_PRECISION_H

#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================*/
/* Precision Configuration                                                    */
/*===========================================================================*/

#if defined(STONESOUP_PRECISION_SINGLE)
    /** Single precision floating point (32-bit) */
    typedef float ss_real_t;
    #define SS_REAL_MIN FLT_MIN
    #define SS_REAL_MAX FLT_MAX
    #define SS_REAL_EPSILON FLT_EPSILON
    #define SS_REAL_DIG FLT_DIG
    #define SS_REAL_MANT_DIG FLT_MANT_DIG
    #define SS_REAL(x) (x##f)
    #define ss_sqrt sqrtf
    #define ss_sin sinf
    #define ss_cos cosf
    #define ss_atan2 atan2f
    #define ss_fabs fabsf
    #define ss_floor floorf
    #define ss_ceil ceilf
    #define ss_exp expf
    #define ss_log logf
    #define ss_pow powf

#elif defined(STONESOUP_PRECISION_EXTENDED)
    /** Extended precision floating point (80-bit or 128-bit) */
    typedef long double ss_real_t;
    #define SS_REAL_MIN LDBL_MIN
    #define SS_REAL_MAX LDBL_MAX
    #define SS_REAL_EPSILON LDBL_EPSILON
    #define SS_REAL_DIG LDBL_DIG
    #define SS_REAL_MANT_DIG LDBL_MANT_DIG
    #define SS_REAL(x) (x##L)
    #define ss_sqrt sqrtl
    #define ss_sin sinl
    #define ss_cos cosl
    #define ss_atan2 atan2l
    #define ss_fabs fabsl
    #define ss_floor floorl
    #define ss_ceil ceill
    #define ss_exp expl
    #define ss_log logl
    #define ss_pow powl

#else
    /** Double precision floating point (64-bit) - DEFAULT */
    typedef double ss_real_t;
    #define SS_REAL_MIN DBL_MIN
    #define SS_REAL_MAX DBL_MAX
    #define SS_REAL_EPSILON DBL_EPSILON
    #define SS_REAL_DIG DBL_DIG
    #define SS_REAL_MANT_DIG DBL_MANT_DIG
    #define SS_REAL(x) (x)
    #define ss_sqrt sqrt
    #define ss_sin sin
    #define ss_cos cos
    #define ss_atan2 atan2
    #define ss_fabs fabs
    #define ss_floor floor
    #define ss_ceil ceil
    #define ss_exp exp
    #define ss_log log
    #define ss_pow pow
#endif

/*===========================================================================*/
/* Mathematical Constants                                                     */
/*===========================================================================*/

#ifndef SS_PI
#define SS_PI SS_REAL(3.14159265358979323846)
#endif

#ifndef SS_TWO_PI
#define SS_TWO_PI SS_REAL(6.28318530717958647692)
#endif

#ifndef SS_DEG_TO_RAD
#define SS_DEG_TO_RAD SS_REAL(0.01745329251994329577)
#endif

#ifndef SS_RAD_TO_DEG
#define SS_RAD_TO_DEG SS_REAL(57.29577951308232087680)
#endif

/*===========================================================================*/
/* Physical Constants                                                         */
/*===========================================================================*/

/** Earth mean radius in meters */
#define SS_EARTH_RADIUS_M SS_REAL(6371000.0)

/** Earth mean radius in kilometers */
#define SS_EARTH_RADIUS_KM SS_REAL(6371.0)

/** Earth gravitational parameter (m^3/s^2) */
#define SS_EARTH_MU SS_REAL(3.986004418e14)

/** Speed of sound in seawater (m/s, nominal) */
#define SS_SOUND_SPEED_WATER SS_REAL(1500.0)

/** Speed of light in vacuum (m/s) */
#define SS_SPEED_OF_LIGHT SS_REAL(299792458.0)

/*===========================================================================*/
/* Undersea Domain Limits                                                     */
/*===========================================================================*/

/** Maximum ocean depth in meters (Mariana Trench) */
#define SS_UNDERSEA_DEPTH_MAX SS_REAL(11000.0)

/** Maximum sonar tracking range in meters */
#define SS_UNDERSEA_RANGE_MAX SS_REAL(100000.0)

/** Minimum sound speed in seawater (m/s) */
#define SS_UNDERSEA_SOUND_SPEED_MIN SS_REAL(1400.0)

/** Maximum sound speed in seawater (m/s) */
#define SS_UNDERSEA_SOUND_SPEED_MAX SS_REAL(1600.0)

/** Maximum submarine velocity (m/s) */
#define SS_UNDERSEA_VELOCITY_MAX SS_REAL(30.0)

/** Maximum hydrostatic pressure (bar) at max depth */
#define SS_UNDERSEA_PRESSURE_MAX SS_REAL(1200.0)

/*===========================================================================*/
/* Orbital Domain Limits                                                      */
/*===========================================================================*/

/** Maximum supported altitude in meters (beyond GEO) */
#define SS_ORBITAL_ALTITUDE_MAX SS_REAL(50000000.0)

/** GEO altitude in meters */
#define SS_GEO_ALTITUDE_M SS_REAL(35786000.0)

/** Maximum orbital velocity in m/s (escape velocity) */
#define SS_ORBITAL_VELOCITY_MAX SS_REAL(12000.0)

/** LEO altitude upper bound in meters */
#define SS_LEO_ALTITUDE_MAX SS_REAL(2000000.0)

/*===========================================================================*/
/* Cislunar Domain Limits                                                     */
/*===========================================================================*/

/** Maximum cislunar tracking distance in meters */
#define SS_CISLUNAR_DISTANCE_MAX SS_REAL(500000000.0)

/** Earth-Moon mean distance in meters */
#define SS_EARTH_MOON_DISTANCE_M SS_REAL(384400000.0)

/** Maximum trans-lunar velocity in m/s */
#define SS_CISLUNAR_VELOCITY_MAX SS_REAL(15000.0)

/** Lunar sphere of influence radius in meters */
#define SS_LUNAR_SOI_M SS_REAL(66000000.0)

/*===========================================================================*/
/* Interplanetary Domain Limits                                               */
/*===========================================================================*/

/** Maximum interplanetary tracking distance in meters (inner solar system) */
#define SS_INTERPLANETARY_DISTANCE_MAX SS_REAL(1.0e12)

/** Maximum interplanetary velocity in m/s */
#define SS_INTERPLANETARY_VELOCITY_MAX SS_REAL(60000.0)

/*===========================================================================*/
/* Range Checking Macros                                                      */
/*===========================================================================*/

#if defined(STONESOUP_DEBUG) || defined(STONESOUP_RANGE_CHECK)

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Check that value is within specified range
 *
 * In debug builds, this macro validates that a value falls within the
 * specified range. If the check fails, it prints an error message and
 * aborts the program.
 *
 * @param val Value to check
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 */
#define SS_CHECK_RANGE(val, min, max) \
    do { \
        ss_real_t _v = (ss_real_t)(val); \
        ss_real_t _min = (ss_real_t)(min); \
        ss_real_t _max = (ss_real_t)(max); \
        if (_v < _min || _v > _max) { \
            fprintf(stderr, \
                "Stone Soup range violation at %s:%d\n" \
                "  Value: %g not in range [%g, %g]\n", \
                __FILE__, __LINE__, \
                (double)_v, (double)_min, (double)_max); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Check that value is positive
 */
#define SS_CHECK_POSITIVE(val) \
    do { \
        ss_real_t _v = (ss_real_t)(val); \
        if (_v <= SS_REAL(0.0)) { \
            fprintf(stderr, \
                "Stone Soup positive value required at %s:%d\n" \
                "  Value: %g\n", \
                __FILE__, __LINE__, (double)_v); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Check that value is non-negative
 */
#define SS_CHECK_NON_NEGATIVE(val) \
    do { \
        ss_real_t _v = (ss_real_t)(val); \
        if (_v < SS_REAL(0.0)) { \
            fprintf(stderr, \
                "Stone Soup non-negative value required at %s:%d\n" \
                "  Value: %g\n", \
                __FILE__, __LINE__, (double)_v); \
            abort(); \
        } \
    } while(0)

/**
 * @brief Check that value is finite (not NaN or Inf)
 */
#define SS_CHECK_FINITE(val) \
    do { \
        ss_real_t _v = (ss_real_t)(val); \
        if (!isfinite((double)_v)) { \
            fprintf(stderr, \
                "Stone Soup finite value required at %s:%d\n" \
                "  Value: %g\n", \
                __FILE__, __LINE__, (double)_v); \
            abort(); \
        } \
    } while(0)

#else

/* In release builds, range checks are no-ops */
#define SS_CHECK_RANGE(val, min, max) ((void)0)
#define SS_CHECK_POSITIVE(val) ((void)0)
#define SS_CHECK_NON_NEGATIVE(val) ((void)0)
#define SS_CHECK_FINITE(val) ((void)0)

#endif /* STONESOUP_DEBUG || STONESOUP_RANGE_CHECK */

/*===========================================================================*/
/* Domain-Specific Validation Macros                                          */
/*===========================================================================*/

/** Validate undersea depth */
#define SS_CHECK_DEPTH(val) \
    SS_CHECK_RANGE(val, SS_REAL(0.0), SS_UNDERSEA_DEPTH_MAX)

/** Validate undersea range */
#define SS_CHECK_SONAR_RANGE(val) \
    SS_CHECK_RANGE(val, SS_REAL(0.0), SS_UNDERSEA_RANGE_MAX)

/** Validate sound speed */
#define SS_CHECK_SOUND_SPEED(val) \
    SS_CHECK_RANGE(val, SS_UNDERSEA_SOUND_SPEED_MIN, SS_UNDERSEA_SOUND_SPEED_MAX)

/** Validate orbital altitude */
#define SS_CHECK_ORBITAL_ALTITUDE(val) \
    SS_CHECK_RANGE(val, SS_REAL(0.0), SS_ORBITAL_ALTITUDE_MAX)

/** Validate cislunar distance */
#define SS_CHECK_CISLUNAR_DISTANCE(val) \
    SS_CHECK_RANGE(val, SS_REAL(0.0), SS_CISLUNAR_DISTANCE_MAX)

/** Validate angle in radians (0 to 2*Pi) */
#define SS_CHECK_ANGLE_RAD(val) \
    SS_CHECK_RANGE(val, SS_REAL(0.0), SS_TWO_PI)

/** Validate latitude in radians */
#define SS_CHECK_LATITUDE_RAD(val) \
    SS_CHECK_RANGE(val, -SS_PI/SS_REAL(2.0), SS_PI/SS_REAL(2.0))

/** Validate longitude in radians */
#define SS_CHECK_LONGITUDE_RAD(val) \
    SS_CHECK_RANGE(val, -SS_PI, SS_PI)

/*===========================================================================*/
/* Precision Information                                                      */
/*===========================================================================*/

/**
 * @brief Get string describing current precision mode
 * @return Precision mode string ("single", "double", or "extended")
 */
static inline const char* ss_precision_mode(void) {
#if defined(STONESOUP_PRECISION_SINGLE)
    return "single";
#elif defined(STONESOUP_PRECISION_EXTENDED)
    return "extended";
#else
    return "double";
#endif
}

/**
 * @brief Get size of ss_real_t in bytes
 * @return Size in bytes
 */
static inline size_t ss_real_size(void) {
    return sizeof(ss_real_t);
}

/**
 * @brief Get number of significant decimal digits
 * @return Number of significant digits
 */
static inline int ss_real_digits(void) {
    return SS_REAL_DIG;
}

#ifdef __cplusplus
}
#endif

#endif /* STONESOUP_PRECISION_H */
