/**
 * @file version.h
 * @brief Version information for the Stone Soup C library
 */

#ifndef STONESOUP_VERSION_H
#define STONESOUP_VERSION_H

/**
 * @defgroup version Version Information
 * @brief Version macros and constants
 * @{
 */

/** Major version number */
#define STONESOUP_VERSION_MAJOR 0

/** Minor version number */
#define STONESOUP_VERSION_MINOR 1

/** Patch version number */
#define STONESOUP_VERSION_PATCH 0

/** Full version string */
#define STONESOUP_VERSION_STRING "0.1.0"

/**
 * @brief Combined version number for comparison
 *
 * Format: MMmmpp (Major, minor, patch as 2 digits each)
 */
#define STONESOUP_VERSION \
    (STONESOUP_VERSION_MAJOR * 10000 + \
     STONESOUP_VERSION_MINOR * 100 + \
     STONESOUP_VERSION_PATCH)

/**
 * @brief Get version string at runtime
 * @return Version string (e.g., "0.1.0")
 */
const char* stonesoup_version(void);

/**
 * @brief Get major version number at runtime
 * @return Major version number
 */
int stonesoup_version_major(void);

/**
 * @brief Get minor version number at runtime
 * @return Minor version number
 */
int stonesoup_version_minor(void);

/**
 * @brief Get patch version number at runtime
 * @return Patch version number
 */
int stonesoup_version_patch(void);

/** @} */

#endif /* STONESOUP_VERSION_H */
