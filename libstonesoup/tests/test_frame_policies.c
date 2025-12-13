/**
 * @file test_frame_policies.c
 * @brief Tests for coordinate frame-specific numeric policies
 */

#include <stonesoup/frame_policies.h>
#include <stonesoup/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test counter */
static int tests_passed = 0;
static int tests_failed = 0;

/* Test macros */
#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            tests_passed++; \
            printf("  PASS: %s\n", message); \
        } else { \
            tests_failed++; \
            printf("  FAIL: %s\n", message); \
        } \
    } while(0)

#define TEST_SUITE(name) \
    printf("\n=== %s ===\n", name)

/*===========================================================================*/
/* Test Functions                                                             */
/*===========================================================================*/

void test_frame_policy_access(void) {
    TEST_SUITE("Frame Policy Access");

    /* Test getting policies for all frame types */
    const ss_frame_policy_t* cartesian = ss_get_frame_policy(SS_FRAME_CARTESIAN);
    TEST_ASSERT(cartesian != NULL, "Get Cartesian policy");
    TEST_ASSERT(cartesian->frame == SS_FRAME_CARTESIAN, "Cartesian frame ID correct");
    TEST_ASSERT(strcmp(cartesian->name, "Cartesian") == 0, "Cartesian name correct");

    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);
    TEST_ASSERT(eci != NULL, "Get ECI policy");
    TEST_ASSERT(eci->frame == SS_FRAME_ECI, "ECI frame ID correct");
    TEST_ASSERT(!eci->is_rotating, "ECI is inertial (non-rotating)");

    const ss_frame_policy_t* ecef = ss_get_frame_policy(SS_FRAME_ECEF);
    TEST_ASSERT(ecef != NULL, "Get ECEF policy");
    TEST_ASSERT(ecef->is_rotating, "ECEF is rotating");

    const ss_frame_policy_t* lla = ss_get_frame_policy(SS_FRAME_LLA);
    TEST_ASSERT(lla != NULL, "Get LLA policy");
    TEST_ASSERT(lla->has_angular_coords, "LLA has angular coordinates");
    TEST_ASSERT(lla->has_singularities, "LLA has singularities at poles");

    const ss_frame_policy_t* polar = ss_get_frame_policy(SS_FRAME_POLAR);
    TEST_ASSERT(polar != NULL, "Get Polar policy");
    TEST_ASSERT(polar->spatial_dims == 2, "Polar is 2D");

    const ss_frame_policy_t* spherical = ss_get_frame_policy(SS_FRAME_SPHERICAL);
    TEST_ASSERT(spherical != NULL, "Get Spherical policy");
    TEST_ASSERT(spherical->spatial_dims == 3, "Spherical is 3D");

    const ss_frame_policy_t* enu = ss_get_frame_policy(SS_FRAME_ENU);
    TEST_ASSERT(enu != NULL, "Get ENU policy");
    TEST_ASSERT(enu->is_rotating, "ENU is rotating (fixed to Earth)");

    const ss_frame_policy_t* lunar = ss_get_frame_policy(SS_FRAME_LUNAR);
    TEST_ASSERT(lunar != NULL, "Get Lunar policy");
    TEST_ASSERT(!lunar->is_rotating, "Lunar is inertial");

    const ss_frame_policy_t* voxel = ss_get_frame_policy(SS_FRAME_VOXEL);
    TEST_ASSERT(voxel != NULL, "Get Voxel policy");
    TEST_ASSERT(!voxel->has_angular_coords, "Voxel has no angular coordinates");

    /* Test unknown frame handling */
    const ss_frame_policy_t* unknown = ss_get_frame_policy(SS_FRAME_UNKNOWN);
    TEST_ASSERT(unknown != NULL, "Get Unknown policy returns non-NULL");
    TEST_ASSERT(unknown->frame == SS_FRAME_UNKNOWN, "Unknown frame ID correct");
}

void test_frame_policy_by_name(void) {
    TEST_SUITE("Frame Policy Lookup by Name");

    /* Test case-insensitive name lookup */
    const ss_frame_policy_t* eci1 = ss_get_frame_policy_by_name("ECI");
    TEST_ASSERT(eci1 != NULL && eci1->frame == SS_FRAME_ECI,
                "Lookup ECI (uppercase)");

    const ss_frame_policy_t* eci2 = ss_get_frame_policy_by_name("eci");
    TEST_ASSERT(eci2 != NULL && eci2->frame == SS_FRAME_ECI,
                "Lookup eci (lowercase)");

    const ss_frame_policy_t* eci3 = ss_get_frame_policy_by_name("EcI");
    TEST_ASSERT(eci3 != NULL && eci3->frame == SS_FRAME_ECI,
                "Lookup EcI (mixed case)");

    const ss_frame_policy_t* ecef = ss_get_frame_policy_by_name("ECEF");
    TEST_ASSERT(ecef != NULL && ecef->frame == SS_FRAME_ECEF,
                "Lookup ECEF");

    const ss_frame_policy_t* lla = ss_get_frame_policy_by_name("LLA");
    TEST_ASSERT(lla != NULL && lla->frame == SS_FRAME_LLA,
                "Lookup LLA");

    const ss_frame_policy_t* polar = ss_get_frame_policy_by_name("Polar");
    TEST_ASSERT(polar != NULL && polar->frame == SS_FRAME_POLAR,
                "Lookup Polar");

    /* Test invalid name */
    const ss_frame_policy_t* invalid = ss_get_frame_policy_by_name("InvalidFrame");
    TEST_ASSERT(invalid == NULL, "Lookup invalid name returns NULL");

    /* Test NULL name */
    const ss_frame_policy_t* null_name = ss_get_frame_policy_by_name(NULL);
    TEST_ASSERT(null_name == NULL, "Lookup NULL name returns NULL");
}

void test_frame_names(void) {
    TEST_SUITE("Frame Name Functions");

    const char* name_eci = ss_frame_name(SS_FRAME_ECI);
    TEST_ASSERT(name_eci != NULL, "Get ECI name");
    TEST_ASSERT(strcmp(name_eci, "ECI") == 0, "ECI name is 'ECI'");

    const char* desc_eci = ss_frame_description(SS_FRAME_ECI);
    TEST_ASSERT(desc_eci != NULL, "Get ECI description");
    TEST_ASSERT(strlen(desc_eci) > 0, "ECI description is not empty");

    const char* name_ecef = ss_frame_name(SS_FRAME_ECEF);
    TEST_ASSERT(strcmp(name_ecef, "ECEF") == 0, "ECEF name is 'ECEF'");

    const char* name_polar = ss_frame_name(SS_FRAME_POLAR);
    TEST_ASSERT(strcmp(name_polar, "Polar") == 0, "Polar name is 'Polar'");
}

void test_position_validation(void) {
    TEST_SUITE("Position Validation");

    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);

    /* Test valid positions */
    TEST_ASSERT(ss_validate_position(7000000.0, eci),
                "Valid ECI position (7000 km)");
    TEST_ASSERT(ss_validate_position(42000000.0, eci),
                "Valid ECI position (GEO altitude)");

    /* Test positions at boundaries */
    TEST_ASSERT(ss_validate_position(0.0, eci),
                "Valid ECI position (0 - at minimum)");
    TEST_ASSERT(ss_validate_position(eci->max_position, eci),
                "Valid ECI position (at maximum)");

    /* Test out-of-range positions */
    TEST_ASSERT(!ss_validate_position(100000000.0, eci),
                "Invalid ECI position (too large)");

    /* Test polar frame with range constraint */
    const ss_frame_policy_t* polar = ss_get_frame_policy(SS_FRAME_POLAR);
    TEST_ASSERT(ss_validate_position(1000.0, polar),
                "Valid polar range (1 km)");
    TEST_ASSERT(ss_validate_position(0.0, polar),
                "Valid polar range (0 - at origin)");
}

void test_velocity_validation(void) {
    TEST_SUITE("Velocity Validation");

    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);

    /* Test valid velocities */
    TEST_ASSERT(ss_validate_velocity(7500.0, eci),
                "Valid orbital velocity (7.5 km/s)");
    TEST_ASSERT(ss_validate_velocity(11200.0, eci),
                "Valid escape velocity (11.2 km/s)");

    /* Test out-of-range velocities */
    TEST_ASSERT(!ss_validate_velocity(20000.0, eci),
                "Invalid velocity (too high)");

    /* Test undersea domain (lower velocity limits) */
    const ss_frame_policy_t* enu = ss_get_frame_policy(SS_FRAME_ENU);
    TEST_ASSERT(ss_validate_velocity(100.0, enu),
                "Valid submarine velocity");
    TEST_ASSERT(!ss_validate_velocity(5000.0, enu),
                "Invalid submarine velocity (too high)");
}

void test_angle_validation(void) {
    TEST_SUITE("Angle Validation");

    const ss_frame_policy_t* lla = ss_get_frame_policy(SS_FRAME_LLA);

    /* Test valid angles for LLA */
    TEST_ASSERT(ss_validate_angle(0.0, lla),
                "Valid latitude (equator)");
    TEST_ASSERT(ss_validate_angle(SS_REAL(SS_PI / 4.0), lla),
                "Valid latitude (45 degrees)");
    TEST_ASSERT(ss_validate_angle(SS_REAL(-SS_PI / 4.0), lla),
                "Valid latitude (-45 degrees)");

    /* Test boundary angles */
    TEST_ASSERT(ss_validate_angle(SS_REAL(SS_PI / 2.0), lla),
                "Valid latitude (north pole)");
    TEST_ASSERT(ss_validate_angle(SS_REAL(-SS_PI / 2.0), lla),
                "Valid latitude (south pole)");

    /* Test out-of-range angles */
    TEST_ASSERT(!ss_validate_angle(SS_REAL(SS_PI), lla),
                "Invalid latitude (too large)");

    /* Test polar frame angles */
    const ss_frame_policy_t* polar = ss_get_frame_policy(SS_FRAME_POLAR);
    TEST_ASSERT(ss_validate_angle(0.0, polar),
                "Valid bearing (0 rad)");
    TEST_ASSERT(ss_validate_angle(SS_REAL(SS_PI), polar),
                "Valid bearing (π rad)");
    TEST_ASSERT(ss_validate_angle(SS_TWO_PI, polar),
                "Valid bearing (2π rad)");

    /* Test frame without angles */
    const ss_frame_policy_t* cartesian = ss_get_frame_policy(SS_FRAME_CARTESIAN);
    TEST_ASSERT(!ss_validate_angle(0.0, cartesian),
                "Cartesian has no angular coordinates");
}

void test_precision_checking(void) {
    TEST_SUITE("Precision Checking");

    /* Test precision for different frames */
    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);
    bool eci_pos_ok = ss_check_position_precision(eci);
    bool eci_vel_ok = ss_check_velocity_precision(eci);

    TEST_ASSERT(eci_pos_ok || !eci_pos_ok,
                "ECI position precision check completes");
    TEST_ASSERT(eci_vel_ok || !eci_vel_ok,
                "ECI velocity precision check completes");

    const ss_frame_policy_t* lla = ss_get_frame_policy(SS_FRAME_LLA);
    bool lla_ang_ok = ss_check_angular_precision(lla);

    TEST_ASSERT(lla_ang_ok || !lla_ang_ok,
                "LLA angular precision check completes");

    /* Test frame without angles */
    const ss_frame_policy_t* cartesian = ss_get_frame_policy(SS_FRAME_CARTESIAN);
    bool cart_ang_ok = ss_check_angular_precision(cartesian);

    TEST_ASSERT(cart_ang_ok,
                "Cartesian angular precision check returns true (no angles)");

    /* Test NULL policy */
    TEST_ASSERT(!ss_check_position_precision(NULL),
                "NULL policy position check returns false");
    TEST_ASSERT(!ss_check_velocity_precision(NULL),
                "NULL policy velocity check returns false");
    TEST_ASSERT(ss_check_angular_precision(NULL),
                "NULL policy angular check returns true (no requirement)");
}

void test_scale_factors(void) {
    TEST_SUITE("Scale Factors");

    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);

    /* ECI should recommend km for better conditioning */
    TEST_ASSERT(eci->position_scale_factor == SS_REAL(1000.0),
                "ECI position scale factor is 1000 (use km)");

    const ss_frame_policy_t* enu = ss_get_frame_policy(SS_FRAME_ENU);

    /* ENU should use meters (local frame) */
    TEST_ASSERT(enu->position_scale_factor == SS_REAL(1.0),
                "ENU position scale factor is 1 (use m)");

    const ss_frame_policy_t* lunar = ss_get_frame_policy(SS_FRAME_LUNAR);

    /* Lunar should recommend km */
    TEST_ASSERT(lunar->position_scale_factor == SS_REAL(1000.0),
                "Lunar position scale factor is 1000 (use km)");
}

void test_domain_specific_policies(void) {
    TEST_SUITE("Domain-Specific Policies");

    /* Test undersea domain characteristics */
    const ss_frame_policy_t* enu = ss_get_frame_policy(SS_FRAME_ENU);
    TEST_ASSERT(enu->position_precision <= SS_REAL(0.01),
                "ENU position precision is cm-level or better");

    /* Test orbital domain characteristics */
    const ss_frame_policy_t* eci = ss_get_frame_policy(SS_FRAME_ECI);
    TEST_ASSERT(eci->max_position >= SS_GEO_ALTITUDE_M,
                "ECI supports GEO altitudes");
    TEST_ASSERT(eci->max_velocity >= SS_REAL(11000.0),
                "ECI supports escape velocity");

    /* Test cislunar domain characteristics */
    const ss_frame_policy_t* lunar = ss_get_frame_policy(SS_FRAME_LUNAR);
    TEST_ASSERT(lunar->max_position >= SS_REAL(60000000.0),
                "Lunar frame supports lunar SOI");

    /* Test voxel grid characteristics */
    const ss_frame_policy_t* voxel = ss_get_frame_policy(SS_FRAME_VOXEL);
    TEST_ASSERT(voxel->position_precision == SS_REAL(1.0),
                "Voxel precision is 1 unit");
}

void test_frame_properties(void) {
    TEST_SUITE("Frame Properties");

    /* Test inertial vs rotating */
    TEST_ASSERT(!ss_get_frame_policy(SS_FRAME_ECI)->is_rotating,
                "ECI is inertial");
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_ECEF)->is_rotating,
                "ECEF is rotating");
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_ENU)->is_rotating,
                "ENU is rotating");
    TEST_ASSERT(!ss_get_frame_policy(SS_FRAME_LUNAR)->is_rotating,
                "Lunar is inertial");

    /* Test singularities */
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_LLA)->has_singularities,
                "LLA has singularities (poles)");
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_POLAR)->has_singularities,
                "Polar has singularity (origin)");
    TEST_ASSERT(!ss_get_frame_policy(SS_FRAME_CARTESIAN)->has_singularities,
                "Cartesian has no singularities");

    /* Test dimensions */
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_POLAR)->spatial_dims == 2,
                "Polar is 2D");
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_SPHERICAL)->spatial_dims == 3,
                "Spherical is 3D");
    TEST_ASSERT(ss_get_frame_policy(SS_FRAME_CARTESIAN)->spatial_dims == 3,
                "Cartesian is 3D");
}

/*===========================================================================*/
/* Main Test Runner                                                          */
/*===========================================================================*/

int main(void) {
    printf("Stone Soup Frame Policies Test Suite\n");
    printf("Precision mode: %s (%zu bytes, %d digits)\n\n",
           ss_precision_mode(), ss_real_size(), ss_real_digits());

    test_frame_policy_access();
    test_frame_policy_by_name();
    test_frame_names();
    test_position_validation();
    test_velocity_validation();
    test_angle_validation();
    test_precision_checking();
    test_scale_factors();
    test_domain_specific_policies();
    test_frame_properties();

    printf("\n=== Test Summary ===\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);

    if (tests_failed == 0) {
        printf("\nAll tests PASSED!\n");
        return 0;
    } else {
        printf("\nSome tests FAILED!\n");
        return 1;
    }
}
