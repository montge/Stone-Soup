/**
 * @file test_coordinates.c
 * @brief Comprehensive tests for coordinate transformations and precision preservation
 *
 * This test suite validates:
 * - Coordinate transformation correctness
 * - Precision preservation across transformations
 * - Round-trip transformation accuracy
 * - Multi-scale domain transitions
 * - Edge cases and boundary conditions
 */

#define _USE_MATH_DEFINES
#include <stonesoup/stonesoup.h>
#include <stonesoup/precision.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Define M_PI if not available */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON_SINGLE 1e-6     /* Tolerance for single precision */
#define EPSILON_DOUBLE 1e-12    /* Tolerance for double precision */
#define EPSILON_EXTENDED 1e-16  /* Tolerance for extended precision */

/* Use appropriate epsilon based on precision mode */
#ifdef STONESOUP_PRECISION_SINGLE
    #define EPSILON EPSILON_SINGLE
#elif defined(STONESOUP_PRECISION_EXTENDED)
    #define EPSILON EPSILON_EXTENDED
#else
    #define EPSILON EPSILON_DOUBLE
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("Running test: %s ... ", #name); \
        fflush(stdout); \
        tests_run++; \
        if (name()) { \
            tests_passed++; \
            printf("PASSED\n"); \
        } else { \
            printf("FAILED\n"); \
        } \
    } while(0)

/* Helper function for comparing doubles with relative tolerance */
static int double_equals(double a, double b, double epsilon) {
    if (fabs(a) < 1e-10 && fabs(b) < 1e-10) {
        return fabs(a - b) < epsilon;
    }
    double rel_error = fabs((a - b) / (fabs(a) > fabs(b) ? a : b));
    return rel_error < epsilon;
}

/* Helper to check if values are close with absolute tolerance */
static int approx_equal(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

/*===========================================================================*/
/* Basic Coordinate Transformation Tests                                     */
/*===========================================================================*/

/**
 * Test basic Cartesian to polar conversion (2D)
 */
static int test_cart2polar_basic(void) {
    stonesoup_state_vector_t* cart = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* polar = stonesoup_state_vector_create(2);

    if (!cart || !polar) {
        stonesoup_state_vector_free(cart);
        stonesoup_state_vector_free(polar);
        return 0;
    }

    /* Test point (3, 4) -> range=5, bearing=atan2(4,3)=0.927... */
    cart->data[0] = 3.0;
    cart->data[1] = 4.0;

    stonesoup_error_t err = stonesoup_cart2polar(cart, polar);

    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(polar->data[0], 5.0, EPSILON) &&
                  approx_equal(polar->data[1], atan2(4.0, 3.0), EPSILON);

    stonesoup_state_vector_free(cart);
    stonesoup_state_vector_free(polar);

    return success;
}

/**
 * Test polar to Cartesian conversion (2D)
 */
static int test_polar2cart_basic(void) {
    stonesoup_state_vector_t* polar = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* cart = stonesoup_state_vector_create(2);

    if (!polar || !cart) {
        stonesoup_state_vector_free(polar);
        stonesoup_state_vector_free(cart);
        return 0;
    }

    /* Test range=5, bearing=45 degrees (π/4) -> (3.535..., 3.535...) */
    polar->data[0] = 5.0;
    polar->data[1] = M_PI / 4.0;

    stonesoup_error_t err = stonesoup_polar2cart(polar, cart);

    double expected = 5.0 / sqrt(2.0);
    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(cart->data[0], expected, EPSILON) &&
                  approx_equal(cart->data[1], expected, EPSILON);

    stonesoup_state_vector_free(polar);
    stonesoup_state_vector_free(cart);

    return success;
}

/**
 * Test Cartesian to spherical conversion (3D)
 */
static int test_cart2sphere_basic(void) {
    stonesoup_state_vector_t* cart = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);

    if (!cart || !sphere) {
        stonesoup_state_vector_free(cart);
        stonesoup_state_vector_free(sphere);
        return 0;
    }

    /* Test point (1, 0, 0) -> range=1, azimuth=0, elevation=0 */
    cart->data[0] = 1.0;
    cart->data[1] = 0.0;
    cart->data[2] = 0.0;

    stonesoup_error_t err = stonesoup_cart2sphere(cart, sphere);

    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(sphere->data[0], 1.0, EPSILON) &&
                  approx_equal(sphere->data[1], 0.0, EPSILON) &&
                  approx_equal(sphere->data[2], 0.0, EPSILON);

    stonesoup_state_vector_free(cart);
    stonesoup_state_vector_free(sphere);

    return success;
}

/**
 * Test spherical to Cartesian conversion (3D)
 */
static int test_sphere2cart_basic(void) {
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart = stonesoup_state_vector_create(3);

    if (!sphere || !cart) {
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart);
        return 0;
    }

    /* Test range=10, azimuth=90°, elevation=0° -> (0, 10, 0) */
    sphere->data[0] = 10.0;
    sphere->data[1] = M_PI / 2.0;  /* 90 degrees */
    sphere->data[2] = 0.0;

    stonesoup_error_t err = stonesoup_sphere2cart(sphere, cart);

    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(cart->data[0], 0.0, EPSILON) &&
                  approx_equal(cart->data[1], 10.0, EPSILON) &&
                  approx_equal(cart->data[2], 0.0, EPSILON);

    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart);

    return success;
}

/*===========================================================================*/
/* Round-Trip Precision Tests                                                */
/*===========================================================================*/

/**
 * Test round-trip precision for polar coordinates
 */
static int test_polar_roundtrip_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* polar = stonesoup_state_vector_create(2);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(2);

    if (!cart1 || !polar || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(polar);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* Original point */
    cart1->data[0] = 123.456;
    cart1->data[1] = 789.012;

    /* Forward and back */
    stonesoup_cart2polar(cart1, polar);
    stonesoup_polar2cart(polar, cart2);

    /* Check precision preserved */
    int success = approx_equal(cart1->data[0], cart2->data[0], EPSILON * 10.0) &&
                  approx_equal(cart1->data[1], cart2->data[1], EPSILON * 10.0);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(polar);
    stonesoup_state_vector_free(cart2);

    return success;
}

/**
 * Test round-trip precision for spherical coordinates
 */
static int test_sphere_roundtrip_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* Original point */
    cart1->data[0] = 100.0;
    cart1->data[1] = 200.0;
    cart1->data[2] = 300.0;

    /* Forward and back */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Check precision preserved */
    int success = approx_equal(cart1->data[0], cart2->data[0], EPSILON * 100.0) &&
                  approx_equal(cart1->data[1], cart2->data[1], EPSILON * 100.0) &&
                  approx_equal(cart1->data[2], cart2->data[2], EPSILON * 100.0);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/*===========================================================================*/
/* Domain-Specific Precision Tests                                           */
/*===========================================================================*/

/**
 * Test precision preservation for undersea domain coordinates
 * Range: 0-100 km, required precision: 1 cm
 */
static int test_undersea_domain_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* Typical undersea tracking scenario: 50 km range, 1 km depth */
    cart1->data[0] = 30000.0;  /* 30 km */
    cart1->data[1] = 40000.0;  /* 40 km */
    cart1->data[2] = -1000.0;  /* 1 km depth */

    /* Round-trip transformation */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Precision should be better than 1 cm (0.01 m) */
    double precision_required = 0.01;
    int success = approx_equal(cart1->data[0], cart2->data[0], precision_required) &&
                  approx_equal(cart1->data[1], cart2->data[1], precision_required) &&
                  approx_equal(cart1->data[2], cart2->data[2], precision_required);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/**
 * Test precision preservation for orbital domain coordinates (LEO)
 * Range: 400-2000 km altitude, required precision: 1 m
 */
static int test_leo_domain_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* LEO orbit: ~7000 km from Earth center */
    double orbit_radius = 7000000.0;  /* 7000 km in meters */
    cart1->data[0] = orbit_radius / sqrt(3.0);
    cart1->data[1] = orbit_radius / sqrt(3.0);
    cart1->data[2] = orbit_radius / sqrt(3.0);

    /* Round-trip transformation */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Precision should be better than 1 m */
    double precision_required = 1.0;
    int success = approx_equal(cart1->data[0], cart2->data[0], precision_required) &&
                  approx_equal(cart1->data[1], cart2->data[1], precision_required) &&
                  approx_equal(cart1->data[2], cart2->data[2], precision_required);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/**
 * Test precision preservation for GEO orbital domain
 * Range: ~36000 km altitude, required precision: 10 m
 */
static int test_geo_domain_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* GEO orbit: ~42000 km from Earth center */
    double orbit_radius = 42000000.0;  /* 42000 km in meters */
    cart1->data[0] = orbit_radius;
    cart1->data[1] = 0.0;
    cart1->data[2] = 0.0;

    /* Round-trip transformation */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Precision should be better than 10 m */
    double precision_required = 10.0;
    int success = approx_equal(cart1->data[0], cart2->data[0], precision_required) &&
                  approx_equal(cart1->data[1], cart2->data[1], precision_required) &&
                  approx_equal(cart1->data[2], cart2->data[2], precision_required);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/**
 * Test precision preservation for cislunar domain
 * Range: ~384,000 km, required precision: 100 m
 */
static int test_cislunar_domain_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* Earth-Moon distance: ~384,000 km */
    double moon_distance = 384400000.0;  /* meters */
    cart1->data[0] = moon_distance;
    cart1->data[1] = 0.0;
    cart1->data[2] = 0.0;

    /* Round-trip transformation */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Precision should be better than 100 m */
    double precision_required = 100.0;
    int success = approx_equal(cart1->data[0], cart2->data[0], precision_required) &&
                  approx_equal(cart1->data[1], cart2->data[1], precision_required) &&
                  approx_equal(cart1->data[2], cart2->data[2], precision_required);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/*===========================================================================*/
/* Multi-Scale Transition Tests                                              */
/*===========================================================================*/

/**
 * Test transition from LEO to GEO (multi-scale within orbital domain)
 */
static int test_leo_to_geo_transition(void) {
    stonesoup_state_vector_t* leo = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* geo = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);

    if (!leo || !geo || !sphere) {
        stonesoup_state_vector_free(leo);
        stonesoup_state_vector_free(geo);
        stonesoup_state_vector_free(sphere);
        return 0;
    }

    /* LEO position */
    double leo_radius = 7000000.0;  /* 7000 km */
    leo->data[0] = leo_radius;
    leo->data[1] = 0.0;
    leo->data[2] = 0.0;

    /* GEO position */
    double geo_radius = 42000000.0;  /* 42000 km */
    geo->data[0] = geo_radius;
    geo->data[1] = 0.0;
    geo->data[2] = 0.0;

    /* Convert both to spherical */
    stonesoup_state_vector_t* leo_sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* geo_sphere = stonesoup_state_vector_create(3);

    stonesoup_cart2sphere(leo, leo_sphere);
    stonesoup_cart2sphere(geo, geo_sphere);

    /* Verify both transformations preserve relative accuracy */
    int success = approx_equal(leo_sphere->data[0], leo_radius, 1.0) &&
                  approx_equal(geo_sphere->data[0], geo_radius, 10.0);

    stonesoup_state_vector_free(leo);
    stonesoup_state_vector_free(geo);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(leo_sphere);
    stonesoup_state_vector_free(geo_sphere);

    return success;
}

/**
 * Test transition from GEO to lunar orbit (orbital to cislunar)
 */
static int test_geo_to_lunar_transition(void) {
    stonesoup_state_vector_t* geo = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* lunar = stonesoup_state_vector_create(3);

    if (!geo || !lunar) {
        stonesoup_state_vector_free(geo);
        stonesoup_state_vector_free(lunar);
        return 0;
    }

    /* GEO position */
    double geo_radius = 42000000.0;
    geo->data[0] = geo_radius;
    geo->data[1] = 0.0;
    geo->data[2] = 0.0;

    /* Lunar orbit position (simplified) */
    double lunar_radius = 384400000.0;
    lunar->data[0] = lunar_radius;
    lunar->data[1] = 0.0;
    lunar->data[2] = 0.0;

    /* Convert both */
    stonesoup_state_vector_t* geo_sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* lunar_sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* geo_back = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* lunar_back = stonesoup_state_vector_create(3);

    /* Round-trip for both scales */
    stonesoup_cart2sphere(geo, geo_sphere);
    stonesoup_sphere2cart(geo_sphere, geo_back);

    stonesoup_cart2sphere(lunar, lunar_sphere);
    stonesoup_sphere2cart(lunar_sphere, lunar_back);

    /* Both should maintain precision appropriate to their scale */
    int success = approx_equal(geo->data[0], geo_back->data[0], 10.0) &&
                  approx_equal(lunar->data[0], lunar_back->data[0], 100.0);

    stonesoup_state_vector_free(geo);
    stonesoup_state_vector_free(lunar);
    stonesoup_state_vector_free(geo_sphere);
    stonesoup_state_vector_free(lunar_sphere);
    stonesoup_state_vector_free(geo_back);
    stonesoup_state_vector_free(lunar_back);

    return success;
}

/**
 * Test transition from undersea to orbital domain
 * (extreme scale change: ~10 km to ~10,000 km)
 */
static int test_undersea_to_orbital_transition(void) {
    stonesoup_state_vector_t* undersea = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* orbital = stonesoup_state_vector_create(3);

    if (!undersea || !orbital) {
        stonesoup_state_vector_free(undersea);
        stonesoup_state_vector_free(orbital);
        return 0;
    }

    /* Undersea position: 50 km range */
    undersea->data[0] = 30000.0;
    undersea->data[1] = 40000.0;
    undersea->data[2] = -1000.0;

    /* Orbital position: 7000 km */
    orbital->data[0] = 5000000.0;
    orbital->data[1] = 5000000.0;
    orbital->data[2] = 0.0;

    /* Test transformations maintain appropriate precision for each scale */
    stonesoup_state_vector_t* undersea_sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* orbital_sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* undersea_back = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* orbital_back = stonesoup_state_vector_create(3);

    stonesoup_cart2sphere(undersea, undersea_sphere);
    stonesoup_sphere2cart(undersea_sphere, undersea_back);

    stonesoup_cart2sphere(orbital, orbital_sphere);
    stonesoup_sphere2cart(orbital_sphere, orbital_back);

    /* Undersea: 1 cm precision, Orbital: 1 m precision */
    int success = approx_equal(undersea->data[0], undersea_back->data[0], 0.01) &&
                  approx_equal(undersea->data[1], undersea_back->data[1], 0.01) &&
                  approx_equal(orbital->data[0], orbital_back->data[0], 1.0) &&
                  approx_equal(orbital->data[1], orbital_back->data[1], 1.0);

    stonesoup_state_vector_free(undersea);
    stonesoup_state_vector_free(orbital);
    stonesoup_state_vector_free(undersea_sphere);
    stonesoup_state_vector_free(orbital_sphere);
    stonesoup_state_vector_free(undersea_back);
    stonesoup_state_vector_free(orbital_back);

    return success;
}

/**
 * Test very large scale transformation (interplanetary distances)
 */
static int test_interplanetary_scale_precision(void) {
    stonesoup_state_vector_t* cart1 = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* sphere = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* cart2 = stonesoup_state_vector_create(3);

    if (!cart1 || !sphere || !cart2) {
        stonesoup_state_vector_free(cart1);
        stonesoup_state_vector_free(sphere);
        stonesoup_state_vector_free(cart2);
        return 0;
    }

    /* Mars distance: ~225 million km */
    double mars_distance = 225000000000.0;  /* meters */
    cart1->data[0] = mars_distance;
    cart1->data[1] = 0.0;
    cart1->data[2] = 0.0;

    /* Round-trip transformation */
    stonesoup_cart2sphere(cart1, sphere);
    stonesoup_sphere2cart(sphere, cart2);

    /* Precision should be better than 1 km at this scale */
    double precision_required = 1000.0;
    int success = approx_equal(cart1->data[0], cart2->data[0], precision_required);

    stonesoup_state_vector_free(cart1);
    stonesoup_state_vector_free(sphere);
    stonesoup_state_vector_free(cart2);

    return success;
}

/*===========================================================================*/
/* Geodetic Transformation Tests                                             */
/*===========================================================================*/

/**
 * Test geodetic to ECEF conversion
 */
static int test_geodetic2ecef_basic(void) {
    stonesoup_state_vector_t* geodetic = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* ecef = stonesoup_state_vector_create(3);

    if (!geodetic || !ecef) {
        stonesoup_state_vector_free(geodetic);
        stonesoup_state_vector_free(ecef);
        return 0;
    }

    /* Equator, prime meridian, sea level: (0, 0, 0) */
    geodetic->data[0] = 0.0;  /* latitude */
    geodetic->data[1] = 0.0;  /* longitude */
    geodetic->data[2] = 0.0;  /* altitude */

    stonesoup_error_t err = stonesoup_geodetic2ecef(geodetic, ecef);

    /* Should give (~6378137, 0, 0) - Earth's semi-major axis */
    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(ecef->data[0], 6378137.0, 1.0) &&
                  approx_equal(ecef->data[1], 0.0, 1.0) &&
                  approx_equal(ecef->data[2], 0.0, 1.0);

    stonesoup_state_vector_free(geodetic);
    stonesoup_state_vector_free(ecef);

    return success;
}

/**
 * Test geodetic conversion at North Pole
 */
static int test_geodetic2ecef_north_pole(void) {
    stonesoup_state_vector_t* geodetic = stonesoup_state_vector_create(3);
    stonesoup_state_vector_t* ecef = stonesoup_state_vector_create(3);

    if (!geodetic || !ecef) {
        stonesoup_state_vector_free(geodetic);
        stonesoup_state_vector_free(ecef);
        return 0;
    }

    /* North pole, sea level */
    geodetic->data[0] = M_PI / 2.0;  /* latitude 90° */
    geodetic->data[1] = 0.0;         /* longitude (irrelevant at pole) */
    geodetic->data[2] = 0.0;         /* altitude */

    stonesoup_error_t err = stonesoup_geodetic2ecef(geodetic, ecef);

    /* Should give (0, 0, ~6356752) - Earth's semi-minor axis */
    int success = (err == STONESOUP_SUCCESS) &&
                  approx_equal(ecef->data[0], 0.0, 1.0) &&
                  approx_equal(ecef->data[1], 0.0, 1.0) &&
                  approx_equal(ecef->data[2], 6356752.0, 1000.0);  /* Allow larger tolerance */

    stonesoup_state_vector_free(geodetic);
    stonesoup_state_vector_free(ecef);

    return success;
}

/*===========================================================================*/
/* Main Test Runner                                                          */
/*===========================================================================*/

int main(void) {
    printf("========================================\n");
    printf("Stone Soup Coordinate Transformation Tests\n");
    printf("Precision mode: %s\n", ss_precision_mode());
    printf("Size of ss_real_t: %zu bytes\n", ss_real_size());
    printf("Significant digits: %d\n", ss_real_digits());
    printf("Test epsilon: %e\n", EPSILON);
    printf("========================================\n\n");

    /* Basic transformation tests */
    TEST(test_cart2polar_basic);
    TEST(test_polar2cart_basic);
    TEST(test_cart2sphere_basic);
    TEST(test_sphere2cart_basic);

    /* Round-trip precision tests */
    TEST(test_polar_roundtrip_precision);
    TEST(test_sphere_roundtrip_precision);

    /* Domain-specific precision tests */
    TEST(test_undersea_domain_precision);
    TEST(test_leo_domain_precision);
    TEST(test_geo_domain_precision);
    TEST(test_cislunar_domain_precision);

    /* Multi-scale transition tests */
    TEST(test_leo_to_geo_transition);
    TEST(test_geo_to_lunar_transition);
    TEST(test_undersea_to_orbital_transition);
    TEST(test_interplanetary_scale_precision);

    /* Geodetic transformation tests */
    TEST(test_geodetic2ecef_basic);
    TEST(test_geodetic2ecef_north_pole);

    printf("\n========================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("========================================\n");

    return (tests_run == tests_passed) ? 0 : 1;
}
