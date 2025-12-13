/**
 * @file frame_policies.c
 * @brief Implementation of coordinate frame-specific numeric policies
 */

#include "stonesoup/frame_policies.h"
#include <string.h>
#include <ctype.h>

/*===========================================================================*/
/* Frame Policy Definitions                                                  */
/*===========================================================================*/

/* Generic Cartesian frame - wide range for flexibility */
static const ss_frame_policy_t policy_cartesian = {
    .frame = SS_FRAME_CARTESIAN,
    .name = "Cartesian",
    .description = "Generic Cartesian coordinates (x, y, z)",
    .position_precision = SS_REAL(0.001),        /* 1 mm */
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(1.0e12),             /* 1 billion km */
    .position_scale_factor = SS_REAL(1.0),       /* meters */
    .velocity_precision = SS_REAL(0.0001),       /* 0.1 mm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(100000.0),           /* 100 km/s */
    .velocity_scale_factor = SS_REAL(1.0),       /* m/s */
    .time_precision = SS_REAL(0.001),            /* 1 ms */
    .max_time_interval = SS_REAL(86400.0 * 365.25), /* 1 year */
    .angular_precision = SS_REAL(0.0),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_REAL(0.0),
    .is_rotating = false,
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/* 2D Polar coordinates */
static const ss_frame_policy_t policy_polar = {
    .frame = SS_FRAME_POLAR,
    .name = "Polar",
    .description = "2D polar coordinates (range, bearing)",
    .position_precision = SS_REAL(0.01),         /* 1 cm */
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(100000.0),           /* 100 km typical */
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.001),        /* 1 mm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(1000.0),             /* 1 km/s typical */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(3600.0),        /* 1 hour typical */
    .angular_precision = SS_REAL(1.0e-6),        /* ~1 micro-radian */
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = false,
    .has_angular_coords = true,
    .has_singularities = true,                   /* singularity at origin */
    .spatial_dims = 2
};

/* 3D Spherical coordinates */
static const ss_frame_policy_t policy_spherical = {
    .frame = SS_FRAME_SPHERICAL,
    .name = "Spherical",
    .description = "3D spherical coordinates (range, azimuth, elevation)",
    .position_precision = SS_REAL(0.1),          /* 10 cm */
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(1000000.0),          /* 1000 km typical */
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.01),         /* 1 cm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(10000.0),            /* 10 km/s typical */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(3600.0),
    .angular_precision = SS_REAL(1.0e-6),
    .min_angle = SS_REAL(-SS_PI / 2.0),          /* elevation range */
    .max_angle = SS_REAL(SS_PI / 2.0),
    .is_rotating = false,
    .has_angular_coords = true,
    .has_singularities = true,                   /* singularity at origin, poles */
    .spatial_dims = 3
};

/* Earth-Centered Inertial (ECI) */
static const ss_frame_policy_t policy_eci = {
    .frame = SS_FRAME_ECI,
    .name = "ECI",
    .description = "Earth-Centered Inertial (non-rotating)",
    .position_precision = SS_REAL(1.0),          /* 1 m */
    .min_position = SS_REAL(0.0),
    .max_position = SS_ORBITAL_ALTITUDE_MAX,     /* 50,000 km */
    .position_scale_factor = SS_REAL(1000.0),    /* use km for better conditioning */
    .velocity_precision = SS_REAL(0.01),         /* 1 cm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_ORBITAL_VELOCITY_MAX,     /* 12 km/s */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(86400.0 * 30.0), /* 30 days */
    .angular_precision = SS_REAL(1.0e-9),        /* ~1 nano-radian for precision orbits */
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = false,
    .has_angular_coords = false,                 /* Cartesian representation */
    .has_singularities = false,
    .spatial_dims = 3
};

/* Earth-Centered Earth-Fixed (ECEF) */
static const ss_frame_policy_t policy_ecef = {
    .frame = SS_FRAME_ECEF,
    .name = "ECEF",
    .description = "Earth-Centered Earth-Fixed (rotating)",
    .position_precision = SS_REAL(1.0),          /* 1 m */
    .min_position = SS_REAL(0.0),
    .max_position = SS_ORBITAL_ALTITUDE_MAX,
    .position_scale_factor = SS_REAL(1000.0),    /* use km */
    .velocity_precision = SS_REAL(0.01),         /* 1 cm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_ORBITAL_VELOCITY_MAX,
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(86400.0),       /* 1 day (frame rotates) */
    .angular_precision = SS_REAL(1.0e-9),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = true,                         /* rotates with Earth */
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/* Latitude-Longitude-Altitude (Geodetic) */
static const ss_frame_policy_t policy_lla = {
    .frame = SS_FRAME_LLA,
    .name = "LLA",
    .description = "Latitude-Longitude-Altitude (geodetic)",
    .position_precision = SS_REAL(1.0),          /* 1 m altitude precision */
    .min_position = SS_REAL(-500.0),             /* below sea level */
    .max_position = SS_ORBITAL_ALTITUDE_MAX,     /* max altitude */
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.01),
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_ORBITAL_VELOCITY_MAX,
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(86400.0),
    .angular_precision = SS_REAL(1.0e-9),        /* ~6 cm at equator */
    .min_angle = SS_REAL(-SS_PI / 2.0),          /* latitude range */
    .max_angle = SS_REAL(SS_PI / 2.0),
    .is_rotating = true,
    .has_angular_coords = true,
    .has_singularities = true,                   /* poles */
    .spatial_dims = 3
};

/* East-North-Up (Local Tangent Plane) */
static const ss_frame_policy_t policy_enu = {
    .frame = SS_FRAME_ENU,
    .name = "ENU",
    .description = "East-North-Up local tangent plane",
    .position_precision = SS_REAL(0.01),         /* 1 cm */
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(500000.0),           /* 500 km typical local range */
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.001),        /* 1 mm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(1000.0),             /* 1 km/s typical */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(3600.0),        /* 1 hour typical */
    .angular_precision = SS_REAL(1.0e-6),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = true,                         /* rotates with Earth */
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/* Lunar-Centered Inertial */
static const ss_frame_policy_t policy_lunar = {
    .frame = SS_FRAME_LUNAR,
    .name = "Lunar",
    .description = "Lunar-Centered Inertial",
    .position_precision = SS_REAL(10.0),         /* 10 m */
    .min_position = SS_REAL(0.0),
    .max_position = SS_LUNAR_SOI_M,              /* Lunar sphere of influence */
    .position_scale_factor = SS_REAL(1000.0),    /* use km */
    .velocity_precision = SS_REAL(0.1),          /* 10 cm/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_CISLUNAR_VELOCITY_MAX,    /* 15 km/s */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.01),             /* 10 ms */
    .max_time_interval = SS_REAL(86400.0 * 30.0), /* 30 days */
    .angular_precision = SS_REAL(1.0e-8),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = false,
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/* Voxel Grid (discrete integer coordinates) */
static const ss_frame_policy_t policy_voxel = {
    .frame = SS_FRAME_VOXEL,
    .name = "Voxel",
    .description = "Discrete voxel grid coordinates",
    .position_precision = SS_REAL(1.0),          /* 1 voxel = minimum unit */
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(10000.0),            /* 10k voxels per dimension typical */
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.1),          /* 0.1 voxels/s */
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(100.0),              /* 100 voxels/s typical */
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.01),
    .max_time_interval = SS_REAL(3600.0),
    .angular_precision = SS_REAL(0.0),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_REAL(0.0),
    .is_rotating = false,
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/* Unknown/undefined frame - conservative defaults */
static const ss_frame_policy_t policy_unknown = {
    .frame = SS_FRAME_UNKNOWN,
    .name = "Unknown",
    .description = "Unknown or undefined reference frame",
    .position_precision = SS_REAL(0.001),
    .min_position = SS_REAL(0.0),
    .max_position = SS_REAL(1.0e12),
    .position_scale_factor = SS_REAL(1.0),
    .velocity_precision = SS_REAL(0.001),
    .min_velocity = SS_REAL(0.0),
    .max_velocity = SS_REAL(100000.0),
    .velocity_scale_factor = SS_REAL(1.0),
    .time_precision = SS_REAL(0.001),
    .max_time_interval = SS_REAL(86400.0 * 365.25),
    .angular_precision = SS_REAL(1.0e-6),
    .min_angle = SS_REAL(0.0),
    .max_angle = SS_TWO_PI,
    .is_rotating = false,
    .has_angular_coords = false,
    .has_singularities = false,
    .spatial_dims = 3
};

/*===========================================================================*/
/* Frame Policy Access Functions                                             */
/*===========================================================================*/

const ss_frame_policy_t* ss_get_frame_policy(ss_reference_frame_t frame) {
    switch (frame) {
        case SS_FRAME_CARTESIAN:
            return &policy_cartesian;
        case SS_FRAME_POLAR:
            return &policy_polar;
        case SS_FRAME_SPHERICAL:
            return &policy_spherical;
        case SS_FRAME_ECI:
            return &policy_eci;
        case SS_FRAME_ECEF:
            return &policy_ecef;
        case SS_FRAME_LLA:
            return &policy_lla;
        case SS_FRAME_ENU:
            return &policy_enu;
        case SS_FRAME_LUNAR:
            return &policy_lunar;
        case SS_FRAME_VOXEL:
            return &policy_voxel;
        case SS_FRAME_UNKNOWN:
        default:
            return &policy_unknown;
    }
}

const ss_frame_policy_t* ss_get_frame_policy_by_name(const char* name) {
    if (name == NULL) {
        return NULL;
    }

    /* Convert input name to lowercase for case-insensitive comparison */
    char lower_name[32];
    size_t i;
    for (i = 0; i < sizeof(lower_name) - 1 && name[i] != '\0'; i++) {
        lower_name[i] = (char)tolower((unsigned char)name[i]);
    }
    lower_name[i] = '\0';

    /* Check each frame type */
    static const ss_reference_frame_t all_frames[] = {
        SS_FRAME_CARTESIAN,
        SS_FRAME_POLAR,
        SS_FRAME_SPHERICAL,
        SS_FRAME_ECI,
        SS_FRAME_ECEF,
        SS_FRAME_LLA,
        SS_FRAME_ENU,
        SS_FRAME_LUNAR,
        SS_FRAME_VOXEL,
        SS_FRAME_UNKNOWN
    };

    for (i = 0; i < sizeof(all_frames) / sizeof(all_frames[0]); i++) {
        const ss_frame_policy_t* policy = ss_get_frame_policy(all_frames[i]);

        /* Convert policy name to lowercase for comparison */
        char policy_lower[32];
        size_t j;
        for (j = 0; j < sizeof(policy_lower) - 1 && policy->name[j] != '\0'; j++) {
            policy_lower[j] = (char)tolower((unsigned char)policy->name[j]);
        }
        policy_lower[j] = '\0';

        if (strcmp(lower_name, policy_lower) == 0) {
            return policy;
        }
    }

    return NULL;
}

const char* ss_frame_name(ss_reference_frame_t frame) {
    const ss_frame_policy_t* policy = ss_get_frame_policy(frame);
    return policy->name;
}

const char* ss_frame_description(ss_reference_frame_t frame) {
    const ss_frame_policy_t* policy = ss_get_frame_policy(frame);
    return policy->description;
}

/*===========================================================================*/
/* Precision Checking Functions                                              */
/*===========================================================================*/

bool ss_check_position_precision(const ss_frame_policy_t* policy) {
    if (policy == NULL) {
        return false;
    }

    /* Calculate the relative precision at maximum position */
    ss_real_t max_pos = policy->max_position;
    ss_real_t required_precision = policy->position_precision;

    /* The achievable precision is approximately max_value * epsilon */
    ss_real_t achievable_precision = max_pos * SS_REAL_EPSILON;

    /* We need achievable precision to be better than (smaller than) required */
    /* Add safety factor of 10x */
    return (achievable_precision * SS_REAL(10.0) < required_precision);
}

bool ss_check_velocity_precision(const ss_frame_policy_t* policy) {
    if (policy == NULL) {
        return false;
    }

    ss_real_t max_vel = policy->max_velocity;
    ss_real_t required_precision = policy->velocity_precision;
    ss_real_t achievable_precision = max_vel * SS_REAL_EPSILON;

    return (achievable_precision * SS_REAL(10.0) < required_precision);
}

bool ss_check_angular_precision(const ss_frame_policy_t* policy) {
    if (policy == NULL || !policy->has_angular_coords) {
        return true;  /* No angular precision required */
    }

    ss_real_t max_angle = policy->max_angle - policy->min_angle;
    ss_real_t required_precision = policy->angular_precision;
    ss_real_t achievable_precision = max_angle * SS_REAL_EPSILON;

    return (achievable_precision * SS_REAL(10.0) < required_precision);
}
