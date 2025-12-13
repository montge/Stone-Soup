# Design: Safety-Critical Numeric Precision

## Context

Safety-critical tracking systems require explicit numeric bounds to prevent overflow,
underflow, and precision loss. This is mandatory for DO-178C certification where all
numeric operations must be proven safe. Different tracking domains (undersea to
interplanetary) span many orders of magnitude, requiring domain-specific type designs.

## Goals / Non-Goals

**Goals:**
- Define explicit numeric ranges for each tracking domain
- Provide Ada types with range constraints for SPARK proofs
- Support configurable precision in C library
- Enable compile-time verification of numeric safety

**Non-Goals:**
- Arbitrary precision arithmetic (out of scope)
- GPU/SIMD optimization (separate concern)
- Real-time scheduling (separate concern)

## Numeric Range Analysis

### Domain-Specific Ranges

| Domain | Position Range | Velocity Range | Acceleration Range | Precision |
|--------|---------------|----------------|-------------------|-----------|
| Undersea | 0-11 km depth, 40 km range | 0-40 m/s (subs), 1500 m/s (sound) | 0-5 m/s² | 0.01 m |
| Terrestrial | 0-13,000 km | 0-1000 m/s (aircraft), 8000 m/s (ballistic) | 0-100 m/s² | 0.1 m |
| LEO | 200-2000 km altitude | 7-8 km/s orbital | 0-10 m/s² (maneuver) | 1 m |
| MEO/GEO | 2000-42000 km | 2-4 km/s orbital | 0-1 m/s² | 10 m |
| Cislunar | 0-400,000 km | 0-11 km/s | 0-10 m/s² | 100 m |
| Interplanetary | 0-10^12 m | 0-60 km/s | 0-0.1 m/s² | 1 km |

### Coordinate Transformation Precision

| Transformation | Input Range | Output Range | Precision Required |
|---------------|------------|--------------|-------------------|
| ECEF ↔ LLA | ±10^7 m | ±180°, ±90° | 10^-9 degrees |
| ENU local | ±10^6 m | ±10^6 m | 0.001 m |
| Orbital elements | 10^6-10^8 m | 0-2π rad | 10^-12 rad |
| Bearing/range | ±10^12 m | 0-2π, 0-10^12 m | domain-dependent |

### Numeric Type Requirements

**Undersea Domain:**
- Position: 32-bit float sufficient (10 km / 0.01 m = 10^6 values)
- Velocity: 32-bit float sufficient
- Sound speed: 1400-1600 m/s, need 0.01 m/s precision
- Pressure: 0-1200 bar, 0.01 bar precision

**LEO/MEO Domain:**
- Position: 64-bit float required (10^8 m / 1 m = 10^8 values)
- Velocity: 64-bit float (10 km/s / 0.001 m/s = 10^7 values)
- Time: 64-bit float for nanosecond precision over days

**Cislunar/Interplanetary:**
- Position: 64-bit float or extended precision
- Time: Extended precision for long mission durations
- Consider: Hierarchical coordinate frames to maintain precision

## Decisions

### Decision 1: Domain-Specific Type Packages

Create separate Ada packages for each domain with appropriate range constraints:
- `Stone_Soup.Undersea_Types` - ranges for underwater tracking
- `Stone_Soup.Orbital_Types` - ranges for LEO/MEO/GEO
- `Stone_Soup.Cislunar_Types` - ranges for Earth-Moon operations
- `Stone_Soup.Interplanetary_Types` - ranges for deep space

**Rationale:** Allows compile-time selection of appropriate precision and enables
SPARK proofs specific to each domain.

### Decision 2: Fixed-Point Types for Deterministic Operations

Use Ada fixed-point types for critical calculations requiring deterministic behavior:
- Angles: fixed-point with 2^-32 resolution (0.00000008 degrees)
- Small deltas: fixed-point for integration steps
- Time: fixed-point for real-time scheduling

**Rationale:** Fixed-point arithmetic is deterministic and avoids floating-point
non-associativity issues critical for safety proofs.

### Decision 3: C Library Compile-Time Configuration

Add preprocessor configuration for C library precision:
```c
#ifdef STONESOUP_PRECISION_SINGLE
  typedef float ss_real_t;
#elif STONESOUP_PRECISION_EXTENDED
  typedef long double ss_real_t;
#else
  typedef double ss_real_t;  // default
#endif
```

**Rationale:** Allows embedded systems to use 32-bit floats while scientific
applications can use extended precision.

### Decision 4: Overflow Checking Strategy

- Ada: Use SPARK preconditions to prove no overflow at compile time
- C: Use checked macros in debug builds, unchecked in release
- All: Document numeric assumptions in API contracts

## Risks / Trade-offs

**Risk 1: Performance impact from range checking**
- Mitigation: SPARK proves at compile time, no runtime overhead

**Risk 2: Type conversion overhead between domains**
- Mitigation: Explicit conversion functions with overflow checking

**Risk 3: Backward compatibility with existing code**
- Mitigation: New types are additive, existing Long_Float code continues to work

## Ada Type Specifications

### Undersea Domain Types

```ada
package Stone_Soup.Undersea_Types is

   -- Depth: 0 to 11,000 meters (Mariana Trench depth)
   subtype Depth_Meters is Long_Float range 0.0 .. 11_000.0;

   -- Horizontal range: 0 to 100 km (extended sonar range)
   subtype Range_Meters is Long_Float range 0.0 .. 100_000.0;

   -- Sound speed: 1400 to 1600 m/s
   subtype Sound_Speed is Long_Float range 1400.0 .. 1600.0;

   -- Velocity: -50 to 50 m/s (submarine speeds)
   subtype Velocity_MPS is Long_Float range -50.0 .. 50.0;

   -- Bearing: 0 to 2*Pi radians
   subtype Bearing_Radians is Long_Float range 0.0 .. 6.283185307;

   -- Pressure: 0 to 1200 bar (11km depth)
   subtype Pressure_Bar is Long_Float range 0.0 .. 1200.0;

   -- Temperature: -2 to 40 Celsius (ocean range)
   subtype Temperature_C is Long_Float range -2.0 .. 40.0;

   -- Salinity: 0 to 45 PSU
   subtype Salinity_PSU is Long_Float range 0.0 .. 45.0;

end Stone_Soup.Undersea_Types;
```

### Orbital Domain Types

```ada
package Stone_Soup.Orbital_Types is

   -- Altitude: 0 to 42,000 km (GEO)
   subtype Altitude_Km is Long_Float range 0.0 .. 42_000.0;

   -- Position: Earth radius + altitude
   subtype Radius_Km is Long_Float range 6_371.0 .. 50_000.0;

   -- Orbital velocity: 0 to 12 km/s
   subtype Velocity_KmPS is Long_Float range 0.0 .. 12.0;

   -- Semi-major axis: 6500 to 50000 km
   subtype SemiMajor_Km is Long_Float range 6_500.0 .. 50_000.0;

   -- Eccentricity: 0 to 1 (open orbits not supported)
   subtype Eccentricity is Long_Float range 0.0 .. 1.0;

   -- Inclination: 0 to Pi radians
   subtype Inclination_Rad is Long_Float range 0.0 .. 3.141592654;

   -- RAAN, argument of perigee, true anomaly: 0 to 2*Pi
   subtype Angle_Rad is Long_Float range 0.0 .. 6.283185307;

end Stone_Soup.Orbital_Types;
```

### Cislunar Domain Types

```ada
package Stone_Soup.Cislunar_Types is

   -- Distance: 0 to 500,000 km (beyond lunar orbit)
   subtype Distance_Km is Long_Float range 0.0 .. 500_000.0;

   -- Velocity: 0 to 15 km/s (trans-lunar injection)
   subtype Velocity_KmPS is Long_Float range 0.0 .. 15.0;

   -- Position in ECEF or Moon-centered: ±500,000 km
   subtype Position_Km is Long_Float range -500_000.0 .. 500_000.0;

end Stone_Soup.Cislunar_Types;
```

## C Library Configuration

### Precision Configuration Header

```c
// stonesoup/precision.h
#ifndef STONESOUP_PRECISION_H
#define STONESOUP_PRECISION_H

// Select precision at compile time
#if defined(STONESOUP_PRECISION_SINGLE)
    typedef float ss_real_t;
    #define SS_REAL_MIN FLT_MIN
    #define SS_REAL_MAX FLT_MAX
    #define SS_REAL_EPSILON FLT_EPSILON
#elif defined(STONESOUP_PRECISION_EXTENDED)
    typedef long double ss_real_t;
    #define SS_REAL_MIN LDBL_MIN
    #define SS_REAL_MAX LDBL_MAX
    #define SS_REAL_EPSILON LDBL_EPSILON
#else
    typedef double ss_real_t;
    #define SS_REAL_MIN DBL_MIN
    #define SS_REAL_MAX DBL_MAX
    #define SS_REAL_EPSILON DBL_EPSILON
#endif

// Domain-specific limits
#define SS_UNDERSEA_DEPTH_MAX 11000.0
#define SS_UNDERSEA_RANGE_MAX 100000.0
#define SS_ORBITAL_ALTITUDE_MAX 42000000.0
#define SS_CISLUNAR_DISTANCE_MAX 500000000.0

// Overflow checking (debug builds only)
#ifdef STONESOUP_DEBUG
    #define SS_CHECK_RANGE(val, min, max) \
        do { if ((val) < (min) || (val) > (max)) { \
            fprintf(stderr, "Range violation: %g not in [%g, %g]\n", \
                    (double)(val), (double)(min), (double)(max)); \
            abort(); \
        }} while(0)
#else
    #define SS_CHECK_RANGE(val, min, max) ((void)0)
#endif

#endif // STONESOUP_PRECISION_H
```

## Migration Plan

1. Add new type packages (non-breaking, additive)
2. Update internal calculations to use domain-specific types
3. Add SPARK contracts gradually
4. Run GNATprove on each module
5. Update C library to use configurable precision type
6. Add overflow checking macros

## Open Questions

1. Should we support automatic precision escalation (e.g., switching from 32-bit
   to 64-bit when approaching range limits)?
2. How to handle cross-domain transfers (e.g., aircraft to orbit)?
3. Should fixed-point types be exposed in public API or internal only?
