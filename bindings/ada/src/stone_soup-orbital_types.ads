-------------------------------------------------------------------------------
-- Stone Soup Orbital Domain Types
--
-- This package defines domain-specific numeric types for orbital tracking
-- applications (LEO, MEO, GEO) with explicit range constraints.
--
-- Numeric ranges based on orbital mechanics:
-- - LEO altitude: 200-2000 km
-- - GEO altitude: ~35,786 km (42,164 km from Earth center)
-- - Orbital velocity: 3-8 km/s
-- - Earth radius: 6,371 km mean
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

package Stone_Soup.Orbital_Types
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Physical Constants
   ---------------------------------------------------------------------------

   -- Earth mean radius in kilometers
   Earth_Radius_Km : constant := 6_371.0;

   -- Earth gravitational parameter (km^3/s^2)
   Earth_Mu : constant := 398_600.4418;

   -- GEO altitude in km
   GEO_Altitude_Nominal : constant := 35_786.0;

   -- Maximum supported altitude (beyond GEO)
   Max_Altitude_Km : constant := 50_000.0;

   ---------------------------------------------------------------------------
   -- Position Types
   ---------------------------------------------------------------------------

   -- Altitude above Earth surface in kilometers
   subtype Altitude_Km is Long_Float range 0.0 .. Max_Altitude_Km;

   -- LEO altitude range (200-2000 km)
   subtype LEO_Altitude_Km is Long_Float range 200.0 .. 2_000.0;

   -- MEO altitude range (2000-35786 km)
   subtype MEO_Altitude_Km is Long_Float range 2_000.0 .. GEO_Altitude_Nominal;

   -- GEO altitude (approximately 35786 km, allow some margin)
   subtype GEO_Altitude_Range is Long_Float range 35_700.0 .. 35_900.0;

   -- Radius from Earth center in kilometers
   subtype Geocentric_Radius_Km is Long_Float
     range Earth_Radius_Km .. (Earth_Radius_Km + Max_Altitude_Km);

   -- ECEF position component in kilometers (±Earth radius + max altitude)
   Max_ECEF_Km : constant := Earth_Radius_Km + Max_Altitude_Km;
   subtype ECEF_Position_Km is Long_Float range -Max_ECEF_Km .. Max_ECEF_Km;

   -- Position in meters for high-precision tracking
   subtype Position_Meters is Long_Float range -60_000_000.0 .. 60_000_000.0;

   ---------------------------------------------------------------------------
   -- Velocity Types
   ---------------------------------------------------------------------------

   -- Maximum orbital velocity (escape velocity at Earth surface ~11.2 km/s)
   Max_Orbital_Velocity : constant := 12.0;

   -- Orbital velocity in km/s
   subtype Velocity_KmPS is Long_Float range 0.0 .. Max_Orbital_Velocity;

   -- Velocity component in km/s (can be negative for direction)
   subtype Velocity_Component_KmPS is Long_Float
     range -Max_Orbital_Velocity .. Max_Orbital_Velocity;

   -- Velocity in m/s for high-precision
   subtype Velocity_MPS is Long_Float range -12_000.0 .. 12_000.0;

   -- Delta-V for maneuvers in m/s (typical spacecraft: 0-4000 m/s budget)
   subtype Delta_V_MPS is Long_Float range 0.0 .. 10_000.0;

   ---------------------------------------------------------------------------
   -- Keplerian Element Types
   ---------------------------------------------------------------------------

   -- Semi-major axis in kilometers
   Min_SemiMajor : constant := Earth_Radius_Km + 100.0;  -- Min viable orbit
   Max_SemiMajor : constant := Earth_Radius_Km + Max_Altitude_Km;
   subtype SemiMajor_Axis_Km is Long_Float range Min_SemiMajor .. Max_SemiMajor;

   -- Eccentricity (0 = circular, <1 = elliptical, 1 = parabolic)
   subtype Eccentricity is Long_Float range 0.0 .. 0.99;

   -- Inclination in radians (0 to Pi)
   subtype Inclination_Rad is Long_Float range 0.0 .. 3.14159265358979323846;

   -- RAAN (Right Ascension of Ascending Node) in radians (0 to 2*Pi)
   subtype RAAN_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- Argument of perigee in radians (0 to 2*Pi)
   subtype Arg_Perigee_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- True anomaly in radians (0 to 2*Pi)
   subtype True_Anomaly_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- Mean anomaly in radians (0 to 2*Pi)
   subtype Mean_Anomaly_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- Eccentric anomaly in radians (0 to 2*Pi)
   subtype Eccentric_Anomaly_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   ---------------------------------------------------------------------------
   -- Angular Types
   ---------------------------------------------------------------------------

   -- General angle in radians (0 to 2*Pi)
   subtype Angle_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- Signed angle in radians (-Pi to +Pi)
   subtype Signed_Angle_Rad is Long_Float
     range -3.14159265358979323846 .. 3.14159265358979323846;

   -- Latitude in radians (-Pi/2 to +Pi/2)
   subtype Latitude_Rad is Long_Float
     range -1.57079632679489661923 .. 1.57079632679489661923;

   -- Longitude in radians (-Pi to +Pi)
   subtype Longitude_Rad is Long_Float
     range -3.14159265358979323846 .. 3.14159265358979323846;

   -- Latitude in degrees (-90 to +90)
   subtype Latitude_Deg is Long_Float range -90.0 .. 90.0;

   -- Longitude in degrees (-180 to +180)
   subtype Longitude_Deg is Long_Float range -180.0 .. 180.0;

   ---------------------------------------------------------------------------
   -- Time Types
   ---------------------------------------------------------------------------

   -- Orbital period in seconds (ISS ~92 min, GEO ~24 hr)
   subtype Orbital_Period_S is Long_Float range 5_000.0 .. 100_000.0;

   -- Mean motion in radians per second
   subtype Mean_Motion_RadPS is Long_Float range 0.0 .. 0.002;

   -- Julian date (useful for epoch specification)
   subtype Julian_Date is Long_Float range 2_400_000.0 .. 2_500_000.0;

   -- Time since epoch in seconds (allow for long propagation)
   subtype Time_Since_Epoch_S is Long_Float range -1.0e9 .. 1.0e9;

   ---------------------------------------------------------------------------
   -- Fixed-Point Types for Deterministic Orbit Propagation
   ---------------------------------------------------------------------------

   -- High-resolution angle (~0.0002 arcsec resolution for orbit determination)
   type Fixed_Angle_Rad is delta 1.0e-12 range -7.0 .. 7.0
     with Small => 1.0e-12;

   -- High-resolution position in km (1 meter resolution)
   type Fixed_Position_Km is delta 0.001 range -60_000.0 .. 60_000.0
     with Small => 0.001;

   ---------------------------------------------------------------------------
   -- Orbital Regime Classification
   ---------------------------------------------------------------------------

   type Orbital_Regime is (LEO, MEO, HEO, GEO, Beyond_GEO);

   -- Classify orbit based on altitude
   function Classify_Orbit (Altitude : Altitude_Km) return Orbital_Regime
     with Post =>
       (if Altitude < 2_000.0 then Classify_Orbit'Result = LEO
        elsif Altitude < 35_700.0 then Classify_Orbit'Result in MEO | HEO
        elsif Altitude <= 35_900.0 then Classify_Orbit'Result = GEO
        else Classify_Orbit'Result = Beyond_GEO);

   ---------------------------------------------------------------------------
   -- Validation Functions
   ---------------------------------------------------------------------------

   -- Check if altitude is viable for orbit
   function Is_Orbital_Altitude (Alt : Long_Float) return Boolean is
     (Alt >= 100.0 and Alt <= Max_Altitude_Km);

   -- Check if eccentricity is valid for closed orbit
   function Is_Valid_Eccentricity (E : Long_Float) return Boolean is
     (E >= 0.0 and E < 1.0);

   ---------------------------------------------------------------------------
   -- Safe Arithmetic Operations (Overflow-Checked)
   ---------------------------------------------------------------------------

   -- Safe altitude addition with saturation
   function Safe_Add_Altitude
     (A1, A2 : Altitude_Km) return Altitude_Km
     with Post => Safe_Add_Altitude'Result <= Max_Altitude_Km;

   -- Compute orbital radius from altitude
   function Altitude_To_Radius (Alt : Altitude_Km) return Geocentric_Radius_Km
     with Post => Altitude_To_Radius'Result >= Earth_Radius_Km;

   -- Compute altitude from orbital radius
   function Radius_To_Altitude (Radius : Geocentric_Radius_Km) return Altitude_Km
     with Pre  => Radius >= Earth_Radius_Km,
          Post => Radius_To_Altitude'Result >= 0.0;

   ---------------------------------------------------------------------------
   -- Orbital Mechanics Functions with SPARK Contracts
   ---------------------------------------------------------------------------

   -- Compute orbital velocity at given radius
   -- V = sqrt(mu/r) where mu = Earth_Mu (km^3/s^2)
   function Orbital_Velocity (Radius : Geocentric_Radius_Km) return Velocity_KmPS
     with Pre  => Radius >= Earth_Radius_Km,
          Post => Orbital_Velocity'Result >= 0.0 and
                  Orbital_Velocity'Result <= Max_Orbital_Velocity;

   -- Compute orbital period at given semi-major axis
   -- T = 2*Pi*sqrt(a^3/mu)
   function Orbital_Period (SemiMajor : SemiMajor_Axis_Km) return Orbital_Period_S
     with Pre  => SemiMajor >= Min_SemiMajor,
          Post => Orbital_Period'Result >= 5_000.0;

   ---------------------------------------------------------------------------
   -- Coordinate Transformation Preconditions
   ---------------------------------------------------------------------------

   -- Check if ECEF position is valid
   function Valid_ECEF_Position
     (X, Y, Z : ECEF_Position_Km) return Boolean is
     (X >= -Max_ECEF_Km and X <= Max_ECEF_Km and
      Y >= -Max_ECEF_Km and Y <= Max_ECEF_Km and
      Z >= -Max_ECEF_Km and Z <= Max_ECEF_Km);

   -- Check if Keplerian elements are valid
   function Valid_Keplerian
     (A : SemiMajor_Axis_Km;
      E : Eccentricity;
      I : Inclination_Rad) return Boolean is
     (A >= Min_SemiMajor and A <= Max_SemiMajor and
      E >= 0.0 and E < 1.0 and
      I >= 0.0 and I <= 3.14159265358979323846);

end Stone_Soup.Orbital_Types;
