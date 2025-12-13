-------------------------------------------------------------------------------
-- Stone Soup Cislunar Domain Types
--
-- This package defines domain-specific numeric types for cislunar tracking
-- (Earth-Moon system) with explicit range constraints.
--
-- Numeric ranges based on Earth-Moon system:
-- - Earth-Moon distance: ~384,400 km (mean)
-- - Moon radius: 1,737 km
-- - Lunar orbit period: ~27.3 days
-- - Trans-lunar injection velocity: ~10.9 km/s
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

package Stone_Soup.Cislunar_Types
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Physical Constants
   ---------------------------------------------------------------------------

   -- Earth mean radius in kilometers
   Earth_Radius_Km : constant := 6_371.0;

   -- Moon mean radius in kilometers
   Moon_Radius_Km : constant := 1_737.4;

   -- Mean Earth-Moon distance in kilometers
   Earth_Moon_Distance_Km : constant := 384_400.0;

   -- Lunar perigee (closest approach) in km
   Lunar_Perigee_Km : constant := 356_500.0;

   -- Lunar apogee (farthest distance) in km
   Lunar_Apogee_Km : constant := 406_700.0;

   -- Earth gravitational parameter (km^3/s^2)
   Earth_Mu : constant := 398_600.4418;

   -- Moon gravitational parameter (km^3/s^2)
   Moon_Mu : constant := 4_902.8;

   -- Maximum tracking distance (beyond lunar orbit for L2, etc.)
   Max_Distance_Km : constant := 500_000.0;

   ---------------------------------------------------------------------------
   -- Position Types
   ---------------------------------------------------------------------------

   -- Distance from Earth center in km (0 to beyond Moon)
   subtype Earth_Distance_Km is Long_Float range 0.0 .. Max_Distance_Km;

   -- Distance from Moon center in km
   subtype Moon_Distance_Km is Long_Float range 0.0 .. Max_Distance_Km;

   -- Position component in Earth-centered frame (km)
   subtype ECI_Position_Km is Long_Float range -Max_Distance_Km .. Max_Distance_Km;

   -- Position component in Moon-centered frame (km)
   subtype MCI_Position_Km is Long_Float range -Max_Distance_Km .. Max_Distance_Km;

   -- Position in meters for high-precision near-Earth/near-Moon operations
   subtype Position_Meters is Long_Float range -5.0e11 .. 5.0e11;

   -- Altitude above Earth surface
   subtype Earth_Altitude_Km is Long_Float range 0.0 .. Max_Distance_Km;

   -- Altitude above Moon surface
   subtype Moon_Altitude_Km is Long_Float range 0.0 .. 100_000.0;

   ---------------------------------------------------------------------------
   -- Velocity Types
   ---------------------------------------------------------------------------

   -- Maximum velocity (Earth escape + margin: ~15 km/s)
   Max_Velocity_KmPS : constant := 15.0;

   -- Velocity magnitude in km/s
   subtype Velocity_Magnitude_KmPS is Long_Float range 0.0 .. Max_Velocity_KmPS;

   -- Velocity component in km/s
   subtype Velocity_Component_KmPS is Long_Float
     range -Max_Velocity_KmPS .. Max_Velocity_KmPS;

   -- Velocity in m/s for high-precision
   subtype Velocity_MPS is Long_Float range -15_000.0 .. 15_000.0;

   -- Trans-lunar injection delta-V (typically ~3.1 km/s from LEO)
   subtype TLI_Delta_V_MPS is Long_Float range 3_000.0 .. 3_500.0;

   -- Lunar orbit insertion delta-V
   subtype LOI_Delta_V_MPS is Long_Float range 800.0 .. 1_200.0;

   ---------------------------------------------------------------------------
   -- Lagrange Point Types
   ---------------------------------------------------------------------------

   -- Earth-Moon Lagrange points (L1, L2, L3, L4, L5)
   type Lagrange_Point is (L1, L2, L3, L4, L5);

   -- L1 distance from Earth (approximately 326,000 km)
   L1_Distance_Km : constant := 326_000.0;

   -- L2 distance from Earth (approximately 449,000 km)
   L2_Distance_Km : constant := 449_000.0;

   -- Distance from Lagrange point for halo/NRHO orbits
   subtype Lagrange_Offset_Km is Long_Float range 0.0 .. 100_000.0;

   ---------------------------------------------------------------------------
   -- Angular Types
   ---------------------------------------------------------------------------

   -- Phase angle in radians (0 to 2*Pi)
   subtype Phase_Angle_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   -- Lunar phase (0 = new moon, Pi = full moon)
   subtype Lunar_Phase_Rad is Long_Float range 0.0 .. 6.28318530717958647692;

   ---------------------------------------------------------------------------
   -- Time Types
   ---------------------------------------------------------------------------

   -- Trans-lunar coast time in seconds (typically 3-4 days)
   subtype TLC_Duration_S is Long_Float range 200_000.0 .. 400_000.0;

   -- Lunar orbital period in seconds (~2 hours for low lunar orbit)
   subtype Lunar_Orbit_Period_S is Long_Float range 6_000.0 .. 20_000.0;

   -- Mission elapsed time in seconds (allow for multi-year missions)
   subtype Mission_Time_S is Long_Float range 0.0 .. 1.0e9;

   ---------------------------------------------------------------------------
   -- Fixed-Point Types
   ---------------------------------------------------------------------------

   -- High-resolution distance (1 meter resolution over cislunar range)
   type Fixed_Distance_Km is delta 0.001 range -500_000.0 .. 500_000.0
     with Small => 0.001;

   -- High-resolution angle for trajectory correction
   type Fixed_Angle_Rad is delta 1.0e-12 range -7.0 .. 7.0
     with Small => 1.0e-12;

   ---------------------------------------------------------------------------
   -- Frame Types
   ---------------------------------------------------------------------------

   type Reference_Frame is
     (Earth_Centered_Inertial,    -- ECI (J2000)
      Earth_Centered_Fixed,       -- ECEF
      Moon_Centered_Inertial,     -- MCI
      Moon_Centered_Fixed,        -- MCMF (Moon-Centered Moon-Fixed)
      Earth_Moon_Rotating,        -- Synodic frame
      Sun_Earth_L2);              -- For deep space reference

   ---------------------------------------------------------------------------
   -- Validation Functions
   ---------------------------------------------------------------------------

   -- Check if within cislunar operational range
   function Is_Cislunar_Range (Dist : Long_Float) return Boolean is
     (Dist >= 0.0 and Dist <= Max_Distance_Km);

   -- Check if in lunar sphere of influence (~66,000 km from Moon)
   Lunar_SOI_Km : constant := 66_000.0;
   function Is_In_Lunar_SOI (Moon_Dist : Long_Float) return Boolean is
     (Moon_Dist >= 0.0 and Moon_Dist <= Lunar_SOI_Km);

   ---------------------------------------------------------------------------
   -- Safe Arithmetic Operations (Overflow-Checked)
   ---------------------------------------------------------------------------

   -- Safe distance addition with saturation
   function Safe_Add_Distance
     (D1, D2 : Earth_Distance_Km) return Earth_Distance_Km
     with Post => Safe_Add_Distance'Result <= Max_Distance_Km;

   -- Compute distance from Lagrange point
   function Distance_From_Lagrange
     (Point    : Lagrange_Point;
      Position : ECI_Position_Km) return Lagrange_Offset_Km
     with Post => Distance_From_Lagrange'Result >= 0.0;

   ---------------------------------------------------------------------------
   -- Coordinate Transformation Preconditions
   ---------------------------------------------------------------------------

   -- Check if ECI position is valid for cislunar operations
   function Valid_ECI_Position
     (X, Y, Z : ECI_Position_Km) return Boolean is
     (X >= -Max_Distance_Km and X <= Max_Distance_Km and
      Y >= -Max_Distance_Km and Y <= Max_Distance_Km and
      Z >= -Max_Distance_Km and Z <= Max_Distance_Km);

   -- Check if position is in Earth's sphere of influence
   -- (Roughly, distance < 929,000 km, but we use practical cislunar range)
   function Is_In_Earth_SOI (Earth_Dist : Earth_Distance_Km) return Boolean is
     (Earth_Dist >= 0.0 and Earth_Dist <= Max_Distance_Km);

   ---------------------------------------------------------------------------
   -- Trajectory Validation Functions
   ---------------------------------------------------------------------------

   -- Validate trans-lunar injection velocity
   function Is_Valid_TLI_DeltaV (DV : Long_Float) return Boolean is
     (DV >= 3_000.0 and DV <= 3_500.0);

   -- Validate lunar orbit insertion velocity
   function Is_Valid_LOI_DeltaV (DV : Long_Float) return Boolean is
     (DV >= 800.0 and DV <= 1_200.0);

end Stone_Soup.Cislunar_Types;
