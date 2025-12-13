-------------------------------------------------------------------------------
-- Stone Soup Undersea Domain Types
--
-- This package defines domain-specific numeric types for underwater tracking
-- applications with explicit range constraints for SPARK verification.
--
-- Numeric ranges are based on physical limits:
-- - Maximum ocean depth: ~11,000m (Mariana Trench)
-- - Maximum sonar range: ~100km (long-range passive sonar)
-- - Sound speed: 1400-1600 m/s (typical ocean conditions)
-- - Submarine speeds: up to ~40 knots (~20 m/s)
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

package Stone_Soup.Undersea_Types
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Depth Types
   ---------------------------------------------------------------------------

   -- Maximum ocean depth in meters (Challenger Deep = 10,935m, rounded up)
   Max_Depth_Meters : constant := 11_000.0;

   -- Depth below sea surface in meters (positive downward)
   subtype Depth_Meters is Long_Float range 0.0 .. Max_Depth_Meters;

   -- Height above seafloor in meters
   subtype Height_Above_Bottom is Long_Float range 0.0 .. Max_Depth_Meters;

   -- Relative depth change
   subtype Depth_Delta is Long_Float range -Max_Depth_Meters .. Max_Depth_Meters;

   ---------------------------------------------------------------------------
   -- Horizontal Range Types
   ---------------------------------------------------------------------------

   -- Maximum tracking range in meters (100km for long-range passive sonar)
   Max_Range_Meters : constant := 100_000.0;

   -- Horizontal range from sensor in meters
   subtype Range_Meters is Long_Float range 0.0 .. Max_Range_Meters;

   -- 3D slant range
   subtype Slant_Range is Long_Float range 0.0 .. 110_000.0;

   -- East/North position relative to reference in meters
   subtype Position_Meters is Long_Float range -Max_Range_Meters .. Max_Range_Meters;

   ---------------------------------------------------------------------------
   -- Sound Speed Types
   ---------------------------------------------------------------------------

   -- Sound speed range in m/s (typical ocean: 1450-1550, extremes: 1400-1600)
   Min_Sound_Speed : constant := 1_400.0;
   Max_Sound_Speed : constant := 1_600.0;

   subtype Sound_Speed_MPS is Long_Float range Min_Sound_Speed .. Max_Sound_Speed;

   -- Sound speed gradient (dC/dZ) in (m/s)/m, typically -0.017 to +0.05
   subtype Sound_Speed_Gradient is Long_Float range -0.1 .. 0.1;

   ---------------------------------------------------------------------------
   -- Velocity Types
   ---------------------------------------------------------------------------

   -- Maximum underwater vehicle speed in m/s (~50 knots for torpedoes)
   Max_Velocity : constant := 30.0;

   -- Velocity components in m/s
   subtype Velocity_MPS is Long_Float range -Max_Velocity .. Max_Velocity;

   -- Vertical velocity (positive = ascending)
   subtype Vertical_Velocity is Long_Float range -10.0 .. 10.0;

   -- Current speed in m/s (ocean currents typically < 3 m/s)
   subtype Current_Speed is Long_Float range 0.0 .. 5.0;

   ---------------------------------------------------------------------------
   -- Angular Types
   ---------------------------------------------------------------------------

   -- Bearing in radians (0 = North, clockwise)
   subtype Bearing_Radians is Long_Float range 0.0 .. 6.283185307179586;

   -- Bearing in degrees (0 = North, clockwise)
   subtype Bearing_Degrees is Long_Float range 0.0 .. 360.0;

   -- Elevation angle in radians (-Pi/2 to +Pi/2)
   subtype Elevation_Radians is Long_Float range -1.570796326794897 .. 1.570796326794897;

   -- Launch/arrival angle for ray tracing
   subtype Ray_Angle is Long_Float range -1.570796326794897 .. 1.570796326794897;

   ---------------------------------------------------------------------------
   -- Pressure Types
   ---------------------------------------------------------------------------

   -- Maximum pressure in bar at max depth (1 bar per ~10m + 1 atm)
   Max_Pressure : constant := 1_200.0;

   -- Hydrostatic pressure in bar
   subtype Pressure_Bar is Long_Float range 1.0 .. Max_Pressure;

   -- Pressure in dbar (decibar, ~1m depth equivalent)
   subtype Pressure_Dbar is Long_Float range 0.0 .. 12_000.0;

   ---------------------------------------------------------------------------
   -- Environmental Types
   ---------------------------------------------------------------------------

   -- Temperature in Celsius (-2 to 40 for ocean)
   subtype Temperature_Celsius is Long_Float range -2.0 .. 40.0;

   -- Salinity in PSU (practical salinity units, 0-45)
   subtype Salinity_PSU is Long_Float range 0.0 .. 45.0;

   -- Acoustic frequency in kHz (0.01 to 500 kHz for sonar)
   subtype Frequency_KHz is Long_Float range 0.01 .. 500.0;

   -- Transmission loss in dB (positive value)
   subtype Transmission_Loss_DB is Long_Float range 0.0 .. 200.0;

   -- Source level / received level in dB re 1 uPa
   subtype Sound_Level_DB is Long_Float range -50.0 .. 250.0;

   ---------------------------------------------------------------------------
   -- Time Types
   ---------------------------------------------------------------------------

   -- Acoustic travel time in seconds (max ~67s for 100km at 1500 m/s)
   subtype Travel_Time_Seconds is Long_Float range 0.0 .. 100.0;

   -- Time delay of arrival in seconds
   subtype TDOA_Seconds is Long_Float range -100.0 .. 100.0;

   ---------------------------------------------------------------------------
   -- Fixed-Point Types for Deterministic Arithmetic
   ---------------------------------------------------------------------------

   -- High-resolution angle (32-bit fixed point, ~0.00008 degree resolution)
   type Fixed_Angle is delta 0.000001 range -7.0 .. 7.0
     with Small => 0.000001;

   -- High-resolution depth (0.001m = 1mm resolution)
   type Fixed_Depth is delta 0.001 range 0.0 .. 11_000.0
     with Small => 0.001;

   -- High-resolution position (1mm resolution)
   type Fixed_Position is delta 0.001 range -100_000.0 .. 100_000.0
     with Small => 0.001;

   ---------------------------------------------------------------------------
   -- Conversion Functions
   ---------------------------------------------------------------------------

   -- Convert degrees to radians with range checking
   function To_Radians (Degrees : Bearing_Degrees) return Bearing_Radians
     with Post => To_Radians'Result >= 0.0 and To_Radians'Result <= 6.283185307179586;

   -- Convert radians to degrees with range checking
   function To_Degrees (Radians : Bearing_Radians) return Bearing_Degrees
     with Post => To_Degrees'Result >= 0.0 and To_Degrees'Result <= 360.0;

   -- Normalize angle to [0, 2*Pi)
   function Normalize_Bearing (Angle : Long_Float) return Bearing_Radians;

   ---------------------------------------------------------------------------
   -- Validation Functions
   ---------------------------------------------------------------------------

   -- Check if depth is valid (in ocean)
   function Is_Valid_Depth (D : Long_Float) return Boolean is
     (D >= 0.0 and D <= Max_Depth_Meters);

   -- Check if range is valid
   function Is_Valid_Range (R : Long_Float) return Boolean is
     (R >= 0.0 and R <= Max_Range_Meters);

   -- Check if sound speed is physically reasonable
   function Is_Valid_Sound_Speed (C : Long_Float) return Boolean is
     (C >= Min_Sound_Speed and C <= Max_Sound_Speed);

end Stone_Soup.Undersea_Types;
