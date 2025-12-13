-------------------------------------------------------------------------------
-- Stone Soup Numeric Types Boundary Value Tests
--
-- Tests domain-specific numeric types at their boundaries:
-- - Minimum and maximum valid values
-- - Edge case operations
-- - Safe arithmetic functions
-- - Coordinate validation functions
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Test_Caller;
with AUnit.Test_Suites;
with AUnit.Assertions;
with Stone_Soup.Undersea_Types;
with Stone_Soup.Orbital_Types;
with Stone_Soup.Cislunar_Types;

package body Test_Numeric_Types is

   use AUnit.Assertions;
   use Stone_Soup.Undersea_Types;
   use Stone_Soup.Orbital_Types;
   use Stone_Soup.Cislunar_Types;

   Epsilon : constant Long_Float := 1.0e-10;

   ---------------------------------------------------------------------------
   -- Undersea Type Tests
   ---------------------------------------------------------------------------

   procedure Test_Undersea_Depth_Boundaries
     (T : in out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      Min_Depth : Depth_Meters := 0.0;
      Max_Depth : Depth_Meters := Max_Depth_Meters;
      Mid_Depth : Depth_Meters := 5500.0;
   begin
      -- Test boundary values are valid
      Assert (Is_Valid_Depth (Long_Float (Min_Depth)),
              "Minimum depth (0) should be valid");
      Assert (Is_Valid_Depth (Long_Float (Max_Depth)),
              "Maximum depth (11000) should be valid");
      Assert (Is_Valid_Depth (Long_Float (Mid_Depth)),
              "Middle depth (5500) should be valid");

      -- Test invalid values
      Assert (not Is_Valid_Depth (-1.0),
              "Negative depth should be invalid");
      Assert (not Is_Valid_Depth (11001.0),
              "Depth > 11000 should be invalid");
   end Test_Undersea_Depth_Boundaries;

   procedure Test_Undersea_Sound_Speed_Boundaries
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      Min_Speed : Sound_Speed_MPS := Min_Sound_Speed;
      Max_Speed : Sound_Speed_MPS := Max_Sound_Speed;
   begin
      Assert (Is_Valid_Sound_Speed (Long_Float (Min_Speed)),
              "Minimum sound speed (1400) should be valid");
      Assert (Is_Valid_Sound_Speed (Long_Float (Max_Speed)),
              "Maximum sound speed (1600) should be valid");
      Assert (Is_Valid_Sound_Speed (1500.0),
              "Nominal sound speed (1500) should be valid");

      Assert (not Is_Valid_Sound_Speed (1399.0),
              "Sound speed < 1400 should be invalid");
      Assert (not Is_Valid_Sound_Speed (1601.0),
              "Sound speed > 1600 should be invalid");
   end Test_Undersea_Sound_Speed_Boundaries;

   procedure Test_Undersea_Safe_Add_Depth
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      D1 : Depth_Meters := 5000.0;
      D2 : Depth_Meters := 4000.0;
      D3 : Depth_Meters := 8000.0;
      Result : Depth_Meters;
   begin
      -- Normal addition
      Result := Safe_Add_Depth (D1, D2);
      Assert (abs (Long_Float (Result) - 9000.0) < Epsilon,
              "5000 + 4000 should equal 9000");

      -- Saturation (would overflow to 13000, saturates to 11000)
      Result := Safe_Add_Depth (D1, D3);
      Assert (Long_Float (Result) <= Max_Depth_Meters,
              "Sum should saturate at max depth");
   end Test_Undersea_Safe_Add_Depth;

   procedure Test_Undersea_Compute_Slant_Range
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      H_Range : Range_Meters := 3000.0;
      Depth_D : Depth_Delta := 4000.0;
      Result : Slant_Range;
   begin
      -- 3-4-5 triangle
      Result := Compute_Slant_Range (H_Range, Depth_D);
      Assert (abs (Long_Float (Result) - 5000.0) < 1.0,
              "Slant range of 3000, 4000 triangle should be 5000");
   end Test_Undersea_Compute_Slant_Range;

   procedure Test_Undersea_Bearing_Conversion
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      Degrees : Bearing_Degrees := 180.0;
      Radians : Bearing_Radians;
   begin
      Radians := To_Radians (Degrees);
      Assert (abs (Long_Float (Radians) - 3.14159265) < 0.0001,
              "180 degrees should convert to Pi radians");

      Degrees := To_Degrees (Radians);
      Assert (abs (Long_Float (Degrees) - 180.0) < 0.01,
              "Pi radians should convert to 180 degrees");
   end Test_Undersea_Bearing_Conversion;

   ---------------------------------------------------------------------------
   -- Orbital Type Tests
   ---------------------------------------------------------------------------

   procedure Test_Orbital_Altitude_Classification
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
   begin
      Assert (Classify_Orbit (400.0) = LEO,
              "400 km altitude should be LEO (ISS)");
      Assert (Classify_Orbit (1500.0) = LEO,
              "1500 km altitude should be LEO");
      Assert (Classify_Orbit (20200.0) = MEO,
              "20200 km altitude should be MEO (GPS)");
      Assert (Classify_Orbit (35786.0) = GEO,
              "35786 km altitude should be GEO");
      Assert (Classify_Orbit (40000.0) = Beyond_GEO,
              "40000 km altitude should be Beyond_GEO");
   end Test_Orbital_Altitude_Classification;

   procedure Test_Orbital_Altitude_Radius_Conversion
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      Alt : Altitude_Km := 400.0;
      Radius : Geocentric_Radius_Km;
   begin
      Radius := Altitude_To_Radius (Alt);
      Assert (abs (Long_Float (Radius) - (6371.0 + 400.0)) < 0.01,
              "400 km altitude should give radius of ~6771 km");

      Alt := Radius_To_Altitude (Radius);
      Assert (abs (Long_Float (Alt) - 400.0) < 0.01,
              "Round-trip conversion should preserve altitude");
   end Test_Orbital_Altitude_Radius_Conversion;

   procedure Test_Orbital_Velocity_Calculation
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      Radius : Geocentric_Radius_Km := 6771.0;  -- ISS altitude
      Vel : Velocity_KmPS;
   begin
      Vel := Orbital_Velocity (Radius);
      -- ISS orbital velocity is about 7.66 km/s
      Assert (Long_Float (Vel) > 7.0 and Long_Float (Vel) < 8.0,
              "LEO orbital velocity should be 7-8 km/s");
   end Test_Orbital_Velocity_Calculation;

   procedure Test_Orbital_Period_Calculation
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      SMA : SemiMajor_Axis_Km := 6771.0;  -- ISS-like orbit
      Period : Orbital_Period_S;
   begin
      Period := Orbital_Period (SMA);
      -- ISS orbital period is about 92.68 minutes = 5560 seconds
      Assert (Long_Float (Period) > 5000.0 and Long_Float (Period) < 6000.0,
              "LEO orbital period should be ~5500 seconds");
   end Test_Orbital_Period_Calculation;

   procedure Test_Orbital_Eccentricity_Validation
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
   begin
      Assert (Is_Valid_Eccentricity (0.0),
              "Circular orbit (e=0) should be valid");
      Assert (Is_Valid_Eccentricity (0.5),
              "Elliptical orbit (e=0.5) should be valid");
      Assert (Is_Valid_Eccentricity (0.99),
              "Highly elliptical orbit (e=0.99) should be valid");
      Assert (not Is_Valid_Eccentricity (1.0),
              "Parabolic orbit (e=1) should be invalid");
      Assert (not Is_Valid_Eccentricity (-0.1),
              "Negative eccentricity should be invalid");
   end Test_Orbital_Eccentricity_Validation;

   ---------------------------------------------------------------------------
   -- Cislunar Type Tests
   ---------------------------------------------------------------------------

   procedure Test_Cislunar_Distance_Boundaries
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
   begin
      Assert (Is_Cislunar_Range (0.0),
              "0 km should be valid cislunar range");
      Assert (Is_Cislunar_Range (384400.0),
              "Earth-Moon distance should be valid");
      Assert (Is_Cislunar_Range (500000.0),
              "Maximum range (500000 km) should be valid");
      Assert (not Is_Cislunar_Range (-1.0),
              "Negative distance should be invalid");
      Assert (not Is_Cislunar_Range (600000.0),
              "Distance > max should be invalid");
   end Test_Cislunar_Distance_Boundaries;

   procedure Test_Cislunar_Lunar_SOI
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
   begin
      Assert (Is_In_Lunar_SOI (0.0),
              "Moon surface should be in lunar SOI");
      Assert (Is_In_Lunar_SOI (66000.0),
              "SOI boundary should be in lunar SOI");
      Assert (not Is_In_Lunar_SOI (70000.0),
              "70000 km from Moon should be outside SOI");
   end Test_Cislunar_Lunar_SOI;

   procedure Test_Cislunar_TLI_Validation
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
   begin
      Assert (Is_Valid_TLI_DeltaV (3100.0),
              "Nominal TLI delta-V (3100 m/s) should be valid");
      Assert (Is_Valid_TLI_DeltaV (3000.0),
              "Min TLI delta-V (3000 m/s) should be valid");
      Assert (Is_Valid_TLI_DeltaV (3500.0),
              "Max TLI delta-V (3500 m/s) should be valid");
      Assert (not Is_Valid_TLI_DeltaV (2900.0),
              "Below min TLI should be invalid");
      Assert (not Is_Valid_TLI_DeltaV (3600.0),
              "Above max TLI should be invalid");
   end Test_Cislunar_TLI_Validation;

   procedure Test_Cislunar_Safe_Add_Distance
     (T : in Out AUnit.Test_Cases.Test_Case'Class)
   is
      pragma Unreferenced (T);
      D1 : Earth_Distance_Km := 200000.0;
      D2 : Earth_Distance_Km := 200000.0;
      D3 : Earth_Distance_Km := 400000.0;
      Result : Earth_Distance_Km;
   begin
      -- Normal addition
      Result := Safe_Add_Distance (D1, D2);
      Assert (abs (Long_Float (Result) - 400000.0) < 1.0,
              "200000 + 200000 should equal 400000");

      -- Saturation
      Result := Safe_Add_Distance (D1, D3);
      Assert (Long_Float (Result) <= Max_Distance_Km,
              "Sum should saturate at max distance");
   end Test_Cislunar_Safe_Add_Distance;

   ---------------------------------------------------------------------------
   -- Test Suite Setup
   ---------------------------------------------------------------------------

   package Undersea_Caller is new AUnit.Test_Caller
     (Numeric_Test_Case);
   package Orbital_Caller is new AUnit.Test_Caller
     (Numeric_Test_Case);
   package Cislunar_Caller is new AUnit.Test_Caller
     (Numeric_Test_Case);

   function Suite return AUnit.Test_Suites.Access_Test_Suite is
      Result : constant AUnit.Test_Suites.Access_Test_Suite :=
         AUnit.Test_Suites.New_Suite;
   begin
      -- Undersea tests
      Result.Add_Test
        (Undersea_Caller.Create
           ("Test undersea depth boundaries",
            Test_Undersea_Depth_Boundaries'Access));
      Result.Add_Test
        (Undersea_Caller.Create
           ("Test undersea sound speed boundaries",
            Test_Undersea_Sound_Speed_Boundaries'Access));
      Result.Add_Test
        (Undersea_Caller.Create
           ("Test undersea safe add depth",
            Test_Undersea_Safe_Add_Depth'Access));
      Result.Add_Test
        (Undersea_Caller.Create
           ("Test undersea compute slant range",
            Test_Undersea_Compute_Slant_Range'Access));
      Result.Add_Test
        (Undersea_Caller.Create
           ("Test undersea bearing conversion",
            Test_Undersea_Bearing_Conversion'Access));

      -- Orbital tests
      Result.Add_Test
        (Orbital_Caller.Create
           ("Test orbital altitude classification",
            Test_Orbital_Altitude_Classification'Access));
      Result.Add_Test
        (Orbital_Caller.Create
           ("Test orbital altitude radius conversion",
            Test_Orbital_Altitude_Radius_Conversion'Access));
      Result.Add_Test
        (Orbital_Caller.Create
           ("Test orbital velocity calculation",
            Test_Orbital_Velocity_Calculation'Access));
      Result.Add_Test
        (Orbital_Caller.Create
           ("Test orbital period calculation",
            Test_Orbital_Period_Calculation'Access));
      Result.Add_Test
        (Orbital_Caller.Create
           ("Test orbital eccentricity validation",
            Test_Orbital_Eccentricity_Validation'Access));

      -- Cislunar tests
      Result.Add_Test
        (Cislunar_Caller.Create
           ("Test cislunar distance boundaries",
            Test_Cislunar_Distance_Boundaries'Access));
      Result.Add_Test
        (Cislunar_Caller.Create
           ("Test cislunar lunar SOI",
            Test_Cislunar_Lunar_SOI'Access));
      Result.Add_Test
        (Cislunar_Caller.Create
           ("Test cislunar TLI validation",
            Test_Cislunar_TLI_Validation'Access));
      Result.Add_Test
        (Cislunar_Caller.Create
           ("Test cislunar safe add distance",
            Test_Cislunar_Safe_Add_Distance'Access));

      return Result;
   end Suite;

end Test_Numeric_Types;
