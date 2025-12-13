-------------------------------------------------------------------------------
-- Stone Soup Unit Scaling Tests Implementation
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stone_Soup.Unit_Scaling; use Stone_Soup.Unit_Scaling;

package body Test_Unit_Scaling is

   Test_Count   : Natural := 0;
   Failed_Count : Natural := 0;

   procedure Assert
     (Condition : Boolean;
      Message   : String)
   is
   begin
      Test_Count := Test_Count + 1;
      if not Condition then
         Put_Line ("  [FAIL] " & Message);
         Failed_Count := Failed_Count + 1;
      else
         Put_Line ("  [PASS] " & Message);
      end if;
   end Assert;

   procedure Test_Scale_Multipliers is
   begin
      Put_Line ("Testing Scale Multipliers...");
      Assert (Scale_Multiplier (Nano) = 1.0e-9, "Nano scale");
      Assert (Scale_Multiplier (Micro) = 1.0e-6, "Micro scale");
      Assert (Scale_Multiplier (Milli) = 1.0e-3, "Milli scale");
      Assert (Scale_Multiplier (Base) = 1.0, "Base scale");
      Assert (Scale_Multiplier (Kilo) = 1.0e3, "Kilo scale");
      Assert (Scale_Multiplier (Mega) = 1.0e6, "Mega scale");
      Assert (Scale_Multiplier (Giga) = 1.0e9, "Giga scale");
   end Test_Scale_Multipliers;

   procedure Test_Auto_Scaling_Distance is
      SV : Scaled_Value;
      Epsilon : constant Long_Float := 1.0e-10;
   begin
      Put_Line ("Testing Auto Scaling for Distance...");

      -- Test millimeters (small values)
      SV := Auto_Scale (0.001, Distance);
      Assert (SV.Scale = Milli and abs (SV.Value - 1.0) < Epsilon,
              "Auto scale 1mm -> 1.0 mm");

      -- Test meters (medium values)
      SV := Auto_Scale (10.0, Distance);
      Assert (SV.Scale = Base and abs (SV.Value - 10.0) < Epsilon,
              "Auto scale 10m -> 10.0 m");

      -- Test kilometers (large values)
      SV := Auto_Scale (5000.0, Distance);
      Assert (SV.Scale = Kilo and abs (SV.Value - 5.0) < Epsilon,
              "Auto scale 5000m -> 5.0 km");

      -- Test megameters (very large values)
      SV := Auto_Scale (2_000_000.0, Distance);
      Assert (SV.Scale = Mega and abs (SV.Value - 2.0) < Epsilon,
              "Auto scale 2,000,000m -> 2.0 Mm");
   end Test_Auto_Scaling_Distance;

   procedure Test_To_Base_Units is
      SV : Scaled_Value;
      Base_Value : Long_Float;
      Epsilon : constant Long_Float := 1.0e-10;
   begin
      Put_Line ("Testing To_Base_Units conversion...");

      SV := Create_Scaled (5.0, Kilo, Distance);
      Base_Value := To_Base_Units (SV);
      Assert (abs (Base_Value - 5000.0) < Epsilon,
              "5.0 km -> 5000.0 m");

      SV := Create_Scaled (100.0, Milli, Distance);
      Base_Value := To_Base_Units (SV);
      Assert (abs (Base_Value - 0.1) < Epsilon,
              "100.0 mm -> 0.1 m");

      SV := Create_Scaled (2.0, Mega, Distance);
      Base_Value := To_Base_Units (SV);
      Assert (abs (Base_Value - 2_000_000.0) < Epsilon,
              "2.0 Mm -> 2,000,000.0 m");
   end Test_To_Base_Units;

   procedure Test_Scale_Conversion is
      SV1, SV2 : Scaled_Value;
      Epsilon : constant Long_Float := 1.0e-10;
   begin
      Put_Line ("Testing Scale Conversion...");

      -- Convert 5 km to meters
      SV1 := Create_Scaled (5.0, Kilo, Distance);
      SV2 := Convert_Scale (SV1, Base);
      Assert (SV2.Scale = Base and abs (SV2.Value - 5000.0) < Epsilon,
              "Convert 5 km to 5000 m");

      -- Convert 1000 mm to meters
      SV1 := Create_Scaled (1000.0, Milli, Distance);
      SV2 := Convert_Scale (SV1, Base);
      Assert (SV2.Scale = Base and abs (SV2.Value - 1.0) < Epsilon,
              "Convert 1000 mm to 1 m");

      -- Convert 0.5 km to mm
      SV1 := Create_Scaled (0.5, Kilo, Distance);
      SV2 := Convert_Scale (SV1, Milli);
      Assert (SV2.Scale = Milli and abs (SV2.Value - 500_000.0) < Epsilon,
              "Convert 0.5 km to 500,000 mm");
   end Test_Scale_Conversion;

   procedure Test_Convenience_Functions is
      SV : Scaled_Value;
      Value : Long_Float;
      Epsilon : constant Long_Float := 1.0e-9;
   begin
      Put_Line ("Testing Convenience Functions...");

      -- Distance functions
      SV := From_Kilometers (5.0);
      Value := To_Meters (SV);
      Assert (abs (Value - 5000.0) < Epsilon,
              "From_Kilometers/To_Meters: 5 km = 5000 m");

      SV := From_Meters (100.0);
      Value := To_Kilometers (SV);
      Assert (abs (Value - 0.1) < Epsilon,
              "From_Meters/To_Kilometers: 100 m = 0.1 km");

      -- Velocity functions
      SV := From_Kilometers_Per_Second (3.0);
      Value := To_Meters_Per_Second (SV);
      Assert (abs (Value - 3000.0) < Epsilon,
              "From_Kilometers_Per_Second: 3 km/s = 3000 m/s");

      -- Time functions
      SV := From_Hours (2.0);
      Value := To_Seconds (SV);
      Assert (abs (Value - 7200.0) < Epsilon,
              "From_Hours/To_Seconds: 2 hr = 7200 s");

      SV := From_Minutes (30.0);
      Value := To_Seconds (SV);
      Assert (abs (Value - 1800.0) < Epsilon,
              "From_Minutes/To_Seconds: 30 min = 1800 s");
   end Test_Convenience_Functions;

   procedure Test_Arithmetic_Operations is
      SV1, SV2, SV3 : Scaled_Value;
      Epsilon : constant Long_Float := 1.0e-9;
   begin
      Put_Line ("Testing Arithmetic Operations...");

      -- Addition
      SV1 := Create_Scaled (5.0, Kilo, Distance);
      SV2 := Create_Scaled (2000.0, Base, Distance);
      SV3 := SV1 + SV2;
      Assert (abs (To_Base_Units (SV3) - 7000.0) < Epsilon,
              "Addition: 5 km + 2000 m = 7000 m");

      -- Subtraction
      SV1 := Create_Scaled (10.0, Kilo, Distance);
      SV2 := Create_Scaled (3000.0, Base, Distance);
      SV3 := SV1 - SV2;
      Assert (abs (To_Base_Units (SV3) - 7000.0) < Epsilon,
              "Subtraction: 10 km - 3000 m = 7000 m");

      -- Multiplication by scalar
      SV1 := Create_Scaled (5.0, Kilo, Distance);
      SV2 := 2.0 * SV1;
      Assert (abs (To_Base_Units (SV2) - 10000.0) < Epsilon,
              "Multiplication: 2 * 5 km = 10000 m");

      -- Division by scalar
      SV1 := Create_Scaled (10.0, Kilo, Distance);
      SV2 := SV1 / 2.0;
      Assert (abs (To_Base_Units (SV2) - 5000.0) < Epsilon,
              "Division: 10 km / 2 = 5000 m");
   end Test_Arithmetic_Operations;

   procedure Test_Domain_Specific_Scaling is
      SV : Scaled_Value;
      Epsilon : constant Long_Float := 1.0e-9;
   begin
      Put_Line ("Testing Domain-Specific Scaling...");

      -- Undersea (should use meters for medium depths)
      SV := Scale_For_Undersea (500.0);
      Assert (SV.Scale = Base and abs (SV.Value - 500.0) < Epsilon,
              "Undersea: 500m depth uses meters");

      -- Orbital (should use kilometers)
      SV := Scale_For_Orbital (500_000.0);
      Assert (SV.Scale = Kilo and abs (SV.Value - 500.0) < Epsilon,
              "Orbital: 500 km uses kilometers");

      -- Cislunar (should use megameters for large distances)
      SV := Scale_For_Cislunar (400_000_000.0);
      Assert (SV.Scale = Mega and abs (SV.Value - 400.0) < Epsilon,
              "Cislunar: 400,000 km uses megameters");

      -- Interplanetary (should use gigameters for very large distances)
      SV := Scale_For_Interplanetary (2_000_000_000.0);
      Assert (SV.Scale = Giga and abs (SV.Value - 2.0) < Epsilon,
              "Interplanetary: 2,000,000 km uses gigameters");
   end Test_Domain_Specific_Scaling;

   procedure Test_Precision_Preservation is
      SV1, SV2 : Scaled_Value;
      Original, Recovered : Long_Float;
      Epsilon : constant Long_Float := 1.0e-12;
   begin
      Put_Line ("Testing Precision Preservation...");

      -- Round-trip conversion should preserve value
      Original := 12345.678;
      SV1 := Auto_Scale (Original, Distance);
      Recovered := To_Base_Units (SV1);
      Assert (abs (Recovered - Original) / abs Original < Epsilon,
              "Round-trip preserves precision");

      -- Scale conversion should preserve value
      SV1 := Create_Scaled (5.5, Kilo, Distance);
      SV2 := Convert_Scale (SV1, Base);
      Assert (abs (To_Base_Units (SV1) - To_Base_Units (SV2)) < Epsilon,
              "Scale conversion preserves value");
   end Test_Precision_Preservation;

   procedure Run_All_Tests is
   begin
      Put_Line ("========================================");
      Put_Line ("Stone Soup Unit Scaling Test Suite");
      Put_Line ("========================================");
      New_Line;

      Test_Scale_Multipliers;
      New_Line;

      Test_Auto_Scaling_Distance;
      New_Line;

      Test_To_Base_Units;
      New_Line;

      Test_Scale_Conversion;
      New_Line;

      Test_Convenience_Functions;
      New_Line;

      Test_Arithmetic_Operations;
      New_Line;

      Test_Domain_Specific_Scaling;
      New_Line;

      Test_Precision_Preservation;
      New_Line;

      Put_Line ("========================================");
      Put_Line ("Total Tests: " & Natural'Image (Test_Count));
      Put_Line ("Failed:      " & Natural'Image (Failed_Count));
      Put_Line ("Passed:      " & Natural'Image (Test_Count - Failed_Count));

      if Failed_Count = 0 then
         Put_Line ("Result: ALL TESTS PASSED");
      else
         Put_Line ("Result: SOME TESTS FAILED");
      end if;
      Put_Line ("========================================");
   end Run_All_Tests;

end Test_Unit_Scaling;
