-------------------------------------------------------------------------------
-- Stone Soup Automatic Unit Scaling Package
--
-- This package provides automatic unit scaling for large values with
-- precision preservation guarantees via SPARK contracts.
--
-- Supports automatic unit selection based on magnitude for:
-- - Distances: mm, m, km, Mm (megameters), Gm (gigameters)
-- - Velocities: mm/s, m/s, km/s
-- - Time: ns, us, ms, s, min, hr, days
--
-- All operations maintain precision by using Long_Float internally and
-- tracking scale factors explicitly.
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

package Stone_Soup.Unit_Scaling
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Unit Categories and Scale Factors
   ---------------------------------------------------------------------------

   -- Category of physical unit
   type Unit_Category is (Distance, Velocity, Time_Duration);

   -- Scale factor for each unit
   type Scale_Factor is
     (Nano,      -- 10^-9 (nanoseconds)
      Micro,     -- 10^-6 (microseconds)
      Milli,     -- 10^-3 (millimeters, milliseconds, mm/s)
      Base,      -- 10^0  (meters, seconds, m/s)
      Kilo,      -- 10^3  (kilometers, km/s)
      Mega,      -- 10^6  (megameters)
      Giga);     -- 10^9  (gigameters)

   -- Get numeric multiplier for scale factor
   function Scale_Multiplier (SF : Scale_Factor) return Long_Float
     with Post =>
       (case SF is
          when Nano  => Scale_Multiplier'Result = 1.0e-9,
          when Micro => Scale_Multiplier'Result = 1.0e-6,
          when Milli => Scale_Multiplier'Result = 1.0e-3,
          when Base  => Scale_Multiplier'Result = 1.0,
          when Kilo  => Scale_Multiplier'Result = 1.0e3,
          when Mega  => Scale_Multiplier'Result = 1.0e6,
          when Giga  => Scale_Multiplier'Result = 1.0e9);

   -- Get unit symbol string for scale factor and category
   function Unit_Symbol
     (SF  : Scale_Factor;
      Cat : Unit_Category) return String
     with Post =>
       (case Cat is
          when Distance =>
            (case SF is
               when Nano  => Unit_Symbol'Result = "nm",
               when Micro => Unit_Symbol'Result = "um",
               when Milli => Unit_Symbol'Result = "mm",
               when Base  => Unit_Symbol'Result = "m",
               when Kilo  => Unit_Symbol'Result = "km",
               when Mega  => Unit_Symbol'Result = "Mm",
               when Giga  => Unit_Symbol'Result = "Gm"),
          when Velocity =>
            (case SF is
               when Nano  => Unit_Symbol'Result = "nm/s",
               when Micro => Unit_Symbol'Result = "um/s",
               when Milli => Unit_Symbol'Result = "mm/s",
               when Base  => Unit_Symbol'Result = "m/s",
               when Kilo  => Unit_Symbol'Result = "km/s",
               when Mega  => Unit_Symbol'Result = "Mm/s",
               when Giga  => Unit_Symbol'Result = "Gm/s"),
          when Time_Duration =>
            (case SF is
               when Nano  => Unit_Symbol'Result = "ns",
               when Micro => Unit_Symbol'Result = "us",
               when Milli => Unit_Symbol'Result = "ms",
               when Base  => Unit_Symbol'Result = "s",
               when Kilo  => Unit_Symbol'Result = "ks",
               when Mega  => Unit_Symbol'Result = "Ms",
               when Giga  => Unit_Symbol'Result = "Gs"));

   -- Get conventional time unit name for convenience
   function Time_Unit_Name (SF : Scale_Factor) return String
     with Post =>
       (case SF is
          when Nano  => Time_Unit_Name'Result = "ns",
          when Micro => Time_Unit_Name'Result = "us",
          when Milli => Time_Unit_Name'Result = "ms",
          when Base  => Time_Unit_Name'Result = "s",
          when Kilo  => Time_Unit_Name'Result = "min",   -- 60 s (approx)
          when Mega  => Time_Unit_Name'Result = "hr",    -- 3600 s (approx)
          when Giga  => Time_Unit_Name'Result = "days"); -- 86400 s (approx)

   ---------------------------------------------------------------------------
   -- Scaled Value Type
   ---------------------------------------------------------------------------

   -- Value with explicit scale factor
   type Scaled_Value is record
      Value    : Long_Float;
      Scale    : Scale_Factor;
      Category : Unit_Category;
   end record;

   -- Maximum absolute value we support (below Long_Float'Last for safety)
   Max_Supported_Value : constant := 1.0e15;

   ---------------------------------------------------------------------------
   -- Automatic Scaling Functions
   ---------------------------------------------------------------------------

   -- Automatically choose appropriate scale for value
   -- Raw_Value is assumed to be in base units (meters, m/s, seconds)
   function Auto_Scale
     (Raw_Value : Long_Float;
      Unit_Type : Unit_Category) return Scaled_Value
     with Pre  => abs Raw_Value <= Max_Supported_Value,
          Post => Auto_Scale'Result.Category = Unit_Type and then
                  abs Auto_Scale'Result.Value >= 1.0 and then
                  abs Auto_Scale'Result.Value < 1000.0;

   -- Convert scaled value back to base units with precision preservation
   function To_Base_Units (SV : Scaled_Value) return Long_Float
     with Post =>
       (abs (To_Base_Units'Result -
             (SV.Value * Scale_Multiplier (SV.Scale))) < 1.0e-12 *
             abs (SV.Value * Scale_Multiplier (SV.Scale)) or else
        abs (SV.Value * Scale_Multiplier (SV.Scale)) < 1.0e-15);

   -- Convert from one scale to another with precision preservation
   function Convert_Scale
     (SV        : Scaled_Value;
      New_Scale : Scale_Factor) return Scaled_Value
     with Post => Convert_Scale'Result.Scale = New_Scale and then
                  Convert_Scale'Result.Category = SV.Category and then
                  (abs (Convert_Scale'Result.Value *
                        Scale_Multiplier (New_Scale) -
                        SV.Value * Scale_Multiplier (SV.Scale)) <
                   1.0e-12 * abs (SV.Value * Scale_Multiplier (SV.Scale)) or else
                   abs (SV.Value * Scale_Multiplier (SV.Scale)) < 1.0e-15);

   ---------------------------------------------------------------------------
   -- Direct Construction Functions
   ---------------------------------------------------------------------------

   -- Create a scaled value with explicit scale (no auto-scaling)
   function Create_Scaled
     (Value     : Long_Float;
      Scale     : Scale_Factor;
      Unit_Type : Unit_Category) return Scaled_Value
     with Post => Create_Scaled'Result.Value = Value and then
                  Create_Scaled'Result.Scale = Scale and then
                  Create_Scaled'Result.Category = Unit_Type;

   -- Create from raw value in specific units
   function From_Millimeters (Value : Long_Float) return Scaled_Value
     with Post => From_Millimeters'Result.Category = Distance;

   function From_Meters (Value : Long_Float) return Scaled_Value
     with Post => From_Meters'Result.Category = Distance;

   function From_Kilometers (Value : Long_Float) return Scaled_Value
     with Post => From_Kilometers'Result.Category = Distance;

   function From_Meters_Per_Second (Value : Long_Float) return Scaled_Value
     with Post => From_Meters_Per_Second'Result.Category = Velocity;

   function From_Kilometers_Per_Second (Value : Long_Float) return Scaled_Value
     with Post => From_Kilometers_Per_Second'Result.Category = Velocity;

   function From_Seconds (Value : Long_Float) return Scaled_Value
     with Post => From_Seconds'Result.Category = Time_Duration;

   function From_Minutes (Value : Long_Float) return Scaled_Value
     with Post => From_Minutes'Result.Category = Time_Duration;

   function From_Hours (Value : Long_Float) return Scaled_Value
     with Post => From_Hours'Result.Category = Time_Duration;

   function From_Days (Value : Long_Float) return Scaled_Value
     with Post => From_Days'Result.Category = Time_Duration;

   ---------------------------------------------------------------------------
   -- Extraction Functions
   ---------------------------------------------------------------------------

   -- Extract value in specific units
   function To_Millimeters (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Distance;

   function To_Meters (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Distance;

   function To_Kilometers (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Distance;

   function To_Meters_Per_Second (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Velocity;

   function To_Kilometers_Per_Second (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Velocity;

   function To_Seconds (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Time_Duration;

   function To_Minutes (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Time_Duration;

   function To_Hours (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Time_Duration;

   function To_Days (SV : Scaled_Value) return Long_Float
     with Pre => SV.Category = Time_Duration;

   ---------------------------------------------------------------------------
   -- Arithmetic Operations with Scale Preservation
   ---------------------------------------------------------------------------

   -- Add two scaled values (converts to common scale)
   function "+" (Left, Right : Scaled_Value) return Scaled_Value
     with Pre  => Left.Category = Right.Category,
          Post => "+"'Result.Category = Left.Category;

   -- Subtract two scaled values (converts to common scale)
   function "-" (Left, Right : Scaled_Value) return Scaled_Value
     with Pre  => Left.Category = Right.Category,
          Post => "-"'Result.Category = Left.Category;

   -- Multiply scaled value by dimensionless scalar
   function "*" (Factor : Long_Float; SV : Scaled_Value) return Scaled_Value
     with Post => "*"'Result.Category = SV.Category and then
                  "*"'Result.Scale = SV.Scale;

   -- Divide scaled value by dimensionless scalar
   function "/" (SV : Scaled_Value; Divisor : Long_Float) return Scaled_Value
     with Pre  => abs Divisor > 1.0e-15,
          Post => "/"'Result.Category = SV.Category and then
                  "/"'Result.Scale = SV.Scale;

   ---------------------------------------------------------------------------
   -- Domain-Specific Convenience Functions
   ---------------------------------------------------------------------------

   -- Choose optimal scale for undersea tracking (0-11 km depth)
   function Scale_For_Undersea (Depth_Meters : Long_Float) return Scaled_Value
     with Pre  => Depth_Meters >= 0.0 and Depth_Meters <= 11_000.0,
          Post => Scale_For_Undersea'Result.Category = Distance;

   -- Choose optimal scale for orbital tracking (500-36000 km)
   function Scale_For_Orbital (Altitude_Meters : Long_Float) return Scaled_Value
     with Pre  => Altitude_Meters >= 0.0,
          Post => Scale_For_Orbital'Result.Category = Distance;

   -- Choose optimal scale for cislunar tracking (~400,000 km)
   function Scale_For_Cislunar (Distance_Meters : Long_Float) return Scaled_Value
     with Pre  => Distance_Meters >= 0.0,
          Post => Scale_For_Cislunar'Result.Category = Distance;

   -- Choose optimal scale for interplanetary tracking (~10^12 m)
   function Scale_For_Interplanetary
     (Distance_Meters : Long_Float) return Scaled_Value
     with Pre  => Distance_Meters >= 0.0 and
                  Distance_Meters <= Max_Supported_Value,
          Post => Scale_For_Interplanetary'Result.Category = Distance;

end Stone_Soup.Unit_Scaling;
