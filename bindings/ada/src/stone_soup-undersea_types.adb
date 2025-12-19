-------------------------------------------------------------------------------
-- Stone Soup Undersea Domain Types Implementation
-------------------------------------------------------------------------------

with Ada.Numerics.Long_Elementary_Functions;

package body Stone_Soup.Undersea_Types
  with SPARK_Mode => Off  -- Off due to use of Elementary_Functions
is
   use Ada.Numerics.Long_Elementary_Functions;

   Pi : constant := 3.14159265358979323846;
   Two_Pi : constant := 6.28318530717958647692;
   Deg_To_Rad : constant := Pi / 180.0;
   Rad_To_Deg : constant := 180.0 / Pi;

   ---------------------------------------------------------------------------
   -- Convert degrees to radians with range checking
   ---------------------------------------------------------------------------
   function To_Radians (Degrees : Bearing_Degrees) return Bearing_Radians is
   begin
      return Bearing_Radians (Long_Float (Degrees) * Deg_To_Rad);
   end To_Radians;

   ---------------------------------------------------------------------------
   -- Convert radians to degrees with range checking
   ---------------------------------------------------------------------------
   function To_Degrees (Radians : Bearing_Radians) return Bearing_Degrees is
   begin
      return Bearing_Degrees (Long_Float (Radians) * Rad_To_Deg);
   end To_Degrees;

   ---------------------------------------------------------------------------
   -- Normalize angle to [0, 2*Pi)
   ---------------------------------------------------------------------------
   function Normalize_Bearing (Angle : Long_Float) return Bearing_Radians is
      Result : Long_Float := Angle;
   begin
      -- Reduce to [-2*Pi, 2*Pi] range first
      if abs (Result) > Two_Pi then
         Result := Result - Long_Float'Floor (Result / Two_Pi) * Two_Pi;
      end if;

      -- Shift negative angles to positive
      while Result < 0.0 loop
         Result := Result + Two_Pi;
      end loop;

      -- Handle edge case where Result = 2*Pi exactly
      if Result >= Two_Pi then
         Result := Result - Two_Pi;
      end if;

      return Bearing_Radians (Result);
   end Normalize_Bearing;

   ---------------------------------------------------------------------------
   -- Safe Arithmetic Operations
   ---------------------------------------------------------------------------

   function Safe_Add_Depth
     (D1, D2 : Depth_Meters) return Depth_Meters is
      Sum : constant Long_Float := Long_Float (D1) + Long_Float (D2);
   begin
      if Sum > Max_Depth_Meters then
         return Depth_Meters (Max_Depth_Meters);
      else
         return Depth_Meters (Sum);
      end if;
   end Safe_Add_Depth;

   function Safe_Sub_Depth
     (D1, D2 : Depth_Meters) return Depth_Delta is
   begin
      return Depth_Delta (Long_Float (D1) - Long_Float (D2));
   end Safe_Sub_Depth;

   function Safe_Add_Range
     (R1, R2 : Range_Meters) return Range_Meters is
      Sum : constant Long_Float := Long_Float (R1) + Long_Float (R2);
   begin
      if Sum > Max_Range_Meters then
         return Range_Meters (Max_Range_Meters);
      else
         return Range_Meters (Sum);
      end if;
   end Safe_Add_Range;

   ---------------------------------------------------------------------------
   -- Coordinate Transformation Functions
   ---------------------------------------------------------------------------

   function Compute_Slant_Range
     (Horizontal_Range : Range_Meters;
      Depth_Diff       : Depth_Delta) return Slant_Range is
      H2 : constant Long_Float := Long_Float (Horizontal_Range) ** 2;
      D2 : constant Long_Float := Long_Float (Depth_Diff) ** 2;
   begin
      return Slant_Range (Sqrt (H2 + D2));
   end Compute_Slant_Range;

   function Compute_Travel_Time
     (Range_M     : Range_Meters;
      Sound_Speed : Sound_Speed_MPS) return Travel_Time_Seconds is
   begin
      return Travel_Time_Seconds (Long_Float (Range_M) / Long_Float (Sound_Speed));
   end Compute_Travel_Time;

end Stone_Soup.Undersea_Types;
