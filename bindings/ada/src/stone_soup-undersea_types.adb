-------------------------------------------------------------------------------
-- Stone Soup Undersea Domain Types Implementation
-------------------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;

package body Stone_Soup.Undersea_Types
  with SPARK_Mode => On
is
   use Ada.Numerics.Elementary_Functions;

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

end Stone_Soup.Undersea_Types;
