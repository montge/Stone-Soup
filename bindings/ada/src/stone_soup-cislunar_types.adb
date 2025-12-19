-------------------------------------------------------------------------------
-- Stone Soup Cislunar Domain Types Implementation
-------------------------------------------------------------------------------

with Ada.Numerics.Long_Elementary_Functions;

package body Stone_Soup.Cislunar_Types
  with SPARK_Mode => Off  -- Off due to use of Elementary_Functions
is
   use Ada.Numerics.Long_Elementary_Functions;

   ---------------------------------------------------------------------------
   -- Safe Arithmetic Operations
   ---------------------------------------------------------------------------

   function Safe_Add_Distance
     (D1, D2 : Earth_Distance_Km) return Earth_Distance_Km is
      Sum : constant Long_Float := Long_Float (D1) + Long_Float (D2);
   begin
      if Sum > Max_Distance_Km then
         return Earth_Distance_Km (Max_Distance_Km);
      else
         return Earth_Distance_Km (Sum);
      end if;
   end Safe_Add_Distance;

   ---------------------------------------------------------------------------
   -- Lagrange Point Functions
   ---------------------------------------------------------------------------

   function Distance_From_Lagrange
     (Point    : Lagrange_Point;
      Position : ECI_Position_Km) return Lagrange_Offset_Km
   is
      LP_X : Long_Float;
      LP_Y : constant Long_Float := 0.0;
      LP_Z : constant Long_Float := 0.0;
      DX, DY, DZ : Long_Float;
      Dist : Long_Float;
   begin
      -- Approximate Lagrange point positions along Earth-Moon line
      case Point is
         when L1 =>
            LP_X := L1_Distance_Km;
         when L2 =>
            LP_X := L2_Distance_Km;
         when L3 =>
            LP_X := -Earth_Moon_Distance_Km;
         when L4 =>
            -- L4 is 60 degrees ahead of Moon
            LP_X := Earth_Moon_Distance_Km * 0.5;
         when L5 =>
            -- L5 is 60 degrees behind Moon
            LP_X := Earth_Moon_Distance_Km * 0.5;
      end case;

      DX := Long_Float (Position) - LP_X;
      DY := 0.0 - LP_Y;  -- Simplified: assuming Position is radial distance
      DZ := 0.0 - LP_Z;

      Dist := Sqrt (DX**2 + DY**2 + DZ**2);

      -- Saturate to valid offset range
      if Dist > 100_000.0 then
         return Lagrange_Offset_Km (100_000.0);
      else
         return Lagrange_Offset_Km (Dist);
      end if;
   end Distance_From_Lagrange;

end Stone_Soup.Cislunar_Types;
