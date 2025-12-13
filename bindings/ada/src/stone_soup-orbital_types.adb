-------------------------------------------------------------------------------
-- Stone Soup Orbital Domain Types Implementation
-------------------------------------------------------------------------------

with Ada.Numerics.Long_Elementary_Functions;

package body Stone_Soup.Orbital_Types
  with SPARK_Mode => Off  -- Off due to use of Elementary_Functions
is
   use Ada.Numerics.Long_Elementary_Functions;

   Two_Pi : constant := 6.28318530717958647692;

   ---------------------------------------------------------------------------
   -- Classify orbit based on altitude
   ---------------------------------------------------------------------------
   function Classify_Orbit (Altitude : Altitude_Km) return Orbital_Regime is
   begin
      if Altitude < 2_000.0 then
         return LEO;
      elsif Altitude < 20_000.0 then
         return MEO;
      elsif Altitude < 35_700.0 then
         return HEO;
      elsif Altitude <= 35_900.0 then
         return GEO;
      else
         return Beyond_GEO;
      end if;
   end Classify_Orbit;

   ---------------------------------------------------------------------------
   -- Safe Arithmetic Operations
   ---------------------------------------------------------------------------

   function Safe_Add_Altitude
     (A1, A2 : Altitude_Km) return Altitude_Km is
      Sum : constant Long_Float := Long_Float (A1) + Long_Float (A2);
   begin
      if Sum > Max_Altitude_Km then
         return Altitude_Km (Max_Altitude_Km);
      else
         return Altitude_Km (Sum);
      end if;
   end Safe_Add_Altitude;

   function Altitude_To_Radius (Alt : Altitude_Km) return Geocentric_Radius_Km is
   begin
      return Geocentric_Radius_Km (Earth_Radius_Km + Long_Float (Alt));
   end Altitude_To_Radius;

   function Radius_To_Altitude (Radius : Geocentric_Radius_Km) return Altitude_Km is
   begin
      return Altitude_Km (Long_Float (Radius) - Earth_Radius_Km);
   end Radius_To_Altitude;

   ---------------------------------------------------------------------------
   -- Orbital Mechanics Functions
   ---------------------------------------------------------------------------

   function Orbital_Velocity (Radius : Geocentric_Radius_Km) return Velocity_KmPS is
      V : Long_Float;
   begin
      -- V = sqrt(mu/r) where mu = Earth_Mu (km^3/s^2)
      V := Sqrt (Earth_Mu / Long_Float (Radius));
      if V > Max_Orbital_Velocity then
         return Velocity_KmPS (Max_Orbital_Velocity);
      else
         return Velocity_KmPS (V);
      end if;
   end Orbital_Velocity;

   function Orbital_Period (SemiMajor : SemiMajor_Axis_Km) return Orbital_Period_S is
      A3 : Long_Float;
      T  : Long_Float;
   begin
      -- T = 2*Pi*sqrt(a^3/mu)
      A3 := Long_Float (SemiMajor) ** 3;
      T := Two_Pi * Sqrt (A3 / Earth_Mu);
      return Orbital_Period_S (T);
   end Orbital_Period;

end Stone_Soup.Orbital_Types;
