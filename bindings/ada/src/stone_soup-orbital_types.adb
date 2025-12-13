-------------------------------------------------------------------------------
-- Stone Soup Orbital Domain Types Implementation
-------------------------------------------------------------------------------

package body Stone_Soup.Orbital_Types
  with SPARK_Mode => On
is

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

end Stone_Soup.Orbital_Types;
