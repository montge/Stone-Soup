-------------------------------------------------------------------------------
-- Stone Soup Cross-Domain Coordinate Transfer Implementation
-------------------------------------------------------------------------------

with Ada.Numerics.Long_Elementary_Functions;

package body Stone_Soup.Domain_Transfer
  with SPARK_Mode => Off  -- Off due to use of Elementary_Functions
is

   use Ada.Numerics.Long_Elementary_Functions;
   use Stone_Soup.Undersea_Types;
   use Stone_Soup.Orbital_Types;
   use Stone_Soup.Cislunar_Types;

   -- Constants
   Earth_Radius_Meters : constant := 6_371_000.0;  -- meters

   ---------------------------------------------------------------------------
   -- Helper Functions
   ---------------------------------------------------------------------------

   -- Compute Euclidean distance in 3D
   function Euclidean_Distance (X1, Y1, Z1, X2, Y2, Z2 : Long_Float) return Long_Float is
      DX : constant Long_Float := X2 - X1;
      DY : constant Long_Float := Y2 - Y1;
      DZ : constant Long_Float := Z2 - Z1;
   begin
      return Sqrt (DX * DX + DY * DY + DZ * DZ);
   end Euclidean_Distance;

   -- Estimate precision loss for coordinate transformation
   function Estimate_Precision_Loss
     (From_Domain : Tracking_Domain;
      To_Domain   : Tracking_Domain;
      Base_Precision : Precision_Meters) return Precision_Meters is
      Loss_Factor : Long_Float := 1.0;
   begin
      -- Simple model: precision degrades based on domain transition
      case From_Domain is
         when Undersea =>
            case To_Domain is
               when Surface => Loss_Factor := 1.1;    -- Minimal loss
               when LEO     => Loss_Factor := 100.0;  -- Large scale change
               when others  => Loss_Factor := 1000.0;
            end case;

         when Surface =>
            case To_Domain is
               when Undersea => Loss_Factor := 1.1;
               when LEO      => Loss_Factor := 10.0;
               when MEO | GEO => Loss_Factor := 50.0;
               when others   => Loss_Factor := 100.0;
            end case;

         when LEO =>
            case To_Domain is
               when Surface => Loss_Factor := 10.0;
               when MEO | GEO => Loss_Factor := 2.0;
               when Cislunar => Loss_Factor := 10.0;
               when others   => Loss_Factor := 5.0;
            end case;

         when MEO =>
            case To_Domain is
               when LEO => Loss_Factor := 2.0;
               when GEO => Loss_Factor := 1.5;
               when Cislunar => Loss_Factor := 5.0;
               when others => Loss_Factor := 3.0;
            end case;

         when GEO =>
            case To_Domain is
               when LEO | MEO => Loss_Factor := 2.0;
               when Cislunar => Loss_Factor := 3.0;
               when Interplanetary => Loss_Factor := 10.0;
               when others => Loss_Factor := 5.0;
            end case;

         when Cislunar =>
            case To_Domain is
               when GEO => Loss_Factor := 3.0;
               when Interplanetary => Loss_Factor := 5.0;
               when others => Loss_Factor := 10.0;
            end case;

         when Interplanetary =>
            Loss_Factor := 10.0;  -- High loss coming back to local frames
      end case;

      return Precision_Meters'Min (Base_Precision * Loss_Factor, 1.0e12);
   end Estimate_Precision_Loss;

   ---------------------------------------------------------------------------
   -- Public Functions
   ---------------------------------------------------------------------------

   function Create_Domain_Position
     (Domain    : Tracking_Domain;
      X, Y, Z   : Long_Float;
      Precision : Precision_Meters := 0.0) return Domain_Position is
   begin
      return (Domain           => Domain,
              X                => X,
              Y                => Y,
              Z                => Z,
              Precision => Precision);
   end Create_Domain_Position;

   function Get_Precision_Threshold
     (Domain : Tracking_Domain) return Long_Float is
   begin
      case Domain is
         when Undersea       => return Undersea_Precision_Threshold;
         when Surface        => return Surface_Precision_Threshold;
         when LEO            => return LEO_Precision_Threshold;
         when MEO            => return MEO_Precision_Threshold;
         when GEO            => return GEO_Precision_Threshold;
         when Cislunar       => return Cislunar_Precision_Threshold;
         when Interplanetary => return Interplanetary_Precision_Threshold;
      end case;
   end Get_Precision_Threshold;

   function Check_Precision
     (Pos : Domain_Position) return Precision_Check_Result is
      Threshold : constant Long_Float := Get_Precision_Threshold (Pos.Domain);
      Loss_Factor : Long_Float;
   begin
      if Pos.Precision <= Threshold then
         return (Status           => OK,
                 Current_Meters   => Pos.Precision,
                 Threshold_Meters => Threshold,
                 Loss_Factor      => 0.0);
      else
         Loss_Factor := Long_Float (Pos.Precision) / Threshold;

         if Loss_Factor >= Max_Precision_Loss_Factor then
            return (Status           => Critical,
                    Current_Meters   => Pos.Precision,
                    Threshold_Meters => Threshold,
                    Loss_Factor      => Loss_Factor);
         else
            return (Status           => Warning,
                    Current_Meters   => Pos.Precision,
                    Threshold_Meters => Threshold,
                    Loss_Factor      => Loss_Factor);
         end if;
      end if;
   end Check_Precision;

   function Can_Transfer
     (From_Domain : Tracking_Domain;
      To_Domain   : Tracking_Domain) return Boolean is
   begin
      -- All transfers are supported (some may be indirect)
      -- Same domain is always allowed
      if From_Domain = To_Domain then
         return True;
      end if;

      -- Adjacent domain transfers
      case From_Domain is
         when Undersea =>
            return To_Domain in Surface .. Interplanetary;
         when Surface =>
            return To_Domain in Undersea | LEO | MEO | GEO | Cislunar | Interplanetary;
         when LEO =>
            return To_Domain in Surface | MEO | GEO | Cislunar | Interplanetary;
         when MEO =>
            return To_Domain in LEO | GEO | Cislunar | Interplanetary;
         when GEO =>
            return To_Domain in LEO | MEO | Cislunar | Interplanetary;
         when Cislunar =>
            return To_Domain in GEO | Interplanetary;
         when Interplanetary =>
            return True;  -- Can transfer to any domain
      end case;
   end Can_Transfer;

   function Is_Valid_Position (Pos : Domain_Position) return Boolean is
   begin
      -- Basic sanity checks on position coordinates
      case Pos.Domain is
         when Undersea =>
            -- Z should be negative depth (0 to -11000 m)
            return Pos.Z >= -11_000.0 and Pos.Z <= 0.0;

         when Surface =>
            -- Z should be near zero (± 10 km for terrain)
            return Pos.Z >= -10_000.0 and Pos.Z <= 10_000.0;

         when LEO =>
            -- Altitude should be 200-2000 km above Earth surface
            declare
               Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
               Altitude : constant Long_Float := Radius - Earth_Radius_Meters;
            begin
               return Altitude >= 200_000.0 and Altitude <= 2_000_000.0;
            end;

         when MEO =>
            -- Altitude 2000-35786 km
            declare
               Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
               Altitude : constant Long_Float := Radius - Earth_Radius_Meters;
            begin
               return Altitude >= 2_000_000.0 and Altitude <= 35_786_000.0;
            end;

         when GEO =>
            -- Altitude ~35786 km
            declare
               Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
               Altitude : constant Long_Float := Radius - Earth_Radius_Meters;
            begin
               return Altitude >= 35_700_000.0 and Altitude <= 35_900_000.0;
            end;

         when Cislunar =>
            -- Distance up to 500,000 km from Earth
            declare
               Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
            begin
               return Radius <= 500_000_000.0;
            end;

         when Interplanetary =>
            -- Very large distances, just check not NaN/Inf
            return abs (Pos.X) < 1.0e15 and abs (Pos.Y) < 1.0e15 and abs (Pos.Z) < 1.0e15;
      end case;
   end Is_Valid_Position;

   ---------------------------------------------------------------------------
   -- Domain Transfer Functions
   ---------------------------------------------------------------------------

   function Undersea_To_Surface
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Undersea, Surface, Pos.Precision);
   begin
      -- Convert depth (negative Z) to surface altitude (Z near 0)
      return (Domain           => Surface,
              X                => Pos.X,
              Y                => Pos.Y,
              Z                => 0.0,  -- Surface level
              Precision => New_Precision);
   end Undersea_To_Surface;

   function Surface_To_LEO
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Surface, LEO, Pos.Precision);

      -- Convert surface position to ECEF-like coordinates at LEO altitude
      -- Assume nominal LEO altitude of 500 km
      LEO_Altitude : constant Long_Float := 500_000.0;  -- meters
      Radius : constant Long_Float := Earth_Radius_Meters + LEO_Altitude;
      Scale : constant Long_Float := Radius / Earth_Radius_Meters;
   begin
      return (Domain           => LEO,
              X                => Pos.X * Scale,
              Y                => Pos.Y * Scale,
              Z                => Pos.Z * Scale,
              Precision => New_Precision);
   end Surface_To_LEO;

   function LEO_To_GEO
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (LEO, GEO, Pos.Precision);

      -- Scale from LEO to GEO radius
      LEO_Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
      GEO_Radius : constant Long_Float := Earth_Radius_Meters + 35_786_000.0;
      Scale : Long_Float;
   begin
      if LEO_Radius > 0.0 then
         Scale := GEO_Radius / LEO_Radius;
      else
         Scale := 1.0;
      end if;

      return (Domain           => GEO,
              X                => Pos.X * Scale,
              Y                => Pos.Y * Scale,
              Z                => Pos.Z * Scale,
              Precision => New_Precision);
   end LEO_To_GEO;

   function GEO_To_Cislunar
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (GEO, Cislunar, Pos.Precision);
   begin
      -- GEO coordinates can be used directly in cislunar frame (Earth-centered)
      return (Domain           => Cislunar,
              X                => Pos.X,
              Y                => Pos.Y,
              Z                => Pos.Z,
              Precision => New_Precision);
   end GEO_To_Cislunar;

   function Cislunar_To_Interplanetary
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Cislunar, Interplanetary, Pos.Precision);
   begin
      -- Convert from Earth-centered to heliocentric frame
      -- For simplicity, just use same coordinates (would need Earth position for real conversion)
      return (Domain           => Interplanetary,
              X                => Pos.X,
              Y                => Pos.Y,
              Z                => Pos.Z,
              Precision => New_Precision);
   end Cislunar_To_Interplanetary;

   ---------------------------------------------------------------------------
   -- Reverse Transfer Functions
   ---------------------------------------------------------------------------

   function Surface_To_Undersea
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Surface, Undersea, Pos.Precision);
   begin
      -- Convert surface altitude to depth (negative Z)
      -- Assume at surface, so depth is 0
      return (Domain           => Undersea,
              X                => Pos.X,
              Y                => Pos.Y,
              Z                => 0.0,  -- At surface
              Precision => New_Precision);
   end Surface_To_Undersea;

   function LEO_To_Surface
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (LEO, Surface, Pos.Precision);

      -- Project down to Earth surface
      Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
      Scale : Long_Float;
   begin
      if Radius > 0.0 then
         Scale := Earth_Radius_Meters / Radius;
      else
         Scale := 1.0;
      end if;

      return (Domain           => Surface,
              X                => Pos.X * Scale,
              Y                => Pos.Y * Scale,
              Z                => 0.0,  -- Project to surface
              Precision => New_Precision);
   end LEO_To_Surface;

   function GEO_To_LEO
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (GEO, LEO, Pos.Precision);

      -- Scale from GEO to LEO radius
      GEO_Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
      LEO_Radius : constant Long_Float := Earth_Radius_Meters + 500_000.0;
      Scale : Long_Float;
   begin
      if GEO_Radius > 0.0 then
         Scale := LEO_Radius / GEO_Radius;
      else
         Scale := 1.0;
      end if;

      return (Domain           => LEO,
              X                => Pos.X * Scale,
              Y                => Pos.Y * Scale,
              Z                => Pos.Z * Scale,
              Precision => New_Precision);
   end GEO_To_LEO;

   function Cislunar_To_GEO
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Cislunar, GEO, Pos.Precision);

      -- Scale to GEO radius
      Radius : constant Long_Float := Sqrt (Pos.X * Pos.X + Pos.Y * Pos.Y + Pos.Z * Pos.Z);
      GEO_Radius : constant Long_Float := Earth_Radius_Meters + 35_786_000.0;
      Scale : Long_Float;
   begin
      if Radius > 0.0 then
         Scale := GEO_Radius / Radius;
      else
         Scale := 1.0;
      end if;

      return (Domain           => GEO,
              X                => Pos.X * Scale,
              Y                => Pos.Y * Scale,
              Z                => Pos.Z * Scale,
              Precision => New_Precision);
   end Cislunar_To_GEO;

   function Interplanetary_To_Cislunar
     (Pos : Domain_Position) return Domain_Position is
      New_Precision : constant Precision_Meters :=
        Estimate_Precision_Loss (Interplanetary, Cislunar, Pos.Precision);
   begin
      -- Convert from heliocentric to Earth-centered
      -- For simplicity, use same coordinates
      return (Domain           => Cislunar,
              X                => Pos.X,
              Y                => Pos.Y,
              Z                => Pos.Z,
              Precision => New_Precision);
   end Interplanetary_To_Cislunar;

   ---------------------------------------------------------------------------
   -- General Transfer Function
   ---------------------------------------------------------------------------

   function Transfer_Domain
     (Pos           : Domain_Position;
      Target_Domain : Tracking_Domain) return Domain_Position is
   begin
      -- If already in target domain, return as-is
      if Pos.Domain = Target_Domain then
         return Pos;
      end if;

      -- Route through intermediate domains as needed
      case Pos.Domain is
         when Undersea =>
            case Target_Domain is
               when Surface => return Undersea_To_Surface (Pos);
               when LEO     => return Surface_To_LEO (Undersea_To_Surface (Pos));
               when MEO     => return LEO_To_GEO (Surface_To_LEO (Undersea_To_Surface (Pos)));
               when GEO     => return LEO_To_GEO (Surface_To_LEO (Undersea_To_Surface (Pos)));
               when Cislunar => return GEO_To_Cislunar (LEO_To_GEO (Surface_To_LEO (Undersea_To_Surface (Pos))));
               when Interplanetary => return Cislunar_To_Interplanetary (GEO_To_Cislunar (LEO_To_GEO (Surface_To_LEO (Undersea_To_Surface (Pos)))));
               when others => return Pos;  -- Should not reach
            end case;

         when Surface =>
            case Target_Domain is
               when Undersea => return Surface_To_Undersea (Pos);
               when LEO => return Surface_To_LEO (Pos);
               when MEO | GEO => return LEO_To_GEO (Surface_To_LEO (Pos));
               when Cislunar => return GEO_To_Cislunar (LEO_To_GEO (Surface_To_LEO (Pos)));
               when Interplanetary => return Cislunar_To_Interplanetary (GEO_To_Cislunar (LEO_To_GEO (Surface_To_LEO (Pos))));
               when others => return Pos;
            end case;

         when LEO =>
            case Target_Domain is
               when Undersea => return Surface_To_Undersea (LEO_To_Surface (Pos));
               when Surface => return LEO_To_Surface (Pos);
               when MEO | GEO => return LEO_To_GEO (Pos);
               when Cislunar => return GEO_To_Cislunar (LEO_To_GEO (Pos));
               when Interplanetary => return Cislunar_To_Interplanetary (GEO_To_Cislunar (LEO_To_GEO (Pos)));
               when others => return Pos;
            end case;

         when MEO | GEO =>
            case Target_Domain is
               when Undersea => return Surface_To_Undersea (LEO_To_Surface (GEO_To_LEO (Pos)));
               when Surface => return LEO_To_Surface (GEO_To_LEO (Pos));
               when LEO => return GEO_To_LEO (Pos);
               when MEO => return Pos;  -- Treat MEO and GEO as equivalent for this transfer
               when Cislunar => return GEO_To_Cislunar (Pos);
               when Interplanetary => return Cislunar_To_Interplanetary (GEO_To_Cislunar (Pos));
               when others => return Pos;
            end case;

         when Cislunar =>
            case Target_Domain is
               when GEO | MEO => return Cislunar_To_GEO (Pos);
               when LEO => return GEO_To_LEO (Cislunar_To_GEO (Pos));
               when Surface => return LEO_To_Surface (GEO_To_LEO (Cislunar_To_GEO (Pos)));
               when Undersea => return Surface_To_Undersea (LEO_To_Surface (GEO_To_LEO (Cislunar_To_GEO (Pos))));
               when Interplanetary => return Cislunar_To_Interplanetary (Pos);
               when others => return Pos;
            end case;

         when Interplanetary =>
            case Target_Domain is
               when Cislunar => return Interplanetary_To_Cislunar (Pos);
               when GEO | MEO => return Cislunar_To_GEO (Interplanetary_To_Cislunar (Pos));
               when LEO => return GEO_To_LEO (Cislunar_To_GEO (Interplanetary_To_Cislunar (Pos)));
               when Surface => return LEO_To_Surface (GEO_To_LEO (Cislunar_To_GEO (Interplanetary_To_Cislunar (Pos))));
               when Undersea => return Surface_To_Undersea (LEO_To_Surface (GEO_To_LEO (Cislunar_To_GEO (Interplanetary_To_Cislunar (Pos)))));
               when others => return Pos;
            end case;
      end case;
   end Transfer_Domain;

   ---------------------------------------------------------------------------
   -- Utility Functions
   ---------------------------------------------------------------------------

   function Distance
     (Pos1, Pos2 : Domain_Position) return Long_Float is
      -- Convert both to common frame (Cislunar) for comparison
      P1 : constant Domain_Position := Transfer_Domain (Pos1, Cislunar);
      P2 : constant Domain_Position := Transfer_Domain (Pos2, Cislunar);
   begin
      return Euclidean_Distance (P1.X, P1.Y, P1.Z, P2.X, P2.Y, P2.Z);
   end Distance;

   function Domain_Name (Domain : Tracking_Domain) return String is
   begin
      case Domain is
         when Undersea       => return "Undersea";
         when Surface        => return "Surface";
         when LEO            => return "LEO";
         when MEO            => return "MEO";
         when GEO            => return "GEO";
         when Cislunar       => return "Cislunar";
         when Interplanetary => return "Interplanetary";
      end case;
   end Domain_Name;

   function Precision_Status_Name (Status : Precision_Status) return String is
   begin
      case Status is
         when OK       => return "OK";
         when Warning  => return "Warning";
         when Critical => return "Critical";
      end case;
   end Precision_Status_Name;

end Stone_Soup.Domain_Transfer;
