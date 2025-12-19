-------------------------------------------------------------------------------
-- Stone Soup Automatic Unit Scaling Package Implementation
-------------------------------------------------------------------------------

package body Stone_Soup.Unit_Scaling
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Scale Multiplier Function
   ---------------------------------------------------------------------------

   function Scale_Multiplier (SF : Scale_Factor) return Long_Float is
   begin
      case SF is
         when Nano  => return 1.0e-9;
         when Micro => return 1.0e-6;
         when Milli => return 1.0e-3;
         when Base  => return 1.0;
         when Kilo  => return 1.0e3;
         when Mega  => return 1.0e6;
         when Giga  => return 1.0e9;
      end case;
   end Scale_Multiplier;

   ---------------------------------------------------------------------------
   -- Unit Symbol Functions
   ---------------------------------------------------------------------------

   function Unit_Symbol
     (SF  : Scale_Factor;
      Cat : Unit_Category) return String is
   begin
      case Cat is
         when Distance =>
            case SF is
               when Nano  => return "nm";
               when Micro => return "um";
               when Milli => return "mm";
               when Base  => return "m";
               when Kilo  => return "km";
               when Mega  => return "Mm";
               when Giga  => return "Gm";
            end case;
         when Velocity =>
            case SF is
               when Nano  => return "nm/s";
               when Micro => return "um/s";
               when Milli => return "mm/s";
               when Base  => return "m/s";
               when Kilo  => return "km/s";
               when Mega  => return "Mm/s";
               when Giga  => return "Gm/s";
            end case;
         when Time_Duration =>
            case SF is
               when Nano  => return "ns";
               when Micro => return "us";
               when Milli => return "ms";
               when Base  => return "s";
               when Kilo  => return "ks";
               when Mega  => return "Ms";
               when Giga  => return "Gs";
            end case;
      end case;
   end Unit_Symbol;

   function Time_Unit_Name (SF : Scale_Factor) return String is
   begin
      case SF is
         when Nano  => return "ns";
         when Micro => return "us";
         when Milli => return "ms";
         when Base  => return "s";
         when Kilo  => return "min";
         when Mega  => return "hr";
         when Giga  => return "days";
      end case;
   end Time_Unit_Name;

   ---------------------------------------------------------------------------
   -- Automatic Scaling Functions
   ---------------------------------------------------------------------------

   function Auto_Scale
     (Raw_Value : Long_Float;
      Unit_Type : Unit_Category) return Scaled_Value
   is
      Abs_Value : constant Long_Float := abs Raw_Value;
      Result    : Scaled_Value;
   begin
      Result.Category := Unit_Type;

      -- Choose scale so that value is in range [1, 1000)
      if Abs_Value < 1.0e-6 then
         Result.Scale := Nano;
         Result.Value := Raw_Value / Scale_Multiplier (Nano);
      elsif Abs_Value < 1.0e-3 then
         Result.Scale := Micro;
         Result.Value := Raw_Value / Scale_Multiplier (Micro);
      elsif Abs_Value < 1.0 then
         Result.Scale := Milli;
         Result.Value := Raw_Value / Scale_Multiplier (Milli);
      elsif Abs_Value < 1.0e3 then
         Result.Scale := Base;
         Result.Value := Raw_Value;
      elsif Abs_Value < 1.0e6 then
         Result.Scale := Kilo;
         Result.Value := Raw_Value / Scale_Multiplier (Kilo);
      elsif Abs_Value < 1.0e9 then
         Result.Scale := Mega;
         Result.Value := Raw_Value / Scale_Multiplier (Mega);
      else
         Result.Scale := Giga;
         Result.Value := Raw_Value / Scale_Multiplier (Giga);
      end if;

      return Result;
   end Auto_Scale;

   function To_Base_Units (SV : Scaled_Value) return Long_Float is
   begin
      return SV.Value * Scale_Multiplier (SV.Scale);
   end To_Base_Units;

   function Convert_Scale
     (SV        : Scaled_Value;
      New_Scale : Scale_Factor) return Scaled_Value
   is
      Base_Value : constant Long_Float := To_Base_Units (SV);
      Result     : Scaled_Value;
   begin
      Result.Category := SV.Category;
      Result.Scale := New_Scale;
      Result.Value := Base_Value / Scale_Multiplier (New_Scale);
      return Result;
   end Convert_Scale;

   ---------------------------------------------------------------------------
   -- Direct Construction Functions
   ---------------------------------------------------------------------------

   function Create_Scaled
     (Value     : Long_Float;
      Scale     : Scale_Factor;
      Unit_Type : Unit_Category) return Scaled_Value is
   begin
      return (Value => Value, Scale => Scale, Category => Unit_Type);
   end Create_Scaled;

   function From_Millimeters (Value : Long_Float) return Scaled_Value is
      Base_Meters : constant Long_Float := Value * 1.0e-3;
   begin
      return Auto_Scale (Base_Meters, Distance);
   end From_Millimeters;

   function From_Meters (Value : Long_Float) return Scaled_Value is
   begin
      return Auto_Scale (Value, Distance);
   end From_Meters;

   function From_Kilometers (Value : Long_Float) return Scaled_Value is
      Base_Meters : constant Long_Float := Value * 1.0e3;
   begin
      return Auto_Scale (Base_Meters, Distance);
   end From_Kilometers;

   function From_Meters_Per_Second (Value : Long_Float) return Scaled_Value is
   begin
      return Auto_Scale (Value, Velocity);
   end From_Meters_Per_Second;

   function From_Kilometers_Per_Second
     (Value : Long_Float) return Scaled_Value is
      Base_MPS : constant Long_Float := Value * 1.0e3;
   begin
      return Auto_Scale (Base_MPS, Velocity);
   end From_Kilometers_Per_Second;

   function From_Seconds (Value : Long_Float) return Scaled_Value is
   begin
      return Auto_Scale (Value, Time_Duration);
   end From_Seconds;

   function From_Minutes (Value : Long_Float) return Scaled_Value is
      Base_Seconds : constant Long_Float := Value * 60.0;
   begin
      return Auto_Scale (Base_Seconds, Time_Duration);
   end From_Minutes;

   function From_Hours (Value : Long_Float) return Scaled_Value is
      Base_Seconds : constant Long_Float := Value * 3600.0;
   begin
      return Auto_Scale (Base_Seconds, Time_Duration);
   end From_Hours;

   function From_Days (Value : Long_Float) return Scaled_Value is
      Base_Seconds : constant Long_Float := Value * 86400.0;
   begin
      return Auto_Scale (Base_Seconds, Time_Duration);
   end From_Days;

   ---------------------------------------------------------------------------
   -- Extraction Functions
   ---------------------------------------------------------------------------

   function To_Millimeters (SV : Scaled_Value) return Long_Float is
      Base_Meters : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_Meters / 1.0e-3;
   end To_Millimeters;

   function To_Meters (SV : Scaled_Value) return Long_Float is
   begin
      return To_Base_Units (SV);
   end To_Meters;

   function To_Kilometers (SV : Scaled_Value) return Long_Float is
      Base_Meters : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_Meters / 1.0e3;
   end To_Kilometers;

   function To_Meters_Per_Second (SV : Scaled_Value) return Long_Float is
   begin
      return To_Base_Units (SV);
   end To_Meters_Per_Second;

   function To_Kilometers_Per_Second (SV : Scaled_Value) return Long_Float is
      Base_MPS : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_MPS / 1.0e3;
   end To_Kilometers_Per_Second;

   function To_Seconds (SV : Scaled_Value) return Long_Float is
   begin
      return To_Base_Units (SV);
   end To_Seconds;

   function To_Minutes (SV : Scaled_Value) return Long_Float is
      Base_Seconds : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_Seconds / 60.0;
   end To_Minutes;

   function To_Hours (SV : Scaled_Value) return Long_Float is
      Base_Seconds : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_Seconds / 3600.0;
   end To_Hours;

   function To_Days (SV : Scaled_Value) return Long_Float is
      Base_Seconds : constant Long_Float := To_Base_Units (SV);
   begin
      return Base_Seconds / 86400.0;
   end To_Days;

   ---------------------------------------------------------------------------
   -- Arithmetic Operations
   ---------------------------------------------------------------------------

   function "+" (Left, Right : Scaled_Value) return Scaled_Value is
      Left_Base  : constant Long_Float := To_Base_Units (Left);
      Right_Base : constant Long_Float := To_Base_Units (Right);
      Sum_Base   : constant Long_Float := Left_Base + Right_Base;
   begin
      -- Return in the same scale as left operand
      return Convert_Scale
        (Auto_Scale (Sum_Base, Left.Category), Left.Scale);
   end "+";

   function "-" (Left, Right : Scaled_Value) return Scaled_Value is
      Left_Base  : constant Long_Float := To_Base_Units (Left);
      Right_Base : constant Long_Float := To_Base_Units (Right);
      Diff_Base  : constant Long_Float := Left_Base - Right_Base;
   begin
      -- Return in the same scale as left operand
      return Convert_Scale
        (Auto_Scale (Diff_Base, Left.Category), Left.Scale);
   end "-";

   function "*" (Factor : Long_Float; SV : Scaled_Value) return Scaled_Value is
      Result : Scaled_Value;
   begin
      Result.Value := SV.Value * Factor;
      Result.Scale := SV.Scale;
      Result.Category := SV.Category;
      return Result;
   end "*";

   function "/" (SV : Scaled_Value; Divisor : Long_Float) return Scaled_Value
   is
      Result : Scaled_Value;
   begin
      Result.Value := SV.Value / Divisor;
      Result.Scale := SV.Scale;
      Result.Category := SV.Category;
      return Result;
   end "/";

   ---------------------------------------------------------------------------
   -- Domain-Specific Convenience Functions
   ---------------------------------------------------------------------------

   function Scale_For_Undersea
     (Depth_Meters : Long_Float) return Scaled_Value is
   begin
      -- Undersea: 0-11 km, typically use meters for precision
      if Depth_Meters < 1.0 then
         return Create_Scaled (Depth_Meters * 1000.0, Milli, Distance);
      elsif Depth_Meters < 1000.0 then
         return Create_Scaled (Depth_Meters, Base, Distance);
      else
         return Create_Scaled (Depth_Meters / 1000.0, Kilo, Distance);
      end if;
   end Scale_For_Undersea;

   function Scale_For_Orbital
     (Altitude_Meters : Long_Float) return Scaled_Value is
   begin
      -- Orbital: 500 km to 36,000 km, use kilometers
      return Create_Scaled (Altitude_Meters / 1000.0, Kilo, Distance);
   end Scale_For_Orbital;

   function Scale_For_Cislunar
     (Distance_Meters : Long_Float) return Scaled_Value is
   begin
      -- Cislunar: ~400,000 km, use megameters for convenience
      if Distance_Meters < 1.0e6 then
         return Create_Scaled (Distance_Meters / 1000.0, Kilo, Distance);
      else
         return Create_Scaled (Distance_Meters / 1.0e6, Mega, Distance);
      end if;
   end Scale_For_Cislunar;

   function Scale_For_Interplanetary
     (Distance_Meters : Long_Float) return Scaled_Value is
   begin
      -- Interplanetary: up to 10^12 m, use gigameters
      if Distance_Meters < 1.0e6 then
         return Create_Scaled (Distance_Meters / 1000.0, Kilo, Distance);
      elsif Distance_Meters < 1.0e9 then
         return Create_Scaled (Distance_Meters / 1.0e6, Mega, Distance);
      else
         return Create_Scaled (Distance_Meters / 1.0e9, Giga, Distance);
      end if;
   end Scale_For_Interplanetary;

end Stone_Soup.Unit_Scaling;
