-------------------------------------------------------------------------------
-- Stone Soup Cross-Domain Coordinate Transfer with Precision Management
--
-- This package provides coordinate transformation between tracking domains
-- with explicit precision tracking and management. Supports transfers:
-- - Undersea ↔ Surface (depth to altitude)
-- - Surface ↔ LEO (low Earth orbit)
-- - LEO ↔ GEO (geostationary orbit)
-- - GEO ↔ Cislunar (Earth-Moon system)
-- - Cislunar ↔ Interplanetary
--
-- Precision management:
-- - Tracks accumulated precision loss across transformations
-- - Warns when precision degrades below domain-specific thresholds
-- - Supports hierarchical coordinate frames with precision propagation
--
-- SPARK contracts ensure:
-- - Domain transfer validity (pre/post conditions)
-- - Precision bounds maintained
-- - No numeric overflow in transformations
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with Stone_Soup.Undersea_Types;
with Stone_Soup.Orbital_Types;
with Stone_Soup.Cislunar_Types;

package Stone_Soup.Domain_Transfer
  with SPARK_Mode => On
is

   ---------------------------------------------------------------------------
   -- Tracking Domain Enumeration
   ---------------------------------------------------------------------------

   type Tracking_Domain is
     (Undersea,        -- Underwater tracking (0-11 km depth)
      Surface,         -- Surface/terrestrial (0 km altitude)
      LEO,             -- Low Earth Orbit (200-2000 km)
      MEO,             -- Medium Earth Orbit (2000-35786 km)
      GEO,             -- Geostationary Orbit (~35786 km)
      Cislunar,        -- Earth-Moon system (up to 500,000 km)
      Interplanetary); -- Beyond cislunar (> 500,000 km)

   ---------------------------------------------------------------------------
   -- Precision Thresholds (in meters)
   ---------------------------------------------------------------------------

   -- Domain-specific precision requirements
   Undersea_Precision_Threshold       : constant := 0.01;      -- 1 cm
   Surface_Precision_Threshold        : constant := 0.1;       -- 10 cm
   LEO_Precision_Threshold            : constant := 1.0;       -- 1 m
   MEO_Precision_Threshold            : constant := 10.0;      -- 10 m
   GEO_Precision_Threshold            : constant := 10.0;      -- 10 m
   Cislunar_Precision_Threshold       : constant := 100.0;     -- 100 m
   Interplanetary_Precision_Threshold : constant := 1000.0;    -- 1 km

   -- Maximum precision loss allowed before warning
   Max_Precision_Loss_Factor : constant := 10.0;

   ---------------------------------------------------------------------------
   -- Precision Types (declared before Domain_Position)
   ---------------------------------------------------------------------------

   -- Type for precision tracking
   subtype Precision_Meters is Long_Float range 0.0 .. 1.0e12;

   ---------------------------------------------------------------------------
   -- Domain Position Type
   ---------------------------------------------------------------------------

   -- Position that can span multiple tracking domains
   type Domain_Position is record
      Domain           : Tracking_Domain;
      X                : Long_Float;  -- meters in domain-specific frame
      Y                : Long_Float;  -- meters
      Z                : Long_Float;  -- meters (depth negative for undersea)
      Precision        : Precision_Meters := 0.0;
   end record;

   -- Create a domain position
   function Create_Domain_Position
     (Domain    : Tracking_Domain;
      X, Y, Z   : Long_Float;
      Precision : Precision_Meters := 0.0) return Domain_Position
     with Post => Create_Domain_Position'Result.Domain = Domain and then
                  Create_Domain_Position'Result.X = X and then
                  Create_Domain_Position'Result.Y = Y and then
                  Create_Domain_Position'Result.Z = Z and then
                  Create_Domain_Position'Result.Precision = Precision;

   ---------------------------------------------------------------------------
   -- Precision Management Types
   ---------------------------------------------------------------------------

   type Precision_Status is (OK, Warning, Critical);

   -- Result of precision check
   type Precision_Check_Result is record
      Status         : Precision_Status;
      Current_Meters : Precision_Meters;
      Threshold_Meters : Long_Float;
      Loss_Factor    : Long_Float;  -- How many times threshold exceeded
   end record;

   -- Check precision against domain requirements
   function Check_Precision
     (Pos : Domain_Position) return Precision_Check_Result
     with Post =>
       (if Check_Precision'Result.Status = OK then
          Check_Precision'Result.Current_Meters <= Check_Precision'Result.Threshold_Meters
        elsif Check_Precision'Result.Status = Warning then
          Check_Precision'Result.Current_Meters > Check_Precision'Result.Threshold_Meters
        else
          Check_Precision'Result.Loss_Factor >= Max_Precision_Loss_Factor);

   -- Get precision threshold for a domain
   function Get_Precision_Threshold
     (Domain : Tracking_Domain) return Long_Float
     with Post => Get_Precision_Threshold'Result > 0.0;

   ---------------------------------------------------------------------------
   -- Domain Transfer Validation
   ---------------------------------------------------------------------------

   -- Check if transfer between domains is supported
   function Can_Transfer
     (From_Domain : Tracking_Domain;
      To_Domain   : Tracking_Domain) return Boolean
     with Post =>
       (if From_Domain = To_Domain then Can_Transfer'Result = True);

   -- Check if position is valid for its domain
   function Is_Valid_Position (Pos : Domain_Position) return Boolean;

   ---------------------------------------------------------------------------
   -- Core Transfer Functions
   ---------------------------------------------------------------------------

   -- Transfer position between domains
   function Transfer_Domain
     (Pos           : Domain_Position;
      Target_Domain : Tracking_Domain) return Domain_Position
     with Pre  => Is_Valid_Position (Pos) and then
                  Can_Transfer (Pos.Domain, Target_Domain),
          Post => Transfer_Domain'Result.Domain = Target_Domain and then
                  Is_Valid_Position (Transfer_Domain'Result) and then
                  Transfer_Domain'Result.Precision >= Pos.Precision;

   ---------------------------------------------------------------------------
   -- Specific Domain Transfer Functions
   ---------------------------------------------------------------------------

   -- Undersea to Surface transfer (depth becomes negative altitude)
   function Undersea_To_Surface
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Undersea and then
                  Is_Valid_Position (Pos),
          Post => Undersea_To_Surface'Result.Domain = Surface and then
                  Is_Valid_Position (Undersea_To_Surface'Result);

   -- Surface to LEO transfer (altitude becomes orbital altitude)
   function Surface_To_LEO
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Surface and then
                  Is_Valid_Position (Pos),
          Post => Surface_To_LEO'Result.Domain = LEO and then
                  Is_Valid_Position (Surface_To_LEO'Result);

   -- LEO to GEO transfer (coordinate frame change)
   function LEO_To_GEO
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = LEO and then
                  Is_Valid_Position (Pos),
          Post => LEO_To_GEO'Result.Domain = GEO and then
                  Is_Valid_Position (LEO_To_GEO'Result);

   -- GEO to Cislunar transfer
   function GEO_To_Cislunar
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = GEO and then
                  Is_Valid_Position (Pos),
          Post => GEO_To_Cislunar'Result.Domain = Cislunar and then
                  Is_Valid_Position (GEO_To_Cislunar'Result);

   -- Cislunar to Interplanetary transfer
   function Cislunar_To_Interplanetary
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Cislunar and then
                  Is_Valid_Position (Pos),
          Post => Cislunar_To_Interplanetary'Result.Domain = Interplanetary and then
                  Is_Valid_Position (Cislunar_To_Interplanetary'Result);

   ---------------------------------------------------------------------------
   -- Reverse Transfer Functions
   ---------------------------------------------------------------------------

   -- Surface to Undersea (altitude becomes depth)
   function Surface_To_Undersea
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Surface and then
                  Is_Valid_Position (Pos),
          Post => Surface_To_Undersea'Result.Domain = Undersea and then
                  Is_Valid_Position (Surface_To_Undersea'Result);

   -- LEO to Surface
   function LEO_To_Surface
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = LEO and then
                  Is_Valid_Position (Pos),
          Post => LEO_To_Surface'Result.Domain = Surface and then
                  Is_Valid_Position (LEO_To_Surface'Result);

   -- GEO to LEO
   function GEO_To_LEO
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = GEO and then
                  Is_Valid_Position (Pos),
          Post => GEO_To_LEO'Result.Domain = LEO and then
                  Is_Valid_Position (GEO_To_LEO'Result);

   -- Cislunar to GEO
   function Cislunar_To_GEO
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Cislunar and then
                  Is_Valid_Position (Pos),
          Post => Cislunar_To_GEO'Result.Domain = GEO and then
                  Is_Valid_Position (Cislunar_To_GEO'Result);

   -- Interplanetary to Cislunar
   function Interplanetary_To_Cislunar
     (Pos : Domain_Position) return Domain_Position
     with Pre  => Pos.Domain = Interplanetary and then
                  Is_Valid_Position (Pos),
          Post => Interplanetary_To_Cislunar'Result.Domain = Cislunar and then
                  Is_Valid_Position (Interplanetary_To_Cislunar'Result);

   ---------------------------------------------------------------------------
   -- Utility Functions
   ---------------------------------------------------------------------------

   -- Compute distance between two domain positions (converts to common frame)
   function Distance
     (Pos1, Pos2 : Domain_Position) return Long_Float
     with Pre  => Is_Valid_Position (Pos1) and then
                  Is_Valid_Position (Pos2),
          Post => Distance'Result >= 0.0;

   -- Get domain name as string
   function Domain_Name (Domain : Tracking_Domain) return String
     with Post => Domain_Name'Result'Length > 0;

   -- Get precision status description
   function Precision_Status_Name (Status : Precision_Status) return String
     with Post => Precision_Status_Name'Result'Length > 0;

end Stone_Soup.Domain_Transfer;
