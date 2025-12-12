-------------------------------------------------------------------------------
-- Stone Soup Ada Bindings Specification
--
-- This package provides Ada bindings to the Stone Soup tracking framework.
-- Stone Soup is a framework for target tracking and state estimation.
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with Interfaces.C;
with Interfaces.C.Strings;

package Stone_Soup is

   pragma Elaborate_Body;

   -- Version information
   Version : constant String := "0.1.0";

   -- Exception types
   Initialization_Error : exception;
   Runtime_Error : exception;
   Invalid_Parameter : exception;
   Null_Pointer_Error : exception;

   -- Initialize the Stone Soup library
   -- Raises: Initialization_Error if initialization fails
   procedure Initialize;

   -- Clean up Stone Soup resources
   -- Raises: Runtime_Error if cleanup fails
   procedure Cleanup;

   -- State vector type
   type State_Vector is private;

   -- Create a new state vector
   function Create_State_Vector (Dimension : Positive) return State_Vector;

   -- Get dimensionality of state vector
   function Dims (SV : State_Vector) return Natural;

   -- Get value at index (1-based indexing)
   function Get (SV : State_Vector; Index : Positive) return Float;

   -- Set value at index (1-based indexing)
   procedure Set (SV : in out State_Vector; Index : Positive; Value : Float);

   -- Covariance matrix type
   type Covariance_Matrix is private;

   -- Create a new covariance matrix
   function Create_Covariance (Dimension : Positive) return Covariance_Matrix;

   -- Gaussian state type
   type Gaussian_State is private;

   -- Create a new Gaussian state
   function Create_Gaussian_State
      (SV : State_Vector; Covar : Covariance_Matrix) return Gaussian_State;

   -- Get state vector from Gaussian state
   function Get_State_Vector (GS : Gaussian_State) return State_Vector;

   -- Get covariance from Gaussian state
   function Get_Covariance (GS : Gaussian_State) return Covariance_Matrix;

   -- Detection type
   type Detection is private;

   -- Create a new detection
   function Create_Detection
      (Measurement : State_Vector; Timestamp : Float) return Detection;

   -- Get measurement from detection
   function Get_Measurement (D : Detection) return State_Vector;

   -- Get timestamp from detection
   function Get_Timestamp (D : Detection) return Float;

   -- Track type
   type Track is private;

   -- Create a new track
   function Create_Track (ID : String) return Track;

   -- Get track ID
   function Get_ID (T : Track) return String;

   -- Get track length (number of states)
   function Length (T : Track) return Natural;

private

   use Interfaces.C;

   -- Private type implementations
   type State_Vector_Data;
   type State_Vector is access State_Vector_Data;

   type Covariance_Matrix_Data;
   type Covariance_Matrix is access Covariance_Matrix_Data;

   type Gaussian_State_Data;
   type Gaussian_State is access Gaussian_State_Data;

   type Detection_Data;
   type Detection is access Detection_Data;

   type Track_Data;
   type Track is access Track_Data;

   -- C API bindings (to be implemented when C library is available)
   -- pragma Import (C, stonesoup_init, "stonesoup_init");
   -- pragma Import (C, stonesoup_cleanup, "stonesoup_cleanup");

end Stone_Soup;
