-------------------------------------------------------------------------------
-- Stone Soup Ada Bindings Specification
--
-- This package provides Ada bindings to the Stone Soup tracking framework.
-- Stone Soup is a framework for target tracking and state estimation.
--
-- SPARK_Mode is enabled for formal verification of critical operations.
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with Interfaces.C;
with Interfaces.C.Strings;

package Stone_Soup
  with SPARK_Mode => On
is

   -- Version information
   Version : constant String := "0.1.0";

   -- Maximum supported dimensions for state vectors
   Max_Dimension : constant := 100;

   -- Dimension subtype for bounds checking
   subtype Dimension_Range is Positive range 1 .. Max_Dimension;
   subtype Dimension_Index is Natural range 0 .. Max_Dimension - 1;

   -- Exception types
   Initialization_Error : exception;
   Runtime_Error        : exception;
   Invalid_Parameter    : exception;
   Null_Pointer_Error   : exception;
   Dimension_Mismatch   : exception;
   Singular_Matrix      : exception;

   ---------------------------------------------------------------------------
   -- State Vector Type
   ---------------------------------------------------------------------------

   -- Fixed-size array for state vector data (1-based indexing for Ada)
   type State_Vector_Array is array (Dimension_Range range <>) of Long_Float;

   -- State vector with known dimension
   type State_Vector (Dim : Dimension_Range) is record
      Data : State_Vector_Array (1 .. Dim) := [others => 0.0];
   end record
     with Dynamic_Predicate => Dim > 0;

   -- Create a zero state vector
   function Zeros (Dim : Dimension_Range) return State_Vector
     with Post => Zeros'Result.Dim = Dim and then
                  (for all I in 1 .. Dim =>
                     Zeros'Result.Data (I) = 0.0);

   -- Create a state vector filled with a value
   function Fill (Dim : Dimension_Range; Value : Long_Float) return State_Vector
     with Post => Fill'Result.Dim = Dim and then
                  (for all I in 1 .. Dim =>
                     Fill'Result.Data (I) = Value);

   -- Get the dimension of a state vector
   function Get_Dim (SV : State_Vector) return Dimension_Range
     with Post => Get_Dim'Result = SV.Dim;

   -- Get value at index (1-based)
   function Get (SV : State_Vector; Index : Dimension_Range) return Long_Float
     with Pre => Index <= SV.Dim,
          Post => Get'Result = SV.Data (Index);

   -- Set value at index (1-based)
   procedure Set
     (SV    : in out State_Vector;
      Index : Dimension_Range;
      Value : Long_Float)
     with Pre  => Index <= SV.Dim,
          Post => SV.Data (Index) = Value and then
                  (for all I in 1 .. SV.Dim =>
                     (if I /= Index then SV.Data (I) = SV.Data'Old (I)));

   -- Compute Euclidean norm
   function Norm (SV : State_Vector) return Long_Float
     with Post => Norm'Result >= 0.0;

   -- Add two state vectors
   function "+" (Left, Right : State_Vector) return State_Vector
     with Pre  => Left.Dim = Right.Dim,
          Post => "+"'Result.Dim = Left.Dim;

   -- Subtract two state vectors
   function "-" (Left, Right : State_Vector) return State_Vector
     with Pre  => Left.Dim = Right.Dim,
          Post => "-"'Result.Dim = Left.Dim;

   -- Scale a state vector
   function "*" (Factor : Long_Float; SV : State_Vector) return State_Vector
     with Post => "*"'Result.Dim = SV.Dim;

   ---------------------------------------------------------------------------
   -- Covariance Matrix Type
   ---------------------------------------------------------------------------

   -- Fixed-size 2D array for matrix data (row-major, 1-based indexing)
   type Matrix_Array is array (Dimension_Range range <>,
                               Dimension_Range range <>) of Long_Float;

   -- Covariance matrix with known dimension
   type Covariance_Matrix (Dim : Dimension_Range) is record
      Data : Matrix_Array (1 .. Dim, 1 .. Dim) := [others => [others => 0.0]];
   end record
     with Dynamic_Predicate => Dim > 0;

   -- Create an identity matrix
   function Identity (Dim : Dimension_Range) return Covariance_Matrix
     with Post => Identity'Result.Dim = Dim;

   -- Create a zero matrix
   function Zero_Matrix (Dim : Dimension_Range) return Covariance_Matrix
     with Post => Zero_Matrix'Result.Dim = Dim;

   -- Create a diagonal matrix
   function Diagonal (Diag : State_Vector) return Covariance_Matrix
     with Post => Diagonal'Result.Dim = Diag.Dim;

   -- Get matrix dimension
   function Get_Dim (M : Covariance_Matrix) return Dimension_Range
     with Post => Get_Dim'Result = M.Dim;

   -- Get element at (row, col) - 1-based indexing
   function Get
     (M   : Covariance_Matrix;
      Row : Dimension_Range;
      Col : Dimension_Range) return Long_Float
     with Pre  => Row <= M.Dim and Col <= M.Dim,
          Post => Get'Result = M.Data (Row, Col);

   -- Set element at (row, col) - 1-based indexing
   procedure Set
     (M     : in out Covariance_Matrix;
      Row   : Dimension_Range;
      Col   : Dimension_Range;
      Value : Long_Float)
     with Pre  => Row <= M.Dim and Col <= M.Dim,
          Post => M.Data (Row, Col) = Value;

   -- Compute trace (sum of diagonal)
   function Trace (M : Covariance_Matrix) return Long_Float;

   -- Add two matrices
   function "+" (Left, Right : Covariance_Matrix) return Covariance_Matrix
     with Pre  => Left.Dim = Right.Dim,
          Post => "+"'Result.Dim = Left.Dim;

   -- Subtract two matrices
   function "-" (Left, Right : Covariance_Matrix) return Covariance_Matrix
     with Pre  => Left.Dim = Right.Dim,
          Post => "-"'Result.Dim = Left.Dim;

   -- Scale a matrix
   function "*" (Factor : Long_Float; M : Covariance_Matrix) return Covariance_Matrix
     with Post => "*"'Result.Dim = M.Dim;

   -- Matrix multiplication
   function "*" (Left, Right : Covariance_Matrix) return Covariance_Matrix
     with Pre  => Left.Dim = Right.Dim,
          Post => "*"'Result.Dim = Left.Dim;

   -- Matrix-vector multiplication
   function "*" (M : Covariance_Matrix; V : State_Vector) return State_Vector
     with Pre  => M.Dim = V.Dim,
          Post => "*"'Result.Dim = V.Dim;

   -- Transpose a matrix
   function Transpose (M : Covariance_Matrix) return Covariance_Matrix
     with Post => Transpose'Result.Dim = M.Dim;

   ---------------------------------------------------------------------------
   -- Gaussian State Type
   ---------------------------------------------------------------------------

   -- Gaussian state with mean and covariance
   type Gaussian_State (Dim : Dimension_Range) is record
      State_Vector : Stone_Soup.State_Vector (Dim);
      Covariance   : Covariance_Matrix (Dim);
      Timestamp    : Long_Float := 0.0;
      Has_Timestamp : Boolean := False;
   end record;

   -- Create a Gaussian state
   function Create_Gaussian_State
     (SV        : State_Vector;
      Covar     : Covariance_Matrix;
      Timestamp : Long_Float := 0.0) return Gaussian_State
     with Pre  => SV.Dim = Covar.Dim,
          Post => Create_Gaussian_State'Result.Dim = SV.Dim;

   -- Get the state vector from a Gaussian state
   function Get_State_Vector (GS : Gaussian_State) return State_Vector
     with Post => Get_State_Vector'Result.Dim = GS.Dim;

   -- Get the covariance from a Gaussian state
   function Get_Covariance (GS : Gaussian_State) return Covariance_Matrix
     with Post => Get_Covariance'Result.Dim = GS.Dim;

   -- Get state element
   function Get_State (GS : Gaussian_State; Index : Dimension_Index) return Long_Float
     with Pre => Index < GS.Dim;

   -- Get variance (diagonal covariance element)
   function Get_Variance (GS : Gaussian_State; Index : Dimension_Index) return Long_Float
     with Pre => Index < GS.Dim;

   ---------------------------------------------------------------------------
   -- Kalman Filter Operations
   ---------------------------------------------------------------------------

   -- Kalman filter prediction step
   -- x_pred = F * x
   -- P_pred = F * P * F^T + Q
   function Kalman_Predict
     (Prior             : Gaussian_State;
      Transition_Matrix : Covariance_Matrix;
      Process_Noise     : Covariance_Matrix) return Gaussian_State
     with Pre  => Prior.Dim = Transition_Matrix.Dim and then
                  Prior.Dim = Process_Noise.Dim,
          Post => Kalman_Predict'Result.Dim = Prior.Dim;

   -- Kalman filter update step
   -- Innovation: y = z - H * x_pred
   -- Innovation covariance: S = H * P * H^T + R
   -- Kalman gain: K = P * H^T * S^-1
   -- Posterior: x_post = x_pred + K * y
   --           P_post = (I - K * H) * P_pred
   function Kalman_Update
     (Predicted         : Gaussian_State;
      Measurement       : State_Vector;
      Measurement_Matrix : Covariance_Matrix;
      Measurement_Noise : Covariance_Matrix) return Gaussian_State
     with Pre  => Predicted.Dim = Measurement_Matrix.Dim and then
                  Measurement.Dim = Measurement_Noise.Dim,
          Post => Kalman_Update'Result.Dim = Predicted.Dim;

   -- Create a constant velocity transition matrix
   -- For 2D: state is [x, vx, y, vy], transition is:
   -- | 1  dt  0   0 |
   -- | 0   1  0   0 |
   -- | 0   0  1  dt |
   -- | 0   0  0   1 |
   function Constant_Velocity_Transition
     (Spatial_Dims : Positive;
      Dt           : Long_Float) return Covariance_Matrix
     with Pre  => Spatial_Dims * 2 <= Max_Dimension,
          Post => Constant_Velocity_Transition'Result.Dim = Spatial_Dims * 2;

   -- Create a position-only measurement matrix
   -- For 2D: extracts [x, y] from [x, vx, y, vy]
   function Position_Measurement (Spatial_Dims : Positive) return Covariance_Matrix
     with Pre  => Spatial_Dims * 2 <= Max_Dimension,
          Post => Position_Measurement'Result.Dim = Spatial_Dims * 2;

   ---------------------------------------------------------------------------
   -- Detection Type
   ---------------------------------------------------------------------------

   -- Detection from a sensor
   type Detection (Dim : Dimension_Range) is record
      Measurement : State_Vector (Dim);
      Timestamp   : Long_Float := 0.0;
   end record;

   -- Create a detection
   function Create_Detection
     (Measurement : State_Vector;
      Timestamp   : Long_Float) return Detection
     with Post => Create_Detection'Result.Dim = Measurement.Dim;

   ---------------------------------------------------------------------------
   -- Track Type
   ---------------------------------------------------------------------------

   -- Maximum track length
   Max_Track_Length : constant := 1000;

   -- Array of Gaussian states for track history
   type State_History is array (Positive range <>) of access Gaussian_State;

   -- Track ID string
   subtype Track_ID is String (1 .. 36);

   -- Track representing a target over time
   type Track (Dim : Dimension_Range; Max_States : Positive) is record
      ID          : Track_ID := (others => ' ');
      States      : State_History (1 .. Max_States);
      Num_States  : Natural := 0;
   end record;

   -- Create a track
   function Create_Track
     (ID         : String;
      Dim        : Dimension_Range;
      Max_States : Positive := Max_Track_Length) return Track;

   -- Get number of states in track
   function Length (T : Track) return Natural
     with Post => Length'Result = T.Num_States;

   -- Check if track is empty
   function Is_Empty (T : Track) return Boolean
     with Post => Is_Empty'Result = (T.Num_States = 0);

   ---------------------------------------------------------------------------
   -- Library Initialization
   ---------------------------------------------------------------------------

   -- Initialize the Stone Soup library
   procedure Initialize;

   -- Clean up Stone Soup resources
   procedure Cleanup;

   -- Check if library is initialized
   function Is_Initialized return Boolean;

end Stone_Soup;
