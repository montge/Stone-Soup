-------------------------------------------------------------------------------
-- Stone Soup Ada Unit Tests
--
-- This file contains AUnit tests for the Stone Soup Ada bindings.
-- Run with: gnatmake -P stonesoup_tests.gpr && ./obj/test_runner
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Test_Caller;
with AUnit.Test_Suites;
with AUnit.Assertions;
with Stone_Soup;

package body Test_Stone_Soup is

   use AUnit.Assertions;
   use Stone_Soup;

   Epsilon : constant Long_Float := 1.0e-10;

   ---------------------------------------------------------------------------
   -- Required AUnit Overrides
   ---------------------------------------------------------------------------

   overriding
   function Name (T : Test_Case) return AUnit.Message_String is
      pragma Unreferenced (T);
   begin
      return AUnit.Format ("Stone Soup Ada Tests");
   end Name;

   overriding
   procedure Register_Tests (T : in Out Test_Case) is
      pragma Unreferenced (T);
   begin
      -- Tests are registered in Suite function using Test_Caller
      null;
   end Register_Tests;

   ---------------------------------------------------------------------------
   -- State Vector Tests
   ---------------------------------------------------------------------------

   procedure Test_State_Vector_Zeros (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV : State_Vector := Zeros (3);
   begin
      Assert (Get_Dim (SV) = 3, "Dimension should be 3");
      Assert (abs (Get (SV, 1)) < Epsilon, "Element 1 should be 0");
      Assert (abs (Get (SV, 2)) < Epsilon, "Element 2 should be 0");
      Assert (abs (Get (SV, 3)) < Epsilon, "Element 3 should be 0");
   end Test_State_Vector_Zeros;

   procedure Test_State_Vector_Fill (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV : State_Vector := Fill (2, 5.0);
   begin
      Assert (Get_Dim (SV) = 2, "Dimension should be 2");
      Assert (abs (Get (SV, 1) - 5.0) < Epsilon, "Element 1 should be 5");
      Assert (abs (Get (SV, 2) - 5.0) < Epsilon, "Element 2 should be 5");
   end Test_State_Vector_Fill;

   procedure Test_State_Vector_Set_Get (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV : State_Vector := Zeros (3);
   begin
      Set (SV, 1, 1.0);
      Set (SV, 2, 2.0);
      Set (SV, 3, 3.0);

      Assert (abs (Get (SV, 1) - 1.0) < Epsilon, "Element 1 should be 1");
      Assert (abs (Get (SV, 2) - 2.0) < Epsilon, "Element 2 should be 2");
      Assert (abs (Get (SV, 3) - 3.0) < Epsilon, "Element 3 should be 3");
   end Test_State_Vector_Set_Get;

   procedure Test_State_Vector_Norm (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV : State_Vector := Zeros (2);
   begin
      Set (SV, 1, 3.0);
      Set (SV, 2, 4.0);

      -- ||[3, 4]|| = 5
      Assert (abs (Norm (SV) - 5.0) < Epsilon, "Norm of [3,4] should be 5");
   end Test_State_Vector_Norm;

   procedure Test_State_Vector_Add (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      A, B, Sum : State_Vector := Zeros (2);
   begin
      Set (A, 1, 1.0);
      Set (A, 2, 2.0);
      Set (B, 1, 3.0);
      Set (B, 2, 4.0);

      Sum := A + B;

      Assert (abs (Get (Sum, 1) - 4.0) < Epsilon, "Sum(1) should be 4");
      Assert (abs (Get (Sum, 2) - 6.0) < Epsilon, "Sum(2) should be 6");
   end Test_State_Vector_Add;

   procedure Test_State_Vector_Subtract (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      A, B, Diff : State_Vector := Zeros (2);
   begin
      Set (A, 1, 5.0);
      Set (A, 2, 6.0);
      Set (B, 1, 2.0);
      Set (B, 2, 1.0);

      Diff := A - B;

      Assert (abs (Get (Diff, 1) - 3.0) < Epsilon, "Diff(1) should be 3");
      Assert (abs (Get (Diff, 2) - 5.0) < Epsilon, "Diff(2) should be 5");
   end Test_State_Vector_Subtract;

   procedure Test_State_Vector_Scale (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV, Scaled : State_Vector := Zeros (2);
   begin
      Set (SV, 1, 2.0);
      Set (SV, 2, 3.0);

      Scaled := 2.0 * SV;

      Assert (abs (Get (Scaled, 1) - 4.0) < Epsilon, "Scaled(1) should be 4");
      Assert (abs (Get (Scaled, 2) - 6.0) < Epsilon, "Scaled(2) should be 6");
   end Test_State_Vector_Scale;

   ---------------------------------------------------------------------------
   -- Covariance Matrix Tests
   ---------------------------------------------------------------------------

   procedure Test_Covariance_Identity (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      M : Covariance_Matrix := Identity (3);
   begin
      Assert (Get_Dim (M) = 3, "Dimension should be 3");
      Assert (abs (Get (M, 1, 1) - 1.0) < Epsilon, "(1,1) should be 1");
      Assert (abs (Get (M, 2, 2) - 1.0) < Epsilon, "(2,2) should be 1");
      Assert (abs (Get (M, 3, 3) - 1.0) < Epsilon, "(3,3) should be 1");
      Assert (abs (Get (M, 1, 2)) < Epsilon, "(1,2) should be 0");
      Assert (abs (Get (M, 2, 1)) < Epsilon, "(2,1) should be 0");
   end Test_Covariance_Identity;

   procedure Test_Covariance_Trace (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      M : Covariance_Matrix := Identity (4);
   begin
      Assert (abs (Trace (M) - 4.0) < Epsilon, "Trace of 4x4 identity should be 4");
   end Test_Covariance_Trace;

   procedure Test_Covariance_Multiply (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      A, B, C : Covariance_Matrix := Zero_Matrix (2);
   begin
      -- A = [[1, 2], [3, 4]]
      Set (A, 1, 1, 1.0);
      Set (A, 1, 2, 2.0);
      Set (A, 2, 1, 3.0);
      Set (A, 2, 2, 4.0);

      -- B = [[5, 6], [7, 8]]
      Set (B, 1, 1, 5.0);
      Set (B, 1, 2, 6.0);
      Set (B, 2, 1, 7.0);
      Set (B, 2, 2, 8.0);

      C := A * B;

      -- C[1,1] = 1*5 + 2*7 = 19
      -- C[1,2] = 1*6 + 2*8 = 22
      -- C[2,1] = 3*5 + 4*7 = 43
      -- C[2,2] = 3*6 + 4*8 = 50
      Assert (abs (Get (C, 1, 1) - 19.0) < Epsilon, "C(1,1) should be 19");
      Assert (abs (Get (C, 1, 2) - 22.0) < Epsilon, "C(1,2) should be 22");
      Assert (abs (Get (C, 2, 1) - 43.0) < Epsilon, "C(2,1) should be 43");
      Assert (abs (Get (C, 2, 2) - 50.0) < Epsilon, "C(2,2) should be 50");
   end Test_Covariance_Multiply;

   procedure Test_Matrix_Vector_Multiply (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      M : Covariance_Matrix := Zero_Matrix (2);
      V, Result : State_Vector := Zeros (2);
   begin
      -- M = [[1, 2], [3, 4]]
      Set (M, 1, 1, 1.0);
      Set (M, 1, 2, 2.0);
      Set (M, 2, 1, 3.0);
      Set (M, 2, 2, 4.0);

      -- V = [1, 2]
      Set (V, 1, 1.0);
      Set (V, 2, 2.0);

      Result := M * V;

      -- Result = [1*1+2*2, 3*1+4*2] = [5, 11]
      Assert (abs (Get (Result, 1) - 5.0) < Epsilon, "Result(1) should be 5");
      Assert (abs (Get (Result, 2) - 11.0) < Epsilon, "Result(2) should be 11");
   end Test_Matrix_Vector_Multiply;

   procedure Test_Covariance_Transpose (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      M, Trans : Covariance_Matrix := Zero_Matrix (2);
   begin
      Set (M, 1, 1, 1.0);
      Set (M, 1, 2, 2.0);
      Set (M, 2, 1, 3.0);
      Set (M, 2, 2, 4.0);

      Trans := Transpose (M);

      Assert (abs (Get (Trans, 1, 1) - 1.0) < Epsilon, "Trans(1,1) should be 1");
      Assert (abs (Get (Trans, 1, 2) - 3.0) < Epsilon, "Trans(1,2) should be 3");
      Assert (abs (Get (Trans, 2, 1) - 2.0) < Epsilon, "Trans(2,1) should be 2");
      Assert (abs (Get (Trans, 2, 2) - 4.0) < Epsilon, "Trans(2,2) should be 4");
   end Test_Covariance_Transpose;

   ---------------------------------------------------------------------------
   -- Gaussian State Tests
   ---------------------------------------------------------------------------

   procedure Test_Gaussian_State_Create (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV    : State_Vector := Zeros (2);
      Covar : Covariance_Matrix := Identity (2);
      GS    : Gaussian_State (2);
   begin
      Set (SV, 1, 1.0);
      Set (SV, 2, 2.0);

      GS := Create_Gaussian_State (SV, Covar, 10.0);

      Assert (GS.Dim = 2, "Dimension should be 2");
      Assert (abs (Get_State (GS, 1) - 1.0) < Epsilon, "State(1) should be 1");
      Assert (abs (Get_State (GS, 2) - 2.0) < Epsilon, "State(2) should be 2");
      Assert (abs (Get_Variance (GS, 1) - 1.0) < Epsilon, "Variance(1) should be 1");
   end Test_Gaussian_State_Create;

   ---------------------------------------------------------------------------
   -- Kalman Filter Tests
   ---------------------------------------------------------------------------

   procedure Test_Kalman_Predict (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      SV        : State_Vector := Zeros (4);
      Covar     : Covariance_Matrix := Identity (4);
      Prior     : Gaussian_State (4);
      F         : Covariance_Matrix (4);
      Q         : Covariance_Matrix (4);
      Predicted : Gaussian_State (4);
   begin
      -- Initial state: x=0, vx=1, y=0, vy=1
      Set (SV, 1, 0.0);
      Set (SV, 2, 1.0);
      Set (SV, 3, 0.0);
      Set (SV, 4, 1.0);

      Prior := Create_Gaussian_State (SV, Covar);
      F := Constant_Velocity_Transition (2, 1.0);
      Q := 0.1 * Identity (4);

      Predicted := Kalman_Predict (Prior, F, Q);

      -- After dt=1: x=0+1*1=1, vx=1, y=0+1*1=1, vy=1
      Assert (abs (Get_State (Predicted, 1) - 1.0) < 0.001,
              "Predicted x should be 1");
      Assert (abs (Get_State (Predicted, 2) - 1.0) < 0.001,
              "Predicted vx should be 1");
      Assert (abs (Get_State (Predicted, 3) - 1.0) < 0.001,
              "Predicted y should be 1");
      Assert (abs (Get_State (Predicted, 4) - 1.0) < 0.001,
              "Predicted vy should be 1");
   end Test_Kalman_Predict;

   procedure Test_Constant_Velocity_Transition (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      F : Covariance_Matrix := Constant_Velocity_Transition (2, 0.5);
   begin
      Assert (Get_Dim (F) = 4, "Dimension should be 4");

      -- Check structure:
      -- | 1  0.5  0   0  |
      -- | 0   1   0   0  |
      -- | 0   0   1  0.5 |
      -- | 0   0   0   1  |
      Assert (abs (Get (F, 1, 1) - 1.0) < Epsilon, "F(1,1) should be 1");
      Assert (abs (Get (F, 1, 2) - 0.5) < Epsilon, "F(1,2) should be 0.5");
      Assert (abs (Get (F, 2, 2) - 1.0) < Epsilon, "F(2,2) should be 1");
      Assert (abs (Get (F, 3, 3) - 1.0) < Epsilon, "F(3,3) should be 1");
      Assert (abs (Get (F, 3, 4) - 0.5) < Epsilon, "F(3,4) should be 0.5");
      Assert (abs (Get (F, 4, 4) - 1.0) < Epsilon, "F(4,4) should be 1");
   end Test_Constant_Velocity_Transition;

   procedure Test_Position_Measurement (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      H : Covariance_Matrix := Position_Measurement (2);
   begin
      Assert (Get_Dim (H) = 4, "Dimension should be 4");

      -- Check structure - extracts x and y from [x, vx, y, vy]
      Assert (abs (Get (H, 1, 1) - 1.0) < Epsilon, "H(1,1) should be 1");
      Assert (abs (Get (H, 1, 2)) < Epsilon, "H(1,2) should be 0");
      Assert (abs (Get (H, 2, 3) - 1.0) < Epsilon, "H(2,3) should be 1");
      Assert (abs (Get (H, 2, 4)) < Epsilon, "H(2,4) should be 0");
   end Test_Position_Measurement;

   ---------------------------------------------------------------------------
   -- Detection Tests
   ---------------------------------------------------------------------------

   procedure Test_Detection_Create (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      Meas : State_Vector := Zeros (2);
      D    : Detection (2);
   begin
      Set (Meas, 1, 100.0);
      Set (Meas, 2, 200.0);

      D := Create_Detection (Meas, 0.0);

      Assert (D.Dim = 2, "Dimension should be 2");
      Assert (abs (Get (D.Measurement, 1) - 100.0) < Epsilon, "Meas(1) should be 100");
      Assert (abs (Get (D.Measurement, 2) - 200.0) < Epsilon, "Meas(2) should be 200");
      Assert (abs (D.Timestamp) < Epsilon, "Timestamp should be 0");
   end Test_Detection_Create;

   ---------------------------------------------------------------------------
   -- Track Tests
   ---------------------------------------------------------------------------

   procedure Test_Track_Create (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
      Trk : Track := Create_Track ("test-track", 4, 100);
   begin
      Assert (Is_Empty (Trk), "Track should be empty");
      Assert (Length (Trk) = 0, "Track length should be 0");
      Assert (Trk.Dim = 4, "Track dimension should be 4");
   end Test_Track_Create;

   ---------------------------------------------------------------------------
   -- Library Initialization Tests
   ---------------------------------------------------------------------------

   procedure Test_Initialize (T : in out AUnit.Test_Cases.Test_Case'Class) is
      pragma Unreferenced (T);
   begin
      Initialize;
      Assert (Is_Initialized, "Library should be initialized");
      Cleanup;
      Assert (not Is_Initialized, "Library should not be initialized after cleanup");
   end Test_Initialize;

   ---------------------------------------------------------------------------
   -- Suite Registration
   ---------------------------------------------------------------------------

   function Suite return AUnit.Test_Suites.Access_Test_Suite is
      Result : constant AUnit.Test_Suites.Access_Test_Suite :=
         AUnit.Test_Suites.New_Suite;

      package State_Vector_Caller is new AUnit.Test_Caller (Test_Case);
      package Covariance_Caller is new AUnit.Test_Caller (Test_Case);
      package Gaussian_State_Caller is new AUnit.Test_Caller (Test_Case);
      package Kalman_Caller is new AUnit.Test_Caller (Test_Case);
      package Detection_Caller is new AUnit.Test_Caller (Test_Case);
      package Track_Caller is new AUnit.Test_Caller (Test_Case);
      package Init_Caller is new AUnit.Test_Caller (Test_Case);
   begin
      -- State Vector Tests
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Zeros", Test_State_Vector_Zeros'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Fill", Test_State_Vector_Fill'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.SetGet", Test_State_Vector_Set_Get'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Norm", Test_State_Vector_Norm'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Add", Test_State_Vector_Add'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Subtract", Test_State_Vector_Subtract'Access));
      Result.Add_Test
         (State_Vector_Caller.Create ("StateVector.Scale", Test_State_Vector_Scale'Access));

      -- Covariance Matrix Tests
      Result.Add_Test
         (Covariance_Caller.Create ("Covariance.Identity", Test_Covariance_Identity'Access));
      Result.Add_Test
         (Covariance_Caller.Create ("Covariance.Trace", Test_Covariance_Trace'Access));
      Result.Add_Test
         (Covariance_Caller.Create ("Covariance.Multiply", Test_Covariance_Multiply'Access));
      Result.Add_Test
         (Covariance_Caller.Create ("Covariance.MatVec", Test_Matrix_Vector_Multiply'Access));
      Result.Add_Test
         (Covariance_Caller.Create ("Covariance.Transpose", Test_Covariance_Transpose'Access));

      -- Gaussian State Tests
      Result.Add_Test
         (Gaussian_State_Caller.Create ("GaussianState.Create", Test_Gaussian_State_Create'Access));

      -- Kalman Filter Tests
      Result.Add_Test
         (Kalman_Caller.Create ("Kalman.Predict", Test_Kalman_Predict'Access));
      Result.Add_Test
         (Kalman_Caller.Create ("Kalman.CVTransition", Test_Constant_Velocity_Transition'Access));
      Result.Add_Test
         (Kalman_Caller.Create ("Kalman.PosMeasurement", Test_Position_Measurement'Access));

      -- Detection Tests
      Result.Add_Test
         (Detection_Caller.Create ("Detection.Create", Test_Detection_Create'Access));

      -- Track Tests
      Result.Add_Test
         (Track_Caller.Create ("Track.Create", Test_Track_Create'Access));

      -- Initialization Tests
      Result.Add_Test
         (Init_Caller.Create ("Initialize", Test_Initialize'Access));

      return Result;
   end Suite;

end Test_Stone_Soup;
