-------------------------------------------------------------------------------
-- Stone Soup Ada Bindings Body
--
-- This file implements the Ada bindings to the Stone Soup tracking framework.
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with Ada.Numerics.Long_Elementary_Functions;

package body Stone_Soup
  with SPARK_Mode => Off  -- Matches spec setting
is
   use Ada.Numerics.Long_Elementary_Functions;

   -- Library state
   Library_Initialized : Boolean := False;

   ---------------------------------------------------------------------------
   -- State Vector Implementation
   ---------------------------------------------------------------------------

   function Zeros (Dim : Dimension_Range) return State_Vector is
   begin
      return (Dim => Dim, Data => (others => 0.0));
   end Zeros;

   function Fill (Dim : Dimension_Range; Value : Long_Float) return State_Vector is
   begin
      return (Dim => Dim, Data => (others => Value));
   end Fill;

   function Get_Dim (SV : State_Vector) return Dimension_Range is
   begin
      return SV.Dim;
   end Get_Dim;

   function Get (SV : State_Vector; Index : Dimension_Range) return Long_Float is
   begin
      return SV.Data (Index);
   end Get;

   procedure Set
     (SV    : in Out State_Vector;
      Index : Dimension_Range;
      Value : Long_Float) is
   begin
      SV.Data (Index) := Value;
   end Set;

   function Norm (SV : State_Vector) return Long_Float is
      Sum : Long_Float := 0.0;
   begin
      for I in 1 .. SV.Dim loop
         Sum := Sum + SV.Data (I) * SV.Data (I);
      end loop;
      return Sqrt (Sum);
   end Norm;

   function "+" (Left, Right : State_Vector) return State_Vector is
      Result : State_Vector (Left.Dim);
   begin
      for I in 1 .. Left.Dim loop
         Result.Data (I) := Left.Data (I) + Right.Data (I);
      end loop;
      return Result;
   end "+";

   function "-" (Left, Right : State_Vector) return State_Vector is
      Result : State_Vector (Left.Dim);
   begin
      for I in 1 .. Left.Dim loop
         Result.Data (I) := Left.Data (I) - Right.Data (I);
      end loop;
      return Result;
   end "-";

   function "*" (Factor : Long_Float; SV : State_Vector) return State_Vector is
      Result : State_Vector (SV.Dim);
   begin
      for I in 1 .. SV.Dim loop
         Result.Data (I) := Factor * SV.Data (I);
      end loop;
      return Result;
   end "*";

   ---------------------------------------------------------------------------
   -- Covariance Matrix Implementation
   ---------------------------------------------------------------------------

   function Identity (Dim : Dimension_Range) return Covariance_Matrix is
      Result : Covariance_Matrix (Dim);
   begin
      for I in 0 .. Dim - 1 loop
         Result.Data (I, I) := 1.0;
      end loop;
      return Result;
   end Identity;

   function Zero_Matrix (Dim : Dimension_Range) return Covariance_Matrix is
   begin
      return (Dim => Dim, Data => (others => (others => 0.0)));
   end Zero_Matrix;

   function Diagonal (Diag : State_Vector) return Covariance_Matrix is
      Result : Covariance_Matrix (Diag.Dim);
   begin
      for I in 1 .. Diag.Dim loop
         Result.Data (I, I) := Diag.Data (I);
      end loop;
      return Result;
   end Diagonal;

   function Get_Dim (M : Covariance_Matrix) return Dimension_Range is
   begin
      return M.Dim;
   end Get_Dim;

   function Get
     (M   : Covariance_Matrix;
      Row : Dimension_Range;
      Col : Dimension_Range) return Long_Float is
   begin
      return M.Data (Row, Col);
   end Get;

   procedure Set
     (M     : in out Covariance_Matrix;
      Row   : Dimension_Range;
      Col   : Dimension_Range;
      Value : Long_Float) is
   begin
      M.Data (Row, Col) := Value;
   end Set;

   function Trace (M : Covariance_Matrix) return Long_Float is
      Sum : Long_Float := 0.0;
   begin
      for I in 1 .. M.Dim loop
         Sum := Sum + M.Data (I, I);
      end loop;
      return Sum;
   end Trace;

   function "+" (Left, Right : Covariance_Matrix) return Covariance_Matrix is
      Result : Covariance_Matrix (Left.Dim);
   begin
      for I in 1 .. Left.Dim loop
         for J in 1 .. Left.Dim loop
            Result.Data (I, J) := Left.Data (I, J) + Right.Data (I, J);
         end loop;
      end loop;
      return Result;
   end "+";

   function "-" (Left, Right : Covariance_Matrix) return Covariance_Matrix is
      Result : Covariance_Matrix (Left.Dim);
   begin
      for I in 1 .. Left.Dim loop
         for J in 1 .. Left.Dim loop
            Result.Data (I, J) := Left.Data (I, J) - Right.Data (I, J);
         end loop;
      end loop;
      return Result;
   end "-";

   function "*" (Factor : Long_Float; M : Covariance_Matrix) return Covariance_Matrix is
      Result : Covariance_Matrix (M.Dim);
   begin
      for I in 1 .. M.Dim loop
         for J in 1 .. M.Dim loop
            Result.Data (I, J) := Factor * M.Data (I, J);
         end loop;
      end loop;
      return Result;
   end "*";

   function "*" (Left, Right : Covariance_Matrix) return Covariance_Matrix is
      Result : Covariance_Matrix (Left.Dim);
      Sum    : Long_Float;
   begin
      for I in 1 .. Left.Dim loop
         for J in 1 .. Left.Dim loop
            Sum := 0.0;
            for K in 1 .. Left.Dim loop
               Sum := Sum + Left.Data (I, K) * Right.Data (K, J);
            end loop;
            Result.Data (I, J) := Sum;
         end loop;
      end loop;
      return Result;
   end "*";

   function "*" (M : Covariance_Matrix; V : State_Vector) return State_Vector is
      Result : State_Vector (V.Dim);
      Sum    : Long_Float;
   begin
      for I in 1 .. M.Dim loop
         Sum := 0.0;
         for J in 1 .. M.Dim loop
            Sum := Sum + M.Data (I, J) * V.Data (J);
         end loop;
         Result.Data (I) := Sum;
      end loop;
      return Result;
   end "*";

   function Transpose (M : Covariance_Matrix) return Covariance_Matrix is
      Result : Covariance_Matrix (M.Dim);
   begin
      for I in 1 .. M.Dim loop
         for J in 1 .. M.Dim loop
            Result.Data (I, J) := M.Data (J, I);
         end loop;
      end loop;
      return Result;
   end Transpose;

   ---------------------------------------------------------------------------
   -- Gaussian State Implementation
   ---------------------------------------------------------------------------

   function Create_Gaussian_State
     (SV        : State_Vector;
      Covar     : Covariance_Matrix;
      Timestamp : Long_Float := 0.0) return Gaussian_State is
   begin
      return (Dim           => SV.Dim,
              State_Vector  => SV,
              Covariance    => Covar,
              Timestamp     => Timestamp,
              Has_Timestamp => Timestamp /= 0.0);
   end Create_Gaussian_State;

   function Get_State_Vector (GS : Gaussian_State) return State_Vector is
   begin
      return GS.State_Vector;
   end Get_State_Vector;

   function Get_Covariance (GS : Gaussian_State) return Covariance_Matrix is
   begin
      return GS.Covariance;
   end Get_Covariance;

   function Get_State (GS : Gaussian_State; Index : Dimension_Index) return Long_Float is
   begin
      return GS.State_Vector.Data (Index);
   end Get_State;

   function Get_Variance (GS : Gaussian_State; Index : Dimension_Index) return Long_Float is
   begin
      return GS.Covariance.Data (Index, Index);
   end Get_Variance;

   ---------------------------------------------------------------------------
   -- Kalman Filter Implementation
   ---------------------------------------------------------------------------

   -- Helper: Multiply A * B^T
   function Multiply_Transpose
     (A : Covariance_Matrix;
      B : Covariance_Matrix) return Covariance_Matrix
     with Pre => A.Dim = B.Dim
   is
      Result : Covariance_Matrix (A.Dim);
      Sum    : Long_Float;
   begin
      for I in 1 .. A.Dim loop
         for J in 1 .. A.Dim loop
            Sum := 0.0;
            for K in 1 .. A.Dim loop
               Sum := Sum + A.Data (I, K) * B.Data (J, K);
            end loop;
            Result.Data (I, J) := Sum;
         end loop;
      end loop;
      return Result;
   end Multiply_Transpose;

   -- Helper: Matrix inversion using Gaussian elimination
   function Invert (M : Covariance_Matrix) return Covariance_Matrix is
      N      : constant Dimension_Range := M.Dim;
      A      : Covariance_Matrix := M;
      Inv    : Covariance_Matrix := Identity (N);
      Max_Row : Dimension_Index;
      Max_Val : Long_Float;
      Factor  : Long_Float;
      Temp    : Long_Float;
   begin
      for Col in 0 .. N - 1 loop
         -- Find pivot
         Max_Row := Col;
         Max_Val := abs (A.Data (Col, Col));
         for Row in Col + 1 .. N - 1 loop
            if abs (A.Data (Row, Col)) > Max_Val then
               Max_Row := Row;
               Max_Val := abs (A.Data (Row, Col));
            end if;
         end loop;

         -- Swap rows
         if Max_Row /= Col then
            for J in 0 .. N - 1 loop
               Temp := A.Data (Col, J);
               A.Data (Col, J) := A.Data (Max_Row, J);
               A.Data (Max_Row, J) := Temp;

               Temp := Inv.Data (Col, J);
               Inv.Data (Col, J) := Inv.Data (Max_Row, J);
               Inv.Data (Max_Row, J) := Temp;
            end loop;
         end if;

         -- Check for singular matrix
         if abs (A.Data (Col, Col)) < 1.0e-12 then
            raise Singular_Matrix;
         end if;

         -- Scale pivot row
         Factor := A.Data (Col, Col);
         for J in 0 .. N - 1 loop
            A.Data (Col, J) := A.Data (Col, J) / Factor;
            Inv.Data (Col, J) := Inv.Data (Col, J) / Factor;
         end loop;

         -- Eliminate column
         for Row in 0 .. N - 1 loop
            if Row /= Col then
               Factor := A.Data (Row, Col);
               for J in 0 .. N - 1 loop
                  A.Data (Row, J) := A.Data (Row, J) - Factor * A.Data (Col, J);
                  Inv.Data (Row, J) := Inv.Data (Row, J) - Factor * Inv.Data (Col, J);
               end loop;
            end if;
         end loop;
      end loop;

      return Inv;
   end Invert;

   function Kalman_Predict
     (Prior             : Gaussian_State;
      Transition_Matrix : Covariance_Matrix;
      Process_Noise     : Covariance_Matrix) return Gaussian_State
   is
      F      : Covariance_Matrix renames Transition_Matrix;
      Q      : Covariance_Matrix renames Process_Noise;
      X_Pred : State_Vector (Prior.Dim);
      P_Pred : Covariance_Matrix (Prior.Dim);
      FP     : Covariance_Matrix (Prior.Dim);
   begin
      -- x_pred = F * x
      X_Pred := F * Prior.State_Vector;

      -- P_pred = F * P * F^T + Q
      FP := F * Prior.Covariance;
      P_Pred := Multiply_Transpose (FP, F) + Q;

      return (Dim           => Prior.Dim,
              State_Vector  => X_Pred,
              Covariance    => P_Pred,
              Timestamp     => Prior.Timestamp,
              Has_Timestamp => Prior.Has_Timestamp);
   end Kalman_Predict;

   function Kalman_Update
     (Predicted         : Gaussian_State;
      Measurement       : State_Vector;
      Measurement_Matrix : Covariance_Matrix;
      Measurement_Noise : Covariance_Matrix) return Gaussian_State
   is
      H            : Covariance_Matrix renames Measurement_Matrix;
      R            : Covariance_Matrix renames Measurement_Noise;
      X_Pred       : State_Vector renames Predicted.State_Vector;
      P_Pred       : Covariance_Matrix renames Predicted.Covariance;

      Hx           : State_Vector (Measurement.Dim);
      Innovation   : State_Vector (Measurement.Dim);
      HP           : Covariance_Matrix (Predicted.Dim);
      HPHt         : Covariance_Matrix (Measurement.Dim);
      S            : Covariance_Matrix (Measurement.Dim);
      S_Inv        : Covariance_Matrix (Measurement.Dim);
      Ht           : Covariance_Matrix (Predicted.Dim);
      PHt          : Covariance_Matrix (Predicted.Dim);
      K            : Covariance_Matrix (Predicted.Dim);
      Ky           : State_Vector (Predicted.Dim);
      X_Post       : State_Vector (Predicted.Dim);
      KH           : Covariance_Matrix (Predicted.Dim);
      I_KH         : Covariance_Matrix (Predicted.Dim);
      P_Post       : Covariance_Matrix (Predicted.Dim);
      I            : Covariance_Matrix (Predicted.Dim);
   begin
      -- y = z - H * x_pred
      Hx := H * X_Pred;
      Innovation := Measurement - Hx;

      -- S = H * P * H^T + R
      HP := H * P_Pred;
      HPHt := Multiply_Transpose (HP, H);
      S := HPHt + R;

      -- K = P * H^T * S^-1
      S_Inv := Invert (S);
      Ht := Transpose (H);
      PHt := P_Pred * Ht;
      K := PHt * S_Inv;

      -- x_post = x_pred + K * y
      Ky := K * Innovation;
      X_Post := X_Pred + Ky;

      -- P_post = (I - K * H) * P_pred
      I := Identity (Predicted.Dim);
      KH := K * H;
      I_KH := I - KH;
      P_Post := I_KH * P_Pred;

      return (Dim           => Predicted.Dim,
              State_Vector  => X_Post,
              Covariance    => P_Post,
              Timestamp     => Predicted.Timestamp,
              Has_Timestamp => Predicted.Has_Timestamp);
   end Kalman_Update;

   function Constant_Velocity_Transition
     (Spatial_Dims : Positive;
      Dt           : Long_Float) return Covariance_Matrix
   is
      State_Dim : constant Dimension_Range := Spatial_Dims * 2;
      Result    : Covariance_Matrix (State_Dim);
   begin
      for I in 0 .. Spatial_Dims - 1 loop
         declare
            Pos_Idx : constant Dimension_Index := Dimension_Index (I * 2);
            Vel_Idx : constant Dimension_Index := Dimension_Index (I * 2 + 1);
         begin
            Result.Data (Pos_Idx, Pos_Idx) := 1.0;
            Result.Data (Pos_Idx, Vel_Idx) := Dt;
            Result.Data (Vel_Idx, Vel_Idx) := 1.0;
         end;
      end loop;
      return Result;
   end Constant_Velocity_Transition;

   function Position_Measurement (Spatial_Dims : Positive) return Covariance_Matrix
   is
      State_Dim : constant Dimension_Range := Spatial_Dims * 2;
      Result    : Covariance_Matrix (State_Dim);
   begin
      for I in 0 .. Spatial_Dims - 1 loop
         declare
            Meas_Idx : constant Dimension_Index := Dimension_Index (I);
            Pos_Idx  : constant Dimension_Index := Dimension_Index (I * 2);
         begin
            Result.Data (Meas_Idx, Pos_Idx) := 1.0;
         end;
      end loop;
      return Result;
   end Position_Measurement;

   ---------------------------------------------------------------------------
   -- Detection Implementation
   ---------------------------------------------------------------------------

   function Create_Detection
     (Measurement : State_Vector;
      Timestamp   : Long_Float) return Detection is
   begin
      return (Dim         => Measurement.Dim,
              Measurement => Measurement,
              Timestamp   => Timestamp);
   end Create_Detection;

   ---------------------------------------------------------------------------
   -- Track Implementation (SPARK_Mode Off due to access types)
   ---------------------------------------------------------------------------

   function Create_Track
     (ID         : String;
      Dim        : Dimension_Range;
      Max_States : Positive := Max_Track_Length) return Track
   is
      Result : Track (Dim, Max_States);
   begin
      -- Copy ID (pad or truncate as needed)
      for I in 1 .. Track_ID'Last loop
         if I <= ID'Length then
            Result.ID (I) := ID (ID'First + I - 1);
         else
            Result.ID (I) := ' ';
         end if;
      end loop;
      return Result;
   end Create_Track;

   function Length (T : Track) return Natural is
   begin
      return T.Num_States;
   end Length;

   function Is_Empty (T : Track) return Boolean is
   begin
      return T.Num_States = 0;
   end Is_Empty;

   ---------------------------------------------------------------------------
   -- Library Initialization
   ---------------------------------------------------------------------------

   procedure Initialize is
   begin
      Library_Initialized := True;
   end Initialize;

   procedure Cleanup is
   begin
      Library_Initialized := False;
   end Cleanup;

   function Is_Initialized return Boolean is
   begin
      return Library_Initialized;
   end Is_Initialized;

end Stone_Soup;
