package stonesoup

import (
	"math"
	"testing"
)

const epsilon = 1e-10

func TestStateVector(t *testing.T) {
	t.Run("NewStateVector", func(t *testing.T) {
		sv := NewStateVector([]float64{1.0, 2.0, 3.0})
		if sv.Dims() != 3 {
			t.Errorf("Expected dims=3, got %d", sv.Dims())
		}
		if sv.Get(0) != 1.0 {
			t.Errorf("Expected Get(0)=1.0, got %f", sv.Get(0))
		}
	})

	t.Run("Set", func(t *testing.T) {
		sv := Zeros(3)
		sv.Set(1, 5.0)
		if sv.Get(1) != 5.0 {
			t.Errorf("Expected 5.0, got %f", sv.Get(1))
		}
	})

	t.Run("Norm", func(t *testing.T) {
		sv := NewStateVector([]float64{3.0, 4.0})
		if math.Abs(sv.Norm()-5.0) > epsilon {
			t.Errorf("Expected norm=5.0, got %f", sv.Norm())
		}
	})

	t.Run("Add", func(t *testing.T) {
		a := NewStateVector([]float64{1.0, 2.0})
		b := NewStateVector([]float64{3.0, 4.0})
		sum, err := a.Add(b)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		if sum.Get(0) != 4.0 || sum.Get(1) != 6.0 {
			t.Errorf("Expected [4,6], got [%f,%f]", sum.Get(0), sum.Get(1))
		}
	})

	t.Run("Sub", func(t *testing.T) {
		a := NewStateVector([]float64{5.0, 3.0})
		b := NewStateVector([]float64{2.0, 1.0})
		diff, err := a.Sub(b)
		if err != nil {
			t.Fatalf("Sub failed: %v", err)
		}
		if diff.Get(0) != 3.0 || diff.Get(1) != 2.0 {
			t.Errorf("Expected [3,2], got [%f,%f]", diff.Get(0), diff.Get(1))
		}
	})

	t.Run("Scale", func(t *testing.T) {
		sv := NewStateVector([]float64{2.0, 3.0})
		scaled := sv.Scale(2.0)
		if scaled.Get(0) != 4.0 || scaled.Get(1) != 6.0 {
			t.Errorf("Expected [4,6], got [%f,%f]", scaled.Get(0), scaled.Get(1))
		}
	})
}

func TestMatrix(t *testing.T) {
	t.Run("Identity", func(t *testing.T) {
		m := IdentityMatrix(3)
		if m.Trace() != 3.0 {
			t.Errorf("Expected trace=3, got %f", m.Trace())
		}
		if m.Get(0, 0) != 1.0 || m.Get(1, 1) != 1.0 || m.Get(2, 2) != 1.0 {
			t.Error("Identity diagonal should be 1")
		}
		if m.Get(0, 1) != 0.0 || m.Get(1, 0) != 0.0 {
			t.Error("Identity off-diagonal should be 0")
		}
	})

	t.Run("Diagonal", func(t *testing.T) {
		m := DiagonalMatrix([]float64{1.0, 2.0, 3.0})
		if m.Get(0, 0) != 1.0 || m.Get(1, 1) != 2.0 || m.Get(2, 2) != 3.0 {
			t.Error("Diagonal values incorrect")
		}
		if m.Trace() != 6.0 {
			t.Errorf("Expected trace=6, got %f", m.Trace())
		}
	})

	t.Run("Transpose", func(t *testing.T) {
		m := NewMatrix(2, 3)
		m.Set(0, 0, 1.0)
		m.Set(0, 1, 2.0)
		m.Set(0, 2, 3.0)
		m.Set(1, 0, 4.0)
		m.Set(1, 1, 5.0)
		m.Set(1, 2, 6.0)

		mt := m.Transpose()
		if mt.Rows() != 3 || mt.Cols() != 2 {
			t.Errorf("Expected 3x2, got %dx%d", mt.Rows(), mt.Cols())
		}
		if mt.Get(0, 0) != 1.0 || mt.Get(2, 1) != 6.0 {
			t.Error("Transpose values incorrect")
		}
	})

	t.Run("Multiply", func(t *testing.T) {
		a := IdentityMatrix(2)
		b := DiagonalMatrix([]float64{2.0, 3.0})
		c, err := a.Multiply(b)
		if err != nil {
			t.Fatalf("Multiply failed: %v", err)
		}
		if c.Get(0, 0) != 2.0 || c.Get(1, 1) != 3.0 {
			t.Error("I*D should equal D")
		}
	})

	t.Run("Inverse2x2", func(t *testing.T) {
		m := DiagonalMatrix([]float64{2.0, 4.0})
		inv, err := m.Inverse()
		if err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}
		if math.Abs(inv.Get(0, 0)-0.5) > epsilon || math.Abs(inv.Get(1, 1)-0.25) > epsilon {
			t.Errorf("Expected [0.5, 0.25] diagonal, got [%f, %f]", inv.Get(0, 0), inv.Get(1, 1))
		}
	})
}

func TestGaussianState(t *testing.T) {
	t.Run("NewGaussianState", func(t *testing.T) {
		sv := NewStateVector([]float64{1.0, 2.0})
		cov := IdentityMatrix(2)
		gs, err := NewGaussianState(sv, cov)
		if err != nil {
			t.Fatalf("NewGaussianState failed: %v", err)
		}
		if gs.Dim() != 2 {
			t.Errorf("Expected dim=2, got %d", gs.Dim())
		}
	})

	t.Run("DimensionMismatch", func(t *testing.T) {
		sv := NewStateVector([]float64{1.0, 2.0})
		cov := IdentityMatrix(3)
		_, err := NewGaussianState(sv, cov)
		if err == nil {
			t.Error("Expected dimension mismatch error")
		}
	})
}

func TestKalmanPredict(t *testing.T) {
	// 2D position-velocity state: [x, vx]
	sv := NewStateVector([]float64{0.0, 1.0})
	cov := IdentityMatrix(2)
	prior, _ := NewGaussianState(sv, cov)

	// Constant velocity transition with dt=1
	F, _ := FromSlice([][]float64{
		{1.0, 1.0},
		{0.0, 1.0},
	})

	Q := DiagonalMatrix([]float64{0.1, 0.1})

	predicted, err := KalmanPredict(prior, F, Q)
	if err != nil {
		t.Fatalf("KalmanPredict failed: %v", err)
	}

	// x_pred should be [0 + 1*1, 1] = [1, 1]
	if math.Abs(predicted.StateVector.Get(0)-1.0) > epsilon {
		t.Errorf("Expected x=1.0, got %f", predicted.StateVector.Get(0))
	}
	if math.Abs(predicted.StateVector.Get(1)-1.0) > epsilon {
		t.Errorf("Expected vx=1.0, got %f", predicted.StateVector.Get(1))
	}
}

func TestKalmanUpdate(t *testing.T) {
	// Predicted state
	sv := NewStateVector([]float64{1.0, 1.0})
	cov := IdentityMatrix(2)
	predicted, _ := NewGaussianState(sv, cov)

	// Measurement: observe position only
	z := NewStateVector([]float64{1.1})
	H, _ := FromSlice([][]float64{{1.0, 0.0}})
	R := DiagonalMatrix([]float64{0.1})

	posterior, err := KalmanUpdate(predicted, z, H, R)
	if err != nil {
		t.Fatalf("KalmanUpdate failed: %v", err)
	}

	// Position should move towards measurement
	x := posterior.StateVector.Get(0)
	if x <= 1.0 || x >= 1.1 {
		t.Errorf("Expected position between 1.0 and 1.1, got %f", x)
	}
}

func TestConstantVelocityTransition(t *testing.T) {
	F := ConstantVelocityTransition(2, 0.5)
	if F.Rows() != 4 || F.Cols() != 4 {
		t.Errorf("Expected 4x4, got %dx%d", F.Rows(), F.Cols())
	}

	// Check structure: dt should appear at (0,1) and (2,3)
	if math.Abs(F.Get(0, 1)-0.5) > epsilon {
		t.Errorf("Expected F[0,1]=0.5, got %f", F.Get(0, 1))
	}
	if math.Abs(F.Get(2, 3)-0.5) > epsilon {
		t.Errorf("Expected F[2,3]=0.5, got %f", F.Get(2, 3))
	}
}

func TestPositionMeasurement(t *testing.T) {
	H := PositionMeasurement(2)
	if H.Rows() != 2 || H.Cols() != 4 {
		t.Errorf("Expected 2x4, got %dx%d", H.Rows(), H.Cols())
	}

	// Should observe x and y (positions at indices 0 and 2)
	if H.Get(0, 0) != 1.0 {
		t.Error("Expected H[0,0]=1")
	}
	if H.Get(1, 2) != 1.0 {
		t.Error("Expected H[1,2]=1")
	}
	if H.Get(0, 1) != 0.0 || H.Get(1, 3) != 0.0 {
		t.Error("Expected zeros for velocity observations")
	}
}

func TestTrack(t *testing.T) {
	track := NewTrack("test-1")
	if track.Length() != 0 {
		t.Error("New track should be empty")
	}

	sv := NewStateVector([]float64{1.0, 2.0})
	cov := IdentityMatrix(2)
	state, _ := NewGaussianState(sv, cov)

	track.AddState(state)
	if track.Length() != 1 {
		t.Error("Track should have 1 state")
	}

	latest := track.Latest()
	if latest == nil {
		t.Error("Latest should not be nil")
	}
	if latest.StateVector.Get(0) != 1.0 {
		t.Error("Latest state value incorrect")
	}
}

func TestVersion(t *testing.T) {
	v := Version()
	if v != "0.1.0" {
		t.Errorf("Expected version 0.1.0, got %s", v)
	}
}

// Additional tests for improved coverage

func TestStateVectorDimensionErrors(t *testing.T) {
	t.Run("AddDimensionMismatch", func(t *testing.T) {
		a := NewStateVector([]float64{1.0, 2.0})
		b := NewStateVector([]float64{1.0, 2.0, 3.0})
		_, err := a.Add(b)
		if err != ErrDimensionMismatch {
			t.Error("Expected dimension mismatch error")
		}
	})

	t.Run("SubDimensionMismatch", func(t *testing.T) {
		a := NewStateVector([]float64{1.0, 2.0})
		b := NewStateVector([]float64{1.0})
		_, err := a.Sub(b)
		if err != ErrDimensionMismatch {
			t.Error("Expected dimension mismatch error")
		}
	})
}

func TestMatrixCopy(t *testing.T) {
	m := DiagonalMatrix([]float64{1.0, 2.0, 3.0})
	copy := m.Copy()

	// Verify copy has same values
	if copy.Get(0, 0) != 1.0 || copy.Get(1, 1) != 2.0 || copy.Get(2, 2) != 3.0 {
		t.Error("Copy values don't match original")
	}

	// Verify modifying copy doesn't affect original
	copy.Set(0, 0, 99.0)
	if m.Get(0, 0) == 99.0 {
		t.Error("Modifying copy should not affect original")
	}
}

func TestMatrixDimensionErrors(t *testing.T) {
	t.Run("AddDimensionMismatch", func(t *testing.T) {
		a := NewMatrix(2, 2)
		b := NewMatrix(3, 3)
		_, err := a.Add(b)
		if err != ErrDimensionMismatch {
			t.Error("Expected dimension mismatch error")
		}
	})

	t.Run("MultiplyDimensionMismatch", func(t *testing.T) {
		a := NewMatrix(2, 3)
		b := NewMatrix(2, 3)
		_, err := a.Multiply(b)
		if err != ErrDimensionMismatch {
			t.Error("Expected dimension mismatch error")
		}
	})

	t.Run("MultiplyVecDimensionMismatch", func(t *testing.T) {
		m := NewMatrix(2, 3)
		v := NewStateVector([]float64{1.0, 2.0})
		_, err := m.MultiplyVec(v)
		if err != ErrDimensionMismatch {
			t.Error("Expected dimension mismatch error")
		}
	})

	t.Run("InverseSingular", func(t *testing.T) {
		// Zero matrix is singular
		m := NewMatrix(2, 2)
		_, err := m.Inverse()
		if err != ErrSingularMatrix {
			t.Error("Expected singular matrix error")
		}
	})
}

func TestMatrixFromSlice(t *testing.T) {
	t.Run("ValidSlice", func(t *testing.T) {
		m, err := FromSlice([][]float64{
			{1.0, 2.0},
			{3.0, 4.0},
		})
		if err != nil {
			t.Fatalf("FromSlice failed: %v", err)
		}
		if m.Get(0, 0) != 1.0 || m.Get(1, 1) != 4.0 {
			t.Error("Values incorrect")
		}
	})

	t.Run("EmptySlice", func(t *testing.T) {
		_, err := FromSlice([][]float64{})
		if err == nil {
			t.Error("Expected error for empty slice")
		}
	})
}

func TestMatrixBoundsCheck(t *testing.T) {
	m := NewMatrix(2, 2)
	m.Set(0, 0, 1.0)

	// Test Get with invalid indices returns 0 (as per implementation)
	val := m.Get(5, 5)
	if val != 0.0 {
		t.Errorf("Expected 0.0 for out of bounds, got %f", val)
	}
}

func TestMatrixAdd(t *testing.T) {
	a := DiagonalMatrix([]float64{1.0, 2.0})
	b := DiagonalMatrix([]float64{3.0, 4.0})
	sum, err := a.Add(b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if sum.Get(0, 0) != 4.0 || sum.Get(1, 1) != 6.0 {
		t.Error("Sum values incorrect")
	}
}

func TestMatrixMultiplyVec(t *testing.T) {
	m := DiagonalMatrix([]float64{2.0, 3.0})
	v := NewStateVector([]float64{1.0, 1.0})
	result, err := m.MultiplyVec(v)
	if err != nil {
		t.Fatalf("MultiplyVec failed: %v", err)
	}
	if result.Get(0) != 2.0 || result.Get(1) != 3.0 {
		t.Error("MultiplyVec result incorrect")
	}
}

func TestMatrixInverse3x3NotSupported(t *testing.T) {
	// 3x3 inverse is not implemented - should return error
	m := DiagonalMatrix([]float64{2.0, 4.0, 8.0})
	_, err := m.Inverse()
	if err == nil {
		t.Error("Expected error for 3x3 inverse (not implemented)")
	}
}

func TestMatrixInverse1x1(t *testing.T) {
	m := DiagonalMatrix([]float64{4.0})
	inv, err := m.Inverse()
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}
	if math.Abs(inv.Get(0, 0)-0.25) > epsilon {
		t.Errorf("Expected inv[0,0]=0.25, got %f", inv.Get(0, 0))
	}
}

func TestMatrixTraceNonSquare(t *testing.T) {
	m := NewMatrix(2, 3)
	// Trace of non-square matrix should still work (sum of min(rows,cols) diagonal)
	trace := m.Trace()
	if trace != 0.0 {
		t.Errorf("Expected trace=0.0 for zero matrix, got %f", trace)
	}
}

func TestTrackEmpty(t *testing.T) {
	track := NewTrack("empty")
	latest := track.Latest()
	if latest != nil {
		t.Error("Latest of empty track should be nil")
	}
}

func TestGaussianStateCovariance(t *testing.T) {
	sv := NewStateVector([]float64{1.0, 2.0})
	cov := DiagonalMatrix([]float64{0.5, 0.25})
	gs, err := NewGaussianState(sv, cov)
	if err != nil {
		t.Fatalf("NewGaussianState failed: %v", err)
	}

	// Check covariance access
	if math.Abs(gs.Covariance.Get(0, 0)-0.5) > epsilon {
		t.Errorf("Expected cov[0,0]=0.5, got %f", gs.Covariance.Get(0, 0))
	}
}
