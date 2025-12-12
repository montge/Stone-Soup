// Package stonesoup provides Go bindings for the Stone Soup tracking framework.
//
// Stone Soup is a framework for target tracking and state estimation.
// These bindings provide pure Go implementations of core algorithms
// with optional cgo bindings for the native C library.
//
// Example usage:
//
//	import "github.com/dstl/stonesoup"
//
//	func main() {
//	    // Create initial state
//	    state := stonesoup.NewStateVector([]float64{0.0, 1.0, 0.0, 1.0})
//	    covar := stonesoup.IdentityMatrix(4)
//	    prior, _ := stonesoup.NewGaussianState(state, covar)
//
//	    // Predict
//	    F := stonesoup.ConstantVelocityTransition(2, 1.0) // 2D, dt=1
//	    Q := stonesoup.DiagonalMatrix([]float64{0.01, 0.1, 0.01, 0.1})
//	    predicted, _ := stonesoup.KalmanPredict(prior, F, Q)
//
//	    // Update with measurement
//	    z := stonesoup.NewStateVector([]float64{1.0, 1.0})
//	    H := stonesoup.PositionMeasurement(2) // Observe position only
//	    R := stonesoup.DiagonalMatrix([]float64{0.5, 0.5})
//	    posterior, _ := stonesoup.KalmanUpdate(predicted, z, H, R)
//	}
package stonesoup

import (
	"errors"
	"fmt"
	"math"
)

// Error types
var (
	ErrDimensionMismatch = errors.New("stonesoup: dimension mismatch")
	ErrSingularMatrix    = errors.New("stonesoup: singular matrix")
	ErrInvalidParam      = errors.New("stonesoup: invalid parameter")
	ErrNullPointer       = errors.New("stonesoup: null pointer")
)

// Version returns the library version string.
func Version() string {
	return "0.1.0"
}

// StateVector represents a state vector in n-dimensional space.
type StateVector struct {
	data []float64
}

// NewStateVector creates a new state vector.
func NewStateVector(data []float64) *StateVector {
	copied := make([]float64, len(data))
	copy(copied, data)
	return &StateVector{data: copied}
}

// Zeros creates a zero state vector of given dimension.
func Zeros(dim int) *StateVector {
	return &StateVector{data: make([]float64, dim)}
}

// Dims returns the dimensionality of the state vector.
func (sv *StateVector) Dims() int {
	return len(sv.data)
}

// Get returns the value at index i.
func (sv *StateVector) Get(i int) float64 {
	if i < 0 || i >= len(sv.data) {
		return 0.0
	}
	return sv.data[i]
}

// Set sets the value at index i.
func (sv *StateVector) Set(i int, val float64) {
	if i >= 0 && i < len(sv.data) {
		sv.data[i] = val
	}
}

// Data returns a copy of the underlying data.
func (sv *StateVector) Data() []float64 {
	result := make([]float64, len(sv.data))
	copy(result, sv.data)
	return result
}

// Norm returns the Euclidean norm of the state vector.
func (sv *StateVector) Norm() float64 {
	sum := 0.0
	for _, v := range sv.data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// Add returns a new state vector that is the sum of two vectors.
func (sv *StateVector) Add(other *StateVector) (*StateVector, error) {
	if sv.Dims() != other.Dims() {
		return nil, ErrDimensionMismatch
	}
	result := make([]float64, sv.Dims())
	for i := range result {
		result[i] = sv.data[i] + other.data[i]
	}
	return &StateVector{data: result}, nil
}

// Sub returns a new state vector that is the difference of two vectors.
func (sv *StateVector) Sub(other *StateVector) (*StateVector, error) {
	if sv.Dims() != other.Dims() {
		return nil, ErrDimensionMismatch
	}
	result := make([]float64, sv.Dims())
	for i := range result {
		result[i] = sv.data[i] - other.data[i]
	}
	return &StateVector{data: result}, nil
}

// Scale returns a new state vector scaled by a factor.
func (sv *StateVector) Scale(factor float64) *StateVector {
	result := make([]float64, sv.Dims())
	for i := range result {
		result[i] = sv.data[i] * factor
	}
	return &StateVector{data: result}
}

// Matrix represents a 2D matrix (for covariance and transition matrices).
type Matrix struct {
	rows int
	cols int
	data []float64 // row-major storage
}

// NewMatrix creates a new matrix with given dimensions.
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

// IdentityMatrix creates an identity matrix of given dimension.
func IdentityMatrix(dim int) *Matrix {
	m := NewMatrix(dim, dim)
	for i := 0; i < dim; i++ {
		m.Set(i, i, 1.0)
	}
	return m
}

// DiagonalMatrix creates a diagonal matrix from given values.
func DiagonalMatrix(diag []float64) *Matrix {
	dim := len(diag)
	m := NewMatrix(dim, dim)
	for i, v := range diag {
		m.Set(i, i, v)
	}
	return m
}

// FromSlice creates a matrix from a 2D slice.
func FromSlice(data [][]float64) (*Matrix, error) {
	if len(data) == 0 {
		return nil, ErrInvalidParam
	}
	rows := len(data)
	cols := len(data[0])
	for _, row := range data {
		if len(row) != cols {
			return nil, ErrInvalidParam
		}
	}

	m := NewMatrix(rows, cols)
	for i, row := range data {
		for j, v := range row {
			m.Set(i, j, v)
		}
	}
	return m, nil
}

// Rows returns the number of rows.
func (m *Matrix) Rows() int { return m.rows }

// Cols returns the number of columns.
func (m *Matrix) Cols() int { return m.cols }

// Get returns the element at (row, col).
func (m *Matrix) Get(row, col int) float64 {
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		return 0.0
	}
	return m.data[row*m.cols+col]
}

// Set sets the element at (row, col).
func (m *Matrix) Set(row, col int, val float64) {
	if row >= 0 && row < m.rows && col >= 0 && col < m.cols {
		m.data[row*m.cols+col] = val
	}
}

// Trace returns the trace of the matrix.
func (m *Matrix) Trace() float64 {
	sum := 0.0
	minDim := m.rows
	if m.cols < minDim {
		minDim = m.cols
	}
	for i := 0; i < minDim; i++ {
		sum += m.Get(i, i)
	}
	return sum
}

// Copy creates a deep copy of the matrix.
func (m *Matrix) Copy() *Matrix {
	result := NewMatrix(m.rows, m.cols)
	copy(result.data, m.data)
	return result
}

// Transpose returns the transpose of the matrix.
func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(j, i, m.Get(i, j))
		}
	}
	return result
}

// Add returns the sum of two matrices.
func (m *Matrix) Add(other *Matrix) (*Matrix, error) {
	if m.rows != other.rows || m.cols != other.cols {
		return nil, ErrDimensionMismatch
	}
	result := NewMatrix(m.rows, m.cols)
	for i := range result.data {
		result.data[i] = m.data[i] + other.data[i]
	}
	return result, nil
}

// Multiply returns the product of two matrices.
func (m *Matrix) Multiply(other *Matrix) (*Matrix, error) {
	if m.cols != other.rows {
		return nil, ErrDimensionMismatch
	}
	result := NewMatrix(m.rows, other.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < other.cols; j++ {
			sum := 0.0
			for k := 0; k < m.cols; k++ {
				sum += m.Get(i, k) * other.Get(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result, nil
}

// MultiplyVec returns the product of a matrix and vector.
func (m *Matrix) MultiplyVec(v *StateVector) (*StateVector, error) {
	if m.cols != v.Dims() {
		return nil, ErrDimensionMismatch
	}
	result := make([]float64, m.rows)
	for i := 0; i < m.rows; i++ {
		sum := 0.0
		for j := 0; j < m.cols; j++ {
			sum += m.Get(i, j) * v.Get(j)
		}
		result[i] = sum
	}
	return &StateVector{data: result}, nil
}

// Inverse returns the inverse of a square matrix (for small matrices).
func (m *Matrix) Inverse() (*Matrix, error) {
	if m.rows != m.cols {
		return nil, ErrDimensionMismatch
	}

	switch m.rows {
	case 1:
		if math.Abs(m.Get(0, 0)) < 1e-10 {
			return nil, ErrSingularMatrix
		}
		result := NewMatrix(1, 1)
		result.Set(0, 0, 1.0/m.Get(0, 0))
		return result, nil

	case 2:
		det := m.Get(0, 0)*m.Get(1, 1) - m.Get(0, 1)*m.Get(1, 0)
		if math.Abs(det) < 1e-10 {
			return nil, ErrSingularMatrix
		}
		result := NewMatrix(2, 2)
		result.Set(0, 0, m.Get(1, 1)/det)
		result.Set(0, 1, -m.Get(0, 1)/det)
		result.Set(1, 0, -m.Get(1, 0)/det)
		result.Set(1, 1, m.Get(0, 0)/det)
		return result, nil

	default:
		return nil, fmt.Errorf("stonesoup: matrix inverse only implemented for 1x1 and 2x2")
	}
}

// GaussianState represents a Gaussian state with mean and covariance.
type GaussianState struct {
	StateVector *StateVector
	Covariance  *Matrix
	Timestamp   float64
}

// NewGaussianState creates a new Gaussian state.
func NewGaussianState(stateVector *StateVector, covariance *Matrix) (*GaussianState, error) {
	n := stateVector.Dims()
	if covariance.Rows() != n || covariance.Cols() != n {
		return nil, ErrDimensionMismatch
	}

	return &GaussianState{
		StateVector: stateVector,
		Covariance:  covariance,
	}, nil
}

// Dim returns the state dimension.
func (gs *GaussianState) Dim() int {
	return gs.StateVector.Dims()
}

// Detection represents a sensor detection.
type Detection struct {
	Measurement *StateVector
	Timestamp   float64
}

// NewDetection creates a new detection.
func NewDetection(measurement []float64, timestamp float64) *Detection {
	return &Detection{
		Measurement: NewStateVector(measurement),
		Timestamp:   timestamp,
	}
}

// Track represents a target track over time.
type Track struct {
	ID     string
	States []*GaussianState
}

// NewTrack creates a new track.
func NewTrack(id string) *Track {
	return &Track{
		ID:     id,
		States: make([]*GaussianState, 0),
	}
}

// Length returns the number of states in the track.
func (t *Track) Length() int {
	return len(t.States)
}

// AddState appends a state to the track.
func (t *Track) AddState(state *GaussianState) {
	t.States = append(t.States, state)
}

// Latest returns the most recent state.
func (t *Track) Latest() *GaussianState {
	if len(t.States) == 0 {
		return nil
	}
	return t.States[len(t.States)-1]
}

// KalmanPredict performs a Kalman filter prediction step.
//
// Computes:
//   - x_pred = F * x
//   - P_pred = F * P * F^T + Q
func KalmanPredict(prior *GaussianState, transitionMatrix *Matrix, processNoise *Matrix) (*GaussianState, error) {
	dim := prior.Dim()

	if transitionMatrix.Rows() != dim || transitionMatrix.Cols() != dim {
		return nil, ErrDimensionMismatch
	}
	if processNoise.Rows() != dim || processNoise.Cols() != dim {
		return nil, ErrDimensionMismatch
	}

	// x_pred = F * x
	xPred, err := transitionMatrix.MultiplyVec(prior.StateVector)
	if err != nil {
		return nil, err
	}

	// P_pred = F * P * F^T + Q
	fp, err := transitionMatrix.Multiply(prior.Covariance)
	if err != nil {
		return nil, err
	}
	ft := transitionMatrix.Transpose()
	fpft, err := fp.Multiply(ft)
	if err != nil {
		return nil, err
	}
	pPred, err := fpft.Add(processNoise)
	if err != nil {
		return nil, err
	}

	return &GaussianState{
		StateVector: xPred,
		Covariance:  pPred,
		Timestamp:   prior.Timestamp,
	}, nil
}

// KalmanUpdate performs a Kalman filter update step.
//
// Computes:
//   - y = z - H * x (innovation)
//   - S = H * P * H^T + R (innovation covariance)
//   - K = P * H^T * S^-1 (Kalman gain)
//   - x_post = x + K * y
//   - P_post = (I - K * H) * P
func KalmanUpdate(predicted *GaussianState, measurement *StateVector, measurementMatrix *Matrix, measurementNoise *Matrix) (*GaussianState, error) {
	stateDim := predicted.Dim()
	measDim := measurement.Dims()

	if measurementMatrix.Rows() != measDim || measurementMatrix.Cols() != stateDim {
		return nil, ErrDimensionMismatch
	}
	if measurementNoise.Rows() != measDim || measurementNoise.Cols() != measDim {
		return nil, ErrDimensionMismatch
	}

	// y = z - H * x (innovation)
	hx, err := measurementMatrix.MultiplyVec(predicted.StateVector)
	if err != nil {
		return nil, err
	}
	y, err := measurement.Sub(hx)
	if err != nil {
		return nil, err
	}

	// S = H * P * H^T + R
	hp, err := measurementMatrix.Multiply(predicted.Covariance)
	if err != nil {
		return nil, err
	}
	ht := measurementMatrix.Transpose()
	hpht, err := hp.Multiply(ht)
	if err != nil {
		return nil, err
	}
	s, err := hpht.Add(measurementNoise)
	if err != nil {
		return nil, err
	}

	// K = P * H^T * S^-1
	sInv, err := s.Inverse()
	if err != nil {
		return nil, err
	}
	pht, err := predicted.Covariance.Multiply(ht)
	if err != nil {
		return nil, err
	}
	k, err := pht.Multiply(sInv)
	if err != nil {
		return nil, err
	}

	// x_post = x + K * y
	ky, err := k.MultiplyVec(y)
	if err != nil {
		return nil, err
	}
	xPost, err := predicted.StateVector.Add(ky)
	if err != nil {
		return nil, err
	}

	// P_post = (I - K * H) * P
	kh, err := k.Multiply(measurementMatrix)
	if err != nil {
		return nil, err
	}
	identity := IdentityMatrix(stateDim)
	for i := 0; i < stateDim; i++ {
		for j := 0; j < stateDim; j++ {
			identity.Set(i, j, identity.Get(i, j)-kh.Get(i, j))
		}
	}
	pPost, err := identity.Multiply(predicted.Covariance)
	if err != nil {
		return nil, err
	}

	return &GaussianState{
		StateVector: xPost,
		Covariance:  pPost,
		Timestamp:   predicted.Timestamp,
	}, nil
}

// ConstantVelocityTransition creates a constant velocity transition matrix.
// For a 2D system (x, vx, y, vy), use ndim=2.
func ConstantVelocityTransition(ndim int, dt float64) *Matrix {
	stateDim := ndim * 2
	m := IdentityMatrix(stateDim)
	for i := 0; i < ndim; i++ {
		m.Set(i*2, i*2+1, dt)
	}
	return m
}

// PositionMeasurement creates a measurement matrix that observes position only.
// For 2D, this observes x and y from state [x, vx, y, vy].
func PositionMeasurement(ndim int) *Matrix {
	stateDim := ndim * 2
	m := NewMatrix(ndim, stateDim)
	for i := 0; i < ndim; i++ {
		m.Set(i, i*2, 1.0)
	}
	return m
}
