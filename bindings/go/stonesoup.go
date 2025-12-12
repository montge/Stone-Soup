// Package stonesoup provides Go bindings for the Stone Soup tracking framework.
//
// Stone Soup is a framework for target tracking and state estimation.
// These bindings use cgo to interface with the native C library.
//
// Example usage:
//
//	import "github.com/dstl/stonesoup"
//
//	func main() {
//	    if err := stonesoup.Initialize(); err != nil {
//	        log.Fatal(err)
//	    }
//	    defer stonesoup.Cleanup()
//
//	    // Use Stone Soup functionality
//	}
package stonesoup

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../lib -lstonesoup

// Placeholder C declarations - will be updated when C API is available
// #include <stdlib.h>
//
// int stonesoup_init(void);
// int stonesoup_cleanup(void);
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Error types
var (
	ErrInitialization = errors.New("stonesoup: initialization failed")
	ErrCleanup        = errors.New("stonesoup: cleanup failed")
	ErrNullPointer    = errors.New("stonesoup: null pointer")
	ErrInvalidParam   = errors.New("stonesoup: invalid parameter")
)

// Version returns the library version string.
func Version() string {
	return "0.1.0"
}

// Initialize initializes the Stone Soup library.
// Must be called before using any other functions.
func Initialize() error {
	// Placeholder - will be implemented when C API is ready
	// result := C.stonesoup_init()
	// if result != 0 {
	//     return ErrInitialization
	// }
	return nil
}

// Cleanup cleans up Stone Soup resources.
// Should be called when done using the library.
func Cleanup() error {
	// Placeholder - will be implemented when C API is ready
	// result := C.stonesoup_cleanup()
	// if result != 0 {
	//     return ErrCleanup
	// }
	return nil
}

// StateVector represents a state vector in n-dimensional space.
type StateVector struct {
	data []float64
}

// NewStateVector creates a new state vector.
func NewStateVector(data []float64) *StateVector {
	return &StateVector{
		data: data,
	}
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

// GaussianState represents a Gaussian state with mean and covariance.
type GaussianState struct {
	StateVector *StateVector
	Covariance  [][]float64
}

// NewGaussianState creates a new Gaussian state.
func NewGaussianState(stateVector *StateVector, covariance [][]float64) (*GaussianState, error) {
	n := stateVector.Dims()
	if len(covariance) != n {
		return nil, ErrInvalidParam
	}
	for i := range covariance {
		if len(covariance[i]) != n {
			return nil, ErrInvalidParam
		}
	}

	return &GaussianState{
		StateVector: stateVector,
		Covariance:  covariance,
	}, nil
}

// Detection represents a sensor detection.
type Detection struct {
	Measurement []float64
	Timestamp   float64
}

// NewDetection creates a new detection.
func NewDetection(measurement []float64, timestamp float64) *Detection {
	return &Detection{
		Measurement: measurement,
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

// Helper function to free C memory (used by cgo)
func freeCMemory(ptr unsafe.Pointer) {
	if ptr != nil {
		C.free(ptr)
	}
}
