## Context

Target tracking applications span ground, air, and space domains, each with preferred coordinate systems and estimation approaches. Stone Soup has strong orbital mechanics foundations but lacks:
1. Grid-based (voxel) tracking approaches useful for radar volume search
2. Viterbi algorithms for optimal trajectory estimation
3. Complete coordinate system coverage for space surveillance

### Stakeholders
- Space surveillance operators (need ECI/ECEF/GCRS transformations)
- Air defense systems (need voxel-based volume search)
- Maritime tracking (need geodetic variants and reference ellipsoids)
- Research community (need complete coordinate system toolkit)

### Constraints
- Must maintain numerical precision across transformations
- Must handle time-varying reference frames (ECI↔ECEF)
- Must support both high-precision (IERS conventions) and simplified models
- Must integrate with existing Stone Soup type system and component architecture

## Goals / Non-Goals

### Goals
- Provide native ECEF and comprehensive ECI↔ECEF transformations
- Support multiple reference ellipsoids beyond WGS84
- Implement voxel-based state estimation for grid-based tracking
- Implement Viterbi algorithms for trajectory smoothing
- Support space surveillance coordinate systems (GCRS, J2000, ICRS)
- Support relative motion frames (RIC/RSW) for proximity operations

### Non-Goals
- Full IAU SOFA implementation (use external library if needed)
- Real-time Earth orientation parameter updates (provide interface for EOP data)
- GPU acceleration for voxel operations (future enhancement)
- Full astrodynamics library (remain focused on tracking)

## Decisions

### Coordinate System Architecture
**Decision**: Create a unified `ReferenceFrame` class hierarchy with composable transformations.

**Rationale**:
- Single source of truth for coordinate definitions
- Transformation chains can be composed automatically
- Supports both time-invariant and time-varying transformations

**Design**:
```
ReferenceFrame (ABC)
├── InertialFrame
│   ├── GCRS
│   ├── J2000
│   └── ICRS
├── BodyFixedFrame
│   ├── ECEF (ITRF)
│   ├── MOON_ME
│   └── PlanetaryFrame
├── TopoCentricFrame
│   ├── ENU
│   ├── NED
│   └── SEZ
└── RelativeFrame
    ├── RIC (Radial/In-track/Cross-track)
    └── RSW (alternative naming)
```

### Reference Ellipsoid Abstraction
**Decision**: Create `ReferenceEllipsoid` class supporting multiple geodetic datums.

| Ellipsoid | Semi-major axis (m) | Flattening |
|-----------|-------------------|------------|
| WGS84 | 6378137.0 | 1/298.257223563 |
| GRS80 | 6378137.0 | 1/298.257222101 |
| WGS72 | 6378135.0 | 1/298.26 |
| PZ-90 | 6378136.0 | 1/298.25784 |
| CGCS2000 | 6378137.0 | 1/298.257222101 |

### ECI↔ECEF Transformation Strategy
**Decision**: Implement tiered precision levels.

| Level | Components | Use Case |
|-------|------------|----------|
| Simple | Earth Rotation Angle (ERA) only | Real-time, low precision |
| Standard | ERA + precession + nutation | General tracking |
| High | IAU 2006/2000A + EOP | Space surveillance |

**Implementation**:
- Default to Standard level
- Accept optional EOP data for High precision
- Use IERS Conventions 2010 as reference

### Voxel Grid Architecture
**Decision**: Use adaptive octree-based voxel representation.

**Rationale**:
- Adaptive resolution reduces memory for large volumes
- Octree enables efficient spatial queries
- Compatible with parallel processing

**Design**:
```python
class VoxelGrid(Type):
    bounds: np.ndarray  # [x_min, y_min, z_min, x_max, y_max, z_max]
    resolution: float   # Base voxel size
    max_depth: int      # Maximum octree depth
    occupancy: OctreeNode  # Probability mass per voxel
```

### Viterbi Implementation Strategy
**Decision**: Implement Viterbi as a Smoother component.

**Rationale**:
- Fits Stone Soup's smoother pattern (post-processing)
- Can be combined with any predictor/updater
- Supports batch processing of track hypotheses

**Components**:
1. `ViterbiSmoother` - Classic Viterbi for HMM state sequence
2. `ViterbiTrackInitiator` - Track-before-detect using Viterbi
3. `GraphViterbiSmoother` - Road/rail constrained tracking

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Numerical precision loss in coordinate chains | High | Unit tests with IERS standard test cases |
| Voxel memory consumption | Medium | Adaptive octree, lazy allocation |
| Viterbi complexity for large state spaces | Medium | Pruning strategies, max hypothesis limits |
| EOP data dependency for high-precision transforms | Low | Default to analytic nutation, optional EOP |

## Migration Plan

### Phase 1: Coordinate Systems Foundation
1. Implement `ReferenceEllipsoid` class
2. Implement `ReferenceFrame` hierarchy
3. Implement ECEF type and transformations
4. Add ECI↔ECEF with ERA (simple level)

### Phase 2: Enhanced Coordinate Systems
1. Implement standard-level ECI↔ECEF (precession/nutation)
2. Add GCRS, J2000, ICRS frames
3. Add RIC/RSW relative frames
4. Add topocentric frames (SEZ)

### Phase 3: Voxel Tracking
1. Implement `VoxelState` type
2. Implement `VoxelGrid` with octree
3. Implement `VoxelPredictor`
4. Implement `VoxelUpdater`

### Phase 4: Viterbi Algorithms
1. Implement `ViterbiSmoother`
2. Implement `ViterbiTrackInitiator`
3. Implement `GraphViterbiSmoother`
4. Integration tests with tracking scenarios

## Open Questions

1. **EOP data source**: Should Stone Soup include EOP data fetching, or require user to provide?
2. **Voxel visualization**: Should voxels be renderable via existing `Plotter` or require specialized viewer?
3. **Lunar/planetary frames**: Include SPICE kernel support or limit to Earth-centric systems?
4. **Viterbi state space**: Continuous states discretized to grid, or discrete mode-based states?
5. **Performance targets**: What voxel grid sizes need to be supported in real-time?
