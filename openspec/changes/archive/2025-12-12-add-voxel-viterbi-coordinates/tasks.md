## 1. Reference Ellipsoid Foundation

- [x] 1.1 Create stonesoup/types/coordinates.py module
- [x] 1.2 Implement ReferenceEllipsoid base class
- [x] 1.3 Implement WGS84 ellipsoid (all revisions G730-G2139)
- [x] 1.4 Implement GRS80, WGS72, PZ-90, CGCS2000 ellipsoids
- [x] 1.5 Add geodetic↔geocentric latitude conversion
- [x] 1.6 Add unit tests for ellipsoid calculations
- [x] 1.7 Document ellipsoid accuracy and precision characteristics

## 2. Reference Frame Hierarchy

- [x] 2.1 Implement ReferenceFrame abstract base class
- [x] 2.2 Implement InertialFrame base class
- [x] 2.3 Implement BodyFixedFrame base class
- [x] 2.4 Implement TopocentricFrame base class
- [x] 2.5 Implement RelativeFrame base class
- [x] 2.6 Add frame transformation composition system
- [x] 2.7 Add time-varying transformation support
- [x] 2.8 Add unit tests for frame transformations

## 3. ECEF Support

- [x] 3.1 Implement ECEFState type
- [x] 3.2 Implement geodetic↔ECEF transformation (native, not pymap3d)
- [x] 3.3 Implement ECEF↔ENU transformation
- [x] 3.4 Implement ECEF↔NED transformation
- [x] 3.5 Add ECEFToGeodeticConverter feeder
- [x] 3.6 Add GeodeticToECEFConverter feeder
- [x] 3.7 Add unit tests with standard test cases
- [x] 3.8 Benchmark against pymap3d for validation

## 4. ECI↔ECEF Transformations

- [x] 4.1 Implement Earth Rotation Angle (ERA) calculation
- [x] 4.2 Implement simple ECI↔ECEF (ERA only)
- [x] 4.3 Implement IAU 2006 precession model
- [x] 4.4 Implement IAU 2000A nutation model
- [x] 4.5 Implement standard-level ECI↔ECEF (ERA + precession + nutation)
- [x] 4.6 Add Earth Orientation Parameters (EOP) interface
- [x] 4.7 Implement high-precision ECI↔ECEF with EOP
- [x] 4.8 Add unit tests with IERS standard test cases
- [x] 4.9 Document precision levels and use cases

## 5. Additional Inertial Frames

- [x] 5.1 Implement GCRS (Geocentric Celestial Reference System)
- [x] 5.2 Implement J2000 mean equator and equinox
- [x] 5.3 Implement ICRS (International Celestial Reference System)
- [x] 5.4 Add GCRS↔J2000 transformation
- [x] 5.5 Add frame conversion utilities to OrbitalState
- [x] 5.6 Add unit tests for inertial frame conversions

## 6. Relative Motion Frames

- [x] 6.1 Implement RIC (Radial/In-track/Cross-track) frame
- [x] 6.2 Implement RSW (alternative naming) frame
- [x] 6.3 Implement LVLH (Local Vertical Local Horizontal)
- [x] 6.4 Add relative state computation from two orbital states
- [x] 6.5 Add proximity operations utilities
- [x] 6.6 Add unit tests for relative motion calculations

## 7. Topocentric Frames

- [x] 7.1 Implement SEZ (South/East/Zenith) frame
- [x] 7.2 Enhance existing ENU with ellipsoid parameterization
- [x] 7.3 Enhance existing NED with ellipsoid parameterization
- [x] 7.4 Add observer-centered transformations
- [x] 7.5 Add radar/sensor line-of-sight calculations
- [x] 7.6 Add unit tests for topocentric conversions

## 8. Voxel Data Structures

- [x] 8.1 Create stonesoup/types/voxel.py module
- [x] 8.2 Implement VoxelGrid base class
- [x] 8.3 Implement OctreeNode for adaptive grids
- [x] 8.4 Implement VoxelState type
- [x] 8.5 Implement voxel↔continuous state conversion
- [x] 8.6 Add grid bounds and resolution configuration
- [x] 8.7 Add occupancy probability storage
- [x] 8.8 Add unit tests for voxel data structures

## 9. Voxel Predictor

- [x] 9.1 Create stonesoup/predictor/voxel.py module
- [x] 9.2 Implement VoxelPredictor base class
- [x] 9.3 Implement occupancy prediction with motion model
- [x] 9.4 Implement adaptive resolution adjustment
- [x] 9.5 Add birth/death process for new voxels
- [x] 9.6 Add unit tests for voxel prediction

## 10. Voxel Updater

- [x] 10.1 Create stonesoup/updater/voxel.py module
- [x] 10.2 Implement VoxelUpdater base class
- [x] 10.3 Implement measurement likelihood per voxel
- [x] 10.4 Implement occupancy update (Bayes rule)
- [x] 10.5 Implement multi-sensor fusion for voxels
- [x] 10.6 Add unit tests for voxel update

## 11. Viterbi Smoother

- [x] 11.1 Create stonesoup/smoother/viterbi.py module
- [x] 11.2 Implement ViterbiSmoother base class
- [x] 11.3 Implement forward-backward message passing
- [x] 11.4 Implement backtracking for optimal sequence
- [x] 11.5 Implement log-domain computation for numerical stability
- [x] 11.6 Add pruning strategies for large state spaces
- [x] 11.7 Add unit tests for Viterbi smoothing

## 12. Viterbi Track Initiator

- [x] 12.1 Create stonesoup/initiator/viterbi.py module
- [x] 12.2 Implement ViterbiTrackInitiator for track-before-detect
- [x] 12.3 Implement detection-to-state lattice construction
- [x] 12.4 Implement multi-scan Viterbi for track extraction
- [x] 12.5 Integrate with existing initiator interface
- [x] 12.6 Add unit tests for Viterbi initiation

## 13. Graph-Based Viterbi

- [x] 13.1 Implement GraphViterbiSmoother for constrained tracking
- [x] 13.2 Add road network graph support
- [x] 13.3 Add rail network graph support
- [x] 13.4 Implement graph-constrained motion model
- [x] 13.5 Add unit tests for graph-based Viterbi

## 14. Integration and Documentation

- [x] 14.1 Integrate coordinate systems with existing OrbitalState
- [x] 14.2 Update orbital functions with new transformations
- [x] 14.3 Add coordinate system tutorial to documentation
- [x] 14.4 Add voxel tracking tutorial
- [x] 14.5 Add Viterbi tracking tutorial
- [x] 14.6 Update API documentation
- [x] 14.7 Add performance benchmarks for coordinate transformations
