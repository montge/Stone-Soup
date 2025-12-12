## Why

Stone Soup currently supports Cartesian/ECI, Keplerian, TLE, and Equinoctial orbital states, plus geodetic conversions via pymap3d. However, it lacks integrated ECEF support, ECI↔ECEF transformations with Earth rotation, and support for alternative reference ellipsoids. Additionally, there are no voxel-based tracking approaches for grid-based spatial estimation or Viterbi algorithm implementations for optimal path/trajectory estimation. These capabilities are essential for comprehensive tracking from ground to air to space.

## What Changes

### Voxel-Based Tracking
- Add voxel grid data structures for 3D spatial discretization
- Add voxel-based predictor for occupancy prediction
- Add voxel-based updater for grid-based measurement update
- Add voxel↔continuous state conversion utilities
- Support adaptive voxel resolution based on uncertainty

### Viterbi Algorithms
- Add Viterbi decoder for Hidden Markov Model (HMM) based tracking
- Add Viterbi smoother for optimal trajectory estimation
- Add multi-hypothesis Viterbi for track-before-detect scenarios
- Add graph-based Viterbi for road-constrained tracking
- Integration with existing hypothesiser/associator framework

### Coordinate Systems Enhancement
- Add native ECEF support with direct transformations (not just via pymap3d dependency)
- Add ECI↔ECEF transformations with Earth Rotation Angle (ERA), precession, and nutation
- Add WGS84 ellipsoid variants (WGS84-G730, WGS84-G873, WGS84-G1150, WGS84-G1674, WGS84-G1762, WGS84-G2139)
- Add ITRF (International Terrestrial Reference Frame) support
- Add GCRS (Geocentric Celestial Reference System) support for space tracking
- Add J2000 and ICRS inertial reference frames
- Add lunar/planetary coordinate systems (MOON_ME, MOON_PA)
- Add topocentric horizon coordinates (SEZ - South/East/Zenith)
- Add RIC/RSW (Radial/In-track/Cross-track) relative motion frames
- Add atmospheric coordinate systems (height above ellipsoid, geopotential altitude)
- Add reference ellipsoid abstraction supporting GRS80, WGS72, PZ-90, CGCS2000

## Impact

- **Affected specs**: voxel-tracking (new), viterbi-algorithms (new), coordinate-systems (new)
- **Affected code**:
  - New `stonesoup/types/voxel.py` for voxel state types
  - New `stonesoup/predictor/voxel.py` for voxel predictors
  - New `stonesoup/updater/voxel.py` for voxel updaters
  - New `stonesoup/smoother/viterbi.py` for Viterbi smoothing
  - New `stonesoup/hypothesiser/viterbi.py` for Viterbi-based hypothesis generation
  - Enhanced `stonesoup/types/orbitalstate.py` with additional coordinate systems
  - Enhanced `stonesoup/functions/orbital.py` with ECI↔ECEF transformations
  - New `stonesoup/functions/coordinates.py` for unified coordinate transformations
  - New `stonesoup/types/coordinates.py` for reference ellipsoid and frame types
  - Enhanced `stonesoup/feeder/geo.py` with ECEF converters
