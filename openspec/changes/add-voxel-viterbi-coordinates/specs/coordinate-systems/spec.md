## ADDED Requirements

### Requirement: Reference Ellipsoid Support
The system SHALL support multiple geodetic reference ellipsoids for coordinate transformations.

#### Scenario: WGS84 ellipsoid
- **WHEN** WGS84 ellipsoid is used
- **THEN** correct semi-major axis (6378137.0m) and flattening (1/298.257223563) are applied

#### Scenario: WGS84 realizations
- **WHEN** specific WGS84 realization is selected (G730, G873, G1150, G1674, G1762, G2139)
- **THEN** corresponding parameters are used

#### Scenario: Alternative ellipsoids
- **WHEN** GRS80, WGS72, PZ-90, or CGCS2000 ellipsoid is selected
- **THEN** correct ellipsoid parameters are applied

#### Scenario: Custom ellipsoid
- **WHEN** custom semi-major axis and flattening are provided
- **THEN** custom ellipsoid is created for transformations

### Requirement: Native ECEF Support
The system SHALL provide native ECEF (Earth-Centered Earth-Fixed) coordinate support without external dependencies.

#### Scenario: Geodetic to ECEF
- **WHEN** longitude, latitude, altitude are converted to ECEF
- **THEN** X, Y, Z coordinates are computed using specified ellipsoid

#### Scenario: ECEF to geodetic
- **WHEN** ECEF X, Y, Z are converted to geodetic
- **THEN** longitude, latitude, altitude are computed iteratively or via closed-form

#### Scenario: Velocity transformation
- **WHEN** velocity in geodetic frame is converted to ECEF
- **THEN** rotation from local frame to ECEF is applied

### Requirement: ECI-ECEF Transformation
The system SHALL provide ECI to ECEF transformations with configurable precision levels.

#### Scenario: Simple transformation (ERA only)
- **WHEN** simple precision level is selected
- **THEN** only Earth Rotation Angle is applied

#### Scenario: Standard transformation
- **WHEN** standard precision level is selected
- **THEN** ERA, precession, and nutation are applied per IAU 2006/2000A

#### Scenario: High-precision transformation
- **WHEN** high precision level is selected with EOP data
- **THEN** full IERS Conventions 2010 transformation is applied

#### Scenario: Time-varying rotation
- **WHEN** transformation is computed at different epochs
- **THEN** rotation matrix varies correctly with time

### Requirement: Inertial Reference Frames
The system SHALL support multiple inertial reference frames for space tracking.

#### Scenario: GCRS frame
- **WHEN** GCRS (Geocentric Celestial Reference System) is selected
- **THEN** positions are in geocentric celestial coordinates

#### Scenario: J2000 frame
- **WHEN** J2000 mean equator and equinox is selected
- **THEN** coordinates use J2000.0 epoch reference

#### Scenario: ICRS frame
- **WHEN** ICRS (International Celestial Reference System) is selected
- **THEN** positions are in quasi-inertial barycentric frame

#### Scenario: Frame conversions
- **WHEN** state is converted between inertial frames
- **THEN** frame bias and precession corrections are applied

### Requirement: Topocentric Frames
The system SHALL support topocentric coordinate frames for sensor-centered tracking.

#### Scenario: ENU frame
- **WHEN** ENU (East-North-Up) frame is used
- **THEN** origin is at observer with axes aligned to local horizontal

#### Scenario: NED frame
- **WHEN** NED (North-East-Down) frame is used
- **THEN** origin is at observer with Z pointing toward Earth center

#### Scenario: SEZ frame
- **WHEN** SEZ (South-East-Zenith) frame is used
- **THEN** radar tracking convention coordinates are provided

#### Scenario: Ellipsoid parameterization
- **WHEN** topocentric frame is created
- **THEN** reference ellipsoid affects local frame orientation

### Requirement: Relative Motion Frames
The system SHALL support relative motion frames for proximity operations.

#### Scenario: RIC frame
- **WHEN** RIC (Radial/In-track/Cross-track) frame is used
- **THEN** relative position is expressed in orbital reference frame

#### Scenario: RSW frame
- **WHEN** RSW frame is used
- **THEN** equivalent to RIC with alternative axis naming

#### Scenario: LVLH frame
- **WHEN** LVLH (Local Vertical Local Horizontal) is used
- **THEN** spacecraft body-fixed orientation is computed

#### Scenario: Relative state computation
- **WHEN** two orbital states are provided
- **THEN** relative state in RIC/RSW frame is computed

### Requirement: ITRF Support
The system SHALL support International Terrestrial Reference Frame for high-precision geodesy.

#### Scenario: ITRF realization
- **WHEN** specific ITRF realization (ITRF2014, ITRF2020) is selected
- **THEN** appropriate reference frame parameters are used

#### Scenario: Plate motion
- **WHEN** high-precision ground station coordinates are needed
- **THEN** tectonic plate motion can be accounted for

### Requirement: Coordinate Transformation Chaining
The system SHALL support chained coordinate transformations with automatic path finding.

#### Scenario: Transformation composition
- **WHEN** transformation from Frame A to Frame C is requested
- **THEN** system finds and composes A→B→C transformations

#### Scenario: Inverse transformation
- **WHEN** inverse transformation is requested
- **THEN** reverse transformation is computed correctly

#### Scenario: Jacobian computation
- **WHEN** covariance transformation is needed
- **THEN** Jacobian of coordinate transformation is available

### Requirement: Atmospheric Coordinates
The system SHALL support atmospheric altitude representations.

#### Scenario: Height above ellipsoid
- **WHEN** geometric altitude is specified
- **THEN** height above reference ellipsoid is used

#### Scenario: Geopotential altitude
- **WHEN** geopotential altitude is needed
- **THEN** conversion accounting for gravity variation is applied

#### Scenario: Mean sea level reference
- **WHEN** MSL altitude is specified
- **THEN** geoid undulation correction is applied

### Requirement: Interplanetary Reference Frames
The system SHALL support coordinate frames for interplanetary tracking.

#### Scenario: Heliocentric frame
- **WHEN** heliocentric inertial frame is selected
- **THEN** positions are relative to Sun center with ecliptic or equatorial orientation

#### Scenario: Solar System Barycentric
- **WHEN** BCRS (Barycentric Celestial Reference System) is selected
- **THEN** positions are relative to solar system barycenter

#### Scenario: Earth-Moon Barycentric
- **WHEN** Earth-Moon system tracking is required
- **THEN** Earth-Moon barycentric frame is available

#### Scenario: Planetary inertial frames
- **WHEN** tracking around Mars, Venus, or other planet
- **THEN** planet-centered inertial frame is available

### Requirement: Lunar Coordinate Systems
The system SHALL support coordinate systems for lunar operations.

#### Scenario: MOON_ME frame
- **WHEN** Moon Mean Earth frame is selected
- **THEN** coordinates align with mean Earth direction

#### Scenario: MOON_PA frame
- **WHEN** Moon Principal Axis frame is selected
- **THEN** coordinates align with lunar principal axes

#### Scenario: Lunar body-fixed
- **WHEN** lunar surface coordinates are needed
- **THEN** selenographic latitude/longitude are available

#### Scenario: Lunar orbit tracking
- **WHEN** object orbits the Moon
- **THEN** lunar-centered inertial frame is available

### Requirement: Mars Coordinate Systems
The system SHALL support coordinate systems for Mars operations.

#### Scenario: Mars body-fixed (IAU)
- **WHEN** Mars surface coordinates are needed
- **THEN** areographic latitude/longitude per IAU are available

#### Scenario: Mars inertial frame
- **WHEN** Mars orbital tracking is required
- **THEN** Mars-centered inertial frame is available

#### Scenario: Mars-Sun line frame
- **WHEN** Mars heliocentric position is relevant
- **THEN** Mars-Sun rotating frame is available

### Requirement: Arbitrary Body Coordinate Systems
The system SHALL support user-defined coordinate systems for arbitrary celestial bodies.

#### Scenario: Custom body definition
- **WHEN** new celestial body is defined with mass and shape
- **THEN** body-centered frames are automatically available

#### Scenario: Body-fixed frame
- **WHEN** custom body rotation parameters are provided
- **THEN** body-fixed frame with correct rotation is available

#### Scenario: Sphere of influence
- **WHEN** tracking near arbitrary body
- **THEN** sphere of influence boundary is computed

### Requirement: Transfer Trajectory Frames
The system SHALL support coordinate frames for interplanetary transfer tracking.

#### Scenario: Patched conic transitions
- **WHEN** object crosses sphere of influence boundary
- **THEN** coordinate frame transitions are handled automatically

#### Scenario: Earth-Moon transfer
- **WHEN** tracking Earth-Moon transfer trajectory
- **THEN** appropriate frame transitions occur at lunar SOI

#### Scenario: Earth-Mars transfer
- **WHEN** tracking Earth-Mars transfer trajectory
- **THEN** heliocentric frame is used in interplanetary space

#### Scenario: Three-body dynamics
- **WHEN** CR3BP or ER3BP model is used
- **THEN** rotating frame centered at system barycenter is available

### Requirement: Time Systems
The system SHALL support multiple time systems for interplanetary operations.

#### Scenario: UTC time
- **WHEN** UTC timestamp is provided
- **THEN** conversions to other time systems are available

#### Scenario: TAI time
- **WHEN** TAI (International Atomic Time) is needed
- **THEN** leap-second-free atomic time is used

#### Scenario: TT time
- **WHEN** TT (Terrestrial Time) is needed
- **THEN** proper time at Earth geoid is computed

#### Scenario: TDB time
- **WHEN** TDB (Barycentric Dynamical Time) is needed
- **THEN** ephemeris time for solar system calculations is used

#### Scenario: Mission elapsed time
- **WHEN** mission-specific time is required
- **THEN** configurable epoch for MET is supported

### Requirement: Light-Time Corrections
The system SHALL support light-time corrections for interplanetary tracking.

#### Scenario: One-way light time
- **WHEN** observation is made at distance
- **THEN** light-time delay is computed

#### Scenario: Aberration correction
- **WHEN** stellar aberration is significant
- **THEN** aberration correction is applied

#### Scenario: Relativistic corrections
- **WHEN** high-precision tracking is required
- **THEN** gravitational light bending is considered

### Requirement: Ephemeris Integration
The system SHALL integrate with standard ephemeris sources.

#### Scenario: JPL ephemeris (DE440)
- **WHEN** JPL DE440 ephemeris is available
- **THEN** planetary positions are computed from ephemeris

#### Scenario: SPICE kernel support
- **WHEN** SPICE kernels are provided
- **THEN** positions and orientations are computed via SPICE

#### Scenario: Custom ephemeris
- **WHEN** custom ephemeris data is provided
- **THEN** interpolation provides positions at arbitrary times
