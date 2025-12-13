## MODIFIED Requirements

### Requirement: MATLAB Bindings
The system SHALL provide MATLAB bindings via MEX interface for integration with MATLAB and Simulink.

#### Scenario: MEX function availability
- **WHEN** MATLAB code calls stonesoup MEX functions
- **THEN** core tracking operations execute via libstonesoup

#### Scenario: MATLAB array compatibility
- **WHEN** MATLAB arrays are passed to stonesoup functions
- **THEN** data is efficiently transferred via MEX API

#### Scenario: Simulink block library
- **WHEN** Simulink model uses Stone Soup blocks
- **THEN** S-function blocks wrap core algorithms for simulation

#### Scenario: MATLAB Coder compatibility
- **WHEN** MATLAB code is compiled with MATLAB Coder
- **THEN** generated C code links against libstonesoup

#### Scenario: Kalman Predictor Simulink block
- **WHEN** Kalman Predictor block receives state and covariance inputs
- **THEN** block outputs predicted state and covariance using transition model

#### Scenario: Kalman Updater Simulink block
- **WHEN** Kalman Updater block receives predicted state and measurement
- **THEN** block outputs posterior state and covariance using measurement model

#### Scenario: Simulink library palette
- **WHEN** user opens Simulink Library Browser
- **THEN** Stone Soup tracking blocks appear in dedicated palette

#### Scenario: Block parameterization
- **WHEN** user configures Simulink block parameters
- **THEN** transition matrix, process noise, and measurement model are configurable via mask

## MODIFIED Requirements

### Requirement: GNU Octave Bindings
The system SHALL provide GNU Octave bindings compatible with MATLAB interface.

#### Scenario: Octave MEX compatibility
- **WHEN** Octave loads stonesoup MEX files
- **THEN** same API as MATLAB is available

#### Scenario: Octave package
- **WHEN** pkg install stonesoup is run in Octave
- **THEN** stonesoup functions are available

#### Scenario: MATLAB script compatibility
- **WHEN** MATLAB scripts using stonesoup run in Octave
- **THEN** scripts execute with same results

#### Scenario: Octave MEX compilation
- **WHEN** mkoctfile compiles stonesoup MEX sources
- **THEN** Octave-compatible MEX files are generated

## MODIFIED Requirements

### Requirement: Scilab Bindings
The system SHALL provide Scilab bindings via gateway interface.

#### Scenario: Scilab gateway functions
- **WHEN** Scilab code calls stonesoup functions
- **THEN** operations execute via C gateway interface

#### Scenario: Xcos palette
- **WHEN** Xcos model uses Stone Soup palette
- **THEN** tracking blocks are available for simulation

#### Scenario: Scilab ATOMS package
- **WHEN** atomsInstall stonesoup is run
- **THEN** stonesoup module is installed

#### Scenario: StateVector creation in Scilab
- **WHEN** Scilab code calls `stonesoup_state_vector_create(dim)`
- **THEN** a state vector of specified dimension is returned as a Scilab column vector

#### Scenario: Kalman filter operations in Scilab
- **WHEN** Scilab code calls `stonesoup_kalman_predict` or `stonesoup_kalman_update`
- **THEN** Kalman filter operations execute via libstonesoup with Scilab-native data types

#### Scenario: Error handling in Scilab
- **WHEN** a gateway function encounters an error
- **THEN** a descriptive Scilab error is raised with error code and message

#### Scenario: Scilab 6.x compatibility
- **WHEN** Scilab bindings are loaded in Scilab 6.0 or later
- **THEN** all gateway functions and macros work correctly

#### Scenario: Xcos Kalman Predictor block
- **WHEN** Xcos model includes Kalman Predictor block
- **THEN** block performs Kalman prediction with configurable parameters

#### Scenario: Xcos Kalman Updater block
- **WHEN** Xcos model includes Kalman Updater block
- **THEN** block performs Kalman update with measurement input
