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
