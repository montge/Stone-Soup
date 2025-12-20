## ADDED Requirements

### Requirement: Scilab CI Integration
The system SHALL include Scilab bindings in CI with conditional execution.

#### Scenario: Scilab CI job execution
- **WHEN** CI runs and Scilab is available
- **THEN** Scilab gateway functions compile and tests execute

#### Scenario: Scilab unavailable gracefully skipped
- **WHEN** CI runs and Scilab is not installed
- **THEN** Scilab job is skipped without failing the pipeline

#### Scenario: Scilab version matrix
- **WHEN** CI runs Scilab tests
- **THEN** both Scilab 6.x and 2024.x are tested if available

### Requirement: Scilab ATOMS Packaging
The system SHALL provide complete ATOMS package scripts for Scilab distribution.

#### Scenario: ATOMS installation script
- **WHEN** user runs builder.sce
- **THEN** gateway functions compile and module installs to Scilab

#### Scenario: ATOMS uninstallation script
- **WHEN** user runs cleaner.sce
- **THEN** module is cleanly removed from Scilab installation

#### Scenario: ATOMS package validation
- **WHEN** ATOMS package structure is validated
- **THEN** DESCRIPTION, DESCRIPTION-FUNCTIONS, and required files are present

### Requirement: Scilab Help Documentation
The system SHALL provide Scilab help files for all public functions.

#### Scenario: Help file availability
- **WHEN** user types help stonesoup_state_vector_create in Scilab
- **THEN** function documentation is displayed

#### Scenario: Help file format
- **WHEN** help files are built
- **THEN** XML help files follow Scilab help format conventions

### Requirement: Simulink Block Library
The system SHALL provide a Simulink library file containing all Stone Soup blocks.

#### Scenario: Simulink library file
- **WHEN** user opens stonesoup_lib.slx in Simulink
- **THEN** library browser shows Stone Soup block palette

#### Scenario: Block masks and icons
- **WHEN** blocks are viewed in Simulink
- **THEN** custom icons and parameter masks are displayed

#### Scenario: Block documentation
- **WHEN** user clicks Help on a block
- **THEN** block documentation is displayed

### Requirement: GNU Octave Testing
The system SHALL verify MEX compatibility with GNU Octave.

#### Scenario: Octave MEX execution
- **WHEN** Octave loads stonesoup MEX files
- **THEN** basic operations execute successfully

#### Scenario: Octave test suite
- **WHEN** Octave test suite runs
- **THEN** all compatible functions pass tests

#### Scenario: Octave limitation documentation
- **WHEN** user reads Octave documentation
- **THEN** known limitations and workarounds are documented

### Requirement: Multi-Target Tracking Demos
The system SHALL provide multi-target tracking demonstration models.

#### Scenario: Simulink multi-target demo
- **WHEN** user opens multi-target Simulink demo
- **THEN** demo shows tracking of multiple simultaneous targets

#### Scenario: Xcos multi-target demo
- **WHEN** user opens multi-target Xcos demo
- **THEN** demo shows tracking of multiple simultaneous targets

#### Scenario: Demo documentation
- **WHEN** user reads demo documentation
- **THEN** step-by-step usage instructions are provided

### Requirement: Binding User Guides
The system SHALL provide comprehensive user guides for MATLAB/Simulink and Scilab/Xcos.

#### Scenario: Simulink user guide
- **WHEN** user accesses Simulink documentation
- **THEN** complete guide for using Stone Soup blocks is available

#### Scenario: Xcos user guide
- **WHEN** user accesses Xcos documentation
- **THEN** complete guide for using Stone Soup Xcos palette is available

#### Scenario: Troubleshooting guide
- **WHEN** user encounters common issues
- **THEN** troubleshooting documentation addresses the problem
