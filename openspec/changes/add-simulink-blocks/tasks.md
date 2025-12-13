# Implementation Tasks

## 1. MATLAB MEX Interface
- [x] 1.1 Create MEX gateway file (`stonesoup_mex.c`)
- [x] 1.2 Implement `stonesoup_mex_state_vector` functions
- [x] 1.3 Implement `stonesoup_mex_gaussian_state` functions
- [x] 1.4 Implement `stonesoup_mex_kalman_predict` function
- [x] 1.5 Implement `stonesoup_mex_kalman_update` function
- [x] 1.6 Create build script (`make.m`) for MEX compilation

## 2. MATLAB Classes
- [x] 2.1 Implement `StateVector.m` class
- [x] 2.2 Implement `GaussianState.m` class
- [x] 2.3 Implement `KalmanPredictor.m` class
- [x] 2.4 Implement `KalmanUpdater.m` class
- [x] 2.5 Add initialization and cleanup functions

## 3. Simulink S-Function Blocks
- [x] 3.1 Create `sfun_kalman_predict.m` Level-2 S-function
- [x] 3.2 Create `sfun_kalman_update.m` Level-2 S-function
- [x] 3.3 Create `sfun_state_source.m` for state initialization
- [x] 3.4 Create `sfun_gaussian_display.m` for visualization
- [ ] 3.5 Create Simulink library (`stonesoup_lib.slx`)
- [ ] 3.6 Design block icons and masks

## 4. Xcos Palette (Scilab)
- [x] 4.1 Create Xcos interface functions
- [x] 4.2 Implement Kalman Predictor Xcos block
- [x] 4.3 Implement Kalman Updater Xcos block
- [x] 4.4 Create palette XML definition
- [x] 4.5 Add palette icons

## 5. GNU Octave Compatibility
- [x] 5.1 Create Octave MEX compatibility wrapper
- [ ] 5.2 Test MEX functions with Octave
- [ ] 5.3 Document Octave-specific limitations

## 6. Demo Models
- [x] 6.1 Create constant velocity tracking demo (Simulink)
- [x] 6.2 Create constant velocity tracking demo (Xcos)
- [ ] 6.3 Create multi-target tracking demo
- [ ] 6.4 Add documentation for each demo

## 7. Testing & Documentation
- [x] 7.1 Create MATLAB unit tests
- [ ] 7.2 Create Octave compatibility tests
- [ ] 7.3 Update bindings/README.md
- [ ] 7.4 Create Simulink/Xcos user guide
