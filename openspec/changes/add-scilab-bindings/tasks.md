# Implementation Tasks

## 1. Project Setup
- [x] 1.1 Create `bindings/scilab/` directory structure
- [x] 1.2 Create Scilab module descriptor (`etc/stonesoup.start`, `etc/stonesoup.quit`)
- [x] 1.3 Create build configuration for gateway functions
- [ ] 1.4 Add Scilab to CI workflow (conditional on Scilab availability)

## 2. Gateway Interface
- [x] 2.1 Create gateway C code (`sci_gateway/` directory)
- [x] 2.2 Implement `sci_stonesoup_state_vector_create` gateway function
- [x] 2.3 Implement `sci_stonesoup_state_vector_add` gateway function
- [x] 2.4 Implement `sci_stonesoup_kalman_predict` gateway function
- [x] 2.5 Implement `sci_stonesoup_kalman_update` gateway function
- [x] 2.6 Create loader script (`loader.sce`)

## 3. Scilab API Layer
- [x] 3.1 Create `macros/` directory for Scilab wrapper functions
- [x] 3.2 Implement `StateVector` Scilab type with constructor and methods
- [x] 3.3 Implement `GaussianState` Scilab type
- [x] 3.4 Implement `kalman_predict` and `kalman_update` high-level functions
- [x] 3.5 Add input validation and error handling

## 4. Xcos Palette (Optional Phase)
- [ ] 4.1 Create `xcos/` directory structure
- [ ] 4.2 Implement Kalman Predictor block (S-function style)
- [ ] 4.3 Implement Kalman Updater block
- [ ] 4.4 Create palette XML definition
- [ ] 4.5 Add demo Xcos models

## 5. ATOMS Package
- [x] 5.1 Create `DESCRIPTION` file for ATOMS
- [x] 5.2 Create `DESCRIPTION-FUNCTIONS` file
- [ ] 5.3 Create installation script
- [ ] 5.4 Create uninstallation script
- [ ] 5.5 Test local ATOMS installation

## 6. Testing
- [x] 6.1 Create unit tests for gateway functions
- [x] 6.2 Create integration tests for Scilab API
- [x] 6.3 Create example scripts demonstrating usage
- [ ] 6.4 Verify compatibility with Scilab 6.x and 2024.x

## 7. Documentation
- [ ] 7.1 Add help files for each public function
- [x] 7.2 Update `bindings/README.md` with Scilab section
- [ ] 7.3 Create Scilab-specific getting started guide
