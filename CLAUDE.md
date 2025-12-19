<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stone Soup is a Python framework for target tracking and state estimation developed by DSTL UK. It provides a modular, component-based architecture for building tracking systems.

## Development Commands

### Installation
```bash
pip install -e .[dev]  # Install with dev dependencies
pip install -e .[dev,ehm,optuna,orbital,roadnet,architectures]  # All optional extras
```

### Testing
```bash
pytest stonesoup                           # Run all tests
pytest stonesoup/types/tests/test_state.py # Run single test file
pytest stonesoup -k "test_name"            # Run tests matching pattern
pytest --slow stonesoup                    # Include slow tests
pytest --cov stonesoup                     # Run with coverage
```

### Linting
```bash
flake8 stonesoup  # Max line length is 99
```

### Documentation
```bash
cd docs && make html  # Build docs to docs/build/html/
```

## Architecture

### Base Classes and Property System

Stone Soup uses a declarative property system defined in `stonesoup/base.py`. All components inherit from `Base` and use `Property` for declaring attributes:

```python
from stonesoup.base import Base, Property

class MyComponent(Base):
    param: int = Property(doc="A parameter")
    optional_param: str = Property(default="value", doc="Optional with default")
    mutable_param: list = Property(default_factory=list, doc="Use default_factory for mutables")
```

The `Property` class auto-generates `__init__` signatures and docstrings. Key features:
- `default_factory` for mutable defaults (lists, dicts)
- `readonly=True` for computed properties
- Property decorators: `.getter()`, `.setter()`, `.deleter()`

### Type System

`stonesoup/types/` contains data structures. All types inherit from `Type` (which inherits from `Base`):
- `State`, `GaussianState`, `ParticleState` - state representations
- `Detection`, `Track`, `GroundTruth` - tracking entities
- `StateVector`, `CovarianceMatrix` - array types (subclass numpy arrays)

### Core Component Hierarchy

Each component type has a base class in `stonesoup/<component>/base.py`:

- **Models** (`models/`): Mathematical models
  - `TransitionModel` - state evolution (e.g., constant velocity)
  - `MeasurementModel` - sensor observations
  - `ControlModel` - external inputs

- **Predictor** (`predictor/`): Predicts states forward using transition models
  - Key method: `predict(prior, timestamp=None, **kwargs)`

- **Updater** (`updater/`): Updates predictions with measurements using measurement models
  - Key methods: `predict_measurement()`, `update(hypothesis)`

- **Hypothesiser** (`hypothesiser/`): Generates hypotheses linking predictions to detections
  - Key method: `hypothesise(track, detections, timestamp)`

- **DataAssociator** (`dataassociator/`): Resolves hypotheses to track-detection associations
  - Key method: `associate(tracks, detections, timestamp)`
  - Uses Hypothesiser internally

- **Initiator** (`initiator/`): Creates new tracks from unassociated detections
  - Key method: `initiate(detections, timestamp)`

- **Deleter** (`deleter/`): Removes stale/dead tracks
  - Key method: `check_for_deletion(track)`

- **Tracker** (`tracker/`): Orchestrates components into complete tracking systems; iterable yielding `(timestamp, tracks)`

### Component Patterns

Components are designed to be interchangeable. A tracker typically composes:
```
Detector → Tracker(predictor, updater, hypothesiser, associator, initiator, deleter) → Tracks
```

Each subpackage follows the pattern:
- `base.py` - abstract base class
- Implementation files (e.g., `kalman.py`, `particle.py`)
- `tests/` - pytest tests

### Additional Components

- **Detector** (`detector/`): Processes sensor data to generate detections
- **Sensor** (`sensor/`): Sensor models and simulations
- **Reader** (`reader/`): Reads detection and ground truth data from files
- **Feeder** (`feeder/`): Data preprocessing and feeding
- **Smoother** (`smoother/`): Post-processing state smoothing
- **Architecture** (`architecture/`): Multi-sensor fusion and distributed tracking

## Python Version

Requires Python 3.10+. Tested on 3.10-3.14.

## Multi-Language SDK Development

Stone Soup includes a C library (`libstonesoup`) and bindings for multiple languages.

### C Library (libstonesoup)

```bash
# Build the C library
cd libstonesoup
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# Run C tests
ctest --output-on-failure

# Static analysis (requires cppcheck)
cmake .. -DENABLE_CPPCHECK=ON
make cppcheck
```

### Rust Bindings

```bash
cd bindings/rust
cargo build
cargo test
cargo clippy  # Linting
```

### Python PyO3 Bindings

```bash
cd bindings/python
pip install maturin
maturin develop  # Build and install locally
pytest tests/
```

### Java Bindings

```bash
cd bindings/java
mvn compile
mvn test
```

### Ada Bindings

```bash
cd bindings/ada
gprbuild -P stonesoup.gpr
```

### Go Bindings

```bash
cd bindings/go
go build ./...
go test ./...
```

### Node.js Bindings

```bash
cd bindings/nodejs
npm install
npm run build
npm test
```
