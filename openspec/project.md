# Project Context

## Purpose
Stone Soup is a Python framework for target tracking and state estimation developed by DSTL UK. It provides a modular, component-based architecture for building tracking systems used in defense, aerospace, and autonomous systems applications.

## Tech Stack
- Python 3.10+ (primary implementation)
- NumPy/SciPy (numerical computation)
- Matplotlib/Plotly (visualization)
- Sphinx (documentation)
- pytest (testing)

## Project Conventions

### Code Style
- Max line length: 99 characters
- Linting: flake8 (current), transitioning to ruff + black
- All components inherit from `Base` class using `Property` declarative system
- Type annotations required for all public APIs

### Architecture Patterns
- Component-based design with abstract base classes
- Each component type: `base.py` + implementations (kalman.py, particle.py, etc.)
- Interchangeable components via common interfaces
- Tracker orchestrates: Predictor → Updater → Hypothesiser → DataAssociator → Initiator → Deleter

### Testing Strategy
- pytest for all testing (unit, integration)
- Target: 90%+ overall coverage, 80%+ branch/function coverage
- `--slow` flag for expensive tests
- Tests in `<component>/tests/` directories

### Git Workflow
- Main branch: `main`
- Feature branches for changes
- PR-based review process
- CI via CircleCI (lint, test across Python 3.10-3.14, docs)

## Domain Context
- Target tracking: estimating object states (position, velocity) from noisy sensor measurements
- Kalman filters: optimal estimation for linear Gaussian systems
- Particle filters: Monte Carlo methods for nonlinear/non-Gaussian systems
- Data association: linking detections to existing tracks
- Multi-target tracking: managing multiple simultaneous targets

## Important Constraints
- Safety-critical applications require formal requirements traceability
- Multi-language bindings must maintain numerical precision
- MISRA compliance required for C/C++ components
- Ada bindings must support SPARK subset for formal verification
- Memory safety critical for real-time embedded deployments

## External Dependencies
- NumPy/SciPy for numerical operations
- Matplotlib/Plotly for visualization
- rtree for spatial indexing
- utm/pymap3d for coordinate transformations
- ruamel.yaml for configuration serialization
