# Change: Add Undersea Tracking Domain Support

## Why
Stone Soup currently focuses on terrestrial and space tracking. Undersea tracking (sonar, submarine, underwater vehicles) has fundamentally different propagation characteristics, coordinate systems, and environmental models that are not addressed.

## What Changes
- Add undersea coordinate systems (depth-based, bathymetric-relative)
- Add underwater propagation models (sound velocity profiles, acoustic channels)
- Add pressure-depth conversions with temperature/salinity effects
- Add sonar-specific measurement models (bearing-only, time-delay, Doppler)
- Add ocean current and drift models for underwater motion prediction

## Impact
- Affected specs: coordinate-systems (new undersea frames), multi-lang-bindings (Ada/safety-critical underwater)
- New spec may be needed: undersea-tracking (domain-specific requirements)
- Affected code: stonesoup/models/, stonesoup/types/
