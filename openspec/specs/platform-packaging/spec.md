# platform-packaging Specification

## Purpose
Defines the platform support and packaging requirements for Stone Soup. This includes Windows, Linux, macOS support, POSIX/RTOS compatibility, cross-compilation for embedded targets, and package distribution via PyPI, crates.io, npm, and Maven Central.
## Requirements
### Requirement: Windows Platform Support
The system SHALL support Windows 10, Windows 11, and Windows Server LTS versions.

#### Scenario: Windows 10 compatibility
- **WHEN** SDK is installed on Windows 10 21H2+
- **THEN** all components function correctly

#### Scenario: Windows 11 compatibility
- **WHEN** SDK is installed on Windows 11
- **THEN** all components function correctly with ARM64 support

#### Scenario: Windows Server 2019/2022
- **WHEN** SDK is deployed on Windows Server
- **THEN** server workloads execute correctly

### Requirement: Linux Platform Support
The system SHALL support major Linux distributions in LTS versions.

#### Scenario: Ubuntu LTS support
- **WHEN** SDK is installed on Ubuntu 20.04/22.04/24.04
- **THEN** all components function correctly

#### Scenario: Debian support
- **WHEN** SDK is installed on Debian 11/12
- **THEN** all components function correctly

#### Scenario: RHEL/CentOS support
- **WHEN** SDK is installed on RHEL 8/9
- **THEN** all components function correctly

#### Scenario: Fedora support
- **WHEN** SDK is installed on latest Fedora
- **THEN** all components function correctly

### Requirement: POSIX/RTOS Compatibility
The system SHALL provide compatibility layer for RTOS environments.

#### Scenario: FreeRTOS compatibility
- **WHEN** libstonesoup core is compiled for FreeRTOS
- **THEN** POSIX-like API layer enables functionality

#### Scenario: VxWorks compatibility
- **WHEN** libstonesoup is deployed on VxWorks
- **THEN** real-time scheduling is respected

#### Scenario: QNX compatibility
- **WHEN** libstonesoup is deployed on QNX
- **THEN** POSIX compliance enables full functionality

### Requirement: macOS Platform Support
The system SHALL support macOS versions 12 through 15.

#### Scenario: macOS Monterey (12)
- **WHEN** SDK is installed on macOS 12
- **THEN** all components function correctly

#### Scenario: macOS Sequoia (15)
- **WHEN** SDK is installed on macOS 15
- **THEN** latest system features are supported

#### Scenario: Universal binary
- **WHEN** SDK is installed on macOS
- **THEN** both Intel and Apple Silicon are supported

### Requirement: Cross-Compilation Support
The system SHALL support cross-compilation for embedded targets.

#### Scenario: ARM cross-compilation
- **WHEN** ARM target is specified
- **THEN** cross-compiled binaries work on ARM devices

#### Scenario: RISC-V cross-compilation
- **WHEN** RISC-V target is specified
- **THEN** cross-compiled binaries work on RISC-V devices

#### Scenario: Embedded target support
- **WHEN** bare-metal target is specified
- **THEN** minimal runtime dependencies are required

### Requirement: Linux DEB Packages
The system SHALL provide DEB packages for Debian-based distributions.

#### Scenario: libstonesoup package
- **WHEN** libstonesoup DEB is installed
- **THEN** shared library is available in system paths

#### Scenario: libstonesoup-dev package
- **WHEN** libstonesoup-dev DEB is installed
- **THEN** headers and static libraries are available

#### Scenario: stonesoup-doc package
- **WHEN** stonesoup-doc DEB is installed
- **THEN** documentation and examples are available

#### Scenario: Static vs dynamic variants
- **WHEN** package variant is selected
- **THEN** either static or dynamic library is installed

### Requirement: Linux RPM Packages
The system SHALL provide RPM packages for Red Hat-based distributions.

#### Scenario: stonesoup package
- **WHEN** stonesoup RPM is installed
- **THEN** shared library is available in system paths

#### Scenario: stonesoup-devel package
- **WHEN** stonesoup-devel RPM is installed
- **THEN** headers and static libraries are available

#### Scenario: stonesoup-doc package
- **WHEN** stonesoup-doc RPM is installed
- **THEN** documentation and examples are available

### Requirement: Windows MSI Installer
The system SHALL provide MSI installer for Windows deployment.

#### Scenario: Component selection
- **WHEN** MSI installer runs
- **THEN** user can select which components to install

#### Scenario: PATH configuration
- **WHEN** installation completes
- **THEN** binaries are optionally added to system PATH

#### Scenario: Uninstall support
- **WHEN** uninstaller runs
- **THEN** all SDK components are cleanly removed

### Requirement: Windows vcpkg Port
The system SHALL provide vcpkg port for C/C++ package management.

#### Scenario: vcpkg install
- **WHEN** vcpkg install stonesoup is run
- **THEN** library and headers are installed

#### Scenario: CMake integration
- **WHEN** find_package(stonesoup) is used
- **THEN** targets are correctly imported

### Requirement: macOS PKG Installer
The system SHALL provide signed PKG installer for macOS.

#### Scenario: Signed installer
- **WHEN** PKG installer is distributed
- **THEN** Apple notarization allows installation

#### Scenario: Framework installation
- **WHEN** PKG installs framework
- **THEN** stonesoup.framework is in /Library/Frameworks

### Requirement: Homebrew Formula
The system SHALL provide Homebrew formula for macOS.

#### Scenario: brew install
- **WHEN** brew install stonesoup is run
- **THEN** library and headers are installed

#### Scenario: brew upgrade
- **WHEN** new version is released
- **THEN** brew upgrade stonesoup updates correctly

### Requirement: Conda Packages
The system SHALL provide Conda packages for cross-platform data science ecosystem.

#### Scenario: conda install
- **WHEN** conda install -c stonesoup stonesoup is run
- **THEN** Python bindings are installed with dependencies

#### Scenario: conda-forge
- **WHEN** package is on conda-forge
- **THEN** community maintenance is enabled

### Requirement: Docker Images
The system SHALL provide Docker images for containerized deployment.

#### Scenario: Base image
- **WHEN** stonesoup/stonesoup:latest is pulled
- **THEN** full SDK is available in container

#### Scenario: Minimal image
- **WHEN** stonesoup/stonesoup:minimal is pulled
- **THEN** core runtime without dev tools is available

#### Scenario: Multi-architecture
- **WHEN** Docker image is pulled on ARM64
- **THEN** native ARM64 image is used

### Requirement: Conan Packages
The system SHALL provide Conan packages for C/C++ ecosystem.

#### Scenario: conan install
- **WHEN** stonesoup is added to conanfile
- **THEN** library is fetched and configured

#### Scenario: Conan Center Index
- **WHEN** package is on Conan Center
- **THEN** official distribution channel is available

### Requirement: Package Documentation
The system SHALL include documentation in all package formats.

#### Scenario: Man pages
- **WHEN** Linux package is installed
- **THEN** man pages are available for CLI tools

#### Scenario: PDF documentation
- **WHEN** documentation package is installed
- **THEN** PDF user guide is included

#### Scenario: Example programs
- **WHEN** documentation package is installed
- **THEN** compilable example programs are included

### Requirement: Package Signing
The system SHALL sign all distributed packages.

#### Scenario: GPG signing for Linux
- **WHEN** DEB/RPM packages are distributed
- **THEN** GPG signatures verify authenticity

#### Scenario: Authenticode for Windows
- **WHEN** MSI installer is distributed
- **THEN** Authenticode signature is present

#### Scenario: Apple notarization
- **WHEN** macOS packages are distributed
- **THEN** Apple notarization allows Gatekeeper passage
