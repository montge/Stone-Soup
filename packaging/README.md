# Stone Soup Packaging

This directory contains packaging configurations for distributing Stone Soup across multiple platforms and package managers.

## Structure

- **linux/**: Linux distribution packages (Debian/Ubuntu and RPM-based)
  - `debian/`: Debian/Ubuntu package configuration
  - `rpm/`: RPM package configuration for RHEL/Fedora/CentOS
- **windows/**: Windows installer configuration using WiX Toolset
- **macos/**: macOS PKG installer configuration
- **docker/**: Docker container images
- **conda/**: Conda package recipe

## Building Packages

### Debian/Ubuntu Package

```bash
cd /home/e/Development/Stone-Soup
dpkg-buildpackage -us -uc
```

### RPM Package

```bash
rpmbuild -ba packaging/linux/rpm/stonesoup.spec
```

### Docker Images

```bash
# Full development image
docker build -t stonesoup:latest -f packaging/docker/Dockerfile .

# Minimal runtime image
docker build -t stonesoup:minimal -f packaging/docker/Dockerfile.minimal .
```

### Conda Package

```bash
conda build packaging/conda
```

### Windows Installer

Requires WiX Toolset:
```bash
candle packaging/windows/stonesoup.wxs
light stonesoup.wixobj -out stonesoup.msi
```

### macOS PKG

```bash
pkgbuild --root /path/to/install --identifier uk.gov.dstl.stonesoup \
         --version 1.0.0 --install-location /usr/local \
         --scripts packaging/macos stonesoup.pkg
```

## Package Contents

All packages include:
- Core Stone Soup Python library
- Documentation (when available)
- Examples and tutorials
- License files

Development packages additionally include:
- Header files
- Development tools
- Test suites

## Requirements

See individual package configurations for specific build dependencies.

## License

Stone Soup is released under the MIT License. See the LICENSE file in the repository root.
