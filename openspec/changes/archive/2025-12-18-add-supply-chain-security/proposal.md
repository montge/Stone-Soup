# Change: Add Supply Chain Security

## Why
Modern software supply chain attacks are increasingly common. Stone Soup needs comprehensive supply chain security measures including Software Bill of Materials (SBOM) generation, secrets detection, artifact signing, and dependency transparency to protect users and meet enterprise security requirements.

## What Changes
- Add SBOM generation in CycloneDX and SPDX formats
- Add secrets detection to pre-commit hooks (gitleaks/detect-secrets)
- Configure artifact signing for releases (GPG, Sigstore)
- Add dependency license compliance checking
- Generate SLSA provenance for build artifacts

## Impact
- Affected specs: security-tooling, ci-cd
- Affected code:
  - `.github/workflows/ci.yml` - SBOM generation steps
  - `.github/workflows/release.yml` - artifact signing
  - `.pre-commit-config.yaml` - secrets detection hooks
  - New `sbom/` directory for SBOM outputs
