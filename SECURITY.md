# Security Policy

## Supported Versions

The following versions of Stone Soup are currently receiving security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Stone Soup seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred):
   - Go to the [Security Advisories](https://github.com/dstl/Stone-Soup/security/advisories) page
   - Click "Report a vulnerability"
   - Fill in the details

2. **Email**:
   - Send an email to the maintainers with the subject line: `[SECURITY] Stone Soup Vulnerability Report`
   - Include as much detail as possible

### What to Include

Please include the following information in your report:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s)** related to the manifestation of the issue
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with an initial assessment
- **Resolution Target**: Within 90 days for most vulnerabilities

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report
2. **Assessment**: We will investigate and validate the reported vulnerability
3. **Communication**: We will keep you informed of our progress
4. **Fix Development**: We will develop and test a fix
5. **Disclosure**: We will coordinate with you on the public disclosure timeline

### Disclosure Policy

- We follow a coordinated disclosure policy
- We aim to release fixes before public disclosure
- We will credit reporters in our security advisories (unless you prefer to remain anonymous)
- We request that you:
  - Give us reasonable time to fix the issue before public disclosure
  - Avoid accessing or modifying user data
  - Act in good faith to avoid privacy violations and data destruction

## Security Best Practices

When using Stone Soup in your applications:

### Input Validation

- Always validate sensor data before processing
- Check array dimensions and bounds
- Sanitize file paths when loading configurations

### Dependency Management

- Keep Stone Soup and its dependencies up to date
- Review the SBOM (Software Bill of Materials) included with releases
- Use virtual environments to isolate dependencies

### Configuration Security

- Never commit credentials or API keys to version control
- Use environment variables for sensitive configuration
- Review configuration files before deployment

### Multi-Language Bindings

- Ensure the C library (libstonesoup) is from a trusted source
- Verify checksums when downloading pre-built binaries
- Use the package manager verification features (PyPI, crates.io, npm)

## Security Features

Stone Soup includes several security features:

- **SBOM Generation**: Software Bill of Materials for dependency transparency
- **Artifact Signing**: Sigstore signatures for release verification
- **Dependency Scanning**: Automated vulnerability scanning in CI
- **SAST**: Static Application Security Testing with Bandit and Semgrep
- **Secret Detection**: Pre-commit hooks to prevent credential leaks

## Verifying Releases

### Python Packages

Releases are signed with Sigstore. To verify:

```bash
pip install sigstore
sigstore verify stonesoup-*.whl
```

### Checksums

Each release includes SHA256 checksums:

```bash
sha256sum -c SHA256SUMS.txt
```

### Build Provenance (SLSA)

Releases include SLSA build provenance attestations that cryptographically prove where and how artifacts were built. To verify provenance:

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Verify provenance for a wheel file
gh attestation verify stonesoup-*.whl --owner dstl

# Verify provenance for source distribution
gh attestation verify stonesoup-*.tar.gz --owner dstl
```

The attestations prove:
- The artifact was built by our GitHub Actions workflow
- The exact commit and workflow that produced the artifact
- No tampering occurred after the build

## Security Updates

Security updates are announced via:

- GitHub Security Advisories
- Release notes
- The project's documentation

Subscribe to the repository's security alerts to receive notifications.

## Contact

For non-vulnerability security questions, please open a GitHub Discussion.

For vulnerability reports, please use the methods described in "How to Report" above.
