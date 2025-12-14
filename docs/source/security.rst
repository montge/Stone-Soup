Security
========

Stone Soup takes security seriously. This page documents our security practices
and how to verify the integrity of Stone Soup releases.

Reporting Vulnerabilities
-------------------------

Please see our `SECURITY.md <https://github.com/dstl/Stone-Soup/blob/main/SECURITY.md>`_
file for information on how to report security vulnerabilities.

Supply Chain Security
---------------------

Stone Soup implements several supply chain security measures to ensure the
integrity and provenance of releases.

Software Bill of Materials (SBOM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each release includes CycloneDX-format SBOMs for all components:

- **Python**: Generated with ``cyclonedx-bom``
- **Rust**: Generated with ``cargo-sbom``
- **Node.js**: Generated with ``@cyclonedx/cyclonedx-npm``
- **Java**: Generated with ``cyclonedx-maven-plugin``

A combined SBOM is also provided that merges all component SBOMs.

SBOMs are available as release artifacts and can be used to:

- Audit dependencies for known vulnerabilities
- Verify license compliance
- Track component versions

Build Provenance
^^^^^^^^^^^^^^^^

Releases include SLSA build provenance attestations that cryptographically prove:

- The artifact was built by our GitHub Actions workflow
- The exact commit and workflow that produced the artifact
- No tampering occurred after the build

To verify provenance::

    gh attestation verify stonesoup-*.whl --owner dstl

Artifact Signing
^^^^^^^^^^^^^^^^

Python packages are signed with Sigstore for verifiable authenticity::

    pip install sigstore
    sigstore verify stonesoup-*.whl

Each release also includes SHA256 checksums for all artifacts::

    sha256sum -c SHA256SUMS.txt

Secrets Detection
^^^^^^^^^^^^^^^^^

The repository uses multiple tools to prevent accidental secret exposure:

- **Gitleaks**: Scans for secrets in commits
- **detect-secrets**: Pre-commit hook for secret detection

License Compliance
^^^^^^^^^^^^^^^^^^

Stone Soup follows the `REUSE <https://reuse.software/>`_ specification for
license compliance. All files are tagged with SPDX license identifiers.

To verify REUSE compliance::

    pip install reuse
    reuse lint

Dependency Scanning
-------------------

Continuous integration includes several security scanning tools:

- **Bandit**: Python static analysis security testing
- **Semgrep**: Multi-language SAST with custom rules
- **Dependabot**: Automated dependency vulnerability alerts

Secure Development
------------------

When contributing to Stone Soup, please follow these security practices:

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Validate inputs**: Always validate data from external sources
3. **Keep dependencies updated**: Regularly update dependencies
4. **Review security alerts**: Respond promptly to Dependabot alerts
5. **Follow secure coding guidelines**: Avoid common vulnerabilities (OWASP Top 10)

For more details, see the :ref:`contributing:Contributing` page.
