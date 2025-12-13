# Change: Add Multi-Language Release Automation

## Why
The ci-cd spec requires automated releases for all language packages (Python, Rust, Node.js, Java), but the current release workflow only publishes Python to PyPI. Multi-language SDK releases should be automated and synchronized.

## What Changes
- Extend release.yml to build and publish Rust crate to crates.io
- Add npm package publishing for Node.js bindings
- Add Maven Central publishing for Java bindings
- Add coordinated version tagging across languages
- Add release verification tests for each language

## Impact
- Affected specs: ci-cd
- Affected code: .github/workflows/release.yml
- Requires secrets: CARGO_REGISTRY_TOKEN, NPM_TOKEN, MAVEN_GPG_*, OSSRH_*
