## 1. Nightly Workflow
- [x] 1.1 Create .github/workflows/nightly.yml
- [x] 1.2 Configure schedule trigger (2 AM UTC)
- [x] 1.3 Add full test suite execution
- [x] 1.4 Configure pre-release version scheme

## 2. Python Nightly
- [x] 2.1 Build Python wheel with dev version
- [x] 2.2 Publish to PyPI (with dev version)

## 3. Rust Nightly
- [x] 3.1 Build Rust crate with pre-release version
- [ ] 3.2 Publish to crates.io with pre-release tag (requires CARGO_REGISTRY_TOKEN secret)

## 4. Node.js Nightly
- [x] 4.1 Build npm package
- [x] 4.2 Publish with nightly dist-tag

## 5. Artifacts
- [x] 5.1 Generate platform-specific installers
- [x] 5.2 Upload as GitHub release assets
- [x] 5.3 Configure 30-day retention cleanup

## 6. README
- [x] 6.1 Add nightly build status badge
