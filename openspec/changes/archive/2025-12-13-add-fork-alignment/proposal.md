# Change: Add Fork Alignment Workflow

## Why

This fork extends the upstream Stone Soup with multi-language SDK features. To maintain compatibility and receive upstream improvements, we need automated synchronization with the upstream repository (dstl/Stone-Soup).

## What Changes

### Upstream Sync Workflow
- Add scheduled workflow to check for upstream changes daily
- Automatically create PRs to merge upstream changes
- Notify maintainers when conflicts require manual resolution

### Versioning
- Configure fork version identifier scheme (X.Y.Z+sdk.N)
- Track both upstream and fork changes in releases

## Impact

- **Affected specs**: ci-cd (implementing Fork Alignment requirement)
- **Affected code**:
  - `.github/workflows/upstream-sync.yml` (new)
