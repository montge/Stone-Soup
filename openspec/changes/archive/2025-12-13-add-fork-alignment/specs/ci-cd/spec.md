## MODIFIED Requirements

### Requirement: Fork Alignment
The system SHALL maintain alignment with upstream Stone Soup while supporting independent releases.

#### Scenario: Upstream sync
- **WHEN** upstream Stone Soup has new commits (checked daily at 3 AM UTC)
- **THEN** automated PR is created to merge upstream changes

#### Scenario: Conflict detection
- **WHEN** upstream merge has conflicts
- **THEN** maintainers are notified via GitHub issue and PR label

#### Scenario: Independent versioning
- **WHEN** fork releases occur
- **THEN** version includes fork identifier (e.g., 1.0.0+sdk.1)

#### Scenario: Changelog tracking
- **WHEN** release is prepared
- **THEN** changes from both upstream and fork are documented in PR description
