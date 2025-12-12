# viterbi-algorithms Specification

## Purpose
TBD - created by archiving change add-voxel-viterbi-coordinates. Update Purpose after archive.
## Requirements
### Requirement: Viterbi Smoother
The system SHALL provide a Viterbi smoother for optimal state sequence estimation in HMM-based tracking.

#### Scenario: Optimal sequence estimation
- **WHEN** ViterbiSmoother processes a track with multiple state estimates
- **THEN** the most likely state sequence is returned

#### Scenario: Log-domain computation
- **WHEN** Viterbi algorithm computes path probabilities
- **THEN** log-domain arithmetic is used for numerical stability

#### Scenario: Smoother integration
- **WHEN** ViterbiSmoother is used
- **THEN** it follows Stone Soup's Smoother interface pattern

#### Scenario: Backtracking
- **WHEN** forward pass completes
- **THEN** backtracking recovers optimal state sequence

### Requirement: Viterbi Track Initiator
The system SHALL provide a track-before-detect initiator using Viterbi algorithm.

#### Scenario: Multi-scan track extraction
- **WHEN** ViterbiTrackInitiator processes multiple scans of detections
- **THEN** tracks are initiated from detection sequences with high cumulative likelihood

#### Scenario: Detection lattice construction
- **WHEN** detections from multiple scans are provided
- **THEN** a lattice of possible track hypotheses is constructed

#### Scenario: Track confirmation
- **WHEN** Viterbi path probability exceeds threshold
- **THEN** track is confirmed and returned as Track object

#### Scenario: False alarm rejection
- **WHEN** detection sequence has low cumulative likelihood
- **THEN** it is rejected as false alarm

### Requirement: Graph-Constrained Viterbi
The system SHALL provide graph-based Viterbi for road/rail constrained tracking.

#### Scenario: Road network constraint
- **WHEN** GraphViterbiSmoother is configured with road network
- **THEN** state estimates are constrained to road segments

#### Scenario: Graph edge transitions
- **WHEN** track crosses road intersection
- **THEN** transition probabilities reflect allowed turns

#### Scenario: Speed limit constraints
- **WHEN** road segment has speed limit
- **THEN** Viterbi considers speed constraints in transition model

#### Scenario: Off-road detection
- **WHEN** detection is far from any road segment
- **THEN** appropriate likelihood penalty is applied

### Requirement: Multi-Hypothesis Viterbi
The system SHALL support multiple hypothesis tracking with Viterbi.

#### Scenario: N-best paths
- **WHEN** ViterbiSmoother is configured for N-best
- **THEN** top N most likely state sequences are returned

#### Scenario: Hypothesis pruning
- **WHEN** number of hypotheses exceeds limit
- **THEN** low-probability hypotheses are pruned

#### Scenario: Track splitting
- **WHEN** ambiguous association exists
- **THEN** multiple track hypotheses are maintained

### Requirement: Viterbi Performance
The system SHALL implement efficient Viterbi computation for real-time tracking.

#### Scenario: Linear complexity
- **WHEN** Viterbi processes T time steps with S states
- **THEN** complexity is O(T * S^2) or better with pruning

#### Scenario: Sparse transitions
- **WHEN** transition matrix is sparse
- **THEN** only non-zero transitions are computed

#### Scenario: Beam search pruning
- **WHEN** beam width is specified
- **THEN** only top-k states are retained per time step

### Requirement: HMM State Definition
The system SHALL support flexible state space definition for HMM-based tracking.

#### Scenario: Discrete state space
- **WHEN** HMM uses discrete states (e.g., road segments)
- **THEN** state space is enumerable with defined transitions

#### Scenario: Discretized continuous space
- **WHEN** continuous state space is discretized
- **THEN** grid-based states are created with spatial relationships

#### Scenario: Hybrid state space
- **WHEN** state has discrete and continuous components
- **THEN** both components are handled in Viterbi computation

