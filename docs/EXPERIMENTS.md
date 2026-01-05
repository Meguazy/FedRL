# Experimental Design Document

This document outlines the complete experimental plan for validating the thesis: **"Clustered Federated Deep Reinforcement Learning with Selective Aggregation: A Framework for Chess Playstyle Preservation"**.

## Table of Contents

1. [Research Hypotheses](#research-hypotheses)
2. [Experimental Overview](#experimental-overview)
3. [Baseline Experiments](#baseline-experiments)
4. [Main System Evaluation](#main-system-evaluation)
5. [Ablation Studies](#ablation-studies)
6. [Performance Evaluation](#performance-evaluation)
7. [Statistical Analysis Plan](#statistical-analysis-plan)
8. [Experimental Timeline](#experimental-timeline)
9. [Resource Requirements](#resource-requirements)

---

## Research Hypotheses

### Primary Hypotheses

| ID | Hypothesis | Validation Method |
|----|------------|-------------------|
| **H1** | Clustered federated learning with selective aggregation preserves distinct playstyles better than standard federated learning | Compare playstyle divergence: Full System vs No Clustering baseline |
| **H2** | Selective aggregation is necessary for playstyle preservation - full layer aggregation destroys cluster-specific patterns | Compare: Selective Aggregation vs Full Aggregation |
| **H3** | Early network layers remain shared (general chess knowledge) while late layers diverge (playstyle-specific) | Analyze per-layer-group divergence metrics |
| **H4** | Policy head diverges more than value head due to different move preferences between playstyles | Compare `policy_head` vs `value_head` divergence index |
| **H5** | Cluster divergence increases over training rounds and eventually stabilizes | Track `mean_divergence` over time, fit plateau curve |

### Secondary Hypotheses

| ID | Hypothesis | Validation Method |
|----|------------|-------------------|
| **H6** | Tactical cluster develops preference for captures, checks, and aggressive moves | Analyze move type distribution in evaluation games |
| **H7** | Positional cluster develops preference for pawn advances, piece maneuvering | Analyze move type distribution in evaluation games |
| **H8** | Both clusters maintain competitive playing strength despite different styles | ELO estimation against Stockfish |
| **H9** | Model divergence correlates with playstyle divergence | Pearson correlation between metrics |
| **H10** | Shared layers enable knowledge transfer between clusters | Compare learning speed vs isolated training |

---

## Experimental Overview

### Experiment Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL DESIGN                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  BASELINES   │    │ MAIN SYSTEM  │    │  ABLATIONS   │      │
│  │              │    │              │    │              │      │
│  │ • No Cluster │    │ • Full       │    │ • Layer      │      │
│  │ • No Select. │    │   System     │    │   Patterns   │      │
│  │ • Isolated   │    │ • Extended   │    │ • Num Clust. │      │
│  │              │    │   Training   │    │ • Agg. Freq. │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                  ┌──────────────────┐                          │
│                  │   PERFORMANCE    │                          │
│                  │   EVALUATION     │                          │
│                  │                  │                          │
│                  │ • ELO Estimation │                          │
│                  │ • Move Analysis  │                          │
│                  │ • Generalization │                          │
│                  └──────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Common Configuration

All experiments share these base parameters unless otherwise specified:

```yaml
# Base configuration for all experiments
base_config:
  # Network Architecture
  model:
    type: "AlphaZeroNet"
    residual_blocks: 19
    channels: 256
    policy_channels: 73

  # Training Parameters
  training:
    games_per_round: 400  # per cluster
    evaluation_interval: 10  # rounds
    checkpoint_interval: 5  # rounds

  # Evaluation
  evaluation:
    games_per_elo_level: 30
    stockfish_elo_levels: [1000, 1200, 1400, 1600, 1800]
    time_per_move: 0.5  # seconds
```

---

## Baseline Experiments

### Experiment B1: No Clustering (Standard Federated Learning)

**Purpose**: Establish baseline for standard federated learning without playstyle clustering.

**Configuration**:
```yaml
experiment_id: "B1_no_clustering"
description: "Standard FedAvg without clustering - all nodes aggregate together"

clusters:
  - id: "unified_cluster"
    nodes: 8  # All nodes in single cluster
    training_type: "mixed"  # Both tactical and positional games

aggregation:
  type: "fedavg"
  selective: false  # Aggregate ALL layers

training:
  total_rounds: 100
  games_per_round: 800  # 8 nodes × 100 games
```

**Metrics to Collect**:
- Playstyle metrics (tactical_score, classification) - expect convergence to "balanced"
- ELO estimation over rounds
- Move type distribution

**Expected Results**:
- Single model emerges with mixed playstyle
- No playstyle divergence (tactical_score ≈ 0.5 for all)
- Serves as baseline for comparison

---

### Experiment B2: No Selective Aggregation (Full Layer Aggregation)

**Purpose**: Demonstrate necessity of selective aggregation for playstyle preservation.

**Configuration**:
```yaml
experiment_id: "B2_no_selective_aggregation"
description: "Clustered FL but aggregate ALL layers (no selective aggregation)"

clusters:
  - id: "cluster_tactical"
    nodes: 4
    training_type: "tactical"

  - id: "cluster_positional"
    nodes: 4
    training_type: "positional"

aggregation:
  intra_cluster:
    type: "fedavg"
    layers: "all"  # Aggregate all layers within cluster

  inter_cluster:
    type: "fedavg"
    layers: "all"  # Share ALL layers between clusters (not selective)

training:
  total_rounds: 100
  games_per_round: 400  # per cluster
```

**Metrics to Collect**:
- Cluster divergence (expect LOW - clusters converge)
- Playstyle metrics (expect convergence)
- Per-layer divergence analysis

**Expected Results**:
- Clusters start different but converge over time
- `mean_divergence` remains low or decreases
- Playstyle metrics converge between clusters
- **Key comparison with B3 (Full System)**

---

### Experiment B3: Isolated Training (No Federation)

**Purpose**: Understand what happens without any knowledge sharing.

**Configuration**:
```yaml
experiment_id: "B3_isolated_training"
description: "Each cluster trains independently - no aggregation between clusters"

clusters:
  - id: "cluster_tactical"
    nodes: 4
    training_type: "tactical"

  - id: "cluster_positional"
    nodes: 4
    training_type: "positional"

aggregation:
  intra_cluster:
    type: "fedavg"
    layers: "all"

  inter_cluster:
    enabled: false  # NO inter-cluster aggregation

training:
  total_rounds: 100
  games_per_round: 400
```

**Metrics to Collect**:
- Cluster divergence (expect MAXIMUM - no sharing)
- Individual cluster ELO (may be lower due to no knowledge transfer)
- Learning curves comparison

**Expected Results**:
- Maximum divergence between clusters
- Potentially slower learning (no knowledge transfer)
- Strong playstyle preservation but possibly weaker overall play
- **Establishes upper bound for divergence**

---

## Main System Evaluation

### Experiment M1: Full System (Primary Experiment)

**Purpose**: Evaluate the complete proposed system with selective aggregation.

**Configuration**:
```yaml
experiment_id: "M1_full_system"
description: "Complete system: Clustered FL with Selective Aggregation"

clusters:
  - id: "cluster_tactical"
    nodes: 4
    training_type: "tactical"
    games_per_round: 400

  - id: "cluster_positional"
    nodes: 4
    training_type: "positional"
    games_per_round: 400

aggregation:
  intra_cluster:
    type: "fedavg"
    weighting: "samples"
    layers: "all"

  inter_cluster:
    type: "selective"
    shared_layers:
      - "input_conv.*"
      - "input_bn.*"
      - "residual.0.*"
      - "residual.1.*"
      - "residual.2.*"
      - "residual.3.*"
      - "residual.4.*"
      - "residual.5.*"
    cluster_specific_layers:
      - "residual.6.*" through "residual.18.*"
      - "policy_head.*"
      - "value_head.*"

training:
  total_rounds: 100
  games_per_round: 400

evaluation:
  interval: 10
  metrics:
    - playstyle
    - divergence
    - weight_statistics
    - elo_estimation
```

**Metrics to Collect**:
- All implemented metrics every 10 rounds
- Detailed per-layer divergence
- Playstyle classification accuracy
- ELO estimation

**Expected Results**:
- Moderate divergence (between B2 and B3)
- Strong playstyle preservation
- Competitive playing strength
- Layer-depth divergence gradient (early shared, late diverged)

---

### Experiment M2: Extended Training

**Purpose**: Observe long-term behavior and divergence plateau.

**Configuration**:
```yaml
experiment_id: "M2_extended_training"
description: "Extended training to observe plateau behavior"

# Same as M1 but with extended training
training:
  total_rounds: 200  # Extended from 100

evaluation:
  interval: 10
```

**Metrics to Collect**:
- Divergence trajectory over 200 rounds
- Plateau detection
- Long-term stability of playstyles

**Expected Results**:
- Divergence increases then plateaus (~round 80-120)
- Playstyles remain stable after plateau
- No divergence collapse

---

## Ablation Studies

### Ablation A1: Shared Layer Boundary

**Purpose**: Determine optimal boundary between shared and cluster-specific layers.

**Configurations**:

```yaml
# A1a: Share only input block
ablation_id: "A1a_share_input_only"
shared_layers:
  - "input_conv.*"
  - "input_bn.*"
# All residual blocks are cluster-specific

# A1b: Share input + early residual (0-2)
ablation_id: "A1b_share_early_2"
shared_layers:
  - "input_conv.*"
  - "input_bn.*"
  - "residual.0.*"
  - "residual.1.*"
  - "residual.2.*"

# A1c: Share input + early residual (0-5) [DEFAULT]
ablation_id: "A1c_share_early_5"
shared_layers:
  - "input_conv.*"
  - "input_bn.*"
  - "residual.0.*" through "residual.5.*"

# A1d: Share input + half residual (0-9)
ablation_id: "A1d_share_half"
shared_layers:
  - "input_conv.*"
  - "input_bn.*"
  - "residual.0.*" through "residual.9.*"

# A1e: Share all except heads
ablation_id: "A1e_share_all_residual"
shared_layers:
  - "input_conv.*"
  - "input_bn.*"
  - "residual.*"  # All residual blocks
# Only heads are cluster-specific
```

**Analysis**:
| Config | Shared Layers | Expected Divergence | Expected Playstyle Preservation |
|--------|---------------|---------------------|--------------------------------|
| A1a | Input only | Very High | Strong |
| A1b | Input + res 0-2 | High | Strong |
| A1c | Input + res 0-5 | Medium | Good (baseline) |
| A1d | Input + res 0-9 | Low-Medium | Moderate |
| A1e | All except heads | Low | Weak |

**Metrics**:
- Per-layer-group divergence
- Playstyle divergence score
- Learning speed (ELO progression)

---

### Ablation A2: Number of Clusters

**Purpose**: Evaluate scalability to multiple playstyles.

**Configurations**:

```yaml
# A2a: 2 Clusters (Default)
ablation_id: "A2a_2_clusters"
clusters:
  - tactical
  - positional

# A2b: 3 Clusters
ablation_id: "A2b_3_clusters"
clusters:
  - tactical
  - positional
  - aggressive  # High-risk, attacking play

# A2c: 4 Clusters
ablation_id: "A2c_4_clusters"
clusters:
  - tactical
  - positional
  - aggressive
  - defensive  # Solid, safety-first play
```

**Analysis**:
- Can the system maintain distinct playstyles with more clusters?
- How does divergence scale?
- Is there a practical limit?

---

### Ablation A3: Aggregation Frequency

**Purpose**: Determine optimal inter-cluster aggregation frequency.

**Configurations**:

```yaml
# A3a: Frequent aggregation (every 5 rounds)
ablation_id: "A3a_freq_5"
inter_cluster_aggregation:
  frequency: 5

# A3b: Default (every 10 rounds)
ablation_id: "A3b_freq_10"
inter_cluster_aggregation:
  frequency: 10

# A3c: Infrequent (every 20 rounds)
ablation_id: "A3c_freq_20"
inter_cluster_aggregation:
  frequency: 20

# A3d: Rare (every 50 rounds)
ablation_id: "A3d_freq_50"
inter_cluster_aggregation:
  frequency: 50
```

**Expected Trade-offs**:
| Frequency | Knowledge Transfer | Playstyle Preservation | Communication Cost |
|-----------|-------------------|------------------------|-------------------|
| Every 5 | High | Lower | High |
| Every 10 | Good | Good | Medium |
| Every 20 | Moderate | Higher | Low |
| Every 50 | Low | Highest | Very Low |

---

### Ablation A4: Training Data Composition

**Purpose**: Understand impact of training data purity on playstyle development.

**Configurations**:

```yaml
# A4a: Pure training data (100% style-specific)
ablation_id: "A4a_pure_100"
clusters:
  tactical:
    data_composition:
      tactical_games: 100%
      positional_games: 0%
  positional:
    data_composition:
      tactical_games: 0%
      positional_games: 100%

# A4b: Mostly pure (80/20)
ablation_id: "A4b_mostly_pure_80"
clusters:
  tactical:
    data_composition:
      tactical_games: 80%
      positional_games: 20%
  positional:
    data_composition:
      tactical_games: 20%
      positional_games: 80%

# A4c: Slightly biased (60/40)
ablation_id: "A4c_biased_60"
clusters:
  tactical:
    data_composition:
      tactical_games: 60%
      positional_games: 40%
  positional:
    data_composition:
      tactical_games: 40%
      positional_games: 60%
```

**Analysis**:
- How pure must training data be for distinct playstyles?
- Is there a minimum bias threshold?

---

### Ablation A5: Node Distribution

**Purpose**: Evaluate impact of unbalanced cluster sizes.

**Configurations**:

```yaml
# A5a: Balanced (4-4)
ablation_id: "A5a_balanced_4_4"
clusters:
  tactical: 4 nodes
  positional: 4 nodes

# A5b: Slightly unbalanced (5-3)
ablation_id: "A5b_unbalanced_5_3"
clusters:
  tactical: 5 nodes
  positional: 3 nodes

# A5c: Highly unbalanced (6-2)
ablation_id: "A5c_unbalanced_6_2"
clusters:
  tactical: 6 nodes
  positional: 2 nodes
```

**Analysis**:
- Does cluster size affect playstyle strength?
- How does aggregation weighting handle imbalance?

---

## Performance Evaluation

### Evaluation E1: ELO Estimation

**Purpose**: Measure absolute playing strength.

**Method**:
```yaml
evaluation_id: "E1_elo_estimation"

stockfish_opponents:
  - elo: 1000
    games: 30
  - elo: 1200
    games: 30
  - elo: 1400
    games: 30
  - elo: 1600
    games: 30
  - elo: 1800
    games: 30
  - elo: 2000
    games: 30

time_control:
  time_per_move: 0.5  # seconds

metrics:
  - win_rate_per_level
  - estimated_elo
  - elo_confidence_interval
```

**Analysis**:
- Compare ELO across experiments
- Verify both clusters maintain competitive strength
- Track ELO progression over training

---

### Evaluation E2: Move Type Analysis

**Purpose**: Quantify playstyle differences through move choices.

**Method**:
```yaml
evaluation_id: "E2_move_analysis"

positions:
  - source: "benchmark_positions.pgn"
    count: 500
    types:
      - tactical (250)
      - quiet (250)

analysis:
  - top_3_moves_per_position
  - move_type_classification:
      - captures
      - checks
      - pawn_advances
      - piece_maneuvers
      - castling
      - quiet_moves
```

**Expected Results**:
| Move Type | Tactical Cluster | Positional Cluster |
|-----------|------------------|-------------------|
| Captures | Higher % | Lower % |
| Checks | Higher % | Lower % |
| Pawn advances | Lower % | Higher % |
| Quiet moves | Lower % | Higher % |

---

### Evaluation E3: Generalization Test

**Purpose**: Test performance on unseen position types.

**Method**:
```yaml
evaluation_id: "E3_generalization"

test_sets:
  - name: "unseen_openings"
    description: "Openings not in training data"
    positions: 100

  - name: "complex_middlegames"
    description: "Positions with high complexity"
    positions: 100

  - name: "technical_endgames"
    description: "Endgame positions requiring precise calculation"
    positions: 100
```

---

## Statistical Analysis Plan

### Required Statistical Tests

| Comparison | Test | Purpose |
|------------|------|---------|
| Full System vs Baselines | Two-sample t-test | Compare mean divergence |
| Divergence over time | Regression analysis | Fit plateau curve |
| Playstyle classification | Chi-square test | Verify distinct classifications |
| ELO comparison | ANOVA | Compare across experiments |
| Correlation analysis | Pearson correlation | Divergence vs playstyle |

### Significance Levels

- Primary hypotheses: α = 0.05
- Secondary hypotheses: α = 0.10
- Multiple comparison correction: Bonferroni

### Sample Size Justification

```
For detecting medium effect size (d = 0.5):
- Power: 0.80
- Alpha: 0.05
- Required games per condition: ~64

We use 100 rounds with evaluation every 10 rounds = 10 data points per experiment
30 games per ELO level × 5 levels = 150 games per evaluation
Total games per experiment: 1500+ (sufficient for statistical power)
```

---

## Experimental Timeline

### Phase 1: Baselines (Week 1-2)

| Day | Experiment | Duration | GPU Hours |
|-----|------------|----------|-----------|
| 1-3 | B1: No Clustering | ~48h | 48 |
| 4-6 | B2: No Selective | ~48h | 48 |
| 7-9 | B3: Isolated | ~48h | 48 |
| 10-14 | Analysis & Documentation | - | - |

### Phase 2: Main System (Week 3-4)

| Day | Experiment | Duration | GPU Hours |
|-----|------------|----------|-----------|
| 15-18 | M1: Full System | ~72h | 72 |
| 19-24 | M2: Extended Training | ~96h | 96 |
| 25-28 | Analysis & Documentation | - | - |

### Phase 3: Ablations (Week 5-8)

| Week | Experiments | GPU Hours |
|------|-------------|-----------|
| 5 | A1: Shared Layer Boundary (5 configs) | 240 |
| 6 | A2: Number of Clusters (3 configs) | 144 |
| 7 | A3: Aggregation Frequency (4 configs) | 192 |
| 8 | A4 & A5: Data & Node Distribution | 192 |

### Phase 4: Performance Evaluation (Week 9-10)

| Day | Evaluation | Duration |
|-----|------------|----------|
| 57-60 | E1: ELO Estimation | ~24h |
| 61-64 | E2: Move Analysis | ~8h |
| 65-68 | E3: Generalization | ~8h |

---

## Resource Requirements

### Hardware

```yaml
compute:
  gpu: "NVIDIA RTX 3090 or better"
  gpu_memory: "24GB+"
  cpu: "8+ cores"
  ram: "32GB+"
  storage: "500GB SSD"

nodes:
  total: 8
  per_cluster: 4
```

### Software

```yaml
dependencies:
  - python: "3.10+"
  - pytorch: "2.0+"
  - stockfish: "15+"
  - redis: "7.0+"
```

### Estimated Total GPU Hours

| Phase | GPU Hours |
|-------|-----------|
| Baselines | 144 |
| Main System | 168 |
| Ablations | 768 |
| Evaluation | 40 |
| **Total** | **~1120** |

---

## Appendix A: Metrics Summary

### Collected Every 10 Rounds

| Metric Category | Metrics | Storage Location |
|-----------------|---------|------------------|
| Playstyle | tactical_score, classification, avg_legal_moves, opening_preferences | `{cluster}/evaluation_round_{N}.json` |
| Divergence | cosine_similarity, l2_distance, divergence_index (per-layer, per-group, global) | `model_divergence/round_{N}.json` |
| Weight Stats | mean, std, sparsity, weight_change, dead_layers | `{cluster}/weight_stats_round_{N}.json` |

### Collected at Evaluation Points

| Metric | Description |
|--------|-------------|
| ELO Estimation | Estimated rating from Stockfish matches |
| Win Rate | Per opponent level and overall |
| Game Length | Average moves per game |
| Move Distribution | % of each move type |

---

## Appendix B: Success Criteria

### Minimum Success (Thesis Defense)

- [ ] H1 validated: Full system shows significantly higher playstyle divergence than B1
- [ ] H3 validated: Clear layer-depth divergence gradient observed
- [ ] Both clusters achieve ELO > 1400

### Target Success

- [ ] All primary hypotheses (H1-H5) validated
- [ ] At least 3 ablation studies completed
- [ ] Clusters achieve ELO > 1600
- [ ] Clear plateau behavior observed

### Stretch Goals

- [ ] All hypotheses validated
- [ ] All ablation studies completed
- [ ] Clusters achieve ELO > 1800
- [ ] Paper-ready visualizations

---

*Document Version: 1.0*
*Created: 2025-11-23*
*Author: Chess Federated Learning Team*
