# Comprehensive Evaluation Guide

## Clustered Federated Deep Reinforcement Learning with Selective Aggregation

**Version:** 1.1
**Last Updated:** 2025-01-XX
**Purpose:** Step-by-step guide for evaluating the federated chess learning framework with layer sharing experiments

---

## Table of Contents

1. [Overview](#1-overview)
2. [Evaluation Architecture](#2-evaluation-architecture)
3. [Metrics Reference](#3-metrics-reference)
4. [Phase 1: Baseline Experiments](#4-phase-1-baseline-experiments)
5. [Phase 2: Partial Layer Sharing Experiments](#5-phase-2-partial-layer-sharing-experiments)
6. [Phase 3: Performance Evaluation](#6-phase-3-performance-evaluation)
7. [Post-Experiment Analysis](#7-post-experiment-analysis)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Results Interpretation](#9-results-interpretation)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### 1.1 Purpose

This guide provides detailed instructions for evaluating the Clustered Federated Deep Reinforcement Learning framework for chess. The evaluation validates ten research hypotheses (H1-H10) through systematic experiments across four phases.

### 1.2 Research Hypotheses

| ID | Hypothesis | Primary Metrics |
|----|------------|-----------------|
| H1 | Clustered FL outperforms centralized training | ELO, Win Rate |
| H2 | Selective aggregation improves cluster models | Cluster Divergence, Weight Statistics |
| H3 | Playstyle clusters emerge naturally | Tactical Score Distribution |
| H4 | Different clusters develop distinct strategies | Move Type Distribution, Playstyle Metrics |
| H5 | Cross-cluster learning enables knowledge transfer | ELO Improvement Rate |
| H6 | System scales with more clusters | Resource Usage, Convergence Time |
| H7 | Clusters maintain stability over training | Divergence Trends, Plateau Detection |
| H8 | Individual clients benefit from clustering | Per-Client ELO Gains |
| H9 | Framework generalizes to new positions | Generalization Test Scores |
| H10 | Behavioral differences are measurable | Move Type Comparison |

### 1.3 Experiment Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Baselines (2 experiments)                                  │
│  ├── B1: Full Sharing (all layers shared between clusters)          │
│  └── B2: No Sharing (completely independent clusters)               │
│                                                                      │
│  Phase 2: Partial Layer Sharing Experiments (4 experiments)          │
│  ├── P1: Share Early Layers Only (input + early residual)           │
│  ├── P2: Share Middle Layers Only (middle residual)                 │
│  ├── P3: Share Late Layers Only (late residual + heads)             │
│  └── P4: Share All Except Heads (backbone only)                     │
│                                                                      │
│  Phase 3: Performance Evaluation (3 evaluations)                     │
│  ├── E1: Playstyle Analysis                                          │
│  ├── E2: Move Type Analysis                                          │
│  └── E3: Generalization Test                                         │
│                                                                      │
│  Post-Experiment: Offline Analysis                                   │
│  ├── Plateau Detection                                               │
│  ├── Correlation Analysis                                            │
│  └── Statistical Significance Tests                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluation Architecture

### 2.1 Storage Structure

All metrics are stored in a structured directory hierarchy:

```
storage/
├── metrics/
│   └── {run_id}/
│       ├── cluster_tactical/
│       │   ├── evaluation_round_{N}.json      # Playstyle metrics
│       │   ├── weight_stats_round_{N}.json    # Weight statistics
│       │   └── move_types_round_{N}.json      # Move type distribution
│       ├── cluster_positional/
│       │   ├── evaluation_round_{N}.json
│       │   ├── weight_stats_round_{N}.json
│       │   └── move_types_round_{N}.json
│       ├── model_divergence/
│       │   └── round_{N}.json                 # Cluster divergence
│       └── events/
│           └── metrics_log.jsonl              # General metric events
├── models/
│   └── {run_id}/
│       └── {cluster_id}/
│           ├── round_{N}.pt                   # Model checkpoints
│           └── best_model.pt                  # Best performing model
└── .metadata/
    └── {run_id}.json                          # Run configuration
```

### 2.2 Metric Collection Timeline

| Training Phase | Metrics Collected | Frequency |
|---------------|-------------------|-----------|
| Every Round | Playstyle Evaluation | Per round |
| Every Round | Weight Statistics | Per round |
| Every Round | Cluster Divergence | Per round |
| Every Round | Training Loss | Per round |
| Every 10 Rounds | Move Type Distribution | Configurable |
| Every 10 Rounds | ELO Estimation | Configurable |
| End of Training | Final Model Checkpoints | Once |

### 2.3 Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PlaystyleEvaluator` | `server/evaluation/playstyle_evaluator.py` | Tactical score calculation |
| `MoveTypeAnalyzer` | `server/evaluation/move_type_analyzer.py` | Move classification |
| `ModelAnalyzer` | `server/evaluation/model_analyzer.py` | Weight statistics |
| `compute_cluster_divergence` | `server/evaluation/model_analyzer.py` | Divergence metrics |
| `ModelEvaluator` | `server/evaluation/model_evaluator.py` | ELO estimation |
| `FileExperimentTracker` | `server/storage/experiment_tracker.py` | Metrics storage |

---

## 3. Metrics Reference

### 3.1 Playstyle Metrics

**Purpose:** Quantify tactical vs positional playing style

**Computation:** Analyze chess positions using core and enhanced metrics

#### Core Metrics (per position)
- **Attacked Material:** Value of opponent pieces under attack (pawns=1, knights/bishops=3, rooks=5, queens=9)
- **Legal Moves:** Number of legal moves available
- **Possible Captures:** Number of capture moves available
- **Center Control:** Squares controlled in center (d4, d5, e4, e5)

#### Tactical Score Formula
```
tactical_score = (0.35 × attacked_material_norm) +
                 (0.25 × legal_moves_norm) +
                 (0.25 × captures_norm) +
                 (0.15 × center_control_norm)
```

#### Classification Scale
| Score Range | Classification |
|-------------|----------------|
| 0.0 - 0.3 | Very Positional |
| 0.3 - 0.45 | Positional |
| 0.45 - 0.55 | Balanced |
| 0.55 - 0.7 | Tactical |
| 0.7 - 1.0 | Very Tactical |

#### Output JSON Schema
```json
{
  "cluster_id": "cluster_tactical",
  "round_num": 50,
  "games_analyzed": 100,
  "tactical_score": {
    "mean": 0.62,
    "std": 0.15,
    "min": 0.31,
    "max": 0.89
  },
  "classification": {
    "label": "Tactical",
    "distribution": {
      "very_positional": 0.05,
      "positional": 0.15,
      "balanced": 0.25,
      "tactical": 0.40,
      "very_tactical": 0.15
    }
  },
  "core_metrics": {
    "attacked_material": {"mean": 4.2, "std": 2.1},
    "legal_moves": {"mean": 32.5, "std": 8.3},
    "possible_captures": {"mean": 3.8, "std": 1.9},
    "center_control": {"mean": 2.1, "std": 0.8}
  }
}
```

### 3.2 Cluster Divergence Metrics

**Purpose:** Track how models drift apart during training

**Computation:** Compare weight tensors across cluster models

#### Divergence Formula (L2 Norm)
```
divergence(A, B) = ||flatten(A) - flatten(B)||₂ / num_parameters
```

#### Layer Groups
| Group | Layers | Purpose |
|-------|--------|---------|
| `input_block` | Conv + BN | Initial feature extraction |
| `early_residual` | Blocks 0-5 | Low-level patterns |
| `middle_residual` | Blocks 6-12 | Mid-level strategies |
| `late_residual` | Blocks 13-18 | High-level planning |
| `policy_head` | Policy conv + FC | Move selection |
| `value_head` | Value conv + FC | Position evaluation |

#### Output JSON Schema
```json
{
  "round_num": 50,
  "timestamp": "2025-01-15T14:30:00",
  "pairwise": {
    "cluster_tactical_vs_cluster_positional": {
      "overall_divergence": 0.0234,
      "by_layer_group": {
        "input_block": 0.0012,
        "early_residual": 0.0089,
        "middle_residual": 0.0156,
        "late_residual": 0.0234,
        "policy_head": 0.0312,
        "value_head": 0.0289
      }
    }
  },
  "global": {
    "mean_divergence": 0.0234,
    "max_divergence": 0.0312,
    "min_divergence": 0.0012
  }
}
```

### 3.3 Weight Statistics Metrics

**Purpose:** Monitor model health and training dynamics

**Computation:** Statistical analysis of weight tensors per layer

#### Statistics Computed
- **Mean:** Average weight value
- **Std:** Standard deviation
- **Min/Max:** Range of values
- **L2 Norm:** Magnitude
- **Sparsity:** Fraction of near-zero weights (|w| < 1e-6)

#### Output JSON Schema
```json
{
  "cluster_id": "cluster_tactical",
  "round_num": 50,
  "timestamp": "2025-01-15T14:30:00",
  "by_layer_group": {
    "input_block": {
      "mean": 0.0012,
      "std": 0.0834,
      "min": -0.2341,
      "max": 0.2567,
      "l2_norm": 2.345,
      "sparsity": 0.0023,
      "num_params": 2048
    }
  },
  "summary": {
    "total_parameters": 23456789,
    "overall_mean": 0.0001,
    "overall_std": 0.0567,
    "overall_sparsity": 0.0034
  }
}
```

### 3.4 Move Type Distribution Metrics

**Purpose:** Classify moves to understand tactical vs positional play

**Move Categories:**
| Category | Description | Style Indicator |
|----------|-------------|-----------------|
| Captures | Takes opponent piece | Tactical |
| Checks | Gives check to king | Tactical |
| Aggressive | Captures + Checks | Tactical |
| Pawn Advances | Non-capture pawn moves | Positional |
| Piece Development | N/B moves (ply 1-20) | Positional |
| Castling | King safety | Positional |
| Quiet Moves | Non-capture, non-check | Positional |

#### Output JSON Schema
```json
{
  "cluster_id": "cluster_tactical",
  "games_analyzed": 50,
  "totals": {
    "total_moves": 3500,
    "captures": 420,
    "checks": 85,
    "pawn_advances": 680,
    "quiet_moves": 2100,
    "aggressive_moves": 505
  },
  "percentages": {
    "captures_pct": 12.0,
    "checks_pct": 2.4,
    "pawn_advances_pct": 19.4,
    "quiet_moves_pct": 60.0,
    "aggressive_pct": 14.4
  },
  "averages_per_game": {
    "avg_captures": 8.4,
    "avg_checks": 1.7,
    "avg_aggressive": 10.1
  }
}
```

### 3.5 ELO Estimation Metrics

**Purpose:** Estimate playing strength of trained models

**Computation:** Self-play matches with Glicko-2 rating system

#### Output JSON Schema
```json
{
  "cluster_id": "cluster_tactical",
  "round_num": 100,
  "elo": {
    "rating": 1850,
    "deviation": 45,
    "volatility": 0.06
  },
  "matches": {
    "total": 100,
    "wins": 52,
    "losses": 38,
    "draws": 10
  }
}
```

---

## 4. Phase 1: Baseline Experiments

### 4.1 Overview

**Duration:** ~10 days
**GPU Hours:** ~240
**Purpose:** Establish two extreme baselines for comparison

The baseline experiments represent the two extremes of the sharing spectrum:
- **Full Sharing (B1):** All layers are shared between clusters - equivalent to standard FedAvg
- **No Sharing (B2):** No layers are shared - clusters train completely independently

All intermediate experiments (Phase 2) will be compared against these two baselines.

### 4.2 Experiment B1: Full Sharing

**Description:** All layers shared between clusters (equivalent to standard FedAvg with clustering)

This baseline represents maximum knowledge sharing where all model parameters are averaged across clusters after each round. Clusters receive identical models but train on different playstyle-specific data.

#### Configuration
```yaml
experiment_id: B1_full_sharing
experiment_type: baseline

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

model:
  architecture: alphazero
  residual_blocks: 19
  filters: 256

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20
  initial_assignment: playstyle_score

aggregation:
  method: fedavg
  layer_sharing:
    mode: full  # Share ALL layers
    shared_layers:
      - input_block
      - early_residual
      - middle_residual
      - late_residual
      - policy_head
      - value_head

clients:
  total_clients: 50
  tactical_ratio: 0.5
  positional_ratio: 0.5

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Metrics Collected

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| Training Loss | Every round | Convergence tracking |
| Playstyle Evaluation | Every round | Style characterization |
| Weight Statistics | Every round | Model health per cluster |
| Cluster Divergence | Every round | Should be ~0 (identical models) |
| Move Type Distribution | Every 10 rounds | Behavioral analysis |
| ELO Estimation | Every 10 rounds | Strength assessment |

#### Expected Outcomes
- **Divergence:** Near-zero (models are identical after aggregation)
- **Playstyle Separation:** Minimal (no model specialization)
- **ELO:** Moderate baseline performance
- **Convergence:** Fast (full knowledge sharing)

### 4.3 Experiment B2: No Sharing

**Description:** Completely independent clusters with no layer sharing

This baseline represents zero knowledge sharing where clusters train completely independently. Each cluster maintains its own model without any cross-cluster aggregation.

#### Configuration
```yaml
experiment_id: B2_no_sharing
experiment_type: baseline

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

model:
  architecture: alphazero
  residual_blocks: 19
  filters: 256

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20
  initial_assignment: playstyle_score

aggregation:
  method: independent  # No cross-cluster sharing
  layer_sharing:
    mode: none  # Share NO layers
    shared_layers: []

clients:
  total_clients: 50
  tactical_ratio: 0.5
  positional_ratio: 0.5

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Metrics Collected

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| Training Loss | Every round | Convergence tracking |
| Playstyle Evaluation | Every round | Style specialization |
| Weight Statistics | Every round | Per-cluster model health |
| Cluster Divergence | Every round | Maximum divergence baseline |
| Move Type Distribution | Every 10 rounds | Behavioral differentiation |
| ELO Estimation | Every 10 rounds | Per-cluster strength |

#### Expected Outcomes
- **Divergence:** High (models evolve independently)
- **Playstyle Separation:** Maximum (full specialization)
- **ELO:** Potentially lower (no knowledge transfer)
- **Convergence:** Slower (smaller effective dataset per cluster)

### 4.4 Baseline Comparison Summary

| Aspect | B1: Full Sharing | B2: No Sharing |
|--------|------------------|----------------|
| Shared Layers | All (100%) | None (0%) |
| Expected Divergence | ~0 | High |
| Playstyle Separation | Minimal | Maximum |
| Knowledge Transfer | Full | None |
| Convergence Speed | Fast | Slow |
| Specialization | None | Full |

---

## 5. Phase 2: Partial Layer Sharing Experiments

### 5.1 Overview

**Duration:** ~20 days
**GPU Hours:** ~480
**Purpose:** Find optimal layer sharing strategy between the two baselines

These experiments explore the spectrum between full sharing and no sharing by selectively sharing specific layer groups. The goal is to identify which layers benefit most from cross-cluster sharing and which should remain cluster-specific.

#### Layer Groups Reference

| Group | Layers | Parameters | Function |
|-------|--------|------------|----------|
| `input_block` | Conv + BN | ~20K | Initial feature extraction |
| `early_residual` | Blocks 0-5 | ~2.5M | Low-level patterns (piece recognition) |
| `middle_residual` | Blocks 6-12 | ~3.5M | Mid-level strategies (tactics) |
| `late_residual` | Blocks 13-18 | ~3.5M | High-level planning (strategy) |
| `policy_head` | Policy conv + FC | ~1.5M | Move selection |
| `value_head` | Value conv + FC | ~0.5M | Position evaluation |

#### Hypothesis

- **Early layers** learn general chess features (piece recognition, board representation) → should benefit from sharing
- **Late layers** learn strategy-specific patterns → should remain cluster-specific
- **Heads** encode playstyle-specific decision making → likely should NOT be shared

### 5.2 Experiment P1: Share Early Layers Only

**Description:** Share input block and early residual layers only

This tests whether sharing low-level feature extraction improves performance while allowing clusters to develop specialized strategies.

#### Configuration
```yaml
experiment_id: P1_share_early
experiment_type: partial_sharing

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20

aggregation:
  method: selective_layer_sharing
  layer_sharing:
    mode: partial
    shared_layers:
      - input_block
      - early_residual
    cluster_specific_layers:
      - middle_residual
      - late_residual
      - policy_head
      - value_head

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Expected Outcomes
- Shared layers: Near-zero divergence for input_block/early_residual
- Cluster-specific layers: Increasing divergence
- Moderate playstyle separation
- Good ELO (shared basic features + specialized strategy)

### 5.3 Experiment P2: Share Middle Layers Only

**Description:** Share only middle residual layers

This tests whether mid-level tactical patterns are universal or playstyle-specific.

#### Configuration
```yaml
experiment_id: P2_share_middle
experiment_type: partial_sharing

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20

aggregation:
  method: selective_layer_sharing
  layer_sharing:
    mode: partial
    shared_layers:
      - middle_residual
    cluster_specific_layers:
      - input_block
      - early_residual
      - late_residual
      - policy_head
      - value_head

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Expected Outcomes
- May show if tactical patterns are shared across playstyles
- Unusual divergence pattern (middle shared, edges divergent)
- Exploratory experiment to understand layer roles

### 5.4 Experiment P3: Share Late Layers Only

**Description:** Share late residual layers and heads only

This is a counter-hypothesis test: what if high-level strategy is universal and only low-level features differ?

#### Configuration
```yaml
experiment_id: P3_share_late
experiment_type: partial_sharing

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20

aggregation:
  method: selective_layer_sharing
  layer_sharing:
    mode: partial
    shared_layers:
      - late_residual
      - policy_head
      - value_head
    cluster_specific_layers:
      - input_block
      - early_residual
      - middle_residual

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Expected Outcomes
- Likely poor performance (heads should encode playstyle differences)
- Minimal playstyle separation (shared decision making)
- Control experiment to validate hypothesis

### 5.5 Experiment P4: Share All Except Heads

**Description:** Share entire backbone, keep only policy/value heads cluster-specific

This tests the hypothesis that playstyle differences are primarily encoded in the output heads.

#### Configuration
```yaml
experiment_id: P4_share_backbone
experiment_type: partial_sharing

training:
  total_rounds: 200
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64

clustering:
  method: playstyle_kmeans
  num_clusters: 2
  cluster_names: ["cluster_tactical", "cluster_positional"]
  reassignment_frequency: 20

aggregation:
  method: selective_layer_sharing
  layer_sharing:
    mode: partial
    shared_layers:
      - input_block
      - early_residual
      - middle_residual
      - late_residual
    cluster_specific_layers:
      - policy_head
      - value_head

metrics_collection:
  playstyle_evaluation: true
  weight_statistics: true
  cluster_divergence: true
  move_type_distribution:
    enabled: true
    frequency: 10
  elo_estimation:
    enabled: true
    frequency: 10
```

#### Expected Outcomes
- Shared backbone: Near-zero divergence for residual layers
- Heads: Significant divergence
- Good playstyle separation IF hypothesis correct
- Potentially best of both worlds (shared features + specialized decisions)

### 5.6 Partial Sharing Comparison Summary

| Experiment | Shared Layers | Cluster-Specific | Hypothesis |
|------------|---------------|------------------|------------|
| P1 | Early (input + early_res) | Middle + Late + Heads | Low-level features universal |
| P2 | Middle only | All others | Tactical patterns universal |
| P3 | Late + Heads | Early + Middle | Strategy universal (control) |
| P4 | All backbone | Heads only | Playstyle in output heads |

#### Key Questions to Answer

1. **Which layers encode playstyle differences?**
   - Compare divergence patterns across experiments
   - Higher divergence in cluster-specific layers indicates playstyle encoding

2. **What is the optimal sharing strategy?**
   - Compare ELO across P1-P4
   - Best ELO indicates optimal balance of sharing vs specialization

3. **Is there a trade-off between ELO and playstyle separation?**
   - Plot ELO vs playstyle separation for all experiments
   - Identify Pareto-optimal configurations

---

## 6. Phase 3: Performance Evaluation

### 6.1 Overview

**Duration:** ~8 days
**GPU Hours:** ~40
**Purpose:** Deep analysis of trained models

### 6.2 Evaluation E1: Playstyle Analysis

**Description:** Comprehensive playstyle characterization

#### Procedure

1. **Load final models** from all experiments (B1, B2, P1-P4)
2. **Generate 500 games** per cluster via self-play
3. **Run PlaystyleEvaluator** on all games
4. **Compute statistics:**
   - Mean tactical score per cluster
   - Standard deviation
   - Classification distribution
   - Core metrics breakdown

#### Expected Output

```json
{
  "evaluation_id": "E1_playstyle",
  "clusters": {
    "cluster_tactical": {
      "games_analyzed": 500,
      "tactical_score": {
        "mean": 0.68,
        "std": 0.12
      },
      "classification": "Tactical",
      "separation_from_other": 0.23
    },
    "cluster_positional": {
      "games_analyzed": 500,
      "tactical_score": {
        "mean": 0.45,
        "std": 0.11
      },
      "classification": "Balanced-Positional",
      "separation_from_other": 0.23
    }
  },
  "statistical_significance": {
    "t_statistic": 28.5,
    "p_value": 1.2e-89,
    "effect_size": 1.92
  }
}
```

#### Hypotheses Validated
- H3: Playstyle clusters emerge naturally
- H4: Different clusters develop distinct strategies

### 6.3 Evaluation E2: Move Type Analysis

**Description:** Behavioral pattern analysis

#### Procedure

1. **Load final models** from all experiments (B1, B2, P1-P4)
2. **Generate 200 games** per cluster via self-play
3. **Run MoveTypeAnalyzer** on all games
4. **Compute cluster comparison:**
   - Per-cluster move type percentages
   - Aggressive move ratio
   - Cluster comparison

#### Expected Output

```json
{
  "evaluation_id": "E2_move_types",
  "clusters": {
    "cluster_tactical": {
      "aggressive_pct": 16.2,
      "captures_pct": 13.5,
      "checks_pct": 2.7,
      "quiet_moves_pct": 55.8
    },
    "cluster_positional": {
      "aggressive_pct": 11.8,
      "captures_pct": 10.2,
      "checks_pct": 1.6,
      "quiet_moves_pct": 64.2
    }
  },
  "comparison": {
    "aggressive_diff": 4.4,
    "tactical_cluster": "cluster_tactical",
    "positional_cluster": "cluster_positional"
  }
}
```

#### Hypotheses Validated
- H4: Different clusters develop distinct strategies
- H10: Behavioral differences are measurable

### 6.4 Evaluation E3: Generalization Test

**Description:** Test model generalization to new positions

#### Procedure

1. **Prepare benchmark positions:**
   - 100 tactical puzzles (mate-in-N, forks, pins)
   - 100 positional puzzles (endgame, pawn structure)
   - 100 mixed complexity positions

2. **Evaluate each cluster model:**
   - Policy accuracy on correct moves
   - Value estimation accuracy
   - Time to solve (MCTS iterations)

3. **Cross-test:**
   - Tactical model on positional puzzles
   - Positional model on tactical puzzles

#### Expected Output

```json
{
  "evaluation_id": "E3_generalization",
  "tactical_puzzles": {
    "cluster_tactical": {"accuracy": 0.78, "avg_iterations": 450},
    "cluster_positional": {"accuracy": 0.62, "avg_iterations": 680}
  },
  "positional_puzzles": {
    "cluster_tactical": {"accuracy": 0.65, "avg_iterations": 520},
    "cluster_positional": {"accuracy": 0.74, "avg_iterations": 380}
  },
  "mixed_puzzles": {
    "cluster_tactical": {"accuracy": 0.71},
    "cluster_positional": {"accuracy": 0.68}
  }
}
```

#### Hypotheses Validated
- H9: Framework generalizes to new positions

---

## 7. Post-Experiment Analysis

> **Note:** The following analyses are computed AFTER experiments complete, using the saved JSON metric files. They do not require running additional training.

### 7.1 Plateau Detection

**Purpose:** Identify when clusters stop improving (convergence)

**Input Files:**
- `storage/metrics/{run_id}/cluster_*/evaluation_round_*.json`
- `storage/metrics/{run_id}/model_divergence/round_*.json`

#### Algorithm

```python
def detect_plateau(metric_series, window=20, threshold=0.01):
    """
    Detect plateau in metric time series.

    Args:
        metric_series: List of (round, value) tuples
        window: Rolling window size
        threshold: Minimum improvement to not be plateau

    Returns:
        plateau_start_round: Round where plateau begins (or None)
    """
    for i in range(window, len(metric_series)):
        recent = metric_series[i-window:i]
        improvement = max(recent) - min(recent)
        if improvement < threshold:
            return metric_series[i-window][0]  # Round number
    return None
```

#### Metrics to Analyze
- Tactical score mean per cluster
- ELO rating per cluster
- Cluster divergence

#### Output

```json
{
  "plateau_analysis": {
    "cluster_tactical": {
      "tactical_score_plateau": 145,
      "elo_plateau": 160,
      "converged": true
    },
    "cluster_positional": {
      "tactical_score_plateau": 138,
      "elo_plateau": 155,
      "converged": true
    },
    "divergence_plateau": 120
  }
}
```

### 7.2 Correlation Analysis

**Purpose:** Discover relationships between metrics

**Input Files:**
- All metric JSON files from experiment run

#### Correlations to Compute

| Metric A | Metric B | Expected Correlation |
|----------|----------|---------------------|
| Tactical Score | Aggressive Move % | Positive |
| Divergence | Training Round | Positive (early) |
| Weight L2 Norm | ELO | Unclear |
| Policy Head Divergence | Playstyle Separation | Positive |
| Cross-Cluster Weight | ELO Improvement | Positive |

#### Algorithm

```python
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_correlations(metrics_df):
    """
    Compute correlation matrix for all metric pairs.

    Returns:
        DataFrame with correlation coefficients and p-values
    """
    correlations = []
    columns = metrics_df.select_dtypes(include='number').columns

    for col1 in columns:
        for col2 in columns:
            if col1 < col2:  # Avoid duplicates
                r, p = pearsonr(metrics_df[col1], metrics_df[col2])
                correlations.append({
                    'metric_a': col1,
                    'metric_b': col2,
                    'pearson_r': r,
                    'p_value': p,
                    'significant': p < 0.05
                })

    return pd.DataFrame(correlations)
```

#### Output

```json
{
  "correlation_analysis": {
    "tactical_score_vs_aggressive_pct": {
      "pearson_r": 0.87,
      "p_value": 1.2e-45,
      "significant": true
    },
    "divergence_vs_round": {
      "pearson_r": 0.72,
      "p_value": 3.4e-28,
      "significant": true
    },
    "policy_divergence_vs_playstyle_separation": {
      "pearson_r": 0.91,
      "p_value": 8.9e-52,
      "significant": true
    }
  }
}
```

### 7.3 Trajectory Analysis

**Purpose:** Visualize how metrics evolve over training

**Input Files:**
- Time series data from all metrics

#### Visualizations to Generate

1. **Playstyle Trajectory Plot**
   - X-axis: Training round
   - Y-axis: Tactical score
   - Lines: One per cluster
   - Shaded: Standard deviation band

2. **Divergence Heatmap**
   - X-axis: Training round
   - Y-axis: Layer group
   - Color: Divergence value

3. **Move Type Evolution**
   - Stacked area chart
   - Shows how move type percentages change over training

4. **ELO Progression**
   - X-axis: Training round
   - Y-axis: ELO rating
   - Lines: One per cluster
   - Error bars: Rating deviation

---

## 8. Statistical Analysis

### 8.1 Required Tests

| Comparison | Test | Purpose |
|------------|------|---------|
| P1-P4 vs B1/B2 | Paired t-test | Validate improvement over baselines |
| Cluster playstyles | Independent t-test | Validate separation |
| Layer sharing effects | ANOVA | Compare P1-P4 configurations |
| Correlation significance | Pearson/Spearman | Validate relationships |

### 8.2 Effect Size Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Cohen's d | (μ₁ - μ₂) / σ_pooled | 0.2=small, 0.5=medium, 0.8=large |
| η² (eta-squared) | SS_between / SS_total | Variance explained |
| r (correlation) | Pearson r | -1 to 1 strength |

### 8.3 Reporting Requirements

For each hypothesis, report:
1. Sample sizes (n)
2. Descriptive statistics (mean, std)
3. Test statistic and degrees of freedom
4. p-value (exact, not just < 0.05)
5. Effect size with interpretation
6. 95% confidence intervals

### 8.4 Multiple Comparison Correction

When performing multiple statistical tests, apply Bonferroni correction:

```python
corrected_alpha = 0.05 / num_tests
```

---

## 9. Results Interpretation

### 9.1 Hypothesis Validation Criteria

| Hypothesis | Validation Criterion |
|------------|---------------------|
| H1 | Best partial sharing ELO > B1 and B2 ELO (p < 0.05) |
| H2 | Partial sharing divergence between B1 (~0) and B2 (high) |
| H3 | Cluster tactical scores significantly different (p < 0.001) |
| H4 | Move type distributions significantly different |
| H5 | Partial sharing ELO improvement > B2 (no sharing) |
| H6 | Layer sharing strategy scales to more clusters |
| H7 | Divergence stabilizes (plateau detected) |
| H8 | Per-client ELO gains positive for >80% of clients |
| H9 | Cross-domain accuracy > 60% |
| H10 | Effect size > 0.5 for behavioral metrics |

### 9.2 Expected Results Summary

| Experiment | Expected Outcome |
|------------|------------------|
| B1 (Full Sharing) | Baseline ELO, ~0 divergence, no specialization |
| B2 (No Sharing) | Lower ELO, high divergence, max specialization |
| P1 (Share Early) | Good ELO, moderate divergence in late layers |
| P2 (Share Middle) | Exploratory - understand tactical layer sharing |
| P3 (Share Late) | Poor separation (control experiment) |
| P4 (Share Backbone) | Potentially best - shared features + specialized heads |

### 9.3 Failure Modes

| Observation | Possible Cause | Action |
|-------------|---------------|--------|
| No playstyle separation | Insufficient client diversity | Check data distribution |
| P1-P4 similar to B1 | Shared layers dominating | Try sharing fewer layers |
| P1-P4 similar to B2 | Cluster-specific layers dominating | Try sharing more layers |
| ELO not improving | Learning rate issues | Tune hyperparameters |
| Unstable divergence | Training instability | Check gradient norms |

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue:** Metrics files not being saved
```bash
# Check storage directory permissions
ls -la storage/metrics/

# Verify experiment tracker initialization
grep "Initialized FileExperimentTracker" logs/server.log
```

**Issue:** Cluster divergence is zero
```bash
# Verify multiple clusters exist
ls storage/models/{run_id}/

# Check if models are actually different
python -c "
import torch
m1 = torch.load('storage/models/{run_id}/cluster_tactical/round_50.pt')
m2 = torch.load('storage/models/{run_id}/cluster_positional/round_50.pt')
print('Models identical:', all(torch.equal(m1[k], m2[k]) for k in m1))
"
```

**Issue:** ELO estimation failing
```bash
# Check self-play is working
grep "Self-play match" logs/server.log

# Verify MCTS configuration
cat configs/mcts.yaml
```

### 10.2 Metric Validation

Run validation checks after each experiment:

```python
async def validate_metrics(run_id: str):
    """Validate all metrics were collected correctly."""

    checks = []

    # Check playstyle metrics exist
    playstyle_files = list(Path(f"storage/metrics/{run_id}").glob("*/evaluation_round_*.json"))
    checks.append(("Playstyle files", len(playstyle_files) > 0))

    # Check divergence metrics
    divergence_files = list(Path(f"storage/metrics/{run_id}/model_divergence").glob("round_*.json"))
    checks.append(("Divergence files", len(divergence_files) > 0))

    # Check weight statistics
    weight_files = list(Path(f"storage/metrics/{run_id}").glob("*/weight_stats_round_*.json"))
    checks.append(("Weight stats files", len(weight_files) > 0))

    # Check move type metrics
    move_files = list(Path(f"storage/metrics/{run_id}").glob("*/move_types_round_*.json"))
    checks.append(("Move type files", len(move_files) > 0))

    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    return all(passed for _, passed in checks)
```

### 10.3 Recovery Procedures

**Recover from interrupted experiment:**
```python
# Load last checkpoint
tracker = FileExperimentTracker(...)
run_info = await tracker.get_run_info(run_id)
last_round = run_info.get("last_completed_round", 0)

# Resume training from checkpoint
await trainer.resume_from_round(last_round)
```

**Regenerate missing metrics:**
```python
# Load model checkpoint
model_state, metadata = await tracker.load_checkpoint(run_id, cluster_id, round_num)

# Regenerate playstyle evaluation
evaluator = PlaystyleEvaluator(model)
games = await generate_self_play_games(model, num_games=100)
metrics = evaluator.evaluate_games(games)
await tracker.log_playstyle_evaluation(run_id, round_num, cluster_id, metrics)
```

---

## Appendix A: Quick Reference

### A.1 Experiment Commands

```bash
# Start baseline experiments
python run_experiment.py --config configs/experiments/B1_full_sharing.yaml
python run_experiment.py --config configs/experiments/B2_no_sharing.yaml

# Start partial layer sharing experiments
python run_experiment.py --config configs/experiments/P1_share_early.yaml
python run_experiment.py --config configs/experiments/P2_share_middle.yaml
python run_experiment.py --config configs/experiments/P3_share_late.yaml
python run_experiment.py --config configs/experiments/P4_share_backbone.yaml

# Run post-experiment analysis
python analyze_results.py --run-id {run_id} --analysis all
```

### A.2 Metric Access Patterns

```python
# Load playstyle metrics for a cluster
import json
with open(f"storage/metrics/{run_id}/{cluster_id}/evaluation_round_{round_num}.json") as f:
    playstyle = json.load(f)

# Load divergence metrics
with open(f"storage/metrics/{run_id}/model_divergence/round_{round_num}.json") as f:
    divergence = json.load(f)

# Load weight statistics
with open(f"storage/metrics/{run_id}/{cluster_id}/weight_stats_round_{round_num}.json") as f:
    weight_stats = json.load(f)
```

### A.3 Key File Locations

| Component | Path |
|-----------|------|
| Experiment configs | `configs/experiments/*.yaml` |
| Model checkpoints | `storage/models/{run_id}/{cluster_id}/` |
| Playstyle metrics | `storage/metrics/{run_id}/{cluster_id}/evaluation_round_*.json` |
| Divergence metrics | `storage/metrics/{run_id}/model_divergence/round_*.json` |
| Weight statistics | `storage/metrics/{run_id}/{cluster_id}/weight_stats_round_*.json` |
| Move type metrics | `storage/metrics/{run_id}/{cluster_id}/move_types_round_*.json` |
| Run metadata | `storage/.metadata/{run_id}.json` |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2025-01-XX | Restructured to two baselines (Full/No Sharing) + partial layer sharing experiments |
| 1.0 | 2025-01-XX | Initial comprehensive evaluation guide |

---

*This guide is part of the Clustered Federated Deep Reinforcement Learning framework documentation.*
