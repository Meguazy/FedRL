# Model Analysis Metrics Documentation

This document provides a comprehensive explanation of the model-level analysis metrics used to understand the internal behavior of the AlphaZero neural network in the federated learning chess framework. These metrics complement the playstyle metrics by providing insights into **how** the model learns, not just **what** it outputs.

## Table of Contents

1. [Overview](#overview)
2. [Motivation and Research Questions](#motivation-and-research-questions)
3. [Network Architecture Reference](#network-architecture-reference)
4. [Cluster Divergence Metrics](#cluster-divergence-metrics)
5. [Weight Statistics Metrics](#weight-statistics-metrics)
6. [Implementation Plan](#implementation-plan)
7. [Expected Results and Hypotheses](#expected-results-and-hypotheses)
8. [Storage Structure](#storage-structure)
9. [Future Extensions](#future-extensions)

---

## Overview

The model analysis metrics system examines the **internal state** of neural networks to understand:

1. **How clusters diverge** - Which layers become cluster-specific vs. remain shared
2. **Weight evolution** - How weights change during training
3. **Learning dynamics** - Are all layers learning? Are any dead?

These metrics are crucial for validating the core thesis hypothesis: **Selective aggregation preserves playstyle by allowing cluster-specific layers to diverge while sharing foundational chess knowledge.**

### Metrics Categories

| Category | Purpose | Complexity |
|----------|---------|------------|
| Cluster Divergence | Measure differentiation between tactical/positional models | Medium |
| Weight Statistics | Track weight distributions and changes | Low |
| Activation Analysis | Understand neuron behavior (future) | High |
| Explainability | Interpret model decisions (future) | High |

---

## Motivation and Research Questions

### Core Research Questions

1. **Q1: Do clusters actually diverge?**
   - If selective aggregation works, tactical and positional models should become different over training
   - Measure: Cosine similarity between models should decrease over rounds

2. **Q2: Which layers diverge most?**
   - Hypothesis: Early layers (shared chess fundamentals) stay similar; late layers (playstyle-specific) diverge
   - Measure: Per-layer divergence analysis

3. **Q3: Do policy and value heads diverge differently?**
   - Policy head: Move selection (should diverge - different preferred moves)
   - Value head: Position evaluation (may stay similar - both understand winning/losing)
   - Measure: Separate divergence metrics for each head

4. **Q4: Is the model actually learning?**
   - Weights should change over training
   - No layers should be "dead" (zero gradients)
   - Measure: Weight change magnitude per layer per round

### Why These Metrics Matter for the Thesis

| Thesis Claim | Supporting Metric |
|--------------|-------------------|
| "Selective aggregation preserves playstyle" | Cluster divergence increases over training |
| "Shared layers transfer knowledge" | Early residual blocks stay similar |
| "Cluster-specific layers capture style" | Late blocks + heads diverge significantly |
| "Framework produces distinct playing styles" | Policy head divergence + playstyle metrics |

---

## Network Architecture Reference

### AlphaZero Network Structure

```
AlphaZeroNet (22.9M parameters)
│
├── input_conv (Conv2d)          [119 → 256 channels, 3x3]
├── input_bn (BatchNorm2d)       [256 channels]
│
├── residual (ModuleDict)        [19 ResidualBlocks]
│   ├── residual.0               [First block - most foundational]
│   │   ├── conv1 (Conv2d)       [256 → 256, 3x3]
│   │   ├── bn1 (BatchNorm2d)
│   │   ├── conv2 (Conv2d)       [256 → 256, 3x3]
│   │   └── bn2 (BatchNorm2d)
│   ├── residual.1
│   │   └── ...
│   └── residual.18              [Last block - most specialized]
│
├── policy_head (PolicyHead)
│   ├── conv (Conv2d)            [256 → 73, 3x3]
│   └── bn (BatchNorm2d)
│
└── value_head (ValueHead)
    ├── conv (Conv2d)            [256 → 1, 1x1]
    ├── bn (BatchNorm2d)
    ├── fc1 (Linear)             [64 → 256]
    └── fc2 (Linear)             [256 → 1]
```

### Layer Groups for Analysis

| Group | Layers | Expected Behavior | Rationale |
|-------|--------|-------------------|-----------|
| **Input Block** | `input_conv`, `input_bn` | Shared (low divergence) | Basic board feature extraction |
| **Early Residual** | `residual.0` - `residual.5` | Mostly shared | Fundamental chess patterns |
| **Middle Residual** | `residual.6` - `residual.12` | Mixed | Transition zone |
| **Late Residual** | `residual.13` - `residual.18` | Divergent | Style-specific patterns |
| **Policy Head** | `policy_head.*` | Highly divergent | Move preferences differ by style |
| **Value Head** | `value_head.*` | Moderately divergent | Position evaluation may be similar |

---

## Cluster Divergence Metrics

### 1. Layer-wise Cosine Similarity

**Definition**: Measures how similar the weights of two models are for each layer.

**Formula**:
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
- A = flattened weight tensor from model A (tactical)
- B = flattened weight tensor from model B (positional)
- · = dot product
- ||x|| = L2 norm
```

**Range**: [-1, 1]
- 1.0 = Identical weights
- 0.0 = Orthogonal (completely different)
- -1.0 = Opposite (rare in practice)

**Per-Layer Calculation**:
```python
def cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """Compute cosine similarity between two weight tensors."""
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    dot_product = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Handle zero tensors

    return (dot_product / (norm_a * norm_b)).item()
```

**Interpretation**:
- `similarity > 0.99`: Layers are nearly identical (effective sharing)
- `similarity ∈ [0.90, 0.99]`: Minor divergence (some specialization)
- `similarity ∈ [0.70, 0.90]`: Moderate divergence (clear differentiation)
- `similarity < 0.70`: Strong divergence (very different representations)

---

### 2. Layer-wise L2 Distance (Weight Divergence)

**Definition**: Euclidean distance between weight tensors, measuring absolute difference.

**Formula**:
```
l2_distance(A, B) = ||A - B||₂ = √(Σᵢ(aᵢ - bᵢ)²)
```

**Normalized Version** (for cross-layer comparison):
```
normalized_l2(A, B) = ||A - B||₂ / √(||A||₂² + ||B||₂²)
```

**Range**:
- Raw: [0, ∞)
- Normalized: [0, √2] ≈ [0, 1.41]

**Per-Layer Calculation**:
```python
def l2_distance(tensor_a: torch.Tensor, tensor_b: torch.Tensor, normalize: bool = True) -> float:
    """Compute L2 distance between two weight tensors."""
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    diff = a_flat - b_flat
    l2 = torch.norm(diff).item()

    if normalize:
        norm_sum = torch.sqrt(torch.norm(a_flat)**2 + torch.norm(b_flat)**2).item()
        if norm_sum > 0:
            l2 = l2 / norm_sum

    return l2
```

**Interpretation**:
- `normalized_l2 < 0.1`: Very similar weights
- `normalized_l2 ∈ [0.1, 0.3]`: Moderate difference
- `normalized_l2 > 0.3`: Significant divergence

---

### 3. Divergence Index (Composite Metric)

**Definition**: Combined metric that captures both direction (cosine) and magnitude (L2) of divergence.

**Formula**:
```
divergence_index = (1 - cosine_similarity) × (1 + normalized_l2)
```

**Range**: [0, ~2.8]
- 0 = Identical
- Higher = More divergent

**Rationale**:
- Cosine similarity alone misses magnitude differences (two vectors pointing same direction but different lengths)
- L2 alone doesn't capture directional alignment
- Combined metric is more robust

---

### 4. Aggregated Divergence Metrics

**Per-Layer-Group Aggregation**:
```python
layer_groups = {
    "input_block": ["input_conv.weight", "input_bn.weight", "input_bn.bias"],
    "early_residual": [f"residual.{i}.*" for i in range(6)],
    "middle_residual": [f"residual.{i}.*" for i in range(6, 13)],
    "late_residual": [f"residual.{i}.*" for i in range(13, 19)],
    "policy_head": ["policy_head.*"],
    "value_head": ["value_head.*"]
}

# Aggregate by averaging metrics within each group
group_divergence = {
    group_name: mean([divergence_index(layer) for layer in layers])
    for group_name, layers in layer_groups.items()
}
```

**Global Divergence**:
```
global_divergence = mean(all layer divergence indices)
```

---

### 5. Temporal Divergence (Change Over Rounds)

**Definition**: Track how divergence evolves during training.

**Metrics**:
```
divergence_velocity = divergence(round_n) - divergence(round_n-10)
divergence_acceleration = velocity(round_n) - velocity(round_n-10)
```

**Interpretation**:
- Positive velocity: Clusters are diverging (good - developing distinct styles)
- Zero velocity: Divergence stabilized (training converged)
- Negative velocity: Clusters converging (aggregation dominating)

---

## Weight Statistics Metrics

### 1. Per-Layer Weight Distribution

**Metrics per layer**:
```python
weight_stats = {
    "mean": tensor.mean().item(),
    "std": tensor.std().item(),
    "min": tensor.min().item(),
    "max": tensor.max().item(),
    "l2_norm": torch.norm(tensor).item(),
    "sparsity": (tensor.abs() < 1e-6).float().mean().item(),  # % near-zero
}
```

**Interpretation**:
- **Mean ≈ 0**: Weights are centered (typical for well-trained networks)
- **Std**: Scale of weights (should be stable, not exploding/vanishing)
- **Sparsity > 0.1**: Many near-zero weights (potential pruning candidates)

---

### 2. Weight Change Magnitude (Per Round)

**Definition**: How much weights changed from previous checkpoint.

**Formula**:
```
weight_change = ||W_current - W_previous||₂ / ||W_previous||₂
```

**Per-Layer Calculation**:
```python
def weight_change_ratio(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Compute relative weight change magnitude."""
    diff_norm = torch.norm(current - previous).item()
    prev_norm = torch.norm(previous).item()

    if prev_norm == 0:
        return float('inf') if diff_norm > 0 else 0.0

    return diff_norm / prev_norm
```

**Interpretation**:
- `change > 0.1`: Significant learning happening
- `change ∈ [0.01, 0.1]`: Normal learning
- `change < 0.001`: Layer barely changing (may be frozen or dead)

---

### 3. Gradient Flow Indicators

**Definition**: Indirect measures of gradient health (computed from weight changes).

**Dead Layer Detection**:
```python
def is_layer_dead(weight_change_history: List[float], threshold: float = 0.001) -> bool:
    """Check if layer has stopped learning."""
    recent_changes = weight_change_history[-5:]  # Last 5 checkpoints
    return all(change < threshold for change in recent_changes)
```

**Interpretation**:
- Dead layers indicate gradient flow problems
- May need skip connections or learning rate adjustments

---

## Implementation Plan

### Phase 1: Cluster Divergence (Priority: HIGH)

**When computed**: Every 10 rounds (alongside playstyle evaluation)

**Input**: Two model state dicts (tactical, positional)

**Output**: JSON file with per-layer and aggregated divergence metrics

**Storage**: `storage/metrics/{run_id}/model_divergence/round_{N}.json`

**Algorithm**:
```python
async def compute_cluster_divergence(
    model_a: Dict[str, torch.Tensor],  # tactical
    model_b: Dict[str, torch.Tensor],  # positional
    round_num: int
) -> Dict[str, Any]:
    """
    Compute divergence metrics between two cluster models.
    """
    results = {
        "round_num": round_num,
        "per_layer": {},
        "per_group": {},
        "global": {}
    }

    # Per-layer metrics
    for layer_name in model_a.keys():
        if layer_name not in model_b:
            continue

        tensor_a = model_a[layer_name]
        tensor_b = model_b[layer_name]

        cos_sim = cosine_similarity(tensor_a, tensor_b)
        l2_dist = l2_distance(tensor_a, tensor_b, normalize=True)
        div_idx = (1 - cos_sim) * (1 + l2_dist)

        results["per_layer"][layer_name] = {
            "cosine_similarity": cos_sim,
            "l2_distance_normalized": l2_dist,
            "divergence_index": div_idx,
            "tensor_shape": list(tensor_a.shape),
            "num_parameters": tensor_a.numel()
        }

    # Aggregate by layer group
    results["per_group"] = aggregate_by_group(results["per_layer"])

    # Global metrics
    all_divergences = [v["divergence_index"] for v in results["per_layer"].values()]
    results["global"] = {
        "mean_divergence": np.mean(all_divergences),
        "max_divergence": np.max(all_divergences),
        "min_divergence": np.min(all_divergences),
        "std_divergence": np.std(all_divergences)
    }

    return results
```

---

### Phase 2: Weight Statistics (Priority: MEDIUM)

**When computed**: Every 10 rounds

**Input**: Single model state dict

**Output**: Per-cluster weight statistics

**Storage**: `storage/metrics/{run_id}/{cluster}/weight_stats_round_{N}.json`

**Algorithm**:
```python
async def compute_weight_statistics(
    model: Dict[str, torch.Tensor],
    previous_model: Optional[Dict[str, torch.Tensor]],
    round_num: int
) -> Dict[str, Any]:
    """
    Compute weight statistics for a single model.
    """
    results = {
        "round_num": round_num,
        "per_layer": {},
        "summary": {}
    }

    total_params = 0
    total_near_zero = 0

    for layer_name, tensor in model.items():
        stats = {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "l2_norm": torch.norm(tensor).item(),
            "num_parameters": tensor.numel(),
            "sparsity": (tensor.abs() < 1e-6).float().mean().item()
        }

        # Weight change from previous round
        if previous_model and layer_name in previous_model:
            stats["weight_change"] = weight_change_ratio(
                tensor, previous_model[layer_name]
            )

        results["per_layer"][layer_name] = stats
        total_params += tensor.numel()
        total_near_zero += (tensor.abs() < 1e-6).sum().item()

    results["summary"] = {
        "total_parameters": total_params,
        "global_sparsity": total_near_zero / total_params,
        "layers_analyzed": len(model)
    }

    return results
```

---

## Expected Results and Hypotheses

### Hypothesis 1: Layer-Depth Divergence Gradient

**Prediction**: Divergence increases with layer depth.

**Expected Pattern**:
```
Layer Group          | Cosine Similarity | Divergence Index
---------------------|-------------------|------------------
input_block          | 0.98 - 0.99       | 0.01 - 0.03
early_residual (0-5) | 0.95 - 0.98       | 0.03 - 0.08
middle_residual      | 0.85 - 0.95       | 0.08 - 0.20
late_residual        | 0.70 - 0.85       | 0.20 - 0.45
policy_head          | 0.50 - 0.75       | 0.35 - 0.75
value_head           | 0.75 - 0.90       | 0.15 - 0.35
```

**If not observed**: Selective aggregation may not be working as intended, or training duration insufficient.

---

### Hypothesis 2: Policy Head Shows Maximum Divergence

**Prediction**: Policy head diverges most because tactical/positional players prefer different moves.

**Supporting Evidence**:
- Tactical: Prefers aggressive captures, checks, threats
- Positional: Prefers pawn advances, piece maneuvers, prophylaxis

**Measurement**: `policy_head.conv.weight` should have lowest cosine similarity.

---

### Hypothesis 3: Divergence Correlates with Playstyle Difference

**Prediction**: Higher model divergence correlates with higher playstyle divergence (from playstyle metrics).

**Formula**:
```
correlation = pearson(
    [model_divergence at round 10, 20, 30, ...],
    [playstyle_divergence at round 10, 20, 30, ...]
)
```

**Expected**: Strong positive correlation (r > 0.7)

---

### Hypothesis 4: Divergence Stabilizes

**Prediction**: After sufficient training, divergence reaches plateau.

**Expected Pattern**:
```
Round 10:  Low divergence (models still similar from init)
Round 50:  Increasing divergence (clusters specializing)
Round 100: Divergence plateau (distinct styles established)
Round 150: Stable (no further increase)
```

---

## Storage Structure

### Directory Layout

```
storage/metrics/{run_id}/
├── tactical/
│   ├── evaluation_round_10.json      # Playstyle metrics
│   ├── evaluation_round_20.json
│   ├── weight_stats_round_10.json    # Weight statistics
│   └── weight_stats_round_20.json
├── positional/
│   ├── evaluation_round_10.json
│   ├── evaluation_round_20.json
│   ├── weight_stats_round_10.json
│   └── weight_stats_round_20.json
└── model_divergence/
    ├── round_10.json                  # Cluster divergence
    ├── round_20.json
    └── summary.json                   # Aggregated over all rounds
```

### JSON Schema: Cluster Divergence

```json
{
  "round_num": 10,
  "timestamp": "2025-11-22T19:19:27.289926",
  "clusters_compared": ["cluster_tactical", "cluster_positional"],

  "per_layer": {
    "input_conv.weight": {
      "cosine_similarity": 0.9847,
      "l2_distance_normalized": 0.0234,
      "divergence_index": 0.0389,
      "tensor_shape": [256, 119, 3, 3],
      "num_parameters": 274176
    },
    "residual.0.conv1.weight": {
      "cosine_similarity": 0.9712,
      "l2_distance_normalized": 0.0456,
      "divergence_index": 0.0601
    },
    // ... more layers
    "policy_head.conv.weight": {
      "cosine_similarity": 0.6823,
      "l2_distance_normalized": 0.2341,
      "divergence_index": 0.3921
    }
  },

  "per_group": {
    "input_block": {
      "mean_cosine_similarity": 0.9823,
      "mean_divergence_index": 0.0412,
      "num_layers": 3
    },
    "early_residual": {
      "mean_cosine_similarity": 0.9634,
      "mean_divergence_index": 0.0723
    },
    "middle_residual": {
      "mean_cosine_similarity": 0.8921,
      "mean_divergence_index": 0.1456
    },
    "late_residual": {
      "mean_cosine_similarity": 0.7834,
      "mean_divergence_index": 0.2891
    },
    "policy_head": {
      "mean_cosine_similarity": 0.6912,
      "mean_divergence_index": 0.3812
    },
    "value_head": {
      "mean_cosine_similarity": 0.8234,
      "mean_divergence_index": 0.2134
    }
  },

  "global": {
    "mean_divergence": 0.1823,
    "max_divergence": 0.4123,
    "min_divergence": 0.0234,
    "std_divergence": 0.1234,
    "total_parameters_compared": 22892949
  }
}
```

### JSON Schema: Weight Statistics

```json
{
  "round_num": 10,
  "timestamp": "2025-11-22T19:19:27.289926",
  "cluster_id": "cluster_tactical",

  "per_layer": {
    "input_conv.weight": {
      "mean": 0.00234,
      "std": 0.0891,
      "min": -0.3421,
      "max": 0.3567,
      "l2_norm": 12.456,
      "num_parameters": 274176,
      "sparsity": 0.0023,
      "weight_change": 0.0456
    },
    // ... more layers
  },

  "summary": {
    "total_parameters": 22892949,
    "global_sparsity": 0.0034,
    "layers_analyzed": 78,
    "dead_layers": [],
    "highly_active_layers": ["residual.15.conv1.weight", "policy_head.conv.weight"]
  }
}
```

---

## Future Extensions

### 1. Activation Analysis (Complexity: High)

**Goal**: Understand how neurons respond to different chess positions and whether tactical/positional clusters develop distinct activation patterns.

#### 1.1 Theoretical Background

Neural network activations represent the intermediate computations between input and output. In the context of chess:

- **Early layer activations**: Encode basic board features (piece positions, attack patterns)
- **Middle layer activations**: Encode tactical motifs (pins, forks, discovered attacks)
- **Late layer activations**: Encode strategic concepts (pawn structure evaluation, king safety)

By analyzing activations, we can understand **what the model "sees"** at each processing stage.

#### 1.2 Method: Forward Hook Registration

PyTorch allows registering hooks on layers to capture activations during forward pass:

```python
class ActivationRecorder:
    """Records activations from specified layers during forward pass."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)

    def _create_hook(self, layer_name: str):
        def hook(module, input, output):
            # Store activation tensor (detached to avoid memory leaks)
            self.activations[layer_name] = output.detach().cpu()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

#### 1.3 Metrics to Compute

**Per-Layer Activation Statistics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Mean Activation | `μ = mean(activations)` | Baseline activity level |
| Activation Std | `σ = std(activations)` | Variability in responses |
| Sparsity | `% where activation ≈ 0` | How selective the layer is |
| Max Activation | `max(activations)` | Strongest response |
| Activation Entropy | `H = -Σ p(a) log p(a)` | Diversity of activation patterns |

**Per-Neuron Analysis**:

```python
def compute_neuron_statistics(activations: torch.Tensor) -> Dict:
    """
    Compute statistics for each neuron in a convolutional layer.

    Args:
        activations: Shape (batch, channels, height, width)

    Returns:
        Statistics per channel/neuron
    """
    # Average over spatial dimensions and batch
    per_channel_mean = activations.mean(dim=(0, 2, 3))  # Shape: (channels,)
    per_channel_std = activations.std(dim=(0, 2, 3))

    # Dead neuron detection (never activates above threshold)
    dead_threshold = 1e-6
    dead_neurons = (per_channel_mean.abs() < dead_threshold).sum().item()

    # Highly active neurons
    active_threshold = per_channel_mean.mean() + 2 * per_channel_mean.std()
    highly_active = (per_channel_mean > active_threshold).sum().item()

    return {
        "per_channel_mean": per_channel_mean.numpy().tolist(),
        "per_channel_std": per_channel_std.numpy().tolist(),
        "dead_neurons_count": dead_neurons,
        "dead_neurons_percent": dead_neurons / len(per_channel_mean) * 100,
        "highly_active_count": highly_active
    }
```

#### 1.4 Comparative Analysis: Tactical vs Positional

**Hypothesis**: Tactical and positional clusters should have different activation patterns for the same board positions.

**Method**:
1. Select a set of **benchmark positions** (e.g., 100 diverse middlegame positions)
2. Run both models on the same positions
3. Compare activation patterns:

```python
def compare_cluster_activations(
    tactical_activations: Dict[str, torch.Tensor],
    positional_activations: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compare how differently two clusters process the same position.
    """
    results = {}

    for layer_name in tactical_activations.keys():
        act_t = tactical_activations[layer_name].flatten()
        act_p = positional_activations[layer_name].flatten()

        # Cosine similarity of activation patterns
        cos_sim = F.cosine_similarity(act_t, act_p, dim=0).item()

        # L2 distance (normalized)
        l2_dist = torch.norm(act_t - act_p).item()
        l2_norm = torch.sqrt(torch.norm(act_t)**2 + torch.norm(act_p)**2).item()
        normalized_l2 = l2_dist / l2_norm if l2_norm > 0 else 0

        results[layer_name] = {
            "activation_similarity": cos_sim,
            "activation_distance": normalized_l2
        }

    return results
```

#### 1.5 Position-Type Analysis

**Goal**: Understand if models respond differently to tactical vs quiet positions.

**Position Categories**:
- **Tactical positions**: Multiple captures available, checks, hanging pieces
- **Quiet positions**: No immediate tactics, maneuvering required
- **Endgame positions**: Few pieces, pawn structure important

**Expected Results**:
- Tactical cluster: Higher activations in tactical positions
- Positional cluster: More uniform activations across position types

#### 1.6 Output Schema

```json
{
  "round_num": 50,
  "cluster_id": "cluster_tactical",
  "positions_analyzed": 100,

  "per_layer_stats": {
    "residual.10": {
      "mean_activation": 0.234,
      "std_activation": 0.567,
      "sparsity_percent": 23.4,
      "dead_neurons_percent": 2.1,
      "highly_active_neurons": ["channel_45", "channel_128", "channel_201"]
    }
  },

  "position_type_comparison": {
    "tactical_positions": {
      "mean_activation": 0.312,
      "activation_variance": 0.089
    },
    "quiet_positions": {
      "mean_activation": 0.198,
      "activation_variance": 0.045
    }
  },

  "cluster_comparison": {
    "residual.10": {
      "activation_similarity_tactical_vs_positional": 0.823,
      "activation_distance": 0.234
    }
  }
}
```

#### 1.7 Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| Memory usage | Process positions in small batches, clear hooks between batches |
| Computation time | Sample subset of layers (every 3rd residual block) |
| Storage size | Store statistics only, not raw activations |
| Hook management | Use context manager to ensure cleanup |

---

### 2. Saliency Maps (Complexity: High)

**Goal**: Visualize which squares and pieces on the chess board most influence the model's decisions, revealing what the model "looks at" when choosing moves.

#### 2.1 Theoretical Background

Saliency maps use **gradient-based attribution** to determine input importance. The core idea:

> If changing a specific input feature significantly changes the output, that feature is "salient" (important) for the decision.

For chess, this translates to:
- Which squares influence the chosen move?
- Does the tactical model focus on attacking squares?
- Does the positional model focus on pawn structure?

#### 2.2 Mathematical Foundation

**Vanilla Gradient Saliency**:

For input `x` (board representation) and output `y` (policy logits for a specific move):

```
Saliency(x) = |∂y/∂x|
```

The gradient tells us: "How much would the output change if we slightly changed this input?"

**For Chess Board Input**:
- Input shape: `(119, 8, 8)` - 119 planes, 8x8 board
- Each plane encodes different information (piece positions, history, castling rights)
- We aggregate across planes to get an 8x8 saliency map

**Aggregation Formula**:
```
Saliency_square(r, c) = Σᵢ |∂y/∂x[i, r, c]|    # Sum over all 119 planes
```

Or using L2 norm:
```
Saliency_square(r, c) = √(Σᵢ (∂y/∂x[i, r, c])²)
```

#### 2.3 Implementation

```python
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class ChessSaliencyAnalyzer:
    """
    Compute saliency maps for chess model decisions.

    Reveals which board squares most influence the model's move choice.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_policy_saliency(
        self,
        board_tensor: torch.Tensor,
        target_move_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute saliency map for policy head output.

        Args:
            board_tensor: Input board state, shape (1, 119, 8, 8)
            target_move_idx: If None, use the model's top predicted move
                            Otherwise, compute saliency for specific move

        Returns:
            Saliency map of shape (8, 8) showing square importance
        """
        board_tensor = board_tensor.to(self.device)
        board_tensor.requires_grad_(True)

        # Forward pass
        policy_logits, value = self.model(board_tensor)

        # Select target output
        if target_move_idx is None:
            target_move_idx = policy_logits.argmax(dim=1).item()

        target_output = policy_logits[0, target_move_idx]

        # Backward pass to compute gradients
        self.model.zero_grad()
        target_output.backward()

        # Get gradient w.r.t. input
        gradient = board_tensor.grad.detach().cpu()  # Shape: (1, 119, 8, 8)

        # Aggregate across channels using L2 norm
        saliency = torch.norm(gradient[0], dim=0)  # Shape: (8, 8)

        # Normalize to [0, 1]
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency.numpy()

    def compute_value_saliency(
        self,
        board_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Compute saliency map for value head output.

        Shows which squares influence position evaluation.
        """
        board_tensor = board_tensor.to(self.device)
        board_tensor.requires_grad_(True)

        # Forward pass
        policy_logits, value = self.model(board_tensor)

        # Backward from value output
        self.model.zero_grad()
        value[0].backward()

        # Get gradient
        gradient = board_tensor.grad.detach().cpu()

        # Aggregate
        saliency = torch.norm(gradient[0], dim=0)
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency.numpy()

    def compute_comparison_saliency(
        self,
        board_tensor: torch.Tensor,
        move_idx_a: int,
        move_idx_b: int
    ) -> np.ndarray:
        """
        Compute differential saliency between two moves.

        Shows which squares differentiate why move A is preferred over move B.
        Useful for understanding tactical vs positional move preferences.
        """
        saliency_a = self.compute_policy_saliency(board_tensor.clone(), move_idx_a)
        saliency_b = self.compute_policy_saliency(board_tensor.clone(), move_idx_b)

        # Differential saliency
        diff_saliency = saliency_a - saliency_b

        return diff_saliency
```

#### 2.4 Advanced Techniques

**Integrated Gradients** (more accurate but slower):

Instead of computing gradient at a single point, integrate gradients along a path from baseline to input:

```
IntegratedGradients(x) = (x - baseline) × ∫₀¹ (∂F(baseline + α(x - baseline))/∂x) dα
```

Where `baseline` is typically a zero board or averaged board.

```python
def integrated_gradients(
    self,
    board_tensor: torch.Tensor,
    target_move_idx: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution.

    More accurate than vanilla gradients, satisfies axioms of attribution.
    """
    if baseline is None:
        baseline = torch.zeros_like(board_tensor)

    # Generate interpolation path
    scaled_inputs = [
        baseline + (float(i) / steps) * (board_tensor - baseline)
        for i in range(steps + 1)
    ]

    # Compute gradients at each step
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.to(self.device)
        scaled_input.requires_grad_(True)

        policy_logits, _ = self.model(scaled_input)
        target_output = policy_logits[0, target_move_idx]

        self.model.zero_grad()
        target_output.backward()

        gradients.append(scaled_input.grad.detach().cpu())

    # Average gradients and multiply by (input - baseline)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grad = (board_tensor - baseline) * avg_gradients

    # Aggregate to 8x8
    saliency = torch.norm(integrated_grad[0], dim=0)
    saliency = saliency / (saliency.max() + 1e-8)

    return saliency.numpy()
```

#### 2.5 Visualization

```python
import matplotlib.pyplot as plt
import chess

def visualize_saliency_on_board(
    saliency: np.ndarray,
    board: chess.Board,
    title: str = "Saliency Map",
    save_path: Optional[str] = None
):
    """
    Overlay saliency map on chess board visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Raw saliency heatmap
    ax1 = axes[0]
    im = ax1.imshow(saliency, cmap='hot', interpolation='nearest')
    ax1.set_title(f"{title} - Heatmap")
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax1.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    plt.colorbar(im, ax=ax1)

    # Right: Saliency overlaid on board
    ax2 = axes[1]
    # Draw checkerboard pattern
    for r in range(8):
        for c in range(8):
            color = '#F0D9B5' if (r + c) % 2 == 0 else '#B58863'
            ax2.add_patch(plt.Rectangle((c, 7-r), 1, 1, color=color))

    # Overlay saliency with transparency
    saliency_display = np.flipud(saliency)  # Flip for display
    ax2.imshow(saliency_display, cmap='Reds', alpha=0.6,
               extent=[0, 8, 0, 8], interpolation='nearest')

    # Add piece symbols
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = chess.square_rank(square)
            symbol = piece_symbols.get(piece.symbol(), piece.symbol())
            ax2.text(col + 0.5, row + 0.5, symbol,
                    ha='center', va='center', fontsize=24)

    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')
    ax2.set_title(f"{title} - Board Overlay")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

#### 2.6 Cluster Comparison Analysis

**Goal**: Compare what tactical vs positional models focus on for the same position.

```python
def compare_cluster_saliency(
    tactical_model: nn.Module,
    positional_model: nn.Module,
    board_tensor: torch.Tensor,
    board: chess.Board
) -> Dict[str, Any]:
    """
    Compare saliency maps between tactical and positional clusters.
    """
    analyzer_tactical = ChessSaliencyAnalyzer(tactical_model)
    analyzer_positional = ChessSaliencyAnalyzer(positional_model)

    # Compute saliency for each model's top move
    tactical_saliency = analyzer_tactical.compute_policy_saliency(board_tensor)
    positional_saliency = analyzer_positional.compute_policy_saliency(board_tensor)

    # Compute difference
    saliency_diff = tactical_saliency - positional_saliency

    # Metrics
    cosine_sim = np.dot(tactical_saliency.flatten(), positional_saliency.flatten())
    cosine_sim /= (np.linalg.norm(tactical_saliency) * np.linalg.norm(positional_saliency) + 1e-8)

    # Which squares does tactical model focus on MORE than positional?
    tactical_focus_squares = np.argwhere(saliency_diff > 0.2)
    positional_focus_squares = np.argwhere(saliency_diff < -0.2)

    return {
        "tactical_saliency": tactical_saliency,
        "positional_saliency": positional_saliency,
        "saliency_difference": saliency_diff,
        "saliency_similarity": cosine_sim,
        "tactical_focus_squares": tactical_focus_squares.tolist(),
        "positional_focus_squares": positional_focus_squares.tolist()
    }
```

#### 2.7 Expected Findings

| Position Type | Tactical Model Focus | Positional Model Focus |
|---------------|---------------------|------------------------|
| **Open position** | Attack squares, king area, loose pieces | Center control, piece coordination |
| **Closed position** | Breakthrough squares, pawn levers | Pawn structure, outposts |
| **Endgame** | Passed pawns, king activity | Pawn promotion squares, king position |

**Specific Hypotheses**:

1. **Tactical cluster**: Higher saliency on squares with hanging pieces, fork targets, pin lines
2. **Positional cluster**: Higher saliency on center squares, pawn chains, piece mobility squares
3. **Value head saliency**: Both models should focus on king safety squares

#### 2.8 Output Schema

```json
{
  "round_num": 50,
  "position_fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "analysis_type": "policy_saliency",

  "tactical_model": {
    "top_move": "Ng5",
    "top_move_idx": 1234,
    "saliency_map": [[0.12, 0.34, ...], ...],  // 8x8 array
    "high_saliency_squares": ["f7", "e5", "g5"],
    "mean_saliency": 0.234,
    "saliency_concentration": 0.67  // How focused vs spread out
  },

  "positional_model": {
    "top_move": "d3",
    "top_move_idx": 567,
    "saliency_map": [[0.08, 0.21, ...], ...],
    "high_saliency_squares": ["d4", "e4", "c4"],
    "mean_saliency": 0.198,
    "saliency_concentration": 0.45
  },

  "comparison": {
    "saliency_similarity": 0.623,
    "tactical_unique_focus": ["f7", "g5"],
    "positional_unique_focus": ["d4", "c4"],
    "move_agreement": false
  }
}
```

#### 2.9 Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| Gradient computation requires grad mode | Use `torch.enable_grad()` context |
| Memory for large batch analysis | Process one position at a time |
| Numerical stability | Add small epsilon to normalization |
| Interpretation complexity | Combine with human chess analysis |
| Input plane aggregation | Test both sum and L2 norm aggregation |

---

### 3. Concept Activation Vectors (CAVs) (Complexity: Very High)

**Goal**: Test if model learns chess concepts (pins, forks, etc.).

**Method**:
- Collect positions with/without specific concepts
- Train linear classifier on activations
- Measure concept presence in model representations

**Challenge**: Requires labeled dataset of chess concepts.

---

### 4. Attention Analysis (Complexity: Medium)

**Goal**: If attention layers added, visualize attention patterns.

**Note**: Current AlphaZero architecture uses convolutions, not attention.
Would require architecture modification.

---

## References

1. **AlphaZero Paper**: Silver, D., et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
2. **Federated Learning Analysis**: McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
3. **Neural Network Interpretability**: Olah, C., et al. "Feature Visualization" (2017)
4. **Model Similarity Metrics**: Kornblith, S., et al. "Similarity of Neural Network Representations Revisited" (2019)

---

*Document Version: 1.0*
*Last Updated: 2025-11-22*
*Author: Chess Federated Learning Team*
