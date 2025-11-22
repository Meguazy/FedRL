"""
Model Divergence Metrics for Cluster Comparison.

This module computes divergence metrics between cluster models to understand
how tactical and positional clusters differentiate during federated learning.

Metrics computed:
- Per-layer cosine similarity
- Per-layer L2 distance (normalized)
- Divergence index (composite metric)
- Per-layer-group aggregations
- Global divergence statistics

These metrics validate the core thesis hypothesis: selective aggregation
preserves playstyle by allowing cluster-specific layers to diverge.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


# Layer groupings for AlphaZero network analysis
LAYER_GROUPS = {
    "input_block": ["input_conv", "input_bn"],
    "early_residual": [f"residual.{i}" for i in range(6)],
    "middle_residual": [f"residual.{i}" for i in range(6, 13)],
    "late_residual": [f"residual.{i}" for i in range(13, 19)],
    "policy_head": ["policy_head"],
    "value_head": ["value_head"],
}


def cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two weight tensors.

    Args:
        tensor_a: First weight tensor
        tensor_b: Second weight tensor

    Returns:
        Cosine similarity in range [-1, 1]
        - 1.0 = Identical direction
        - 0.0 = Orthogonal
        - -1.0 = Opposite direction
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    dot_product = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (dot_product / (norm_a * norm_b)).item()


def l2_distance(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    normalize: bool = True
) -> float:
    """
    Compute L2 distance between two weight tensors.

    Args:
        tensor_a: First weight tensor
        tensor_b: Second weight tensor
        normalize: If True, normalize by combined norms for cross-layer comparison

    Returns:
        L2 distance (normalized to ~[0, 1.41] if normalize=True)
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    diff = a_flat - b_flat
    l2 = torch.norm(diff).item()

    if normalize:
        norm_sum = torch.sqrt(torch.norm(a_flat)**2 + torch.norm(b_flat)**2).item()
        if norm_sum > 0:
            l2 = l2 / norm_sum

    return l2


def divergence_index(cos_sim: float, l2_dist: float) -> float:
    """
    Compute composite divergence index.

    Combines cosine similarity and L2 distance into a single metric
    that captures both directional and magnitude differences.

    Formula: (1 - cosine_similarity) * (1 + normalized_l2)

    Args:
        cos_sim: Cosine similarity value
        l2_dist: Normalized L2 distance value

    Returns:
        Divergence index in range [0, ~2.8]
        - 0 = Identical weights
        - Higher = More divergent
    """
    return (1 - cos_sim) * (1 + l2_dist)


def get_layer_group(layer_name: str) -> Optional[str]:
    """
    Determine which layer group a layer belongs to.

    Args:
        layer_name: Full layer name (e.g., "residual.5.conv1.weight")

    Returns:
        Group name or None if not matched
    """
    for group_name, prefixes in LAYER_GROUPS.items():
        for prefix in prefixes:
            if layer_name.startswith(prefix):
                return group_name
    return None


def compute_cluster_divergence(
    model_a_state: Dict[str, torch.Tensor],
    model_b_state: Dict[str, torch.Tensor],
    round_num: int,
    cluster_a_id: str = "cluster_tactical",
    cluster_b_id: str = "cluster_positional"
) -> Dict[str, Any]:
    """
    Compute comprehensive divergence metrics between two cluster models.

    This is the main entry point for divergence analysis. It computes:
    1. Per-layer metrics (cosine similarity, L2 distance, divergence index)
    2. Per-layer-group aggregations
    3. Global summary statistics

    Args:
        model_a_state: State dict from first model (typically tactical)
        model_b_state: State dict from second model (typically positional)
        round_num: Current training round
        cluster_a_id: Identifier for first cluster
        cluster_b_id: Identifier for second cluster

    Returns:
        Dictionary containing all divergence metrics

    Example:
        >>> tactical_state = torch.load("tactical_checkpoint.pt")["model_state_dict"]
        >>> positional_state = torch.load("positional_checkpoint.pt")["model_state_dict"]
        >>> metrics = compute_cluster_divergence(tactical_state, positional_state, round_num=10)
    """
    results = {
        "round_num": round_num,
        "timestamp": datetime.now().isoformat(),
        "clusters_compared": [cluster_a_id, cluster_b_id],
        "per_layer": {},
        "per_group": {},
        "global": {}
    }

    # Track metrics for aggregation
    all_divergences = []
    all_cosine_sims = []
    all_l2_dists = []
    group_metrics: Dict[str, List[Dict[str, float]]] = {
        group: [] for group in LAYER_GROUPS.keys()
    }
    total_params = 0

    # Compute per-layer metrics
    for layer_name, tensor_a in model_a_state.items():
        if layer_name not in model_b_state:
            logger.warning(f"Layer {layer_name} not found in model B, skipping")
            continue

        tensor_b = model_b_state[layer_name]

        # Ensure tensors have same shape
        if tensor_a.shape != tensor_b.shape:
            logger.warning(
                f"Shape mismatch for {layer_name}: {tensor_a.shape} vs {tensor_b.shape}"
            )
            continue

        # Compute metrics
        cos_sim = cosine_similarity(tensor_a, tensor_b)
        l2_dist = l2_distance(tensor_a, tensor_b, normalize=True)
        div_idx = divergence_index(cos_sim, l2_dist)

        layer_metrics = {
            "cosine_similarity": round(cos_sim, 6),
            "l2_distance_normalized": round(l2_dist, 6),
            "divergence_index": round(div_idx, 6),
            "tensor_shape": list(tensor_a.shape),
            "num_parameters": tensor_a.numel()
        }

        results["per_layer"][layer_name] = layer_metrics

        # Track for aggregation
        all_divergences.append(div_idx)
        all_cosine_sims.append(cos_sim)
        all_l2_dists.append(l2_dist)
        total_params += tensor_a.numel()

        # Add to layer group
        group = get_layer_group(layer_name)
        if group:
            group_metrics[group].append({
                "cosine_similarity": cos_sim,
                "l2_distance": l2_dist,
                "divergence_index": div_idx
            })

    # Compute per-group aggregations
    for group_name, metrics_list in group_metrics.items():
        if not metrics_list:
            continue

        cos_sims = [m["cosine_similarity"] for m in metrics_list]
        l2_dists = [m["l2_distance"] for m in metrics_list]
        div_indices = [m["divergence_index"] for m in metrics_list]

        results["per_group"][group_name] = {
            "mean_cosine_similarity": round(float(np.mean(cos_sims)), 6),
            "std_cosine_similarity": round(float(np.std(cos_sims)), 6),
            "mean_l2_distance": round(float(np.mean(l2_dists)), 6),
            "mean_divergence_index": round(float(np.mean(div_indices)), 6),
            "std_divergence_index": round(float(np.std(div_indices)), 6),
            "num_layers": len(metrics_list)
        }

    # Compute global statistics
    if all_divergences:
        results["global"] = {
            "mean_divergence": round(float(np.mean(all_divergences)), 6),
            "std_divergence": round(float(np.std(all_divergences)), 6),
            "max_divergence": round(float(np.max(all_divergences)), 6),
            "min_divergence": round(float(np.min(all_divergences)), 6),
            "mean_cosine_similarity": round(float(np.mean(all_cosine_sims)), 6),
            "mean_l2_distance": round(float(np.mean(all_l2_dists)), 6),
            "total_parameters_compared": total_params,
            "num_layers_compared": len(all_divergences)
        }

    logger.info(
        f"Computed divergence metrics for round {round_num}: "
        f"mean_divergence={results['global'].get('mean_divergence', 'N/A')}, "
        f"mean_cosine_sim={results['global'].get('mean_cosine_similarity', 'N/A')}"
    )

    return results


def compute_divergence_summary(
    divergence_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute summary statistics across multiple rounds of divergence metrics.

    Useful for analyzing divergence trends over training.

    Args:
        divergence_history: List of divergence metric dicts from different rounds

    Returns:
        Summary statistics including trends and final state
    """
    if not divergence_history:
        return {}

    rounds = [d["round_num"] for d in divergence_history]
    mean_divs = [d["global"]["mean_divergence"] for d in divergence_history]

    # Compute trend (simple linear regression slope)
    if len(rounds) > 1:
        x = np.array(rounds)
        y = np.array(mean_divs)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    # Per-group trends
    group_trends = {}
    for group_name in LAYER_GROUPS.keys():
        group_divs = []
        for d in divergence_history:
            if group_name in d.get("per_group", {}):
                group_divs.append(d["per_group"][group_name]["mean_divergence_index"])

        if len(group_divs) > 1:
            group_trends[group_name] = {
                "initial": round(group_divs[0], 6),
                "final": round(group_divs[-1], 6),
                "change": round(group_divs[-1] - group_divs[0], 6),
                "trend": "increasing" if group_divs[-1] > group_divs[0] else "stable_or_decreasing"
            }

    return {
        "rounds_analyzed": len(divergence_history),
        "round_range": [min(rounds), max(rounds)],
        "initial_mean_divergence": round(mean_divs[0], 6),
        "final_mean_divergence": round(mean_divs[-1], 6),
        "divergence_change": round(mean_divs[-1] - mean_divs[0], 6),
        "divergence_slope": round(slope, 8),
        "trend": "increasing" if slope > 0.001 else ("decreasing" if slope < -0.001 else "stable"),
        "per_group_trends": group_trends
    }


def get_most_divergent_layers(
    divergence_metrics: Dict[str, Any],
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the most divergent layers from divergence metrics.

    Args:
        divergence_metrics: Output from compute_cluster_divergence
        top_n: Number of top layers to return

    Returns:
        List of (layer_name, divergence_index) tuples, sorted descending
    """
    layer_divs = [
        (name, metrics["divergence_index"])
        for name, metrics in divergence_metrics.get("per_layer", {}).items()
    ]

    return sorted(layer_divs, key=lambda x: x[1], reverse=True)[:top_n]


def get_most_similar_layers(
    divergence_metrics: Dict[str, Any],
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the most similar layers from divergence metrics.

    Args:
        divergence_metrics: Output from compute_cluster_divergence
        top_n: Number of top layers to return

    Returns:
        List of (layer_name, cosine_similarity) tuples, sorted descending
    """
    layer_sims = [
        (name, metrics["cosine_similarity"])
        for name, metrics in divergence_metrics.get("per_layer", {}).items()
    ]

    return sorted(layer_sims, key=lambda x: x[1], reverse=True)[:top_n]
