"""
Weight Statistics Metrics for Model Analysis.

This module computes per-layer weight statistics to understand
how the model's parameters evolve during training.

Metrics computed:
- Mean, std, min, max per layer
- L2 norm per layer
- Sparsity (% near-zero weights)
- Weight change from previous checkpoint
- Dead layer detection

These metrics help identify:
- Which layers are actively learning
- Potential gradient issues (dead layers)
- Weight distribution health
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger


# Layer groupings (same as model_divergence for consistency)
LAYER_GROUPS = {
    "input_block": ["input_conv", "input_bn"],
    "early_residual": [f"residual.{i}" for i in range(6)],
    "middle_residual": [f"residual.{i}" for i in range(6, 13)],
    "late_residual": [f"residual.{i}" for i in range(13, 19)],
    "policy_head": ["policy_head"],
    "value_head": ["value_head"],
}


def compute_layer_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for a single weight tensor.

    Args:
        tensor: Weight tensor to analyze

    Returns:
        Dictionary with mean, std, min, max, l2_norm, sparsity
    """
    tensor_float = tensor.float()

    # Basic statistics
    mean = tensor_float.mean().item()
    std = tensor_float.std().item()
    min_val = tensor_float.min().item()
    max_val = tensor_float.max().item()
    l2_norm = torch.norm(tensor_float).item()

    # Sparsity: percentage of near-zero weights
    near_zero_threshold = 1e-6
    sparsity = (tensor_float.abs() < near_zero_threshold).float().mean().item()

    return {
        "mean": round(mean, 8),
        "std": round(std, 8),
        "min": round(min_val, 8),
        "max": round(max_val, 8),
        "l2_norm": round(l2_norm, 6),
        "sparsity": round(sparsity, 6),
    }


def compute_weight_change(
    current: torch.Tensor,
    previous: torch.Tensor
) -> Dict[str, float]:
    """
    Compute how much weights changed from previous checkpoint.

    Args:
        current: Current weight tensor
        previous: Previous weight tensor

    Returns:
        Dictionary with absolute and relative change metrics
    """
    current_float = current.float()
    previous_float = previous.float()

    # Absolute L2 change
    diff = current_float - previous_float
    absolute_change = torch.norm(diff).item()

    # Relative change (normalized by previous norm)
    prev_norm = torch.norm(previous_float).item()
    if prev_norm > 0:
        relative_change = absolute_change / prev_norm
    else:
        relative_change = float('inf') if absolute_change > 0 else 0.0

    # Max element-wise change
    max_change = diff.abs().max().item()

    # Mean element-wise change
    mean_change = diff.abs().mean().item()

    return {
        "absolute_l2_change": round(absolute_change, 8),
        "relative_change": round(relative_change, 8),
        "max_element_change": round(max_change, 8),
        "mean_element_change": round(mean_change, 8),
    }


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


def compute_weight_statistics(
    model_state: Dict[str, torch.Tensor],
    previous_state: Optional[Dict[str, torch.Tensor]],
    round_num: int,
    cluster_id: str,
    dead_threshold: float = 0.001
) -> Dict[str, Any]:
    """
    Compute comprehensive weight statistics for a model.

    Args:
        model_state: Current model state dict
        previous_state: Previous model state dict (for computing changes)
        round_num: Current training round
        cluster_id: Cluster identifier
        dead_threshold: Relative change threshold below which layer is "dead"

    Returns:
        Dictionary containing all weight statistics

    Example:
        >>> current = torch.load("checkpoint_round_10.pt")["model_state_dict"]
        >>> previous = torch.load("checkpoint_round_0.pt")["model_state_dict"]
        >>> stats = compute_weight_statistics(current, previous, 10, "cluster_tactical")
    """
    results = {
        "round_num": round_num,
        "timestamp": datetime.now().isoformat(),
        "cluster_id": cluster_id,
        "per_layer": {},
        "per_group": {},
        "summary": {}
    }

    # Track for summary
    total_params = 0
    total_near_zero = 0
    all_relative_changes = []
    dead_layers = []
    highly_active_layers = []
    group_stats: Dict[str, List[Dict[str, float]]] = {
        group: [] for group in LAYER_GROUPS.keys()
    }

    # Compute per-layer statistics
    for layer_name, tensor in model_state.items():
        # Basic statistics
        layer_stats = compute_layer_statistics(tensor)
        layer_stats["num_parameters"] = tensor.numel()
        layer_stats["tensor_shape"] = list(tensor.shape)

        # Weight change from previous
        if previous_state and layer_name in previous_state:
            change_stats = compute_weight_change(tensor, previous_state[layer_name])
            layer_stats.update(change_stats)

            # Track relative change for dead layer detection
            rel_change = change_stats["relative_change"]
            all_relative_changes.append(rel_change)

            if rel_change < dead_threshold:
                dead_layers.append(layer_name)

        results["per_layer"][layer_name] = layer_stats

        # Track totals
        total_params += tensor.numel()
        total_near_zero += (tensor.float().abs() < 1e-6).sum().item()

        # Add to layer group
        group = get_layer_group(layer_name)
        if group:
            group_stats[group].append({
                "mean": layer_stats["mean"],
                "std": layer_stats["std"],
                "l2_norm": layer_stats["l2_norm"],
                "sparsity": layer_stats["sparsity"],
                "relative_change": layer_stats.get("relative_change", 0),
            })

    # Identify highly active layers (top 10% by relative change)
    if all_relative_changes:
        change_threshold = np.percentile(all_relative_changes, 90)
        for layer_name, layer_stats in results["per_layer"].items():
            if layer_stats.get("relative_change", 0) >= change_threshold:
                highly_active_layers.append(layer_name)

    # Compute per-group aggregations
    for group_name, stats_list in group_stats.items():
        if not stats_list:
            continue

        results["per_group"][group_name] = {
            "mean_l2_norm": round(float(np.mean([s["l2_norm"] for s in stats_list])), 6),
            "mean_sparsity": round(float(np.mean([s["sparsity"] for s in stats_list])), 6),
            "mean_std": round(float(np.mean([s["std"] for s in stats_list])), 8),
            "num_layers": len(stats_list),
        }

        # Add change metrics if available
        changes = [s["relative_change"] for s in stats_list if s.get("relative_change", 0) > 0]
        if changes:
            results["per_group"][group_name]["mean_relative_change"] = round(float(np.mean(changes)), 8)

    # Compute summary
    results["summary"] = {
        "total_parameters": total_params,
        "global_sparsity": round(total_near_zero / total_params, 6) if total_params > 0 else 0,
        "layers_analyzed": len(model_state),
        "dead_layers_count": len(dead_layers),
        "dead_layers": dead_layers[:10],  # Limit to first 10
        "highly_active_layers_count": len(highly_active_layers),
        "highly_active_layers": highly_active_layers[:10],  # Limit to first 10
    }

    # Add change summary if we have previous state
    if all_relative_changes:
        results["summary"]["mean_relative_change"] = round(float(np.mean(all_relative_changes)), 8)
        results["summary"]["max_relative_change"] = round(float(np.max(all_relative_changes)), 8)
        results["summary"]["min_relative_change"] = round(float(np.min(all_relative_changes)), 8)

    logger.info(
        f"Computed weight statistics for {cluster_id} round {round_num}: "
        f"total_params={total_params:,}, "
        f"sparsity={results['summary']['global_sparsity']:.4f}, "
        f"dead_layers={len(dead_layers)}"
    )

    return results


def detect_anomalies(
    weight_stats: Dict[str, Any],
    std_threshold: float = 10.0,
    sparsity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Detect potential anomalies in weight statistics.

    Args:
        weight_stats: Output from compute_weight_statistics
        std_threshold: Flag layers with std above this value
        sparsity_threshold: Flag layers with sparsity above this value

    Returns:
        List of anomaly reports
    """
    anomalies = []

    for layer_name, stats in weight_stats.get("per_layer", {}).items():
        # High standard deviation (exploding weights)
        if stats.get("std", 0) > std_threshold:
            anomalies.append({
                "layer": layer_name,
                "type": "high_std",
                "value": stats["std"],
                "threshold": std_threshold,
                "severity": "warning"
            })

        # High sparsity (many dead neurons)
        if stats.get("sparsity", 0) > sparsity_threshold:
            anomalies.append({
                "layer": layer_name,
                "type": "high_sparsity",
                "value": stats["sparsity"],
                "threshold": sparsity_threshold,
                "severity": "info"
            })

        # Near-zero mean with high std (unstable)
        if abs(stats.get("mean", 0)) < 1e-6 and stats.get("std", 0) > 1.0:
            anomalies.append({
                "layer": layer_name,
                "type": "unstable_weights",
                "mean": stats["mean"],
                "std": stats["std"],
                "severity": "warning"
            })

    return anomalies


def get_learning_summary(
    current_stats: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a human-readable learning summary.

    Args:
        current_stats: Current weight statistics
        history: List of previous statistics (optional)

    Returns:
        Summary dictionary with learning insights
    """
    summary = current_stats.get("summary", {})

    insights = {
        "cluster_id": current_stats.get("cluster_id"),
        "round_num": current_stats.get("round_num"),
        "health_status": "healthy",
        "issues": [],
        "observations": []
    }

    # Check for dead layers
    dead_count = summary.get("dead_layers_count", 0)
    if dead_count > 0:
        insights["issues"].append(f"{dead_count} layers showing minimal learning")
        if dead_count > 10:
            insights["health_status"] = "warning"

    # Check global sparsity
    sparsity = summary.get("global_sparsity", 0)
    if sparsity > 0.1:
        insights["observations"].append(f"High sparsity ({sparsity:.1%}) - consider pruning")

    # Check active layers
    active_count = summary.get("highly_active_layers_count", 0)
    if active_count > 0:
        insights["observations"].append(f"{active_count} layers actively learning")

    # Mean change analysis
    mean_change = summary.get("mean_relative_change", 0)
    if mean_change > 0:
        if mean_change < 0.001:
            insights["observations"].append("Learning rate may be too low")
        elif mean_change > 0.5:
            insights["observations"].append("High weight changes - potential instability")
        else:
            insights["observations"].append(f"Normal learning progression ({mean_change:.4f} avg change)")

    return insights
