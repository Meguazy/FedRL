"""
Diversity metric computation plugin.

Computes metrics related to model diversity and inter-cluster distance:
- Inter-cluster distance: L2 distance between cluster models
- Layer divergence: Per-layer distance metrics
- Diversity score: Normalized diversity measure
"""

from typing import Any, Dict, List

import torch
from loguru import logger

from .metric_registry import BaseMetricComputer


class DiversityMetricComputer(BaseMetricComputer):
    """
    Computes diversity metrics for federated learning.

    Measures how different cluster models are from each other,
    which is critical for preserving playstyle diversity.
    """

    def get_name(self) -> str:
        """Get display name."""
        return "Diversity Metrics Computer"

    def get_required_context_keys(self) -> List[str]:
        """Required context keys."""
        return ["models"]

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute diversity metrics.

        Args:
            context: Must contain:
                - models: Dict[str, Dict[str, Any]] - cluster models

        Returns:
            Dictionary with:
            - inter_cluster_distance: Average L2 distance between clusters
            - diversity_score: Normalized diversity (0-1)
            - layer_divergence: Dict of per-layer distances
            - max_divergence_layer: Layer with maximum divergence
        """
        models = context.get("models", {})

        if len(models) < 2:
            logger.warning("Need at least 2 models to compute diversity")
            return {
                "inter_cluster_distance": 0.0,
                "diversity_score": 0.0,
                "layer_divergence": {},
                "max_divergence_layer": None
            }

        # Compute pairwise distances
        model_ids = list(models.keys())
        distances = []
        layer_distances = {}

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                model_a = models[model_ids[i]]
                model_b = models[model_ids[j]]

                # Compute overall distance
                distance = self._compute_model_distance(model_a, model_b)
                distances.append(distance)

                # Compute per-layer distances
                layer_dist = self._compute_layer_distances(model_a, model_b)
                for layer, dist in layer_dist.items():
                    if layer not in layer_distances:
                        layer_distances[layer] = []
                    layer_distances[layer].append(dist)

        # Average distance
        avg_distance = sum(distances) / len(distances) if distances else 0.0

        # Average layer distances
        avg_layer_distances = {
            layer: sum(dists) / len(dists)
            for layer, dists in layer_distances.items()
        }

        # Find layer with max divergence
        max_divergence_layer = None
        if avg_layer_distances:
            max_divergence_layer = max(
                avg_layer_distances.items(),
                key=lambda x: x[1]
            )[0]

        # Diversity score (normalized, 0-1)
        # Higher distance = higher diversity
        # Normalize by expected range (heuristic)
        diversity_score = min(avg_distance / 10.0, 1.0)

        return {
            "inter_cluster_distance": float(avg_distance),
            "diversity_score": float(diversity_score),
            "layer_divergence": {
                k: float(v) for k, v in avg_layer_distances.items()
            },
            "max_divergence_layer": max_divergence_layer
        }

    def _compute_model_distance(
        self,
        model_a: Dict[str, Any],
        model_b: Dict[str, Any]
    ) -> float:
        """
        Compute L2 distance between two models.

        Args:
            model_a: First model state dict
            model_b: Second model state dict

        Returns:
            L2 distance between models
        """
        total_distance = 0.0
        param_count = 0

        # Get common keys
        common_keys = set(model_a.keys()) & set(model_b.keys())

        for key in common_keys:
            param_a = model_a[key]
            param_b = model_b[key]

            # Convert to tensors if needed
            if not isinstance(param_a, torch.Tensor):
                param_a = torch.tensor(param_a)
            if not isinstance(param_b, torch.Tensor):
                param_b = torch.tensor(param_b)

            # Compute L2 distance
            distance = torch.norm(param_a - param_b, p=2).item()
            total_distance += distance
            param_count += 1

        # Average distance per parameter group
        return total_distance / param_count if param_count > 0 else 0.0

    def _compute_layer_distances(
        self,
        model_a: Dict[str, Any],
        model_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute per-layer distances.

        Args:
            model_a: First model state dict
            model_b: Second model state dict

        Returns:
            Dictionary of layer_name -> distance
        """
        layer_distances = {}

        # Get common keys
        common_keys = set(model_a.keys()) & set(model_b.keys())

        for key in common_keys:
            param_a = model_a[key]
            param_b = model_b[key]

            # Convert to tensors if needed
            if not isinstance(param_a, torch.Tensor):
                param_a = torch.tensor(param_a)
            if not isinstance(param_b, torch.Tensor):
                param_b = torch.tensor(param_b)

            # Compute L2 distance for this layer
            distance = torch.norm(param_a - param_b, p=2).item()
            layer_distances[key] = distance

        return layer_distances


class SharedLayerSyncMetric(BaseMetricComputer):
    """
    Computes metrics about shared layer synchronization.

    Measures how well shared layers are synchronized across clusters.
    """

    def get_name(self) -> str:
        """Get display name."""
        return "Shared Layer Sync Metrics"

    def get_required_context_keys(self) -> List[str]:
        """Required context keys."""
        return ["models", "shared_layer_patterns"]

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute shared layer synchronization metrics.

        Args:
            context: Must contain:
                - models: Dict of cluster models
                - shared_layer_patterns: List of shared layer patterns

        Returns:
            Dictionary with:
            - shared_layers_synchronized: bool
            - max_shared_layer_distance: Maximum distance in shared layers
            - avg_shared_layer_distance: Average distance in shared layers
        """
        import re

        models = context.get("models", {})
        shared_patterns = context.get("shared_layer_patterns", [])

        if len(models) < 2:
            return {
                "shared_layers_synchronized": True,
                "max_shared_layer_distance": 0.0,
                "avg_shared_layer_distance": 0.0
            }

        # Identify shared layers
        model_ids = list(models.keys())
        first_model = models[model_ids[0]]
        shared_layers = []

        for key in first_model.keys():
            for pattern in shared_patterns:
                if re.match(pattern, key):
                    shared_layers.append(key)
                    break

        if not shared_layers:
            return {
                "shared_layers_synchronized": True,
                "max_shared_layer_distance": 0.0,
                "avg_shared_layer_distance": 0.0
            }

        # Compute distances for shared layers
        distances = []

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                model_a = models[model_ids[i]]
                model_b = models[model_ids[j]]

                for layer in shared_layers:
                    if layer in model_a and layer in model_b:
                        param_a = model_a[layer]
                        param_b = model_b[layer]

                        # Convert to tensors if needed
                        if not isinstance(param_a, torch.Tensor):
                            param_a = torch.tensor(param_a)
                        if not isinstance(param_b, torch.Tensor):
                            param_b = torch.tensor(param_b)

                        # Compute distance
                        distance = torch.norm(param_a - param_b, p=2).item()
                        distances.append(distance)

        max_distance = max(distances) if distances else 0.0
        avg_distance = sum(distances) / len(distances) if distances else 0.0

        # Synchronized if max distance is very small (tolerance)
        synchronized = max_distance < 1e-6

        return {
            "shared_layers_synchronized": synchronized,
            "max_shared_layer_distance": float(max_distance),
            "avg_shared_layer_distance": float(avg_distance)
        }
