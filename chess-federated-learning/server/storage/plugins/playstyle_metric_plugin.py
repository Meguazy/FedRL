"""
Playstyle metrics plugin for metric registry.

This plugin integrates playstyle evaluation results into the storage system,
allowing playstyle metrics to be logged alongside other training metrics.
"""

from typing import Any, Dict, List
from loguru import logger

from server.storage.plugins.metric_registry import BaseMetricComputer


class PlaystyleMetricsPlugin(BaseMetricComputer):
    """
    Metric computer for playstyle evaluation results.

    This plugin processes playstyle evaluation data and formats it
    for storage in the metrics system.
    """

    def get_name(self) -> str:
        """Get plugin name."""
        return "PlaystyleMetricsComputer"

    def get_required_context_keys(self) -> List[str]:
        """Get required context keys."""
        return ["evaluation_results"]

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metrics from playstyle evaluation results.

        Args:
            context: Dict containing:
                - evaluation_results: Results from ModelEvaluator.evaluate_models()
                - round_num: Current training round (optional)

        Returns:
            Dict of metrics ready for storage
        """
        log = logger.bind(context="PlaystyleMetricsPlugin.compute")

        evaluation_results = context["evaluation_results"]
        round_num = context.get("round_num", 0)

        # Extract cluster metrics
        cluster_metrics = evaluation_results.get("cluster_metrics", {})
        summary = evaluation_results.get("summary", {})

        metrics = {}

        # Add cluster-specific metrics
        for cluster_id, cm_dict in cluster_metrics.items():
            # Prefix all cluster metrics with cluster ID
            cluster_prefix = f"{cluster_id}_"

            metrics[f"{cluster_prefix}tactical_score"] = cm_dict["tactical_score"]
            metrics[f"{cluster_prefix}classification"] = cm_dict["classification"]
            metrics[f"{cluster_prefix}estimated_elo"] = cm_dict["estimated_elo"]
            metrics[f"{cluster_prefix}elo_confidence"] = cm_dict["elo_confidence"]
            metrics[f"{cluster_prefix}win_rate"] = cm_dict["win_rate"]
            metrics[f"{cluster_prefix}games_analyzed"] = cm_dict["games_analyzed"]

            # Raw metrics
            metrics[f"{cluster_prefix}avg_attacked_material"] = cm_dict["avg_attacked_material"]
            metrics[f"{cluster_prefix}avg_legal_moves"] = cm_dict["avg_legal_moves"]
            metrics[f"{cluster_prefix}avg_captures"] = cm_dict["avg_captures"]
            metrics[f"{cluster_prefix}avg_center_control"] = cm_dict["avg_center_control"]

            # Normalized metrics
            metrics[f"{cluster_prefix}avg_attacks_metric"] = cm_dict["avg_attacks_metric"]
            metrics[f"{cluster_prefix}avg_moves_metric"] = cm_dict["avg_moves_metric"]
            metrics[f"{cluster_prefix}avg_material_metric"] = cm_dict["avg_material_metric"]

            # Tactical score statistics
            metrics[f"{cluster_prefix}tactical_score_std"] = cm_dict["tactical_score_std"]
            metrics[f"{cluster_prefix}tactical_score_min"] = cm_dict["tactical_score_min"]
            metrics[f"{cluster_prefix}tactical_score_max"] = cm_dict["tactical_score_max"]

            # Classification distribution
            for classification, count in cm_dict["classification_distribution"].items():
                safe_classification = classification.replace(" ", "_").lower()
                metrics[f"{cluster_prefix}dist_{safe_classification}"] = count

        # Add summary metrics
        metrics["playstyle_divergence"] = summary.get("playstyle_divergence", 0.0)
        metrics["elo_spread"] = summary.get("elo_spread", 0)
        metrics["total_clusters_evaluated"] = summary.get("total_clusters_evaluated", 0)

        log.debug(f"Computed {len(metrics)} playstyle metrics for round {round_num}")

        return metrics
