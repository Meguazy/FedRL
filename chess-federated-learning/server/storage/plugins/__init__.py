"""
Plugin system for extensible metric computation.

This package provides a plugin architecture for computing custom metrics:
- MetricComputer: Protocol for metric computation plugins
- MetricRegistry: Dynamic registration and management of plugins
- Built-in plugins: diversity, system metrics

Example:
    from server.storage.plugins import MetricRegistry, DiversityMetricComputer

    registry = MetricRegistry()
    registry.register("diversity", DiversityMetricComputer())

    context = {"models": cluster_models, "round": 10}
    metrics = registry.compute_all(context)
"""

from server.storage.plugins.metric_registry import (
    BaseMetricComputer,
    MetricComputer,
    MetricRegistry,
)
from server.storage.plugins.diversity_metrics import (
    DiversityMetricComputer,
    SharedLayerSyncMetric,
)
from server.storage.plugins.system_metrics import (
    AggregationTimingMetric,
    SystemMetricComputer,
)

__all__ = [
    "BaseMetricComputer",
    "MetricComputer",
    "MetricRegistry",
    "DiversityMetricComputer",
    "SharedLayerSyncMetric",
    "SystemMetricComputer",
    "AggregationTimingMetric",
]

__version__ = "1.0.0"
