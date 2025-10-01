"""
System performance metric computation plugin.

Computes system-level metrics:
- CPU usage
- Memory usage
- Throughput metrics
- Timing information
"""

import os
import time
from typing import Any, Dict, List

from loguru import logger

from server.storage.plugins.metric_registry import BaseMetricComputer


class SystemMetricComputer(BaseMetricComputer):
    """
    Computes system performance metrics.

    Tracks resource usage and performance characteristics.
    """

    def __init__(self):
        """Initialize with timing tracking."""
        self._last_compute_time = time.time()
        self._compute_count = 0

    def get_name(self) -> str:
        """Get display name."""
        return "System Performance Metrics"

    def get_required_context_keys(self) -> List[str]:
        """No required keys - uses system data."""
        return []

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute system metrics.

        Args:
            context: Optional context data (not required)

        Returns:
            Dictionary with:
            - cpu_usage_percent: CPU usage percentage
            - memory_usage_mb: Memory usage in MB
            - memory_usage_percent: Memory usage percentage
            - compute_interval_seconds: Time since last compute
        """
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed, system metrics unavailable")
            return {}

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent

        # Timing
        current_time = time.time()
        interval = current_time - self._last_compute_time
        self._last_compute_time = current_time
        self._compute_count += 1

        return {
            "cpu_usage_percent": float(cpu_percent),
            "memory_usage_mb": float(memory_mb),
            "memory_usage_percent": float(memory_percent),
            "compute_interval_seconds": float(interval),
            "compute_count": self._compute_count
        }


class AggregationTimingMetric(BaseMetricComputer):
    """
    Computes aggregation timing metrics.

    Tracks time spent in aggregation operations.
    """

    def get_name(self) -> str:
        """Get display name."""
        return "Aggregation Timing Metrics"

    def get_required_context_keys(self) -> List[str]:
        """Required context keys."""
        return []

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute aggregation timing metrics.

        Args:
            context: May contain:
                - aggregation_start_time: Start timestamp
                - aggregation_end_time: End timestamp
                - node_count: Number of nodes aggregated

        Returns:
            Dictionary with:
            - aggregation_duration_seconds: Time taken
            - nodes_per_second: Throughput metric
        """
        start_time = context.get("aggregation_start_time")
        end_time = context.get("aggregation_end_time")
        node_count = context.get("node_count", 0)

        if start_time is None or end_time is None:
            return {}

        duration = end_time - start_time
        throughput = node_count / duration if duration > 0 else 0

        return {
            "aggregation_duration_seconds": float(duration),
            "nodes_per_second": float(throughput),
            "node_count": node_count
        }
