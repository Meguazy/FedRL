"""
Server aggregation package.

This package contains aggregation algorithms for federated learning model updates.
It handles both intra-cluster and inter-cluster aggregation strategies for different
chess playstyles.

Modules:
    base_aggregator: Base classes and interfaces for aggregation algorithms
    intra_cluster_aggregator: Aggregation within clusters (same playstyle)
    inter_cluster_aggregator: Cross-cluster aggregation for global models

Available aggregation implementations:
- IntraClusterAggregator: FedAvg-based aggregation within clusters
- InterClusterAggregator: Cross-cluster aggregation for global models
"""

# Import aggregator classes
from .base_aggregator import BaseAggregator, AggregationMetrics
from .intra_cluster_aggregator import IntraClusterAggregator
from .inter_cluster_aggregator import InterClusterAggregator

__all__ = [
    BaseAggregator.__name__,
    AggregationMetrics.__name__,
    IntraClusterAggregator.__name__,
    InterClusterAggregator.__name__,
]

__version__ = '1.0.0'