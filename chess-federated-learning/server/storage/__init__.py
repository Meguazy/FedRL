"""
Storage layer for federated learning experiments.

This package provides a modular, schema-free storage system for:
- Metrics: Time-series metrics from nodes, clusters, and global aggregation
- Models: Model checkpoints with metadata and integrity checks
- Experiments: Lifecycle management and coordination

Key Features:
- Schema-free: Add new metrics without code changes
- Plugin-based: Extensible metric computation
- Local filesystem: Simple, no external dependencies
- Git-friendly: Separate code from data

Example:
    from server.storage import create_experiment_tracker

    tracker = create_experiment_tracker(base_path="./storage")
    run_id = await tracker.start_run(config={"nodes": 64})

    await tracker.log_metrics(
        run_id=run_id,
        round_num=1,
        entity_type=EntityType.NODE,
        entity_id="agg_001",
        metrics={"loss": 0.5, "accuracy": 0.85}
    )

    await tracker.end_run(run_id)
"""

from .base import (
    EntityType,
    MetricEvent,
    ModelCheckpointMetadata,
    MetricsStore,
    ModelRepository,
    ExperimentTracker,
)
from .file_metrics_store import FileMetricsStore
from .local_model_repository import LocalModelRepository
from .experiment_tracker import FileExperimentTracker
from .factory import (
    create_experiment_tracker,
    create_metrics_store,
    create_model_repository,
)

__all__ = [
    "EntityType",
    "MetricEvent",
    "ModelCheckpointMetadata",
    "MetricsStore",
    "ModelRepository",
    "ExperimentTracker",
    "FileMetricsStore",
    "LocalModelRepository",
    "FileExperimentTracker",
    "create_experiment_tracker",
    "create_metrics_store",
    "create_model_repository",
]

__version__ = "1.0.0"
