"""
Factory functions for creating storage components.

Provides convenient functions for creating pre-configured storage instances.
"""

from pathlib import Path
from typing import Optional

from .experiment_tracker import FileExperimentTracker
from .file_metrics_store import FileMetricsStore
from .local_model_repository import LocalModelRepository


def create_experiment_tracker(
    base_path: str | Path = ".chess-federated-learning/storage",
    metrics_backend: str = "file",
    model_backend: str = "local",
    compression: bool = True,
    organize_by_entity: bool = False,
    keep_last_n: Optional[int] = None,
    keep_best: bool = True,
    compute_checksums: bool = True
) -> FileExperimentTracker:
    """
    Create a fully configured experiment tracker.

    Args:
        base_path: Base directory for all storage
        metrics_backend: Metrics storage backend ('file' only for now)
        model_backend: Model storage backend ('local' only for now)
        compression: Enable gzip compression for metrics
        organize_by_entity: Organize metrics files by entity
        keep_last_n: Keep only last N checkpoints (None = keep all)
        keep_best: Always keep best checkpoint
        compute_checksums: Compute SHA256 checksums for models

    Returns:
        Configured FileExperimentTracker instance

    Example:
        tracker = create_experiment_tracker(base_path="./storage")
        run_id = await tracker.start_run(config={"nodes": 64})
    """
    base_path = Path(base_path)

    # Create metrics store
    if metrics_backend == "file":
        metrics_store = FileMetricsStore(
            base_path=base_path / "metrics",
            compression=compression,
            organize_by_entity=organize_by_entity
        )
    else:
        raise ValueError(f"Unsupported metrics backend: {metrics_backend}")

    # Create model repository
    if model_backend == "local":
        model_repository = LocalModelRepository(
            base_path=base_path / "models",
            keep_last_n=keep_last_n,
            keep_best=keep_best,
            compute_checksums=compute_checksums
        )
    else:
        raise ValueError(f"Unsupported model backend: {model_backend}")

    # Create experiment tracker
    tracker = FileExperimentTracker(
        metrics_store=metrics_store,
        model_repository=model_repository,
        base_path=base_path
    )

    return tracker


def create_metrics_store(
    base_path: str | Path = ".chess-federated-learning/storage/metrics",
    backend: str = "file",
    compression: bool = True,
    organize_by_entity: bool = False
) -> FileMetricsStore:
    """
    Create a standalone metrics store.

    Args:
        base_path: Base directory for metrics storage
        backend: Storage backend ('file' only for now)
        compression: Enable gzip compression
        organize_by_entity: Organize files by entity

    Returns:
        Configured metrics store instance
    """
    if backend == "file":
        return FileMetricsStore(
            base_path=base_path,
            compression=compression,
            organize_by_entity=organize_by_entity
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def create_model_repository(
    base_path: str | Path = ".chess-federated-learning/storage/models",
    backend: str = "local",
    keep_last_n: Optional[int] = None,
    keep_best: bool = True,
    compute_checksums: bool = True
) -> LocalModelRepository:
    """
    Create a standalone model repository.

    Args:
        base_path: Base directory for model storage
        backend: Storage backend ('local' only for now)
        keep_last_n: Keep only last N checkpoints (None = keep all)
        keep_best: Always keep best checkpoint
        compute_checksums: Compute SHA256 checksums

    Returns:
        Configured model repository instance
    """
    if backend == "local":
        return LocalModelRepository(
            base_path=base_path,
            keep_last_n=keep_last_n,
            keep_best=keep_best,
            compute_checksums=compute_checksums
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
