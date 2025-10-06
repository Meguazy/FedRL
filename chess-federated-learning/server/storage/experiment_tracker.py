"""
Experiment tracker for coordinating storage components.

This module provides high-level experiment lifecycle management:
- Start/end experiment runs
- Unified logging interface for metrics
- Checkpoint saving/loading coordination
- Run metadata management
- Integration with metrics store and model repository

The ExperimentTracker is the main entry point for the storage system.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .base import (
    EntityType,
    ExperimentTracker,
    MetricEvent,
    MetricsStore,
    ModelCheckpointMetadata,
    ModelRepository,
)


class FileExperimentTracker(ExperimentTracker):
    """
    File-based experiment tracker.

    Coordinates MetricsStore and ModelRepository to provide
    a unified interface for experiment tracking.
    """

    def __init__(
        self,
        metrics_store: MetricsStore,
        model_repository: ModelRepository,
        base_path: str | Path
    ):
        """
        Initialize experiment tracker.

        Args:
            metrics_store: Metrics storage backend
            model_repository: Model checkpoint storage backend
            base_path: Base directory for experiment metadata
        """
        self.metrics_store = metrics_store
        self.model_repository = model_repository
        self.base_path = Path(base_path)

        # Create metadata directory
        self.metadata_path = self.base_path / ".metadata"
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        # Active runs tracking
        self._active_runs: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized FileExperimentTracker at {self.base_path}")

    def _get_run_metadata_path(self, run_id: str) -> Path:
        """Get path to run metadata file."""
        return self.metadata_path / f"{run_id}.json"

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_id}"

    async def start_run(
        self,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """Start a new experiment run."""
        # Generate run ID if not provided
        if run_id is None:
            run_id = self._generate_run_id()

        # Check if run already exists
        if run_id in self._active_runs:
            raise ValueError(f"Run {run_id} is already active")

        # Create run metadata
        run_info = {
            "run_id": run_id,
            "description": description or "",
            "config": config,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "final_results": None
        }

        # Save metadata to file
        import json
        metadata_path = self._get_run_metadata_path(run_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        # Track active run
        self._active_runs[run_id] = run_info

        logger.info(f"Started experiment run: {run_id}")
        if description:
            logger.info(f"  Description: {description}")

        return run_id

    async def end_run(
        self,
        run_id: str,
        final_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """End an experiment run."""
        # Load run info
        run_info = await self.get_run_info(run_id)

        # Update metadata
        run_info["status"] = "completed"
        run_info["end_time"] = datetime.now().isoformat()
        run_info["final_results"] = final_results or {}

        # Save updated metadata
        import json
        metadata_path = self._get_run_metadata_path(run_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        # Remove from active runs
        if run_id in self._active_runs:
            del self._active_runs[run_id]

        logger.info(f"Ended experiment run: {run_id}")
        if final_results:
            logger.info(f"  Final results: {final_results}")

    async def log_metrics(
        self,
        run_id: str,
        round_num: int,
        entity_type: EntityType,
        entity_id: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics for an entity."""
        # Create metric event
        event = MetricEvent(
            run_id=run_id,
            round_num=round_num,
            timestamp=datetime.now(),
            entity_type=entity_type,
            entity_id=entity_id,
            metrics=metrics,
            metadata=metadata or {}
        )

        # Record to metrics store
        await self.metrics_store.record_event(event)

        logger.debug(
            f"Logged metrics for {entity_type.value}/{entity_id} "
            f"round {round_num}: {list(metrics.keys())}"
        )

    async def save_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int,
        model_state: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelCheckpointMetadata:
        """Save a model checkpoint."""
        checkpoint_metadata = await self.model_repository.save_model(
            run_id=run_id,
            cluster_id=cluster_id,
            round_num=round_num,
            model_state=model_state,
            metrics=metrics,
            metadata=metadata
        )

        logger.debug(
            f"Saved checkpoint for {cluster_id} round {round_num}: "
            f"{checkpoint_metadata.model_size_bytes / 1024 / 1024:.2f} MB"
        )

        return checkpoint_metadata

    async def load_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: Optional[int] = None,
        version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ModelCheckpointMetadata]:
        """Load a model checkpoint."""
        model_state, metadata = await self.model_repository.load_model(
            run_id=run_id,
            cluster_id=cluster_id,
            round_num=round_num,
            version=version
        )

        logger.debug(
            f"Loaded checkpoint for {cluster_id} "
            f"{'round ' + str(round_num) if round_num else version}"
        )

        return model_state, metadata

    async def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get information about a run."""
        # Check active runs first
        if run_id in self._active_runs:
            return self._active_runs[run_id].copy()

        # Load from file
        metadata_path = self._get_run_metadata_path(run_id)
        if not metadata_path.exists():
            raise ValueError(f"Run {run_id} not found")

        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def list_runs(self) -> List[Dict[str, Any]]:
        """List all experiment runs."""
        runs = []

        # Load all metadata files
        import json
        for metadata_file in self.metadata_path.glob("*.json"):
            # Try to load as run metadata
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    run_info = json.load(f)
                    # Verify it's a run metadata file (has run_id key)
                    if "run_id" in run_info:
                        runs.append(run_info)
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue

        # Sort by start time (most recent first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return runs

    async def get_metrics_summary(
        self,
        run_id: str,
        entity_type: Optional[EntityType] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of metrics for a run.

        Returns:
            Dictionary with:
            - available_metrics: List of metric names
            - entity_counts: Number of entities per type
            - round_range: (min, max) rounds
        """
        # Get available metrics
        available_metrics = await self.metrics_store.get_available_metrics(
            run_id, entity_type
        )

        # Query all events to get summary stats
        events = await self.metrics_store.query_events(run_id, entity_type)

        # Count entities
        entity_counts = {}
        round_nums = []

        for event in events:
            # Count entities by type
            entity_type_str = event.entity_type.value
            if entity_type_str not in entity_counts:
                entity_counts[entity_type_str] = set()
            entity_counts[entity_type_str].add(event.entity_id)

            # Track rounds
            round_nums.append(event.round_num)

        # Convert sets to counts
        entity_counts = {k: len(v) for k, v in entity_counts.items()}

        # Round range
        round_range = None
        if round_nums:
            round_range = (min(round_nums), max(round_nums))

        return {
            "available_metrics": available_metrics,
            "entity_counts": entity_counts,
            "round_range": round_range,
            "total_events": len(events)
        }

    async def get_checkpoint_summary(
        self,
        run_id: str,
        cluster_id: str
    ) -> Dict[str, Any]:
        """
        Get a summary of checkpoints for a cluster.

        Returns:
            Dictionary with:
            - checkpoint_count: Total number of checkpoints
            - latest_round: Latest checkpoint round
            - best_checkpoint: Best checkpoint metadata
            - total_size_mb: Total storage used
        """
        checkpoints = await self.model_repository.list_checkpoints(run_id, cluster_id)

        if not checkpoints:
            return {
                "checkpoint_count": 0,
                "latest_round": None,
                "best_checkpoint": None,
                "total_size_mb": 0
            }

        # Calculate total size
        total_size = sum(cp.model_size_bytes for cp in checkpoints)

        # Get best checkpoint
        best_checkpoint = await self.model_repository.get_best_checkpoint(
            run_id, cluster_id
        )

        return {
            "checkpoint_count": len(checkpoints),
            "latest_round": checkpoints[-1].round_num,
            "best_checkpoint": best_checkpoint.to_dict() if best_checkpoint else None,
            "total_size_mb": total_size / 1024 / 1024
        }

    async def export_metrics(
        self,
        run_id: str,
        output_path: str | Path,
        format: str = "csv"
    ) -> None:
        """
        Export metrics to file.

        Args:
            run_id: Experiment run ID
            output_path: Path to output file
            format: Export format ('csv' or 'json')
        """
        output_path = Path(output_path)

        if format == "csv":
            # Export to CSV via DataFrame
            df = await self.metrics_store.to_dataframe(run_id)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported metrics to CSV: {output_path}")

        elif format == "json":
            # Export to JSON
            events = await self.metrics_store.query_events(run_id)
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [event.to_dict() for event in events],
                    f,
                    indent=2
                )
            logger.info(f"Exported metrics to JSON: {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")

    async def cleanup_run(
        self,
        run_id: str,
        delete_metrics: bool = True,
        delete_models: bool = True
    ) -> None:
        """
        Clean up data for a run.

        Args:
            run_id: Experiment run ID
            delete_metrics: If True, delete metrics
            delete_models: If True, delete model checkpoints
        """
        if delete_metrics:
            await self.metrics_store.delete_run(run_id)
            logger.info(f"Deleted metrics for run {run_id}")

        if delete_models:
            await self.model_repository.delete_run(run_id)
            logger.info(f"Deleted models for run {run_id}")

        # Delete metadata
        metadata_path = self._get_run_metadata_path(run_id)
        if metadata_path.exists():
            metadata_path.unlink()

        # Remove from active runs
        if run_id in self._active_runs:
            del self._active_runs[run_id]

        logger.info(f"Cleaned up run {run_id}")
