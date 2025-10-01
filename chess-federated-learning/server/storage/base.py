"""
Base abstractions for the storage layer.

This module defines the core interfaces and data structures for the storage system:
- MetricEvent: Universal container for any metric data (schema-free)
- EntityType: Types of entities that can have metrics (node, cluster, global, system)
- MetricsStore: Abstract interface for metric storage backends
- ModelRepository: Abstract interface for model checkpoint storage
- ExperimentTracker: Abstract interface for experiment lifecycle management

The storage system is designed to be:
1. Schema-free: Add new metrics without code changes
2. Extensible: Plugin-based metric computation
3. Simple: Local filesystem only, no external dependencies
4. Git-friendly: Separate code from data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


class EntityType(str, Enum):
    """Types of entities that can have metrics."""
    NODE = "node"           # Individual training node (e.g., agg_001)
    CLUSTER = "cluster"     # Cluster aggregate (e.g., cluster_aggressive)
    GLOBAL = "global"       # Global/system-wide metrics
    SYSTEM = "system"       # System performance metrics (CPU, memory, etc.)
    CUSTOM = "custom"       # User-defined custom entities


@dataclass
class MetricEvent:
    """
    Universal container for metric events.

    Schema-free design: The 'metrics' dict can contain any structure.
    No predefined schema - add new metrics without code changes.

    Attributes:
        run_id: Unique identifier for the experiment run
        round_num: Training round number
        timestamp: When the metric was recorded
        entity_type: Type of entity (node, cluster, global, etc.)
        entity_id: Specific entity identifier
        metrics: Arbitrary metric data (schema-free dict)
        metadata: Optional additional context
    """
    run_id: str
    round_num: int
    timestamp: datetime
    entity_type: EntityType
    entity_id: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "round_num": self.round_num,
            "timestamp": self.timestamp.isoformat(),
            "entity_type": self.entity_type.value,
            "entity_id": self.entity_id,
            "metrics": self.metrics,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricEvent':
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            round_num=data["round_num"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            entity_type=EntityType(data["entity_type"]),
            entity_id=data["entity_id"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'MetricEvent':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ModelCheckpointMetadata:
    """Metadata for a model checkpoint."""
    run_id: str
    cluster_id: str
    round_num: int
    timestamp: datetime
    checksum: str  # SHA256 checksum for integrity
    metrics: Dict[str, Any]  # Metrics at time of checkpoint
    model_size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "cluster_id": self.cluster_id,
            "round_num": self.round_num,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "metrics": self.metrics,
            "model_size_bytes": self.model_size_bytes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCheckpointMetadata':
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            cluster_id=data["cluster_id"],
            round_num=data["round_num"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            checksum=data["checksum"],
            metrics=data["metrics"],
            model_size_bytes=data["model_size_bytes"],
            metadata=data.get("metadata", {})
        )


class MetricsStore(ABC):
    """
    Abstract interface for metric storage backends.

    Implementations store MetricEvent objects and support queries.
    Schema-free: No predefined metric schema required.
    """

    @abstractmethod
    async def record_event(self, event: MetricEvent) -> None:
        """Record a single metric event."""
        pass

    @abstractmethod
    async def record_events(self, events: List[MetricEvent]) -> None:
        """Record multiple metric events (batch operation)."""
        pass

    @abstractmethod
    async def query_events(
        self,
        run_id: str,
        entity_type: Optional[EntityType] = None,
        entity_id: Optional[str] = None,
        round_range: Optional[Tuple[int, int]] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[MetricEvent]:
        """
        Query metric events with filters.

        Args:
            run_id: Experiment run ID
            entity_type: Filter by entity type
            entity_id: Filter by specific entity
            round_range: Filter by round range (min, max) inclusive
            metric_names: Filter events containing these metric names

        Returns:
            List of matching metric events
        """
        pass

    @abstractmethod
    async def get_available_metrics(
        self,
        run_id: str,
        entity_type: Optional[EntityType] = None
    ) -> List[str]:
        """
        Discover available metric names (automatic schema discovery).

        Args:
            run_id: Experiment run ID
            entity_type: Optional filter by entity type

        Returns:
            List of unique metric names found in the data
        """
        pass

    @abstractmethod
    async def get_metric_series(
        self,
        run_id: str,
        entity_id: str,
        metric_name: str
    ) -> List[Tuple[int, Any]]:
        """
        Get time series for a specific metric.

        Args:
            run_id: Experiment run ID
            entity_id: Specific entity
            metric_name: Name of metric to retrieve

        Returns:
            List of (round_num, metric_value) tuples
        """
        pass

    @abstractmethod
    async def delete_run(self, run_id: str) -> None:
        """Delete all metrics for a run."""
        pass


class ModelRepository(ABC):
    """
    Abstract interface for model checkpoint storage.

    Handles saving/loading model checkpoints with metadata and integrity checks.
    """

    @abstractmethod
    async def save_model(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int,
        model_state: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelCheckpointMetadata:
        """
        Save a model checkpoint.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier
            round_num: Training round number
            model_state: Model state dict (e.g., PyTorch state_dict)
            metrics: Metrics at time of checkpoint
            metadata: Optional additional metadata

        Returns:
            Metadata for the saved checkpoint
        """
        pass

    @abstractmethod
    async def load_model(
        self,
        run_id: str,
        cluster_id: str,
        round_num: Optional[int] = None,
        version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ModelCheckpointMetadata]:
        """
        Load a model checkpoint.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier
            round_num: Specific round number (if None, uses version)
            version: Version to load ('latest', 'best', or None)

        Returns:
            Tuple of (model_state, metadata)
        """
        pass

    @abstractmethod
    async def list_checkpoints(
        self,
        run_id: str,
        cluster_id: str
    ) -> List[ModelCheckpointMetadata]:
        """
        List all checkpoints for a cluster.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier

        Returns:
            List of checkpoint metadata, sorted by round_num
        """
        pass

    @abstractmethod
    async def get_best_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        metric_name: str = "loss",
        minimize: bool = True
    ) -> Optional[ModelCheckpointMetadata]:
        """
        Get the best checkpoint based on a metric.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier
            metric_name: Metric to optimize
            minimize: If True, minimize metric; if False, maximize

        Returns:
            Metadata for best checkpoint, or None if no checkpoints
        """
        pass

    @abstractmethod
    async def delete_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int
    ) -> None:
        """Delete a specific checkpoint."""
        pass

    @abstractmethod
    async def delete_run(self, run_id: str) -> None:
        """Delete all checkpoints for a run."""
        pass


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment lifecycle management.

    High-level coordination of metrics and model storage.
    Provides unified API for experiment tracking.
    """

    @abstractmethod
    async def start_run(
        self,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Start a new experiment run.

        Args:
            config: Experiment configuration
            run_id: Optional custom run ID (auto-generated if None)
            description: Optional run description

        Returns:
            Run ID for the new experiment
        """
        pass

    @abstractmethod
    async def end_run(
        self,
        run_id: str,
        final_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End an experiment run.

        Args:
            run_id: Experiment run ID
            final_results: Optional final results/summary
        """
        pass

    @abstractmethod
    async def log_metrics(
        self,
        run_id: str,
        round_num: int,
        entity_type: EntityType,
        entity_id: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log metrics for an entity.

        Args:
            run_id: Experiment run ID
            round_num: Training round number
            entity_type: Type of entity
            entity_id: Entity identifier
            metrics: Metric data (schema-free)
            metadata: Optional additional context
        """
        pass

    @abstractmethod
    async def save_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int,
        model_state: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelCheckpointMetadata:
        """
        Save a model checkpoint.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier
            round_num: Training round number
            model_state: Model state dict
            metrics: Metrics at checkpoint time
            metadata: Optional metadata

        Returns:
            Checkpoint metadata
        """
        pass

    @abstractmethod
    async def load_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: Optional[int] = None,
        version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ModelCheckpointMetadata]:
        """
        Load a model checkpoint.

        Args:
            run_id: Experiment run ID
            cluster_id: Cluster identifier
            round_num: Specific round (if None, uses version)
            version: Version to load ('latest', 'best')

        Returns:
            Tuple of (model_state, metadata)
        """
        pass

    @abstractmethod
    async def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a run.

        Args:
            run_id: Experiment run ID

        Returns:
            Dictionary with run information (config, status, timestamps, etc.)
        """
        pass

    @abstractmethod
    async def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all experiment runs.

        Returns:
            List of run information dictionaries
        """
        pass
