"""
JSONL-based metrics storage implementation.

This module provides a file-based metrics storage backend using JSONL format:
- Append-only JSONL files for metric events
- Optional gzip compression
- Automatic index generation for fast queries
- Schema discovery (auto-detect available metrics)
- DataFrame export for analysis

Storage structure:
    storage/metrics/{run_id}/
    ├── events.jsonl.gz          # All metric events (compressed)
    ├── index.json               # Fast query index
    └── by_entity/               # Optional organized view
        ├── node/
        │   └── {entity_id}.jsonl.gz
        ├── cluster/
        │   └── {entity_id}.jsonl.gz
        └── global/
            └── {entity_id}.jsonl.gz
"""

import asyncio
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from server.storage.base import EntityType, MetricEvent, MetricsStore


class FileMetricsStore(MetricsStore):
    """
    File-based metrics storage using JSONL format.

    Features:
    - Append-only writes for performance
    - Optional gzip compression
    - Automatic index generation
    - Schema-free metric storage
    - Fast queries with index
    """

    def __init__(
        self,
        base_path: str | Path,
        compression: bool = True,
        organize_by_entity: bool = False,
        auto_index: bool = True
    ):
        """
        Initialize file metrics store.

        Args:
            base_path: Base directory for metrics storage
            compression: If True, use gzip compression
            organize_by_entity: If True, create per-entity files
            auto_index: If True, automatically update index on writes
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.organize_by_entity = organize_by_entity
        self.auto_index = auto_index

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Cache for indices
        self._index_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized FileMetricsStore at {self.base_path} "
            f"(compression={compression}, organize_by_entity={organize_by_entity})"
        )

    def _get_run_path(self, run_id: str) -> Path:
        """Get path for a specific run."""
        return self.base_path / run_id

    def _get_events_file(self, run_id: str) -> Path:
        """Get path to main events file."""
        run_path = self._get_run_path(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        filename = "events.jsonl.gz" if self.compression else "events.jsonl"
        return run_path / filename

    def _get_entity_file(self, run_id: str, entity_type: EntityType, entity_id: str) -> Path:
        """Get path to entity-specific file."""
        run_path = self._get_run_path(run_id)
        entity_dir = run_path / "by_entity" / entity_type.value
        entity_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{entity_id}.jsonl.gz" if self.compression else f"{entity_id}.jsonl"
        return entity_dir / filename

    def _get_index_file(self, run_id: str) -> Path:
        """Get path to index file."""
        return self._get_run_path(run_id) / "index.json"

    async def record_event(self, event: MetricEvent) -> None:
        """Record a single metric event."""
        await self.record_events([event])

    async def record_events(self, events: List[MetricEvent]) -> None:
        """Record multiple metric events (batch operation)."""
        if not events:
            return

        # Group events by run_id
        events_by_run: Dict[str, List[MetricEvent]] = {}
        for event in events:
            if event.run_id not in events_by_run:
                events_by_run[event.run_id] = []
            events_by_run[event.run_id].append(event)

        # Write events for each run
        for run_id, run_events in events_by_run.items():
            await self._write_events(run_id, run_events)

            if self.auto_index:
                await self._update_index(run_id, run_events)

    async def _write_events(self, run_id: str, events: List[MetricEvent]) -> None:
        """Write events to file(s)."""
        # Write to main events file
        events_file = self._get_events_file(run_id)
        await self._append_to_file(events_file, events)

        # Optionally write to entity-specific files
        if self.organize_by_entity:
            events_by_entity: Dict[Tuple[EntityType, str], List[MetricEvent]] = {}
            for event in events:
                key = (event.entity_type, event.entity_id)
                if key not in events_by_entity:
                    events_by_entity[key] = []
                events_by_entity[key].append(event)

            for (entity_type, entity_id), entity_events in events_by_entity.items():
                entity_file = self._get_entity_file(run_id, entity_type, entity_id)
                await self._append_to_file(entity_file, entity_events)

    async def _append_to_file(self, file_path: Path, events: List[MetricEvent]) -> None:
        """Append events to a JSONL file."""
        def _write():
            if self.compression:
                with gzip.open(file_path, "at", encoding="utf-8") as f:
                    for event in events:
                        f.write(event.to_json() + "\n")
            else:
                with open(file_path, "a", encoding="utf-8") as f:
                    for event in events:
                        f.write(event.to_json() + "\n")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)

    async def _update_index(self, run_id: str, new_events: List[MetricEvent]) -> None:
        """Update the index with new events."""
        # Load existing index or create new
        index = await self._load_index(run_id) or {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "round_range": [float('inf'), -float('inf')],
            "entity_ids": {},
            "metrics_by_entity": {},
            "event_count": 0
        }

        # Convert lists back to sets if loading from file
        if "entity_ids" in index:
            index["entity_ids"] = {
                k: set(v) if isinstance(v, list) else v
                for k, v in index["entity_ids"].items()
            }
        if "metrics_by_entity" in index:
            index["metrics_by_entity"] = {
                k: set(v) if isinstance(v, list) else v
                for k, v in index["metrics_by_entity"].items()
            }

        # Update with new events
        for event in new_events:
            # Update round range
            index["round_range"][0] = min(index["round_range"][0], event.round_num)
            index["round_range"][1] = max(index["round_range"][1], event.round_num)

            # Update entity IDs
            entity_type_key = event.entity_type.value
            if entity_type_key not in index["entity_ids"]:
                index["entity_ids"][entity_type_key] = set()
            index["entity_ids"][entity_type_key].add(event.entity_id)

            # Update metrics by entity
            if entity_type_key not in index["metrics_by_entity"]:
                index["metrics_by_entity"][entity_type_key] = set()
            index["metrics_by_entity"][entity_type_key].update(event.metrics.keys())

            # Increment event count
            index["event_count"] += 1

        # Convert sets to lists for JSON serialization
        index["last_updated"] = datetime.now().isoformat()
        serializable_index = {
            **index,
            "entity_ids": {k: sorted(list(v)) for k, v in index["entity_ids"].items()},
            "metrics_by_entity": {k: sorted(list(v)) for k, v in index["metrics_by_entity"].items()}
        }

        # Save index
        index_file = self._get_index_file(run_id)

        def _write():
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(serializable_index, f, indent=2)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)

        # Update cache
        self._index_cache[run_id] = serializable_index

    async def _load_index(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load index from file."""
        # Check cache first
        if run_id in self._index_cache:
            return self._index_cache[run_id]

        index_file = self._get_index_file(run_id)
        if not index_file.exists():
            return None

        def _read():
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)

        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(None, _read)

        # Cache it
        self._index_cache[run_id] = index
        return index

    async def query_events(
        self,
        run_id: str,
        entity_type: Optional[EntityType] = None,
        entity_id: Optional[str] = None,
        round_range: Optional[Tuple[int, int]] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[MetricEvent]:
        """Query metric events with filters."""
        # Determine which file to read from
        if self.organize_by_entity and entity_type and entity_id:
            # Read from entity-specific file
            file_path = self._get_entity_file(run_id, entity_type, entity_id)
        else:
            # Read from main events file
            file_path = self._get_events_file(run_id)

        if not file_path.exists():
            return []

        def _read():
            events = []
            if self.compression:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        event = MetricEvent.from_json(line.strip())
                        if self._matches_filters(
                            event, entity_type, entity_id, round_range, metric_names
                        ):
                            events.append(event)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        event = MetricEvent.from_json(line.strip())
                        if self._matches_filters(
                            event, entity_type, entity_id, round_range, metric_names
                        ):
                            events.append(event)
            return events

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)

    def _matches_filters(
        self,
        event: MetricEvent,
        entity_type: Optional[EntityType],
        entity_id: Optional[str],
        round_range: Optional[Tuple[int, int]],
        metric_names: Optional[List[str]]
    ) -> bool:
        """Check if event matches filters."""
        if entity_type and event.entity_type != entity_type:
            return False

        if entity_id and event.entity_id != entity_id:
            return False

        if round_range:
            min_round, max_round = round_range
            if event.round_num < min_round or event.round_num > max_round:
                return False

        if metric_names:
            # Check if event contains any of the requested metrics
            if not any(name in event.metrics for name in metric_names):
                return False

        return True

    async def get_available_metrics(
        self,
        run_id: str,
        entity_type: Optional[EntityType] = None
    ) -> List[str]:
        """Discover available metric names (automatic schema discovery)."""
        index = await self._load_index(run_id)

        if not index:
            # No index, scan events file
            return await self._discover_metrics_from_events(run_id, entity_type)

        # Use index for fast lookup
        if entity_type:
            return index.get("metrics_by_entity", {}).get(entity_type.value, [])
        else:
            # Return all unique metrics across all entity types
            all_metrics = set()
            for metrics in index.get("metrics_by_entity", {}).values():
                all_metrics.update(metrics)
            return sorted(list(all_metrics))

    async def _discover_metrics_from_events(
        self,
        run_id: str,
        entity_type: Optional[EntityType]
    ) -> List[str]:
        """Discover metrics by scanning events file."""
        events_file = self._get_events_file(run_id)
        if not events_file.exists():
            return []

        def _scan():
            metrics = set()
            if self.compression:
                with gzip.open(events_file, "rt", encoding="utf-8") as f:
                    for line in f:
                        event = MetricEvent.from_json(line.strip())
                        if entity_type is None or event.entity_type == entity_type:
                            metrics.update(event.metrics.keys())
            else:
                with open(events_file, "r", encoding="utf-8") as f:
                    for line in f:
                        event = MetricEvent.from_json(line.strip())
                        if entity_type is None or event.entity_type == entity_type:
                            metrics.update(event.metrics.keys())
            return sorted(list(metrics))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _scan)

    async def get_metric_series(
        self,
        run_id: str,
        entity_id: str,
        metric_name: str
    ) -> List[Tuple[int, Any]]:
        """Get time series for a specific metric."""
        # Query events for this entity
        events = await self.query_events(
            run_id=run_id,
            entity_id=entity_id,
            metric_names=[metric_name]
        )

        # Extract metric values
        series = [
            (event.round_num, event.metrics.get(metric_name))
            for event in events
            if metric_name in event.metrics
        ]

        # Sort by round number
        series.sort(key=lambda x: x[0])
        return series

    async def delete_run(self, run_id: str) -> None:
        """Delete all metrics for a run."""
        run_path = self._get_run_path(run_id)

        if run_path.exists():
            def _delete():
                import shutil
                shutil.rmtree(run_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _delete)

            # Clear cache
            if run_id in self._index_cache:
                del self._index_cache[run_id]

            logger.info(f"Deleted metrics for run {run_id}")

    async def to_dataframe(self, run_id: str) -> Any:
        """
        Export metrics to pandas DataFrame.

        Returns:
            pandas.DataFrame with columns:
            - run_id, round_num, timestamp, entity_type, entity_id
            - metric_* (one column per metric, dynamically discovered)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export. Install with: pip install pandas")

        # Load all events
        events = await self.query_events(run_id)

        if not events:
            return pd.DataFrame()

        # Convert to records
        records = []
        for event in events:
            record = {
                "run_id": event.run_id,
                "round_num": event.round_num,
                "timestamp": event.timestamp,
                "entity_type": event.entity_type.value,
                "entity_id": event.entity_id,
            }
            # Add metrics with 'metric_' prefix
            for key, value in event.metrics.items():
                record[f"metric_{key}"] = value
            records.append(record)

        return pd.DataFrame(records)
