"""
Unit tests for FileMetricsStore.

Tests JSONL-based metrics storage implementation.
"""

import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from server.storage.base import EntityType, MetricEvent
from server.storage.file_metrics_store import FileMetricsStore


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir()
    yield storage_dir
    # Cleanup
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def metrics_store(temp_storage_dir):
    """Create FileMetricsStore instance."""
    return FileMetricsStore(base_path=temp_storage_dir, compression=True)


@pytest.fixture
def sample_event():
    """Create a sample MetricEvent."""
    return MetricEvent(
        run_id="test_run",
        round_num=1,
        timestamp=datetime.now(),
        entity_type=EntityType.NODE,
        entity_id="node_001",
        metrics={"loss": 0.5, "accuracy": 0.85}
    )


class TestFileMetricsStoreInitialization:
    """Test FileMetricsStore initialization."""

    def test_create_with_default_settings(self, temp_storage_dir):
        """Test creating store with default settings."""
        store = FileMetricsStore(base_path=temp_storage_dir)

        assert store.base_path == temp_storage_dir
        assert store.compression is True
        assert store.organize_by_entity is False
        assert store.auto_index is True

    def test_create_with_custom_compression(self, temp_storage_dir):
        """Test creating store with compression disabled."""
        store = FileMetricsStore(base_path=temp_storage_dir, compression=False)

        assert store.compression is False

    def test_create_with_organize_by_entity(self, temp_storage_dir):
        """Test creating store with entity organization."""
        store = FileMetricsStore(
            base_path=temp_storage_dir,
            organize_by_entity=True
        )

        assert store.organize_by_entity is True

    def test_directory_creation_on_init(self, tmp_path):
        """Test that base directory is created on init."""
        storage_dir = tmp_path / "new_storage"
        assert not storage_dir.exists()

        FileMetricsStore(base_path=storage_dir)

        assert storage_dir.exists()
        assert storage_dir.is_dir()


class TestSingleEventRecording:
    """Test recording single events."""

    @pytest.mark.asyncio
    async def test_record_single_event(self, metrics_store, sample_event):
        """Test recording a single event successfully."""
        await metrics_store.record_event(sample_event)

        # Verify file was created
        events_file = metrics_store._get_events_file("test_run")
        assert events_file.exists()

    @pytest.mark.asyncio
    async def test_file_in_correct_location(self, metrics_store, sample_event):
        """Test that event file is created in correct location."""
        await metrics_store.record_event(sample_event)

        expected_path = metrics_store.base_path / "test_run" / "events.jsonl.gz"
        assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_jsonl_format_validation(self, temp_storage_dir, sample_event):
        """Test that events are stored in valid JSONL format."""
        store = FileMetricsStore(base_path=temp_storage_dir, compression=False)
        await store.record_event(sample_event)

        # Read and parse JSONL
        events_file = store._get_events_file("test_run")
        with open(events_file, "r") as f:
            line = f.readline()
            data = json.loads(line)

        assert data["run_id"] == "test_run"
        assert data["entity_id"] == "node_001"
        assert data["metrics"]["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_gzip_compression_when_enabled(self, metrics_store, sample_event):
        """Test that gzip compression is used when enabled."""
        await metrics_store.record_event(sample_event)

        events_file = metrics_store._get_events_file("test_run")

        # File should have .gz extension
        assert events_file.suffix == ".gz"

        # Should be readable with gzip
        with gzip.open(events_file, "rt") as f:
            line = f.readline()
            data = json.loads(line)
        assert data["run_id"] == "test_run"

    @pytest.mark.asyncio
    async def test_no_compression_when_disabled(self, temp_storage_dir, sample_event):
        """Test that compression is skipped when disabled."""
        store = FileMetricsStore(base_path=temp_storage_dir, compression=False)
        await store.record_event(sample_event)

        events_file = store._get_events_file("test_run")

        # File should NOT have .gz extension
        assert events_file.suffix == ".jsonl"

        # Should be readable as plain text
        with open(events_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
        assert data["run_id"] == "test_run"


class TestBatchEventRecording:
    """Test recording multiple events."""

    @pytest.mark.asyncio
    async def test_record_multiple_events(self, metrics_store):
        """Test recording multiple events efficiently."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id=f"node_{i:03d}",
                metrics={"loss": 0.5 - i * 0.01}
            )
            for i in range(5)
        ]

        await metrics_store.record_events(events)

        # Verify all events written
        events_file = metrics_store._get_events_file("test_run")
        with gzip.open(events_file, "rt") as f:
            lines = f.readlines()

        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_events_grouped_by_run_id(self, metrics_store):
        """Test that events are grouped by run_id."""
        events = [
            MetricEvent(
                run_id="run_1",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5}
            ),
            MetricEvent(
                run_id="run_2",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_2",
                metrics={"loss": 0.4}
            ),
            MetricEvent(
                run_id="run_1",
                round_num=2,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.3}
            )
        ]

        await metrics_store.record_events(events)

        # Verify separate files
        run1_file = metrics_store._get_events_file("run_1")
        run2_file = metrics_store._get_events_file("run_2")

        assert run1_file.exists()
        assert run2_file.exists()

        # run_1 should have 2 events
        with gzip.open(run1_file, "rt") as f:
            assert len(f.readlines()) == 2

        # run_2 should have 1 event
        with gzip.open(run2_file, "rt") as f:
            assert len(f.readlines()) == 1

    @pytest.mark.asyncio
    async def test_empty_events_list(self, metrics_store):
        """Test that empty events list is handled gracefully."""
        await metrics_store.record_events([])

        # Should not create any files
        assert len(list(metrics_store.base_path.glob("**/events.jsonl*"))) == 0


class TestIndexManagement:
    """Test automatic index generation and updates."""

    @pytest.mark.asyncio
    async def test_index_auto_creation(self, metrics_store, sample_event):
        """Test that index is created automatically on first write."""
        await metrics_store.record_event(sample_event)

        index_file = metrics_store._get_index_file("test_run")
        assert index_file.exists()

    @pytest.mark.asyncio
    async def test_index_updates_incrementally(self, metrics_store):
        """Test that index updates incrementally."""
        # Record first event
        event1 = MetricEvent(
            run_id="test_run",
            round_num=1,
            timestamp=datetime.now(),
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5}
        )
        await metrics_store.record_event(event1)

        index1 = await metrics_store._load_index("test_run")
        assert index1["event_count"] == 1

        # Record second event
        event2 = MetricEvent(
            run_id="test_run",
            round_num=2,
            timestamp=datetime.now(),
            entity_type=EntityType.NODE,
            entity_id="node_2",
            metrics={"loss": 0.4}
        )
        await metrics_store.record_event(event2)

        index2 = await metrics_store._load_index("test_run")
        assert index2["event_count"] == 2

    @pytest.mark.asyncio
    async def test_index_accuracy(self, metrics_store):
        """Test that index contains accurate information."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE if i % 2 == 0 else EntityType.CLUSTER,
                entity_id=f"entity_{i}",
                metrics={"loss": 0.5, "accuracy": 0.8, f"metric_{i}": i}
            )
            for i in range(1, 6)
        ]

        await metrics_store.record_events(events)

        index = await metrics_store._load_index("test_run")

        # Check round range
        assert index["round_range"] == [1, 5]

        # Check entity counts
        assert "node" in index["entity_ids"]
        assert "cluster" in index["entity_ids"]

        # Check metrics discovered
        node_metrics = set(index["metrics_by_entity"]["node"])
        assert "loss" in node_metrics
        assert "accuracy" in node_metrics

    @pytest.mark.asyncio
    async def test_index_caching(self, metrics_store, sample_event):
        """Test that index is cached correctly."""
        await metrics_store.record_event(sample_event)

        # First load
        index1 = await metrics_store._load_index("test_run")

        # Second load should come from cache
        index2 = await metrics_store._load_index("test_run")

        assert index1 is index2  # Same object from cache


class TestQuerying:
    """Test querying events with filters."""

    @pytest.mark.asyncio
    async def test_query_all_events(self, metrics_store):
        """Test querying all events for a run."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id=f"node_{i}",
                metrics={"loss": 0.5}
            )
            for i in range(3)
        ]

        await metrics_store.record_events(events)

        results = await metrics_store.query_events("test_run")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_filter_by_entity_type(self, metrics_store):
        """Test filtering by entity type."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5}
            ),
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.CLUSTER,
                entity_id="cluster_1",
                metrics={"loss": 0.3}
            )
        ]

        await metrics_store.record_events(events)

        node_events = await metrics_store.query_events(
            "test_run",
            entity_type=EntityType.NODE
        )
        assert len(node_events) == 1
        assert node_events[0].entity_type == EntityType.NODE

    @pytest.mark.asyncio
    async def test_filter_by_entity_id(self, metrics_store):
        """Test filtering by entity ID."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1" if i % 2 == 0 else "node_2",
                metrics={"loss": 0.5}
            )
            for i in range(4)
        ]

        await metrics_store.record_events(events)

        node1_events = await metrics_store.query_events(
            "test_run",
            entity_id="node_1"
        )
        assert len(node1_events) == 2
        assert all(e.entity_id == "node_1" for e in node1_events)

    @pytest.mark.asyncio
    async def test_filter_by_round_range(self, metrics_store):
        """Test filtering by round range."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5}
            )
            for i in range(1, 11)
        ]

        await metrics_store.record_events(events)

        filtered = await metrics_store.query_events(
            "test_run",
            round_range=(3, 7)
        )
        assert len(filtered) == 5
        assert all(3 <= e.round_num <= 7 for e in filtered)

    @pytest.mark.asyncio
    async def test_filter_by_metric_names(self, metrics_store):
        """Test filtering by metric names."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5, "accuracy": 0.8}
            ),
            MetricEvent(
                run_id="test_run",
                round_num=2,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.4}  # No accuracy
            )
        ]

        await metrics_store.record_events(events)

        with_accuracy = await metrics_store.query_events(
            "test_run",
            metric_names=["accuracy"]
        )
        assert len(with_accuracy) == 1

    @pytest.mark.asyncio
    async def test_combined_filters(self, metrics_store):
        """Test using multiple filters together."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE if i % 2 == 0 else EntityType.CLUSTER,
                entity_id=f"entity_{i}",
                metrics={"loss": 0.5}
            )
            for i in range(1, 11)
        ]

        await metrics_store.record_events(events)

        filtered = await metrics_store.query_events(
            "test_run",
            entity_type=EntityType.NODE,
            round_range=(2, 8)
        )

        # Should only include NODE entities in rounds 2-8
        assert all(e.entity_type == EntityType.NODE for e in filtered)
        assert all(2 <= e.round_num <= 8 for e in filtered)

    @pytest.mark.asyncio
    async def test_empty_result_handling(self, metrics_store):
        """Test that empty results are handled correctly."""
        results = await metrics_store.query_events("nonexistent_run")
        assert results == []


class TestMetricDiscovery:
    """Test automatic metric discovery."""

    @pytest.mark.asyncio
    async def test_get_available_metrics_with_index(self, metrics_store):
        """Test getting available metrics using index."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5, "accuracy": 0.8, "custom": 42}
            )
        ]

        await metrics_store.record_events(events)

        metrics = await metrics_store.get_available_metrics("test_run")
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "custom" in metrics

    @pytest.mark.asyncio
    async def test_filter_by_entity_type_discovery(self, metrics_store):
        """Test metric discovery filtered by entity type."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"node_metric": 1.0}
            ),
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.CLUSTER,
                entity_id="cluster_1",
                metrics={"cluster_metric": 2.0}
            )
        ]

        await metrics_store.record_events(events)

        node_metrics = await metrics_store.get_available_metrics(
            "test_run",
            entity_type=EntityType.NODE
        )
        assert "node_metric" in node_metrics
        assert "cluster_metric" not in node_metrics


class TestTimeSeries:
    """Test time series retrieval."""

    @pytest.mark.asyncio
    async def test_get_metric_series(self, metrics_store):
        """Test getting time series for a specific metric."""
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=i,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5 - i * 0.05}
            )
            for i in range(1, 6)
        ]

        await metrics_store.record_events(events)

        series = await metrics_store.get_metric_series(
            "test_run",
            "node_1",
            "loss"
        )

        assert len(series) == 5
        assert series[0] == (1, 0.45)
        assert series[4] == (5, 0.25)

    @pytest.mark.asyncio
    async def test_series_sorted_by_round(self, metrics_store):
        """Test that series is sorted by round number."""
        # Insert out of order
        events = [
            MetricEvent(
                run_id="test_run",
                round_num=3,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.3}
            ),
            MetricEvent(
                run_id="test_run",
                round_num=1,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5}
            ),
            MetricEvent(
                run_id="test_run",
                round_num=2,
                timestamp=datetime.now(),
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.4}
            )
        ]

        await metrics_store.record_events(events)

        series = await metrics_store.get_metric_series(
            "test_run",
            "node_1",
            "loss"
        )

        # Should be sorted
        assert series == [(1, 0.5), (2, 0.4), (3, 0.3)]


class TestDeletion:
    """Test deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_run(self, metrics_store, sample_event):
        """Test deleting all data for a run."""
        await metrics_store.record_event(sample_event)

        # Verify file exists
        run_path = metrics_store._get_run_path("test_run")
        assert run_path.exists()

        # Delete
        await metrics_store.delete_run("test_run")

        # Verify removed
        assert not run_path.exists()

    @pytest.mark.asyncio
    async def test_cache_cleared_on_deletion(self, metrics_store, sample_event):
        """Test that cache is cleared when run is deleted."""
        await metrics_store.record_event(sample_event)

        # Load index to cache it
        await metrics_store._load_index("test_run")
        assert "test_run" in metrics_store._index_cache

        # Delete run
        await metrics_store.delete_run("test_run")

        # Cache should be cleared
        assert "test_run" not in metrics_store._index_cache

    @pytest.mark.asyncio
    async def test_delete_nonexistent_run(self, metrics_store):
        """Test deleting a non-existent run doesn't raise error."""
        # Should not raise
        await metrics_store.delete_run("nonexistent_run")
