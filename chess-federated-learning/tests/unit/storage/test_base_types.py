"""
Unit tests for base storage types.

Tests MetricEvent and ModelCheckpointMetadata data structures.
"""

import json
from datetime import datetime, timezone

import pytest

from server.storage.base import EntityType, MetricEvent, ModelCheckpointMetadata


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_type_values(self):
        """Test all EntityType values are defined."""
        assert EntityType.NODE == "node"
        assert EntityType.CLUSTER == "cluster"
        assert EntityType.GLOBAL == "global"
        assert EntityType.SYSTEM == "system"
        assert EntityType.CUSTOM == "custom"

    def test_entity_type_from_string(self):
        """Test creating EntityType from string."""
        assert EntityType("node") == EntityType.NODE
        assert EntityType("cluster") == EntityType.CLUSTER
        assert EntityType("global") == EntityType.GLOBAL
        assert EntityType("system") == EntityType.SYSTEM
        assert EntityType("custom") == EntityType.CUSTOM

    def test_invalid_entity_type(self):
        """Test invalid entity type raises error."""
        with pytest.raises(ValueError):
            EntityType("invalid")


class TestMetricEvent:
    """Test MetricEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating MetricEvent with all fields."""
        timestamp = datetime.now()
        event = MetricEvent(
            run_id="test_run",
            round_num=5,
            timestamp=timestamp,
            entity_type=EntityType.NODE,
            entity_id="node_001",
            metrics={"loss": 0.5, "accuracy": 0.85},
            metadata={"extra": "data"}
        )

        assert event.run_id == "test_run"
        assert event.round_num == 5
        assert event.timestamp == timestamp
        assert event.entity_type == EntityType.NODE
        assert event.entity_id == "node_001"
        assert event.metrics == {"loss": 0.5, "accuracy": 0.85}
        assert event.metadata == {"extra": "data"}

    def test_creation_with_defaults(self):
        """Test creating MetricEvent with default metadata."""
        event = MetricEvent(
            run_id="test_run",
            round_num=1,
            timestamp=datetime.now(),
            entity_type=EntityType.CLUSTER,
            entity_id="cluster_1",
            metrics={"loss": 0.3}
        )

        assert event.metadata == {}

    def test_to_dict_serialization(self):
        """Test to_dict() converts to dictionary correctly."""
        timestamp = datetime(2025, 1, 15, 10, 30, 45)
        event = MetricEvent(
            run_id="exp_001",
            round_num=10,
            timestamp=timestamp,
            entity_type=EntityType.NODE,
            entity_id="agg_001",
            metrics={"loss": 0.25, "samples": 1000},
            metadata={"node_version": "1.0"}
        )

        result = event.to_dict()

        assert result["run_id"] == "exp_001"
        assert result["round_num"] == 10
        assert result["timestamp"] == timestamp.isoformat()
        assert result["entity_type"] == "node"
        assert result["entity_id"] == "agg_001"
        assert result["metrics"] == {"loss": 0.25, "samples": 1000}
        assert result["metadata"] == {"node_version": "1.0"}

    def test_from_dict_deserialization(self):
        """Test from_dict() recreates MetricEvent from dict."""
        data = {
            "run_id": "exp_002",
            "round_num": 15,
            "timestamp": "2025-01-15T14:30:00",
            "entity_type": "cluster",
            "entity_id": "cluster_aggressive",
            "metrics": {"diversity": 0.8},
            "metadata": {"aggregation_time": 2.5}
        }

        event = MetricEvent.from_dict(data)

        assert event.run_id == "exp_002"
        assert event.round_num == 15
        assert event.timestamp == datetime(2025, 1, 15, 14, 30, 0)
        assert event.entity_type == EntityType.CLUSTER
        assert event.entity_id == "cluster_aggressive"
        assert event.metrics == {"diversity": 0.8}
        assert event.metadata == {"aggregation_time": 2.5}

    def test_from_dict_without_metadata(self):
        """Test from_dict() handles missing metadata field."""
        data = {
            "run_id": "exp_003",
            "round_num": 1,
            "timestamp": "2025-01-15T10:00:00",
            "entity_type": "global",
            "entity_id": "global",
            "metrics": {"total_loss": 0.5}
        }

        event = MetricEvent.from_dict(data)
        assert event.metadata == {}

    def test_to_json_serialization(self):
        """Test to_json() creates valid JSON string."""
        timestamp = datetime(2025, 1, 15, 12, 0, 0)
        event = MetricEvent(
            run_id="test",
            round_num=1,
            timestamp=timestamp,
            entity_type=EntityType.SYSTEM,
            entity_id="system",
            metrics={"cpu": 45.2}
        )

        json_str = event.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "test"
        assert parsed["entity_type"] == "system"
        assert parsed["metrics"]["cpu"] == 45.2

    def test_from_json_deserialization(self):
        """Test from_json() recreates MetricEvent from JSON."""
        json_str = json.dumps({
            "run_id": "json_test",
            "round_num": 5,
            "timestamp": "2025-01-15T15:45:00",
            "entity_type": "node",
            "entity_id": "pos_001",
            "metrics": {"loss": 0.15},
            "metadata": {}
        })

        event = MetricEvent.from_json(json_str)

        assert event.run_id == "json_test"
        assert event.round_num == 5
        assert event.entity_type == EntityType.NODE
        assert event.metrics == {"loss": 0.15}

    def test_round_trip_serialization(self):
        """Test full round-trip: event -> dict -> event."""
        original = MetricEvent(
            run_id="roundtrip",
            round_num=99,
            timestamp=datetime.now(),
            entity_type=EntityType.CUSTOM,
            entity_id="custom_123",
            metrics={"metric_a": 1.0, "metric_b": 2.0},
            metadata={"source": "test"}
        )

        # Round trip through dict
        data = original.to_dict()
        restored = MetricEvent.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.round_num == original.round_num
        assert restored.entity_type == original.entity_type
        assert restored.entity_id == original.entity_id
        assert restored.metrics == original.metrics
        assert restored.metadata == original.metadata

    def test_empty_metrics_dict(self):
        """Test MetricEvent with empty metrics dictionary."""
        event = MetricEvent(
            run_id="empty_test",
            round_num=1,
            timestamp=datetime.now(),
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={}
        )

        assert event.metrics == {}

        # Should serialize/deserialize correctly
        restored = MetricEvent.from_dict(event.to_dict())
        assert restored.metrics == {}

    def test_large_nested_metrics(self):
        """Test MetricEvent with large nested structure."""
        large_metrics = {
            "layer_weights": {
                f"layer_{i}": {
                    "mean": float(i),
                    "std": float(i * 0.1),
                    "values": list(range(10))
                }
                for i in range(100)
            },
            "summary": {"total_layers": 100}
        }

        event = MetricEvent(
            run_id="large_test",
            round_num=1,
            timestamp=datetime.now(),
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics=large_metrics
        )

        # Should serialize successfully
        json_str = event.to_json()
        assert len(json_str) > 0

        # Should deserialize successfully
        restored = MetricEvent.from_json(json_str)
        assert restored.metrics == large_metrics

    def test_timestamp_timezone_handling(self):
        """Test timestamp handling with timezone-aware datetime."""
        # Timezone-aware timestamp
        timestamp_aware = datetime.now(timezone.utc)
        event = MetricEvent(
            run_id="tz_test",
            round_num=1,
            timestamp=timestamp_aware,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5}
        )

        # Should serialize with timezone info
        iso_str = event.to_dict()["timestamp"]
        assert "+" in iso_str or "Z" in iso_str or "-" in iso_str  # Timezone indicator

    def test_metrics_with_various_types(self):
        """Test metrics dictionary with various Python types."""
        event = MetricEvent(
            run_id="types_test",
            round_num=1,
            timestamp=datetime.now(),
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={
                "int_val": 42,
                "float_val": 3.14159,
                "str_val": "test_string",
                "bool_val": True,
                "list_val": [1, 2, 3],
                "dict_val": {"nested": "value"},
                "none_val": None
            }
        )

        # Round trip
        restored = MetricEvent.from_dict(event.to_dict())
        assert restored.metrics["int_val"] == 42
        assert restored.metrics["float_val"] == 3.14159
        assert restored.metrics["str_val"] == "test_string"
        assert restored.metrics["bool_val"] is True
        assert restored.metrics["list_val"] == [1, 2, 3]
        assert restored.metrics["dict_val"] == {"nested": "value"}
        assert restored.metrics["none_val"] is None


class TestModelCheckpointMetadata:
    """Test ModelCheckpointMetadata dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating ModelCheckpointMetadata with all fields."""
        timestamp = datetime.now()
        metadata = ModelCheckpointMetadata(
            run_id="test_run",
            cluster_id="cluster_aggressive",
            round_num=10,
            timestamp=timestamp,
            checksum="abc123def456",
            metrics={"loss": 0.25, "accuracy": 0.90},
            model_size_bytes=1024000,
            metadata={"framework": "pytorch"}
        )

        assert metadata.run_id == "test_run"
        assert metadata.cluster_id == "cluster_aggressive"
        assert metadata.round_num == 10
        assert metadata.timestamp == timestamp
        assert metadata.checksum == "abc123def456"
        assert metadata.metrics == {"loss": 0.25, "accuracy": 0.90}
        assert metadata.model_size_bytes == 1024000
        assert metadata.metadata == {"framework": "pytorch"}

    def test_to_dict_serialization(self):
        """Test to_dict() converts to dictionary correctly."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0)
        metadata = ModelCheckpointMetadata(
            run_id="exp_001",
            cluster_id="cluster_positional",
            round_num=25,
            timestamp=timestamp,
            checksum="sha256hash",
            metrics={"loss": 0.15},
            model_size_bytes=2048000,
            metadata={}
        )

        result = metadata.to_dict()

        assert result["run_id"] == "exp_001"
        assert result["cluster_id"] == "cluster_positional"
        assert result["round_num"] == 25
        assert result["timestamp"] == timestamp.isoformat()
        assert result["checksum"] == "sha256hash"
        assert result["metrics"] == {"loss": 0.15}
        assert result["model_size_bytes"] == 2048000
        assert result["metadata"] == {}

    def test_from_dict_deserialization(self):
        """Test from_dict() recreates ModelCheckpointMetadata from dict."""
        data = {
            "run_id": "exp_002",
            "cluster_id": "cluster_aggressive",
            "round_num": 50,
            "timestamp": "2025-01-15T16:00:00",
            "checksum": "checksum123",
            "metrics": {"loss": 0.10, "elo": 1850},
            "model_size_bytes": 5120000,
            "metadata": {"version": "2.0"}
        }

        metadata = ModelCheckpointMetadata.from_dict(data)

        assert metadata.run_id == "exp_002"
        assert metadata.cluster_id == "cluster_aggressive"
        assert metadata.round_num == 50
        assert metadata.timestamp == datetime(2025, 1, 15, 16, 0, 0)
        assert metadata.checksum == "checksum123"
        assert metadata.metrics == {"loss": 0.10, "elo": 1850}
        assert metadata.model_size_bytes == 5120000
        assert metadata.metadata == {"version": "2.0"}

    def test_round_trip_serialization(self):
        """Test full round-trip: metadata -> dict -> metadata."""
        original = ModelCheckpointMetadata(
            run_id="roundtrip",
            cluster_id="cluster_test",
            round_num=100,
            timestamp=datetime.now(),
            checksum="hash_abc",
            metrics={"metric": 42.0},
            model_size_bytes=999999,
            metadata={"test": True}
        )

        # Round trip through dict
        data = original.to_dict()
        restored = ModelCheckpointMetadata.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.cluster_id == original.cluster_id
        assert restored.round_num == original.round_num
        assert restored.checksum == original.checksum
        assert restored.metrics == original.metrics
        assert restored.model_size_bytes == original.model_size_bytes
        assert restored.metadata == original.metadata

    def test_empty_checksum(self):
        """Test ModelCheckpointMetadata with empty checksum."""
        metadata = ModelCheckpointMetadata(
            run_id="test",
            cluster_id="cluster_1",
            round_num=1,
            timestamp=datetime.now(),
            checksum="",  # Empty checksum
            metrics={"loss": 0.5},
            model_size_bytes=1000
        )

        assert metadata.checksum == ""

        # Should serialize/deserialize correctly
        restored = ModelCheckpointMetadata.from_dict(metadata.to_dict())
        assert restored.checksum == ""

    def test_large_model_size(self):
        """Test ModelCheckpointMetadata with very large model size."""
        large_size = 10 * 1024 * 1024 * 1024  # 10 GB

        metadata = ModelCheckpointMetadata(
            run_id="large_model",
            cluster_id="cluster_1",
            round_num=1,
            timestamp=datetime.now(),
            checksum="large_hash",
            metrics={"loss": 0.1},
            model_size_bytes=large_size
        )

        assert metadata.model_size_bytes == large_size

        # Should serialize/deserialize correctly
        restored = ModelCheckpointMetadata.from_dict(metadata.to_dict())
        assert restored.model_size_bytes == large_size
