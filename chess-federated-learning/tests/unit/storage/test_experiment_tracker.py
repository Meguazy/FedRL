"""
Unit tests for FileExperimentTracker.

Tests experiment lifecycle management and coordination.
"""

import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch

from server.storage.base import EntityType
from server.storage.experiment_tracker import FileExperimentTracker
from server.storage.file_metrics_store import FileMetricsStore
from server.storage.local_model_repository import LocalModelRepository


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "test_experiments"
    storage_dir.mkdir()
    yield storage_dir
    # Cleanup
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def metrics_store(temp_storage_dir):
    """Create metrics store."""
    return FileMetricsStore(base_path=temp_storage_dir / "metrics")


@pytest.fixture
def model_repo(temp_storage_dir):
    """Create model repository."""
    return LocalModelRepository(base_path=temp_storage_dir / "models")


@pytest.fixture
def tracker(metrics_store, model_repo, temp_storage_dir):
    """Create experiment tracker."""
    return FileExperimentTracker(
        metrics_store=metrics_store,
        model_repository=model_repo,
        base_path=temp_storage_dir
    )


@pytest.fixture
def sample_model():
    """Create a sample model state dict."""
    return {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10)
    }


class TestRunLifecycle:
    """Test run lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_run_creates_run_id(self, tracker):
        """Test that start_run creates a run_id."""
        run_id = await tracker.start_run(config={"nodes": 8})

        assert run_id is not None
        assert run_id.startswith("run_")

    @pytest.mark.asyncio
    async def test_start_run_with_custom_run_id(self, tracker):
        """Test starting run with custom run_id."""
        custom_id = "custom_experiment_001"
        run_id = await tracker.start_run(
            config={"nodes": 8},
            run_id=custom_id
        )

        assert run_id == custom_id

    @pytest.mark.asyncio
    async def test_run_metadata_saved(self, tracker):
        """Test that run metadata is saved correctly."""
        run_id = await tracker.start_run(
            config={"nodes": 8, "rounds": 50},
            description="Test experiment"
        )

        run_info = await tracker.get_run_info(run_id)

        assert run_info["run_id"] == run_id
        assert run_info["config"] == {"nodes": 8, "rounds": 50}
        assert run_info["description"] == "Test experiment"
        assert run_info["status"] == "running"

    @pytest.mark.asyncio
    async def test_duplicate_run_id_rejected(self, tracker):
        """Test that duplicate run_id is rejected."""
        run_id = "duplicate_test"
        await tracker.start_run(config={}, run_id=run_id)

        # Second start with same ID should raise
        with pytest.raises(ValueError, match="already active"):
            await tracker.start_run(config={}, run_id=run_id)

    @pytest.mark.asyncio
    async def test_end_run_updates_metadata(self, tracker):
        """Test that end_run updates metadata."""
        run_id = await tracker.start_run(config={"nodes": 8})

        await tracker.end_run(run_id, final_results={"accuracy": 0.95})

        run_info = await tracker.get_run_info(run_id)

        assert run_info["status"] == "completed"
        assert run_info["end_time"] is not None
        assert run_info["final_results"] == {"accuracy": 0.95}

    @pytest.mark.asyncio
    async def test_end_run_removes_from_active(self, tracker):
        """Test that end_run removes from active runs."""
        run_id = await tracker.start_run(config={})

        assert run_id in tracker._active_runs

        await tracker.end_run(run_id)

        assert run_id not in tracker._active_runs


class TestMetricLogging:
    """Test metric logging."""

    @pytest.mark.asyncio
    async def test_log_metrics_creates_event(self, tracker):
        """Test that log_metrics creates a MetricEvent."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_001",
            metrics={"loss": 0.5, "accuracy": 0.85}
        )

        # Query to verify
        events = await tracker.metrics_store.query_events(run_id)
        assert len(events) == 1
        assert events[0].entity_id == "node_001"
        assert events[0].metrics["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_log_metrics_all_entity_types(self, tracker):
        """Test logging metrics for all entity types."""
        run_id = await tracker.start_run(config={})

        for entity_type in [EntityType.NODE, EntityType.CLUSTER, EntityType.GLOBAL]:
            await tracker.log_metrics(
                run_id=run_id,
                round_num=1,
                entity_type=entity_type,
                entity_id=f"{entity_type.value}_1",
                metrics={"metric": 1.0}
            )

        events = await tracker.metrics_store.query_events(run_id)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_log_metrics_with_metadata(self, tracker):
        """Test logging metrics with optional metadata."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_001",
            metrics={"loss": 0.5},
            metadata={"version": "1.0", "extra": "data"}
        )

        events = await tracker.metrics_store.query_events(run_id)
        assert events[0].metadata == {"version": "1.0", "extra": "data"}


class TestCheckpointOperations:
    """Test checkpoint save/load coordination."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_delegates(self, tracker, sample_model):
        """Test that save_checkpoint delegates to repository."""
        run_id = await tracker.start_run(config={})

        metadata = await tracker.save_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        assert metadata.run_id == run_id
        assert metadata.cluster_id == "cluster_1"
        assert metadata.round_num == 1

    @pytest.mark.asyncio
    async def test_load_checkpoint_by_round_num(self, tracker, sample_model):
        """Test loading checkpoint by round number."""
        run_id = await tracker.start_run(config={})

        await tracker.save_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            round_num=5,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        loaded_state, metadata = await tracker.load_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            round_num=5
        )

        assert metadata.round_num == 5
        assert loaded_state is not None

    @pytest.mark.asyncio
    async def test_load_checkpoint_by_version(self, tracker, sample_model):
        """Test loading checkpoint by version."""
        run_id = await tracker.start_run(config={})

        # Save multiple checkpoints
        for i in range(1, 4):
            await tracker.save_checkpoint(
                run_id=run_id,
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": 0.5}
            )

        loaded_state, metadata = await tracker.load_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            version="latest"
        )

        assert metadata.round_num == 3


class TestRunInformation:
    """Test retrieving run information."""

    @pytest.mark.asyncio
    async def test_get_run_info_for_active_run(self, tracker):
        """Test getting info for active run."""
        run_id = await tracker.start_run(
            config={"test": True},
            description="Active run"
        )

        run_info = await tracker.get_run_info(run_id)

        assert run_info["run_id"] == run_id
        assert run_info["status"] == "running"
        assert run_info["config"] == {"test": True}

    @pytest.mark.asyncio
    async def test_get_run_info_for_completed_run(self, tracker):
        """Test getting info for completed run."""
        run_id = await tracker.start_run(config={})
        await tracker.end_run(run_id, final_results={"done": True})

        run_info = await tracker.get_run_info(run_id)

        assert run_info["status"] == "completed"
        assert run_info["final_results"] == {"done": True}

    @pytest.mark.asyncio
    async def test_get_run_info_loads_from_file(self, tracker):
        """Test that get_run_info loads from file after restart."""
        run_id = await tracker.start_run(config={"persisted": True})
        await tracker.end_run(run_id)

        # Clear active runs (simulating restart)
        tracker._active_runs.clear()

        # Should still load from file
        run_info = await tracker.get_run_info(run_id)
        assert run_info["config"] == {"persisted": True}

    @pytest.mark.asyncio
    async def test_get_run_info_nonexistent_raises(self, tracker):
        """Test that getting non-existent run raises error."""
        with pytest.raises(ValueError, match="not found"):
            await tracker.get_run_info("nonexistent_run")


class TestListRuns:
    """Test listing runs."""

    @pytest.mark.asyncio
    async def test_list_runs_returns_all(self, tracker):
        """Test that list_runs returns all runs."""
        # Create 3 runs
        for i in range(3):
            run_id = await tracker.start_run(
                config={"experiment": i},
                run_id=f"exp_{i}"
            )
            await tracker.end_run(run_id)

        runs = await tracker.list_runs()

        assert len(runs) == 3
        run_ids = [r["run_id"] for r in runs]
        assert "exp_0" in run_ids
        assert "exp_1" in run_ids
        assert "exp_2" in run_ids

    @pytest.mark.asyncio
    async def test_list_runs_sorted_by_time(self, tracker):
        """Test that runs are sorted by start_time (most recent first)."""
        # Create runs with slight delay
        import asyncio

        run_ids = []
        for i in range(3):
            run_id = await tracker.start_run(config={}, run_id=f"run_{i}")
            run_ids.append(run_id)
            await tracker.end_run(run_id)
            await asyncio.sleep(0.01)  # Small delay

        runs = await tracker.list_runs()

        # Should be in reverse order (most recent first)
        assert runs[0]["run_id"] == "run_2"
        assert runs[1]["run_id"] == "run_1"
        assert runs[2]["run_id"] == "run_0"

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, tracker):
        """Test list_runs when no runs exist."""
        runs = await tracker.list_runs()
        assert runs == []


class TestMetricsSummary:
    """Test metrics summary retrieval."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, tracker):
        """Test getting metrics summary for a run."""
        run_id = await tracker.start_run(config={})

        # Log some metrics
        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5, "accuracy": 0.8}
        )
        await tracker.log_metrics(
            run_id=run_id,
            round_num=2,
            entity_type=EntityType.CLUSTER,
            entity_id="cluster_1",
            metrics={"loss": 0.3}
        )

        summary = await tracker.get_metrics_summary(run_id)

        assert "loss" in summary["available_metrics"]
        assert "accuracy" in summary["available_metrics"]
        assert summary["entity_counts"]["node"] == 1
        assert summary["entity_counts"]["cluster"] == 1
        assert summary["round_range"] == (1, 2)
        assert summary["total_events"] == 2

    @pytest.mark.asyncio
    async def test_get_metrics_summary_with_filter(self, tracker):
        """Test getting metrics summary filtered by entity type."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"node_metric": 1.0}
        )
        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.CLUSTER,
            entity_id="cluster_1",
            metrics={"cluster_metric": 2.0}
        )

        summary = await tracker.get_metrics_summary(
            run_id,
            entity_type=EntityType.NODE
        )

        assert "node_metric" in summary["available_metrics"]
        assert "cluster_metric" not in summary["available_metrics"]


class TestCheckpointSummary:
    """Test checkpoint summary retrieval."""

    @pytest.mark.asyncio
    async def test_get_checkpoint_summary(self, tracker, sample_model):
        """Test getting checkpoint summary."""
        run_id = await tracker.start_run(config={})

        # Save multiple checkpoints
        for i in range(1, 4):
            await tracker.save_checkpoint(
                run_id=run_id,
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": 0.5 - i * 0.1}
            )

        summary = await tracker.get_checkpoint_summary(run_id, "cluster_1")

        assert summary["checkpoint_count"] == 3
        assert summary["latest_round"] == 3
        assert summary["best_checkpoint"]["round_num"] == 3  # Lowest loss
        assert summary["total_size_mb"] > 0

    @pytest.mark.asyncio
    async def test_get_checkpoint_summary_empty(self, tracker):
        """Test checkpoint summary when no checkpoints exist."""
        run_id = await tracker.start_run(config={})

        summary = await tracker.get_checkpoint_summary(run_id, "cluster_1")

        assert summary["checkpoint_count"] == 0
        assert summary["latest_round"] is None
        assert summary["best_checkpoint"] is None
        assert summary["total_size_mb"] == 0


class TestExport:
    """Test metrics export."""

    @pytest.mark.asyncio
    async def test_export_metrics_to_csv(self, tracker, tmp_path):
        """Test exporting metrics to CSV."""
        run_id = await tracker.start_run(config={})

        # Log some metrics
        for i in range(3):
            await tracker.log_metrics(
                run_id=run_id,
                round_num=i + 1,
                entity_type=EntityType.NODE,
                entity_id="node_1",
                metrics={"loss": 0.5 - i * 0.1}
            )

        output_file = tmp_path / "metrics.csv"
        await tracker.export_metrics(run_id, output_file, format="csv")

        assert output_file.exists()

        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert "metric_loss" in df.columns

    @pytest.mark.asyncio
    async def test_export_metrics_to_json(self, tracker, tmp_path):
        """Test exporting metrics to JSON."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5}
        )

        output_file = tmp_path / "metrics.json"
        await tracker.export_metrics(run_id, output_file, format="json")

        assert output_file.exists()

        # Verify JSON content
        import json
        with open(output_file, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["entity_id"] == "node_1"

    @pytest.mark.asyncio
    async def test_export_invalid_format_raises(self, tracker, tmp_path):
        """Test that invalid export format raises error."""
        run_id = await tracker.start_run(config={})

        output_file = tmp_path / "metrics.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            await tracker.export_metrics(run_id, output_file, format="xml")


class TestCleanup:
    """Test cleanup operations."""

    @pytest.mark.asyncio
    async def test_cleanup_run_deletes_metrics(self, tracker):
        """Test that cleanup_run can delete metrics only."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5}
        )

        await tracker.cleanup_run(run_id, delete_metrics=True, delete_models=False)

        # Metrics should be gone
        events = await tracker.metrics_store.query_events(run_id)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_cleanup_run_deletes_models(self, tracker, sample_model):
        """Test that cleanup_run can delete models only."""
        run_id = await tracker.start_run(config={})

        await tracker.save_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        await tracker.cleanup_run(run_id, delete_metrics=False, delete_models=True)

        # Models should be gone
        checkpoints = await tracker.model_repository.list_checkpoints(run_id, "cluster_1")
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_cleanup_run_deletes_both(self, tracker, sample_model):
        """Test that cleanup_run can delete both metrics and models."""
        run_id = await tracker.start_run(config={})

        await tracker.log_metrics(
            run_id=run_id,
            round_num=1,
            entity_type=EntityType.NODE,
            entity_id="node_1",
            metrics={"loss": 0.5}
        )
        await tracker.save_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        await tracker.cleanup_run(run_id, delete_metrics=True, delete_models=True)

        # Both should be gone
        events = await tracker.metrics_store.query_events(run_id)
        checkpoints = await tracker.model_repository.list_checkpoints(run_id, "cluster_1")
        assert len(events) == 0
        assert len(checkpoints) == 0

        # Metadata should also be removed
        with pytest.raises(ValueError, match="not found"):
            await tracker.get_run_info(run_id)


class TestIntegration:
    """Test integration of all tracker features."""

    @pytest.mark.asyncio
    async def test_complete_experiment_workflow(self, tracker, sample_model):
        """Test complete experiment workflow from start to finish."""
        # Start experiment
        run_id = await tracker.start_run(
            config={"nodes": 8, "rounds": 5},
            description="Integration test"
        )

        # Simulate 5 rounds of training
        for round_num in range(1, 6):
            # Log node metrics
            for node_id in ["node_1", "node_2"]:
                await tracker.log_metrics(
                    run_id=run_id,
                    round_num=round_num,
                    entity_type=EntityType.NODE,
                    entity_id=node_id,
                    metrics={"loss": 0.5 - round_num * 0.05}
                )

            # Log cluster metrics
            await tracker.log_metrics(
                run_id=run_id,
                round_num=round_num,
                entity_type=EntityType.CLUSTER,
                entity_id="cluster_1",
                metrics={"aggregated_loss": 0.4 - round_num * 0.04}
            )

            # Save checkpoint
            await tracker.save_checkpoint(
                run_id=run_id,
                cluster_id="cluster_1",
                round_num=round_num,
                model_state=sample_model,
                metrics={"loss": 0.4 - round_num * 0.04}
            )

        # End experiment
        await tracker.end_run(run_id, final_results={"final_loss": 0.2})

        # Verify everything was tracked
        run_info = await tracker.get_run_info(run_id)
        assert run_info["status"] == "completed"
        assert run_info["final_results"]["final_loss"] == 0.2

        # Verify metrics
        metrics_summary = await tracker.get_metrics_summary(run_id)
        assert metrics_summary["total_events"] == 15  # 2 nodes × 5 + 5 cluster
        assert metrics_summary["round_range"] == (1, 5)

        # Verify checkpoints
        checkpoint_summary = await tracker.get_checkpoint_summary(run_id, "cluster_1")
        assert checkpoint_summary["checkpoint_count"] == 5
        assert checkpoint_summary["latest_round"] == 5
        assert checkpoint_summary["best_checkpoint"]["round_num"] == 5  # Lowest loss

        # Verify can load checkpoint
        loaded_state, metadata = await tracker.load_checkpoint(
            run_id=run_id,
            cluster_id="cluster_1",
            version="best"
        )
        assert metadata.round_num == 5
