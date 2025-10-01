"""
Unit tests for LocalModelRepository.

Tests model checkpoint storage with PyTorch state dicts.
"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import torch

from server.storage.base import ModelCheckpointMetadata
from server.storage.local_model_repository import LocalModelRepository


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "test_models"
    storage_dir.mkdir()
    yield storage_dir
    # Cleanup
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def model_repo(temp_storage_dir):
    """Create LocalModelRepository instance."""
    return LocalModelRepository(
        base_path=temp_storage_dir,
        keep_last_n=None,
        keep_best=True,
        compute_checksums=True
    )


@pytest.fixture
def sample_model():
    """Create a sample PyTorch model state dict."""
    return {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 10),
        'layer2.bias': torch.randn(5)
    }


class TestLocalModelRepositoryInitialization:
    """Test LocalModelRepository initialization."""

    def test_create_with_default_settings(self, temp_storage_dir):
        """Test creating repository with default settings."""
        repo = LocalModelRepository(base_path=temp_storage_dir)

        assert repo.base_path == temp_storage_dir
        assert repo.keep_last_n is None
        assert repo.keep_best is True
        assert repo.compute_checksums is True

    def test_create_with_keep_last_n(self, temp_storage_dir):
        """Test creating repository with keep_last_n option."""
        repo = LocalModelRepository(base_path=temp_storage_dir, keep_last_n=5)

        assert repo.keep_last_n == 5

    def test_create_with_keep_best_disabled(self, temp_storage_dir):
        """Test creating repository with keep_best disabled."""
        repo = LocalModelRepository(base_path=temp_storage_dir, keep_best=False)

        assert repo.keep_best is False

    def test_directory_creation(self, tmp_path):
        """Test that base directory is created."""
        storage_dir = tmp_path / "new_models"
        assert not storage_dir.exists()

        LocalModelRepository(base_path=storage_dir)

        assert storage_dir.exists()
        assert storage_dir.is_dir()


class TestSaveCheckpoint:
    """Test saving model checkpoints."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_successfully(self, model_repo, sample_model):
        """Test saving a checkpoint successfully."""
        metadata = await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        assert metadata.run_id == "test_run"
        assert metadata.cluster_id == "cluster_1"
        assert metadata.round_num == 1
        assert metadata.metrics == {"loss": 0.5}
        assert metadata.model_size_bytes > 0

    @pytest.mark.asyncio
    async def test_checkpoint_file_created(self, model_repo, sample_model):
        """Test that checkpoint file is created."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        checkpoint_path = model_repo._get_checkpoint_path(
            "test_run", "cluster_1", 1
        )
        assert checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_metadata_file_created(self, model_repo, sample_model):
        """Test that metadata file is created."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        metadata_path = model_repo._get_metadata_path(
            "test_run", "cluster_1", 1
        )
        assert metadata_path.exists()

    @pytest.mark.asyncio
    async def test_checksum_computed_when_enabled(self, model_repo, sample_model):
        """Test that checksum is computed when enabled."""
        metadata = await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        assert metadata.checksum != ""
        assert len(metadata.checksum) == 64  # SHA256 hex digest length

    @pytest.mark.asyncio
    async def test_checksum_skipped_when_disabled(self, temp_storage_dir, sample_model):
        """Test that checksum is skipped when disabled."""
        repo = LocalModelRepository(
            base_path=temp_storage_dir,
            compute_checksums=False
        )

        metadata = await repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        assert metadata.checksum == ""

    @pytest.mark.asyncio
    async def test_file_size_recorded(self, model_repo, sample_model):
        """Test that file size is recorded correctly."""
        metadata = await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Verify size matches actual file
        checkpoint_path = model_repo._get_checkpoint_path(
            "test_run", "cluster_1", 1
        )
        actual_size = checkpoint_path.stat().st_size
        assert metadata.model_size_bytes == actual_size

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, model_repo, sample_model):
        """Test that timestamp is recorded."""
        before = datetime.now()
        metadata = await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )
        after = datetime.now()

        assert before <= metadata.timestamp <= after


class TestSymlinkManagement:
    """Test symlink creation and updates."""

    @pytest.mark.asyncio
    async def test_latest_symlink_created(self, model_repo, sample_model):
        """Test that latest.pt symlink is created."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        latest_symlink = model_repo._get_latest_symlink("test_run", "cluster_1")
        assert latest_symlink.exists()
        assert latest_symlink.is_symlink()

    @pytest.mark.asyncio
    async def test_latest_symlink_updated(self, model_repo, sample_model):
        """Test that latest.pt is updated on new save."""
        # Save round 1
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Save round 2
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=2,
            model_state=sample_model,
            metrics={"loss": 0.4}
        )

        # latest should point to round 2
        latest_symlink = model_repo._get_latest_symlink("test_run", "cluster_1")
        target = latest_symlink.resolve()
        assert "round_0002.pt" in target.name

    @pytest.mark.asyncio
    async def test_best_symlink_created_for_first(self, model_repo, sample_model):
        """Test that best.pt is created for first checkpoint."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        best_symlink = model_repo._get_best_symlink("test_run", "cluster_1")
        assert best_symlink.exists()
        assert best_symlink.is_symlink()

    @pytest.mark.asyncio
    async def test_best_symlink_updated_when_better(self, model_repo, sample_model):
        """Test that best.pt is updated when better checkpoint saved."""
        # Save round 1 with loss 0.5
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Save round 2 with lower loss (better)
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=2,
            model_state=sample_model,
            metrics={"loss": 0.3}
        )

        # best should point to round 2
        best_symlink = model_repo._get_best_symlink("test_run", "cluster_1")
        target = best_symlink.resolve()
        assert "round_0002.pt" in target.name

    @pytest.mark.asyncio
    async def test_best_symlink_unchanged_when_worse(self, model_repo, sample_model):
        """Test that best.pt is unchanged when worse checkpoint saved."""
        # Save round 1 with loss 0.3
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.3}
        )

        # Save round 2 with higher loss (worse)
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=2,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # best should still point to round 1
        best_symlink = model_repo._get_best_symlink("test_run", "cluster_1")
        target = best_symlink.resolve()
        assert "round_0001.pt" in target.name

    @pytest.mark.asyncio
    async def test_relative_symlinks(self, model_repo, sample_model):
        """Test that symlinks are relative, not absolute."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        latest_symlink = model_repo._get_latest_symlink("test_run", "cluster_1")
        # Read raw symlink (not resolved)
        import os
        target = os.readlink(latest_symlink)

        # Should be relative (just filename)
        assert "/" not in target
        assert target == "round_0001.pt"


class TestBestCheckpointTracking:
    """Test best checkpoint tracking."""

    @pytest.mark.asyncio
    async def test_track_best_by_loss_minimize(self, model_repo, sample_model):
        """Test tracking best by loss (minimize)."""
        for i, loss in enumerate([0.5, 0.3, 0.6, 0.2, 0.4], start=1):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": loss}
            )

        best = await model_repo.get_best_checkpoint(
            "test_run", "cluster_1", metric_name="loss", minimize=True
        )

        assert best.round_num == 4  # loss=0.2
        assert best.metrics["loss"] == 0.2

    @pytest.mark.asyncio
    async def test_track_best_by_custom_metric_minimize(self, model_repo, sample_model):
        """Test tracking best by custom metric (minimize)."""
        for i, error in enumerate([10, 5, 15, 3, 8], start=1):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"error": error}
            )

        best = await model_repo.get_best_checkpoint(
            "test_run", "cluster_1", metric_name="error", minimize=True
        )

        assert best.round_num == 4  # error=3

    @pytest.mark.asyncio
    async def test_track_best_by_custom_metric_maximize(self, model_repo, sample_model):
        """Test tracking best by custom metric (maximize)."""
        for i, accuracy in enumerate([0.5, 0.8, 0.6, 0.9, 0.7], start=1):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"accuracy": accuracy}
            )

        best = await model_repo.get_best_checkpoint(
            "test_run", "cluster_1", metric_name="accuracy", minimize=False
        )

        assert best.round_num == 4  # accuracy=0.9

    @pytest.mark.asyncio
    async def test_missing_metric_handling(self, model_repo, sample_model):
        """Test handling when metric is missing in some checkpoints."""
        # First checkpoint has metric
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5, "accuracy": 0.8}
        )

        # Second checkpoint missing accuracy
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=2,
            model_state=sample_model,
            metrics={"loss": 0.3}
        )

        best = await model_repo.get_best_checkpoint(
            "test_run", "cluster_1", metric_name="accuracy", minimize=False
        )

        # Should return the only checkpoint with accuracy
        assert best.round_num == 1


class TestLoadCheckpoint:
    """Test loading checkpoints."""

    @pytest.mark.asyncio
    async def test_load_by_round_num(self, model_repo, sample_model):
        """Test loading checkpoint by round number."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=5,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        loaded_state, metadata = await model_repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=5
        )

        assert metadata.round_num == 5
        assert set(loaded_state.keys()) == set(sample_model.keys())

    @pytest.mark.asyncio
    async def test_load_by_version_latest(self, model_repo, sample_model):
        """Test loading checkpoint by version='latest'."""
        # Save multiple checkpoints
        for i in range(1, 4):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": 0.5}
            )

        loaded_state, metadata = await model_repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            version="latest"
        )

        assert metadata.round_num == 3  # Latest

    @pytest.mark.asyncio
    async def test_load_by_version_best(self, model_repo, sample_model):
        """Test loading checkpoint by version='best'."""
        # Save checkpoints with different losses
        for i, loss in enumerate([0.5, 0.3, 0.6], start=1):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": loss}
            )

        loaded_state, metadata = await model_repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            version="best"
        )

        assert metadata.round_num == 2  # Best (loss=0.3)

    @pytest.mark.asyncio
    async def test_model_state_matches_saved(self, model_repo, sample_model):
        """Test that loaded model state matches saved state."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        loaded_state, _ = await model_repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1
        )

        # Compare tensors
        for key in sample_model:
            assert torch.allclose(loaded_state[key], sample_model[key])


class TestChecksumVerification:
    """Test checksum verification."""

    @pytest.mark.asyncio
    async def test_verify_checksum_on_load(self, model_repo, sample_model):
        """Test that checksum is verified on load."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Should load without error (checksum valid)
        loaded_state, metadata = await model_repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1
        )

        assert loaded_state is not None

    @pytest.mark.asyncio
    async def test_corrupted_file_detected(self, model_repo, sample_model):
        """Test that corrupted file is detected by checksum."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Corrupt the file
        checkpoint_path = model_repo._get_checkpoint_path(
            "test_run", "cluster_1", 1
        )
        with open(checkpoint_path, "ab") as f:
            f.write(b"corrupted_data")

        # Should raise error on load
        with pytest.raises(ValueError, match="Checksum mismatch"):
            await model_repo.load_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=1
            )

    @pytest.mark.asyncio
    async def test_skip_verification_when_disabled(self, temp_storage_dir, sample_model):
        """Test that verification is skipped when checksums disabled."""
        repo = LocalModelRepository(
            base_path=temp_storage_dir,
            compute_checksums=False
        )

        await repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        # Verify no checksum in metadata
        metadata_path = repo._get_metadata_path("test_run", "cluster_1", 1)
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["checksum"] == ""

        # Load should work fine (no checksum verification happens)
        loaded_state, loaded_metadata = await repo.load_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1
        )

        assert loaded_metadata.checksum == ""


class TestCleanup:
    """Test checkpoint cleanup."""

    @pytest.mark.asyncio
    async def test_keep_last_n_enforced(self, temp_storage_dir, sample_model):
        """Test that keep_last_n is enforced correctly."""
        repo = LocalModelRepository(
            base_path=temp_storage_dir,
            keep_last_n=3,
            keep_best=False
        )

        # Save 5 checkpoints
        for i in range(1, 6):
            await repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": 0.5}
            )

        checkpoints = await repo.list_checkpoints("test_run", "cluster_1")

        # Should only keep last 3
        assert len(checkpoints) == 3
        assert checkpoints[0].round_num == 3
        assert checkpoints[-1].round_num == 5

    @pytest.mark.asyncio
    async def test_best_checkpoint_preserved(self, temp_storage_dir, sample_model):
        """Test that best checkpoint is preserved when keep_best=True."""
        repo = LocalModelRepository(
            base_path=temp_storage_dir,
            keep_last_n=2,
            keep_best=True
        )

        # Save checkpoints: round 1 is best (loss=0.1)
        for i, loss in enumerate([0.1, 0.5, 0.6, 0.7], start=1):
            await repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": loss}
            )

        checkpoints = await repo.list_checkpoints("test_run", "cluster_1")
        rounds = [cp.round_num for cp in checkpoints]

        # Should keep: round 1 (best), round 3, round 4 (last 2)
        assert 1 in rounds  # Best preserved
        assert 3 in rounds  # Last 2
        assert 4 in rounds


class TestListCheckpoints:
    """Test listing checkpoints."""

    @pytest.mark.asyncio
    async def test_returns_all_checkpoints_sorted(self, model_repo, sample_model):
        """Test that list returns all checkpoints sorted."""
        # Save out of order
        for round_num in [3, 1, 4, 2]:
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=round_num,
                model_state=sample_model,
                metrics={"loss": 0.5}
            )

        checkpoints = await model_repo.list_checkpoints("test_run", "cluster_1")

        assert len(checkpoints) == 4
        assert [cp.round_num for cp in checkpoints] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_excludes_symlinks(self, model_repo, sample_model):
        """Test that symlinks are excluded from list."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        checkpoints = await model_repo.list_checkpoints("test_run", "cluster_1")

        # Should only return 1 (not counting latest/best symlinks)
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_empty_list_for_no_checkpoints(self, model_repo):
        """Test empty list when no checkpoints exist."""
        checkpoints = await model_repo.list_checkpoints("test_run", "cluster_1")
        assert checkpoints == []


class TestDeleteOperations:
    """Test delete operations."""

    @pytest.mark.asyncio
    async def test_delete_checkpoint_removes_files(self, model_repo, sample_model):
        """Test that delete_checkpoint removes both model and metadata."""
        await model_repo.save_model(
            run_id="test_run",
            cluster_id="cluster_1",
            round_num=1,
            model_state=sample_model,
            metrics={"loss": 0.5}
        )

        checkpoint_path = model_repo._get_checkpoint_path("test_run", "cluster_1", 1)
        metadata_path = model_repo._get_metadata_path("test_run", "cluster_1", 1)

        assert checkpoint_path.exists()
        assert metadata_path.exists()

        await model_repo.delete_checkpoint("test_run", "cluster_1", 1)

        assert not checkpoint_path.exists()
        assert not metadata_path.exists()

    @pytest.mark.asyncio
    async def test_delete_run_removes_all(self, model_repo, sample_model):
        """Test that delete_run removes all cluster data."""
        # Save multiple checkpoints
        for i in range(1, 4):
            await model_repo.save_model(
                run_id="test_run",
                cluster_id="cluster_1",
                round_num=i,
                model_state=sample_model,
                metrics={"loss": 0.5}
            )

        run_path = model_repo.base_path / "test_run"
        assert run_path.exists()

        await model_repo.delete_run("test_run")

        assert not run_path.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_gracefully(self, model_repo):
        """Test deleting non-existent checkpoint doesn't raise error."""
        # Should not raise
        await model_repo.delete_checkpoint("test_run", "cluster_1", 999)
