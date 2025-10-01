"""
Local filesystem model checkpoint repository.

This module provides local filesystem storage for model checkpoints:
- Save/load PyTorch model state dicts
- SHA256 checksum verification for data integrity
- Version tracking with latest/best symlinks
- Metadata JSON files per checkpoint
- Automatic cleanup of old checkpoints

Storage structure:
    storage/models/{run_id}/
    ├── {cluster_id}/
    │   ├── round_0001.pt
    │   ├── round_0001_metadata.json
    │   ├── round_0002.pt
    │   ├── round_0002_metadata.json
    │   ├── ...
    │   ├── latest.pt -> round_0050.pt
    │   ├── best.pt -> round_0023.pt
    │   └── checksums.json
"""

import asyncio
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from server.storage.base import ModelCheckpointMetadata, ModelRepository


class LocalModelRepository(ModelRepository):
    """
    Local filesystem repository for model checkpoints.

    Features:
    - PyTorch state_dict persistence
    - SHA256 checksum verification
    - Version tracking (latest/best)
    - Metadata per checkpoint
    - Automatic cleanup options
    """

    def __init__(
        self,
        base_path: str | Path,
        keep_last_n: Optional[int] = None,
        keep_best: bool = True,
        compute_checksums: bool = True
    ):
        """
        Initialize local model repository.

        Args:
            base_path: Base directory for model storage
            keep_last_n: Keep only last N checkpoints (None = keep all)
            keep_best: Always keep the best checkpoint
            compute_checksums: If True, compute SHA256 checksums
        """
        self.base_path = Path(base_path)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.compute_checksums = compute_checksums

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized LocalModelRepository at {self.base_path} "
            f"(keep_last_n={keep_last_n}, keep_best={keep_best})"
        )

    def _get_cluster_path(self, run_id: str, cluster_id: str) -> Path:
        """Get path for a specific cluster."""
        path = self.base_path / run_id / cluster_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_checkpoint_path(self, run_id: str, cluster_id: str, round_num: int) -> Path:
        """Get path to checkpoint file."""
        cluster_path = self._get_cluster_path(run_id, cluster_id)
        return cluster_path / f"round_{round_num:04d}.pt"

    def _get_metadata_path(self, run_id: str, cluster_id: str, round_num: int) -> Path:
        """Get path to metadata file."""
        cluster_path = self._get_cluster_path(run_id, cluster_id)
        return cluster_path / f"round_{round_num:04d}_metadata.json"

    def _get_checksums_path(self, run_id: str, cluster_id: str) -> Path:
        """Get path to checksums file."""
        return self._get_cluster_path(run_id, cluster_id) / "checksums.json"

    def _get_latest_symlink(self, run_id: str, cluster_id: str) -> Path:
        """Get path to 'latest' symlink."""
        return self._get_cluster_path(run_id, cluster_id) / "latest.pt"

    def _get_best_symlink(self, run_id: str, cluster_id: str) -> Path:
        """Get path to 'best' symlink."""
        return self._get_cluster_path(run_id, cluster_id) / "best.pt"

    async def save_model(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int,
        model_state: Dict[str, Any],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelCheckpointMetadata:
        """Save a model checkpoint."""
        checkpoint_path = self._get_checkpoint_path(run_id, cluster_id, round_num)
        metadata_path = self._get_metadata_path(run_id, cluster_id, round_num)

        # Save model state
        def _save_model():
            torch.save(model_state, checkpoint_path)
            return checkpoint_path.stat().st_size

        loop = asyncio.get_event_loop()
        model_size = await loop.run_in_executor(None, _save_model)

        # Compute checksum if enabled
        checksum = ""
        if self.compute_checksums:
            checksum = await self._compute_checksum(checkpoint_path)
            await self._update_checksums(run_id, cluster_id, round_num, checksum)

        # Create metadata
        checkpoint_metadata = ModelCheckpointMetadata(
            run_id=run_id,
            cluster_id=cluster_id,
            round_num=round_num,
            timestamp=datetime.now(),
            checksum=checksum,
            metrics=metrics,
            model_size_bytes=model_size,
            metadata=metadata or {}
        )

        # Save metadata
        def _save_metadata():
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_metadata.to_dict(), f, indent=2)

        await loop.run_in_executor(None, _save_metadata)

        # Update latest symlink
        await self._update_symlink(
            self._get_latest_symlink(run_id, cluster_id),
            checkpoint_path
        )

        # Update best symlink if this is the best checkpoint
        await self._maybe_update_best(run_id, cluster_id, checkpoint_metadata)

        # Cleanup old checkpoints if needed
        if self.keep_last_n:
            await self._cleanup_old_checkpoints(run_id, cluster_id)

        logger.info(
            f"Saved checkpoint for {cluster_id} round {round_num}: "
            f"{model_size / 1024 / 1024:.2f} MB"
        )

        return checkpoint_metadata

    async def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        def _compute():
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute)

    async def _update_checksums(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int,
        checksum: str
    ) -> None:
        """Update checksums file."""
        checksums_path = self._get_checksums_path(run_id, cluster_id)

        def _update():
            # Load existing checksums
            checksums = {}
            if checksums_path.exists():
                with open(checksums_path, "r", encoding="utf-8") as f:
                    checksums = json.load(f)

            # Add new checksum
            checksums[f"round_{round_num:04d}"] = checksum

            # Save updated checksums
            with open(checksums_path, "w", encoding="utf-8") as f:
                json.dump(checksums, f, indent=2)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)

    async def _update_symlink(self, symlink_path: Path, target_path: Path) -> None:
        """Update a symlink to point to target."""
        def _update():
            # Remove existing symlink if it exists
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            # Create new symlink (relative path)
            relative_target = target_path.name
            symlink_path.symlink_to(relative_target)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)

    async def _maybe_update_best(
        self,
        run_id: str,
        cluster_id: str,
        new_metadata: ModelCheckpointMetadata
    ) -> None:
        """Update best symlink if this checkpoint is better."""
        if not self.keep_best:
            return

        best_symlink = self._get_best_symlink(run_id, cluster_id)

        # Get current best metadata
        current_best = None
        if best_symlink.exists():
            # Read current best round number from symlink
            target_name = best_symlink.resolve().name
            round_num = int(target_name.split("_")[1].split(".")[0])
            current_best = await self._load_metadata(run_id, cluster_id, round_num)

        # Determine if new checkpoint is better (lower loss by default)
        is_better = False
        if current_best is None:
            is_better = True
        else:
            new_loss = new_metadata.metrics.get("loss", float('inf'))
            current_loss = current_best.metrics.get("loss", float('inf'))
            is_better = new_loss < current_loss

        if is_better:
            checkpoint_path = self._get_checkpoint_path(
                run_id, cluster_id, new_metadata.round_num
            )
            await self._update_symlink(best_symlink, checkpoint_path)
            logger.info(
                f"Updated best checkpoint for {cluster_id} to round {new_metadata.round_num}"
            )

    async def _cleanup_old_checkpoints(self, run_id: str, cluster_id: str) -> None:
        """Remove old checkpoints, keeping only last N."""
        if not self.keep_last_n:
            return

        # List all checkpoints
        checkpoints = await self.list_checkpoints(run_id, cluster_id)

        # Keep last N
        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by round number (already sorted from list_checkpoints)
        to_delete = checkpoints[:-self.keep_last_n]

        # Get best checkpoint round number if we're keeping best
        best_round = None
        if self.keep_best:
            best_symlink = self._get_best_symlink(run_id, cluster_id)
            if best_symlink.exists():
                target_name = best_symlink.resolve().name
                best_round = int(target_name.split("_")[1].split(".")[0])

        # Delete old checkpoints (except best)
        for checkpoint in to_delete:
            if self.keep_best and checkpoint.round_num == best_round:
                continue  # Don't delete best

            await self.delete_checkpoint(run_id, cluster_id, checkpoint.round_num)

    async def load_model(
        self,
        run_id: str,
        cluster_id: str,
        round_num: Optional[int] = None,
        version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ModelCheckpointMetadata]:
        """Load a model checkpoint."""
        # Determine which checkpoint to load
        if round_num is not None:
            checkpoint_path = self._get_checkpoint_path(run_id, cluster_id, round_num)
        elif version == "latest":
            checkpoint_path = self._get_latest_symlink(run_id, cluster_id)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"No latest checkpoint for {cluster_id}")
        elif version == "best":
            checkpoint_path = self._get_best_symlink(run_id, cluster_id)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"No best checkpoint for {cluster_id}")
        else:
            raise ValueError("Must specify either round_num or version ('latest'/'best')")

        # Resolve symlink if needed
        checkpoint_path = checkpoint_path.resolve()

        # Extract round number from filename
        round_num = int(checkpoint_path.name.split("_")[1].split(".")[0])

        # Load model state
        def _load_model():
            return torch.load(checkpoint_path)

        loop = asyncio.get_event_loop()
        model_state = await loop.run_in_executor(None, _load_model)

        # Load metadata
        metadata = await self._load_metadata(run_id, cluster_id, round_num)

        # Verify checksum if enabled
        if self.compute_checksums and metadata.checksum:
            actual_checksum = await self._compute_checksum(checkpoint_path)
            if actual_checksum != metadata.checksum:
                raise ValueError(
                    f"Checksum mismatch for {cluster_id} round {round_num}: "
                    f"expected {metadata.checksum}, got {actual_checksum}"
                )

        logger.info(f"Loaded checkpoint for {cluster_id} round {round_num}")
        return model_state, metadata

    async def _load_metadata(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int
    ) -> ModelCheckpointMetadata:
        """Load metadata for a checkpoint."""
        metadata_path = self._get_metadata_path(run_id, cluster_id, round_num)

        def _load():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _load)

        return ModelCheckpointMetadata.from_dict(data)

    async def list_checkpoints(
        self,
        run_id: str,
        cluster_id: str
    ) -> List[ModelCheckpointMetadata]:
        """List all checkpoints for a cluster."""
        cluster_path = self._get_cluster_path(run_id, cluster_id)

        # Find all checkpoint files
        def _list():
            checkpoints = []
            for file_path in sorted(cluster_path.glob("round_*.pt")):
                # Skip symlinks
                if file_path.is_symlink():
                    continue

                round_num = int(file_path.name.split("_")[1].split(".")[0])
                metadata_path = self._get_metadata_path(run_id, cluster_id, round_num)

                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = ModelCheckpointMetadata.from_dict(json.load(f))
                        checkpoints.append(metadata)

            # Sort by round number
            checkpoints.sort(key=lambda x: x.round_num)
            return checkpoints

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def get_best_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        metric_name: str = "loss",
        minimize: bool = True
    ) -> Optional[ModelCheckpointMetadata]:
        """Get the best checkpoint based on a metric."""
        checkpoints = await self.list_checkpoints(run_id, cluster_id)

        if not checkpoints:
            return None

        # Filter checkpoints that have the metric
        valid_checkpoints = [
            cp for cp in checkpoints
            if metric_name in cp.metrics
        ]

        if not valid_checkpoints:
            return None

        # Find best
        if minimize:
            best = min(valid_checkpoints, key=lambda x: x.metrics[metric_name])
        else:
            best = max(valid_checkpoints, key=lambda x: x.metrics[metric_name])

        return best

    async def delete_checkpoint(
        self,
        run_id: str,
        cluster_id: str,
        round_num: int
    ) -> None:
        """Delete a specific checkpoint."""
        checkpoint_path = self._get_checkpoint_path(run_id, cluster_id, round_num)
        metadata_path = self._get_metadata_path(run_id, cluster_id, round_num)

        def _delete():
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)

        logger.info(f"Deleted checkpoint for {cluster_id} round {round_num}")

    async def delete_run(self, run_id: str) -> None:
        """Delete all checkpoints for a run."""
        run_path = self.base_path / run_id

        if run_path.exists():
            def _delete():
                shutil.rmtree(run_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _delete)

            logger.info(f"Deleted all checkpoints for run {run_id}")
