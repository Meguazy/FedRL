"""
Supervised trainer for bootstrapping AlphaZero from game databases.

This trainer implements supervised learning from high-quality human games
to provide an initial policy/value network before transitioning to self-play.

Key Features:
- Loads games from PGN databases (Lichess, Chess.com, etc.)
- Filters by playstyle (tactical/positional) for specialized clusters
- Extracts training samples (position, move, outcome)
- Trains AlphaZero network on supervised data
- Uses board encoder (119 planes) and move encoder (4672 actions)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import time
import numpy as np
from loguru import logger

from .trainer_interface import TrainerInterface, TrainingConfig, TrainingResult, TrainingError
from .models.alphazero_net import AlphaZeroNet
from data.sample_extractor import SampleExtractor, ExtractionConfig, TrainingSample
from data.board_encoder import BoardEncoder
from data.move_encoder import MoveEncoder
from common.model_serialization import PyTorchSerializer


class ChessDataset(Dataset):
    """PyTorch dataset for chess training samples."""

    def __init__(self, samples: List[TrainingSample],
                 board_encoder: BoardEncoder,
                 move_encoder: MoveEncoder):
        """
        Initialize dataset.

        Args:
            samples: List of training samples
            board_encoder: Encoder for board positions
            move_encoder: Encoder for moves
        """
        self.samples = samples
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training sample.

        Returns:
            Tuple of (board_tensor, policy_target, value_target)
        """
        sample = self.samples[idx]

        # Encode board with history (119, 8, 8)
        # History is list of up to 7 previous board positions
        board_tensor = self.board_encoder.encode(sample.board, history=sample.history)

        # Encode move to action index
        move_index = self.move_encoder.encode(sample.move_played, sample.board)

        # Create one-hot policy target
        policy_target = np.zeros(4672, dtype=np.float32)
        policy_target[move_index] = 1.0

        # Value target is game outcome from player's perspective
        value_target = np.array([sample.game_outcome], dtype=np.float32)

        return (
            torch.from_numpy(board_tensor),
            torch.from_numpy(policy_target),
            torch.from_numpy(value_target)
        )


class SupervisedTrainer(TrainerInterface):
    """
    Supervised trainer for AlphaZero using game databases.

    This trainer bootstraps the AlphaZero network by training on
    high-quality human games before transitioning to self-play.

    Training Pipeline:
    1. Extract samples from PGN database (filtered by playstyle, rating)
    2. Encode positions to 119-plane tensors
    3. Encode moves to action indices (0-4671)
    4. Train policy head (cross-entropy) and value head (MSE)
    5. Return updated model weights

    Example:
        >>> config = TrainingConfig(
        ...     games_per_round=100,
        ...     batch_size=256,
        ...     learning_rate=0.001,
        ...     playstyle="tactical"
        ... )
        >>> trainer = SupervisedTrainer("node_1", "tactical_cluster", config)
        >>> result = await trainer.train(initial_model_state)
    """

    def __init__(self, node_id: str, cluster_id: str, config: TrainingConfig,
                 pgn_database_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize supervised trainer.

        Args:
            node_id: Unique node identifier
            cluster_id: Cluster identifier
            config: Training configuration
            pgn_database_path: Path to PGN database (optional, can be set later)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(node_id, cluster_id, config)

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        # Model serializer - use base64 encoding for JSON compatibility
        self.serializer = PyTorchSerializer(compression=True, encoding='base64')

        # PGN database path
        self.pgn_database_path = pgn_database_path

        # Model (will be created when training starts)
        self.model: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Sample extractor config
        self.extraction_config = ExtractionConfig(
            skip_opening_moves=10,
            skip_endgame_moves=6,
            sample_rate=1.0,
            max_positions_per_game=None,
            shuffle_games=True
        )

        # Calculate unique base offset for this node to ensure data diversity within cluster
        # Extract numeric part from node_id (e.g., "agg_001" -> 1, "pos_003" -> 3)
        node_number = int(''.join(filter(str.isdigit, node_id)) or '0')
        self.node_base_offset = node_number * config.games_per_round
        
        # Current round offset (incremented each round for diversity across rounds)
        self.current_round = 0
        
        log = logger.bind(context=f"SupervisedTrainer.{node_id}")
        log.info(f"Initialized supervised trainer on device: {self.device}")
        log.info(f"Node base offset: {self.node_base_offset} (ensures intra-cluster diversity)")

    def set_pgn_database(self, path: str):
        """Set the PGN database path."""
        self.pgn_database_path = path
        log = logger.bind(context=f"SupervisedTrainer.{self.node_id}")
        log.info(f"Set PGN database path: {path}")

    async def train(self, initial_model_state: Dict[str, Any]) -> TrainingResult:
        """
        Perform supervised training from game database.

        Args:
            initial_model_state: Starting model weights (serialized state_dict)

        Returns:
            TrainingResult with updated model and metrics

        Raises:
            TrainingError: If training fails
        """
        log = logger.bind(context=f"SupervisedTrainer.{self.node_id}")
        log.info("Starting supervised training")

        if not self.pgn_database_path:
            raise TrainingError("PGN database path not set")

        if not Path(self.pgn_database_path).exists():
            raise TrainingError(f"PGN database not found: {self.pgn_database_path}")

        start_time = time.time()

        try:
            # 1. Load model
            self._initialize_model(initial_model_state)
            log.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

            # 2. Extract training samples with unique offset for this node and round
            # Offset = node_base_offset + (current_round * games_per_round * nodes_per_cluster)
            # This ensures each node in a cluster gets different games, and different games each round
            round_offset = self.current_round * self.config.games_per_round * 10  # Assume max 10 nodes per cluster
            total_offset = self.node_base_offset + round_offset
            
            log.info(f"[{self.node_id}] Starting data extraction from database...")
            log.info(f"[{self.node_id}] Offset calculation: base={self.node_base_offset}, "
                    f"round={self.current_round}, total={total_offset}")
            log.info(f"[{self.node_id}] Will extract {self.config.games_per_round} games from offset {total_offset}")
            
            extraction_start = time.time()
            samples = await self._extract_samples(offset=total_offset)
            extraction_time = time.time() - extraction_start
            
            log.success(f"[{self.node_id}] Data extraction complete: {len(samples)} samples in {extraction_time:.1f}s")

            if len(samples) == 0:
                raise TrainingError("No training samples extracted")

            # 3. Create dataset and dataloader
            dataset = ChessDataset(samples, self.board_encoder, self.move_encoder)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=True if self.device.type == "cuda" else False
            )
            log.info(f"Created dataloader with {len(dataloader)} batches")

            # 4. Train
            log.info(f"[{self.node_id}] Starting model training with {len(dataloader)} batches...")
            training_start = time.time()
            metrics = await self._train_epoch(dataloader)
            training_time = time.time() - training_start
            log.success(f"[{self.node_id}] Training complete: loss={metrics['total_loss']:.4f}, time={training_time:.1f}s")

            # 5. Get updated model state (keep as state_dict for aggregation)
            # The state_dict will be serialized at the protocol layer when needed
            updated_model_state = self.model.state_dict()

            # 6. Update statistics
            training_time = time.time() - start_time
            self.total_games_played += self.config.games_per_round
            self.total_training_time += training_time
            self.current_round += 1  # Increment for next round's diversity

            # 7. Create result
            result = TrainingResult(
                model_state=updated_model_state,
                samples=len(samples),
                loss=metrics['total_loss'],
                games_played=self.config.games_per_round,
                training_time=training_time,
                metrics=metrics,
                game_data=None,
                success=True,
                error_message=None
            )

            self.training_history.append(result)
            return result

        except Exception as e:
            log.error(f"Training failed: {e}")
            raise TrainingError(f"Supervised training failed: {e}") from e

    async def evaluate(self, model_state: Dict[str, Any],
                      num_games: int = 10) -> Dict[str, Any]:
        """
        Evaluate model on validation set.

        Args:
            model_state: Model weights to evaluate (serialized)
            num_games: Number of games to use for evaluation

        Returns:
            Dict with evaluation metrics
        """
        log = logger.bind(context=f"SupervisedTrainer.{self.node_id}")
        log.info(f"Evaluating model on {num_games} games")

        try:
            # Load model
            self._initialize_model(model_state)
            self.model.eval()

            # Extract validation samples (use offset for different games)
            extractor = SampleExtractor(self.pgn_database_path, self.extraction_config)
            samples = extractor.extract_samples(
                num_games=num_games,
                playstyle=self.config.playstyle,
                min_rating=2000,
                offset=self.sample_offset + 1000  # Offset to avoid training data
            )

            if len(samples) == 0:
                return {"error": "No validation samples"}

            # Create dataset
            dataset = ChessDataset(samples, self.board_encoder, self.move_encoder)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

            # Evaluate
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for boards, policy_targets, value_targets in dataloader:
                    boards = boards.to(self.device)
                    policy_targets = policy_targets.to(self.device)
                    value_targets = value_targets.to(self.device)

                    policy_logits, value_preds = self.model(boards)

                    policy_loss = self.policy_loss_fn(policy_logits, policy_targets)
                    value_loss = self.value_loss_fn(value_preds, value_targets)

                    total_policy_loss += policy_loss.item() * len(boards)
                    total_value_loss += value_loss.item() * len(boards)
                    total_samples += len(boards)

            metrics = {
                "policy_loss": total_policy_loss / total_samples,
                "value_loss": total_value_loss / total_samples,
                "total_loss": (total_policy_loss + total_value_loss) / total_samples,
                "samples": total_samples
            }

            log.success(f"Evaluation complete: {metrics}")
            return metrics

        except Exception as e:
            log.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

    def _initialize_model(self, model_state: Dict[str, Any]):
        """
        Initialize model and optimizer.

        Args:
            model_state: Either serialized model data (from protocol) or raw state_dict
        """
        if self.model is None:
            self.model = AlphaZeroNet().to(self.device)

        # Load model state
        if model_state:
            # Check if this is serialized data from protocol
            if "serialized_data" in model_state:
                # Deserialize the model state
                state_dict = self.serializer.deserialize(model_state["serialized_data"])
            else:
                # Assume it's already a state_dict (for backward compatibility)
                state_dict = model_state

            self.model.load_state_dict(state_dict)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

    async def _extract_samples(self, offset: int) -> List[TrainingSample]:
        """
        Extract training samples from PGN database.
        
        Args:
            offset: Starting position in the database for sample extraction
        
        Returns:
            List of training samples
        """
        # Run extraction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def extract():
            extractor = SampleExtractor(self.pgn_database_path, self.extraction_config)
            return extractor.extract_samples(
                num_games=self.config.games_per_round,
                playstyle=self.config.playstyle,
                min_rating=2000,  # High-quality games
                offset=offset
            )

        return await loop.run_in_executor(None, extract)

    async def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch."""
        log = logger.bind(context=f"SupervisedTrainer.{self.node_id}")

        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (boards, policy_targets, value_targets) in enumerate(dataloader):
            # Move to device
            boards = boards.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            # Forward pass
            policy_logits, value_preds = self.model(boards)

            # Compute losses
            policy_loss = self.policy_loss_fn(policy_logits, policy_targets)
            value_loss = self.value_loss_fn(value_preds, value_targets)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                log.info(f"Batch {batch_idx + 1}/{len(dataloader)}: "
                        f"loss={loss.item():.4f}, "
                        f"policy={policy_loss.item():.4f}, "
                        f"value={value_loss.item():.4f}")

            # Allow other async tasks to run
            await asyncio.sleep(0)

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "num_batches": num_batches
        }
