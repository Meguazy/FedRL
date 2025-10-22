"""
Puzzle trainer for tactical pattern recognition.

This trainer uses Lichess puzzle database to train the model on tactical motifs.
Unlike supervised learning from full games, puzzles focus on critical positions
where there's a clear best move sequence (tactics, combinations, checkmates).

Key differences from supervised trainer:
- Uses puzzle database (FEN + move sequence) instead of PGN games
- Focuses on tactical positions only
- Can filter by puzzle rating and themes
- Each position has a verified correct solution
"""

import asyncio
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from client.trainer.trainer_interface import (
    TrainerInterface,
    TrainingConfig,
    TrainingResult,
    TrainingError
)
from client.trainer.models.alphazero_net import AlphaZeroNet
from data.board_encoder import BoardEncoder
from data.move_encoder import MoveEncoder
from data.redis_puzzle_cache import RedisPuzzleCache
from common.model_serialization import PyTorchSerializer


@dataclass
class Puzzle:
    """Represents a single chess puzzle."""
    puzzle_id: str
    fen: str
    moves: List[str]  # UCI format moves
    rating: int
    themes: List[str]
    
    def get_position(self) -> chess.Board:
        """Get the starting position."""
        return chess.Board(self.fen)
    
    def get_solution_move(self) -> str:
        """Get the first move of the solution (the move to find)."""
        # Puzzles format: opponent_move solution_move1 solution_move2 ...
        # We want solution_move1 (index 1)
        return self.moves[1] if len(self.moves) > 1 else self.moves[0]


@dataclass
class PuzzlePosition:
    """Represents a single position in a puzzle sequence."""
    board: chess.Board
    solution_move: str
    position_index: int  # Which move in the sequence (0 = first move)


class PuzzleDataset(Dataset):
    """
    PyTorch dataset for chess puzzles.
    
    Extracts ALL positions from multi-move puzzles, creating one training
    sample per move in the solution sequence.
    """
    
    def __init__(self,
                 puzzles: List[Puzzle],
                 board_encoder: BoardEncoder,
                 move_encoder: MoveEncoder):
        """
        Initialize puzzle dataset.
        
        Args:
            puzzles: List of puzzles
            board_encoder: Encoder for board positions
            move_encoder: Encoder for moves
        """
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        
        # Extract all positions from all puzzles
        self.positions = self._extract_positions(puzzles)
        
    def _extract_positions(self, puzzles: List[Puzzle]) -> List[PuzzlePosition]:
        """
        Extract all training positions from puzzles.
        
        For a puzzle with moves: [setup, move1, reply1, move2, reply2, ...]
        We create samples for: move1, move2, ... (all player moves)
        
        Args:
            puzzles: List of puzzles
            
        Returns:
            List of positions ready for training
        """
        positions = []
        
        for puzzle in puzzles:
            board = puzzle.get_position()
            
            # Play through the entire move sequence
            # moves[0] = opponent setup move
            # moves[1] = our first move (LEARN THIS)
            # moves[2] = opponent reply
            # moves[3] = our second move (LEARN THIS)
            # etc.
            
            for move_idx, move_uci in enumerate(puzzle.moves):
                # Odd indices are OUR moves (1, 3, 5, ...) - these are the ones to learn
                if move_idx > 0 and move_idx % 2 == 1:
                    # Save current position before making the move
                    position = PuzzlePosition(
                        board=board.copy(),
                        solution_move=move_uci,
                        position_index=(move_idx - 1) // 2  # 0 for first move, 1 for second, etc.
                    )
                    positions.append(position)
                
                # Make the move to advance to next position
                try:
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                except:
                    # Invalid move, skip rest of puzzle
                    break
        
        return positions
        
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a training sample.
        
        Returns:
            Tuple of (board_tensor, policy_target, value_target):
                - board_tensor: Shape (119, 8, 8) - encoded board state
                - policy_target: Scalar integer (0-4671) - move class index
                - value_target: Shape (1,) - value is 1.0 (winning move)
        """
        position = self.positions[idx]
        
        # Encode the position
        board_tensor = self.board_encoder.encode(position.board, history=[])
        
        # Encode the solution move
        solution_move = chess.Move.from_uci(position.solution_move)
        move_index = self.move_encoder.encode(solution_move, position.board)
        
        # Policy target: the correct move
        policy_target = move_index
        
        # Value target: puzzles are winning positions, so value = 1.0
        value_target = np.array([1.0], dtype=np.float32)
        
        return (
            torch.from_numpy(board_tensor),
            torch.tensor(policy_target, dtype=torch.long),
            torch.from_numpy(value_target)
        )


class PuzzleTrainer(TrainerInterface):
    """
    Puzzle trainer for tactical pattern recognition.
    
    Trains on Lichess puzzle database, focusing on tactical motifs.
    """
    
    def __init__(self,
                 node_id: str,
                 config: TrainingConfig,
                 puzzle_database_path: str,
                 device: str = 'cpu',
                 redis_host: str = 'localhost',
                 redis_port: int = 6381):
        """
        Initialize puzzle trainer.
        
        Args:
            node_id: Unique identifier for this node (e.g., "agg_001", "pos_003")
            config: Training configuration
            puzzle_database_path: Path to Lichess puzzle database (legacy, not used if Redis available)
            device: Device to train on ('cpu' or 'cuda')
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.node_id = node_id
        self.config = config
        self.puzzle_database_path = puzzle_database_path
        self.device = torch.device(device)
        
        # Extract node index from node_id (e.g., "agg_001" -> 0, "agg_002" -> 1, etc.)
        # Format: {prefix}_{index:03d}
        try:
            self.node_index = int(node_id.split('_')[-1]) - 1  # 0-indexed
        except (ValueError, IndexError):
            logger.warning(f"Could not extract node index from node_id '{node_id}', defaulting to 0")
            self.node_index = 0
        
        # Track current round (for offset calculation in federated setting)
        self.current_round = 0
        self.round_offset = 0
        
        # Redis cache for puzzles
        self.redis_cache = RedisPuzzleCache(host=redis_host, port=redis_port)
        
        # Initialize encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        
        # Model serializer
        self.serializer = PyTorchSerializer(compression=True, encoding='base64')
        
        # Model (will be created when training starts)
        self.model: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Track best loss for scheduler
        self.best_loss = float('inf')
        
        # Puzzle filtering
        self.min_rating = config.additional_params.get('min_puzzle_rating', 1500)
        self.max_rating = config.additional_params.get('max_puzzle_rating', 2500)
        self.themes = config.additional_params.get('themes', None)  # None = all themes
        
    def set_current_round(self, round_num: int):
        """Set the current training round number."""
        self.current_round = round_num
        
    def set_round_offset(self, offset: int):
        """Set the round offset for resume training."""
        self.round_offset = offset
        
    async def train(self, initial_model_state: Dict[str, Any]) -> TrainingResult:
        """
        Perform puzzle training.
        
        Args:
            initial_model_state: Starting model weights (serialized state_dict)
            
        Returns:
            TrainingResult with updated model and metrics
        """
        log = logger.bind(context=f"PuzzleTrainer.{self.node_id}")
        log.info("Starting puzzle training")
        
        if not Path(self.puzzle_database_path).exists():
            raise TrainingError(f"Puzzle database not found: {self.puzzle_database_path}")
        
        start_time = time.time()
        
        try:
            # 1. Load model
            self._initialize_model(initial_model_state)
            log.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
            # 2. Load puzzles
            log.info(f"Loading puzzles from database (rating {self.min_rating}-{self.max_rating})")
            puzzles = await self._load_puzzles()
            log.success(f"Loaded {len(puzzles)} puzzles")
            
            if len(puzzles) == 0:
                raise TrainingError("No puzzles loaded")
            
            # 3. Create dataset and dataloader
            # Note: Dataset extracts ALL positions from multi-move puzzles
            dataset = PuzzleDataset(puzzles, self.board_encoder, self.move_encoder)
            log.info(f"Extracted {len(dataset)} training positions from {len(puzzles)} puzzles "
                    f"(avg {len(dataset)/len(puzzles):.1f} positions per puzzle)")
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False
            )
            log.info(f"Created dataloader with {len(dataloader)} batches")
            
            # 4. Train for one epoch
            metrics = await self._train_epoch(dataloader)
            
            # 5. Update learning rate scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(metrics['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            log.info(f"Current learning rate: {current_lr:.6f}")
            if current_lr != old_lr:
                log.warning(f"Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}")
            
            # 6. Track best loss
            if metrics['total_loss'] < self.best_loss:
                self.best_loss = metrics['total_loss']
                log.info(f"New best loss: {self.best_loss:.4f}")
            
            # 7. Serialize model
            model_state = self.serializer.serialize(self.model.state_dict())
            
            training_time = time.time() - start_time
            
            log.success(f"Training complete: loss={metrics['total_loss']:.4f}")
            
            return TrainingResult(
                model_state={"serialized_data": model_state},
                samples=len(puzzles),
                loss=metrics['total_loss'],
                games_played=len(puzzles),  # Each puzzle is like a "game"
                training_time=training_time,
                metrics={
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss'],
                    'num_batches': metrics['num_batches'],
                    'min_puzzle_rating': self.min_rating,
                    'max_puzzle_rating': self.max_rating
                },
                success=True
            )
            
        except Exception as e:
            log.error(f"Training failed: {e}")
            raise TrainingError(f"Puzzle training failed: {e}")
    
    def _initialize_model(self, model_state: Optional[Dict[str, Any]] = None):
        """Initialize or update the model."""
        log = logger.bind(context=f"PuzzleTrainer.{self.node_id}")
        
        # Create model if it doesn't exist
        if self.model is None:
            self.model = AlphaZeroNet(
                num_res_blocks=19,
                num_channels=256
            )
            self.model.to(self.device)
        
        # Load model state
        if model_state:
            if "serialized_data" in model_state:
                state_dict = self.serializer.deserialize(model_state["serialized_data"])
            else:
                state_dict = model_state
            
            self.model.load_state_dict(state_dict)
        
        # Create optimizer and scheduler only once (persist across rounds)
        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=15,
                min_lr=1e-6
            )
            
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Optimizer created with learning rate: {current_lr}")
        else:
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Reusing optimizer with learning rate: {current_lr}")
    
    async def _load_puzzles(self) -> List[Puzzle]:
        """
        Load puzzles from Redis cache with proper offset calculation.
        
        Offset Strategy (Federated Learning):
        - Nodes within the SAME cluster must see DIFFERENT puzzles (for aggregation benefit)
        - Nodes in DIFFERENT clusters can see SAME puzzles (for fair comparison)
        
        Offset Formula:
            offset = (current_round + round_offset) * (nodes_per_cluster * games_per_round) + node_index * games_per_round
        
        Where:
            - current_round: Current training round (0, 1, 2, ...)
            - round_offset: Offset for resume training (e.g., 50 if resuming from round 50)
            - nodes_per_cluster: Number of nodes in this cluster (e.g., 4)
            - games_per_round: Puzzles per round per node (e.g., 500)
            - node_index: Index within cluster (0, 1, 2, 3 for a 4-node cluster)
        
        Example with 4 nodes, 500 puzzles/round, resuming from round 50:
            Round 0 (actual round 50):
                node_index=0: offset = 50 * (4 * 500) + 0 * 500 = 100,000
                node_index=1: offset = 50 * (4 * 500) + 1 * 500 = 100,500
                node_index=2: offset = 50 * (4 * 500) + 2 * 500 = 101,000
                node_index=3: offset = 50 * (4 * 500) + 3 * 500 = 101,500
            Round 1 (actual round 51):
                node_index=0: offset = 51 * (4 * 500) + 0 * 500 = 102,000
                ...
        
        Returns:
            List of Puzzle objects loaded from Redis
        """
        log = logger.bind(context=f"PuzzleTrainer.{self.node_id}")
        loop = asyncio.get_event_loop()
        
        def load():
            # Get total puzzle count from Redis
            total_puzzles = self.redis_cache.get_total_puzzles()
            if total_puzzles == 0:
                log.error("No puzzles found in Redis. Please run index_puzzles_to_redis.py first.")
                raise TrainingError("Redis puzzle cache is empty")
            
            log.info(f"Total puzzles in Redis: {total_puzzles:,}")
            
            # Calculate nodes per cluster from config (assume symmetric clusters)
            # This should ideally come from cluster topology, but we can infer from node_id
            # For now, we'll use a reasonable default of 4 nodes per cluster
            # TODO: Pass this from cluster_topology.yaml in the future
            nodes_per_cluster = 4
            
            # Calculate offset for this node
            # Formula: (round + round_offset) * (nodes_per_cluster * games_per_round) + node_index * games_per_round
            effective_round = self.current_round + self.round_offset
            round_base = effective_round * (nodes_per_cluster * self.config.games_per_round)
            node_offset = self.node_index * self.config.games_per_round
            offset = round_base + node_offset
            
            log.info(f"Calculating offset: round={self.current_round}, round_offset={self.round_offset}, "
                    f"effective_round={effective_round}, nodes_per_cluster={nodes_per_cluster}, "
                    f"games_per_round={self.config.games_per_round}, node_index={self.node_index}")
            log.info(f"Offset calculation: round_base={round_base}, node_offset={node_offset}, total_offset={offset}")
            
            # Ensure offset doesn't exceed available puzzles (wrap around if needed)
            if offset >= total_puzzles:
                log.warning(f"Offset {offset} >= total puzzles {total_puzzles}. Wrapping around.")
                offset = offset % total_puzzles
            
            # Load puzzles from Redis
            count = self.config.games_per_round
            log.info(f"Loading {count} puzzles from Redis starting at offset {offset}")
            
            puzzle_dicts = self.redis_cache.get_puzzles(offset=offset, count=count)
            
            if not puzzle_dicts:
                log.error(f"No puzzles retrieved from Redis at offset {offset}")
                raise TrainingError(f"Failed to load puzzles from Redis at offset {offset}")
            
            # Convert to Puzzle objects
            puzzles = []
            for pd in puzzle_dicts:
                try:
                    # Apply rating filter if configured
                    rating = pd['rating']
                    if rating < self.min_rating or rating > self.max_rating:
                        continue
                    
                    # Apply theme filter if configured
                    themes = pd['themes']
                    if self.themes and not any(t in themes for t in self.themes):
                        continue
                    
                    puzzle = Puzzle(
                        puzzle_id=pd['puzzle_id'],
                        fen=pd['fen'],
                        moves=pd['moves'],
                        rating=rating,
                        themes=themes
                    )
                    puzzles.append(puzzle)
                    
                except (ValueError, KeyError) as e:
                    log.warning(f"Skipping invalid puzzle: {e}")
                    continue
            
            # If filters reduced puzzle count too much, log warning
            if len(puzzles) < count * 0.5:  # Less than 50% of expected
                log.warning(f"Filters reduced puzzle count significantly: {len(puzzles)}/{count} "
                          f"(rating: {self.min_rating}-{self.max_rating}, themes: {self.themes})")
            
            log.info(f"Loaded {len(puzzles)} puzzles after filtering (offset={offset}, count={count})")
            return puzzles
        
        return await loop.run_in_executor(None, load)
    
    async def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch."""
        log = logger.bind(context=f"PuzzleTrainer.{self.node_id}")
        
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
            
            # Free memory
            del boards, policy_targets, value_targets
            del policy_logits, value_preds, policy_loss, value_loss, loss
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                log.info(f"Batch {batch_idx + 1}/{len(dataloader)}: "
                        f"loss={total_loss / num_batches:.4f}, "
                        f"policy={total_policy_loss / num_batches:.4f}, "
                        f"value={total_value_loss / num_batches:.4f}")
            
            # Memory cleanup
            if (batch_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            await asyncio.sleep(0)
        
        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "num_batches": num_batches
        }
