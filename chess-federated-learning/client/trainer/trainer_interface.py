"""
Abstract trainer interface for federated learning nodes.

This module defines the interface that all trainers must implement, allowing
the node logic to remain agnostic to the specific training implementation.
This enables easy swapping between dummy trainers (for testing) and real
chess engines (AlphaZero, supervised learning, etc.).

Key Features:
    - Framework-agnostic design (works with PyTorch, TensorFlow, etc.)
    - Async-first API for non-blocking training
    - Structured training results with metrics
    - Support for both self-play and supervised learning
    - Easy testing with mock implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class TrainingConfig:
    """
    Configuration for a training session.
    
    This dataclass encapsulates all parameters needed for local training,
    allowing easy serialization and modification without changing code.
    
    Attributes:
        games_per_round: Number of games/episodes to play this round
        batch_size: Batch size for training updates
        learning_rate: Learning rate for optimizer
        exploration_factor: Exploration parameter (e.g., temperature for MCTS)
        max_game_length: Maximum moves per game before draw
        save_games: Whether to save game data (PGNs, trajectories)
        playstyle: Cluster playstyle ("tactical", "positional", etc.)
        additional_params: Dict for trainer-specific parameters
    """
    games_per_round: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    exploration_factor: float = 1.0
    max_game_length: int = 200
    save_games: bool = True
    playstyle: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class TrainingResult:
    """
    Results from a training session.
    
    Contains all information about the training run, including updated
    model weights and comprehensive metrics for storage and analysis.
    
    Attributes:
        model_state: Updated model state dict after training
        samples: Number of training samples used
        loss: Average training loss
        games_played: Number of games completed
        training_time: Wall-clock time for training (seconds)
        metrics: Additional metrics (accuracy, win rate, etc.)
        game_data: Optional game trajectories/PGNs for storage
        success: Whether training completed successfully
        error_message: Error details if training failed
    """
    model_state: Dict[str, Any]
    samples: int
    loss: float
    games_played: int
    training_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    game_data: Optional[List[str]] = None  # e.g., PGN strings
    success: bool = True
    error_message: Optional[str] = None
    

class TrainerInterface(ABC):
    """
    Abstract base class for all trainer implementations.
    
    This interface defines the contract that all trainers must follow,
    enabling the node logic to work with any training implementation
    without modification.
    
    Implementations might include:
    - DummyTrainer: For testing (random moves, synthetic data)
    - SupervisedTrainer: Training on human game databases
    - AlphaZeroTrainer: Self-play with MCTS
    - HybridTrainer: Combination of supervised + self-play
    
    The interface is async-first to allow non-blocking training that can
    be cancelled or monitored in real-time.
    """
    
    def __init__(self, node_id: str, cluster_id: str, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            node_id: Unique identifier for this node
            cluster_id: Cluster this node belongs to
            config: Training configuration
        """
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.config = config

        self.current_model_state: Optional[Dict[str, Any]] = None
        self.training_history: List[TrainingResult] = []
        self.total_games_played: int = 0
        self.total_training_time: float = 0.0
        
        log = logger.bind(context=f"{self.__class__.__name__}.{node_id}")
        log.info(f"Initialized trainer for node {node_id} in cluster {cluster_id}")
        
    @abstractmethod
    async def train(self, initial_model_state: Dict[str, Any]) -> TrainingResult:
        """
        Perform local training starting from initial model state.
        
        This is the main training method that nodes call each round.
        The implementation should:
        1. Load the initial model state
        2. Perform training (self-play, supervised, etc.)
        3. Update the model based on training data
        4. Collect metrics and results
        5. Return TrainingResult with updated model
        
        Args:
            initial_model_state: Starting model weights for this round
        
        Returns:
            TrainingResult: Updated model and training metrics
        
        Raises:
            TrainingError: If training fails
        """
        pass
    
    @abstractmethod
    async def evaluate(self, model_state: Dict[str, Any], 
                      num_games: int = 10) -> Dict[str, Any]:
        """
        Evaluate model performance without training.
        
        Useful for validation, testing against baselines, or
        computing metrics without modifying the model.
        
        Args:
            model_state: Model weights to evaluate
            num_games: Number of evaluation games
        
        Returns:
            Dict containing evaluation metrics (win rate, avg loss, etc.)
        """
        pass
    
    def update_config(self, config: TrainingConfig):
        """
        Update training configuration.
        
        Allows dynamic adjustment of training parameters between rounds
        (e.g., decreasing exploration over time).
        
        Args:
            config: New training configuration
        """
        log = logger.bind(context=f"{self.__class__.__name__}.{self.node_id}")
        log.info(f"Updating training config: {config}")
        self.config = config
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trainer statistics.
        
        Returns:
            Dict containing cumulative training statistics
        """
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "total_games_played": self.total_games_played,
            "total_training_time": self.total_training_time,
            "rounds_completed": len(self.training_history),
            "average_loss": sum(r.loss for r in self.training_history) / len(self.training_history)
                           if self.training_history else 0.0,
        }
    
    def reset_statistics(self):
        """Reset training statistics (useful for new experiments)."""
        log = logger.bind(context=f"{self.__class__.__name__}.{self.node_id}")
        log.info("Resetting trainer statistics")
        
        self.training_history.clear()
        self.total_games_played = 0
        self.total_training_time = 0.0
        
        
class TrainingError(Exception):
    """Exception raised when training fails."""
    pass
