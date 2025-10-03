"""
Client trainer package for federated learning.

This package contains trainer implementations for local training on nodes.
"""

from client.trainer.trainer_interface import (
    TrainerInterface,
    TrainingConfig,
    TrainingResult,
    TrainingError,
)
from client.trainer.trainer_dummy import DummyTrainer
from client.trainer.factory import create_trainer

__all__ = [
    "TrainerInterface",
    "TrainingConfig",
    "TrainingResult",
    "TrainingError",
    "DummyTrainer",
    "create_trainer",
]
