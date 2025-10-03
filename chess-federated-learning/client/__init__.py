"""
Client package for federated learning chess system.

This package contains all client-side components for participating nodes in the
federated learning network. It handles connection to the server, training coordination,
and model updates.

Modules:
    communication: Client communication protocols and WebSocket client
    trainer: Training interface and implementations (dummy, supervised, AlphaZero)
    node: Main federated learning node orchestrator
    config: Configuration loading and management
"""

# Import communication submodules
from .communication.client_socket import FederatedLearningClient, ClientState, ConnectionStats

# Import trainer components
from .trainer import (
    TrainerInterface,
    TrainingConfig,
    TrainingResult,
    TrainingError,
    DummyTrainer,
    create_trainer,
)

# Import node
from .node import FederatedLearningNode, NodeLifecycleState

# Import config
from .config import NodeConfig, create_default_config, load_config_with_overrides

__all__ = [
    # Communication
    'FederatedLearningClient',
    'ClientState',
    'ConnectionStats',
    # Trainer
    'TrainerInterface',
    'TrainingConfig',
    'TrainingResult',
    'TrainingError',
    'DummyTrainer',
    'create_trainer',
    # Node
    'FederatedLearningNode',
    'NodeLifecycleState',
    # Config
    'NodeConfig',
    'create_default_config',
    'load_config_with_overrides',
]

__version__ = '1.0.0'