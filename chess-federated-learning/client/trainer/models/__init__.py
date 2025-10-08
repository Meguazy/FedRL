"""
Neural network models for federated learning trainers.

This package contains model architectures used by various trainers,
including AlphaZero-style networks for chess.
"""

from .alphazero_net import (
    AlphaZeroNet,
    ResidualBlock,
    PolicyHead,
    ValueHead,
    create_alphazero_net
)

__all__ = [
    "AlphaZeroNet",
    "ResidualBlock",
    "PolicyHead",
    "ValueHead",
    "create_alphazero_net"
]
