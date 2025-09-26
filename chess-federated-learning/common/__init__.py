"""
Common utilities package for federated learning chess system.

This package contains shared utilities and common functionality used by both
server and client components.

Modules:
    model_serialization: Model serialization and deserialization utilities
"""

from .model_serialization import ModelSerializer, PyTorchSerializer, TensorFlowSerializer

__all__ = [
    'ModelSerializer',
    'PyTorchSerializer',
    'TensorFlowSerializer',
]

__version__ = '1.0.0'