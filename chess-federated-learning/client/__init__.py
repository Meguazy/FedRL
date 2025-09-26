"""
Client package for federated learning chess system.

This package contains all client-side components for participating nodes in the
federated learning network. It handles connection to the server, training coordination,
and model updates.

Modules:
    communication: Client communication protocols and WebSocket client
"""

# Import communication submodules for convenience
from .communication.client_socket import FederatedLearningClient, ClientState, ConnectionStats

__all__ = [
    'FederatedLearningClient',
    'ClientState',
    'ConnectionStats',
]

__version__ = '1.0.0'