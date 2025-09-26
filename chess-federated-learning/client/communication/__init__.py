"""
Client communication package.

This package handles all client-side communication including:
- WebSocket client implementation for connecting to federated learning server
- Message handling and routing for training coordination
- Connection management with automatic reconnection

Modules:
    client_socket: WebSocket client for federated learning nodes
"""

from .client_socket import FederatedLearningClient, ClientState, ConnectionStats

__all__ = [
    'FederatedLearningClient',
    'ClientState',
    'ConnectionStats',
]

__version__ = '1.0.0'