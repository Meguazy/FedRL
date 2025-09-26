"""
Server communication package.

This package handles all server-side communication including:
- WebSocket server implementation for federated learning
- Protocol definitions and message factories
- Connection management and routing

Modules:
    server_socket: WebSocket server for federated learning coordination
    protocol: Message protocol definitions and utilities
"""

from .server_socket import FederatedLearningServer, NodeState, ConnectedNode
from .protocol import Message, MessageType, MessageFactory

__all__ = [
    'FederatedLearningServer',
    'NodeState',
    'ConnectedNode',
    'Message',
    'MessageType',
    'MessageFactory',
]

__version__ = '1.0.0'