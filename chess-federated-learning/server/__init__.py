"""
Server package for federated learning chess system.

This package contains all server-side components including:
- Cluster management and topology
- Communication protocols and server socket
- Aggregation algorithms for model updates

Modules:
    cluster_manager: Manages cluster topology, node assignments, and validation
    communication: Server communication protocols and WebSocket server
    aggregation: Model aggregation algorithms and strategies
"""

from .cluster_manager import ClusterManager, Cluster

# Import communication submodules for convenience
from .communication.protocol import Message, MessageType, MessageFactory
from .communication.server_socket import FederatedLearningServer, NodeState, ConnectedNode

__all__ = [
    # Cluster management
    'ClusterManager',
    'Cluster',

    # Communication
    'Message',
    'MessageType',
    'MessageFactory',
    'FederatedLearningServer',
    'NodeState',
    'ConnectedNode',
]

__version__ = '1.0.0'