"""
Pytest configuration and shared fixtures for federated learning tests.

This module provides reusable fixtures for testing the federated learning system
using in-process server and client instances. All tests run within the same process
using asyncio for realistic but controlled testing.

Key Fixtures:
    - event_loop: Custom event loop for async tests
    - test_server: Running FL server instance
    - test_client: Connected FL client instance
    - multiple_clients: Multiple connected clients for multi-node testing
"""

import pytest
import asyncio
import socket
import time
from typing import List, Dict, Any
from loguru import logger

from server.communication.server_socket import FederatedLearningServer
from client.communication.client_socket import FederatedLearningClient
#from server.cluster_manager import ClusterManager


def get_free_port() -> int:
    """
    Get an available port for testing.
    
    Returns:
        int: Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the entire test session.
    
    This ensures all async tests run in the same event loop,
    which is necessary for proper cleanup and resource management.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_port():
    """
    Provide a free port for testing.
    
    Returns:
        int: Available port number for test server
    """
    return get_free_port()


@pytest.fixture
async def cluster_manager():
    """
    Provide a basic cluster manager for testing.
    
    Returns:
        ClusterManager: Configured cluster manager with test clusters
    """
    # Create a simple cluster manager for testing
    # This is a placeholder - you'll replace with actual ClusterManager
    class MockClusterManager:
        def __init__(self):
            self.nodes = {}
            self.clusters = {
                "test_cluster": {"nodes": set()},
                "cluster_aggressive": {"nodes": set()},
                "cluster_positional": {"nodes": set()}
            }
        
        def is_valid_node(self, node_id: str, cluster_id: str) -> bool:
            return cluster_id in self.clusters
        
        def register_node(self, node_id: str, cluster_id: str):
            self.nodes[node_id] = cluster_id
            if cluster_id in self.clusters:
                self.clusters[cluster_id]["nodes"].add(node_id)
        
        def unregister_node(self, node_id: str):
            if node_id in self.nodes:
                cluster_id = self.nodes[node_id]
                self.clusters[cluster_id]["nodes"].discard(node_id)
                del self.nodes[node_id]
    
    return MockClusterManager()


@pytest.fixture
async def test_server(test_port, cluster_manager):
    """
    Provide a running FL server instance for testing.
    
    The server is started in a background task and will be automatically
    cleaned up after the test completes.
    
    Args:
        test_port: Port to run the server on
        cluster_manager: Cluster manager instance
    
    Yields:
        FederatedLearningServer: Running server instance
    """
    log = logger.bind(context="test_server_fixture")
    log.info(f"Starting test server on port {test_port}")
    
    # Create server instance
    server = FederatedLearningServer(
        host="localhost", 
        port=test_port,
        #cluster_manager=cluster_manager
    )
    
    # Start server in background task
    server_task = asyncio.create_task(server.start_server())
    
    # Wait a moment for server to start listening
    await asyncio.sleep(0.1)
    
    # Verify server is running
    start_time = time.time()
    timeout = 5.0
    while not server.is_running and (time.time() - start_time) < timeout:
        await asyncio.sleep(0.1)
    
    if not server.is_running:
        server_task.cancel()
        raise RuntimeError("Test server failed to start within timeout")
    
    log.info(f"Test server started successfully on port {test_port}")
    
    try:
        yield server
    finally:
        # Cleanup: stop server
        log.info("Stopping test server")
        await server.stop_server()
        
        # Cancel server task if still running
        if not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
        
        log.info("Test server stopped")


@pytest.fixture
async def test_client(test_server, test_port):
    """
    Provide a connected FL client instance for testing.
    
    The client is automatically connected and registered with the test server.
    
    Args:
        test_server: Running server instance
        test_port: Server port
    
    Yields:
        FederatedLearningClient: Connected and registered client
    """
    log = logger.bind(context="test_client_fixture")
    log.info("Creating test client")
    
    # Create client instance
    client = FederatedLearningClient(
        node_id="test_node_001",
        cluster_id="test_cluster",
        server_host="localhost",
        server_port=test_port
    )
    
    # Disable auto-reconnect for cleaner test behavior
    client.auto_reconnect = False
    
    # Start client in background task
    client_task = asyncio.create_task(client.start())
    
    # Wait for client to connect and register
    start_time = time.time()
    timeout = 10.0
    while not client.is_connected() and (time.time() - start_time) < timeout:
        await asyncio.sleep(0.1)
    
    if not client.is_connected():
        await client.stop()
        client_task.cancel()
        raise RuntimeError("Test client failed to connect within timeout")
    
    log.info("Test client connected and registered")
    
    try:
        yield client
    finally:
        # Cleanup: stop client
        log.info("Stopping test client")
        await client.stop()
        
        # Cancel client task if still running
        if not client_task.done():
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass
        
        log.info("Test client stopped")


@pytest.fixture
async def multiple_clients(test_server, test_port):
    """
    Provide multiple connected FL clients for testing.
    
    Creates clients for both aggressive and positional clusters to simulate
    the actual federated learning setup.
    
    Args:
        test_server: Running server instance
        test_port: Server port
    
    Yields:
        Dict[str, FederatedLearningClient]: Dictionary of client_id -> client instance
    """
    log = logger.bind(context="multiple_clients_fixture")
    log.info("Creating multiple test clients")
    
    clients = {}
    client_tasks = []
    
    # Create clients for different clusters
    client_configs = [
        ("agg_001", "cluster_aggressive"),
        ("agg_002", "cluster_aggressive"),
        ("pos_001", "cluster_positional"),
        ("pos_002", "cluster_positional"),
        ("test_001", "test_cluster")
    ]
    
    try:
        # Create and start all clients
        for node_id, cluster_id in client_configs:
            client = FederatedLearningClient(
                node_id=node_id,
                cluster_id=cluster_id,
                server_host="localhost",
                server_port=test_port
            )
            
            # Disable auto-reconnect for cleaner test behavior
            client.auto_reconnect = False
            
            # Start client
            client_task = asyncio.create_task(client.start())
            client_tasks.append(client_task)
            clients[node_id] = client
        
        # Wait for all clients to connect
        start_time = time.time()
        timeout = 15.0
        
        while (time.time() - start_time) < timeout:
            connected_count = sum(1 for client in clients.values() if client.is_connected())
            if connected_count == len(clients):
                break
            await asyncio.sleep(0.1)
        
        # Verify all clients connected
        connected_clients = [cid for cid, client in clients.items() if client.is_connected()]
        if len(connected_clients) != len(clients):
            failed_clients = [cid for cid, client in clients.items() if not client.is_connected()]
            raise RuntimeError(f"Some clients failed to connect: {failed_clients}")
        
        log.info(f"All {len(clients)} test clients connected successfully")
        
        yield clients
        
    finally:
        # Cleanup: stop all clients
        log.info("Stopping all test clients")
        
        # Stop all clients
        stop_tasks = []
        for client in clients.values():
            stop_tasks.append(asyncio.create_task(client.stop()))
        
        # Wait for all stops to complete
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel all client tasks
        for task in client_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        log.info("All test clients stopped")


@pytest.fixture
def dummy_model_state():
    """
    Provide dummy model state for testing serialization.
    
    Returns:
        Dict: Dummy PyTorch-style state dict
    """
    return {
        "layer1.weight": [[0.1, 0.2], [0.3, 0.4]],  # Simulated tensor as nested list
        "layer1.bias": [0.1, 0.2],
        "layer2.weight": [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        "layer2.bias": [0.3, 0.4, 0.5]
    }


@pytest.fixture
def training_metrics():
    """
    Provide dummy training metrics for testing.
    
    Returns:
        Dict: Training metrics
    """
    return {
        "loss": 0.25,
        "samples": 1000,
        "accuracy": 0.85,
        "training_time": 45.2,
        "games_played": 100
    }


# Configure pytest for async testing
pytest_plugins = ['pytest_asyncio']

# Set default timeout for async tests
def pytest_configure(config):
    """Configure pytest settings."""
    # Set up logging for tests
    logger.remove()  # Remove default logger
    logger.add(
        "tests.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB"
    )
    
    # Also log to console during tests
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger.info("Starting federated learning test session")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    logger.info(f"Federated learning test session finished with status: {exitstatus}")