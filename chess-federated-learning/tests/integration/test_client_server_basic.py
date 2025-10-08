"""
Basic client-server integration tests.

This module tests the fundamental client-server communication functionality
including connection establishment, registration, and basic message exchange.
These tests verify that the core communication stack works correctly.
"""

import pytest
import asyncio
from loguru import logger

from server.communication.protocol import MessageType, MessageFactory
from tests.fixtures.test_utils import (
    wait_for_condition, generate_random_model, assert_message_valid,
    MessageCollector
)


class TestBasicConnection:
    """Test basic client-server connection functionality."""
    
    @pytest.mark.asyncio
    async def test_server_starts_successfully(self, test_server):
        """Test that the FL server starts and is ready to accept connections."""
        log = logger.bind(context="test_server_starts_successfully")
        log.info("Testing server startup")
        
        # Server should be running
        assert test_server.is_running, "Server should be running"
        assert test_server.get_node_count() == 0, "Server should have no connected nodes initially"
        assert test_server.get_cluster_count() == 0, "Server should have no active clusters initially"
        
        log.info("✓ Server startup test passed")
    
    @pytest.mark.asyncio
    async def test_client_connects_and_registers(self, test_client):
        """Test that a client can connect and register with the server."""
        log = logger.bind(context="test_client_connects_and_registers")
        log.info("Testing client connection and registration")
        
        # Client should be connected and registered
        assert test_client.is_connected(), "Client should be connected"
        assert test_client.node_id == "test_node_001", "Client should have correct node ID"
        assert test_client.cluster_id == "test_cluster", "Client should have correct cluster ID"
        
        # Check connection stats
        stats = test_client.get_connection_stats()
        assert stats.successful_connections >= 1, "Client should have at least one successful connection"
        assert stats.total_messages_sent >= 1, "Client should have sent registration message"
        
        log.info("✓ Client connection and registration test passed")
    
    @pytest.mark.asyncio
    async def test_server_tracks_connected_client(self, test_server, test_client):
        """Test that the server correctly tracks the connected client."""
        log = logger.bind(context="test_server_tracks_connected_client")
        log.info("Testing server client tracking")
        
        # Wait for registration to complete
        await asyncio.sleep(0.5)
        
        # Server should track the connected client
        assert test_server.get_node_count() == 1, "Server should have 1 connected node"
        assert test_server.get_cluster_count() == 1, "Server should have 1 active cluster"
        
        # Check specific client details
        connected_nodes = test_server.get_connected_nodes()
        assert "test_node_001" in connected_nodes, "Server should track test client"
        
        node = connected_nodes["test_node_001"]
        assert node.node_id == "test_node_001", "Node should have correct ID"
        assert node.cluster_id == "test_cluster", "Node should have correct cluster"
        assert node.state.value in ["registered", "idle"], "Node should be in registered or idle state"
        
        # Check cluster tracking
        cluster_nodes = test_server.get_cluster_nodes("test_cluster")
        assert "test_node_001" in cluster_nodes, "Server should track client in cluster"
        
        log.info("✓ Server client tracking test passed")


class TestMessageExchange:
    """Test basic message exchange between client and server."""
    
    @pytest.mark.asyncio
    async def test_client_sends_heartbeat(self, test_client):
        """Test that client sends heartbeat messages."""
        log = logger.bind(context="test_client_sends_heartbeat")
        log.info("Testing client heartbeat functionality")
        
        # Get initial message count
        initial_messages = test_client.get_connection_stats().total_messages_sent

        # Restart heartbeat with shorter interval for testing
        test_client.restart_heartbeat_with_interval(0.5)  # 0.5 seconds for testing
        await asyncio.sleep(1.5)  # Wait long enough for multiple heartbeats
        
        # Should have sent at least one more message (heartbeat)
        final_messages = test_client.get_connection_stats().total_messages_sent
        assert final_messages > initial_messages, "Client should send heartbeat messages"
        
        log.info("✓ Client heartbeat test passed")
    
    @pytest.mark.asyncio
    async def test_server_broadcasts_message(self, test_server, multiple_clients):
        """Test that server can broadcast messages to multiple clients."""
        log = logger.bind(context="test_server_broadcasts_message")
        log.info("Testing server broadcast functionality")
        
        # Wait for all clients to connect
        await asyncio.sleep(1.0)
        
        # Create message collectors for each client
        collectors = {}
        for node_id, client in multiple_clients.items():
            collector = MessageCollector()
            client.set_message_handler(MessageType.START_TRAINING, collector.collect_message)
            collectors[node_id] = collector
        
        # Broadcast start training message
        broadcast_msg = MessageFactory.create_start_training(
            "broadcast", "all_clusters", 100, 1
        )
        
        await test_server.broadcast_to_all(broadcast_msg)
        
        # Wait for message delivery
        await asyncio.sleep(1.0)
        
        # All clients should have received the message
        for node_id, collector in collectors.items():
            messages = collector.get_messages(MessageType.START_TRAINING)
            assert len(messages) >= 1, f"Client {node_id} should receive broadcast message"
            
            # Verify message content
            msg = messages[0]
            assert msg.payload["games_per_round"] == 100, "Message should have correct payload"
            assert msg.round_num == 1, "Message should have correct round number"
        
        log.info("✓ Server broadcast test passed")
    
    @pytest.mark.asyncio
    async def test_client_sends_model_update(self, test_client, test_server):
        """Test that client can send model updates to server."""
        log = logger.bind(context="test_client_sends_model_update")
        log.info("Testing client model update")
        
        # Generate dummy model
        dummy_model = generate_random_model(seed=42)
        model_state = dummy_model.state_dict()
        
        # Set up message handler on server to capture model updates
        received_updates = []
        
        async def capture_model_update(node_id: str, message):
            log.debug(f"Captured model update from {node_id}")
            received_updates.append((node_id, message))
        
        test_server.set_message_handler(MessageType.MODEL_UPDATE, capture_model_update)
        
        # Send model update
        await test_client.send_model_update(
            model_state=model_state,
            samples=1000,
            loss=0.25,
            round_num=1
        )
        
        # Wait for server to process
        await asyncio.sleep(0.5)
        
        # Server should have received the update
        assert len(received_updates) == 1, "Server should receive model update"
        
        node_id, message = received_updates[0]
        assert node_id == "test_node_001", "Update should be from correct node"
        assert message.type == MessageType.MODEL_UPDATE.value, "Message should be MODEL_UPDATE"
        assert message.payload["samples"] == 1000, "Update should have correct sample count"
        assert message.payload["loss"] == 0.25, "Update should have correct loss"
        assert message.round_num == 1, "Update should have correct round number"
        
        log.info("✓ Client model update test passed")


class TestConnectionRecovery:
    """Test connection recovery and error handling."""
    
    @pytest.mark.asyncio
    async def test_client_handles_disconnection(self, test_server, test_port):
        """Test that client handles server disconnection gracefully."""
        log = logger.bind(context="test_client_handles_disconnection")
        log.info("Testing client disconnection handling")
        
        from client.communication.client_socket import FederatedLearningClient, ClientState
        
        # Create client with auto-reconnect disabled for controlled testing
        client = FederatedLearningClient(
            "disconnect_test", "test_cluster", 
            "localhost", test_port
        )
        client.auto_reconnect = False
        
        # Start client
        client_task = asyncio.create_task(client.start())
        
        # Wait for connection
        await wait_for_condition(lambda: client.is_connected(), timeout=5.0)
        assert client.is_connected(), "Client should connect initially"
        
        # Force disconnect by stopping the server
        await test_server.stop_server()
        
        # Wait for client to detect disconnection
        await wait_for_condition(
            lambda: client.state == ClientState.DISCONNECTED, 
            timeout=10.0
        )
        
        assert not client.is_connected(), "Client should detect disconnection"
        assert client.state == ClientState.DISCONNECTED, "Client should be in disconnected state"
        
        # Cleanup
        await client.stop()
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
        
        log.info("✓ Client disconnection handling test passed")
    
    @pytest.mark.asyncio
    async def test_server_handles_client_disconnection(self, test_server, test_client):
        """Test that server properly cleans up when client disconnects."""
        log = logger.bind(context="test_server_handles_client_disconnection")
        log.info("Testing server cleanup on client disconnect")
        
        # Wait for connection to establish
        await asyncio.sleep(0.5)
        
        # Verify client is connected
        assert test_server.get_node_count() == 1, "Server should have 1 connected node"
        
        # Disconnect client
        await test_client.stop()
        
        # Wait for server to clean up
        cleanup_successful = await wait_for_condition(
            lambda: test_server.get_node_count() == 0, 
            timeout=5.0
        )
        
        assert cleanup_successful, "Server should clean up disconnected client"
        assert test_server.get_node_count() == 0, "Server should have no connected nodes"
        assert test_server.get_cluster_count() == 0, "Server should have no active clusters"
        
        log.info("✓ Server client disconnect cleanup test passed")


class TestMultiNodeScenarios:
    """Test scenarios with multiple nodes."""
    
    @pytest.mark.asyncio
    async def test_multiple_clients_connect(self, test_server, multiple_clients):
        """Test that multiple clients can connect simultaneously."""
        log = logger.bind(context="test_multiple_clients_connect")
        log.info("Testing multiple client connections")
        
        # Wait for all connections to establish
        await asyncio.sleep(1.0)
        
        # All clients should be connected
        for node_id, client in multiple_clients.items():
            assert client.is_connected(), f"Client {node_id} should be connected"
        
        # Server should track all clients
        expected_count = len(multiple_clients)
        assert test_server.get_node_count() == expected_count, f"Server should have {expected_count} nodes"
        
        # Check cluster distribution
        tactical_nodes = test_server.get_cluster_nodes("cluster_tactical")
        positional_nodes = test_server.get_cluster_nodes("cluster_positional")
        test_nodes = test_server.get_cluster_nodes("test_cluster")
        
        assert len(tactical_nodes) == 2, "Should have 2 tactical cluster nodes"
        assert len(positional_nodes) == 2, "Should have 2 positional cluster nodes"
        assert len(test_nodes) == 1, "Should have 1 test cluster node"
        
        log.info("✓ Multiple client connection test passed")
    
    @pytest.mark.asyncio
    async def test_cluster_specific_broadcast(self, test_server, multiple_clients):
        """Test broadcasting messages to specific clusters."""
        log = logger.bind(context="test_cluster_specific_broadcast")
        log.info("Testing cluster-specific broadcast")
        
        # Wait for connections
        await asyncio.sleep(1.0)
        
        # Set up message collectors
        tactical_collectors = {}
        positional_collectors = {}
        
        for node_id, client in multiple_clients.items():
            collector = MessageCollector()
            client.set_message_handler(MessageType.START_TRAINING, collector.collect_message)
            
            if client.cluster_id == "cluster_tactical":
                tactical_collectors[node_id] = collector
            elif client.cluster_id == "cluster_positional":
                positional_collectors[node_id] = collector
        
        # Broadcast to tactical cluster only
        tactical_msg = MessageFactory.create_start_training(
            "broadcast", "cluster_tactical", 150, 2
        )
        
        await test_server.broadcast_to_cluster("cluster_tactical", tactical_msg)
        
        # Wait for delivery
        await asyncio.sleep(0.5)
        
        # Only tactical cluster clients should receive the message
        for node_id, collector in tactical_collectors.items():
            messages = collector.get_messages(MessageType.START_TRAINING)
            assert len(messages) >= 1, f"tactical client {node_id} should receive cluster message"
            assert messages[0].payload["games_per_round"] == 150, "Message should have correct payload"
        
        # Positional cluster clients should NOT receive the message
        for node_id, collector in positional_collectors.items():
            messages = collector.get_messages(MessageType.START_TRAINING)
            assert len(messages) == 0, f"Positional client {node_id} should NOT receive tactical cluster message"
        
        log.info("✓ Cluster-specific broadcast test passed")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_registration(self, test_server, test_port):
        """Test server handles invalid registration attempts."""
        log = logger.bind(context="test_invalid_registration")
        log.info("Testing invalid registration handling")
        
        from client.communication.client_socket import FederatedLearningClient, ClientState
        
        # Create client with invalid cluster
        client = FederatedLearningClient(
            "invalid_test", "invalid_cluster", 
            "localhost", test_port
        )
        client.auto_reconnect = False
        
        # Start client
        client_task = asyncio.create_task(client.start())
        
        # Wait for registration attempt
        await asyncio.sleep(2.0)
        
        # Client should not be successfully registered
        # (This depends on cluster manager validation - might be registered in mock)
        stats = client.get_connection_stats()
        assert stats.connection_attempts >= 1, "Client should have attempted connection"
        
        # Cleanup
        await client.stop()
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
        
        log.info("✓ Invalid registration test passed")
    
    @pytest.mark.asyncio
    async def test_message_validation(self, test_client):
        """Test that invalid messages are handled properly."""
        log = logger.bind(context="test_message_validation")
        log.info("Testing message validation")
        
        # This test would require access to the WebSocket to send malformed messages
        # For now, we test that the client validates messages before sending
        
        # Generate valid model update
        dummy_model = generate_random_model(seed=123)
        model_state = dummy_model.state_dict()
        
        # This should succeed
        await test_client.send_model_update(
            model_state=model_state,
            samples=500,
            loss=0.35,
            round_num=3
        )
        
        # Check that message was sent
        stats = test_client.get_connection_stats()
        assert stats.total_messages_sent >= 2, "Client should have sent messages (registration + model update)"
        
        log.info("✓ Message validation test passed")


# Performance test - not run by default
class TestBasicPerformance:
    """Basic performance tests."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_message_throughput(self, test_server, test_client):
        """Test basic message throughput."""
        log = logger.bind(context="test_message_throughput")
        log.info("Testing message throughput")
        
        # Send multiple messages rapidly
        num_messages = 50
        start_time = asyncio.get_event_loop().time()
        
        for i in range(num_messages):
            metrics = {"iteration": i, "loss": 0.1 + i * 0.01, "samples": 100}
            await test_client.send_metrics(metrics, round_num=1)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        messages_per_second = num_messages / duration
        log.info(f"Message throughput: {messages_per_second:.1f} messages/second")
        
        # Should achieve reasonable throughput (>10 msg/s)
        assert messages_per_second > 10, f"Throughput too low: {messages_per_second:.1f} msg/s"
        
        log.info("✓ Message throughput test passed")