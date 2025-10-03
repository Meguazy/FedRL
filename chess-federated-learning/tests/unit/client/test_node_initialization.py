"""
Unit tests for FederatedLearningNode initialization.

Tests verify that the node is correctly initialized with all required
components and proper default states.
"""

import pytest
from unittest.mock import MagicMock, patch

from client.node import FederatedLearningNode, NodeLifecycleState
from client.trainer.trainer_interface import TrainerInterface, TrainingConfig
from server.communication.protocol import MessageType


class MockTrainer(TrainerInterface):
    """Mock trainer for testing."""

    async def train(self, initial_model_state):
        return MagicMock(
            model_state={"mock": "state"},
            samples=100,
            loss=0.5,
            games_played=10,
            training_time=1.0,
            success=True
        )

    async def evaluate(self, model_state, num_games=10):
        return {"win_rate": 0.5, "avg_loss": 0.4}


@pytest.fixture
def training_config():
    """Create a training configuration for testing."""
    return TrainingConfig(
        games_per_round=100,
        batch_size=32,
        learning_rate=0.001,
        playstyle="aggressive"
    )


@pytest.fixture
def mock_trainer(training_config):
    """Create a mock trainer instance."""
    return MockTrainer(
        node_id="test_node_001",
        cluster_id="test_cluster",
        config=training_config
    )


def test_node_initialization_basic(mock_trainer):
    """Test basic node initialization with required parameters."""
    node = FederatedLearningNode(
        node_id="test_node_001",
        cluster_id="test_cluster",
        trainer=mock_trainer,
        server_host="localhost",
        server_port=8765
    )

    # Verify node identity
    assert node.node_id == "test_node_001"
    assert node.cluster_id == "test_cluster"
    assert node.trainer is mock_trainer

    # Verify lifecycle state
    assert node.lifecycle_state == NodeLifecycleState.INITIALIZING
    assert node.is_running is False

    # Verify model state
    assert node.current_model_state is None
    assert node.current_round is None

    # Verify training task management
    assert node.training_task is None

    # Verify statistics initialization
    assert node.rounds_completed == 0
    assert node.total_training_time == 0.0
    assert node.total_samples == 0
    assert node.start_time == 0.0


def test_node_initialization_with_custom_server(mock_trainer):
    """Test node initialization with custom server settings."""
    node = FederatedLearningNode(
        node_id="custom_node",
        cluster_id="custom_cluster",
        trainer=mock_trainer,
        server_host="192.168.1.100",
        server_port=9999
    )

    assert node.node_id == "custom_node"
    assert node.cluster_id == "custom_cluster"
    assert node.client.server_host == "192.168.1.100"
    assert node.client.server_port == 9999


def test_node_initialization_auto_reconnect_default(mock_trainer):
    """Test that auto_reconnect defaults to True."""
    node = FederatedLearningNode(
        node_id="test_node",
        cluster_id="test_cluster",
        trainer=mock_trainer
    )

    assert node.client.auto_reconnect is True


def test_node_initialization_auto_reconnect_disabled(mock_trainer):
    """Test node initialization with auto_reconnect disabled."""
    node = FederatedLearningNode(
        node_id="test_node",
        cluster_id="test_cluster",
        trainer=mock_trainer,
        auto_reconnect=False
    )

    assert node.client.auto_reconnect is False


def test_node_creates_communication_client(mock_trainer):
    """Test that node creates a communication client with correct parameters."""
    node = FederatedLearningNode(
        node_id="comm_test_node",
        cluster_id="comm_cluster",
        trainer=mock_trainer,
        server_host="testhost",
        server_port=7777
    )

    # Verify client is created
    assert node.client is not None

    # Verify client has correct configuration
    assert node.client.node_id == "comm_test_node"
    assert node.client.cluster_id == "comm_cluster"
    assert node.client.server_host == "testhost"
    assert node.client.server_port == 7777


def test_node_trainer_association(mock_trainer):
    """Test that trainer is correctly associated with the node."""
    node = FederatedLearningNode(
        node_id="trainer_node",
        cluster_id="trainer_cluster",
        trainer=mock_trainer
    )

    # Verify trainer reference
    assert node.trainer is mock_trainer
    assert node.trainer.node_id == "test_node_001"  # From fixture
    assert node.trainer.cluster_id == "test_cluster"  # From fixture


def test_node_statistics_counters_initialized(mock_trainer):
    """Test that all statistics counters are properly initialized to zero."""
    node: FederatedLearningNode = FederatedLearningNode(
        node_id="stats_node",
        cluster_id="stats_cluster",
        trainer=mock_trainer
    )

    assert node.rounds_completed == 0
    assert node.total_training_time == 0.0
    assert node.total_samples == 0
    assert node.start_time == 0.0


def test_node_initial_state_values(mock_trainer):
    """Test that node state values are correctly initialized."""
    node = FederatedLearningNode(
        node_id="state_node",
        cluster_id="state_cluster",
        trainer=mock_trainer
    )

    # Lifecycle and running state
    assert node.lifecycle_state == NodeLifecycleState.INITIALIZING
    assert node.is_running is False

    # Model and round state
    assert node.current_model_state is None
    assert node.current_round is None

    # Task management
    assert node.training_task is None


def test_node_default_server_values(mock_trainer):
    """Test that default server host and port are used when not specified."""
    node = FederatedLearningNode(
        node_id="default_node",
        cluster_id="default_cluster",
        trainer=mock_trainer
    )

    assert node.client.server_host == "localhost"
    assert node.client.server_port == 8765


def test_multiple_nodes_independent(training_config):
    """Test that multiple node instances are independent."""
    trainer1 = MockTrainer("node_1", "cluster_1", training_config)
    trainer2 = MockTrainer("node_2", "cluster_2", training_config)

    node1 = FederatedLearningNode(
        node_id="node_1",
        cluster_id="cluster_1",
        trainer=trainer1
    )

    node2 = FederatedLearningNode(
        node_id="node_2",
        cluster_id="cluster_2",
        trainer=trainer2
    )

    # Verify independence
    assert node1.node_id != node2.node_id
    assert node1.cluster_id != node2.cluster_id
    assert node1.trainer is not node2.trainer
    assert node1.client is not node2.client


def test_node_initialization_with_different_clusters(training_config):
    """Test nodes can be initialized in different clusters."""
    clusters = ["aggressive", "positional", "defensive"]

    for i, cluster in enumerate(clusters):
        trainer = MockTrainer(f"node_{i}", cluster, training_config)
        node = FederatedLearningNode(
            node_id=f"node_{i}",
            cluster_id=cluster,
            trainer=trainer
        )

        assert node.cluster_id == cluster
        assert node.node_id == f"node_{i}"


def test_node_message_handlers_registered(mock_trainer):
    """Test that message handlers are registered during initialization."""
    node = FederatedLearningNode(
        node_id="handler_node",
        cluster_id="handler_cluster",
        trainer=mock_trainer
    )

    # Verify handlers are registered for expected message types
    expected_handlers = [
        MessageType.START_TRAINING,
        MessageType.CLUSTER_MODEL,
        MessageType.ERROR,
        MessageType.REGISTER_ACK,
    ]

    for message_type in expected_handlers:
        assert message_type in node.client.message_handlers, \
            f"Handler for {message_type} not registered"
        assert node.client.message_handlers[message_type] is not None, \
            f"Handler for {message_type} is None"


def test_node_handler_methods_exist(mock_trainer):
    """Test that handler methods exist on the node instance."""
    node = FederatedLearningNode(
        node_id="method_node",
        cluster_id="method_cluster",
        trainer=mock_trainer
    )

    # Verify handler methods exist
    expected_methods = [
        "_handle_start_training",
        "_handle_cluster_model",
        "_handle_server_error",
        "_handle_register_ack",
    ]

    for method_name in expected_methods:
        assert hasattr(node, method_name), \
            f"Node missing handler method: {method_name}"
        assert callable(getattr(node, method_name)), \
            f"Handler {method_name} is not callable"


def test_node_handlers_bound_correctly(mock_trainer):
    """Test that handlers are bound to the correct node instance."""
    node = FederatedLearningNode(
        node_id="binding_node",
        cluster_id="binding_cluster",
        trainer=mock_trainer
    )

    # Get a handler from the client
    handler = node.client.message_handlers.get(MessageType.START_TRAINING)

    # Verify it's bound to the node instance
    assert handler is not None
    # The handler should be a method bound to this specific node instance
    if hasattr(handler, '__self__'):
        assert handler.__self__ is node, \
            "Handler not bound to correct node instance"


def test_node_setup_message_handlers_called(mock_trainer):
    """Test that _setup_message_handlers is called during initialization."""
    with patch.object(FederatedLearningNode, '_setup_message_handlers') as mock_setup:
        node = FederatedLearningNode(
            node_id="setup_node",
            cluster_id="setup_cluster",
            trainer=mock_trainer
        )

        # Verify _setup_message_handlers was called during __init__
        mock_setup.assert_called_once()


def test_node_all_required_handlers_present(mock_trainer):
    """Test that all required message handlers are present after initialization."""
    node = FederatedLearningNode(
        node_id="complete_node",
        cluster_id="complete_cluster",
        trainer=mock_trainer
    )

    # Verify we have at least the minimum required handlers
    assert len(node.client.message_handlers) >= 4, \
        f"Expected at least 4 handlers, got {len(node.client.message_handlers)}"

    # Verify no handlers are None
    for msg_type, handler in node.client.message_handlers.items():
        assert handler is not None, \
            f"Handler for {msg_type} is None"
