"""
Test utilities and helper functions for federated learning tests.

This module provides utility functions, mock objects, and helpers that are
commonly used across different test modules. It includes model generators,
message validators, and test orchestration helpers.
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from server.communication.protocol import Message, MessageType, MessageFactory


class MockModel:
    """
    Mock model for testing that simulates PyTorch-like behavior.
    
    Provides state_dict() method and other model-like functionality
    without requiring actual PyTorch installation in tests.
    """
    
    def __init__(self, layer_sizes: List[Tuple[int, int]] = None):
        """
        Initialize mock model with specified layer sizes.
        
        Args:
            layer_sizes: List of (input_size, output_size) tuples for each layer
        """
        log = logger.bind(context="MockModel.__init__")
        
        if layer_sizes is None:
            layer_sizes = [(784, 128), (128, 64), (64, 10)]  # Default sizes
        
        self.layers = {}
        for i, (in_size, out_size) in enumerate(layer_sizes):
            # Create random weights and biases (as nested lists, not tensors)
            weight = [[random.uniform(-1, 1) for _ in range(in_size)] for _ in range(out_size)]
            bias = [random.uniform(-1, 1) for _ in range(out_size)]
            
            self.layers[f"layer{i+1}.weight"] = weight
            self.layers[f"layer{i+1}.bias"] = bias
        
        log.debug(f"Created MockModel with {len(layer_sizes)} layers")
    
    def state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary (like PyTorch)."""
        log = logger.bind(context="MockModel.state_dict")
        log.trace(f"Returning state dict with {len(self.layers)} parameters")
        return self.layers.copy()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dictionary (like PyTorch)."""
        log = logger.bind(context="MockModel.load_state_dict")
        log.debug(f"Loading state dict with {len(state_dict)} parameters")
        self.layers = state_dict.copy()
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters in the model."""
        log = logger.bind(context="MockModel.get_parameter_count")
        
        total = 0
        for key, value in self.layers.items():
            if isinstance(value, list):
                if isinstance(value[0], list):  # 2D weight matrix
                    total += len(value) * len(value[0])
                else:  # 1D bias vector
                    total += len(value)
        
        log.trace(f"Model has {total} parameters")
        return total


def generate_random_model(seed: int = None) -> MockModel:
    """
    Generate a random mock model for testing.
    
    Args:
        seed: Random seed for reproducible models
    
    Returns:
        MockModel: Generated model instance
    """
    log = logger.bind(context="generate_random_model")
    
    if seed is not None:
        random.seed(seed)
        log.debug(f"Using random seed: {seed}")
    
    # Random architecture
    num_layers = random.randint(2, 5)
    layer_sizes = []
    
    input_size = random.randint(100, 1000)
    for i in range(num_layers):
        output_size = random.randint(10, 500)
        layer_sizes.append((input_size, output_size))
        input_size = output_size
    
    model = MockModel(layer_sizes)
    log.info(f"Generated random model with {num_layers} layers, {model.get_parameter_count()} parameters")
    
    return model


def generate_model_variations(base_model: MockModel, count: int, 
                            variation_factor: float = 0.1) -> List[MockModel]:
    """
    Generate variations of a base model for testing aggregation.
    
    Args:
        base_model: Base model to create variations from
        count: Number of variations to generate
        variation_factor: How much to vary the weights (0.0-1.0)
    
    Returns:
        List[MockModel]: List of model variations
    """
    log = logger.bind(context="generate_model_variations")
    log.info(f"Generating {count} model variations with factor {variation_factor}")
    
    variations = []
    base_state = base_model.state_dict()
    
    for i in range(count):
        # Create variation by adding noise to base model
        varied_state = {}
        for key, value in base_state.items():
            if isinstance(value, list):
                if isinstance(value[0], list):  # 2D weight matrix
                    varied_value = []
                    for row in value:
                        varied_row = [
                            x + random.uniform(-variation_factor, variation_factor) 
                            for x in row
                        ]
                        varied_value.append(varied_row)
                else:  # 1D bias vector
                    varied_value = [
                        x + random.uniform(-variation_factor, variation_factor) 
                        for x in value
                    ]
                varied_state[key] = varied_value
            else:
                varied_state[key] = value
        
        # Create new model with varied weights
        model = MockModel()
        model.load_state_dict(varied_state)
        variations.append(model)
        
        log.trace(f"Created variation {i+1}")
    
    log.info(f"Generated {len(variations)} model variations")
    return variations


async def wait_for_condition(condition_func: callable, timeout: float = 10.0, 
                           interval: float = 0.1) -> bool:
    """
    Wait for a condition to become true with timeout.
    
    Args:
        condition_func: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
    
    Returns:
        bool: True if condition was met, False if timeout
    """
    log = logger.bind(context="wait_for_condition")
    log.debug(f"Waiting for condition with timeout={timeout}s, interval={interval}s")
    
    start_time = time.time()
    checks = 0
    
    while (time.time() - start_time) < timeout:
        checks += 1
        try:
            if condition_func():
                elapsed = time.time() - start_time
                log.debug(f"Condition met after {elapsed:.2f}s ({checks} checks)")
                return True
        except Exception as e:
            log.warning(f"Condition check failed: {e}")
        
        await asyncio.sleep(interval)
    
    elapsed = time.time() - start_time
    log.warning(f"Condition timeout after {elapsed:.2f}s ({checks} checks)")
    return False


async def wait_for_clients_connected(clients: List, timeout: float = 15.0) -> bool:
    """
    Wait for all clients to be connected and registered.
    
    Args:
        clients: List of client instances
        timeout: Maximum time to wait
    
    Returns:
        bool: True if all clients connected, False otherwise
    """
    log = logger.bind(context="wait_for_clients_connected")
    log.info(f"Waiting for {len(clients)} clients to connect")
    
    def all_connected():
        connected_states = []
        for i, client in enumerate(clients):
            is_connected = client.is_connected()
            connected_states.append(is_connected)
            log.trace(f"Client {i}: connected={is_connected}")
        
        all_conn = all(connected_states)
        log.debug(f"All connected: {all_conn} ({sum(connected_states)}/{len(connected_states)})")
        return all_conn
    
    result = await wait_for_condition(all_connected, timeout)
    
    if result:
        log.info(f"All {len(clients)} clients connected successfully")
    else:
        connected_count = sum(1 for client in clients if client.is_connected())
        log.warning(f"Timeout: Only {connected_count}/{len(clients)} clients connected")
    
    return result


async def wait_for_server_nodes(server, expected_count: int, timeout: float = 15.0) -> bool:
    """
    Wait for server to have expected number of connected nodes.
    
    Args:
        server: Server instance
        expected_count: Expected number of connected nodes
        timeout: Maximum time to wait
    
    Returns:
        bool: True if expected nodes connected, False otherwise
    """
    log = logger.bind(context="wait_for_server_nodes")
    log.info(f"Waiting for server to have {expected_count} nodes")
    
    def expected_nodes_connected():
        current_count = server.get_node_count()
        log.debug(f"Server has {current_count}/{expected_count} nodes")
        return current_count >= expected_count
    
    result = await wait_for_condition(expected_nodes_connected, timeout)
    
    if result:
        log.info(f"Server has expected {expected_count} nodes connected")
    else:
        current_count = server.get_node_count()
        log.warning(f"Timeout: Server has only {current_count}/{expected_count} nodes")
    
    return result


def validate_message_structure(message: Message, expected_type: MessageType, 
                              required_payload_keys: List[str] = None) -> bool:
    """
    Validate message structure and content.
    
    Args:
        message: Message to validate
        expected_type: Expected message type
        required_payload_keys: Keys that must be present in payload
    
    Returns:
        bool: True if message is valid
    """
    log = logger.bind(context="validate_message_structure")
    log.debug(f"Validating {message.type} message")
    
    # Check basic message structure
    if not message.validate():
        log.warning("Message failed basic validation")
        return False
    
    # Check message type
    if message.type != expected_type.value:
        log.warning(f"Expected {expected_type}, got {message.type}")
        return False
    
    # Check required payload keys
    if required_payload_keys:
        missing_keys = []
        for key in required_payload_keys:
            if key not in message.payload:
                missing_keys.append(key)
        
        if missing_keys:
            log.warning(f"Missing required payload keys: {missing_keys}")
            return False
    
    log.debug(f"Message validation passed for {expected_type}")
    return True


async def simulate_training_round(clients: Dict[str, Any], server, round_num: int = 1) -> bool:
    """
    Simulate a complete training round with multiple clients.
    
    Args:
        clients: Dictionary of client_id -> client instance
        server: Server instance
        round_num: Training round number
    
    Returns:
        bool: True if simulation completed successfully
    """
    log = logger.bind(context="simulate_training_round")
    log.info(f"Simulating training round {round_num} with {len(clients)} clients")
    
    try:
        # Step 1: Server broadcasts START_TRAINING
        log.info("Step 1: Broadcasting START_TRAINING")
        start_msg = MessageFactory.create_start_training(
            "broadcast", "all_clusters", 100, round_num
        )
        await server.broadcast_to_all(start_msg)
        log.info("Broadcasted START_TRAINING message")
        
        # Step 2: Wait for clients to receive training command
        log.info("Step 2: Waiting for message delivery")
        await asyncio.sleep(0.5)
        
        # Step 3: Simulate clients sending model updates
        log.info("Step 3: Simulating model updates from clients")
        update_tasks = []
        
        for client_id, client in clients.items():
            # Generate dummy model and metrics
            dummy_model = generate_random_model(seed=hash(client_id))
            model_state = dummy_model.state_dict()
            
            # Create update task
            update_task = client.send_model_update(
                model_state=model_state,
                samples=random.randint(800, 1200),
                loss=random.uniform(0.1, 0.5),
                round_num=round_num
            )
            update_tasks.append(update_task)
        
        # Wait for all updates to complete
        await asyncio.gather(*update_tasks)
        log.info("All clients sent model updates")
        
        # Step 4: Wait for server to process
        log.info("Step 4: Waiting for server processing")
        await asyncio.sleep(1.0)
        
        log.info(f"Training round {round_num} simulation completed successfully")
        return True
        
    except Exception as e:
        log.error(f"Training round simulation failed: {e}")
        return False


def create_test_messages() -> Dict[str, Message]:
    """
    Create a set of test messages for validation.
    
    Returns:
        Dict[str, Message]: Dictionary of message_name -> message instance
    """
    log = logger.bind(context="create_test_messages")
    log.info("Creating test message set")
    
    messages = {}
    
    # Registration message
    messages["register"] = MessageFactory.create_register_message(
        "test_node", "test_cluster"
    )
    
    # Model update message
    dummy_model = generate_random_model(seed=42)
    messages["model_update"] = MessageFactory.create_model_update(
        "test_node", "test_cluster", dummy_model.state_dict(), 1000, 0.25, 1
    )
    
    # Start training message
    messages["start_training"] = MessageFactory.create_start_training(
        "test_node", "test_cluster", 100, 1
    )
    
    # Cluster model message
    messages["cluster_model"] = MessageFactory.create_cluster_model(
        "test_node", "test_cluster", dummy_model.state_dict(), 1
    )
    
    # Metrics message
    messages["metrics"] = MessageFactory.create_metrics(
        "test_node", "test_cluster", 
        {"loss": 0.25, "samples": 1000, "accuracy": 0.85}, 1
    )
    
    # Error message
    messages["error"] = MessageFactory.create_error(
        "test_node", "test_cluster", "Test error message"
    )
    
    # Heartbeat message
    messages["heartbeat"] = MessageFactory.create_heartbeat(
        "test_node", "test_cluster"
    )
    
    log.info(f"Created {len(messages)} test messages")
    return messages


class MessageCollector:
    """
    Utility class to collect messages during tests.
    
    Can be used as a message handler to capture messages sent to clients
    or received by the server for later verification.
    """
    
    def __init__(self):
        """Initialize empty message collector."""
        log = logger.bind(context="MessageCollector.__init__")
        
        self.messages: List[Message] = []
        self.messages_by_type: Dict[str, List[Message]] = {}
        
        log.debug("Created new MessageCollector")
    
    async def collect_message(self, message: Message):
        """
        Collect a message (can be used as message handler).
        
        Args:
            message: Message to collect
        """
        log = logger.bind(context="MessageCollector.collect_message")
        log.trace(f"Collecting message: {message.type}")
        
        self.messages.append(message)
        
        msg_type = message.type
        if msg_type not in self.messages_by_type:
            self.messages_by_type[msg_type] = []
        self.messages_by_type[msg_type].append(message)
        
        log.debug(f"Collected {message.type}, total: {len(self.messages)}")
    
    def get_messages(self, message_type: MessageType = None) -> List[Message]:
        """
        Get collected messages, optionally filtered by type.
        
        Args:
            message_type: Optional message type filter
        
        Returns:
            List[Message]: Collected messages
        """
        log = logger.bind(context="MessageCollector.get_messages")
        
        if message_type is None:
            result = self.messages.copy()
            log.debug(f"Returning all {len(result)} messages")
        else:
            result = self.messages_by_type.get(message_type.value, []).copy()
            log.debug(f"Returning {len(result)} messages of type {message_type}")
        
        return result
    
    def clear(self):
        """Clear all collected messages."""
        log = logger.bind(context="MessageCollector.clear")
        
        old_count = len(self.messages)
        self.messages.clear()
        self.messages_by_type.clear()
        
        log.debug(f"Cleared {old_count} messages")
    
    def count(self, message_type: MessageType = None) -> int:
        """
        Count messages, optionally by type.
        
        Args:
            message_type: Optional message type filter
        
        Returns:
            int: Number of messages
        """
        if message_type is None:
            return len(self.messages)
        else:
            return len(self.messages_by_type.get(message_type.value, []))


def assert_message_valid(message: Message, expected_type: MessageType,
                        expected_node_id: str = None, expected_cluster_id: str = None):
    """
    Assert that a message has the expected structure and content.
    
    Args:
        message: Message to validate
        expected_type: Expected message type
        expected_node_id: Expected node ID (optional)
        expected_cluster_id: Expected cluster ID (optional)
    
    Raises:
        AssertionError: If message doesn't match expectations
    """
    log = logger.bind(context="assert_message_valid")
    log.debug(f"Asserting message validity: {message.type}")
    
    assert message.validate(), "Message failed basic validation"
    assert message.type == expected_type.value, f"Expected {expected_type}, got {message.type}"
    
    if expected_node_id:
        assert message.node_id == expected_node_id, f"Expected node_id {expected_node_id}, got {message.node_id}"
    
    if expected_cluster_id:
        assert message.cluster_id == expected_cluster_id, f"Expected cluster_id {expected_cluster_id}, got {message.cluster_id}"
    
    log.debug("Message assertion passed")


def generate_test_metrics(node_id: str, round_num: int) -> Dict[str, Any]:
    """
    Generate realistic test metrics for a training round.
    
    Args:
        node_id: Node identifier (affects seed for consistency)
        round_num: Training round number
    
    Returns:
        Dict: Training metrics
    """
    log = logger.bind(context="generate_test_metrics")
    
    # Use node_id and round for reproducible metrics
    seed = hash(f"{node_id}_{round_num}") % 1000000
    random.seed(seed)
    
    metrics = {
        "loss": round(random.uniform(0.1, 0.8), 4),
        "samples": random.randint(500, 1500),
        "accuracy": round(random.uniform(0.6, 0.95), 3),
        "training_time": round(random.uniform(30.0, 120.0), 2),
        "games_played": random.randint(80, 120),
        "elo_estimate": random.randint(1000, 2000)
    }
    
    log.debug(f"Generated metrics for {node_id} round {round_num}: {metrics}")
    return metrics


async def cleanup_test_resources(*resources):
    """
    Clean up test resources (servers, clients, etc.).
    
    Args:
        *resources: Resources with stop() or close() methods
    """
    log = logger.bind(context="cleanup_test_resources")
    log.info(f"Cleaning up {len(resources)} test resources")
    
    cleanup_tasks = []
    
    for resource in resources:
        if hasattr(resource, 'stop'):
            cleanup_tasks.append(asyncio.create_task(resource.stop()))
        elif hasattr(resource, 'close'):
            if asyncio.iscoroutinefunction(resource.close):
                cleanup_tasks.append(asyncio.create_task(resource.close()))
            else:
                resource.close()
    
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        log.info("All async cleanup tasks completed")
    
    log.info("Test resource cleanup complete")