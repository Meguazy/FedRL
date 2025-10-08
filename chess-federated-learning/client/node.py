"""
Federated learning node implementation.

This module implements the main node logic that coordinates all client-side
activities including communication, training, and state management. The node
acts as the primary orchestrator for participating in federated learning.

Key Responsibilities:
    - Manage WebSocket connection to FL server
    - Coordinate local training via trainer interface
    - Handle model updates (receive and send)
    - Report metrics to server
    - Maintain node state and lifecycle
    - Integrate with storage layer for local persistence

Architecture:
    Node uses composition to separate concerns:
    - FederatedLearningClient: Handles network communication
    - TrainerInterface: Performs actual training
    - Node: Orchestrates the workflow
"""

import asyncio
from typing import Dict, Any, Optional
from enum import Enum
from loguru import logger
import time

from .communication.client_socket import FederatedLearningClient, ClientState
from .trainer.trainer_interface import (
    TrainerInterface,
    TrainingConfig,
    TrainingResult,
    TrainingError
)
from server.communication.protocol import Message, MessageType


class NodeLifecycleState(Enum):
    """
    Enum representing the lifecycle states of the node.
    """
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    UPDATING = "updating"
    IDLE = "idle"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    
    
class FederatedLearningNode:
    """
    Main federated learning node implementation.
    
    This class orchestrates all node activities and manages the complete
    lifecycle of participating in federated learning. It coordinates:
    - Communication with the FL server
    - Local training execution
    - Model state management
    - Metrics reporting
    
    The node operates asynchronously, responding to server commands while
    managing local training in the background.
    
    Example:
        >>> config = TrainingConfig(games_per_round=100, playstyle="tactical")
        >>> trainer = DummyTrainer("agg_001", "cluster_tactical", config)
        >>> node = FederatedLearningNode(
        ...     node_id="agg_001",
        ...     cluster_id="cluster_tactical",
        ...     trainer=trainer,
        ...     server_host="localhost",
        ...     server_port=8765
        ... )
        >>> await node.start()
    """
    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        trainer: TrainerInterface,
        server_host: str = "localhost",
        server_port: int = 8765,
        auto_reconnect: bool = True,
    ):
        """
        Initialize the federated learning node.
        
        Args:
            node_id: Unique identifier for this node
            cluster_id: Cluster this node belongs to
            trainer: TrainerInterface implementation for local training
            server_host: FL server hostname or IP
            server_port: FL server port
            reconnect_delay: Delay before reconnecting on failure (seconds)
        """
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.trainer = trainer

        log = logger.bind(context=f"FLNode.{node_id}")
        log.info(f"Node {node_id} initialized in cluster {cluster_id}")
        
        # Communication client
        self.client: FederatedLearningClient = FederatedLearningClient(
            node_id=node_id,
            cluster_id=cluster_id,
            server_host=server_host,
            server_port=server_port,
        )
        self.client.auto_reconnect = auto_reconnect
        
        # Node state
        self.lifecycle_state = NodeLifecycleState.INITIALIZING
        self.current_model_state: Optional[Dict[str, Any]] = None
        self.current_round: Optional[int] = None
        self.is_running: bool = False
        
        # Training task management
        self.training_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.rounds_completed: int = 0
        self.total_training_time: float = 0.0
        self.total_samples: int = 0
        self.start_time: float = 0.0

        # Register message handler
        self._setup_message_handlers()
        
        log.info(f"Node {node_id} setup complete")
        
    def _setup_message_handlers(self):
        """
        Setup message handlers for different message types from the server.
        """
        log = logger.bind(context=f"FLNode.{self.node_id}._setup_message_handlers")
        log.debug("Setting up message handlers")
        
        self.client.set_message_handler(MessageType.START_TRAINING, self._handle_start_training)
        self.client.set_message_handler(MessageType.CLUSTER_MODEL, self._handle_cluster_model)
        self.client.set_message_handler(MessageType.ERROR, self._handle_server_error)
        self.client.set_message_handler(MessageType.REGISTER_ACK, self._handle_register_ack)
        
        log.debug("Message handlers setup complete")
        
    async def start(self):
        """
        Start the node and begin participating in federated learning.
        
        This method:
        1. Connects to the FL server
        2. Enters the main event loop
        3. Responds to server commands
        4. Manages local training
        """
        log = logger.bind(context=f"FLNode.{self.node_id}")
        log.info(f"Starting federated learning node {self.node_id}")
        
        self.is_running = True
        self.start_time = time.time()
        self.lifecycle_state = NodeLifecycleState.READY
        
        # Start connect task in background
        client_task = asyncio.create_task(self.client.start())

        # Wait for connection
        log.info("Waiting for connection to server...")
        connection_timeout = 30  # seconds
        start_wait = time.time()
        
        while not self.client.is_connected() and (time.time() - start_wait) < connection_timeout:
            await asyncio.sleep(1)
            
        if not self.client.is_connected():
            log.error("Failed to connect to server within timeout")
            self.lifecycle_state = NodeLifecycleState.ERROR
            self.is_running = False
            await self.client.stop()
            return
        
        log.info(f"Node {self.node_id} connected and ready for training")
        self.lifecycle_state = NodeLifecycleState.IDLE
        
        # Run until stopped
        try:
            while self.is_running:
                await asyncio.sleep(1)
                
                # Check for completed training task
                if self.training_task and self.training_task.done():
                    try:
                        result: TrainingResult = self.training_task.result()
                        await self._handle_training_complete(result)
                    except Exception as e:
                        log.error(f"Training task failed: {e}")
                        self.lifecycle_state = NodeLifecycleState.ERROR
                    finally:
                        self.training_task = None

        except Exception as e:
            log.error(f"Unexpected error in main loop: {e}")
            self.lifecycle_state = NodeLifecycleState.ERROR
        finally:
            await self.stop()
            
    async def stop(self):
        """
        Stop the node and clean up resources.
        
        This gracefully shuts down the node, cancelling any ongoing
        training and disconnecting from the server.
        """
        log = logger.bind(context=f"FLNode.{self.node_id}")
        log.info("Stopping federated learning node")
        
        self.is_running = False
        self.lifecycle_state = NodeLifecycleState.SHUTDOWN
        
        # Cancel training if in progress
        if self.training_task and not self.training_task.done():
            log.info("Cancelling ongoing training")
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from server
        await self.client.stop()
        
        # Log final statistics
        self._log_final_statistics()
        
        log.info(f"Node {self.node_id} stopped")
            
    async def _handle_start_training(self, message: Message):
        """
        Handle START_TRAINING message from server.

        This message indicates that the node should begin local training
        for the specified round.

        Args:
            message: Message object containing round info
        """
        log = logger.bind(context=f"FLNode.{self.node_id}._handle_start_training")
        log.info("Received START_TRAINING message")

        if self.lifecycle_state == NodeLifecycleState.TRAINING:
            log.warning("Already in training state, ignoring START_TRAINING")
            return

        games_per_round = message.payload.get("games_per_round", 100)
        round_num = message.round_num

        log.info(f"Starting training for round {round_num} with {games_per_round} games")

        self.current_round = round_num
        self.lifecycle_state = NodeLifecycleState.TRAINING

        # Update trainer config if needed
        if self.trainer.config.games_per_round != games_per_round:
            new_config = TrainingConfig(games_per_round=games_per_round)
            self.trainer.update_config(new_config)
            log.info(f"Updated trainer config: {new_config}")

        # Start training in background task
        if self.current_model_state is None:
            log.warning("No model state available, using uninitialized model")
            self.current_model_state = {}

        self.training_task = asyncio.create_task(
            self._run_training(self.current_model_state)
        )
        
        log.info(f"Training task started for round {round_num}")
        
    async def _run_training(self, initial_model_state: Dict[str, Any]) -> TrainingResult:
        """
        Execute training with the trainer.
        
        Args:
            initial_model_state: Starting model state
        
        Returns:
            TrainingResult with updated model and metrics
        """
        log = logger.bind(context=f"FLNode.{self.node_id}")
        log.info("Running local training")
        
        try:
            result: TrainingResult = await self.trainer.train(initial_model_state)
            log.info(f"Training complete: loss={result.loss}, samples={result.samples}")
            return result
        except TrainingError as e:
            log.error(f"Training failed: {e}")
            raise
        except Exception as e:
            log.exception(f"Unexpected error during training: {e}")
            raise
        
    async def _handle_training_complete(self, result: TrainingResult):
        """
        Handle completion of local training.
        
        Args:
            result: TrainingResult from trainer
        """
        log = logger.bind(context=f"FLNode.{self.node_id}")
        log.info("Handling training completion")
        
        if not result.success:
            log.error("Training was not successful, entering ERROR state")
            self.lifecycle_state = NodeLifecycleState.ERROR
            return
        
        # Update statistics
        self.rounds_completed += 1
        self.total_training_time += result.training_time
        self.total_samples += result.samples
        self.current_model_state = result.model_state
        
        # Send mode update to server
        log.info(f"Sending model update for round {self.current_round}")
        
        try:
            await self.client.send_model_update(
                model_state=self.current_model_state,
                round_num=self.current_round,
                loss=result.loss,
                samples=result.samples
            )
            
            # Also send additional metrics
            await  self.client.send_metrics(
                metrics=result.metrics,
                round_num=self.current_round
            )
            
            log.info("Model update sent successfully")
            self.lifecycle_state = NodeLifecycleState.IDLE
        except Exception as e:
            log.error(f"Failed to send model update: {e}")
            self.lifecycle_state = NodeLifecycleState.ERROR
            
    async def _handle_cluster_model(self, message: Message):
        """
        Handle CLUSTER_MODEL message from server.
        
        This message contains the latest global model state for the cluster.
        
        Args:
            message: Message object containing model state
        """
        log = logger.bind(context=f"FLNode.{self.node_id}._handle_cluster_model")
        log.info("Received CLUSTER_MODEL message")
        
        round_num = message.round_num
        model_state = message.payload.get("model_state", {})
        
        log.info(f"Updating model state for round {round_num}")
        
        # Update current model state
        self.current_model_state = model_state
        self.lifecycle_state = NodeLifecycleState.IDLE
        log.info("Model state updated successfully. Ready for next round.")
        
    async def _handle_server_error(self, message: Message):
        """
        Handle ERROR message from server.
        
        Args:
            message: ERROR message from server
        """
        log = logger.bind(context=f"FLNode.{self.node_id}")
        
        error_msg = message.payload.get('message', 'Unknown error')
        error_code = message.payload.get('error_code', 0)
        
        log.error(f"Server error [{error_code}]: {error_msg}")
        
        # Decide whether to stop or continue based on error severity
        # For now, just log it
        self.lifecycle_state = NodeLifecycleState.ERROR
        
    async def _handle_register_ack(self, message: Message):
        """
        Handle REGISTER_ACK message from server.
        
        This confirms successful registration with the server.
        
        Args:
            message: REGISTER_ACK message
        """
        log = logger.bind(context=f"FLNode.{self.node_id}._handle_register_ack")
        log.info("Received REGISTER_ACK from server")
        self.lifecycle_state = NodeLifecycleState.READY
        log.info("Node is now READY for training")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get node statistics.
        
        Returns:
            Dict containing comprehensive node statistics
        """
        uptime = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "lifecycle_state": self.lifecycle_state.value,
            "connected": self.client.is_connected(),
            "rounds_completed": self.rounds_completed,
            "current_round": self.current_round,
            "total_training_time": self.total_training_time,
            "total_samples": self.total_samples,
            "uptime": uptime,
            "trainer_stats": self.trainer.get_statistics(),
            "client_stats": self.client.get_stats(),
        }
    
    def _log_final_statistics(self):
        """Log final statistics when node stops."""
        log = logger.bind(context=f"FLNode.{self.node_id}")
        
        stats = self.get_statistics()
        
        log.info("=== Final Node Statistics ===")
        log.info(f"Node ID: {stats['node_id']}")
        log.info(f"Cluster: {stats['cluster_id']}")
        log.info(f"Rounds completed: {stats['rounds_completed']}")
        log.info(f"Total training time: {stats['total_training_time']:.1f}s")
        log.info(f"Total samples: {stats['total_samples']}")
        log.info(f"Total uptime: {stats['uptime']:.1f}s")
        log.info(f"Final state: {stats['lifecycle_state']}")


# async def main():
#     """Example usage of FederatedLearningNode."""
#     from client.trainer.trainer_interface import TrainingConfig
#     from client.trainer.trainer_dummy import DummyTrainer
    
#     # Configuration
#     node_id = "agg_001"
#     cluster_id = "cluster_tactical"
    
#     # Create trainer
#     config = TrainingConfig(
#         games_per_round=50,
#         playstyle="tactical"
#     )
#     trainer = DummyTrainer(node_id, cluster_id, config)
    
#     # Create node
#     node = FederatedLearningNode(
#         node_id=node_id,
#         cluster_id=cluster_id,
#         trainer=trainer,
#         server_host="localhost",
#         server_port=8765
#     )
    
#     # Start node
#     await node.start()


# if __name__ == "__main__":
#     asyncio.run(main())
