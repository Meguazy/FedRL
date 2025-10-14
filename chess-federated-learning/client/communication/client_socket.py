"""
WebSocket client for federated learning nodes.

This module implements the client-side communication for federated learning nodes.
It manages connection to the server, handles message routing, and provides a clean
interface for the node training logic.

Key Components:
    - FederatedLearningClient: Main WebSocket client class
    - Automatic reconnection and connection management
    - Message sending and receiving with proper error handling
    - Integration with node training workflow
    - Heartbeat mechanism for connection health

Architecture:
    - Async WebSocket client using websockets library
    - Event-driven message handling
    - Robust reconnection logic with exponential backoff
    - Thread-safe message queuing and callbacks
"""

import asyncio
import json
import time
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from enum import Enum
import websockets
from loguru import logger

from server.communication.protocol import Message, MessageType, MessageFactory


class ClientState(Enum):
    """States that the client can be in."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    REGISTERING = "registering"
    REGISTERED = "registered"
    TRAINING = "training"
    UPLOADING = "uploading"
    ERROR = "error"
    
    
@dataclass
class ConnectionStats:
    """Statistics about the connection."""
    connection_attempts: int = 0
    successful_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    last_connection_time: float = 0.0
    total_uptime: float = 0.0
    reconnections: int = 0
    ping_failures: int = 0
    keepalive_timeouts: int = 0
    last_ping_time: float = 0.0
    message_size_errors: int = 0
    large_message_warnings: int = 0
    
    
class FederatedLearningClient:
    """
    WebSocket client for federated learning nodes.
    
    This client handles all communication with the federated learning server,
    including registration, training coordination, model updates, and heartbeats.
    
    The client operates asynchronously and provides callbacks for different
    events, allowing the node training logic to respond appropriately.
    
    Typical workflow:
    1. Connect to server and register with node/cluster ID
    2. Wait for training commands from server
    3. Perform local training when requested
    4. Send model updates back to server
    5. Receive aggregated models and continue training
    """
    
    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        server_host: str = "localhost",
        server_port: int = 8765,
    ):
        """
        Initialize the federated learning client.
        
        Args:
            node_id: Unique identifier for this node (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_tactical")
            server_host: FL server hostname (default: localhost)
            server_port: FL server port (default: 8765)
        """
        log = logger.bind(context="FederatedLearningClient.__init__")
        log.info("Initializing FederatedLearningClient")
        
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{self.server_host}:{self.server_port}"

        # Connection state and stats
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ClientState.DISCONNECTED
        self.is_running = False
        
        # Reconnection settings
        self.auto_reconnect = True
        self.reconnect_delay = 10  # seconds
        self.max_reconnect_delay = 300  # seconds
        self.reconnect_backoff = 1.5  # exponential backoff factor
        
        # Heartbeat settings
        self.heartbeat_interval = 45  # seconds - increased to reduce server load
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Stats
        self.stats = ConnectionStats()
        
        # Current training state
        self.current_round: Optional[int] = None
        self.training_in_progress = False
        
        log.info(f"Initialized client for node {self.node_id} in cluster {self.cluster_id}")
        log.info(f"Will connect to server URL: {self.server_url}")
        
    def set_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a handler for specific message types.
        
        This allows the node logic to handle specific messages like
        START_TRAINING, CLUSTER_MODEL, etc.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        log = logger.bind(context="FederatedLearningClient.set_message_handler")
        log.info(f"Registered handler for {message_type}")
        self.message_handlers[message_type] = handler
        
    async def start(self):
        """
        Start the client and connect to the server.
        
        This method initiates the connection process and starts
        the main event loop for handling messages.
        """
        log = logger.bind(context="FederatedLearningClient.start")
        log.info("Starting client")
        
        self.is_running = True
        self.state = ClientState.DISCONNECTED
        
        # Main client loop with reconnection logic
        while self.is_running:
            try:
                await self._connect_and_run()
            except Exception as e:
                log.error(f"Connection error: {e}")
                
                if self.auto_reconnect and self.is_running:
                    log.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    
                    # Exponential backoff for reconnection delay
                    self.reconnect_delay = min(
                        self.reconnect_delay * self.reconnect_backoff,
                        self.max_reconnect_delay
                    )
                    
                    self.stats.reconnections += 1
                else:
                    break

        log.info(f"FL client {self.node_id} stopped")
        
    async def stop(self):
        """Stop the client and close the connection."""
        log = logger.bind(context="FederatedLearningClient.stop")
        log.info("Stopping client")
        
        self.is_running = False
        
        # Cancel heartbeat task if running
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            
        # Send disconnect message
        if self.websocket and self.state in [ClientState.CONNECTED, ClientState.REGISTERED]:
            try:
                disconnect_msg = MessageFactory.create_disconnect(
                    node_id=self.node_id,
                    cluster_id=self.cluster_id,
                    reason="Client shutting down"
                )
                await self._send_message(disconnect_msg)
            except Exception as e:
                log.warning(f"Error sending disconnect message: {e}")
                
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self._log_final_stats()
        
    async def _connect_and_run(self):
        """
        Connect to server and run main message loop.
        
        This method handles the full connection lifecycle:
        1. Connect to WebSocket server
        2. Register with the server
        3. Start heartbeat mechanism
        4. Handle incoming messages
        """
        log = logger.bind(context="FederatedLearningClient._connect_and_run")
        
        # Update connection stats
        self.stats.connection_attempts += 1
        connection_start = time.time()
        
        # Connect to server
        log.info(f"Establishing WebSocket connection at {self.server_url}")
        self.state = ClientState.CONNECTING
        
        try:
            self.websocket = await websockets.connect(
                uri=self.server_url,
                ping_interval=60,  # Increased to 60 seconds for server load
                ping_timeout=45,   # Increased to 45 seconds for better tolerance
                max_size=500 * 1024 * 1024,  # 500 MB for large model updates
                close_timeout=20,  # Increased close timeout
            )
            
            self.state = ClientState.CONNECTED
            self.stats.successful_connections += 1
            self.stats.last_connection_time = time.time()
            
            # Reset reconnection delay on successful connection
            self.reconnect_delay = 1.0
            
            log.info("WebSocket connection established")
            
            # Start message loop and registration concurrently
            message_task = asyncio.create_task(self._message_loop())
            registration_task = asyncio.create_task(self._register_with_server())

            # Wait for registration to complete
            await registration_task

            # Start heartbeat task after successful registration
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Continue handling messages
            await message_task
            
        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"Connection closed by server: {e}")
            
            # Check if this is a keepalive ping timeout
            if "keepalive ping timeout" in str(e):
                log.warning("Connection closed due to keepalive ping timeout - network latency or server load")
                self.stats.keepalive_timeouts += 1
                # Use shorter delay for ping timeout reconnections
                self.reconnect_delay = max(self.reconnect_delay / 2, 5.0)
                log.info(f"Adjusting reconnect delay to {self.reconnect_delay}s for ping timeout recovery")
            # Check if this is a message too big error
            elif "message too big" in str(e):
                log.error("Connection closed due to message size limit exceeded")
                log.error("Model update message is too large - consider implementing compression")
                self.stats.message_size_errors += 1
                # Use normal reconnect delay for size issues
                self.reconnect_delay = min(self.reconnect_delay * 1.2, self.max_reconnect_delay)
            
            self.state = ClientState.DISCONNECTED
            # Don't continue trying if auto_reconnect is disabled
            if not self.auto_reconnect:
                self.is_running = False
        except websockets.exceptions.InvalidURI:
            log.error(f"Invalid server URL: {self.server_url}")
            self.state = ClientState.ERROR
            raise
        except OSError as e:
            log.error(f"Network error connecting to server: {e}")
            self.state = ClientState.DISCONNECTED
            # Don't continue trying if auto_reconnect is disabled
            if not self.auto_reconnect:
                self.is_running = False
        except Exception as e:
            log.error(f"Unexpected error during connection: {e}")
            self.state = ClientState.ERROR
            raise
        finally:
            # Update uptime stats
            if connection_start:
                self.stats.total_uptime += time.time() - connection_start
            
            # Cancel heartbeat if running
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up WebSocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
    async def _register_with_server(self):
        """
        Register this node with the federated learning server.
        
        Sends registration message and waits for acknowledgment.
        """
        log = logger.bind(context="FederatedLearningClient._register_with_server")
        log.info(f"Registering node {self.node_id} with cluster {self.cluster_id}")
        
        self.state = ClientState.REGISTERING
        
        # Send registration message
        register_msg = MessageFactory.create_register_message(
            node_id=self.node_id,
            cluster_id=self.cluster_id
        )
        await self._send_message(register_msg)
        
        # Wait for registration response
        try:
            response: Message = await self._wait_for_message_type(MessageType.REGISTER_ACK, timeout=30.0)

            if response.payload.get('success', False):
                self.state = ClientState.REGISTERED
                log.info(f"Node {self.node_id} successfully registered")
            else:
                error_msg = response.payload.get('error_msg', 'Unknown error')
                log.error(f"Registration failed: {error_msg}")
                self.state = ClientState.ERROR
                raise Exception(f"Registration failed: {error_msg}")
            
        except asyncio.TimeoutError:
            log.error("Registration timed out")
            self.state = ClientState.ERROR
            raise

    async def _message_loop(self):
        """
        Main loop for receiving and handling messages from the server.
        
        This method continuously listens for incoming messages and dispatches
        them to the appropriate handlers based on message type.
        """
        log = logger.bind(context="FederatedLearningClient._message_loop")
        log.info("Starting message loop")
        
        try:
            async for raw_msg in self.websocket:
                self.stats.total_messages_received += 1
                
                try:
                    # Parse incoming message
                    message = Message.from_json(raw_msg)
                    log.debug(f"Received message: {message.type} - {message.payload}")
                    
                    # Validate message
                    if not message.validate():
                        log.warning(f"Invalid message received: {message}")
                        continue
                    
                    # Route message to handler
                    await self._handle_message(message)
                
                except json.JSONDecodeError:
                    log.error(f"Failed to decode message: {raw_msg}")
                except Exception as e:
                    log.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed as e:
            log.error(f"Message loop ended. WebSocket connection closed: {e}")
            
            # Check if this is a keepalive ping timeout
            if "keepalive ping timeout" in str(e):
                log.warning("Message loop ended due to keepalive ping timeout")
                self.stats.keepalive_timeouts += 1
                # This will trigger reconnection logic in the main loop
            # Check if this is a message too big error
            elif "message too big" in str(e):
                log.error("Message loop ended due to message size limit exceeded")
                log.error("Consider implementing message compression or chunking for large model updates")
                self.stats.message_size_errors += 1
            
        except Exception as e:
            log.error(f"Unexpected error in message loop: {e}")
            raise
        
    async def _handle_message(self, message: Message):
        """
        Handle incoming message from server.
        
        Routes messages to appropriate handlers or resolves pending responses.
        
        Args:
            message: Parsed message from server
        """
        log = logger.bind(context="FederatedLearningClient._handle_message")
        message_type = MessageType(message.type)
        
        # Handle built-in message types
        handled_builtin = False
        if message_type == MessageType.REGISTER_ACK:
            await self._handle_register_ack(message)
            handled_builtin = True
        elif message_type == MessageType.START_TRAINING:
            await self._handle_start_training(message)
            handled_builtin = True
        elif message_type == MessageType.CLUSTER_MODEL:
            await self._handle_cluster_model(message)
            handled_builtin = True
        elif message_type == MessageType.ERROR:
            await self._handle_error(message)
            handled_builtin = True
        elif message_type == MessageType.DISCONNECT:
            await self._handle_disconnect(message)
            handled_builtin = True

        # Always check for external handlers (in addition to built-in ones)
        if message_type in self.message_handlers:
            log.debug(f"Routing {message_type} to external handler")
            try:
                await self.message_handlers[message_type](message)
            except Exception as e:
                log.error(f"External handler failed for {message_type}: {e}")
        elif not handled_builtin:
            log.warning(f"No handler for message type: {message_type}")
                
        # Check for pending response waiters
        response_key = f"{message_type.value}_{message.round_num}"
        pending_key = f"{message_type.value}_pending"

        # Check both possible keys (round-specific and general pending)
        if response_key in self.pending_responses:
            future = self.pending_responses.pop(response_key)
            if not future.done():
                future.set_result(message)
        elif pending_key in self.pending_responses:
            future = self.pending_responses.pop(pending_key)
            if not future.done():
                future.set_result(message)
                
    async def _handle_register_ack(self, message: Message):
        """Handle registration acknowledgment from server."""
        log = logger.bind(context="FederatedLearningClient._handle_register_ack")
        
        success = message.payload.get('success', False)
        msg = message.payload.get('message', '')
        
        if success:
            log.info(f"Registration acknowledged: {msg}")
        else:
            log.error(f"Registration rejected: {msg}")
    
    async def _handle_start_training(self, message: Message):
        """Handle start training command from server."""
        log = logger.bind(context="FederatedLearningClient._handle_start_training")
        
        games_per_round = message.payload.get('games_per_round', 100)
        self.current_round = message.round_num
        
        log.info(f"Training started for round {self.current_round}, {games_per_round} games")
        self.state = ClientState.TRAINING
        self.training_in_progress = True
        
        # TODO: This would typically trigger the actual training logic
        # For now, just log the training command
    
    async def _handle_cluster_model(self, message: Message):
        """Handle cluster model update from server."""
        log = logger.bind(context="FederatedLearningClient._handle_cluster_model")
        
        round_num = message.round_num
        log.info(f"Received cluster model for round {round_num}")
        
        # TODO: This would typically load the new model weights
        # For now, just log the model update
        self.state = ClientState.REGISTERED
    
    async def _handle_error(self, message: Message):
        """Handle error message from server."""
        log = logger.bind(context="FederatedLearningClient._handle_error")

        error_msg = message.payload.get('message', 'Unknown error')
        log.error(f"Server error: {error_msg}")

        # TODO: Could trigger error recovery logic here

    async def _handle_disconnect(self, message: Message):
        """Handle disconnect message from server."""
        log = logger.bind(context="FederatedLearningClient._handle_disconnect")

        reason = message.payload.get('reason', 'Server disconnect')
        log.info(f"Server requested disconnect: {reason}")

        # Disable auto-reconnect when server initiates disconnect
        self.auto_reconnect = False
        self.state = ClientState.DISCONNECTED
        
    async def _heartbeat_loop(self):
        """
        Send periodic heartbeat messages to server.
        
        Keeps the connection alive and allows server to detect
        disconnected clients.
        """
        log = logger.bind(context="FederatedLearningClient._heartbeat_loop")
        log.debug(f"Starting heartbeat loop (interval={self.heartbeat_interval}s)")
        
        try:
            while self.is_running and self.state == ClientState.REGISTERED:
                if self.websocket and self.websocket.close_code is None:
                    heartbeat_msg = MessageFactory.create_heartbeat(
                        node_id=self.node_id,
                        cluster_id=self.cluster_id
                    )
                    try:
                        await self._send_message(heartbeat_msg)
                        self.stats.last_ping_time = time.time()
                        log.trace("Sent heartbeat")
                    except Exception as e:
                        log.error(f"Failed to send heartbeat: {e}")
                        self.stats.ping_failures += 1
                        break
                else:
                    log.warning("WebSocket closed, stopping heartbeat")
                    break

                # Sleep after sending heartbeat (check interval dynamically)
                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            log.debug("Heartbeat loop cancelled")
        except Exception as e:
            log.error(f"Unexpected error in heartbeat loop: {e}")
            raise
        
    async def _send_message(self, message: Message):
        """
        Send message to server.
        
        Args:
            message: Message to send
        """
        log = logger.bind(context="FederatedLearningClient._send_message")
        log.info(f"Sending message: {message.type}")
        
        if not self.websocket or self.websocket.close_code is not None:
            log.error("Cannot send message, WebSocket is not connected")
            raise ConnectionError("WebSocket is not connected")
        
        try:
            message_json = message.to_json()
            message_size = len(message_json.encode('utf-8'))
            
            # Log message size for monitoring
            if message_size > 100 * 1024 * 1024:  # 100MB warning
                log.warning(f"Large message detected: {message_size / (1024*1024):.1f}MB for {message.type}")
                self.stats.large_message_warnings += 1
            elif message_size > 10 * 1024 * 1024:  # 10MB info
                log.info(f"Message size: {message_size / (1024*1024):.1f}MB for {message.type}")
            
            await self.websocket.send(message_json)
            self.stats.total_messages_sent += 1
            log.trace(f"Message sent successfully (size: {message_size / 1024:.1f}KB)")
        except Exception as e:
            log.error(f"Error sending message: {e}")
            raise
        
    async def _wait_for_message_type(self, message_type: MessageType, 
                                   timeout: float = 30.0) -> Message:
        """
        Wait for a specific message type from server.
        
        Args:
            message_type: Type of message to wait for
            timeout: Maximum time to wait in seconds
        
        Returns:
            Message of the requested type
        
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        log = logger.bind(context="FederatedLearningClient._wait_for_message_type")
        log.debug(f"Waiting for {message_type} (timeout={timeout}s)")
        
        # Create future for this message type
        response_key = f"{message_type.value}_pending"
        future = asyncio.Future()
        self.pending_responses[response_key] = future
        
        try:
            # Wait for response with timeout
            message = await asyncio.wait_for(future, timeout=timeout)
            log.debug(f"Received expected {message_type}")
            return message
        except asyncio.TimeoutError:
            # Clean up future on timeout
            self.pending_responses.pop(response_key, None)
            log.warning(f"Timeout waiting for {message_type}")
            raise
    
    async def send_model_update(self, model_state: Dict, samples: int, 
                              loss: float, round_num: int):
        """
        Send model update to server after local training.
        
        Args:
            model_state: Trained model weights (raw state_dict with tensors)
            samples: Number of training samples used
            loss: Training loss achieved
            round_num: Current training round
        """
        log = logger.bind(context="FederatedLearningClient.send_model_update")
        log.info(f"Sending model update for round {round_num}: {samples} samples, loss={loss:.4f}")
        
        self.state = ClientState.UPLOADING
        
        try:
            # Check if model_state is already packaged (has 'serialized_data' key)
            # If so, use it directly. Otherwise, serialize it.
            if isinstance(model_state, dict) and "serialized_data" in model_state:
                log.debug("Model state is already serialized, using as-is")
                packaged_model_state = model_state
            else:
                # Serialize model state for network transmission
                # Import here to avoid circular dependency
                # Use base64 encoding for JSON compatibility over WebSocket
                from common.model_serialization import PyTorchSerializer
                serializer = PyTorchSerializer(compression=True, encoding='base64')
                serialized_data = serializer.serialize(model_state)
                
                # Package serialized model with metadata
                packaged_model_state = {
                    "serialized_data": serialized_data,
                    "framework": "pytorch",
                    "compression": True,
                    "encoding": "base64"
                }
            
            # Create model update message
            update_msg = MessageFactory.create_model_update(
                self.node_id, self.cluster_id, packaged_model_state, samples, loss, round_num
            )
            
            # Send to server
            await self._send_message(update_msg)
            
            log.info("Model update sent successfully")
            self.state = ClientState.REGISTERED
            self.training_in_progress = False
            
        except Exception as e:
            log.error(f"Failed to send model update: {e}")
            self.state = ClientState.ERROR
            raise
    
    async def send_metrics(self, metrics: Dict, round_num: int):
        """
        Send training metrics to server.
        
        Args:
            metrics: Dictionary of metric name -> value
            round_num: Current training round
        """
        log = logger.bind(context="FederatedLearningClient.send_metrics")
        log.debug(f"Sending metrics for round {round_num}")
        
        try:
            # Extract required metrics from the dict
            loss = metrics.get('loss', 0.0)
            samples = metrics.get('samples', 0)

            metrics_msg = MessageFactory.create_metrics(
                self.node_id, self.cluster_id, loss, samples, round_num
            )
            
            await self._send_message(metrics_msg)
            log.debug("Metrics sent successfully")
            
        except Exception as e:
            log.error(f"Failed to send metrics: {e}")
            raise
    
    def get_connection_stats(self) -> ConnectionStats:
        """Get connection statistics."""
        return self.stats
    
    def is_connected(self) -> bool:
        """Check if client is connected and registered."""
        return self.state == ClientState.REGISTERED
    
    def is_training(self) -> bool:
        """Check if client is currently training."""
        return self.training_in_progress
    
    def get_current_round(self) -> Optional[int]:
        """Get current training round number."""
        return self.current_round

    def restart_heartbeat_with_interval(self, new_interval: float):
        """
        Restart heartbeat task with a new interval.

        Args:
            new_interval: New heartbeat interval in seconds
        """
        log = logger.bind(context="FederatedLearningClient.restart_heartbeat_with_interval")
        log.debug(f"Restarting heartbeat with interval {new_interval}s")

        # Cancel existing heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Set new interval and start new task
        self.heartbeat_interval = new_interval
        if self.state == ClientState.REGISTERED:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    def _log_final_stats(self):
        """Log final connection statistics."""
        log = logger.bind(context="FederatedLearningClient._log_final_stats")
        
        log.info("=== Final Connection Statistics ===")
        log.info(f"Connection attempts: {self.stats.connection_attempts}")
        log.info(f"Successful connections: {self.stats.successful_connections}")
        log.info(f"Total uptime: {self.stats.total_uptime:.1f} seconds")
        log.info(f"Messages sent: {self.stats.total_messages_sent}")
        log.info(f"Messages received: {self.stats.total_messages_received}")
        log.info(f"Reconnections: {self.stats.reconnections}")
        log.info(f"Ping failures: {self.stats.ping_failures}")
        log.info(f"Keepalive timeouts: {self.stats.keepalive_timeouts}")
        log.info(f"Message size errors: {self.stats.message_size_errors}")
        log.info(f"Large message warnings: {self.stats.large_message_warnings}")
        
        if self.stats.last_ping_time > 0:
            time_since_last_ping = time.time() - self.stats.last_ping_time
            log.info(f"Time since last successful ping: {time_since_last_ping:.1f} seconds")
        
        if self.stats.total_uptime > 0:
            msg_rate = (self.stats.total_messages_sent + self.stats.total_messages_received) / self.stats.total_uptime
            log.info(f"Average message rate: {msg_rate:.2f} messages/second")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics as a dictionary."""
        return {
            "connection_attempts": self.stats.connection_attempts,
            "successful_connections": self.stats.successful_connections,
            "total_uptime": self.stats.total_uptime,
            "messages_sent": self.stats.total_messages_sent,
            "messages_received": self.stats.total_messages_received,
            "reconnections": self.stats.reconnections,
            "ping_failures": self.stats.ping_failures,
            "keepalive_timeouts": self.stats.keepalive_timeouts,
            "last_ping_time": self.stats.last_ping_time,
            "message_size_errors": self.stats.message_size_errors,
            "large_message_warnings": self.stats.large_message_warnings,
        }