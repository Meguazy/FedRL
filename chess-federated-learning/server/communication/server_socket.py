"""
WebSocket server for federated learning node communication.

This module implements the central communication hub for the federated learning system.
It manages client connections, handles message routing, and coordinates the training workflow.

Key Components:
    - FederatedLearningServer: Main WebSocket server class
    - Connection management for multiple concurrent nodes
    - Message routing and broadcasting capabilities
    - Integration with cluster management and aggregation
    - Graceful handling of node disconnections and failures

Architecture:
    - Async WebSocket server using websockets library
    - Each client connection handled as a separate task
    - Thread-safe node registry and message queuing
    - Support for 64+ concurrent connections
"""

import asyncio
import json
import time
from typing import Dict, Set, Optional, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
import websockets
from loguru import logger

from .protocol import Message, MessageType, MessageFactory
from ..cluster_manager import ClusterManager


class NodeState(Enum):
    """States that a connected node can be in."""
    CONNECTING = "connecting"
    REGISTERED = "registered"
    TRAINING = "training"
    UPDATING = "updating"
    IDLE = "idle"
    DISCONNECTED = "disconnected"


@dataclass
class ConnectedNode:
    """
    Information about a connected node.
    
    Tracks the WebSocket connection, node metadata, and current state
    for each client connected to the federated learning server.
    
    Attributes:
        node_id: Unique node identifier (e.g., "agg_001")
        cluster_id: Cluster this node belongs to
        websocket: Active WebSocket connection
        state: Current state of the node
        last_heartbeat: Timestamp of last heartbeat received
        registration_time: When the node registered
        current_round: Training round node is participating in
    """
    node_id: str
    cluster_id: str
    websocket: websockets.WebSocketServerProtocol
    state: NodeState
    last_heartbeat: float
    registration_time: float
    current_round: Optional[int] = None


class FederatedLearningServer:
    """
    Central WebSocket server for federated learning coordination.
    
    This server manages all client connections, routes messages between nodes
    and the aggregation system, and coordinates the training workflow across
    all participating nodes.
    
    The server operates asynchronously, handling multiple client connections
    concurrently while maintaining thread-safe state management.
    
    Typical workflow:
    1. Start server and listen for connections
    2. Nodes connect and register with their cluster assignments
    3. Server initiates training rounds by broadcasting START_TRAINING
    4. Collect model updates from nodes
    5. Trigger aggregation and redistribute updated models
    6. Repeat training cycle
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 cluster_config_path: str = "config/cluster_topology.yaml"):
        """
        Initialize the federated learning server.
        
        Args:
            host: Server host address (default: localhost)
            port: Server port (default: 8765)
            cluster_config_path: Path to cluster topology YAML file (required)
        
        Raises:
            FileNotFoundError: If cluster configuration file not found
            ValueError: If cluster configuration is invalid
        """
        log = logger.bind(context="FederatedLearningServer.__init__")
        log.info("Initializing FederatedLearningServer...")
        
        self.host = host
        self.port = port
        
        # Initialize cluster manager directly (required)
        log.info(f"Loading cluster manager from config: {cluster_config_path}")
        self.cluster_manager: ClusterManager = ClusterManager(cluster_config_path)
        log.info(f"Cluster manager initialized with {self.cluster_manager.get_cluster_count()} clusters")
        
        # Connection management
        self.connected_nodes: Dict[str, ConnectedNode] = {}
        self.connections_by_cluster: Dict[str, Set[str]] = {}
        
        # Server state
        self.is_running = False
        self.server_instance = None
        self.current_round = 0
        
        # Callbacks for external integration
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.round_complete_callback: Optional[Callable] = None
        
        # Performance monitoring
        self.total_messages_handled = 0
        self.start_time = 0
        
        log.info(f"FederatedLearningServer initialized for {host}:{port}")
    
    def set_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a handler for specific message types.
        
        This allows external components (like aggregators) to handle
        specific messages while the server manages connections.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        log = logger.bind(context="FederatedLearningServer.set_message_handler")
        log.info(f"Setting handler for message type: {message_type}")
        self.message_handlers[message_type] = handler
    
    def set_round_complete_callback(self, callback: Callable):
        """
        Register a callback to be invoked when a training round completes.
        
        Args:
            callback: Function to call when round completes
        """
        log = logger.bind(context="FederatedLearningServer.set_round_complete_callback")
        log.info("Setting round complete callback")
        self.round_complete_callback = callback
    
    async def start_server(self):
        """
        Start the WebSocket server and begin listening for connections.
        
        This method starts the async server and will run indefinitely,
        handling client connections as they arrive.
        """
        log = logger.bind(context="FederatedLearningServer.start_server")
        log.info(f"Starting FL server on {self.host}:{self.port}")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Start the WebSocket server
            self.server_instance = await websockets.serve(
                handler=self._handle_client_connection,
                host=self.host,
                port=self.port,
                ping_interval=30,  # Send pings every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                max_size=50 * 1024 * 1024  # 50MB max message size
            )
            
            log.info("FL server started successfully")
            log.info("Server ready to accept node connections")
            
            # Keep server running
            await self.server_instance.wait_closed()
            
        except Exception as e:
            log.error(f"Error starting server: {e}")
            self.is_running = False
            raise
    
    async def stop_server(self):
        """
        Stop the WebSocket server and disconnect all clients.
        
        This method gracefully shuts down the server, closing all
        active connections and cleaning up resources.
        """
        log = logger.bind(context="FederatedLearningServer.stop_server")
        log.info("Stopping FL server...")
        
        # Notify all connected nodes of shutdown
        if self.connected_nodes:
            disconnect_tasks = []
            for node_id in list(self.connected_nodes.keys()):
                task = asyncio.create_task(self._disconnect_node(node_id, "server_shutdown"))
                disconnect_tasks.append(task)
            
            # Wait for all disconnects to complete
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        # Stop the server instance
        if self.server_instance:
            self.server_instance.close()
            await self.server_instance.wait_closed()
            self.server_instance = None
        
        self.is_running = False
        uptime = time.time() - self.start_time
        log.info(f"FL server stopped. Uptime: {uptime:.2f} seconds")
    
    async def _handle_client_connection(self, websocket: websockets.WebSocketServerProtocol):
        """
        Handle a new client connection.
        
        This method manages the lifecycle of a client connection,
        including registration, message handling, and disconnection.
        
        Args:
            websocket: Active WebSocket connection
        """
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        log = logger.bind(context="FederatedLearningServer._handle_client_connection")
        log.info(f"New client connection from {client_addr}")
        
        node_id = None
        try:
            # Wait for registration message
            node_id = await self._wait_for_registration(websocket)
            if node_id is None:
                log.warning(f"Client {client_addr} failed to register. Closing connection.")
                return
            
            log.info(f"Client {client_addr} registered as node {node_id}")
            
            # Handle messages from this client
            await self._handle_node_messages(node_id)
            
        except websockets.ConnectionClosed:
            log.info(f"Connection closed by client {client_addr}")
        except Exception as e:
            log.error(f"Error handling client {client_addr}: {e}")
        finally:
            # Cleanup connection
            if node_id:
                await self._cleanup_node_connection(node_id)
            else:
                log.info(f"Client {client_addr} disconnected before registration")
    
    async def _wait_for_registration(self, websocket: websockets.WebSocketServerProtocol) -> Optional[str]:
        """
        Wait for a registration message from the client.
        
        This method listens for the initial registration message
        and validates it before adding the node to the connected list.
        
        Args:
            websocket: Active WebSocket connection
        
        Returns:
            node_id if registration successful, else None
        """
        log = logger.bind(context="FederatedLearningServer._wait_for_registration")
        
        try:
            # Wait for registration message (with timeout)
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            message = Message.from_json(raw_message)
            
            # Validate registration message
            if message.type != MessageType.REGISTER.value:
                log.warning("First message must be REGISTER")
                await self._send_registration_response(websocket, "", "", False, "Expected registration message")
                return None
            
            if not message.validate():
                log.warning("Invalid registration message format")
                await self._send_registration_response(websocket, message.node_id, message.cluster_id, False, "Invalid message format")
                return None
            
            # Validate with cluster manager (required)
            if not self.cluster_manager.is_valid_node(message.node_id, message.cluster_id):
                log.warning(f"Invalid node/cluster combination: {message.node_id}/{message.cluster_id}")
                await self._send_registration_response(websocket, message.node_id, message.cluster_id, False, "Invalid node/cluster assignment")
                return None
            
            # Check if node already connected
            if message.node_id in self.connected_nodes:
                log.warning(f"Node {message.node_id} already connected")
                await self._send_registration_response(websocket, message.node_id, message.cluster_id, False, "Node already connected")
                return None
            
            # Register the node
            await self._register_node(message.node_id, message.cluster_id, websocket)
            
            # Send successful registration response
            await self._send_registration_response(websocket, message.node_id, message.cluster_id, True, "Registration successful")
            log.info(f"Node {message.node_id} registered successfully in cluster {message.cluster_id}")
            return message.node_id
        
        except asyncio.TimeoutError:
            log.warning("Registration timed out")
            return None
        except Exception as e:
            log.error(f"Error during registration: {e}")
            return None
    
    async def _register_node(self, node_id: str, cluster_id: str, websocket: websockets.WebSocketServerProtocol):
        """
        Add a new node to the connected nodes list.
        
        Args:
            node_id: Unique identifier for the node
            cluster_id: Cluster this node belongs to
            websocket: Active WebSocket connection
        """
        log = logger.bind(context="FederatedLearningServer._register_node")
        log.info(f"Registering node {node_id} in cluster {cluster_id}")
        
        # Create ConnectedNode instance
        connected_node = ConnectedNode(
            node_id=node_id,
            cluster_id=cluster_id,
            websocket=websocket,
            state=NodeState.REGISTERED,
            last_heartbeat=time.time(),
            registration_time=time.time()
        )
        
        # Store in registries
        self.connected_nodes[node_id] = connected_node
        
        # Add to cluster registry
        if cluster_id not in self.connections_by_cluster:
            self.connections_by_cluster[cluster_id] = set()
        self.connections_by_cluster[cluster_id].add(node_id)
        
        # Register with cluster manager (required)
        success = self.cluster_manager.register_node(node_id, cluster_id)
        if not success:
            log.error(f"Cluster manager registration failed for {node_id}")
            raise RuntimeError(f"Failed to register node {node_id} with cluster manager")
        
        log.info(f"Registered node {node_id} to cluster {cluster_id}")
        log.info(f"Total connected nodes: {len(self.connected_nodes)}")
        log.debug(f"Cluster {cluster_id} now has {len(self.connections_by_cluster[cluster_id])} nodes")
    
    async def _send_registration_response(self, websocket: websockets.WebSocketServerProtocol, 
                                        node_id: str, cluster_id: str, success: bool, message: str):
        """
        Send a registration response message to the client.
        
        Args:
            websocket: Active WebSocket connection
            node_id: Node identifier
            cluster_id: Cluster identifier
            success: Whether registration was successful
            message: Additional info message
        """
        log = logger.bind(context="FederatedLearningServer._send_registration_response")
        
        response = MessageFactory.create_register_ack(
            node_id=node_id,
            cluster_id=cluster_id,
            success=success,
            message=message
        )
        
        try:
            await websocket.send(response.to_json())
            log.info(f"Sent registration response to node {node_id}: success={success}")
        except Exception as e:
            log.error(f"Error sending registration response to node {node_id}: {e}")
    
    async def _handle_node_messages(self, node_id: str):
        """
        Main loop to handle messages from a connected node.
        
        This method listens for incoming messages from the node
        and routes them to the appropriate handlers.
        
        Args:
            node_id: Unique identifier for the connected node
        """
        log = logger.bind(context="FederatedLearningServer._handle_node_messages")
        log.info(f"Starting message handling loop for node {node_id}")
        
        node = self.connected_nodes.get(node_id)
        
        try:
            async for raw_message in node.websocket:
                self.total_messages_handled += 1
                
                try:
                    # Parse incoming message
                    message = Message.from_json(raw_message)
                    
                    # Update last activity timestamp
                    node.last_heartbeat = time.time()
                    
                    # Log message receipt
                    log.debug(f"Received message from node {node_id}: {message.type}")
                    
                    # Validate message
                    if not message.validate():
                        log.warning(f"Invalid message format from node {node_id}")
                        await self._send_error(node_id, "Invalid message format")
                        continue
                    
                    # Route message to appropriate handler
                    await self._route_message(node_id, message)
                
                except json.JSONDecodeError as e:
                    log.warning(f"JSON decode error from node {node_id}: {e}")
                    await self._send_error(node_id, "Invalid JSON format")
                except Exception as e:
                    log.error(f"Error processing message from node {node_id}: {e}")
                    await self._send_error(node_id, f"Message processing error: {str(e)}")
        
        except websockets.ConnectionClosed:
            log.info(f"Connection closed by node {node_id}")
        except Exception as e:
            log.error(f"Error in message loop for node {node_id}: {e}")
    
    async def _route_message(self, node_id: str, message: Message):
        """
        Route an incoming message to the appropriate handler.
        
        Args:
            node_id: Unique identifier for the sending node
            message: Parsed Message object
        """
        log = logger.bind(context="FederatedLearningServer._route_message")
        log.debug(f"Routing message of type {message.type} from node {node_id}")
        message_type = MessageType(message.type)
        
        # Handle built-in message types
        handled_builtin = False
        if message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(node_id, message)
            handled_builtin = True
        elif message_type == MessageType.MODEL_UPDATE:
            await self._handle_model_update(node_id, message)
            handled_builtin = True
        elif message_type == MessageType.METRICS:
            await self._handle_metrics(node_id, message)
            handled_builtin = True
        elif message_type == MessageType.DISCONNECT:
            await self._handle_disconnect_request(node_id, message)
            handled_builtin = True
        
        # Check for external handlers (in addition to built-in ones)
        if message_type in self.message_handlers:
            log.debug(f"Routing {message.type} to custom handler")
            try:
                await self.message_handlers[message_type](node_id, message)
            except Exception as e:
                log.error(f"Error in custom handler for {message.type}: {e}")
                await self._send_error(node_id, f"Handler error: {str(e)}")
        elif not handled_builtin:
            log.warning(f"No handler registered for message type {message.type}")
            await self._send_error(node_id, f"No handler for message type {message.type}")
    
    async def _handle_heartbeat(self, node_id: str, message: Message):
        """Handle a heartbeat message from a node."""
        log = logger.bind(context="FederatedLearningServer._handle_heartbeat")
        log.trace(f"Heartbeat from {node_id}")
        # Heartbeat handling is automatic (last_heartbeat updated in _handle_node_messages)
    
    async def _handle_model_update(self, node_id: str, message: Message):
        """Handle a model update message from a node."""
        log = logger.bind(context="FederatedLearningServer._handle_model_update")
        log.info(f"Received model update from node {node_id} for round {message.round_num}")
        
        # Update node state
        node = self.connected_nodes.get(node_id)
        node.state = NodeState.IDLE
        node.current_round = message.round_num
        
        # Log update details
        samples = message.payload.get("samples", 0)
        loss = message.payload.get("loss", 0.0)
        log.info(f"Node {node_id} submitted update: samples={samples}, loss={loss:.4f}")
    
    async def _handle_metrics(self, node_id: str, message: Message):
        """Handle a metrics message from a node."""
        log = logger.bind(context="FederatedLearningServer._handle_metrics")
        log.debug(f"Received metrics from node {node_id}: {message.payload}")
    
    async def _handle_disconnect_request(self, node_id: str, message: Message):
        """Handle disconnect message from node."""
        log = logger.bind(context="FederatedLearningServer._handle_disconnect_request")
        log.info(f"Disconnect request from {node_id}")
        await self._cleanup_node_connection(node_id)
    
    async def _send_error(self, node_id: str, error_message: str):
        """
        Send an error message to a specific node.
        
        Args:
            node_id: Unique identifier for the target node
            error_message: Error message to send
        """
        log = logger.bind(context="FederatedLearningServer._send_error")
        log.debug(f"Sending error to node {node_id}: {error_message}")
        
        if node_id not in self.connected_nodes:
            log.warning(f"Cannot send error, node {node_id} not connected")
            return
        
        node = self.connected_nodes[node_id]
        error_msg = MessageFactory.create_error(
            node_id=node_id,
            cluster_id=node.cluster_id,
            error_msg=error_message
        )
        
        try:
            await node.websocket.send(error_msg.to_json())
        except Exception as e:
            log.error(f"Error sending error message to node {node_id}: {e}")
    
    async def broadcast_to_cluster(self, cluster_id: str, message: Message):
        """
        Broadcast a message to all nodes in a specific cluster.
        
        Args:
            cluster_id: Target cluster identifier
            message: Message object to broadcast
        """
        log = logger.bind(context="FederatedLearningServer.broadcast_to_cluster")
        log.info(f"Broadcasting message of type {message.type} to cluster {cluster_id}")
        
        nodes = self.get_cluster_nodes(cluster_id, active_only=True)
        if not nodes:
            log.warning(f"No active nodes found in cluster {cluster_id} for broadcast")
            return
        
        log.info(f"Broadcasting to {len(nodes)} nodes in cluster {cluster_id}")
        
        # Send to all nodes in cluster
        send_tasks = []
        for node_id in nodes:
            if node_id in self.connected_nodes:
                task = asyncio.create_task(self._send_message_to_node(node_id, message))
                send_tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        # Log broadcast completion
        successful = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Broadcast to cluster {cluster_id} complete: {successful}/{len(send_tasks)} successful")
    
    async def broadcast_to_all(self, message: Message):
        """
        Broadcast a message to all connected nodes.
        
        Args:
            message: Message object to broadcast
        """
        log = logger.bind(context="FederatedLearningServer.broadcast_to_all")
        log.info(f"Broadcasting message of type {message.type} to all connected nodes")
        
        if not self.connected_nodes:
            log.warning("No connected nodes for broadcast")
            return
        
        nodes = list(self.connected_nodes.keys())
        log.info(f"Broadcasting to {len(nodes)} connected nodes")
        
        # Send to all connected nodes
        send_tasks = []
        for node_id in nodes:
            task = asyncio.create_task(self._send_message_to_node(node_id, message))
            send_tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        # Log broadcast completion
        successful = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Broadcast to all nodes complete: {successful}/{len(send_tasks)} successful")
    
    async def _send_message_to_node(self, node_id: str, message: Message):
        """
        Send message to specific node.
        
        Args:
            node_id: Target node
            message: Message to send
        """
        log = logger.bind(context="FederatedLearningServer._send_message_to_node")
        
        if node_id not in self.connected_nodes:
            log.warning(f"Cannot send message, node {node_id} not connected")
            return
        
        node = self.connected_nodes[node_id]
        
        try:
            await node.websocket.send(message.to_json())
            log.trace(f"Sent {message.type} to {node_id}")
        except websockets.ConnectionClosed:
            log.warning(f"Connection to {node_id} closed during send")
            await self._cleanup_node_connection(node_id)
        except Exception as e:
            log.error(f"Failed to send message to {node_id}: {e}")
            raise
    
    async def _disconnect_node(self, node_id: str, reason: str):
        """
        Disconnect a specific node.
        
        Args:
            node_id: Node to disconnect
            reason: Reason for disconnection
        """
        log = logger.bind(context="FederatedLearningServer._disconnect_node")
        log.info(f"Disconnecting node {node_id}: {reason}")
        
        if node_id not in self.connected_nodes:
            log.warning(f"Cannot disconnect, node {node_id} not connected")
            return
        
        node = self.connected_nodes[node_id]
        
        try:
            # Send disconnect message
            disconnect_msg = MessageFactory.create_error(
                node_id=node_id,
                cluster_id=node.cluster_id,
                error_msg=f"Disconnected: {reason}"
            )
            await node.websocket.send(disconnect_msg.to_json())
            
            # Close the WebSocket connection
            await node.websocket.close()
        
        except Exception as e:
            log.error(f"Error disconnecting node {node_id}: {e}")
        
        await self._cleanup_node_connection(node_id)
    
    async def _cleanup_node_connection(self, node_id: str):
        """
        Clean up node connection data.
        
        Args:
            node_id: Node to clean up
        """
        log = logger.bind(context="FederatedLearningServer._cleanup_node_connection")
        log.info(f"Cleaning up connection for node {node_id}")
        
        if node_id not in self.connected_nodes:
            log.warning(f"Node {node_id} not found in connected nodes during cleanup")
            return
        
        node = self.connected_nodes[node_id]
        cluster_id = node.cluster_id
        
        # Remove from connected nodes
        del self.connected_nodes[node_id]
        
        if cluster_id in self.connections_by_cluster:
            self.connections_by_cluster[cluster_id].discard(node_id)
            # Remove cluster entry if empty
            if not self.connections_by_cluster[cluster_id]:
                del self.connections_by_cluster[cluster_id]
        
        # Unregister from cluster manager (required)
        success = self.cluster_manager.unregister_node(node_id)
        if success:
            log.debug(f"Cluster manager unregistered {node_id} successfully")
        else:
            log.warning(f"Cluster manager unregistration failed for {node_id}")
        
        connection_time = time.time() - node.registration_time
        log.info(f"Cleaned up node {node_id} (connected for {connection_time:.1f}s)")
        log.info(f"Remaining nodes: {len(self.connected_nodes)}")
    
    def get_connected_nodes(self) -> Dict[str, ConnectedNode]:
        """Get dictionary of all connected nodes."""
        return self.connected_nodes.copy()
    
    def get_cluster_nodes(self, cluster_id: str, active_only: bool = True) -> List[str]:
        """
        Get list of node IDs in a specific cluster.
        
        Args:
            cluster_id: Target cluster identifier
            active_only: If True, return only active nodes
        
        Returns:
            List of node IDs in the cluster
        """
        log = logger.bind(context="FederatedLearningServer.get_cluster_nodes")
        
        nodes = self.cluster_manager.get_cluster_nodes(cluster_id, active_only=active_only)
        log.debug(f"Cluster manager returned {len(nodes)} nodes for {cluster_id}")
        return nodes
    
    def get_cluster_readiness(self, cluster_id: str, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Get detailed cluster readiness information.
        
        Args:
            cluster_id: Target cluster identifier
            threshold: Readiness threshold (default 0.8)
        
        Returns:
            Dictionary containing readiness information
        """
        cluster = self.cluster_manager.get_cluster(cluster_id)
        if not cluster:
            return {
                "cluster_id": cluster_id,
                "is_ready": False,
                "reason": "Cluster not found",
                "active_nodes": 0,
                "expected_nodes": 0
            }
        
        is_ready = self.cluster_manager.is_cluster_ready(cluster_id, threshold)
        readiness_ratio = cluster.get_readiness_ratio()
        
        return {
            "cluster_id": cluster_id,
            "is_ready": is_ready,
            "readiness_ratio": readiness_ratio,
            "active_nodes": cluster.get_active_node_count(),
            "expected_nodes": cluster.get_expected_node_count(),
            "threshold": threshold,
            "playstyle": cluster.playstyle
        }
    
    def get_all_clusters_readiness(self, threshold: float = 0.8) -> Dict[str, Dict[str, Any]]:
        """Get readiness information for all clusters."""
        readiness_info = {}
        clusters = self.cluster_manager.get_all_clusters()
        for cluster in clusters:
            readiness_info[cluster.cluster_id] = self.get_cluster_readiness(cluster.cluster_id, threshold)
        return readiness_info
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get comprehensive server statistics including cluster information."""
        uptime = time.time() - self.start_time if self.start_time > 0 else 0
        
        base_stats = {
            "server": {
                "host": self.host,
                "port": self.port,
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "current_round": self.current_round,
                "total_messages_handled": self.total_messages_handled
            },
            "connections": {
                "total_connected_nodes": len(self.connected_nodes),
                "total_active_clusters": len(self.connections_by_cluster)
            }
        }
        
        # Add cluster manager statistics (always available)
        cluster_stats = self.cluster_manager.get_statistics()
        base_stats["cluster_manager"] = cluster_stats
        base_stats["cluster_readiness"] = self.get_all_clusters_readiness()
        
        return base_stats
    
    def get_node_count(self) -> int:
        """Get total number of connected nodes."""
        return len(self.connected_nodes)
    
    def get_cluster_count(self) -> int:
        """Get number of clusters with connected nodes."""
        return len(self.connections_by_cluster)