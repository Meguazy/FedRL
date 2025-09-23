"""
Communication protocol for federated learning server-client communication.

This module defines the message types, message structure, and serialization/deserialization
logic for communication between the FL server and client nodes.

Key Components:
    - MessageType: Enum defining all possible message types
    - Message: Dataclass representing a protocol message
    - MessageFactory: Factory class for creating typed messages
    - Serialization utilities for model states
"""

from enum import Enum
from dataclasses import dataclass, asdict
from loguru import logger
from typing import Dict, Any, Optional
import json
import time

class MessageType(Enum):
    """
    Enumeration of all message types used in the federated learning protocol.
    
    Client -> Server messages:
        REGISTER: Node registration request
        MODEL_UPDATE: Trained model weights and metrics
        METRICS: Training metrics without model weights
        HEARTBEAT: Keep-alive signal
    
    Server -> Client messages:
        REGISTER_ACK: Registration acknowledgment
        START_TRAINING: Command to start training round
        CLUSTER_MODEL: Aggregated cluster model distribution
        REQUEST_MODEL: Request current model from node
    
    Bidirectional messages:
        ERROR: Error notification
        DISCONNECT: Graceful disconnection notice
    """
    # Client -> Server
    REGISTER = "register"
    MODEL_UPDATE = "model_update"
    METRICS = "metrics"
    HEARTBEAT = "heartbeat"
    
    # Server -> Client
    REGISTER_ACK = "register_ack"
    START_TRAINING = "start_training"
    CLUSTER_MODEL = "cluster_model"
    REQUEST_MODEL = "request_model"
    
    # Bidirectional
    ERROR = "error"
    DISCONNECT = "disconnect"
    
@dataclass
class Message:
    """
    Base message structure for all communications in the federated learning system.
    
    This dataclass encapsulates all information needed for server-client communication,
    including message routing (node_id, cluster_id), timing (timestamp, round_num),
    and the actual data payload.
    
    Attributes:
        type: Type of message (must be a valid MessageType enum value)
        node_id: Unique identifier of the sending/receiving node (e.g., "agg_001")
        cluster_id: Cluster the node belongs to (e.g., "cluster_aggressive")
        payload: Dictionary containing message-specific data
        timestamp: Unix timestamp when message was created
        round_num: Optional training round number for tracking
    
    Example:
        >>> msg = Message(
        ...     type="register",
        ...     node_id="agg_001",
        ...     cluster_id="cluster_aggressive",
        ...     payload={},
        ...     timestamp=time.time()
        ... )
    """
    type: str
    node_id: str
    cluster_id: str
    payload: Dict[str, Any]
    timestamp: float
    round_num: Optional[int] = None
    
    def to_json(self) -> str:
        """Serialize the message to a JSON string."""
        log = logger.bind(context="Message.to_json")
        log.debug(f"Serializing message: type={self.type}, node_id={self.node_id}")
        
        try:
            json_str = json.dumps(asdict(self))
            log.trace(f"Serialized JSON: {json_str[:100]}...")  # Log first 100 chars
            return json_str
        except Exception as e:
            log.error(f"Error serializing message: {e}")
            raise
        
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Deserialize a JSON string to a Message object."""
        log = logger.bind(context="Message.from_json")
        log.debug(f"Deserializing JSON: {json_str[:100]}...")  # Log first 100 chars
        
        try:
            data = json.loads(json_str)
            log.trace(f"Parsed JSONL: type={data.get('type')}, node_id={data.get('node_id')}")
            
            message = cls(**data)
            log.debug(f"Succesfully deserialized message: type={message.type}")
            return message
        except json.JSONDecodeError as e:
            log.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            log.error(f"Invalid message structure: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate message structure and required fields.
        
        Performs two levels of validation:
        1. Basic structure: type, node_id, cluster_id are valid
        2. Payload validation: message-type-specific requirements
        
        Returns:
            bool: True if message is valid, False otherwise
        
        Example:
            >>> msg = Message(...)
            >>> if msg.validate():
            ...     # Process message
        """
        log = logger.bind(context="Message.validate")
        log.debug(f"Validating message: type={self.type}, node={self.node_id}")

        # Check if tupe is valid MessageType enum value
        if self.type not in [mt.value for mt in MessageType]:
            log.warning(f"Invalid message type: {self.type}")
            return False
        
        # Check required fields are present and non-empty
        if not self.node_id or not self.cluster_id:
            log.warning("Missing required fields: node_id or cluster_id")
            return False
        
        # Validate payload based on message type
        is_valid = self._validate_payload()
        
        if is_valid:
            log.debug("Message validation successful")
        else:
            log.warning("Message payload validation failed")
            
        return is_valid
    
    def _validate_payload(self) -> bool:
        """
        Validate payload contents based on message type.
        
        Each message type has specific requirements for its payload:
        - MODEL_UPDATE: must have model_state and samples
        - METRICS: must have loss and samples
        - START_TRAINING: must have games_per_round
        - CLUSTER_MODEL: must have model_state
        
        Returns:
            bool: True if payload is valid for this message type
        """
        log = logger.bind(context="Message._validate_payload")
        msg_type = MessageType(self.type)
        
        # Registration messages need no special payload
        if msg_type == MessageType.REGISTER:
            log.trace("No payload validation needed for REGISTER")
            return True
        
        # Model update must contain model_state and sample count
        elif msg_type == MessageType.MODEL_UPDATE:
            has_required = "model_state" in self.payload and "samples" in self.payload
            if not has_required:
                log.warning("MODEL_UPDATE payload missing model_state or samples")
            return has_required
        
        # Metrics must have loss and sample count
        elif msg_type == MessageType.METRICS:
            has_required = "loss" in self.payload and "samples" in self.payload
            if not has_required:
                log.warning("METRICS payload missing loss or samples")
            return has_required
        
        # Start training must specify games per round
        elif msg_type == MessageType.START_TRAINING:
            has_required = "games_per_round" in self.payload
            if not has_required:
                log.warning("START_TRAINING payload missing games_per_round")
            return has_required
        
        # Cluster model must include model_state
        elif msg_type == MessageType.CLUSTER_MODEL:
            has_required = "model_state" in self.payload
            if not has_required:
                log.warning("CLUSTER_MODEL payload missing model_state")
            return has_required
        
        # Other message types have no specific payload requirements
        log.trace(f"No specific payload validation for message type: {self.type}")
        return True
    

class MessageFactory:
    """
    Factory class for creating protocol messages with proper structure.
    
    This class provides static methods to create well-formed messages for all
    communication scenarios. Using the factory ensures messages are correctly
    structured and reduces the chance of errors.
    
    All methods return Message objects ready to be serialized and sent.
    """
    
    @staticmethod
    def create_register_message(node_id: str, cluster_id: str) -> Message:
        """
        Create a registration message from client to server.
        
        When a node starts up, it sends this message to register itself
        with the federated learning server.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")

        Returns:
            Message: Registration message ready to send
        """
        log = logger.bind(context="MessageFactory.create_register_message")
        log.info(f"Creating REGISTER message for node={node_id}, cluster={cluster_id}")
        
        return Message(
            type=MessageType.REGISTER.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={},
            timestamp=time.time()
        )
        
    @staticmethod
    def create_register_ack(
        node_id: str,
        cluster_id: str,
        success: bool,
        message: Optional[str] = ""
    ) -> Message:
        """
        Create a registration acknowledgment message from server to client.
        
        The server responds to a REGISTER message with this acknowledgment,
        indicating whether registration was successful.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            success: Whether registration was successful
            message: Optional message providing additional info

        Returns:
            Message: Registration acknowledgment message ready to send
        """
        log = logger.bind(context="MessageFactory.create_register_ack")
        log.info(f"Creating REGISTER_ACK message for node={node_id}, cluster={cluster_id}, success={success}")

        return Message(
            type=MessageType.REGISTER_ACK.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "success": success,
                "message": message
            },
            timestamp=time.time()
        )
        
    @staticmethod
    def create_model_update(
        node_id: str,
        cluster_id: str,
        model_state: Dict[str, Any],
        samples: int,
        loss: float,
        round_num: int
    ) -> Message:
        """
        Create a model update message from client to server.
        
        After training, a node sends this message containing its updated
        model weights and training metrics.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            model_state: Serialized model weights/state dictionary
            samples: Number of training samples used
            loss: Training loss metric
            round_num: Training round number
        
        Returns:
            Message: Model update message ready to send
        """
        log = logger.bind(context="MessageFactory.create_model_update")
        log.info(f"Creating MODEL_UPDATE message for node={node_id}, cluster={cluster_id}, round={round_num}")

        return Message(
            type=MessageType.MODEL_UPDATE.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "model_state": model_state,
                "samples": samples,
                "loss": loss
            },
            timestamp=time.time(),
            round_num=round_num
        )
        
    @staticmethod
    def create_start_training(
        node_id: str,
        cluster_id: str,
        games_per_round: int,
        round_num: int
    ) -> Message:
        """
        Create a start training command message from server to client.
        
        The server sends this message to instruct a node to begin
        a new training round with specified parameters.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            games_per_round: Number of games to play in this training round
            round_num: Training round number

        Returns:
            Message: Start training command message ready to send
        """
        log = logger.bind(context="MessageFactory.create_start_training")
        log.info(f"Creating START_TRAINING message for node={node_id}, cluster={cluster_id}, round={round_num}")

        return Message(
            type=MessageType.START_TRAINING.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "games_per_round": games_per_round
            },
            timestamp=time.time(),
            round_num=round_num
        )

    @staticmethod
    def create_cluster_model(
        node_id: str,
        cluster_id: str,
        model_state: Dict[str, Any],
        round_num: int
    ) -> Message:
        """
        Create a cluster model distribution message from server to client.
        
        The server sends this message to distribute the aggregated
        cluster model to all nodes in the cluster.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            model_state: Serialized aggregated model weights/state dictionary
            round_num: Training round number
        
        Returns:
            Message: Cluster model distribution message ready to send
        """
        log = logger.bind(context="MessageFactory.create_cluster_model")
        log.info(f"Creating CLUSTER_MODEL message for node={node_id}, cluster={cluster_id}, round={round_num}")

        return Message(
            type=MessageType.CLUSTER_MODEL.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "model_state": model_state
            },
            timestamp=time.time(),
            round_num=round_num
        )
        
    @staticmethod
    def create_metrics(
        node_id: str,
        cluster_id: str,
        loss: float,
        samples: int,
        round_num: int
    ) -> Message:
        """
        Create a metrics message from client to server.
        
        A node sends this message to report training metrics
        without sending model weights.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            loss: Training loss metric
            samples: Number of training samples used
            round_num: Training round number

        Returns:
            Message: Metrics message ready to send
        """
        log = logger.bind(context="MessageFactory.create_metrics")
        log.info(f"Creating METRICS message for node={node_id}, cluster={cluster_id}, round={round_num}")

        return Message(
            type=MessageType.METRICS.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "loss": loss,
                "samples": samples
            },
            timestamp=time.time(),
            round_num=round_num
        )

    @staticmethod
    def create_error(
        node_id: str,
        cluster_id: str,
        error_code: int,
        message: str
    ) -> Message:
        """
        Create an error notification message.
        
        Either the server or client can send this message to
        notify the other party of an error condition.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")
            error_code: Numeric error code
            message: Descriptive error message
        """
        log = logger.bind(context="MessageFactory.create_error")
        log.info(f"Creating ERROR message for node={node_id}, cluster={cluster_id}, error_code={error_code}")

        return Message(
            type=MessageType.ERROR.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={
                "error_code": error_code,
                "message": message
            },
            timestamp=time.time()
        )
        
    @staticmethod
    def create_heartbeat(
        node_id: str,
        cluster_id: str
    ) -> Message:
        """
        Create a heartbeat message to keep the connection alive.
        
        Clients periodically send this message to inform the server
        they are still active and connected.
        
        Args:
            node_id: Unique node identifier (e.g., "agg_001")
            cluster_id: Cluster this node belongs to (e.g., "cluster_aggressive")

        Returns:
            Message: Heartbeat message ready to send
        """
        log = logger.bind(context="MessageFactory.create_heartbeat")
        log.info(f"Creating HEARTBEAT message for node={node_id}, cluster={cluster_id}")

        return Message(
            type=MessageType.HEARTBEAT.value,
            node_id=node_id,
            cluster_id=cluster_id,
            payload={},
            timestamp=time.time()
        )
        
        
def serialize_model_state(model_state: Dict, framework: str = 'pytorch', 
                          compression: bool = True) -> Union[bytes, str]:
    """
    Serialize model state dict for network transmission.
    
    This function uses the appropriate serializer based on the framework.
    It's a convenience wrapper around the model_serialization module.
    
    Args:
        model_state: Framework-specific model state (state_dict for PyTorch, 
                    weights list for TensorFlow)
        framework: 'pytorch' or 'tensorflow' (default: 'pytorch')
        compression: Whether to compress the serialized data (default: True)
    
    Returns:
        Serialized model data ready for transmission (bytes or base64 string)
    
    Example:
        >>> state_dict = model.state_dict()  # PyTorch
        >>> data = serialize_model_state(state_dict, framework='pytorch')
    """
    log = logger.bind(context="serialize_model_state")
    log.info(f"Serializing model state using {framework} serializer (compression={compression})")
    
    try:
        from common.model_serialization import get_serializer
        
        # Get appropriate serializer for the framework
        serializer = get_serializer(framework, compression=compression, encoding='binary')
        
        # Serialize the model state
        serialized = serializer.serialize(model_state)
        
        log.info(f"Model serialization complete, size: {len(serialized)} bytes")
        return serialized
        
    except ImportError as e:
        log.error(f"Failed to import model_serialization module: {e}")
        raise
    except Exception as e:
        log.error(f"Model serialization failed: {e}")
        raise


def deserialize_model_state(data: Union[bytes, str], framework: str = 'pytorch',
                            compression: bool = True) -> Union[Dict, List]:
    """
    Deserialize model state dict from network transmission format.
    
    This function uses the appropriate deserializer based on the framework.
    It's a convenience wrapper around the model_serialization module.
    
    Args:
        data: Serialized model data (bytes or base64 string)
        framework: 'pytorch' or 'tensorflow' (default: 'pytorch')
        compression: Whether the data is compressed (default: True)
    
    Returns:
        Framework-specific model state ready to load:
            - PyTorch: state_dict (Dict)
            - TensorFlow: weights list (List of numpy arrays)
    
    Example:
        >>> data = receive_from_network()
        >>> state_dict = deserialize_model_state(data, framework='pytorch')
        >>> model.load_state_dict(state_dict)  # PyTorch
    """
    log = logger.bind(context="deserialize_model_state")
    log.info(f"Deserializing model state using {framework} deserializer (compression={compression})")
    
    try:
        from common.model_serialization import get_serializer
        
        # Get appropriate serializer for the framework
        serializer = get_serializer(framework, compression=compression, encoding='binary')
        
        # Deserialize the model state
        model_state = serializer.deserialize(data)
        
        log.info(f"Model deserialization complete")
        return model_state
        
    except ImportError as e:
        log.error(f"Failed to import model_serialization module: {e}")
        raise
    except Exception as e:
        log.error(f"Model deserialization failed: {e}")
        raise
