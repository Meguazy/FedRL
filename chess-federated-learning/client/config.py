"""
Configuration management for federated learning nodes.

This module provides configuration dataclasses and utilities for loading
node configurations from YAML files, environment variables, or programmatically.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from loguru import logger


@dataclass
class NodeConfig:
    """
    Complete configuration for a federated learning node.

    Attributes:
        node_id: Unique identifier for this node
        cluster_id: Cluster this node belongs to
        server_host: FL server hostname
        server_port: FL server port
        trainer_type: Type of trainer ("dummy", "supervised", "alphazero")
        auto_reconnect: Whether to automatically reconnect on disconnect
        training: Training configuration parameters
        storage: Local storage configuration
        logging: Logging configuration
        config: Additional trainer-specific configuration (e.g., supervised, alphazero)
    """
    node_id: str
    cluster_id: str
    server_host: str = "localhost"
    server_port: int = 8765
    trainer_type: str = "dummy"
    auto_reconnect: bool = True
    training: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        if not self.cluster_id:
            raise ValueError("cluster_id cannot be empty")
        if self.server_port <= 0 or self.server_port > 65535:
            raise ValueError(f"Invalid server port: {self.server_port}")
        
        # Set default training parameters if not provided
        if not self.training:
            self.training = {
                "games_per_round": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "exploration_factor": 1.0,
                "max_game_length": 200,
                "save_games": True,
            }
        
        # Set default storage parameters if not provided
        if not self.storage:
            self.storage = {
                "enabled": False,
                "base_path": "./storage",
                "save_models": True,
                "save_metrics": True,
            }
        
        # Set default logging parameters if not provided
        if not self.logging:
            self.logging = {
                "level": "INFO",
                "file": None,
                "format": "text",
            }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NodeConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            NodeConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        log = logger.bind(context="NodeConfig.from_yaml")
        log.info(f"Loading node configuration from {yaml_path}")

        config_file = Path(yaml_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        # Extract known fields
        known_fields = {
            'node_id', 'cluster_id', 'server_host', 'server_port',
            'trainer_type', 'auto_reconnect', 'training', 'storage', 'logging'
        }

        # Separate known fields from additional config
        node_config_params = {}
        additional_config = {}

        for key, value in config_data.items():
            if key in known_fields:
                node_config_params[key] = value
            else:
                additional_config[key] = value

        # Add additional config to the config dict
        if additional_config:
            node_config_params['config'] = additional_config

        log.debug(f"Loaded configuration with {len(additional_config)} additional config sections")
        return cls(**node_config_params)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NodeConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            NodeConfig instance
        """
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        log = logger.bind(context="NodeConfig.to_yaml")
        log.info(f"Saving configuration to {yaml_path}")
        
        config_file = Path(yaml_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)
        
        log.debug(f"Configuration saved to {yaml_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return asdict(self)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")


def create_default_config(node_id: str, cluster_id: str) -> NodeConfig:
    """
    Create a default node configuration.
    
    Args:
        node_id: Node identifier
        cluster_id: Cluster identifier
    
    Returns:
        NodeConfig with default values
    """
    return NodeConfig(
        node_id=node_id,
        cluster_id=cluster_id,
        server_host="localhost",
        server_port=8765,
        trainer_type="dummy",
        auto_reconnect=True,
    )


def load_config_with_overrides(yaml_path: str, **overrides) -> NodeConfig:
    """
    Load configuration from YAML and apply overrides.
    
    Useful for loading a template config and customizing specific values.
    
    Args:
        yaml_path: Path to base YAML configuration
        **overrides: Parameters to override
    
    Returns:
        NodeConfig with overrides applied
    """
    config = NodeConfig.from_yaml(yaml_path)
    config.update(**overrides)
    return config