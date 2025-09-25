
"""
Cluster management for federated learning nodes.

This module manages cluster topology, node assignments, and cluster state for the
federated learning system. It handles node registration, validation, and provides
cluster-specific information for aggregation and communication.

Key Features:
    - Auto-generation of node IDs based on configuration
    - Dynamic cluster topology management
    - Node registration and validation
    - Cluster readiness checking with configurable thresholds
    - Support for scalable node counts (up to 64+ nodes per cluster)
    - YAML-based configuration loading

Architecture:
    - Cluster class: Represents individual clusters with metadata
    - ClusterManager: Central management of all clusters and nodes
    - Configuration-driven topology setup
    - Thread-safe operations for concurrent node management
"""

import yaml
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime
from loguru import logger


@dataclass
class Cluster:
    """
    Represents a cluster in the federated learning system.
    
    Each cluster groups nodes with the same playstyle (e.g., aggressive, positional)
    and maintains metadata about the nodes, their status, and cluster configuration.
    
    Attributes:
        cluster_id: Unique cluster identifier (e.g., "cluster_aggressive")
        playstyle: Playstyle this cluster represents (e.g., "aggressive", "positional")
        node_count: Target number of nodes for this cluster
        node_prefix: Prefix for auto-generated node IDs (e.g., "agg")
        description: Human-readable description of the cluster
        expected_nodes: Set of expected node IDs for this cluster
        active_nodes: Set of currently registered/active node IDs
        inactive_nodes: Set of registered but currently disconnected nodes
        creation_time: Timestamp when cluster was created
        last_activity: Timestamp of last node activity in this cluster
    """
    cluster_id: str
    playstyle: str
    node_count: int
    node_prefix: str
    description: str = ""
    expected_nodes: Set[str] = field(default_factory=set)
    active_nodes: Set[str] = field(default_factory=set)
    inactive_nodes: Set[str] = field(default_factory=set)
    creation_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize cluster after creation."""
        log = logger.bind(context="Cluster.__post_init__")
        log.debug(f"Initializing cluster {self.cluster_id} with playstyle {self.playstyle}")
        
        # Auto-generate expected node IDs if not provided
        if not self.expected_nodes:
            self.expected_nodes = {
                f"{self.node_prefix}_{i+1:03d}" 
                for i in range(self.node_count)
            }
        
            log.debug(f"Generated {len(self.expected_nodes)} expected nodes for {self.cluster_id}")
        
        log.info(f"Created cluster {self.cluster_id} with {self.node_count} expected nodes")
        
    def get_total_registered_nodes(self) -> int:
        """Return the total number of registered nodes (active + inactive)."""
        log = logger.bind(context="Cluster.get_total_registered_nodes")
        log.debug(f"Cluster {self.cluster_id} has {len(self.active_nodes)} active and {len(self.inactive_nodes)} inactive nodes")
        return len(self.active_nodes) + len(self.inactive_nodes)
    
    def get_active_node_count(self) -> int:
        """Return the number of currently active nodes."""
        log = logger.bind(context="Cluster.get_active_node_count")
        log.debug(f"Cluster {self.cluster_id} has {len(self.active_nodes)} active nodes")
        return len(self.active_nodes)
    
    def get_expected_node_count(self) -> int:
        """Return the number of expected nodes."""
        log = logger.bind(context="Cluster.get_expected_node_count")
        log.debug(f"Cluster {self.cluster_id} has {len(self.expected_nodes)} expected nodes")
        return len(self.expected_nodes)
    
    def is_node_expected(self, node_id: str) -> bool:
        """Check if a node ID is expected in this cluster."""
        log = logger.bind(context="Cluster.is_node_expected")
        log.debug(f"Checking if node {node_id} is expected in cluster {self.cluster_id}")
        return node_id in self.expected_nodes
    
    def is_node_active(self, node_id: str) -> bool:
        """Check if a node ID is currently active in this cluster."""
        log = logger.bind(context="Cluster.is_node_active")
        log.debug(f"Checking if node {node_id} is active in cluster {self.cluster_id}")
        return node_id in self.active_nodes
    
    def get_readiness_ratio(self) -> float:
        """Calculate the readiness ratio of active nodes to expected nodes."""
        log = logger.bind(context="Cluster.get_readiness_ratio")
        log.debug(f"Calculating readiness ratio for cluster {self.cluster_id}")
        expected = self.get_expected_node_count()
        if expected == 0:
            return 1.0
        return self.get_active_node_count() / expected
    
    def is_ready(self, threshold: float = 0.8) -> bool:
        """
        Check if cluster is ready based on active node threshold.
        
        Args:
            threshold: Minimum ratio of active/expected nodes required (0.0-1.0)
        
        Returns:
            bool: True if cluster has enough active nodes
        """
        log = logger.bind(context="Cluster.is_ready")
        log.debug(f"Checking if cluster {self.cluster_id} is ready with threshold {threshold}")
        return self.get_readiness_ratio() >= threshold

    def update_activity(self):
        """Update the last activity timestamp to current time."""
        self.last_activity = time.time()
        log = logger.bind(context="Cluster.update_activity")
        log.debug(f"Updated last activity for cluster {self.cluster_id} to {self.last_activity}")
        
        
class ClusterManager:
    """
    Central management system for all clusters in the federated learning system.
    
    This class handles cluster topology configuration, node registration and validation,
    cluster state tracking, and provides interfaces for aggregation and communication
    components to query cluster information.
    
    Key responsibilities:
    - Load cluster topology from YAML configuration
    - Auto-generate node IDs based on cluster configuration
    - Validate node registrations against expected topology
    - Track node states (active, inactive, disconnected)
    - Provide cluster-specific node lists for aggregation
    - Monitor cluster readiness for training rounds
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the cluster manager.
        
        Args:
            config_path: Path to cluster topology YAML file (optional)
        """
        log = logger.bind(context="ClusterManager.__init__")
        log.info("Initializing ClusterManager...")
        log.debug(f"Loading configuration from {config_path if config_path else 'default path'}")
        
        # Core data structures
        self.clusters: Dict[str, Cluster] = {}
        self.node_to_cluster: Dict[str, str] = {}
        self.node_registration_times: Dict[str, float] = {}
        self.node_last_seen: Dict[str, float] = {}
        
        # Configuration
        self.config_path = config_path
        self.auto_generate_nodes = True
        self.default_threshold = 0.8  # 80% nodes active to be ready

        # Statistics
        self.total_registrations = 0
        self.total_unregistrations = 0
        self.creation_time = time.time()
        
        log.info("ClusterManager initialized.")
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
            log.info("Cluster configuration loaded.")
        else:
            log.warning("No configuration path provided. ClusterManager needs to be configured manually.")
            
    def load_configuration(self, config_path: str):
        """
        Load cluster topology from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
            ValueError: If config contains invalid data
        """
        log = logger.bind(context="ClusterManager.load_configuration")
        log.info(f"Loading cluster configuration from {config_path}")
        
        config_file = Path(config_path)
        if not config_file.exists():
            log.error(f"Configuration file {config_path} not found.")
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            log.debug(f"Loaded YAML configuration with {len(config.get('clusters', []))} clusters.")
            
            # Validate configuration structure
            if 'clusters' not in config:
                log.error("Configuration missing 'clusters' section.")
                raise ValueError("Configuration must contain 'clusters' section.")
            
            clusters_config = config['clusters']
            if not isinstance(clusters_config, list):
                log.error("'clusters' section must be a list.")
                raise ValueError("'clusters' section must be a list.")
            
            # Process each cluster configuration
            for cluster_cfg in clusters_config:
                self._create_cluster_from_config(cluster_cfg)
                
            log.info(f"Successfully loaded {len(self.clusters)} clusters from configuration.")
            self._log_cluster_summary()
            
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML configuration: {e}")
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            log.error(f"Unexpected error loading configuration: {e}")
            raise e
        
    def _create_cluster_from_config(self, cluster_cfg: Dict[str, Any]):
        """
        Create a cluster from configuration dictionary.
        
        Args:
            cluster_config: Dictionary containing cluster configuration
        """
        log = logger.bind(context="ClusterManager._create_cluster_from_config")
        log.debug(f"Creating cluster from config: {cluster_cfg}")
        
        # Validate required fields
        required_fields = ['id', 'playstyle', 'node_count', 'node_prefix']
        for field in required_fields:
            if field not in cluster_cfg:
                log.error(f"Cluster configuration missing required field: {field}")
                raise ValueError(f"Cluster configuration must contain '{field}' field.")
            
        cluster_id = cluster_cfg['id']
        playstyle = cluster_cfg['playstyle']
        node_count = cluster_cfg['node_count']
        node_prefix = cluster_cfg['node_prefix']
        description = cluster_cfg.get('description', "")
        
        log.debug(f"Creating cluster {cluster_id} with playstyle {playstyle}, node_count {node_count}, node_prefix {node_prefix}")
        
        # Validate node count
        if not isinstance(node_count, int) or node_count <= 0:
            log.error(f"Invalid node_count {node_count} for cluster {cluster_id}. Must be a positive integer.")
            raise ValueError(f"node_count must be a positive integer for cluster {cluster_id}.")
        
        # Create Cluster instance
        cluster = Cluster(
            cluster_id=cluster_id,
            playstyle=playstyle,
            node_count=node_count,
            node_prefix=node_prefix,
            description=description
        )
        
        # Add to cluster registry
        self.clusters[cluster_id] = cluster
        
        # Update node to cluster mapping for expected nodes
        for node_id in cluster.expected_nodes:
            self.node_to_cluster[node_id] = cluster_id
            
        log.info(f"Cluster {cluster_id} created with {len(cluster.expected_nodes)} expected nodes.")
        
    def add_cluster(self, cluster_id: str, playstyle: str, node_count: int, 
                   node_prefix: str, description: str = "") -> Cluster:
        """
        Manually add a cluster to the manager.
        
        Args:
            cluster_id: Unique cluster identifier
            playstyle: Playstyle for this cluster
            node_count: Number of nodes expected in this cluster
            node_prefix: Prefix for auto-generated node IDs
            description: Optional description
        
        Returns:
            Cluster: Created cluster instance
        
        Raises:
            ValueError: If cluster_id already exists
        """
        log = logger.bind(context="ClusterManager.add_cluster")
        log.info(f"Adding cluster {cluster_id} manually.")
        
        if cluster_id in self.clusters:
            log.error(f"Cluster ID {cluster_id} already exists.")
            raise ValueError(f"Cluster ID {cluster_id} already exists.")
        
        # Create Cluster instance
        cluster = Cluster(
            cluster_id=cluster_id,
            playstyle=playstyle,
            node_count=node_count,
            node_prefix=node_prefix,
            description=description
        )
        
        # Add to cluster registry
        self.clusters[cluster_id] = cluster
        for node_id in cluster.expected_nodes:
            self.node_to_cluster[node_id] = cluster_id
            
        log.info(f"Cluster {cluster_id} added with {len(cluster.expected_nodes)} expected nodes.")
        return cluster
    
    def is_valid_node(self, node_id: str, cluster_id: str) -> bool:
        """
        Validate if a node can be registered to a cluster.
        
        Args:
            node_id: Node identifier to validate
            cluster_id: Cluster identifier to validate against
        
        Returns:
            bool: True if node/cluster combination is valid
        """
        log = logger.bind(context="ClusterManager.is_valid_node")
        log.debug(f"Validating node {node_id} for cluster {cluster_id}")
        
        # Check if cluster exists
        if cluster_id not in self.clusters:
            log.warning(f"Cluster {cluster_id} does not exist.")
            return False
        
        cluster = self.clusters[cluster_id]
        
        # Check if node is expected in this cluster
        if not cluster.is_node_expected(node_id):
            log.warning(f"Node {node_id} is not expected in cluster {cluster_id}.")
            return False
        
        # Check if node is already registered to a different cluster
        if node_id in self.node_to_cluster:
            current_cluster = self.node_to_cluster[node_id]
            if current_cluster != cluster_id:
                log.warning(f"Node {node_id} is already registered to cluster {current_cluster}.")
                return False
            
        logger.debug(f"Node {node_id} is valid for cluster {cluster_id}.")
        return True
    
    def register_node(self, node_id: str, cluster_id: str) -> bool:
        """
        Register a node to a cluster.
        
        Args:
            node_id: Node identifier to register
            cluster_id: Cluster identifier to register to
        
        Returns:
            bool: True if registration was successful, False otherwise
        """
        log = logger.bind(context="ClusterManager.register_node")
        log.info(f"Registering node {node_id} to cluster {cluster_id}")
        
        # Validate registration
        if not self.is_valid_node(node_id, cluster_id):
            log.error(f"Failed to register node {node_id} to cluster {cluster_id}: invalid node or cluster.")
            return False
        
        cluster = self.clusters[cluster_id]
        current_time = time.time()
        
        # Move node from inactive to active if it was previously registered
        if node_id in cluster.inactive_nodes:
            cluster.inactive_nodes.remove(node_id)
            log.debug(f"Node {node_id} moved from inactive to active in cluster {cluster_id}.")
            
        # Add node to active nodes if not already present
        if node_id not in cluster.active_nodes:
            cluster.active_nodes.add(node_id)            
            log.info(f"Node {node_id} registered as active in cluster {cluster_id}. Total registrations: {self.total_registrations + 1}")
            cluster.update_activity()
            
        # Update tracking data
        self.node_to_cluster[node_id] = cluster_id
        self.node_registration_times[node_id] = current_time
        self.node_last_seen[node_id] = current_time
        self.total_registrations += 1
        
        log.info(f"Node {node_id} successfully registered to cluster {cluster_id}.")
        log.debug(f"Cluster {cluster_id} now has {len(cluster.active_nodes)} active nodes.")
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from its cluster.
        
        Args:
            node_id: Node identifier to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        log = logger.bind(context="ClusterManager.unregister_node")
        log.info(f"Unregistering node {node_id}")
        
        if node_id not in self.node_to_cluster:
            log.warning(f"Node {node_id} is not registered to any cluster.")
            return False
        
        cluster_id = self.node_to_cluster[node_id]
        cluster = self.clusters[cluster_id]
        
        # Remove from active nodes if present, add to inactive
        if node_id in cluster.active_nodes:
            cluster.active_nodes.remove(node_id)
            cluster.inactive_nodes.add(node_id)
            log.debug(f"Node {node_id} moved from active to inactive in cluster {cluster_id}.")
            
        # Update tracking data
        self.node_last_seen[node_id] = time.time()
        self.total_unregistrations += 1
        
        log.info(f"Node {node_id} successfully unregistered from cluster {cluster_id}. Total unregistrations: {self.total_unregistrations}")
        log.debug(f"Cluster {cluster_id} now has {len(cluster.active_nodes)} active nodes.")
        
        return True
    
    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """
        Retrieve a cluster by its ID.
        
        Args:
            cluster_id: Cluster identifier to retrieve
            
        Returns:
            Optional[Cluster]: Cluster instance if found, None otherwise
        """
        log = logger.bind(context="ClusterManager.get_cluster")
        log.debug(f"Retrieving cluster {cluster_id}")
        return self.clusters.get(cluster_id)
    
    def get_all_clusters(self) -> List[Cluster]:
        """
        Retrieve all clusters managed by the ClusterManager.
        
        Returns:
            List[Cluster]: List of all Cluster instances
        """
        log = logger.bind(context="ClusterManager.get_all_clusters")
        log.debug("Retrieving all clusters")
        return list(self.clusters.values())
    
    def get_cluster_nodes(self, cluster_id: str, active_only: bool = True) -> List[str]:
        """
        Get the list of node IDs for a specific cluster.
        
        Args:
            cluster_id: Cluster identifier to query
            active_only: If True, return only active nodes; otherwise return all registered nodes
        
        Returns:
            List[str]: List of node IDs in the cluster
        """
        log = logger.bind(context="ClusterManager.get_cluster_nodes")
        log.debug(f"Getting nodes for cluster {cluster_id} (active_only={active_only})")
        
        if cluster_id not in self.clusters:
            log.warning(f"Cluster {cluster_id} does not exist.")
            return []
        
        cluster = self.get_cluster(cluster_id)
        
        if active_only:
            log.debug(f"Returning {len(cluster.active_nodes)} active nodes for cluster {cluster_id}.")
            return list(cluster.active_nodes)
        else:
            log.debug(f"Returning {len(cluster.active_nodes) + len(cluster.inactive_nodes)} total registered nodes for cluster {cluster_id}.")
            return list(cluster.active_nodes.union(cluster.inactive_nodes))
        
    def get_node_cluster(self, node_id: str) -> Optional[str]:
        """
        Get the cluster ID a node is registered to.
        
        Args:
            node_id: Node identifier to query
            
        Returns:
            Optional[str]: Cluster ID if node is registered, None otherwise
        """
        log = logger.bind(context="ClusterManager.get_node_cluster")
        log.debug(f"Getting cluster for node {node_id}")
        return self.node_to_cluster.get(node_id)
    
    def is_cluster_ready(self, cluster_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a specific cluster is ready based on active node threshold.
        
        Args:
            cluster_id: Cluster identifier to check
            threshold: Optional custom readiness threshold (0.0-1.0)
        Returns:
            bool: True if cluster is ready, False otherwise
        """
        log = logger.bind(context="ClusterManager.is_cluster_ready")
        log.debug(f"Checking readiness for cluster {cluster_id} with threshold {threshold if threshold else self.default_threshold}")
        
        if cluster_id not in self.clusters:
            log.warning(f"Cluster {cluster_id} does not exist.")
            return False

        if threshold is None:
            threshold = self.default_threshold
            
        cluster = self.clusters[cluster_id]
        is_ready = cluster.is_ready(threshold)
        
        log.info(f"Cluster {cluster_id} readiness: {is_ready} (threshold: {threshold})")
        log.debug(f"Cluster {cluster_id} readiness: {is_ready} "
                 f"({cluster.get_active_node_count()}/{cluster.get_expected_node_count()}, "
                 f"threshold={threshold})")
        
        return is_ready
    
    def get_ready_clusters(self, threshold: Optional[float] = None) -> List[str]:
        """
        Get a list of all clusters that are currently ready.
        
        Args:
            threshold: Optional custom readiness threshold (0.0-1.0)
        
        Returns:
            List[str]: List of cluster IDs that are ready
        """
        log = logger.bind(context="ClusterManager.get_ready_clusters")
        log.debug(f"Getting all ready clusters with threshold {threshold if threshold else self.default_threshold}")
        
        if threshold is None:
            threshold = self.default_threshold
            
        ready_clusters = [cluster_id for cluster_id, _ in self.clusters.items() if self.is_cluster_ready(cluster_id, threshold)]

        log.info(f"Found {len(ready_clusters)} ready clusters.")
        log.debug(f"Found {len(ready_clusters)} ready clusters (threshold={threshold})")
        return ready_clusters
    
    def get_total_expected_nodes(self) -> int:
        """Return the total number of expected nodes across all clusters."""
        log = logger.bind(context="ClusterManager.total_expected_nodes")
        total = sum(cluster.get_expected_node_count() for cluster in self.clusters.values())
        log.debug(f"Total expected nodes across all clusters: {total}")
        return total
    
    def get_total_active_nodes(self) -> int:
        """Return the total number of active nodes across all clusters."""
        log = logger.bind(context="ClusterManager.total_active_nodes")
        total = sum(cluster.get_active_node_count() for cluster in self.clusters.values())
        log.debug(f"Total active nodes across all clusters: {total}")
        return total
    
    def get_total_registered_nodes(self) -> int:
        """Return the total number of registered nodes (active + inactive) across all clusters."""
        log = logger.bind(context="ClusterManager.total_registered_nodes")
        total = sum(cluster.get_total_registered_nodes() for cluster in self.clusters.values())
        log.debug(f"Total registered nodes across all clusters: {total}")
        return total
    
    def get_cluster_count(self) -> int:
        """Return the total number of clusters managed."""
        log = logger.bind(context="ClusterManager.get_cluster_count")
        count = len(self.clusters)
        log.debug(f"Total number of clusters managed: {count}")
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about the ClusterManager state.
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics
        """
        log = logger.bind(context="ClusterManager.get_statistics")
        log.debug("Gathering ClusterManager statistics.")
        
        stats = {
            "cluster_count": self.get_cluster_count(),
            "total_expected_nodes": self.get_total_expected_nodes(),
            "total_active_nodes": self.get_total_active_nodes(),
            "total_registered_nodes": self.get_total_registered_nodes(),
            "total_registrations": self.total_registrations,
            "total_unregistrations": self.total_unregistrations,
            "uptime_seconds": time.time() - self.creation_time,
            "ready_clusters": len(self.get_ready_clusters()),
            "cluster_details": {}
        }
        
        # Add per-cluster details
        for cluster_id, cluster in self.clusters.items():
            stats["cluster_details"][cluster_id] = {
                "playstyle": cluster.playstyle,
                "expected_nodes": cluster.get_expected_node_count(),
                "active_nodes": cluster.get_active_node_count(),
                "inactive_nodes": len(cluster.inactive_nodes),
                "readiness_ratio": cluster.get_readiness_ratio(),
                "is_ready": cluster.is_ready(self.default_threshold),
                "last_activity": datetime.fromtimestamp(cluster.last_activity).strftime("%Y-%m-%d %H:%M:%S") if cluster.last_activity else "Never"
            }
        
        log.debug(f"Generated statistics: {len(stats)} top-level fields")
        return stats
    
    def _log_cluster_summary(self):
        """Log a summary of all clusters and their statuses."""
        log = logger.bind(context="ClusterManager._log_cluster_summary")
        log.info("Cluster Summary:")
        for cluster_id, cluster in self.clusters.items():
            log.info(f" - {cluster_id}: {cluster.playstyle}, "
                     f"Expected: {cluster.get_expected_node_count()}, "
                     f"Active: {cluster.get_active_node_count()}, "
                     f"Inactive: {len(cluster.inactive_nodes)}, "
                     f"Ready: {cluster.is_ready(self.default_threshold)}")
            
    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current cluster configuration to dictionary format.
        
        Returns:
            Dictionary in YAML-compatible format
        """
        log = logger.bind(context="ClusterManager.export_configuration")
        
        config = {
            "clusters": []
        }
        
        for cluster in self.clusters.values():
            cluster_config = {
                "id": cluster.cluster_id,
                "playstyle": cluster.playstyle,
                "node_count": cluster.node_count,
                "node_prefix": cluster.node_prefix,
                "description": cluster.description
            }
            config["clusters"].append(cluster_config)
        
        log.debug(f"Exported configuration with {len(config['clusters'])} clusters")
        return config
    
    def save_configuration(self, path: str):
        """
        Save current cluster configuration to a YAML file.
        
        Args:
            path (str): The file path to save the configuration
        """
        log = logger.bind(context="ClusterManager.save_configuration")
        log.info(f"Saving cluster configuration to {path}")
        config = self.export_configuration()
        
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(config, f)
            log.info(f"Cluster configuration successfully saved to {path}")
        except Exception as e:
            log.error(f"Failed to save configuration to {path}: {e}")
            raise e
        
    def get_expected_nodes(self, cluster_id: str) -> Set[str]:
        """Return a set of all expected node IDs across all clusters."""
        log = logger.bind(context="ClusterManager.get_expected_nodes")
        log.debug(f"Getting expected nodes for cluster {cluster_id}")
        expected_nodes = set()
        if cluster_id not in self.clusters:
            log.warning(f"Cluster {cluster_id} does not exist.")
            return expected_nodes
        
        cluster = self.clusters.get(cluster_id)
        if cluster:
            expected_nodes.update(cluster.expected_nodes)
        log.debug(f"Total expected nodes for cluster {cluster_id}: {len(expected_nodes)}")
        return expected_nodes

# Example usage:
if __name__ == "__main__":
    # Load from config file
    log = logger.bind(context="Main")
    manager = ClusterManager("/home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning/config/cluster_topology.yaml")

    # Register nodes
    manager.register_node("agg_001", "cluster_aggressive")
    manager.register_node("pos_001", "cluster_positional")
    manager.register_node("agg_002", "cluster_aggressive")
    manager.register_node("pos_002", "cluster_positional")
    manager.register_node("agg_003", "cluster_aggressive")
    manager.register_node("pos_003", "cluster_positional")
    manager.register_node("agg_004", "cluster_aggressive")
    manager.register_node("pos_004", "cluster_positional")

    expected_aggressive = manager.get_expected_nodes("cluster_aggressive")
    expected_positional = manager.get_expected_nodes("cluster_positional")
    
    log.info(f"Expected aggressive nodes: {expected_aggressive}")
    log.info(f"Expected positional nodes: {expected_positional}")
    # Check cluster readiness
    if manager.is_cluster_ready("cluster_aggressive"):
        nodes = manager.get_cluster_nodes("cluster_aggressive")
        # Start aggregation...

    # Get statistics
    stats = manager.get_statistics()
    # Formatting stats for better readability
    import pprint
    pprint.pprint(stats)
    
    manager.unregister_node("agg_002")
    manager.unregister_node("pos_002")
    
    stats = manager.get_statistics()
    pprint.pprint(stats)