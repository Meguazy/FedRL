#!/usr/bin/env python3
"""
Script to launch multiple federated learning nodes in parallel using asyncio.

This script can launch nodes in multiple ways:
1. From a directory of config files
2. From cluster topology (auto-generate configs)
3. Specific node IDs

All nodes run in parallel as async tasks, each in its own event loop context.

Usage:
    # Launch all nodes from config directory
    python scripts/start_all_nodes.py --config-dir config/nodes

    # Launch nodes from topology
    python scripts/start_all_nodes.py --topology config/cluster_topology.yaml

    # Launch specific nodes
    python scripts/start_all_nodes.py --nodes agg_001,agg_002,pos_001

    # Launch with specific server
    python scripts/start_all_nodes.py --config-dir config/nodes --server-host 192.168.1.100

    # Launch limited number for testing
    python scripts/start_all_nodes.py --topology config/cluster_topology.yaml --limit 4
"""

import asyncio
import argparse
import sys
import signal
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add chess-federated-learning directory to path
chess_fl_dir = Path(__file__).parent.parent
sys.path.insert(0, str(chess_fl_dir))

from loguru import logger
import yaml
from client.node import FederatedLearningNode
from client.config import NodeConfig
from client.trainer.factory import create_trainer
from client.trainer.trainer_interface import TrainingConfig


class NodeLauncher:
    """Manager for launching and monitoring multiple nodes in parallel."""

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8765,
        stagger_delay: float = 0.1
    ):
        """
        Initialize node launcher.

        Args:
            server_host: FL server hostname
            server_port: FL server port
            stagger_delay: Delay between starting each node (seconds)
        """
        self.server_host = server_host
        self.server_port = server_port
        self.stagger_delay = stagger_delay
        self.nodes: List[FederatedLearningNode] = []
        self.node_configs: List[NodeConfig] = []
        self.threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

    def load_configs_from_directory(self, config_dir: str) -> List[NodeConfig]:
        """
        Load all node configurations from directory.

        Args:
            config_dir: Directory containing node YAML configs

        Returns:
            List of NodeConfig objects
        """
        config_path = Path(config_dir)
        if not config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        configs = []
        yaml_files = sorted(config_path.glob("*.yaml"))

        if not yaml_files:
            logger.warning(f"No YAML files found in {config_dir}")
            return configs

        for yaml_file in yaml_files:
            try:
                config = NodeConfig.from_yaml(str(yaml_file))
                # Override server settings if specified
                config.server_host = self.server_host
                config.server_port = self.server_port
                configs.append(config)
            except Exception as e:
                logger.warning(f"Failed to load {yaml_file}: {e}")

        logger.info(f"Loaded {len(configs)} node configurations from {config_dir}")
        return configs

    def load_configs_from_topology(self, topology_path: str) -> List[NodeConfig]:
        """
        Load node configurations from cluster topology.

        Args:
            topology_path: Path to cluster topology YAML

        Returns:
            List of NodeConfig objects
        """
        topology_file = Path(topology_path)
        if not topology_file.exists():
            raise FileNotFoundError(f"Topology file not found: {topology_path}")

        with open(topology_file, 'r') as f:
            topology = yaml.safe_load(f)

        clusters = topology.get("clusters", [])
        configs = []

        for cluster in clusters:
            cluster_id = cluster["id"]
            node_count = cluster["node_count"]
            node_prefix = cluster["node_prefix"]
            playstyle = cluster.get("playstyle", "balanced")

            for i in range(1, node_count + 1):
                node_id = f"{node_prefix}_{i:03d}"

                config = NodeConfig(
                    node_id=node_id,
                    cluster_id=cluster_id,
                    server_host=self.server_host,
                    server_port=self.server_port,
                    trainer_type="dummy",
                    auto_reconnect=True,
                    training={
                        "games_per_round": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "exploration_factor": 1.0,
                        "max_game_length": 200,
                        "save_games": True,
                        "playstyle": playstyle,
                    },
                    storage={
                        "enabled": True,
                        "base_path": "./storage",
                        "save_models": True,
                        "save_metrics": True,
                    },
                    logging={
                        "level": "INFO",
                        "file": f"./logs/{node_id}.log",
                        "format": "text",
                    }
                )
                configs.append(config)

        logger.info(f"Generated {len(configs)} node configurations from topology")
        return configs

    def filter_configs(
        self,
        configs: List[NodeConfig],
        node_ids: Optional[List[str]] = None,
        cluster_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[NodeConfig]:
        """
        Filter configurations by node ID, cluster ID, or limit.

        Args:
            configs: List of node configurations
            node_ids: List of specific node IDs to include
            cluster_ids: List of specific cluster IDs to include
            limit: Maximum number of nodes to include

        Returns:
            Filtered list of configurations
        """
        filtered = configs

        # Filter by node IDs
        if node_ids:
            node_id_set = set(node_ids)
            filtered = [c for c in filtered if c.node_id in node_id_set]

        # Filter by cluster IDs
        if cluster_ids:
            cluster_id_set = set(cluster_ids)
            filtered = [c for c in filtered if c.cluster_id in cluster_id_set]

        # Apply limit
        if limit and limit > 0:
            filtered = filtered[:limit]

        logger.info(f"Filtered to {len(filtered)} nodes")
        return filtered

    def create_node(self, config: NodeConfig) -> FederatedLearningNode:
        """
        Create a node instance from configuration.

        Args:
            config: Node configuration

        Returns:
            FederatedLearningNode instance
        """
        # Create training config
        training_config = TrainingConfig(**config.training)

        # Create trainer
        trainer = create_trainer(
            trainer_type=config.trainer_type,
            node_id=config.node_id,
            cluster_id=config.cluster_id,
            config=training_config
        )

        # Create node
        node = FederatedLearningNode(
            node_id=config.node_id,
            cluster_id=config.cluster_id,
            trainer=trainer,
            server_host=config.server_host,
            server_port=config.server_port,
            auto_reconnect=config.auto_reconnect
        )

        return node

    def run_node_in_thread(self, node: FederatedLearningNode, index: int):
        """
        Run a single node in its own thread with its own event loop.

        Args:
            node: Node instance to run
            index: Node index for staggered startup
        """
        # Stagger node startup to avoid overwhelming the server
        if index > 0 and self.stagger_delay > 0:
            time.sleep(self.stagger_delay * index)

        logger.info(f"Starting node {node.node_id} in thread {threading.current_thread().name}")

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(node.start())
        except KeyboardInterrupt:
            logger.info(f"Node {node.node_id} interrupted")
        except Exception as e:
            logger.error(f"Node {node.node_id} failed: {e}")
        finally:
            # Cleanup
            try:
                if node.client.is_connected():
                    loop.run_until_complete(node.stop())
            except Exception as e:
                logger.error(f"Error stopping node {node.node_id}: {e}")

            loop.close()
            logger.debug(f"Node {node.node_id} stopped")

    def launch_all(self, configs: List[NodeConfig]):
        """
        Launch all nodes in parallel threads.

        Args:
            configs: List of node configurations
        """
        self.node_configs = configs

        if not configs:
            logger.warning("No node configurations to launch")
            return

        logger.info(f"Launching {len(configs)} nodes in parallel threads...")

        # Create all nodes
        for config in configs:
            try:
                node = self.create_node(config)
                self.nodes.append(node)
            except Exception as e:
                logger.error(f"Failed to create node {config.node_id}: {e}")

        # Launch all nodes in separate threads
        for i, node in enumerate(self.nodes):
            thread = threading.Thread(
                target=self.run_node_in_thread,
                args=(node, i),
                name=f"Node-{node.node_id}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

        logger.success(f"Successfully launched {len(self.threads)} node threads")

        # Wait for all threads to complete
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down...")
            self.shutdown_all()

    def shutdown_all(self):
        """Shutdown all running nodes."""
        logger.info(f"Shutting down {len(self.nodes)} nodes...")

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for all threads to complete with timeout
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        logger.success("All nodes shut down")

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all nodes.

        Returns:
            Dictionary with status information
        """
        running = sum(1 for node in self.nodes if node.client.is_connected())

        return {
            "total": len(self.nodes),
            "running": running,
            "stopped": len(self.nodes) - running,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "cluster_id": node.cluster_id,
                    "state": node.lifecycle_state.value,
                    "connected": node.client.is_connected()
                }
                for node in self.nodes
            ]
        }


def main_sync(args):
    """Main function."""
    # Create launcher
    launcher = NodeLauncher(
        server_host=args.server_host,
        server_port=args.server_port,
        stagger_delay=args.delay
    )

    # Load configurations
    try:
        if args.config_dir:
            logger.info(f"Loading configurations from {args.config_dir}")
            configs = launcher.load_configs_from_directory(args.config_dir)
        elif args.topology:
            logger.info(f"Loading configurations from topology {args.topology}")
            configs = launcher.load_configs_from_topology(args.topology)
        elif args.nodes:
            # For specific nodes, need to load from config dir or topology
            logger.error("--nodes requires --config-dir or --topology to be specified")
            sys.exit(1)
        else:
            # Default: try to load from default config directory
            default_config_dir = chess_fl_dir / "config" / "nodes"
            if default_config_dir.exists():
                logger.info(f"Using default config directory: {default_config_dir}")
                configs = launcher.load_configs_from_directory(str(default_config_dir))
            else:
                logger.error("No configuration source specified. Use --config-dir or --topology")
                sys.exit(1)

        # Apply filters
        node_ids = args.nodes.split(',') if args.nodes else None
        cluster_ids = args.clusters.split(',') if args.clusters else None
        configs = launcher.filter_configs(
            configs,
            node_ids=node_ids,
            cluster_ids=cluster_ids,
            limit=args.limit
        )

        if not configs:
            logger.error("No nodes to launch after filtering")
            sys.exit(1)

        # Setup signal handlers for graceful shutdown
        def signal_handler(_signum, _frame):
            logger.info("Signal received, initiating shutdown...")
            launcher.shutdown_all()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Launch nodes
        launcher.launch_all(configs)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Failed to launch nodes: {e}")
        sys.exit(1)
    finally:
        launcher.shutdown_all()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch multiple federated learning nodes in parallel"
    )

    # Configuration source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing node YAML configs"
    )
    source_group.add_argument(
        "--topology",
        type=str,
        help="Path to cluster topology YAML"
    )
    source_group.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node IDs to launch"
    )

    # Filtering
    parser.add_argument(
        "--clusters",
        type=str,
        help="Comma-separated list of cluster IDs to include"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of nodes to launch"
    )

    # Server connection
    parser.add_argument(
        "--server-host",
        type=str,
        default="localhost",
        help="FL server hostname (default: localhost)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8765,
        help="FL server port (default: 8765)"
    )

    # Launch options
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between starting each node (seconds, default: 0.1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # Run main
    try:
        main_sync(args)
    except KeyboardInterrupt:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
