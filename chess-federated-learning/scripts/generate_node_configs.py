#!/usr/bin/env python3
"""
Script to auto-generate node configuration files from cluster topology.

This script reads the cluster_topology.yaml file and generates individual
YAML configuration files for each node based on the cluster definitions.

Usage:
    python scripts/generate_node_configs.py
    python scripts/generate_node_configs.py --topology config/cluster_topology.yaml
    python scripts/generate_node_configs.py --output-dir config/nodes
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add chess-federated-learning directory to path
chess_fl_dir = Path(__file__).parent.parent
sys.path.insert(0, str(chess_fl_dir))

import yaml
from loguru import logger


def load_cluster_topology(topology_path: str) -> Dict[str, Any]:
    """
    Load cluster topology configuration.

    Args:
        topology_path: Path to cluster topology YAML file

    Returns:
        Topology configuration dictionary
    """
    topology_file = Path(topology_path)
    if not topology_file.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    with open(topology_file, 'r') as f:
        topology = yaml.safe_load(f)

    logger.info(f"Loaded cluster topology from {topology_path}")
    return topology


def generate_node_config(
    node_id: str,
    cluster_id: str,
    cluster_info: Dict[str, Any],
    server_host: str = "localhost",
    server_port: int = 8765,
    trainer_type: str = "dummy",
    pgn_database_path: str = None,
    puzzle_database_path: str = None
) -> Dict[str, Any]:
    """
    Generate configuration for a single node.

    Args:
        node_id: Node identifier
        cluster_id: Cluster identifier
        cluster_info: Cluster configuration dictionary
        server_host: FL server hostname
        server_port: FL server port
        trainer_type: Type of trainer to use
        pgn_database_path: Path to PGN database (for supervised trainer)
        puzzle_database_path: Path to puzzle database (for puzzle trainer)

    Returns:
        Node configuration dictionary
    """
    # Extracting playstyle and returning and error if not found
    if "playstyle" not in cluster_info:
        raise ValueError(f"Cluster info for {cluster_id} missing 'playstyle' key")
    if cluster_info["playstyle"] not in ["tactical", "positional", "defensive", "balanced"]:
        raise ValueError(f"Invalid playstyle '{cluster_info['playstyle']}' for cluster {cluster_id}")
    playstyle = cluster_info["playstyle"]

    # Get games_per_round from cluster config or use defaults based on trainer type
    if "games_per_round" in cluster_info:
        games_per_round = cluster_info["games_per_round"]
    elif trainer_type == "puzzle":
        games_per_round = 500  # Default for puzzle trainer
    else:
        games_per_round = 200  # Default for supervised/other trainers
    
    # Determine batch_size based on trainer type
    batch_size = 64 if trainer_type == "puzzle" else 32

    config = {
        "node_id": node_id,
        "cluster_id": cluster_id,
        "server_host": server_host,
        "server_port": server_port,
        "trainer_type": trainer_type,
        "auto_reconnect": True,
        "training": {
            "games_per_round": games_per_round,
            "batch_size": batch_size,
            "learning_rate": 0.003,
            "exploration_factor": 1.0,
            "max_game_length": 200,
            "save_games": True,
            "playstyle": playstyle,
        },
        "storage": {
            "enabled": True,
            "base_path": "./storage",
            "save_models": True,
            "save_metrics": True,
        },
        "logging": {
            "level": "INFO",
            "file": f"./logs/{node_id}.log",
            "format": "text",
        }
    }

    # Add supervised learning specific config
    if trainer_type == "supervised":
        config["supervised"] = {
            "pgn_database_path": pgn_database_path or "./data/databases/lichess_db.pgn.zst",
            "min_rating": 2000,
            "skip_opening_moves": 10,
            "skip_endgame_pieces": 6,
            "sample_rate": 1.0,
        }
    
    # Add puzzle learning specific config
    elif trainer_type == "puzzle":
        # Define themes based on playstyle
        if playstyle == "tactical":
            themes = [
                "fork", "pin", "skewer", "discoveredAttack", "sacrifice",
                "attackingF2F7", "doubleCheck", "deflection", "attraction", "clearance"
            ]
        elif playstyle == "positional":
            themes = [
                "endgame", "advantage", "crushing", "mate", "mateIn2", "mateIn3",
                "queenRookEndgame", "bishopEndgame", "pawnEndgame", "rookEndgame"
            ]
        else:
            # Default to mixed themes
            themes = [
                "fork", "pin", "skewer", "endgame", "mate", "mateIn2"
            ]
        
        config["puzzle"] = {
            "puzzle_database_path": puzzle_database_path or "/home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning/data/databases/lichess_puzzles.csv.zst",
            "redis_host": "localhost",
            "redis_port": 6381,
            "min_rating": 1600,
            "max_rating": 2400,
            "themes": themes,
        }

    return config


def save_node_config(config: Dict[str, Any], output_path: Path):
    """
    Save node configuration to YAML file.

    Args:
        config: Node configuration dictionary
        output_path: Path to save configuration file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Add header comment
        f.write(f"# Configuration for node {config['node_id']}\n")
        f.write(f"# Cluster: {config['cluster_id']}\n")
        f.write(f"# Auto-generated by generate_node_configs.py\n\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Saved configuration to {output_path}")


def generate_all_configs(
    topology_path: str,
    output_dir: str,
    server_host: str = "localhost",
    server_port: int = 8765,
    trainer_type: str = "dummy",
    pgn_database_path: str = None,
    puzzle_database_path: str = None
) -> List[str]:
    """
    Generate configuration files for all nodes in the topology.

    Args:
        topology_path: Path to cluster topology YAML file
        output_dir: Directory to save node configurations
        server_host: FL server hostname
        server_port: FL server port
        trainer_type: Type of trainer to use
        pgn_database_path: Path to PGN database (for supervised trainer)
        puzzle_database_path: Path to puzzle database (for puzzle trainer)

    Returns:
        List of generated configuration file paths
    """
    # Load topology
    topology = load_cluster_topology(topology_path)
    clusters = topology.get("clusters", [])

    if not clusters:
        logger.warning("No clusters defined in topology")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []
    total_nodes = 0

    # Generate configs for each cluster
    for cluster in clusters:
        cluster_id = cluster["id"]
        node_count = cluster["node_count"]
        node_prefix = cluster["node_prefix"]

        logger.info(f"Generating {node_count} nodes for cluster {cluster_id}")

        # Generate configs for each node in this cluster
        for i in range(1, node_count + 1):
            node_id = f"{node_prefix}_{i:03d}"

            # Generate config
            config = generate_node_config(
                node_id=node_id,
                cluster_id=cluster_id,
                cluster_info=cluster,
                server_host=server_host,
                server_port=server_port,
                trainer_type=trainer_type,
                pgn_database_path=pgn_database_path,
                puzzle_database_path=puzzle_database_path
            )

            # Save config
            config_file = output_path / f"{node_id}.yaml"
            save_node_config(config, config_file)
            generated_files.append(str(config_file))
            total_nodes += 1

    logger.info(f"Generated {total_nodes} node configurations in {output_dir}")
    return generated_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate node configuration files from cluster topology"
    )

    parser.add_argument(
        "--topology",
        type=str,
        default="config/cluster_topology.yaml",
        help="Path to cluster topology YAML file (default: config/cluster_topology.yaml)"
    )
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
    
    # Mutually exclusive group for trainer type
    trainer_group = parser.add_mutually_exclusive_group(required=True)
    trainer_group.add_argument(
        "--puzzle",
        action="store_true",
        help="Generate puzzle trainer configurations (saves to config/nodes/puzzle_configs/)"
    )
    trainer_group.add_argument(
        "--supervised",
        action="store_true",
        help="Generate supervised trainer configurations (saves to config/nodes/supervised_configs/)"
    )
    
    parser.add_argument(
        "--pgn-database",
        type=str,
        default=None,
        help="Path to PGN database for supervised learning (default: ./data/databases/lichess_db.pgn.zst)"
    )
    parser.add_argument(
        "--puzzle-database",
        type=str,
        default=None,
        help="Path to puzzle database for puzzle learning (default: /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning/data/databases/lichess_puzzles.csv.zst)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Determine trainer type and output directory
    if args.puzzle:
        trainer_type = "puzzle"
        output_dir = "chess-federated-learning/config/nodes/puzzle_configs"
    elif args.supervised:
        trainer_type = "supervised"
        output_dir = "chess-federated-learning/config/nodes/supervised_configs"
    else:
        # Should never reach here due to required=True
        parser.error("Must specify either --puzzle or --supervised")

    # Setup logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.info("Starting node configuration generation")
    logger.info(f"Topology file: {args.topology}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Server: {args.server_host}:{args.server_port}")
    logger.info(f"Trainer type: {trainer_type}")
    
    if trainer_type == "supervised":
        pgn_path = args.pgn_database or "./data/databases/lichess_db.pgn.zst"
        logger.info(f"PGN database: {pgn_path}")
    elif trainer_type == "puzzle":
        puzzle_path = args.puzzle_database or "/home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning/data/databases/lichess_puzzles.csv.zst"
        logger.info(f"Puzzle database: {puzzle_path}")

    try:
        generated_files = generate_all_configs(
            topology_path=args.topology,
            output_dir=output_dir,
            server_host=args.server_host,
            server_port=args.server_port,
            trainer_type=trainer_type,
            pgn_database_path=args.pgn_database,
            puzzle_database_path=args.puzzle_database
        )

        logger.success(f"Successfully generated {len(generated_files)} configuration files")

        if args.verbose:
            logger.info("Generated files:")
            for file in generated_files:
                logger.info(f"  - {file}")

    except Exception as e:
        logger.error(f"Failed to generate configurations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
