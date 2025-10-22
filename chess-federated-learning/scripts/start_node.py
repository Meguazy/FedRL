#!/usr/bin/env python3
"""
Script to start a federated learning node.

Usage:
    python scripts/start_node.py --config config/nodes/agg_001.yaml
    python scripts/start_node.py --node-id agg_001 --cluster-id cluster_tactical
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add chess-federated-learning directory to path
chess_fl_dir = Path(__file__).parent.parent
sys.path.insert(0, str(chess_fl_dir))

from loguru import logger
from client.node import FederatedLearningNode
from client.config import NodeConfig, create_default_config
from client.trainer.trainer_interface import TrainingConfig
from client.trainer.factory import create_trainer


def setup_logging(config: NodeConfig):
    """
    Configure logging based on node configuration.
    
    Args:
        config: Node configuration
    """
    logger.remove()  # Remove default handler
    
    log_level = config.logging.get("level", "INFO")
    log_format = config.logging.get("format", "text")
    
    if log_format == "json":
        format_str = "{time} {level} {name}:{function}:{line} {message}"
    else:
        format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
    
    # Console logging
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_str,
        colorize=True
    )
    
    # File logging if specified
    log_file = config.logging.get("file")
    if log_file:
        # Clear the log file if it exists (fresh start each run)
        from pathlib import Path
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}",
            rotation="10 MB"
        )
        logger.info(f"Logging to file: {log_file}")


async def start_node_from_config(config: NodeConfig):
    """
    Start a node from configuration.

    Args:
        config: Node configuration
    """
    logger.info(f"Starting node {config.node_id} in cluster {config.cluster_id}")

    # Create training config
    training_config = TrainingConfig(**config.training)

    # Create trainer
    trainer = create_trainer(
        trainer_type=config.trainer_type,
        node_id=config.node_id,
        cluster_id=config.cluster_id,
        config=training_config
    )

    # Configure supervised trainer if applicable
    if config.trainer_type == "supervised":
        supervised_config = config.config.get("supervised", {})
        pgn_database_path = supervised_config.get("pgn_database_path")

        if pgn_database_path:
            trainer.set_pgn_database(pgn_database_path)
            logger.info(f"Configured supervised trainer with PGN database: {pgn_database_path}")
        else:
            logger.warning("Supervised trainer configured but no PGN database path specified")
    
    # Configure puzzle trainer if applicable
    elif config.trainer_type == "puzzle":
        puzzle_config = config.config.get("puzzle", {})
        puzzle_database_path = puzzle_config.get("puzzle_database_path")
        redis_host = puzzle_config.get("redis_host", "localhost")
        redis_port = puzzle_config.get("redis_port", 6381)
        min_rating = puzzle_config.get("min_rating", 1500)
        max_rating = puzzle_config.get("max_rating", 2500)
        themes = puzzle_config.get("themes")
        
        if puzzle_database_path:
            trainer.puzzle_database_path = puzzle_database_path
            logger.info(f"Configured puzzle trainer with database: {puzzle_database_path}")
        else:
            logger.warning("Puzzle trainer configured but no puzzle database path specified")
        
        # Update Redis configuration (should already be set from factory, but ensure it's correct)
        trainer.redis_cache.host = redis_host
        trainer.redis_cache.port = redis_port
        
        # Update puzzle filters
        trainer.min_rating = min_rating
        trainer.max_rating = max_rating
        if themes:
            trainer.themes = themes
        
        logger.info(f"Puzzle trainer config: Redis={redis_host}:{redis_port}, "
                   f"rating={min_rating}-{max_rating}, themes={themes if themes else 'all'}")

    # Create node
    node = FederatedLearningNode(
        node_id=config.node_id,
        cluster_id=config.cluster_id,
        trainer=trainer,
        server_host=config.server_host,
        server_port=config.server_port,
        auto_reconnect=config.auto_reconnect
    )

    # Start node
    try:
        await node.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Node failed: {e}")
        raise
    finally:
        logger.info("Node shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start a federated learning node"
    )
    
    # Configuration source
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Or specify parameters directly
    parser.add_argument(
        "--node-id",
        type=str,
        help="Node identifier (e.g., agg_001)"
    )
    parser.add_argument(
        "--cluster-id",
        type=str,
        help="Cluster identifier (e.g., cluster_tactical)"
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
    parser.add_argument(
        "--trainer-type",
        type=str,
        default="dummy",
        choices=["dummy", "supervised", "alphazero"],
        help="Trainer type (default: dummy)"
    )
    parser.add_argument(
        "--games-per-round",
        type=int,
        default=100,
        help="Games per training round (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        # Load from YAML file
        config = NodeConfig.from_yaml(args.config)
    elif args.node_id and args.cluster_id:
        # Create from command-line arguments
        config = create_default_config(args.node_id, args.cluster_id)
        config.server_host = args.server_host
        config.server_port = args.server_port
        config.trainer_type = args.trainer_type
        config.training["games_per_round"] = args.games_per_round
    else:
        parser.error("Must specify either --config or both --node-id and --cluster-id")
    
    # Setup logging
    setup_logging(config)
    
    # Print configuration
    logger.info("Node Configuration:")
    logger.info(f"  Node ID: {config.node_id}")
    logger.info(f"  Cluster ID: {config.cluster_id}")
    logger.info(f"  Server: {config.server_host}:{config.server_port}")
    logger.info(f"  Trainer: {config.trainer_type}")
    logger.info(f"  Games per round: {config.training.get('games_per_round', 100)}")
    
    # Start node
    asyncio.run(start_node_from_config(config))


if __name__ == "__main__":
    main()