#!/usr/bin/env python3
"""
Automated training script for B1 experiment.
Starts server and immediately begins training without interactive menu.
"""

import asyncio
import signal
import sys
import argparse
from pathlib import Path
from loguru import logger
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.aggregation.base_aggregator import AggregationMetrics
from server.communication.server_socket import FederatedLearningServer
from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
from server.storage.factory import create_experiment_tracker
from server.main import TrainingOrchestrator, RoundConfig, EvaluationConfig


async def run_automated_training(
    num_rounds: int,
    experiment_name: str,
    cluster_config_path: str,
    server_config_path: str
):
    """Run training automatically without interactive menu."""

    log = logger.bind(context="run_training")
    log.info(f"Starting automated training: {experiment_name}")
    log.info(f"Rounds: {num_rounds}")

    # Load server config
    try:
        with open(server_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            server_config = config_data.get("server_config", {})
            orchestrator_config = config_data.get("orchestrator_config", {})
            evaluation_config_data = config_data.get("evaluation_config", {})
        log.info("Loaded configuration from YAML")
    except FileNotFoundError:
        log.error(f"Config file {server_config_path} not found")
        return False

    server_host = server_config.get("host", "localhost")
    server_port = server_config.get("port", 8765)

    # Create server
    log.info(f"Creating server at {server_host}:{server_port}")
    server = FederatedLearningServer(
        host=server_host,
        port=server_port,
        cluster_config_path=cluster_config_path
    )

    # Start server in background
    server_task = asyncio.create_task(server.start_server())

    # Wait for server to start
    log.info("Waiting for server to start...")
    await asyncio.sleep(3.0)

    if not server.is_running:
        log.error("Server failed to start. Exiting.")
        return False

    log.info("Server started successfully")

    # Create round configuration
    round_config = RoundConfig(
        aggregation_threshold=orchestrator_config.get("aggregation_threshold", 0.8),
        timeout_seconds=orchestrator_config.get("timeout_seconds", 1200),
        shared_layer_patterns=orchestrator_config.get("shared_layer_patterns", []),
        cluster_specific_patterns=orchestrator_config.get("cluster_specific_patterns", []),
        checkpoint_interval=orchestrator_config.get("checkpoint_interval", 5)
    )

    # Create evaluation configuration
    evaluation_config = EvaluationConfig(
        enabled=evaluation_config_data.get("enabled", True),
        interval_rounds=evaluation_config_data.get("interval_rounds", 10),
        games_per_elo_level=evaluation_config_data.get("games_per_elo_level", 10),
        stockfish_elo_levels=evaluation_config_data.get("stockfish_elo_levels", [1000, 1200, 1400]),
        time_per_move=evaluation_config_data.get("time_per_move", 0.1),
        skip_check_positions=evaluation_config_data.get("skip_check_positions", True),
        stockfish_path=evaluation_config_data.get("stockfish_path"),
        enable_delta_analysis=evaluation_config_data.get("enable_delta_analysis", True),
        delta_sampling_rate=evaluation_config_data.get("delta_sampling_rate", 3),
        stockfish_depth=evaluation_config_data.get("stockfish_depth", 12)
    )

    # Create storage backend
    log.info("Initializing storage backend...")
    tracker = create_experiment_tracker(
        base_path="./storage",
        compression=True,
        keep_last_n=None,
        keep_best=True
    )
    log.info("Storage backend initialized")

    # Create orchestrator
    log.info("Creating training orchestrator...")
    orchestrator = TrainingOrchestrator(
        server=server,
        round_config=round_config,
        storage_tracker=tracker,
        evaluation_config=evaluation_config
    )
    log.info("Orchestrator initialized")

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        log.info("Shutdown signal received")
        orchestrator.is_running = False
        if not server_task.done():
            server_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Wait for nodes to connect
    log.info("Waiting for nodes to connect (30 seconds)...")
    await asyncio.sleep(30)

    # Check node connections
    connected_nodes = len(server.get_connected_nodes())
    log.info(f"Connected nodes: {connected_nodes}")

    if connected_nodes == 0:
        log.error("No nodes connected. Cannot start training.")
        await server.stop_server()
        return False

    # Start training automatically
    log.info(f"Starting training: {num_rounds} rounds")
    log.info(f"Experiment name: {experiment_name}")

    try:
        await orchestrator.run_training(
            num_rounds=num_rounds,
            experiment_name=experiment_name
        )
        log.info("Training completed successfully!")
        success = True
    except Exception as e:
        log.error(f"Training failed: {e}")
        log.exception("Full traceback:")
        success = False
    finally:
        log.info("Shutting down server...")
        await server.stop_server()

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run federated learning training automatically"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name"
    )
    parser.add_argument(
        "--server-config",
        type=str,
        default="chess-federated-learning/config/server_config.yaml",
        help="Path to server config file"
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="chess-federated-learning/config/cluster_topology.yaml",
        help="Path to cluster topology file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: DEBUG)"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Run training
    success = asyncio.run(run_automated_training(
        num_rounds=args.rounds,
        experiment_name=args.experiment,
        cluster_config_path=args.cluster_config,
        server_config_path=args.server_config
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
