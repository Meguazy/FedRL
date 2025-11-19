"""
Federated Learning Server - Main Orchestration

This module implements the main server orchestration logic that coordinates
the complete federated learning workflow. It manages the training loop,
triggers aggregation, and integrates all server components.

Key Responsibilities:
    - Coordinate training rounds across all nodes
    - Trigger intra-cluster and inter-cluster aggregation
    - Manage model distribution to nodes
    - Integrate storage layer for metrics and checkpoints
    - Handle server lifecycle and graceful shutdown

Architecture:
    FederatedLearningServer (communication) +
    ClusterManager (topology) +
    Aggregators (model combination) +
    FileExperimentTracker (storage) +
    Orchestrator (this module)
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from loguru import logger
import time
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.aggregation.base_aggregator import AggregationMetrics
from server.communication.server_socket import FederatedLearningServer
from server.communication.protocol import Message, MessageFactory, MessageType
from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
from server.cluster_manager import ClusterManager
from server.storage.base import EntityType
from server.storage.experiment_tracker import FileExperimentTracker
from server.storage.factory import create_experiment_tracker
from server.evaluation.model_evaluator import ModelEvaluator


@dataclass
class RoundConfig:
    """Configuration for each training round."""
    aggregation_threshold: float = 0.8  # Fraction of nodes required to proceed
    timeout_seconds: int = 300  # Max wait time for nodes
    intra_cluster_weighting: str = "samples"  # "samples" or "uniform"
    inter_cluster_weighting: str = "uniform"  # "samples" or "uniform"
    shared_layer_patterns: list = field(default_factory=lambda: ["input_conv.*"])
    cluster_specific_patterns: list = field(default_factory=lambda: ["policy_head.*", "value_head.*"])
    checkpoint_interval: int = 5  # Save every N rounds
    # Note: games_per_round is now configured per-cluster in cluster_topology.yaml


@dataclass
class EvaluationConfig:
    """Configuration for playstyle evaluation."""
    enabled: bool = True
    interval_rounds: int = 10  # Run evaluation every N rounds
    games_per_elo_level: int = 10
    stockfish_elo_levels: list = field(default_factory=lambda: [1000, 1200, 1400])
    time_per_move: float = 0.1
    skip_check_positions: bool = True
    stockfish_path: Optional[str] = None
    # Enhanced metrics
    enable_delta_analysis: bool = True
    delta_sampling_rate: int = 3
    stockfish_depth: int = 12
    
    def from_yaml(self, path: str):
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
                
class TrainingOrchestrator:
    """
    Orchestrates the federated learning training process.
    
    This class coordinates the complete training workflow:
    1. Broadcast START_TRAINING to all nodes
    2. Wait for MODEL_UPDATE from nodes (with threshold)
    3. Perform intra-cluster aggregation
    4. Perform inter-cluster selective aggregation
    5. Broadcast CLUSTER_MODEL back to nodes
    6. Log metrics and save checkpoints
    7. Repeat for multiple rounds
    
    The orchestrator integrates all server components and manages
    the timing and coordination of the federated learning process.
    """
    
    def __init__(
        self,
        server: FederatedLearningServer,
        round_config: RoundConfig,
        storage_tracker: Optional[FileExperimentTracker] = None,
        evaluation_config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize the orchestrator with server and configuration.
        
        Args:
            server: FederatedLearningServer instance
            round_config: Configuration for training rounds
            storage_tracker: Optional FileExperimentTracker for metrics
        """
        log = logger.bind(context="TrainingOrchestrator.__init__")
        log.info("Initializing Training Orchestrator...")
        
        # Core components
        self.server = server
        self.round_config = round_config
        self.evaluation_config = evaluation_config or EvaluationConfig()

        # Initialize or create storage tracker
        if storage_tracker is None:
            log.info("No storage tracker provided, creating default")
            self.storage_tracker = create_experiment_tracker(
                base_path="./storage",
                compression=True,
                keep_last_n=None,  # Keep all checkpoints
                keep_best=True
            )
        else:
            self.storage_tracker = storage_tracker

        # Initialize model evaluator if evaluation is enabled
        if self.evaluation_config.enabled:
            try:
                self.model_evaluator = ModelEvaluator(
                    stockfish_path=self.evaluation_config.stockfish_path,
                    device='cpu',  # TODO: Make configurable
                    skip_check_positions=self.evaluation_config.skip_check_positions,
                    enable_delta_analysis=self.evaluation_config.enable_delta_analysis,
                    delta_sampling_rate=self.evaluation_config.delta_sampling_rate,
                    stockfish_depth=self.evaluation_config.stockfish_depth
                )
                log.info("Model evaluator initialized for playstyle analysis")
            except Exception as e:
                log.warning(f"Failed to initialize model evaluator: {e}")
                log.warning("Playstyle evaluation will be disabled")
                self.evaluation_config.enabled = False
                self.model_evaluator = None
        else:
            self.model_evaluator = None
            log.info("Playstyle evaluation is disabled")
        
        # Aggregators
        self.intra_aggregator = IntraClusterAggregator(
            framework="pytorch",
            weighting_strategy=self.round_config.intra_cluster_weighting,
            experiment_tracker=self.storage_tracker
        )
        self.inter_aggregator = InterClusterAggregator(
            framework="pytorch",
            weighting_strategy=self.round_config.inter_cluster_weighting,
            shared_layer_patterns=self.round_config.shared_layer_patterns,
            cluster_specific_patterns=self.round_config.cluster_specific_patterns,
            experiment_tracker=self.storage_tracker
        )
        
        # State tracking
        self.current_round = 0
        self.starting_round = 0  # For resume training: offset for checkpoint naming
        self.current_run_id: Optional[str] = None
        self.is_running = False
        self.cluster_models: Dict[str, Dict[str, Any]] = {}  # cluster_id -> model_state
        self.pending_updates: Dict[str, Set[str]] = {}  # cluster_id -> set of pending node_ids
        
        log.info("Training orchestrator initialized")
        log.info(f"Intra-cluster weighting: {round_config.intra_cluster_weighting}")
        log.info(f"Inter-cluster weighting: {round_config.inter_cluster_weighting}")
        log.info(f"Aggregation threshold: {round_config.aggregation_threshold * 100}%")
        log.info(f"Checkpoint interval: every {round_config.checkpoint_interval} rounds")

    async def run_training(self, num_rounds: int, experiment_name: str = "federated_chess_training"):
        """
        Run the federated training process for a specified number of rounds.
        
        Args:
            num_rounds: Total number of training rounds to execute
            experiment_name: Name for this training experiment
        """
        log = logger.bind(context="TrainingOrchestrator.run_training")
        log.info(f"Starting federated training for {num_rounds} rounds")
        
        self.is_running = True
        
        # Create experiment configuration
        # Note: games_per_round is per-cluster, logged separately in cluster configs
        experiment_config = {
            'num_rounds': num_rounds,
            'aggregation_threshold': self.round_config.aggregation_threshold,
            'intra_cluster_weighting': self.round_config.intra_cluster_weighting,
            'inter_cluster_weighting': self.round_config.inter_cluster_weighting,
            'shared_layer_patterns': self.round_config.shared_layer_patterns,
            'cluster_specific_patterns': self.round_config.cluster_specific_patterns,
            'checkpoint_interval': self.round_config.checkpoint_interval,
            'server_config': {
                'host': self.server.host,
                'port': self.server.port
            },
            'clusters': {
                cluster.cluster_id: {
                    'playstyle': cluster.playstyle,
                    'games_per_round': cluster.games_per_round,
                    'node_count': cluster.node_count
                }
                for cluster in self.server.cluster_manager.get_all_clusters()
            }
        }
        
        # Start experiment tracking
        try:
            self.current_run_id = await self.storage_tracker.start_run(
                config=experiment_config,
                description=f"Federated learning training: {experiment_name}"
            )
            log.info(f"Started experiment tracking with run_id: {self.current_run_id}")
        except Exception as e:
            log.error(f"Failed to start experiment tracking: {e}")
            log.warning("Continuing without experiment tracking")
            self.current_run_id = None

        # Set starting round for resume training (affects checkpoint naming)
        if self.server.cluster_manager.resume_training_enabled:
            self.starting_round = self.server.cluster_manager.starting_round
            log.info(f"Resume training enabled: starting from round {self.starting_round}")
        else:
            self.starting_round = 0

        # Load and broadcast initial models if configured (for resume training)
        try:
            await self._load_and_broadcast_initial_models()
        except Exception as e:
            log.error(f"Failed to load initial models: {e}")
            log.warning("Continuing with random initialization for all clusters")

        try:
            for round_num in range(1, num_rounds + 1):
                self.current_round = round_num
                log.info(f"=" * 60)
                log.info(f"Round {round_num}/{num_rounds} - Starting")
                log.info(f"=" * 60)

                # Execute one complete training round
                success = await self._execute_round(round_num)
                
                if not success:
                    log.error(f"Round {round_num} failed or timed out")
                    # Log failure as metrics
                    if self.current_run_id:
                        await self.storage_tracker.log_metrics(
                            run_id=self.current_run_id,
                            round_num=round_num,
                            entity_type=EntityType.SERVER,
                            entity_id="orchestrator",
                            metrics={'status': 'failed', 'reason': 'execution_failed'}
                        )
                    break  # Stop training on failure
                
                # Log round completion as metrics
                if self.current_run_id:
                    await self.storage_tracker.log_metrics(
                        run_id=self.current_run_id,
                        round_num=round_num,
                        entity_type=EntityType.SERVER,
                        entity_id="orchestrator",
                        metrics={'status': 'complete'}
                    )
                
                log.info(f"Round {round_num} completed successfully")
                
                # Brief pause between rounds (check for shutdown during pause)
                if round_num < num_rounds:
                    log.info("Pausing briefly before next round...")
                    try:
                        await asyncio.sleep(5.0)
                    except asyncio.CancelledError:
                        log.info("Pause interrupted by shutdown signal")
                        raise
                
            log.info("=" * 60)
            log.info(f"TRAINING COMPLETE: {num_rounds} rounds finished")
            log.info("=" * 60)
            
            # End experiment tracking
            if self.current_run_id:
                await self.storage_tracker.end_run(
                    run_id=self.current_run_id,
                    final_results={'total_rounds': num_rounds, 'status': 'completed'}
                )
                log.info(f"Experiment {self.current_run_id} completed successfully")
        
        except asyncio.CancelledError:
            log.warning("Training cancelled by user")
            # Log cancellation
            if self.current_run_id:
                await self.storage_tracker.end_run(
                    run_id=self.current_run_id,
                    final_results={'status': 'cancelled', 'rounds_completed': self.current_round}
                )
            raise
        except Exception as e:
            log.exception(f"Training failed: {e}")
            # Log failure
            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=self.current_round,
                    entity_type=EntityType.SERVER,
                    entity_id="orchestrator",
                    metrics={'status': 'failed', 'error': str(e)}
                )
                await self.storage_tracker.end_run(
                    run_id=self.current_run_id,
                    final_results={'status': 'failed', 'error': str(e)}
                )
            raise
        finally:
            self.is_running = False
            log.info("Orchestrator stopped")

    async def _execute_round(self, round_num: int) -> bool:
        """
        Execute a single training round.

        Args:
            round_num: Current round number

        Returns:
            bool: True if round completed successfully
        """
        log = logger.bind(context=f"TrainingOrchestrator.round_{round_num}")
        log.info(f"Executing training round {round_num}")

        round_start_time = time.time()
        updates = None
        cluster_models = None
        final_models = None

        try:
            # Step 1: Get ready clusters
            log.info("Step 1: Checking cluster readiness...")
            ready_clusters = self.server.cluster_manager.get_ready_clusters(
                threshold=self.round_config.aggregation_threshold
            )
            if not ready_clusters:
                log.error("No clusters are ready. Aborting round.")
                return False

            log.info(f"Ready clusters: {ready_clusters}")

            # Step 2: Broadcast START_TRAINING
            log.info("Step 2: Broadcasting START_TRAINING to all nodes...")
            await self._broadcast_start_training(round_num)

            # Step 3: Wait for MODEL_UPDATE messages
            log.info("Step 3: Waiting for MODEL_UPDATE messages from nodes...")
            updates = await self._collect_model_updates(round_num)
            if not updates:
                log.error("No model updates received. Aborting round.")
                return False
            log.info(f"Received {len(updates)} model updates")

            # Step 4: Intra-cluster aggregation
            log.info("Step 4: Performing intra-cluster aggregation...")
            cluster_models = await self._aggregate_intra_cluster(updates, round_num)
            if not cluster_models:
                log.error("Intra-cluster aggregation failed. Aborting round.")
                return False
            log.info(f"Intra-cluster aggregation produced {len(cluster_models)} cluster models")

            # Step 5: Inter-cluster selective aggregation
            log.info("Step 5: Performing inter-cluster selective aggregation...")
            final_models = await self._aggregate_inter_cluster(cluster_models, round_num)
            if not final_models:
                log.error("Inter-cluster aggregation failed. Aborting round.")
                return False
            log.info(f"Inter-cluster aggregation produced {len(final_models)} final models")

            # Step 6: Save checkpoints (if interval reached)
            if round_num % self.round_config.checkpoint_interval == 0:
                log.info("Step 6: Saving model checkpoints...")
                await self._save_checkpoints(final_models, round_num)

            # Step 6.5: Run playstyle evaluation (if interval reached)
            if (self.evaluation_config.enabled and
                round_num % self.evaluation_config.interval_rounds == 0):
                log.info("Step 6.5: Running playstyle evaluation...")
                await self._run_playstyle_evaluation(round_num, final_models)

            # Step 7: Broadcast CLUSTER_MODEL to nodes
            log.info("Step 7: Broadcasting CLUSTER_MODEL to all nodes...")
            await self._broadcast_cluster_models(final_models, round_num)

            # Log round metrics (before cleanup)
            round_duration = time.time() - round_start_time
            num_updates = len(updates) if updates else 0
            num_clusters = len(cluster_models) if cluster_models else 0

            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.SERVER,
                    entity_id="orchestrator",
                    metrics={
                        'round_duration': round_duration,
                        'updates_received': num_updates,
                        'clusters_aggregated': num_clusters
                    }
                )

            # Clean up after distributing models and logging metrics
            # Keep models in self.cluster_models for reference, but clear local variables
            if updates is not None:
                del updates
            if cluster_models is not None:
                del cluster_models
            if final_models is not None:
                del final_models
            gc.collect()
            log.debug("Cleaned up round models after distribution")

            log.info(f"Round {round_num} execution complete ({round_duration:.2f}s)")
            return True

        except Exception as e:
            log.error(f"Round {round_num} execution failed: {e}")
            return False
        
    async def _broadcast_start_training(self, round_num: int):
        """
        Broadcast START_TRAINING message to all connected nodes.
        
        Args:
            round_num: Current training round number
        """
        log = logger.bind(context="TrainingOrchestrator._broadcast_start_training")

        # Reset pending updates
        self.pending_updates.clear()
        
        # Get all active clusters
        clusters = self.server.cluster_manager.get_all_clusters()
        
        for cluster in clusters:
            cluster_id = cluster.cluster_id
            nodes = self.server.get_cluster_nodes(cluster_id, active_only=True)
            
            if not nodes:
                log.warning(f"No active nodes in cluster {cluster_id}. Skipping.")
                continue
            
            # Track pending updates for this cluster
            self.pending_updates[cluster_id] = set(nodes)

            # Calculate round offset for resume training
            round_offset = 0
            if self.server.cluster_manager.resume_training_enabled:
                round_offset = self.server.cluster_manager.starting_round
                log.debug(f"Resume training enabled: using round offset {round_offset}")

            # Use cluster-specific games_per_round (required)
            if not cluster.games_per_round or cluster.games_per_round <= 0:
                log.error(f"Cluster {cluster_id} missing valid games_per_round configuration")
                raise ValueError(f"Cluster {cluster_id} must have games_per_round > 0 in cluster_topology.yaml")
            
            games_per_round = cluster.games_per_round
            log.debug(f"Cluster {cluster_id} will train with {games_per_round} games per round")

            # Create START_TRAINING message
            message = MessageFactory.create_start_training(
                node_id="server",
                cluster_id=cluster_id,
                games_per_round=games_per_round,
                round_num=round_num,
                round_offset=round_offset
            )
            
            # Send to all nodes in the cluster
            await self.server.broadcast_to_cluster(cluster_id, message)
            log.info(f"Sent START_TRAINING to cluster {cluster_id} with {len(nodes)} nodes ({games_per_round} games/round)")
            
    async def _collect_model_updates(self, round_num: int) -> Dict[str, Any]:
        """
        Collect model updates from nodes with timeout.
        
        Args:
            round_num: Current training round
        
        Returns:
            Dict mapping node_id to model update data
        """
        log = logger.bind(context="TrainingOrchestrator._collect_model_updates")
        
        updates: Dict[str, Any] = {}
        update_event = asyncio.Event()
        
        # Message handler for MODEL_UPDATE
        async def handle_model_update(node_id: str, message: Message):
            if message.round_num != round_num:
                log.warning(f"Ignoring out-of-sync MODEL_UPDATE from {node_id} "
                           f"(expected round {round_num}, got {message.round_num})")
                return
            
            log.debug(f"Received MODEL_UPDATE from {node_id}")
            updates[node_id] = {
                "model_state": message.payload.get("model_state"),
                "samples": message.payload.get("samples", 0),
                "loss": message.payload.get("loss", None),
                "cluster_id": message.cluster_id
            }
            
            # Mark as received
            if message.cluster_id in self.pending_updates:
                self.pending_updates[message.cluster_id].discard(node_id)
                
            # Check if we reached the aggregation threshold
            if self._check_threshold_met():
                update_event.set()

        self.server.set_message_handler(
            MessageType.MODEL_UPDATE, handle_model_update
        )
        
        # Wait for threshold or timeout
        try:
            await asyncio.wait_for(
                update_event.wait(),
                timeout=self.round_config.timeout_seconds
            )
            log.info(f"Threshold met: {len(updates)} updates received")
        except asyncio.TimeoutError:
            log.warning(f"Timeout reached: only {len(updates)} updates received")
        except asyncio.CancelledError:
            log.warning(f"Update collection cancelled: {len(updates)} updates received before cancellation")
            raise
            
            # Log timeout as metrics (not event)
            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.SERVER,
                    entity_id="orchestrator",
                    metrics={
                        'collection_timeout': True,
                        'updates_received': len(updates),
                        'timeout_seconds': self.round_config.timeout_seconds
                    }
                )
        
        return updates
    
    def _check_threshold_met(self) -> bool:
        """
        Check if aggregation threshold is met for all clusters.
        
        Returns:
            bool: True if threshold met
        """
        for cluster_id, pending_nodes in self.pending_updates.items():
            cluster = self.server.cluster_manager.get_cluster(cluster_id)
            if not cluster:
                continue
            
            expected_count = cluster.get_active_node_count()
            received_count = expected_count - len(pending_nodes)
            
            if received_count < expected_count * self.round_config.aggregation_threshold:
                return False
        
        return True
    
    async def _aggregate_intra_cluster(
        self, 
        updates: Dict[str, Dict], 
        round_num: int
    ) -> Dict[str, Any]:
        """
        Perform intra-cluster aggregation.
        
        Args:
            updates: Model updates from nodes
            round_num: Current round number
        
        Returns:
            Dict mapping cluster_id to aggregated model
        """
        log = logger.bind(context="TrainingOrchestrator._aggregate_intra_cluster")

        # Group updates by cluster and deserialize model states
        cluster_updates: Dict[str, Dict[str, Any]] = {}
        cluster_metrics: Dict[str, Dict[str, Any]] = {}

        # Import serializer for deserialization
        # Use base64 encoding for JSON compatibility over WebSocket
        from common.model_serialization import PyTorchSerializer
        serializer = PyTorchSerializer(compression=True, encoding='base64')

        for node_id, update in updates.items():
            cluster_id = update["cluster_id"]
            
            if cluster_id not in cluster_updates:
                cluster_updates[cluster_id] = {}
                cluster_metrics[cluster_id] = {}
            
            # Deserialize model state if it's packaged
            model_state = update["model_state"]
            log.info(f"🔍 RAW model from {node_id}: type={type(model_state)}")
            if isinstance(model_state, dict):
                log.info(f"   Keys in model_state: {list(model_state.keys())}")
                log.info(f"   Has 'serialized_data' key: {'serialized_data' in model_state}")
            log.debug(f"Processing model from {node_id}: type={type(model_state)}, has_serialized_data={'serialized_data' in model_state if isinstance(model_state, dict) else False}")
            
            if isinstance(model_state, dict) and "serialized_data" in model_state:
                try:
                    serialized_data = model_state["serialized_data"]
                    log.debug(f"Serialized data from {node_id}: type={type(serialized_data)}, length={len(serialized_data) if isinstance(serialized_data, (str, bytes)) else 'N/A'}")
                    
                    # Deserialize the model back to state_dict for aggregation
                    deserialized = serializer.deserialize(serialized_data)
                    log.info(f"✓ Successfully deserialized model from {node_id}: {type(deserialized)}, {len(deserialized)} layers")
                    log.debug(f"  Layer names: {list(deserialized.keys())[:5]}")
                    model_state = deserialized
                except Exception as e:
                    log.error(f"✗ Failed to deserialize model from {node_id}: {e}")
                    log.exception("Full traceback:")
                    raise
            else:
                log.debug(f"Model from {node_id} is already deserialized or has unexpected structure: {type(model_state)}")
            
            cluster_updates[cluster_id][node_id] = model_state
            cluster_metrics[cluster_id][node_id] = {
                "samples": update["samples"],
                "loss": update.get("loss", 0.0)
            }
            
        # Aggregate per cluster
        cluster_models: Dict[str, Any] = {}
        
        for cluster_id, models in cluster_updates.items():
            log.info(f"Aggregating {len(models)} models for cluster {cluster_id}")
            
            # Get aggregation weights
            weights = self.intra_aggregator.get_aggregation_weights(
                participant_metrics=cluster_metrics[cluster_id]
            )
            
            # Perform aggregation
            aggregated_model, metrics = await self.intra_aggregator.aggregate(
                models=models,
                weights=weights,
                round_num=round_num
            )
            
            cluster_models[cluster_id] = aggregated_model
            
            # Log aggregation metrics using correct API
            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.CLUSTER,
                    entity_id=cluster_id,
                    metrics={
                        'aggregation_type': 'intra_cluster',
                        'aggregation_time': metrics.aggregation_time,
                        'participant_count': metrics.participant_count,
                        'total_samples': metrics.total_samples,
                        'average_loss': metrics.average_loss
                    }
                )
            
            log.info(f"{cluster_id} aggregation: {metrics.participant_count} nodes, "
                    f"{metrics.total_samples} samples, {metrics.aggregation_time:.2f}s")

        # Clean up individual model updates to free memory after aggregation
        del cluster_updates
        del cluster_metrics
        gc.collect()
        log.debug("Cleaned up individual model updates after intra-cluster aggregation")

        return cluster_models

    async def _aggregate_inter_cluster(
        self,
        cluster_models: Dict[str, Any],
        round_num: int
    ) -> Dict[str, Any]:
        """
        Perform inter-cluster selective aggregation.
        
        Args:
            cluster_models: Aggregated models per cluster
            round_num: Current round number
        
        Returns:
            Dict mapping cluster_id to final model (with shared layers synchronized)
        """
        log = logger.bind(context="TrainingOrchestrator._aggregate_inter_cluster")
        
        if len(cluster_models) < 2:
            log.info("Only one cluster, skipping inter-cluster aggregation")
            return cluster_models
        
        # Create metrics for inter-cluster aggregation
        cluster_metrics = {}
        for cluster_id in cluster_models.keys():
            cluster = self.server.cluster_manager.get_cluster(cluster_id)
            if cluster:
                # Estimate samples: nodes × games_per_round × avg_positions_per_game
                # Note: This is an approximation (50 positions/game is average)
                cluster_metrics[cluster_id] = {
                    "samples": cluster.get_active_node_count() * cluster.games_per_round * 50,
                    "loss": 0.3  # Placeholder
                }
        
        # Get weights
        weights = self.inter_aggregator.get_aggregation_weights(cluster_metrics)
        
        # Aggregate
        final_models, metrics = await self.inter_aggregator.aggregate(
            models=cluster_models,
            weights=weights,
            round_num=round_num
        )
        
        # Log aggregation metrics using correct API
        if self.current_run_id:
            await self.storage_tracker.log_metrics(
                run_id=self.current_run_id,
                round_num=round_num,
                entity_type=EntityType.SERVER,
                entity_id="inter_cluster_aggregator",
                metrics={
                    'aggregation_type': 'inter_cluster',
                    'aggregation_time': metrics.aggregation_time,
                    'cluster_count': metrics.participant_count,
                    'shared_layer_count': metrics.additional_metrics.get('shared_layer_count', 0),
                    'cluster_specific_count': metrics.additional_metrics.get('cluster_specific_count', 0)
                }
            )
        
        log.info(f"Inter-cluster aggregation: {metrics.participant_count} clusters, "
                f"{metrics.additional_metrics.get('shared_layer_count', 0)} shared layers, "
                f"{metrics.aggregation_time:.2f}s")

        # Clean up intermediate cluster models after inter-cluster aggregation
        # (we keep final_models as they are needed for distribution)
        del cluster_metrics
        gc.collect()
        log.debug("Cleaned up intermediate data after inter-cluster aggregation")

        # Store for next round
        self.cluster_models = final_models

        return final_models
    
    async def _save_checkpoints(
        self,
        models: Dict[str, Any],
        round_num: int
    ):
        """
        Save model checkpoints to storage.
        
        Args:
            models: Cluster models to save
            round_num: Current round number (relative to training loop)
        """
        log = logger.bind(context="TrainingOrchestrator._save_checkpoints")
        
        if not self.current_run_id:
            log.warning("No run_id available, skipping checkpoint save")
            return
        
        # Calculate absolute round number (includes resume offset)
        absolute_round = self.starting_round + round_num
        
        for cluster_id, model_state in models.items():
            try:
                await self.storage_tracker.save_checkpoint(
                    run_id=self.current_run_id,
                    cluster_id=cluster_id,
                    round_num=absolute_round,  # Use absolute round number for file naming
                    model_state=model_state,
                    metrics={'round': absolute_round},  # Required metrics parameter
                    metadata={
                        'timestamp': time.time(),
                        'cluster_id': cluster_id,
                        'relative_round': round_num,  # Track both for reference
                        'starting_round': self.starting_round
                    }
                )
                log.info(f"Saved checkpoint for {cluster_id} round {absolute_round} (relative: {round_num})")
            except Exception as e:
                log.error(f"Failed to save checkpoint for {cluster_id}: {e}")
    
    async def _broadcast_cluster_models(
        self,
        models: Dict[str, Any],
        round_num: int
    ):
        """
        Broadcast updated cluster models to nodes.
        
        Args:
            models: Final cluster models
            round_num: Current round number
        """
        log = logger.bind(context="TrainingOrchestrator._broadcast_cluster_models")
        
        # Import serializer for serialization
        # Use base64 encoding for JSON compatibility over WebSocket
        from common.model_serialization import PyTorchSerializer
        serializer = PyTorchSerializer(compression=True, encoding='base64')
        
        for cluster_id, model_state in models.items():
            # Serialize model state for network transmission
            serialized_data = serializer.serialize(model_state)
            packaged_model_state = {
                "serialized_data": serialized_data,
                "framework": "pytorch",
                "compression": True,
                "encoding": "base64"
            }
            
            # Create CLUSTER_MODEL message
            message = MessageFactory.create_cluster_model(
                node_id="server",
                cluster_id=cluster_id,
                model_state=packaged_model_state,
                round_num=round_num
            )
            
            # Broadcast to cluster
            await self.server.broadcast_to_cluster(cluster_id, message)
            
            nodes = self.server.get_cluster_nodes(cluster_id, active_only=True)
            log.info(f"Broadcasted model to {len(nodes)} nodes in {cluster_id}")

    async def _load_and_broadcast_initial_models(self):
        """
        Load initial models from checkpoints (for resume training) and broadcast to nodes.

        This method checks each cluster for an initial_model path in the config.
        If found, it loads the model checkpoint and broadcasts it to all nodes in that cluster.
        """
        log = logger.bind(context="TrainingOrchestrator._load_and_broadcast_initial_models")
        log.info("Checking for initial models to load...")

        import torch
        from common.model_serialization import PyTorchSerializer
        serializer = PyTorchSerializer(compression=True, encoding='base64')

        clusters = self.server.cluster_manager.get_all_clusters()
        loaded_count = 0

        for cluster in clusters:
            cluster_id = cluster.cluster_id
            initial_model_path = self.server.cluster_manager.get_initial_model(cluster_id)

            if not initial_model_path:
                log.debug(f"No initial model configured for cluster {cluster_id}")
                continue

            try:
                # Load the checkpoint
                log.info(f"Loading initial model for cluster {cluster_id} from {initial_model_path}")
                checkpoint = torch.load(initial_model_path, map_location='cpu')

                # Extract model state (checkpoints may contain additional info)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                else:
                    model_state = checkpoint

                # Serialize model state for network transmission
                serialized_data = serializer.serialize(model_state)
                packaged_model_state = {
                    "serialized_data": serialized_data,
                    "framework": "pytorch",
                    "compression": True,
                    "encoding": "base64"
                }

                # Create CLUSTER_MODEL message for round 0
                message = MessageFactory.create_cluster_model(
                    node_id="server",
                    cluster_id=cluster_id,
                    model_state=packaged_model_state,
                    round_num=0
                )

                # Broadcast to cluster
                await self.server.broadcast_to_cluster(cluster_id, message)

                nodes = self.server.get_cluster_nodes(cluster_id, active_only=True)
                log.info(f"Broadcasted initial model to {len(nodes)} nodes in {cluster_id}")
                loaded_count += 1

            except FileNotFoundError:
                log.error(f"Initial model file not found: {initial_model_path}")
                log.warning(f"Cluster {cluster_id} will start with random initialization")
            except Exception as e:
                log.error(f"Failed to load initial model for cluster {cluster_id}: {e}")
                log.warning(f"Cluster {cluster_id} will start with random initialization")

        if loaded_count > 0:
            log.info(f"Successfully loaded and broadcasted {loaded_count} initial model(s)")
        else:
            log.info("No initial models loaded - all clusters will start with random initialization")

    async def _run_playstyle_evaluation(self, round_num: int, cluster_models: Dict[str, Any]):
        """
        Run playstyle evaluation for cluster models.

        Args:
            round_num: Current round number
            cluster_models: Dict of cluster_id -> model_state_dict
        """
        log = logger.bind(context="TrainingOrchestrator._run_playstyle_evaluation")
        log.info(f"=" * 60)
        log.info(f"Running playstyle evaluation at round {round_num}")
        log.info(f"=" * 60)

        if not self.evaluation_config.enabled or self.model_evaluator is None:
            log.warning("Playstyle evaluation is disabled, skipping")
            return

        eval_start_time = time.time()

        try:
            # Run evaluation
            evaluation_results = await self.model_evaluator.evaluate_models(
                cluster_models=cluster_models,
                num_games=self.evaluation_config.games_per_elo_level,
                stockfish_elo_levels=self.evaluation_config.stockfish_elo_levels,
                time_per_move=self.evaluation_config.time_per_move
            )

            eval_duration = time.time() - eval_start_time

            # Log summary to console
            summary = evaluation_results.get("summary", {})
            log.info(f"\n{'=' * 60}")
            log.info(f"Evaluation Summary (Round {round_num}):")
            log.info(f"{'=' * 60}")

            # ELO Rankings
            log.info("\nELO Rankings:")
            for rank in summary.get("elo_rankings", []):
                log.info(f"  {rank['cluster_id']}: "
                        f"ELO {rank['estimated_elo']} ± {rank['elo_confidence']}, "
                        f"Win Rate: {rank['overall_winrate'] * 100:.1f}%")

            # Tactical Rankings
            log.info("\nTactical Score Rankings:")
            for rank in summary.get("tactical_rankings", []):
                log.info(f"  {rank['cluster_id']}: "
                        f"{rank['tactical_score']:.3f} ({rank['classification']})")

            # Opening Preferences (Top 5 per cluster)
            log.info("\nOpening Preferences:")
            for cluster_id, cluster_metrics in evaluation_results.get("cluster_metrics", {}).items():
                log.info(f"  {cluster_id}:")
                top_openings = cluster_metrics.get("top_openings", [])[:5]
                for opening in top_openings:
                    log.info(f"    {opening['eco']}: {opening['name']} ({opening['count']} games)")

            log.info(f"\nPlaystyle Divergence: {summary.get('playstyle_divergence', 0):.3f}")
            log.info(f"ELO Spread: {summary.get('elo_spread', 0)} points")
            log.info(f"Evaluation Duration: {eval_duration:.1f}s")
            log.info(f"{'=' * 60}\n")

            # Store evaluation results in metrics
            if self.current_run_id:
                # Store per-game raw metrics
                for i, game_result in enumerate(evaluation_results.get("game_results", [])):
                    await self.storage_tracker.log_metrics(
                        run_id=self.current_run_id,
                        round_num=round_num,
                        entity_type=EntityType.CUSTOM,
                        entity_id=f"playstyle_eval_game_{i}",
                        metrics=game_result,
                        metadata={
                            "evaluation_type": "playstyle",
                            "game_number": i,
                            "evaluation_round": round_num
                        }
                    )

                # Store per-game computed metrics
                for i, computed_metrics in enumerate(evaluation_results.get("computed_metrics", [])):
                    await self.storage_tracker.log_metrics(
                        run_id=self.current_run_id,
                        round_num=round_num,
                        entity_type=EntityType.CUSTOM,
                        entity_id=f"playstyle_eval_game_{i}_computed",
                        metrics=computed_metrics,
                        metadata={
                            "evaluation_type": "playstyle_computed",
                            "game_number": i,
                            "evaluation_round": round_num
                        }
                    )

                # Store per-cluster aggregated metrics
                for cluster_id, cluster_metrics in evaluation_results.get("cluster_metrics", {}).items():
                    await self.storage_tracker.log_metrics(
                        run_id=self.current_run_id,
                        round_num=round_num,
                        entity_type=EntityType.CLUSTER,
                        entity_id=cluster_id,
                        metrics={"playstyle_evaluation": cluster_metrics},
                        metadata={
                            "evaluation_type": "playstyle_cluster_summary",
                            "evaluation_round": round_num
                        }
                    )

                # Store global evaluation summary
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.SERVER,
                    entity_id="playstyle_evaluator",
                    metrics={
                        "evaluation_summary": summary,
                        "evaluation_duration_seconds": eval_duration
                    },
                    metadata={
                        "evaluation_type": "playstyle_global_summary",
                        "evaluation_round": round_num
                    }
                )

                log.info(f"Playstyle evaluation metrics stored for round {round_num}")

        except Exception as e:
            log.error(f"Playstyle evaluation failed: {e}")
            log.exception("Full traceback:")

            # Log failure
            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.SERVER,
                    entity_id="playstyle_evaluator",
                    metrics={"evaluation_error": str(e), "evaluation_failed": True},
                    metadata={"evaluation_round": round_num}
                )


class ServerMenu:
    """Interactive menu for server control."""
    
    def __init__(self, server: FederatedLearningServer, orchestrator: TrainingOrchestrator):
        self.server = server
        self.orchestrator = orchestrator
        self.is_running = True

    async def _menu_loop(self):
        await asyncio.sleep(2)  # Give server time to start

        while self.is_running:
            print("\n" + "="*60)
            print("🎛️  FEDERATED LEARNING SERVER MENU")
            print("="*60)
            print("1. 📋 Show connected nodes")
            print("2. 🏃 Start training")
            print("3. 📊 Show server stats")
            print("4. 🛑 Stop server")
            print("="*60)

            try:
                choice = await self._get_input("Enter your choice (1-4): ")

                if choice == "1":
                    await self._show_connected_nodes()
                elif choice == "2":
                    await self._start_training()
                elif choice == "3":
                    await self._show_server_stats()
                elif choice == "4":
                    print("🛑 Stopping server...")
                    self.is_running = False
                    self.orchestrator.is_running = False
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                self.is_running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(2)

    async def _get_input(self, prompt: str) -> str:
        """Non-blocking input for async context"""
        print(prompt, end='', flush=True)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sys.stdin.readline)
        return result.strip()

    async def _show_connected_nodes(self):
        nodes = self.server.get_connected_nodes()

        if not nodes:
            print("📭 No nodes currently connected.")
            return

        print(f"\n📋 Connected Nodes ({len(nodes)}):")
        print("-" * 60)
        for node_id, node in nodes.items():
            uptime = time.time() - node.registration_time
            print(f"🔹 {node_id}")
            print(f"   Cluster: {node.cluster_id}")
            print(f"   State: {node.state.value}")
            print(f"   Uptime: {uptime:.1f}s")
            print(f"   Round: {node.current_round or 'N/A'}")
            print()

    async def _start_training(self):
        """Start training with user input."""
        num_rounds = await self._get_input("Enter number of rounds (default 10): ")
        num_rounds = int(num_rounds) if num_rounds else 10
        
        experiment_name = await self._get_input("Enter experiment name (default 'federated_chess_training'): ")
        experiment_name = experiment_name if experiment_name else "federated_chess_training"
        
        print(f"🏃 Starting training: {num_rounds} rounds")
        print(f"📝 Experiment name: {experiment_name}")
        
        # Run training (this will block until complete)
        await self.orchestrator.run_training(
            num_rounds=num_rounds,
            experiment_name=experiment_name
        )

    async def _show_server_stats(self):
        stats = self.server.get_server_statistics()

        print(f"\n📊 Server Statistics:")
        print(f"Connected nodes: {stats['connections']['total_connected_nodes']}")
        print(f"Active clusters: {stats['connections']['total_active_clusters']}")
        print(f"Messages handled: {stats['server']['total_messages_handled']}")
        print(f"Current round: {stats['server']['current_round']}")
        print(f"Server uptime: {stats['server']['uptime_seconds']:.1f}s")

        print(f"\n🏗️  Cluster Manager Statistics:")
        cm_stats = stats['cluster_manager']
        print(f"Total clusters: {cm_stats['cluster_count']}")
        print(f"Total expected nodes: {cm_stats['total_expected_nodes']}")
        print(f"Total registered nodes: {cm_stats['total_registered_nodes']}")
        print(f"Active nodes: {cm_stats['total_active_nodes']}")
        print(f"Ready clusters: {cm_stats['ready_clusters']}")
        print(f"Uptime: {cm_stats['uptime_seconds']:.1f}s")

        print(f"\n🏷️  Cluster Readiness:")
        cluster_readiness = stats['cluster_readiness']
        for cluster_id, readiness in cluster_readiness.items():
            status = "✅" if readiness['is_ready'] else "❌"
            ratio = readiness.get('readiness_ratio', 0.0)
            active = readiness.get('active_nodes', 0)
            expected = readiness.get('expected_nodes', 0)
            playstyle = readiness.get('playstyle', 'unknown')
            print(f"  {status} {cluster_id} ({playstyle}): {active}/{expected} nodes ({ratio:.1%})")

        print(f"\n🔗 Connected Clusters:")
        for cluster_id in self.server.connections_by_cluster:
            node_count = len(self.server.connections_by_cluster[cluster_id])
            print(f"  {cluster_id}: {node_count} connected nodes")


async def main():
    """
    Main entry point to start the federated learning server.
    """
    log = logger.bind(context="server.main")
    log.info("Starting Federated Learning Server...")
    
    # Configuration paths
    cluster_config_path = "chess-federated-learning/config/cluster_topology.yaml"
    server_config_path = "chess-federated-learning/config/server_config.yaml"
    
    # Load server config
    import yaml
    try:
        with open(server_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            server_config = config_data.get("server_config", {})
            orchestrator_config = config_data.get("orchestrator_config", {})
            evaluation_config_data = config_data.get("evaluation_config", {})
        log.info("Loaded configuration from YAML")
    except FileNotFoundError:
        log.warning(f"Config file {server_config_path} not found, using defaults")
        server_config = {}
        orchestrator_config = {}
        evaluation_config_data = {}
    
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
    await asyncio.sleep(2.0)
    
    if not server.is_running:
        log.error("Server failed to start. Exiting.")
        return
    
    log.info("Server started successfully")
    
    # Create round configuration
    # Note: games_per_round is now configured per-cluster in cluster_topology.yaml
    round_config = RoundConfig(
        aggregation_threshold=orchestrator_config.get("aggregation_threshold", 0.8),
        timeout_seconds=orchestrator_config.get("timeout_seconds", 300),
        shared_layer_patterns=orchestrator_config.get("shared_layer_patterns", ["input_conv.*"]),
        cluster_specific_patterns=orchestrator_config.get("cluster_specific_patterns", ["policy_head.*", "value_head.*"]),
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
    
    # Create storage backend using factory
    log.info("Initializing storage backend...")
    tracker = create_experiment_tracker(
        base_path="./storage",
        compression=True,
        keep_last_n=None,  # Keep all checkpoints
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
    
    # Create menu
    menu = ServerMenu(server=server, orchestrator=orchestrator)
    menu_task = asyncio.create_task(menu._menu_loop())

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        log.info("Shutdown signal received")
        orchestrator.is_running = False
        menu.is_running = False
        # Cancel tasks to break out of gather
        if not server_task.done():
            server_task.cancel()
        if not menu_task.done():
            menu_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Run server and menu concurrently
        await asyncio.gather(server_task, menu_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Interrupt received")
    finally:
        # Cleanup
        log.info("Shutting down server...")

        # Stop orchestrator if running
        if orchestrator.is_running:
            orchestrator.is_running = False

        # Cancel all tasks first (before stopping server to break out of any waits)
        tasks_to_cancel = []
        if not server_task.done():
            server_task.cancel()
            tasks_to_cancel.append(server_task)
        if not menu_task.done():
            menu_task.cancel()
            tasks_to_cancel.append(menu_task)

        # Wait for task cancellations with timeout
        if tasks_to_cancel:
            log.info(f"Cancelling {len(tasks_to_cancel)} tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                log.warning("Task cancellation timed out after 5 seconds")

        # Stop server (disconnect clients)
        log.info("Stopping server and disconnecting clients...")
        try:
            await asyncio.wait_for(server.stop_server(), timeout=10.0)
        except asyncio.TimeoutError:
            log.error("Server stop timed out after 10 seconds")

        # Cancel any remaining tasks in the event loop (except the current task)
        current_task = asyncio.current_task()
        pending_tasks = [
            task for task in asyncio.all_tasks()
            if not task.done() and task is not current_task
        ]
        if pending_tasks:
            log.warning(f"Cancelling {len(pending_tasks)} remaining background tasks...")
            for task in pending_tasks:
                task.cancel()
            # Wait briefly for cancellations with return_exceptions to avoid propagating errors
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                log.warning("Some tasks did not cancel within 2 seconds")

    log.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())