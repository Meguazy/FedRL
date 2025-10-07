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


@dataclass
class RoundConfig:
    """Configuration for each training round."""
    games_per_round: int = 100
    aggregation_threshold: float = 0.8  # Fraction of nodes required to proceed
    timeout_seconds: int = 300  # Max wait time for nodes
    intra_cluster_weighting: str = "samples"  # "samples" or "uniform"
    inter_cluster_weighting: str = "uniform"  # "samples" or "uniform"
    shared_layer_patterns: list = field(default_factory=lambda: ["input_conv.*"])
    cluster_specific_patterns: list = field(default_factory=lambda: ["policy_head.*", "value_head.*"])
    checkpoint_interval: int = 5  # Save every N rounds
    
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
        storage_tracker: Optional[FileExperimentTracker] = None
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
        experiment_config = {
            'num_rounds': num_rounds,
            'games_per_round': self.round_config.games_per_round,
            'aggregation_threshold': self.round_config.aggregation_threshold,
            'intra_cluster_weighting': self.round_config.intra_cluster_weighting,
            'inter_cluster_weighting': self.round_config.inter_cluster_weighting,
            'shared_layer_patterns': self.round_config.shared_layer_patterns,
            'cluster_specific_patterns': self.round_config.cluster_specific_patterns,
            'checkpoint_interval': self.round_config.checkpoint_interval,
            'server_config': {
                'host': self.server.host,
                'port': self.server.port
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
                
                # Brief pause between rounds
                if round_num < num_rounds:
                    log.info("Pausing briefly before next round...")
                    await asyncio.sleep(5.0)
                
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
            
            # Step 7: Broadcast CLUSTER_MODEL to nodes
            log.info("Step 7: Broadcasting CLUSTER_MODEL to all nodes...")
            await self._broadcast_cluster_models(final_models, round_num)
            
            # Log round metrics
            round_duration = time.time() - round_start_time
            if self.current_run_id:
                await self.storage_tracker.log_metrics(
                    run_id=self.current_run_id,
                    round_num=round_num,
                    entity_type=EntityType.SERVER,
                    entity_id="orchestrator",
                    metrics={
                        'round_duration': round_duration,
                        'updates_received': len(updates),
                        'clusters_aggregated': len(cluster_models)
                    }
                )
            
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
            
            # Create START_TRAINING message
            message = MessageFactory.create_start_training(
                node_id="server",
                cluster_id=cluster_id,
                games_per_round=self.round_config.games_per_round,
                round_num=round_num
            )
            
            # Send to all nodes in the cluster
            await self.server.broadcast_to_cluster(cluster_id, message)
            log.info(f"Sent START_TRAINING to cluster {cluster_id} with {len(nodes)} nodes")
            
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

        # Group updates by cluster
        cluster_updates: Dict[str, Dict[str, Any]] = {}
        cluster_metrics: Dict[str, Dict[str, Any]] = {}

        for node_id, update in updates.items():
            cluster_id = update["cluster_id"]
            
            if cluster_id not in cluster_updates:
                cluster_updates[cluster_id] = {}
                cluster_metrics[cluster_id] = {}
                
            cluster_updates[cluster_id][node_id] = update["model_state"]
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
                cluster_metrics[cluster_id] = {
                    "samples": cluster.get_active_node_count() * self.round_config.games_per_round * 50,
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
            round_num: Current round number
        """
        log = logger.bind(context="TrainingOrchestrator._save_checkpoints")
        
        if not self.current_run_id:
            log.warning("No run_id available, skipping checkpoint save")
            return
        
        for cluster_id, model_state in models.items():
            try:
                await self.storage_tracker.save_checkpoint(
                    run_id=self.current_run_id,
                    cluster_id=cluster_id,
                    round_num=round_num,
                    model_state=model_state,
                    metrics={'round': round_num},  # Required metrics parameter
                    metadata={
                        'timestamp': time.time(),
                        'cluster_id': cluster_id
                    }
                )
                log.info(f"Saved checkpoint for {cluster_id} round {round_num}")
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
        
        for cluster_id, model_state in models.items():
            # Create CLUSTER_MODEL message
            message = MessageFactory.create_cluster_model(
                node_id="server",
                cluster_id=cluster_id,
                model_state=model_state,
                round_num=round_num
            )
            
            # Broadcast to cluster
            await self.server.broadcast_to_cluster(cluster_id, message)
            
            nodes = self.server.get_cluster_nodes(cluster_id, active_only=True)
            log.info(f"Broadcasted model to {len(nodes)} nodes in {cluster_id}")


class ServerMenu:
    """Interactive menu for server control."""
    
    def __init__(self, server: FederatedLearningServer, orchestrator: TrainingOrchestrator):
        self.server = server
        self.orchestrator = orchestrator
        self.is_running = True

    async def _menu_loop(self):
        await asyncio.sleep(2)  # Give server time to start

        while self.is_running:
            print("\n" + "="*50)
            print("🎛️  FEDERATED LEARNING SERVER MENU")
            print("="*50)
            print("1. 📋 Show connected nodes")
            print("2. 🔔 Broadcast start training")
            print("3. 📊 Show server stats")
            print("="*50)

            try:
                choice = await self._get_input("Enter your choice (1-8): ")

                if choice == "1":
                    await self._show_connected_nodes()
                elif choice == "2":
                    await self._broadcast_start_training()
                elif choice == "3":
                    await self._show_server_stats()
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

    async def _broadcast_start_training(self):
        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num) if round_num else 1

        self.orchestrator.run_training(num_rounds=round_num)
        print(f"📡 Broadcasted START_TRAINING: round {round_num}")

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

if __name__ == "__main__":
    asyncio.run(main())