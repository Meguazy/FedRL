"""
Intra-cluster aggregation for federated learning.

This module implements Step 2 of the 3-tier aggregation process: aggregating
models from nodes within the same cluster. It uses Federated Averaging (FedAvg)
to combine models from nodes with the same playstyle.

Key Features:
    - FedAvg algorithm implementation for cluster-level aggregation
    - Sample-based weighting (nodes with more samples contribute more)
    - Preserves playstyle characteristics within each cluster
    - Comprehensive validation and metrics collection
    - Support for both PyTorch and TensorFlow models 

Architecture:
    - Extends BaseAggregator with cluster-specific logic
    - Operates on models from a single cluster at a time
    - Produces one aggregated model per cluster
    - Prepares models for inter-cluster selective aggregation
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger

from .base_aggregator import BaseAggregator, AggregationMetrics, validate_participant_metrics, normalize_weights
from server.storage.base import EntityType, ExperimentTracker
from server.storage.plugins.metric_registry import MetricRegistry


class IntraClusterAggregator(BaseAggregator):
    """
    Aggregator for combining models within a single cluster.
    
    This aggregator implements Federated Averaging (FedAvg) to combine models
    from nodes that share the same playstyle. The aggregation is weighted by
    the number of training samples each node contributed, ensuring that nodes
    with more data have proportionally more influence on the cluster model.
    
    The intra-cluster aggregation is the first level of model combination in
    the 3-tier architecture:
    1. Local training (nodes)
    2. **Intra-cluster aggregation** (this class) <- Step 2
    3. Inter-cluster selective aggregation (next step)
    
    Example:
        >>> aggregator = IntraClusterAggregator(framework='pytorch')
        >>> models = {
        ...     'agg_001': model_state_1,
        ...     'agg_002': model_state_2,
        ...     'agg_003': model_state_3
        ... }
        >>> weights = {'agg_001': 0.3, 'agg_002': 0.5, 'agg_003': 0.2}
        >>> aggregated, metrics = await aggregator.aggregate(models, weights, round_num=1)
    """
    def __init__(self, framework: str = "pytorch", compression: bool = True,
                 weighting_strategy: str = "samples",
                 experiment_tracker: Optional[ExperimentTracker] = None,
                 metric_registry: Optional[MetricRegistry] = None):
        """
        Initialize the intra-cluster aggregator.

        Args:
            framework: ML framework to use ('pytorch' or 'tensorflow')
            compression: Whether to use compression for model serialization
            weighting_strategy: How to weight node contributions ('samples', 'uniform', 'loss')
            experiment_tracker: Optional experiment tracker for logging metrics and checkpoints
            metric_registry: Optional metric registry for custom metric computation

        Raises:
            ValueError: If weighting strategy is not supported
        """
        log = logger.bind(context="IntraClusterAggregator.__init__")
        log.info(f"Initializing IntraClusterAggregator with {weighting_strategy} weighting")

        # Initialize base aggregator
        super().__init__(framework=framework, compression=compression)

        # Validate weighting strategy
        valid_strategies = ["samples", "uniform", "loss"]
        if weighting_strategy not in valid_strategies:
            log.error(f"Invalid weighting strategy: {weighting_strategy}")
            raise ValueError(f"Weighting strategy must be one of {valid_strategies}")
        self.weighting_strategy = weighting_strategy

        # Intra-cluster specific attributes
        self.min_participants = 1  # Minimum nodes required for aggregation
        self.max_participants = 100  # Max nodes to consider for aggregation

        # Storage integration
        self.experiment_tracker = experiment_tracker
        self.metric_registry = metric_registry

        if experiment_tracker:
            log.info("Storage integration enabled with experiment tracker")
        if metric_registry:
            log.info(f"Metric registry enabled with {len(metric_registry.list_computers())} computers")

        log.info("IntraClusterAggregator initialized successfully")
        
    async def aggregate(
        self,
        models: Dict[str, Any],
        weights: Dict[str, float],
        round_num: int = 0,
        cluster_id: str = "default_cluster",
        run_id: Optional[str] = None
    ) -> Tuple[Any, AggregationMetrics]:
        """
        Aggregate models from nodes within a cluster using FedAvg.

        This method combines multiple node models into a single cluster-level
        model by computing a weighted average of all parameters. The weights
        are typically based on the number of training samples each node used.

        Args:
            models: Dictionary mapping node_id -> model_state_dict
            weights: Dictionary mapping node_id -> aggregation_weight
            round_num: Current training round number
            cluster_id: Identifier for the cluster being aggregated
            run_id: Optional run ID for storage (required if experiment_tracker is set)

        Returns:
            Tuple of (aggregated_model_state, metrics)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If aggregation fails
        """
        log = logger.bind(context="IntraClusterAggregator.aggregate")
        log.info(f"Starting intra-cluster aggregation for round {round_num}, cluster {cluster_id}")
        log.info(f"Aggregating {len(models)} models with {self.weighting_strategy} weighting")

        start_time = time.time()

        try:
            # Step 1: Validate inputs
            if self._validate_inputs:
                log.debug("Step 1: Validating inputs...")
                self.validate_inputs(models, weights)
                self.check_model_compatibility(models)

            # Step 2: Normalize weights
            log.debug("Step 2: Normalizing aggregation weights...")
            normalized_weights = normalize_weights(weights)
            log.debug(f"Normalized weights: {normalized_weights}")

            # Step 3: Perform FedAvg aggregation
            log.debug("Step 3: Performing FedAvg aggregation...")
            aggregated_model = self._fedavg_aggregate(models, normalized_weights)

            # Step 4: Collect metrics
            log.debug("Step 4: Collecting aggregation metrics...")
            aggregation_time = time.time() - start_time

            metrics = self._collect_metrics(
                models=models,
                weights=weights,
                aggregation_time=aggregation_time,
                round_num=round_num
            )

            # Step 5: Compute custom metrics via plugin registry
            custom_metrics = {}
            if self.metric_registry:
                log.debug("Step 5a: Computing custom metrics via registry...")
                context = {
                    "models": {node_id: model for node_id, model in models.items()},
                    "aggregated_model": aggregated_model,
                    "weights": normalized_weights,
                    "round": round_num,
                    "cluster_id": cluster_id,
                    "aggregation_start_time": start_time,
                    "aggregation_end_time": time.time(),
                    "node_count": len(models)
                }
                custom_metrics = self.metric_registry.compute_all(context, skip_on_error=True)
                log.debug(f"Computed {len(custom_metrics)} custom metrics")

            # Step 6: Store metrics and checkpoint if tracker is available
            if self.experiment_tracker and run_id:
                log.debug("Step 6a: Logging metrics to storage...")

                # Combine standard and custom metrics
                all_metrics = {
                    "aggregation_time": aggregation_time,
                    "participant_count": len(models),
                    "total_samples": int(sum(weights.values())),
                    "weighting_strategy": self.weighting_strategy,
                    **custom_metrics
                }

                # Log cluster-level metrics
                self.experiment_tracker.log_metrics(
                    run_id=run_id,
                    entity_type=EntityType.CLUSTER,
                    entity_id=cluster_id,
                    round_num=round_num,
                    metrics=all_metrics
                )

                log.debug("Step 6b: Saving cluster model checkpoint...")

                # Save cluster model checkpoint
                checkpoint_metrics = {"loss": metrics.average_loss} if metrics.average_loss else {}
                self.experiment_tracker.save_checkpoint(
                    run_id=run_id,
                    cluster_id=cluster_id,
                    round_num=round_num,
                    model_state=aggregated_model,
                    metrics=checkpoint_metrics
                )

                log.info(f"Stored metrics and checkpoint for cluster {cluster_id}, round {round_num}")

            # Step 7: Update internal statistics
            self._update_statistics(len(models), aggregation_time)

            log.info(f"Intra-cluster aggregation successful: {len(models)} models -> 1 cluster model")
            return aggregated_model, metrics

        except Exception as e:
            log.error(f"Aggregation failed: {e}")
            raise RuntimeError(f"Aggregation failed: {e}")
        
    def _fedavg_aggregate(self, models: Dict[str, Any], normalized_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform Federated Averaging aggregation.
        
        Computes the weighted average of all model parameters:
        aggregated[key] = Σ(weight_i * model_i[key]) for all nodes i
        
        Args:
            models: Dictionary of node models
            normalized_weights: Normalized aggregation weights (sum to 1.0)
        
        Returns:
            Aggregated model state dictionary
        """
        log = logger.bind(context="IntraClusterAggregator._fedavg_aggregate")
        log.debug("Computing weighted average of models...")
        
        # Initialize aggregated model with zeros
        aggregated_model = {}
        
        # Get the first model to determine structure
        first_node = next(iter(models))
        reference_model = models[first_node]
        
        # For each parameter in the model
        for key in reference_model.keys():
            log.trace(f"Aggregating parameter: {key}")
            
            # Initialize this parameter in the aggregated model
            aggregated_model[key] = self._aggregate_parameter(
                key, models, normalized_weights
            )
            
        log.debug(f"Aggregation complete. Aggregated model has {len(aggregated_model)} parameters.")
        return aggregated_model
        
    def _aggregate_parameter(self, key: str, models: Dict[str, Any],
                           normalized_weights: Dict[str, float]) -> Any:
        """
        Aggregate a single parameter across all models.
        
        Args:
            key: Parameter name/key
            models: Dictionary of node models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated parameter value
        """
        log = logger.bind(context="IntraClusterAggregator._aggregate_parameter")
        log.trace(f"Aggregating parameter: {key} across {len(models)} models")
        # Get reference parameter to determine type/shape
        first_node = next(iter(models))
        reference_param = models[first_node][key]
        
        # Handle 2D parameters (e.g. weight matrices)
        if isinstance(reference_param, (list)) and isinstance(reference_param[0], (list)):
            log.debug(f"Aggregating 2D parameter: {key}...")
            return self._aggregate_2d_parameter(key, models, normalized_weights)
        # Handle 1D parameters (e.g. bias vectors)
        elif isinstance(reference_param, (list)):
            log.debug(f"Aggregating 1D parameter: {key}...")
            return self._aggregate_1d_parameter(key, models, normalized_weights)
        # Handle scalar parameters (e.g. learning rates)
        else:
            log.debug(f"Aggregating scalar parameter: {key}...")
            return self._aggregate_scalar_parameter(key, models, normalized_weights)

    def _aggregate_2d_parameter(self, key: str, models: Dict[str, Any],
                              normalized_weights: Dict[str, float]) -> List[List[float]]:
        """
        Aggregate a 2D parameter (e.g. weight matrix).
        
        Args:
            key: Parameter name/key
            models: Dictionary of node models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated 2D parameter
        """
        log = logger.bind(context="IntraClusterAggregator._aggregate_2d_parameter")
        log.trace(f"Aggregating 2D parameter: {key}")
        
        # Initialize aggregated parameter with zeros
        first_node = next(iter(models))
        reference_param = models[first_node][key]
        rows = len(reference_param)
        cols = len(reference_param[0])

        # Initialize result matrix with zeros
        aggregated_param = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        # Compute weighted sum for each element
        for node_id, model in models.items():
            weight = normalized_weights[node_id]
            param = model[key]
            for i in range(rows):
                for j in range(cols):
                    aggregated_param[i][j] += weight * param[i][j]

        log.trace(f"2D parameter {key} aggregation complete.")
        return aggregated_param
    
    def _aggregate_1d_parameter(self, key: str, models: Dict[str, Any],
                               normalized_weights: Dict[str, float]) -> List[float]:
        """
        Aggregate a 1D parameter (bias vector).
        
        Args:
            key: Parameter name
            models: Dictionary of node models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated 1D parameter
        """
        # Get length from reference model
        first_node = next(iter(models))
        reference_param = models[first_node][key]
        length = len(reference_param)
        
        # Initialize result vector with zeros
        aggregated = [0.0 for _ in range(length)]
        
        # Compute weighted sum for each element
        for node_id, model in models.items():
            weight = normalized_weights[node_id]
            param = model[key]
            
            for i in range(length):
                aggregated[i] += weight * param[i]
        
        return aggregated
    
    def _aggregate_scalar_parameter(self, key: str, models: Dict[str, Any],
                                   normalized_weights: Dict[str, float]) -> float:
        """
        Aggregate a scalar parameter.
        
        Args:
            key: Parameter name
            models: Dictionary of node models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated scalar value
        """
        aggregated = 0.0
        
        for node_id, model in models.items():
            weight = normalized_weights[node_id]
            param = model[key]
            aggregated += weight * param
        
        return aggregated
    
    def get_aggregation_weights(self, participant_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregation weights based on participant metrics.
        
        Implements different weighting strategies:
        - 'samples': Weight by number of training samples (default)
        - 'uniform': Equal weight for all participants
        - 'loss': Inverse weighting by loss (better models contribute more)
        
        Args:
            participant_metrics: Dictionary mapping node_id -> metrics dict
                               Expected keys: 'samples', 'loss'
        
        Returns:
            Dictionary mapping node_id -> aggregation_weight
        
        Raises:
            ValueError: If metrics are invalid
        """
        log = logger.bind(context="IntraClusterAggregator.get_aggregation_weights")
        log.info(f"Calculating weights using '{self.weighting_strategy}' strategy")
        
        # Validate participant metrics
        validate_participant_metrics(participant_metrics)
        
        weights = {}
        if self.weighting_strategy == "samples":
            log.debug("Weighting by number of samples...")
            # Weight by number of samples
            for node_id, metrics in participant_metrics.items():
                weights[node_id] = float(metrics.get('samples', 0))
        elif self.weighting_strategy == "uniform":
            log.debug("Using uniform weighting...")
            # Equal weight for all participants
            for node_id in participant_metrics.keys():
                weights[node_id] = 1.0
        elif self.weighting_strategy == "loss":
            log.debug("Weighting inversely by loss...")
            # Inverse weighting by loss (lower loss -> higher weight)
            # Weight = 1 / (loss + epsilon) to avoid division by zero
            epsilon = 1e-6
            for node_id, metrics in participant_metrics.items():
                loss = float(metrics.get('loss', 0))
                weights[node_id] = 1.0 / (loss + epsilon)
                
        log.debug(f"Raw weights before normalization: {weights}")
        return weights
    
    def _collect_metrics(self, models: Dict[str, Any], weights: Dict[str, float],
                        aggregation_time: float, round_num: int) -> AggregationMetrics:
        """
        Collect metrics about the aggregation process.
        
        Args:
            models: Dictionary of node models
            weights: Aggregation weights used
            aggregation_time: Time taken for aggregation
            round_num: Current training round
        
        Returns:
            AggregationMetrics object with collected metrics
        """
        log = logger.bind(context="IntraClusterAggregator._collect_metrics")
        log.info("Collecting aggregation metrics...")
        
        # Calculate total samples (unnormalized weights represent sample counts)
        total_samples = int(sum(weights.values()))
        
        # Calculate diversity if enabled
        model_diversity = None
        if self.collect_metrics and len(models) > 1:
            model_diversity = self.calculate_model_diversity(models)
        
        metrics = AggregationMetrics(
            aggregation_time=aggregation_time,
            participant_count=len(models),
            total_samples=total_samples,
            average_loss=0.0,  # Would need loss values from participants
            model_diversity=model_diversity,
            framework=self.framework,
            aggregation_round=round_num,
            additional_metrics={
                'weighting_strategy': self.weighting_strategy,
                'aggregation_type': 'intra_cluster'
            }
        )
        
        log.debug(f"Collected metrics: {metrics.participant_count} participants, "
                 f"{metrics.total_samples} total samples, "
                 f"{metrics.aggregation_time:.3f}s")
        
        return metrics
    
    def set_weighting_strategy(self, strategy: str):
        """
        Change the weighting strategy.
        
        Args:
            strategy: New weighting strategy ('samples', 'uniform', 'loss')
        
        Raises:
            ValueError: If strategy is not supported
        """
        log = logger.bind(context="IntraClusterAggregator.set_weighting_strategy")
        
        valid_strategies = ['samples', 'uniform', 'loss']
        if strategy not in valid_strategies:
            log.error(f"Invalid weighting strategy: {strategy}")
            raise ValueError(f"Weighting strategy must be one of {valid_strategies}")
        
        old_strategy = self.weighting_strategy
        self.weighting_strategy = strategy
        log.info(f"Changed weighting strategy from '{old_strategy}' to '{strategy}'")


def create_intra_cluster_aggregator(config: Dict[str, Any]) -> IntraClusterAggregator:
    """
    Factory function to create an intra-cluster aggregator from configuration.

    Args:
        config: Configuration dictionary with keys:
               - framework: 'pytorch' or 'tensorflow'
               - compression: bool
               - weighting_strategy: 'samples', 'uniform', or 'loss'
               - experiment_tracker: Optional ExperimentTracker instance
               - metric_registry: Optional MetricRegistry instance

    Returns:
        Configured IntraClusterAggregator instance
    """
    log = logger.bind(context="create_intra_cluster_aggregator")
    log.info("Creating intra-cluster aggregator from configuration")

    framework = config.get('framework', 'pytorch')
    compression = config.get('compression', True)
    weighting_strategy = config.get('weighting_strategy', 'samples')
    experiment_tracker = config.get('experiment_tracker', None)
    metric_registry = config.get('metric_registry', None)

    aggregator = IntraClusterAggregator(
        framework=framework,
        compression=compression,
        weighting_strategy=weighting_strategy,
        experiment_tracker=experiment_tracker,
        metric_registry=metric_registry
    )

    log.info("Intra-cluster aggregator created successfully")
    return aggregator
