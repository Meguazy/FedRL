"""
Inter-cluster aggregation for federated learning with diversity preservation.

This module implements Step 3 of the 3-tier aggregation process: selectively sharing
weights across clusters while preserving playstyle diversity. Unlike traditional
federated learning, this does NOT create a single global model. Instead, it maintains
separate models per cluster, sharing only generic feature extraction layers while
keeping strategic decision-making layers cluster-specific.

Key Features:
    - Selective layer sharing across clusters (configurable via layer names)
    - Pattern matching for layer selection (e.g., "policy_head.*")
    - Preservation of cluster-specific layers (policy, value heads)
    - Sample-based weighting across clusters
    - Comprehensive validation and metrics collection
    - Support for both PyTorch and TensorFlow models

Architecture:
    - Extends BaseAggregator with inter-cluster logic
    - Operates on cluster-aggregated models (output from IntraClusterAggregator)
    - Produces updated models for each cluster with shared layers synchronized
    - Core innovation: diversity-preserving selective aggregation
    
Example:
    Shared layers (synchronized across clusters):
        - input_conv: Board representation
        - representation.0-2: Early feature extraction
        
    Cluster-specific layers (preserved per cluster):
        - policy_head.*: Playstyle-specific move selection
        - value_head.*: Playstyle-specific position evaluation
        - representation.5-8: Deep strategic layers
"""

import time
import re
from typing import Dict, List, Any, Set, Tuple
from loguru import logger

from .base_aggregator import (
    BaseAggregator, 
    AggregationMetrics, 
    validate_participant_metrics, 
    normalize_weights
)


class InterClusterAggregator(BaseAggregator):
    """
    Aggregator for selective weight sharing across clusters.
    
    This aggregator implements the diversity-preserving mechanism that makes
    this federated learning framework unique. It aggregates only specified
    "shared" layers across clusters using FedAvg, while leaving "cluster-specific"
    layers untouched to preserve distinct playstyles.
    
    The inter-cluster aggregation is the final step in the 3-tier architecture:
    1. Local training (nodes)
    2. Intra-cluster aggregation (within same playstyle)
    3. **Inter-cluster selective aggregation** (this class) <- Step 3
    
    Key Innovation:
        Instead of creating one global model, this maintains N cluster models
        (where N = number of clusters), synchronizing only low-level feature
        extraction while preserving high-level strategic differences.
    
    Example:
        >>> aggregator = InterClusterAggregator(
        ...     framework='pytorch',
        ...     shared_layer_patterns=['input_conv', 'representation.0']
        ... )
        >>> cluster_models = {
        ...     'cluster_aggressive': model_state_1,
        ...     'cluster_positional': model_state_2
        ... }
        >>> cluster_metrics = {
        ...     'cluster_aggressive': {'samples': 16000},
        ...     'cluster_positional': {'samples': 16000}
        ... }
        >>> weights = aggregator.get_aggregation_weights(cluster_metrics)
        >>> updated_models, metrics = await aggregator.aggregate(
        ...     cluster_models, weights, round_num=1
        ... )
    """
    
    def __init__(
        self, 
        framework: str = "pytorch",
        compression: bool = True,
        shared_layer_patterns: List[str] = None,
        cluster_specific_patterns: List[str] = None,
        weighting_strategy: str = "samples"
    ):
        """
        Initialize the inter-cluster aggregator.
        
        Args:
            framework: ML framework to use ('pytorch' or 'tensorflow')
            compression: Whether to use compression for model serialization
            shared_layer_patterns: List of layer name patterns to share across clusters
                                   Supports wildcards (e.g., "input_conv", "representation.0")
            cluster_specific_patterns: List of layer name patterns to keep cluster-specific
                                       Supports wildcards (e.g., "policy_head.*", "value_head.*")
            weighting_strategy: How to weight cluster contributions ('samples', 'uniform')
        
        Raises:
            ValueError: If weighting strategy is not supported
        """
        log = logger.bind(context="InterClusterAggregator.__init__")
        log.info(f"Initializing InterClusterAggregator with {weighting_strategy} weighting")
        super().__init__(framework, compression)
        
        # Validate weighting strategy
        if weighting_strategy not in ["samples", "uniform"]:
            log.error(f"Unsupported weighting strategy: {weighting_strategy}")
            raise ValueError(f"Unsupported weighting strategy: {weighting_strategy}")
        self.weighting_strategy = weighting_strategy
        
        # Set default layer patterns if none provided
        if shared_layer_patterns is None:
            shared_layer_patterns = [
                "input_conv"
            ]
            log.warning(f"No shared_layer_patterns provided, using defaults: {shared_layer_patterns}")
            
        if cluster_specific_patterns is None:
            cluster_specific_patterns = [
                "policy_head.*",
                "value_head.*"
            ]
            log.warning(f"No cluster_specific_patterns provided, using defaults: {cluster_specific_patterns}")
            
        self.shared_layer_patterns = shared_layer_patterns
        self.cluster_specific_patterns = cluster_specific_patterns
        
        # Inter-cluster specific attributes
        self.min_participants = 2  # Need at least 2 clusters to aggregate
        self.max_participants = 10 
        
        log.info(f"Configured with {len(self.shared_layer_patterns)} shared patterns")
        log.info(f"Configured with {len(self.cluster_specific_patterns)} cluster-specific patterns")
        log.debug(f"Shared patterns: {self.shared_layer_patterns}")
        log.debug(f"Cluster-specific patterns: {self.cluster_specific_patterns}")
        log.info("InterClusterAggregator initialized successfully")

    async def aggregate(
        self,
        models: Dict[str, Any],
        weights: Dict[str, float],
        round_num: int = 0
    ) -> Tuple[Dict[str, Any], AggregationMetrics]:
        """
        Aggregate shared layers across cluster models while preserving cluster-specific layers.
        
        This method performs selective aggregation:
        1. Identifies which layers are shared vs. cluster-specific
        2. Aggregates only shared layers using weighted averaging
        3. Updates each cluster model with aggregated shared layers
        4. Preserves cluster-specific layers exactly as they were
        
        Args:
            models: Dictionary mapping cluster_id -> cluster_model_state_dict
            weights: Dictionary mapping cluster_id -> aggregation_weight
            round_num: Current training round number
        
        Returns:
            Tuple of (updated_cluster_models, metrics)
            - updated_cluster_models: Dict[cluster_id, model_state] with shared layers updated
            - metrics: AggregationMetrics instance with details about the aggregation
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If aggregation fails
        """
        log = logger.bind(context="InterClusterAggregator.aggregate")
        log.info(f"Starting inter-cluster aggregation for round {round_num}")
        log.info(f"Aggregating {len(models)} cluster models with {self.weighting_strategy} weighting")
        
        start_time = time.time()
        
        try:
            # Step 1: Validate inputs
            if self._validate_inputs:
                log.debug("Step 1: Validating inputs...")
                self.validate_inputs(models, weights)
                self.check_model_compatibility(models)
                
            # Step 2: Identify shared vs. cluster-specific layers
            log.debug("Step 2: Identifying shared vs. cluster-specific layers...")
            first_cluster = next(iter(models))
            all_layers_name = set(models[first_cluster].keys())
            
            shared_layers = self._identify_shared_layers(all_layers_name)
            cluster_specific_layers = self._identify_cluster_specific_layers(all_layers_name)
            
            log.info(f"Identified {len(shared_layers)} shared layers")
            log.info(f"Identified {len(cluster_specific_layers)} cluster-specific layers")
            log.debug(f"Shared layers: {list(shared_layers)[:5]}...")  # Log first 5
            log.debug(f"Cluster-specific layers: {list(cluster_specific_layers)[:5]}...")

            # Validate layer identification
            self._validate_layer_identification(
                all_layers_name, shared_layers, cluster_specific_layers
            )
            
            # Step 3: Normalize weights
            log.debug("Step 3: Normalizing weights...")
            normalized_weights = normalize_weights(weights)
            log.debug(f"Normalized weights: {normalized_weights}")
            
            # Step 4: Aggregate shared layers across clusters
            log.debug("Step 4: Aggregating shared layers across clusters...")
            aggregated_shared_layers = self._aggregate_shared_layers(
                models, shared_layers, normalized_weights
            )
            log.info("Shared layers aggregated successfully")
            
            # Step 5: Update each cluster model with aggregated shared layers
            log.debug("Step 5: Updating cluster models with aggregated shared layers...")
            updated_models = self._update_cluster_models(
                models, aggregated_shared_layers, cluster_specific_layers
            )
            log.info("Cluster models updated successfully")
            
            # Step 6: Collect metrics
            log.debug("Step 6: Collecting aggregation metrics...")
            aggregation_time = time.time() - start_time
            metrics = self._collect_metrics(
                models=models,
                weights=weights,
                shared_layer_count=len(shared_layers),
                cluster_specific_count=len(cluster_specific_layers),
                aggregation_time=aggregation_time,
                round_num=round_num
            )
            
            # Step 7: Update internal statistics
            self._update_statistics(len(models), aggregation_time)
            
            log.info(f"Inter-cluster aggregation successful: {len(models)} clusters processed")
            log.info(f"Preserved diversity in {len(cluster_specific_layers)} cluster-specific layers")
            
            return updated_models, metrics
        
        except Exception as e:
            log.error(f"Aggregation failed: {e}")
            raise RuntimeError(f"Inter-cluster aggregation failed: {e}")

    def _identify_shared_layers(self, all_layer_names: Set[str]) -> Set[str]:
        """
        Identify which layers should be shared across clusters based on patterns.
        
        Args:
            all_layer_names: Set of all layer names in the model
        
        Returns:
            Set of layer names that match shared patterns
        """
        log = logger.bind(context="InterClusterAggregator._identify_shared_layers")
        log.debug(f"Identifying shared layers from {len(all_layer_names)} total layers")
        
        shared_layers = set()
        
        for layer_name in all_layer_names:
            for pattern in self.shared_layer_patterns:
                if self._matches_pattern(layer_name, pattern):
                    shared_layers.add(layer_name)
                    log.trace(f"Layer '{layer_name}' matched shared pattern '{pattern}'")
                    break  # No need to check other patterns once matched
                
        log.debug(f"Total shared layers identified: {len(shared_layers)}")
        return shared_layers
    
    def _identify_cluster_specific_layers(self, all_layer_names: Set[str]) -> Set[str]:
        """
        Identify which layers should remain cluster-specific based on patterns.
        
        Args:
            all_layer_names: Set of all layer names in the model
        
        Returns:
            Set of layer names that match cluster-specific patterns
        """
        log = logger.bind(context="InterClusterAggregator._identify_cluster_specific_layers")
        log.debug(f"Identifying cluster-specific layers from {len(all_layer_names)} total layers")
        
        cluster_specific_layers = set()
        
        for layer_name in all_layer_names:
            for pattern in self.cluster_specific_patterns:
                if self._matches_pattern(layer_name, pattern):
                    cluster_specific_layers.add(layer_name)
                    log.trace(f"Layer '{layer_name}' matched cluster-specific pattern '{pattern}'")
                    break  # No need to check other patterns once matched
                
        log.debug(f"Total cluster-specific layers identified: {len(cluster_specific_layers)}")
        return cluster_specific_layers
    
    def _matches_pattern(self, layer_name: str, pattern: str) -> bool:
        """
        Check if a layer name matches a pattern (supports wildcards).
        
        Patterns:
            - Exact match: "input_conv" matches "input_conv"
            - Wildcard: "policy_head.*" matches "policy_head.linear", "policy_head.bias", etc.
            - Prefix: "representation." matches "representation.0", "representation.1", etc.
        
        Args:
            layer_name: Name of the layer to check
            pattern: Pattern to match against (may contain wildcards)
        
        Returns:
            True if layer_name matches pattern
        """
        # Convert pattern to regex
        # Replace '.' with '\.' (literal dot) and '*' with '.*' (any characters)
        regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
        regex_pattern = f"^{regex_pattern}$"  # Match full string
        
        return re.match(regex_pattern, layer_name) is not None

    def _validate_layer_identification(
        self, 
        all_layers: Set[str],
        shared_layers: Set[str],
        cluster_specific_layers: Set[str]
    ):
        """
        Validate that layer identification is correct and complete.
        
        Checks:
        1. No overlap between shared and cluster-specific layers
        2. All layers are either shared or cluster-specific
        3. At least some layers are shared (otherwise no point in inter-cluster aggregation)
        
        Args:
            all_layers: All layer names in the model
            shared_layers: Identified shared layers
            cluster_specific_layers: Identified cluster-specific layers
        
        Raises:
            ValueError: If validation fails
        """
        log = logger.bind(context="InterClusterAggregator._validate_layer_identification")
        log.debug("Validating layer identification...")
        
        # Check for overlap
        overlap = shared_layers.intersection(cluster_specific_layers)
        if overlap:
            log.error(f"Layer identification error: Overlap found in layers: {overlap}")
            raise ValueError(f"Layer identification error: Overlap found in layers: {overlap}")
        
        # Check coverage
        identified_layers = shared_layers.union(cluster_specific_layers)
        unidentified_layers = all_layers - identified_layers

        if unidentified_layers:
            log.warning(f"{len(unidentified_layers)} layers not classified as shared or cluster-specific")
            log.warning(f"Unidentified layers will be treated as cluster-specific: {list(unidentified_layers)[:5]}...")
            # This is a warning, not an error - unidentified layers stay cluster-specific by default
        
        # Check that we have shared layers
        if not shared_layers:
            log.error("No shared layers identified - inter-cluster aggregation has no effect")
            raise ValueError("No shared layers identified for aggregation")
        
        # Check that we have cluster-specific layers
        if not cluster_specific_layers and not unidentified_layers:
            log.warning("No cluster-specific layers - all layers will be shared (diversity may be lost)")
        
        log.debug("Layer identification validation passed")
        
    def _aggregate_shared_layers(
        self,
        models: Dict[str, Any],
        shared_layers: Set[str],
        normalized_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Aggregate only the shared layers across all cluster models.
        
        Uses FedAvg algorithm: weighted average of shared layer parameters
        across all clusters.
        
        Args:
            models: Dictionary of cluster_id -> model_state_dict
            shared_layers: Set of layer names to aggregate
            normalized_weights: Normalized aggregation weights (sum to 1.0)
        
        Returns:
            Dictionary containing only the aggregated shared layers
        """
        log = logger.bind(context="InterClusterAggregator._aggregate_shared_layers")
        log.debug(f"Aggregating {len(shared_layers)} shared layers across {len(models)} clusters")
        
        aggregated_shared = {}
        
        # Aggregate based on parameter type
        try:
            for layer_name in shared_layers:
                log.trace(f"Aggregating layer '{layer_name}'")
                
                # Get the layer from first cluster to determine structure
                first_cluster = next(iter(models))
                reference_layer = models[first_cluster][layer_name]
                
                # Aggregate based on parameter type
                if isinstance(reference_layer, list):
                    if isinstance(reference_layer[0], list):
                        # 2D list (e.g., Flow weights)
                        aggregated_value = self._aggregate_2d_shared_layer(layer_name, models, normalized_weights)
                    else:
                        # 1D list (e.g., biases)
                        aggregated_value = self._aggregate_1d_shared_layer(layer_name, models, normalized_weights)
                else:
                    # Scalar parameter (e.g., single value)
                    aggregated_value = self._aggregate_scalar_shared_layer(layer_name, models, normalized_weights)
                    
                aggregated_shared[layer_name] = aggregated_value
                
            log.debug("Shared layer aggregation completed")
            return aggregated_shared
        except Exception as e:
            log.error(f"Failed to aggregate shared layers: {e}")
            raise RuntimeError(f"Shared layer aggregation failed: {e}")
        
    def _aggregate_2d_shared_layer(
        self,
        layer_name: str,
        models: Dict[str, Any],
        normalized_weights: Dict[str, float]
    ) -> List[List[float]]:
        """
        Aggregate a 2D shared layer (weight matrix) across clusters.
        
        Args:
            layer_name: Name of the layer to aggregate
            models: Dictionary of cluster models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated 2D parameter
        """
        log = logger.bind(context="InterClusterAggregator._aggregate_2d_shared_layer")
        log.trace(f"Aggregating 2D shared layer: {layer_name}")
        
        # Get dimensions from first cluster
        first_cluster = next(iter(models))
        reference_param = models[first_cluster][layer_name]
        rows = len(reference_param)
        cols = len(reference_param[0])
        
        # Initialize result matrix with zeros
        aggregated = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        # Compute weighted sum across clusters
        for cluster_id, model in models.items():
            weight = normalized_weights[cluster_id]
            param = model[layer_name]
            
            for i in range(rows):
                for j in range(cols):
                    aggregated[i][j] += weight * param[i][j]
        
        log.trace(f"2D layer {layer_name} aggregation complete")
        return aggregated
    
    def _aggregate_1d_shared_layer(
        self,
        layer_name: str,
        models: Dict[str, Any],
        normalized_weights: Dict[str, float]
    ) -> List[float]:
        """
        Aggregate a 1D shared layer (bias vector) across clusters.
        
        Args:
            layer_name: Name of the layer to aggregate
            models: Dictionary of cluster models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated 1D parameter
        """
        # Get length from first cluster
        first_cluster = next(iter(models))
        reference_param = models[first_cluster][layer_name]
        length = len(reference_param)
        
        # Initialize result vector with zeros
        aggregated = [0.0 for _ in range(length)]
        
        # Compute weighted sum across clusters
        for cluster_id, model in models.items():
            weight = normalized_weights[cluster_id]
            param = model[layer_name]
            
            for i in range(length):
                aggregated[i] += weight * param[i]
        
        return aggregated
    
    def _aggregate_scalar_shared_layer(
        self,
        layer_name: str,
        models: Dict[str, Any],
        normalized_weights: Dict[str, float]
    ) -> float:
        """
        Aggregate a scalar shared parameter across clusters.
        
        Args:
            layer_name: Name of the parameter to aggregate
            models: Dictionary of cluster models
            normalized_weights: Normalized aggregation weights
        
        Returns:
            Aggregated scalar value
        """
        aggregated = 0.0
        
        for cluster_id, model in models.items():
            weight = normalized_weights[cluster_id]
            param = model[layer_name]
            aggregated += weight * param
        
        return aggregated
    
    def _update_cluster_models(
        self,
        models: Dict[str, Any],
        aggregated_shared_layers: Dict[str, Any],
        cluster_specific_layers: Set[str]
    ) -> Dict[str, Any]:
        """
        Update each cluster model with aggregated shared layers.
        
        Creates new model state dicts for each cluster where:
        - Shared layers are replaced with aggregated versions
        - Cluster-specific layers are preserved exactly as they were
        
        Args:
            models: Original cluster models
            aggregated_shared_layers: Aggregated shared layer parameters
            cluster_specific_layers: Set of layer names to preserve
        
        Returns:
            Dictionary of updated cluster models
        """
        log = logger.bind(context="InterClusterAggregator._update_cluster_models")
        log.debug(f"Updating {len(models)} cluster models with aggregated shared layers")
        
        updated_models = {}
        
        for cluster_id, model_state in models.items():
            log.trace(f"Updating model for cluster '{cluster_id}'")

            # Start with a copy of the original model
            updated_model = model_state.copy()
            
            # Replace shared layers with aggregated versions
            for layer_name, aggregated_value in aggregated_shared_layers.items():
                updated_model[layer_name] = aggregated_value
                log.trace(f"Cluster '{cluster_id}': Updated shared layer '{layer_name}'")
                
            # Cluster-specific layers remain unchanged
            cluster_specific_count = sum(
                1 for layer in model_state.keys() if layer in cluster_specific_layers
            )
            log.trace(f"Preserved {cluster_specific_count} cluster-specific layers in {cluster_id}")
            
            updated_models[cluster_id] = updated_model
        
        log.debug(f"Updated {len(updated_models)} cluster models successfully")
        return updated_models

    def get_aggregation_weights(
        self,
        cluster_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate aggregation weights for clusters based on metrics.
        
        Implements different weighting strategies:
        - 'samples': Weight by total training samples in cluster (default)
        - 'uniform': Equal weight for all clusters
        
        Args:
            cluster_metrics: Dictionary mapping cluster_id -> metrics dict
                            Expected keys: 'samples', 'loss'
        
        Returns:
            Dictionary mapping cluster_id -> aggregation_weight
        
        Raises:
            ValueError: If metrics are invalid
        """
        log = logger.bind(context="InterClusterAggregator.get_aggregation_weights")
        log.info(f"Calculating weights using '{self.weighting_strategy}' strategy")
        
        # Validate cluster metrics
        validate_participant_metrics(cluster_metrics)
        
        weights = {}
        
        if self.weighting_strategy == "samples":
            log.debug("Weighting by total samples per cluster...")
            # Weight by total samples in each cluster
            for cluster_id, metrics in cluster_metrics.items():
                weights[cluster_id] = float(metrics.get('samples', 0))
        
        elif self.weighting_strategy == "uniform":
            log.debug("Using uniform weighting...")
            # Equal weight for all clusters
            for cluster_id in cluster_metrics.keys():
                weights[cluster_id] = 1.0
        
        log.debug(f"Raw weights before normalization: {weights}")
        return weights
    
    def _collect_metrics(
        self,
        models: Dict[str, Any],
        weights: Dict[str, float],
        shared_layer_count: int,
        cluster_specific_count: int,
        aggregation_time: float,
        round_num: int
    ) -> AggregationMetrics:
        """
        Collect metrics about the inter-cluster aggregation process.
        
        Args:
            models: Dictionary of cluster models
            weights: Aggregation weights used
            shared_layer_count: Number of shared layers aggregated
            cluster_specific_count: Number of cluster-specific layers preserved
            aggregation_time: Time taken for aggregation
            round_num: Current training round
        
        Returns:
            AggregationMetrics object with collected metrics
        """
        log = logger.bind(context="InterClusterAggregator._collect_metrics")
        log.info("Collecting inter-cluster aggregation metrics...")
        
        # Calculate total samples across all clusters
        total_samples = int(sum(weights.values()))
        
        # Calculate diversity if enabled
        model_diversity = None
        if self.collect_metrics and len(models) > 1:
            # For inter-cluster, diversity should be high (preserved playstyles)
            model_diversity = self.calculate_model_diversity(models)
        
        metrics = AggregationMetrics(
            aggregation_time=aggregation_time,
            participant_count=len(models),
            total_samples=total_samples,
            average_loss=0.0,  # TODO: Would need loss values from clusters
            model_diversity=model_diversity,
            framework=self.framework,
            aggregation_round=round_num,
            additional_metrics={
                'weighting_strategy': self.weighting_strategy,
                'aggregation_type': 'inter_cluster',
                'shared_layer_count': shared_layer_count,
                'cluster_specific_count': cluster_specific_count,
                'shared_layer_patterns': self.shared_layer_patterns,
                'cluster_specific_patterns': self.cluster_specific_patterns
            }
        )
        
        log.debug(f"Collected metrics: {metrics.participant_count} clusters, "
                 f"{shared_layer_count} shared layers, "
                 f"{cluster_specific_count} cluster-specific layers, "
                 f"{metrics.aggregation_time:.3f}s")
        
        return metrics
    
    def set_shared_layer_patterns(self, patterns: List[str]):
        """
        Update the shared layer patterns.
        
        Args:
            patterns: New list of layer name patterns for sharing
        
        Raises:
            ValueError: If patterns list is empty
        """
        log = logger.bind(context="InterClusterAggregator.set_shared_layer_patterns")
        
        if not patterns:
            log.error("Shared layer patterns cannot be empty")
            raise ValueError("Shared layer patterns cannot be empty")
        
        old_patterns = self.shared_layer_patterns
        self.shared_layer_patterns = patterns
        log.info(f"Updated shared layer patterns from {len(old_patterns)} to {len(patterns)} patterns")
        log.debug(f"New shared patterns: {patterns}")
    
    def set_cluster_specific_patterns(self, patterns: List[str]):
        """
        Update the cluster-specific layer patterns.
        
        Args:
            patterns: New list of layer name patterns to keep cluster-specific
        """
        log = logger.bind(context="InterClusterAggregator.set_cluster_specific_patterns")
        
        old_patterns = self.cluster_specific_patterns
        self.cluster_specific_patterns = patterns
        log.info(f"Updated cluster-specific patterns from {len(old_patterns)} to {len(patterns)} patterns")
        log.debug(f"New cluster-specific patterns: {patterns}")


def create_inter_cluster_aggregator(config: Dict[str, Any]) -> InterClusterAggregator:
    """
    Factory function to create an inter-cluster aggregator from configuration.
    
    Args:
        config: Configuration dictionary with keys:
               - framework: 'pytorch' or 'tensorflow'
               - compression: bool
               - shared_layer_patterns: List[str]
               - cluster_specific_patterns: List[str]
               - weighting_strategy: 'samples' or 'uniform'
    
    Returns:
        Configured InterClusterAggregator instance
    """
    log = logger.bind(context="create_inter_cluster_aggregator")
    log.info("Creating inter-cluster aggregator from configuration")
    
    framework = config.get('framework', 'pytorch')
    compression = config.get('compression', True)
    shared_layer_patterns = config.get('shared_layer_patterns', None)
    cluster_specific_patterns = config.get('cluster_specific_patterns', None)
    weighting_strategy = config.get('weighting_strategy', 'samples')
    
    aggregator = InterClusterAggregator(
        framework=framework,
        compression=compression,
        shared_layer_patterns=shared_layer_patterns,
        cluster_specific_patterns=cluster_specific_patterns,
        weighting_strategy=weighting_strategy
    )
    
    log.info("Inter-cluster aggregator created successfully")
    return aggregator