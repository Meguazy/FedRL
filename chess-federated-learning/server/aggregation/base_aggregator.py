"""
Base aggregator for federated learning model aggregation.

This module provides the abstract base class and common utilities for all
aggregation strategies in the federated learning system. It defines the
interface that all aggregators must implement and provides shared functionality.

Key Components:
    - BaseAggregator: Abstract base class for all aggregation strategies
    - AggregationMetrics: Data structure for tracking aggregation statistics
    - Common utilities for weight validation and model state handling
    - Integration with model serialization and cluster management

Architecture:
    - Abstract interface ensures consistent aggregator implementations
    - Standardized metrics collection across all aggregation types
    - Support for both PyTorch and TensorFlow model formats
    - Comprehensive logging and error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from loguru import logger

from common.model_serialization import get_serializer, ModelSerializer


@dataclass
class AggregationMetrics:
    """
    Metrics and metadata collected during aggregation.
    
    This dataclass tracks important information about the aggregation process,
    including performance metrics, participant statistics, and quality measures.
    
    Attributes:
        aggregation_time: Time taken to perform aggregation (seconds)
        participant_count: Number of models that participated in aggregation
        total_samples: Total number of training samples across all participants
        average_loss: Weighted average loss across participants
        model_diversity: Measure of diversity between participant models (optional)
        convergence_metric: Metric indicating convergence progress (optional)
        framework: ML framework used ('pytorch' or 'tensorflow')
        aggregation_round: Training round number
        timestamp: When aggregation was performed
        additional_metrics: Dictionary for custom metrics
    """
    aggregation_time: float = 0.0
    participant_count: int = 0
    total_samples: int = 0
    average_loss: float = 0.0
    model_diversity: Optional[float] = None
    convergence_metric: Optional[float] = None
    framework: str = "pytorch"  # or 'tensorflow'
    aggregation_round: int = 0
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    
class BaseAggregator(ABC):
    """
    Abstract base class for all federated learning aggregation strategies.
    
    This class defines the interface that all aggregators must implement and
    provides common functionality for model aggregation. It supports both
    PyTorch and TensorFlow models and includes comprehensive logging and
    metrics collection.
    
    The aggregation process follows these steps:
    1. Validate input models and weights
    2. Prepare models for aggregation (deserialization if needed)
    3. Perform the actual aggregation algorithm
    4. Collect metrics and validate results
    5. Return aggregated model with metadata
    
    Subclasses must implement the core aggregation logic while the base class
    handles common tasks like validation, serialization, and metrics collection.
    """
    
    def __init__(self, framework: str = "pytorch", compression: bool = True):
        """
        Initialize the base aggregator.
        
        Args:
            framework: ML framework to use ('pytorch' or 'tensorflow')
            compression: Whether to use compression for model serialization
        
        Raises:
            ValueError: If framework is not supported
        """
        log = logger.bind(context="BaseAggregator.__init__")
        log.info(f"Initializing aggregator for {framework}")
        
        # Validate framework
        if framework not in ["pytorch", "tensorflow"]:
            log.error(f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")
        
        self.framework = framework
        self.compression = compression
        
        # Initialize model serializer
        try:
            self.serializer: ModelSerializer = get_serializer(framework, compression=compression)
            log.info(f"Using serializer: {self.serializer.__class__.__name__}")
        except Exception as e:
            log.exception("Failed to initialize model serializer")
            raise e
        
        # Aggregation statistics
        self.total_aggregations = 0
        self.total_aggregation_time = 0.0
        self.total_models_aggregated = 0
        self.creation_time = time.time()
        
        # Configuration
        self._validate_inputs = True  # Whether to validate input models
        self.collect_metrics = True  # Whether to collect aggregation metrics
        self.min_participants = 1    # Minimum participants required for aggregation
        self.max_participants = 1000  # Maximum participants allowed for aggregation
        
    @abstractmethod
    async def aggregate(
        self, 
        models: Dict[str, Any], 
        weights: Dict[str, float], 
        round_number: int = 0
    ) -> Tuple[Any, AggregationMetrics]:
        """
        Abstract method to perform model aggregation.
        
        This method must be implemented by all subclasses. It takes a dictionary
        of models and their corresponding weights, performs the aggregation
        according to the specific strategy, and returns the aggregated model
        along with relevant metrics.

        Args:
            models: Dictionary mapping participant IDs to their model states
            weights: Dictionary mapping participant IDs to their weights
            round_number: Current training round number
            
        Returns:
            A tuple containing:
                - The aggregated model state
                - An AggregationMetrics instance with details about the aggregation
                
        Raises:
            NotImplementedError: If the method is not implemented in a subclass
        """
        pass
    
    @abstractmethod
    def get_aggregation_weights(
        self,
        participant_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate aggregation weights based on participant metrics.
        
        This method determines how much each participant should contribute
        to the aggregated model. Common strategies include:
        - Equal weighting (all participants weighted equally)
        - Sample-based weighting (weight by number of training samples)
        - Performance-based weighting (weight by model quality)
        - Custom weighting strategies
        
        Args:
            participant_metrics: Dictionary mapping participant_id -> metrics dict
                                Expected metrics: {'samples': int, 'loss': float, ...}
        
        Returns:
            Dictionary mapping participant_id -> aggregation_weight
        
        Raises:
            ValueError: If metrics are invalid or insufficient
        """
        pass
    
    def validate_inputs(
        self,
        models: Dict[str, Any],
        weights: Dict[str, float]
    ) -> bool:
        """
        Validate inputs to the aggregation process.
        
        Checks that models and weights are properly formatted and compatible
        for aggregation. This includes validating data types, checking for
        missing participants, and ensuring model compatibility.
        
        Args:
            models: Dictionary of participant models
            weights: Dictionary of aggregation weights
        
        Returns:
            True if inputs are valid
        
        Raises:
            ValueError: If inputs are invalid
        """
        log = logger.bind(context="BaseAggregator.validate_inputs")
        log.debug(f"Validating inputs: {len(models)} models, {len(weights)} weights")
        
        # Check basic requirements
        if not models:
            log.error("No models provided for aggregation")
            raise ValueError("No models provided for aggregation")
        
        if not weights:
            log.error("No weights provided for aggregation")
            raise ValueError("No weights provided for aggregation")
        
        # Check participant counts limits
        if len(models) < self.min_participants:
            log.error(f"Not enough participants: {len(models)} < {self.min_participants}")
            raise ValueError(f"Not enough participants: {len(models)} < {self.min_participants}")
        if len(models) > self.max_participants:
            log.error(f"Too many participants: {len(models)} > {self.max_participants}")
            raise ValueError(f"Too many participants: {len(models)} > {self.max_participants}")
        
        # Check that models and weights have same participants
        model_participants = set(models.keys())
        weight_participants = set(weights.keys())
        
        if model_participants != weight_participants:
            missing_weights = model_participants - weight_participants
            missing_models = weight_participants - model_participants
            
            if missing_weights:
                log.error(f"Missing weights for participants: {missing_weights}")
                raise ValueError(f"Missing weights for participants: {missing_weights}")
            if missing_models:
                log.error(f"Missing models for participants: {missing_models}")
                raise ValueError(f"Missing models for participants: {missing_models}")
            
        # Validate individual models
        for pid, weight in weights.items():
            if not isinstance(weight, (int, float)):
                log.error(f"Invalid weight type for participant {pid}: {type(weight)}")
                raise ValueError(f"Invalid weight type for participant {pid}: {type(weight)}")
            if weight < 0:
                log.error(f"Negative weight for participant {pid}: {weight}")
                raise ValueError(f"Negative weight for participant {pid}: {weight}")
            if weight == 0:
                log.warning(f"Zero weight for participant {pid}")
                
        # Check that total weight is positive
        total_weight = sum(weights.values())
        if total_weight <= 0:
            log.error(f"Total weight must be positive, got {total_weight}")
            raise ValueError(f"Total weight must be positive, got {total_weight}")
        
        log.info("Input validation passed")
        return True
    
    def check_model_compatibility(self, models: Dict[str, Any]) -> bool:
        """
        Check if models have compatible structure for aggregation.
        
        Verifies that all models have the same keys (layers) and compatible
        shapes. This is essential for successful aggregation.
        
        Args:
            models: Dictionary of participant models
        
        Returns:
            True if models are compatible
        
        Raises:
            ValueError: If models are incompatible
        """
        log = logger.bind(context="BaseAggregator.check_model_compatibility")
        log.debug(f"Checking compatibility of {len(models)} models")
        
        if not models:
            return True  # Nothing to check
        
        # Get reference model structure
        reference_id = next(iter(models))
        reference_model = models[reference_id]
        reference_keys = set(reference_model.keys())
        
        log.debug(f"Using {reference_id} as reference model with {len(reference_keys)} keys")

        # Check all other models against reference
        for pid, model in models.items():
            if pid == reference_id:
                continue  # Skip reference
            
            model_keys = set(model.keys())
            
            # Check for missing keys
            missing_keys = reference_keys - model_keys
            if missing_keys:
                log.error(f"Model {pid} is missing keys: {missing_keys}")
                raise ValueError(f"Model {pid} is missing keys: {missing_keys}")
            
            # Check for extra keys
            extra_keys = model_keys - reference_keys
            if extra_keys:
                log.error(f"Model {pid} has extra keys: {extra_keys}")
                raise ValueError(f"Model {pid} has extra keys: {extra_keys}")
            
            # Check shapes for each key
            for key in reference_keys:
                ref_value = reference_model[key]
                model_value = model[key]
                
                # For list-based representations (e.g. TensorFlow)
                if isinstance(ref_value, list) and isinstance(model_value, list):
                    if len(ref_value) != len(model_value):
                        log.error(f"Model {pid} key '{key}' has different length: {len(model_value)} != {len(ref_value)}")
                        raise ValueError(f"Model {pid} key '{key}' has different length: {len(model_value)} != {len(ref_value)}")
                    
                    # Check nested lists for 2D weights
                    if ref_value and isinstance(ref_value[0], list):
                        for i, (ref_row, model_row) in enumerate(zip(ref_value, model_value)):
                            if len(ref_row) != len(model_row):
                                log.error(f"Model {pid} key '{key}' row {i} has different length: {len(model_row)} != {len(ref_row)}")
                                raise ValueError(f"Model {pid} key '{key}' row {i} has different length: {len(model_row)} != {len(ref_row)}")
                            
            log.debug(f"Model {pid} is compatible")
            
        log.info("All models are compatible")
        return True
    
    def calculate_model_diversity(self, models: Dict[str, Any]) -> float:
        """
        Calculate diversity metric between models.
        
        Measures how different the models are from each other. Higher diversity
        indicates that participants have learned different representations.
        
        Args:
            models: Dictionary of participant models
        
        Returns:
            Diversity metric (higher = more diverse)
        """
        log = logger.bind(context="BaseAggregator.calculate_model_diversity")
        log.debug(f"Calculating diversity for {len(models)} models")
        
        if len(models) < 2:
            log.debug("Not enough models to calculate diversity")
            return 0.0  # No diversity with less than 2 models
        
        try:
            # Simple diversity metrics: average pairwise distance
            model_list = list(models.values())
            total_distance = 0.0
            pair_count = 0
            
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    dist = self._calculate_parameter_distance(model_list[i], model_list[j])
                    total_distance += dist
                    pair_count += 1
                    
            diversity = total_distance / pair_count if pair_count > 0 else 0.0
            log.debug(f"Calculated model diversity: {diversity}")
            return diversity
        except Exception as e:
            log.exception("Failed to calculate model diversity")
            return 0.0
        
    def _calculate_parameter_distance(self, model1: Dict[str, Any], model2: Dict[str, Any]) -> float:
        """
        Calculate distance between two models' parameters.
        
        Args:
            model1: First model's parameters
            model2: Second model's parameters
        
        Returns:
            Distance metric between models
        """
        log = logger.bind(context="BaseAggregator._calculate_parameter_distance")
        log.debug("Calculating parameter distance between two models")
        
        total_distance = 0.0
        total_params = 0
        
        for key in model1.keys():
            if key not in model2:
                continue  # Skip missing keys
            
            params1 = model1[key]
            params2 = model2[key]
            
            # Handle list-based parameters (e.g. TensorFlow)
            if isinstance(params1, list) and isinstance(params2, list):
                if isinstance(params1[0], list):  # 2D weights
                    for row1, row2 in zip(params1, params2):
                        for p1, p2 in zip(row1, row2):
                            total_distance += abs(p1 - p2)
                            total_params += 1
                else:  # 1D weights
                    for p1, p2 in zip(params1, params2):
                        total_distance += abs(p1 - p2)
                        total_params += 1

        return total_distance / total_params if total_params > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregator statistics.
        
        Returns:
            Dictionary containing performance and usage statistics
        """
        uptime = time.time() - self.creation_time
        avg_aggregation_time = (self.total_aggregation_time / self.total_aggregations 
                               if self.total_aggregations > 0 else 0.0)
        
        return {
            "framework": self.framework,
            "compression": self.compression,
            "total_aggregations": self.total_aggregations,
            "total_models_aggregated": self.total_models_aggregated,
            "total_aggregation_time": self.total_aggregation_time,
            "average_aggregation_time": avg_aggregation_time,
            "uptime_seconds": uptime,
            "min_participants": self.min_participants,
            "max_participants": self.max_participants
        }
    
    def _update_statistics(self, participant_count: int, aggregation_time: float):
        """
        Update internal statistics after aggregation.
        
        Args:
            participant_count: Number of participants in this aggregation
            aggregation_time: Time taken for aggregation
        """
        self.total_aggregations += 1
        self.total_models_aggregated += participant_count
        self.total_aggregation_time += aggregation_time
        
        log = logger.bind(context="BaseAggregator._update_statistics")
        log.debug(f"Updated stats: {self.total_aggregations} aggregations, "
                 f"{aggregation_time:.3f}s last aggregation")


def validate_participant_metrics(metrics: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate participant metrics dictionary.
    
    Args:
        metrics: Dictionary mapping participant_id -> metrics dict
    
    Returns:
        True if metrics are valid
    
    Raises:
        ValueError: If metrics are invalid
    """
    log = logger.bind(context="validate_participant_metrics")
    log.debug(f"Validating metrics for {len(metrics)} participants")
    
    if not metrics:
        log.error("No participant metrics provided")
        raise ValueError("No participant metrics provided")
    
    required_fields = ['samples', 'loss']
    
    for participant_id, participant_metrics in metrics.items():
        if not isinstance(participant_metrics, dict):
            log.error(f"Metrics for {participant_id} must be a dictionary")
            raise ValueError(f"Metrics for {participant_id} must be a dictionary")
        
        # Check required fields
        for field in required_fields:
            if field not in participant_metrics:
                log.error(f"Missing required field '{field}' in metrics for {participant_id}")
                raise ValueError(f"Missing required field '{field}' in metrics for {participant_id}")
        
        # Validate field values
        samples = participant_metrics['samples']
        if not isinstance(samples, int) or samples <= 0:
            log.error(f"Invalid samples value for {participant_id}: {samples}")
            raise ValueError(f"Invalid samples value for {participant_id}: {samples}")
        
        loss = participant_metrics['loss']
        if not isinstance(loss, (int, float)) or loss < 0:
            log.error(f"Invalid loss value for {participant_id}: {loss}")
            raise ValueError(f"Invalid loss value for {participant_id}: {loss}")
    
    log.debug("Participant metrics validation passed")
    return True


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize aggregation weights to sum to 1.0.
    
    Args:
        weights: Dictionary of weights to normalize
    
    Returns:
        Dictionary with normalized weights
    
    Raises:
        ValueError: If total weight is zero or negative
    """
    log = logger.bind(context="normalize_weights")
    
    total_weight = sum(weights.values())
    if total_weight <= 0:
        log.error(f"Cannot normalize weights with total {total_weight}")
        raise ValueError(f"Cannot normalize weights with total {total_weight}")
    
    normalized = {participant_id: weight / total_weight 
                 for participant_id, weight in weights.items()}
    
    log.debug(f"Normalized {len(weights)} weights (total was {total_weight:.6f})")
    return normalized