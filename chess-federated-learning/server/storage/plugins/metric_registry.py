"""
Metric computation plugin registry.

This module provides a plugin system for computing custom metrics:
- MetricComputer: Protocol defining the plugin interface
- MetricRegistry: Registry for managing and executing plugins

Plugins can compute arbitrary metrics from context data and are
automatically discovered and executed.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class MetricComputer(Protocol):
    """
    Protocol for metric computation plugins.

    Implementations should compute metrics from the provided context
    and return a dictionary of metric name -> value pairs.
    """

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metrics from context.

        Args:
            context: Dictionary containing data needed for computation.
                    Common keys:
                    - 'models': Dict of cluster models
                    - 'round': Current training round
                    - 'node_metrics': Dict of node metrics
                    - 'cluster_metrics': Dict of cluster metrics
                    - Any custom data

        Returns:
            Dictionary of metric_name -> metric_value

        Example:
            def compute(self, context):
                models = context.get('models', {})
                distance = self._compute_distance(models)
                return {
                    "inter_cluster_distance": distance,
                    "diversity_score": 1.0 - distance
                }
        """
        ...

    def get_name(self) -> str:
        """
        Get the name of this metric computer.

        Returns:
            Human-readable name for this plugin
        """
        ...

    def get_required_context_keys(self) -> List[str]:
        """
        Get the list of required context keys.

        Returns:
            List of keys that must be present in context

        Example:
            def get_required_context_keys(self):
                return ['models', 'round']
        """
        ...


class MetricRegistry:
    """
    Registry for metric computation plugins.

    Manages registration, validation, and execution of metric plugins.
    Supports dynamic registration and automatic execution.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._computers: Dict[str, MetricComputer] = {}
        logger.info("Initialized MetricRegistry")

    def register(self, name: str, computer: MetricComputer) -> None:
        """
        Register a metric computer.

        Args:
            name: Unique name for this computer
            computer: MetricComputer instance

        Raises:
            TypeError: If computer doesn't implement MetricComputer protocol
            ValueError: If name already registered
        """
        # Validate protocol implementation
        if not isinstance(computer, MetricComputer):
            raise TypeError(
                f"Computer must implement MetricComputer protocol, got {type(computer)}"
            )

        # Check for duplicate names
        if name in self._computers:
            raise ValueError(f"Metric computer '{name}' already registered")

        self._computers[name] = computer
        logger.info(
            f"Registered metric computer: {name} ({computer.get_name()})"
        )

    def unregister(self, name: str) -> None:
        """
        Unregister a metric computer.

        Args:
            name: Name of computer to remove

        Raises:
            KeyError: If name not found
        """
        if name not in self._computers:
            raise KeyError(f"Metric computer '{name}' not registered")

        del self._computers[name]
        logger.info(f"Unregistered metric computer: {name}")

    def has_computer(self, name: str) -> bool:
        """Check if a computer is registered."""
        return name in self._computers

    def get_computer(self, name: str) -> MetricComputer:
        """
        Get a registered computer.

        Args:
            name: Name of computer

        Returns:
            MetricComputer instance

        Raises:
            KeyError: If name not found
        """
        if name not in self._computers:
            raise KeyError(f"Metric computer '{name}' not registered")

        return self._computers[name]

    def list_computers(self) -> List[str]:
        """
        List all registered computers.

        Returns:
            List of registered computer names
        """
        return list(self._computers.keys())

    def compute(
        self,
        name: str,
        context: Dict[str, Any],
        validate_context: bool = True
    ) -> Dict[str, Any]:
        """
        Compute metrics using a specific computer.

        Args:
            name: Name of computer to use
            context: Context data for computation
            validate_context: If True, validate required context keys

        Returns:
            Dictionary of computed metrics

        Raises:
            KeyError: If computer not found or required context missing
        """
        computer = self.get_computer(name)

        # Validate context if requested
        if validate_context:
            self._validate_context(computer, context)

        # Compute metrics
        try:
            metrics = computer.compute(context)
            logger.debug(
                f"Computed metrics with {name}: {list(metrics.keys())}"
            )
            return metrics
        except Exception as e:
            logger.error(f"Error computing metrics with {name}: {e}")
            raise

    def compute_all(
        self,
        context: Dict[str, Any],
        skip_on_error: bool = True,
        validate_context: bool = False
    ) -> Dict[str, Any]:
        """
        Compute metrics using all registered computers.

        Args:
            context: Context data for computation
            skip_on_error: If True, skip computers that fail
            validate_context: If True, validate required context keys

        Returns:
            Dictionary of all computed metrics (merged)

        Note:
            If computers produce overlapping metric names, later
            computers will overwrite earlier ones.
        """
        all_metrics = {}

        for name, computer in self._computers.items():
            try:
                # Validate context if requested
                if validate_context:
                    self._validate_context(computer, context)

                # Compute metrics
                metrics = computer.compute(context)
                all_metrics.update(metrics)

                logger.debug(
                    f"Computed {len(metrics)} metrics with {name}"
                )

            except Exception as e:
                if skip_on_error:
                    logger.warning(
                        f"Skipping {name} due to error: {e}"
                    )
                else:
                    raise

        return all_metrics

    def _validate_context(
        self,
        computer: MetricComputer,
        context: Dict[str, Any]
    ) -> None:
        """
        Validate that context contains required keys.

        Args:
            computer: MetricComputer to validate for
            context: Context data

        Raises:
            KeyError: If required keys missing
        """
        required_keys = computer.get_required_context_keys()

        missing_keys = [
            key for key in required_keys
            if key not in context
        ]

        if missing_keys:
            raise KeyError(
                f"Missing required context keys for {computer.get_name()}: "
                f"{missing_keys}"
            )

    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about the registry.

        Returns:
            Dictionary with:
            - count: Number of registered computers
            - computers: List of computer info dicts
        """
        computers_info = []

        for name, computer in self._computers.items():
            computers_info.append({
                "name": name,
                "display_name": computer.get_name(),
                "required_keys": computer.get_required_context_keys()
            })

        return {
            "count": len(self._computers),
            "computers": computers_info
        }


class BaseMetricComputer:
    """
    Base class for metric computers (optional convenience).

    Provides default implementations of protocol methods.
    Subclasses only need to implement compute().
    """

    def get_name(self) -> str:
        """Get name from class name by default."""
        return self.__class__.__name__

    def get_required_context_keys(self) -> List[str]:
        """Default: no required keys."""
        return []

    def compute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute()"
        )
