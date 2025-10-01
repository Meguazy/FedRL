"""
Unit tests for MetricRegistry and plugin system.

Tests plugin registration, validation, and execution.
"""

import pytest

from server.storage.plugins.metric_registry import (
    BaseMetricComputer,
    MetricRegistry,
)


class SimpleMetricComputer(BaseMetricComputer):
    """Simple test metric computer."""

    def get_name(self) -> str:
        return "Simple Test Computer"

    def get_required_context_keys(self):
        return ["value"]

    def compute(self, context):
        value = context.get("value", 0)
        return {
            "doubled": value * 2,
            "squared": value ** 2
        }


class NoRequirementsComputer(BaseMetricComputer):
    """Computer with no required context keys."""

    def compute(self, context):
        return {"constant": 42}


class FailingComputer(BaseMetricComputer):
    """Computer that always fails."""

    def compute(self, context):
        raise ValueError("Intentional failure")


class TestMetricRegistry:
    """Test MetricRegistry functionality."""

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = MetricRegistry()
        assert len(registry.list_computers()) == 0

    def test_register_computer(self):
        """Test registering a computer."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()

        registry.register("simple", computer)

        assert registry.has_computer("simple")
        assert "simple" in registry.list_computers()

    def test_register_validates_protocol(self):
        """Test that register validates MetricComputer protocol."""
        registry = MetricRegistry()

        # Invalid type should raise
        with pytest.raises(TypeError, match="MetricComputer protocol"):
            registry.register("invalid", "not a computer")

    def test_duplicate_name_rejected(self):
        """Test that duplicate computer names are rejected."""
        registry = MetricRegistry()
        computer1 = SimpleMetricComputer()
        computer2 = SimpleMetricComputer()

        registry.register("test", computer1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", computer2)

    def test_unregister_computer(self):
        """Test unregistering a computer."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()

        registry.register("test", computer)
        assert registry.has_computer("test")

        registry.unregister("test")
        assert not registry.has_computer("test")

    def test_unregister_nonexistent_raises(self):
        """Test that unregistering non-existent computer raises error."""
        registry = MetricRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_has_computer(self):
        """Test checking if computer exists."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()

        assert not registry.has_computer("test")

        registry.register("test", computer)
        assert registry.has_computer("test")

    def test_get_computer(self):
        """Test retrieving a computer."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()
        registry.register("test", computer)

        retrieved = registry.get_computer("test")
        assert retrieved is computer

    def test_get_nonexistent_computer_raises(self):
        """Test that getting non-existent computer raises error."""
        registry = MetricRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_computer("nonexistent")

    def test_list_computers(self):
        """Test listing all registered computers."""
        registry = MetricRegistry()

        registry.register("comp1", SimpleMetricComputer())
        registry.register("comp2", NoRequirementsComputer())
        registry.register("comp3", SimpleMetricComputer())

        computers = registry.list_computers()
        assert len(computers) == 3
        assert "comp1" in computers
        assert "comp2" in computers
        assert "comp3" in computers


class TestComputation:
    """Test metric computation."""

    def test_compute_single_computer(self):
        """Test computing metrics with a single computer."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())

        context = {"value": 5}
        metrics = registry.compute("simple", context)

        assert metrics["doubled"] == 10
        assert metrics["squared"] == 25

    def test_compute_validates_context(self):
        """Test that compute validates required context keys."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())

        # Missing required key 'value'
        context = {}

        with pytest.raises(KeyError, match="Missing required context keys"):
            registry.compute("simple", context, validate_context=True)

    def test_compute_skips_validation_when_disabled(self):
        """Test that validation can be skipped."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())

        # Missing required key 'value'
        context = {}

        # Should not raise when validation disabled
        metrics = registry.compute("simple", context, validate_context=False)
        assert metrics["doubled"] == 0  # Default value used

    def test_compute_nonexistent_computer_raises(self):
        """Test that computing with non-existent computer raises error."""
        registry = MetricRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.compute("nonexistent", {})

    def test_compute_all_merges_results(self):
        """Test that compute_all merges results from all computers."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())
        registry.register("constant", NoRequirementsComputer())

        context = {"value": 3}
        all_metrics = registry.compute_all(context)

        # Should have metrics from both computers
        assert all_metrics["doubled"] == 6
        assert all_metrics["squared"] == 9
        assert all_metrics["constant"] == 42

    def test_compute_all_skip_on_error(self):
        """Test that compute_all can skip failing computers."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())
        registry.register("failing", FailingComputer())
        registry.register("constant", NoRequirementsComputer())

        context = {"value": 3}
        all_metrics = registry.compute_all(context, skip_on_error=True)

        # Should have metrics from working computers only
        assert "doubled" in all_metrics
        assert "constant" in all_metrics
        # No error raised despite FailingComputer

    def test_compute_all_raise_on_error(self):
        """Test that compute_all can raise on errors."""
        registry = MetricRegistry()
        registry.register("failing", FailingComputer())
        registry.register("simple", SimpleMetricComputer())

        context = {"value": 3}

        with pytest.raises(ValueError, match="Intentional failure"):
            registry.compute_all(context, skip_on_error=False)

    def test_compute_all_no_validation_by_default(self):
        """Test that compute_all doesn't validate context by default."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())

        # Missing required keys
        context = {}

        # Should not raise (validation disabled by default)
        metrics = registry.compute_all(context, validate_context=False)
        assert "doubled" in metrics

    def test_compute_all_with_validation(self):
        """Test compute_all with context validation enabled."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())

        # Missing required keys
        context = {}

        # Need skip_on_error=False to raise validation errors
        with pytest.raises(KeyError, match="Missing required context keys"):
            registry.compute_all(context, validate_context=True, skip_on_error=False)


class TestRegistryInfo:
    """Test registry information retrieval."""

    def test_get_registry_info(self):
        """Test getting registry information."""
        registry = MetricRegistry()
        registry.register("simple", SimpleMetricComputer())
        registry.register("constant", NoRequirementsComputer())

        info = registry.get_registry_info()

        assert info["count"] == 2
        assert len(info["computers"]) == 2

        # Check computer info structure
        comp_names = [c["name"] for c in info["computers"]]
        assert "simple" in comp_names
        assert "constant" in comp_names

        # Check that required keys are listed
        for comp_info in info["computers"]:
            if comp_info["name"] == "simple":
                assert "value" in comp_info["required_keys"]
            elif comp_info["name"] == "constant":
                assert comp_info["required_keys"] == []

    def test_get_registry_info_empty(self):
        """Test getting info for empty registry."""
        registry = MetricRegistry()

        info = registry.get_registry_info()

        assert info["count"] == 0
        assert info["computers"] == []


class TestBaseMetricComputer:
    """Test BaseMetricComputer convenience class."""

    def test_base_computer_get_name_default(self):
        """Test that get_name returns class name by default."""
        computer = NoRequirementsComputer()
        assert computer.get_name() == "NoRequirementsComputer"

    def test_base_computer_required_keys_default(self):
        """Test that get_required_context_keys returns empty by default."""
        computer = NoRequirementsComputer()
        # Override in class, but without override should be []
        assert isinstance(computer.get_required_context_keys(), list)

    def test_base_computer_compute_not_implemented(self):
        """Test that compute must be implemented by subclasses."""
        class IncompleteComputer(BaseMetricComputer):
            pass

        computer = IncompleteComputer()

        with pytest.raises(NotImplementedError):
            computer.compute({})


class TestOverlappingMetrics:
    """Test handling of overlapping metric names."""

    def test_later_computers_overwrite_earlier(self):
        """Test that later computers overwrite metrics from earlier ones."""
        class Computer1(BaseMetricComputer):
            def compute(self, context):
                return {"shared": 1, "unique1": 10}

        class Computer2(BaseMetricComputer):
            def compute(self, context):
                return {"shared": 2, "unique2": 20}

        registry = MetricRegistry()
        registry.register("comp1", Computer1())
        registry.register("comp2", Computer2())

        metrics = registry.compute_all({})

        # comp2's value should overwrite comp1's
        assert metrics["shared"] == 2
        assert metrics["unique1"] == 10
        assert metrics["unique2"] == 20


class TestContextValidation:
    """Test context validation logic."""

    def test_validate_context_with_all_keys(self):
        """Test validation passes when all keys present."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()

        context = {"value": 5, "extra": "data"}

        # Should not raise
        registry._validate_context(computer, context)

    def test_validate_context_missing_keys(self):
        """Test validation fails when keys missing."""
        registry = MetricRegistry()
        computer = SimpleMetricComputer()

        context = {"wrong_key": 5}

        with pytest.raises(KeyError, match="Missing required context keys"):
            registry._validate_context(computer, context)

    def test_validate_context_no_requirements(self):
        """Test validation passes for computer with no requirements."""
        registry = MetricRegistry()
        computer = NoRequirementsComputer()

        context = {}

        # Should not raise
        registry._validate_context(computer, context)
