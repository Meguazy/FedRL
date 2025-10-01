"""
Unit tests for system performance metric computation plugins.

Tests SystemMetricComputer and AggregationTimingMetric plugins.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from server.storage.plugins.system_metrics import (
    AggregationTimingMetric,
    SystemMetricComputer,
)


class TestSystemMetricComputer:
    """Test SystemMetricComputer plugin."""

    def test_get_name(self):
        """Test computer name."""
        computer = SystemMetricComputer()
        assert computer.get_name() == "System Performance Metrics"

    def test_required_context_keys(self):
        """Test required context keys (none required)."""
        computer = SystemMetricComputer()
        assert computer.get_required_context_keys() == []

    def test_compute_with_psutil(self):
        """Test computing metrics when psutil is available."""
        # Create a mock psutil module
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 500  # 500 MB
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory

        # Patch import to return mock
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()
            metrics = computer.compute({})

            assert "cpu_usage_percent" in metrics
            assert "memory_usage_mb" in metrics
            assert "memory_usage_percent" in metrics
            assert "compute_interval_seconds" in metrics
            assert "compute_count" in metrics

            assert metrics["cpu_usage_percent"] == 45.5
            assert metrics["memory_usage_mb"] == 500.0
            assert metrics["memory_usage_percent"] == 60.0
            assert metrics["compute_count"] == 1

    def test_compute_without_psutil(self):
        """Test computing when psutil is not installed."""
        # Create a mock that raises ImportError when accessed
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            computer = SystemMetricComputer()
            metrics = computer.compute({})

            # Should return empty dict when psutil unavailable
            assert metrics == {}

    def test_compute_interval_tracking(self):
        """Test that compute interval is tracked correctly."""
        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 100
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()

            # First compute
            metrics1 = computer.compute({})
            interval1 = metrics1["compute_interval_seconds"]

            # Should have non-zero interval (time since __init__)
            assert interval1 > 0

            # Wait a bit
            time.sleep(0.05)

            # Second compute
            metrics2 = computer.compute({})
            interval2 = metrics2["compute_interval_seconds"]

            # Second interval should be roughly 0.05 seconds
            assert 0.04 < interval2 < 0.1

    def test_compute_count_increments(self):
        """Test that compute count increments correctly."""
        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 100
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()

            # Multiple computes
            for i in range(1, 6):
                metrics = computer.compute({})
                assert metrics["compute_count"] == i

    def test_metric_types(self):
        """Test that all metrics are floats."""
        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 100
        mock_memory.percent = 50
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()
            metrics = computer.compute({})

            # All numeric metrics should be floats
            assert isinstance(metrics["cpu_usage_percent"], float)
            assert isinstance(metrics["memory_usage_mb"], float)
            assert isinstance(metrics["memory_usage_percent"], float)
            assert isinstance(metrics["compute_interval_seconds"], float)

            # Count is int
            assert isinstance(metrics["compute_count"], int)

    def test_memory_conversion_to_mb(self):
        """Test that memory is correctly converted to MB."""
        # Mock psutil with known values
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 0.0
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 1024  # 1 GB = 1024 MB
        mock_memory.percent = 0.0
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()
            metrics = computer.compute({})

            assert metrics["memory_usage_mb"] == 1024.0

    def test_cpu_interval_parameter(self):
        """Test that CPU measurement uses correct interval."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = MagicMock()
        mock_memory.used = 0
        mock_memory.percent = 0.0
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            computer = SystemMetricComputer()
            computer.compute({})

            # Verify cpu_percent was called with 0.1 second interval
            mock_psutil.cpu_percent.assert_called_with(interval=0.1)


class TestAggregationTimingMetric:
    """Test AggregationTimingMetric plugin."""

    def test_get_name(self):
        """Test computer name."""
        computer = AggregationTimingMetric()
        assert computer.get_name() == "Aggregation Timing Metrics"

    def test_required_context_keys(self):
        """Test required context keys (none required)."""
        computer = AggregationTimingMetric()
        assert computer.get_required_context_keys() == []

    def test_compute_with_full_context(self):
        """Test computing with complete context."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "aggregation_end_time": 105.0,
            "node_count": 10
        }

        metrics = computer.compute(context)

        assert metrics["aggregation_duration_seconds"] == 5.0
        assert metrics["nodes_per_second"] == 2.0  # 10 nodes / 5 seconds
        assert metrics["node_count"] == 10

    def test_compute_with_missing_start_time(self):
        """Test computing when start_time is missing."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_end_time": 105.0,
            "node_count": 10
        }

        metrics = computer.compute(context)

        # Should return empty dict when required timing data missing
        assert metrics == {}

    def test_compute_with_missing_end_time(self):
        """Test computing when end_time is missing."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "node_count": 10
        }

        metrics = computer.compute(context)

        assert metrics == {}

    def test_compute_with_zero_duration(self):
        """Test computing when duration is zero."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "aggregation_end_time": 100.0,
            "node_count": 10
        }

        metrics = computer.compute(context)

        assert metrics["aggregation_duration_seconds"] == 0.0
        assert metrics["nodes_per_second"] == 0.0  # Avoid division by zero

    def test_compute_with_zero_nodes(self):
        """Test computing with zero nodes."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "aggregation_end_time": 105.0,
            "node_count": 0
        }

        metrics = computer.compute(context)

        assert metrics["aggregation_duration_seconds"] == 5.0
        assert metrics["nodes_per_second"] == 0.0
        assert metrics["node_count"] == 0

    def test_compute_without_node_count(self):
        """Test computing when node_count is not provided."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "aggregation_end_time": 105.0
        }

        metrics = computer.compute(context)

        # Should default to 0 nodes
        assert metrics["node_count"] == 0
        assert metrics["nodes_per_second"] == 0.0

    def test_throughput_calculation(self):
        """Test various throughput calculations."""
        computer = AggregationTimingMetric()

        # Test case 1: 100 nodes in 10 seconds = 10 nodes/sec
        context1 = {
            "aggregation_start_time": 0.0,
            "aggregation_end_time": 10.0,
            "node_count": 100
        }
        metrics1 = computer.compute(context1)
        assert metrics1["nodes_per_second"] == 10.0

        # Test case 2: 50 nodes in 5 seconds = 10 nodes/sec
        context2 = {
            "aggregation_start_time": 0.0,
            "aggregation_end_time": 5.0,
            "node_count": 50
        }
        metrics2 = computer.compute(context2)
        assert metrics2["nodes_per_second"] == 10.0

        # Test case 3: 1 node in 0.5 seconds = 2 nodes/sec
        context3 = {
            "aggregation_start_time": 0.0,
            "aggregation_end_time": 0.5,
            "node_count": 1
        }
        metrics3 = computer.compute(context3)
        assert metrics3["nodes_per_second"] == 2.0

    def test_metric_types(self):
        """Test that all metrics are correct types."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 100.0,
            "aggregation_end_time": 105.0,
            "node_count": 10
        }

        metrics = computer.compute(context)

        assert isinstance(metrics["aggregation_duration_seconds"], float)
        assert isinstance(metrics["nodes_per_second"], float)
        assert isinstance(metrics["node_count"], int)

    def test_negative_duration(self):
        """Test handling of negative duration (end before start)."""
        computer = AggregationTimingMetric()
        context = {
            "aggregation_start_time": 105.0,
            "aggregation_end_time": 100.0,  # Before start
            "node_count": 10
        }

        metrics = computer.compute(context)

        # Duration will be negative
        assert metrics["aggregation_duration_seconds"] == -5.0
        # Throughput is 0 when duration <= 0 (guard in code)
        assert metrics["nodes_per_second"] == 0.0

    def test_empty_context(self):
        """Test computing with empty context."""
        computer = AggregationTimingMetric()
        metrics = computer.compute({})

        assert metrics == {}

    def test_partial_timing_data(self):
        """Test with only one timestamp."""
        computer = AggregationTimingMetric()

        # Only start time
        context1 = {"aggregation_start_time": 100.0}
        assert computer.compute(context1) == {}

        # Only end time
        context2 = {"aggregation_end_time": 100.0}
        assert computer.compute(context2) == {}
