"""
Unit tests for diversity metric computation plugins.

Tests DiversityMetricComputer and SharedLayerSyncMetric plugins.
"""

import pytest
import torch

from server.storage.plugins.diversity_metrics import (
    DiversityMetricComputer,
    SharedLayerSyncMetric,
)


class TestDiversityMetricComputer:
    """Test DiversityMetricComputer plugin."""

    def test_get_name(self):
        """Test computer name."""
        computer = DiversityMetricComputer()
        assert computer.get_name() == "Diversity Metrics Computer"

    def test_required_context_keys(self):
        """Test required context keys."""
        computer = DiversityMetricComputer()
        assert computer.get_required_context_keys() == ["models"]

    def test_compute_with_no_models(self):
        """Test computing with no models."""
        computer = DiversityMetricComputer()
        context = {"models": {}}

        metrics = computer.compute(context)

        assert metrics["inter_cluster_distance"] == 0.0
        assert metrics["diversity_score"] == 0.0
        assert metrics["layer_divergence"] == {}
        assert metrics["max_divergence_layer"] is None

    def test_compute_with_one_model(self):
        """Test computing with only one model."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {
                    "layer1": torch.tensor([1.0, 2.0, 3.0]),
                }
            }
        }

        metrics = computer.compute(context)

        assert metrics["inter_cluster_distance"] == 0.0
        assert metrics["diversity_score"] == 0.0
        assert metrics["layer_divergence"] == {}
        assert metrics["max_divergence_layer"] is None

    def test_compute_identical_models(self):
        """Test computing with identical models."""
        computer = DiversityMetricComputer()
        model = {
            "layer1": torch.tensor([1.0, 2.0, 3.0]),
            "layer2": torch.tensor([4.0, 5.0]),
        }
        context = {
            "models": {
                "cluster_0": model,
                "cluster_1": model,
            }
        }

        metrics = computer.compute(context)

        # Identical models should have 0 distance
        assert metrics["inter_cluster_distance"] == 0.0
        assert metrics["diversity_score"] == 0.0
        assert "layer1" in metrics["layer_divergence"]
        assert "layer2" in metrics["layer_divergence"]
        assert metrics["layer_divergence"]["layer1"] == 0.0
        assert metrics["layer_divergence"]["layer2"] == 0.0

    def test_compute_different_models(self):
        """Test computing with different models."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {
                    "layer1": torch.tensor([1.0, 2.0, 3.0]),
                    "layer2": torch.tensor([4.0, 5.0]),
                },
                "cluster_1": {
                    "layer1": torch.tensor([2.0, 3.0, 4.0]),
                    "layer2": torch.tensor([5.0, 6.0]),
                },
            }
        }

        metrics = computer.compute(context)

        # Different models should have non-zero distance
        assert metrics["inter_cluster_distance"] > 0.0
        assert metrics["diversity_score"] > 0.0
        assert "layer1" in metrics["layer_divergence"]
        assert "layer2" in metrics["layer_divergence"]
        assert metrics["layer_divergence"]["layer1"] > 0.0
        assert metrics["layer_divergence"]["layer2"] > 0.0

    def test_compute_three_models(self):
        """Test computing with three models (multiple pairwise comparisons)."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {"layer1": torch.tensor([1.0, 2.0])},
                "cluster_1": {"layer1": torch.tensor([2.0, 3.0])},
                "cluster_2": {"layer1": torch.tensor([3.0, 4.0])},
            }
        }

        metrics = computer.compute(context)

        # Should compute all pairwise distances (3 pairs)
        assert metrics["inter_cluster_distance"] > 0.0
        assert metrics["diversity_score"] > 0.0
        assert "layer1" in metrics["layer_divergence"]

    def test_diversity_score_normalization(self):
        """Test that diversity score is normalized to 0-1."""
        computer = DiversityMetricComputer()

        # Create models with very large distance
        context = {
            "models": {
                "cluster_0": {"layer1": torch.tensor([0.0])},
                "cluster_1": {"layer1": torch.tensor([1000.0])},
            }
        }

        metrics = computer.compute(context)

        # Diversity score should be capped at 1.0
        assert 0.0 <= metrics["diversity_score"] <= 1.0

    def test_max_divergence_layer_identification(self):
        """Test that layer with max divergence is correctly identified."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {
                    "layer1": torch.tensor([1.0]),
                    "layer2": torch.tensor([1.0]),
                },
                "cluster_1": {
                    "layer1": torch.tensor([1.1]),  # Small difference
                    "layer2": torch.tensor([10.0]),  # Large difference
                },
            }
        }

        metrics = computer.compute(context)

        # layer2 has larger divergence
        assert metrics["max_divergence_layer"] == "layer2"
        assert metrics["layer_divergence"]["layer2"] > metrics["layer_divergence"]["layer1"]

    def test_compute_with_non_tensor_values(self):
        """Test computing with non-tensor values (should convert)."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {
                    "layer1": [1.0, 2.0, 3.0],  # List, not tensor
                },
                "cluster_1": {
                    "layer1": [2.0, 3.0, 4.0],  # List, not tensor
                },
            }
        }

        metrics = computer.compute(context)

        # Should handle non-tensor values
        assert metrics["inter_cluster_distance"] > 0.0
        assert "layer1" in metrics["layer_divergence"]

    def test_compute_with_partial_overlap(self):
        """Test computing with models that have partial key overlap."""
        computer = DiversityMetricComputer()
        context = {
            "models": {
                "cluster_0": {
                    "layer1": torch.tensor([1.0, 2.0]),
                    "layer2": torch.tensor([3.0, 4.0]),
                },
                "cluster_1": {
                    "layer1": torch.tensor([2.0, 3.0]),
                    "layer3": torch.tensor([5.0, 6.0]),  # Different layer
                },
            }
        }

        metrics = computer.compute(context)

        # Should only compute distance for common layers
        assert "layer1" in metrics["layer_divergence"]
        assert "layer2" not in metrics["layer_divergence"]
        assert "layer3" not in metrics["layer_divergence"]


class TestSharedLayerSyncMetric:
    """Test SharedLayerSyncMetric plugin."""

    def test_get_name(self):
        """Test computer name."""
        computer = SharedLayerSyncMetric()
        assert computer.get_name() == "Shared Layer Sync Metrics"

    def test_required_context_keys(self):
        """Test required context keys."""
        computer = SharedLayerSyncMetric()
        required = computer.get_required_context_keys()
        assert "models" in required
        assert "shared_layer_patterns" in required

    def test_compute_with_no_models(self):
        """Test computing with no models."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {},
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        assert metrics["shared_layers_synchronized"] is True
        assert metrics["max_shared_layer_distance"] == 0.0
        assert metrics["avg_shared_layer_distance"] == 0.0

    def test_compute_with_one_model(self):
        """Test computing with only one model."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {"shared.layer1": torch.tensor([1.0])}
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        assert metrics["shared_layers_synchronized"] is True
        assert metrics["max_shared_layer_distance"] == 0.0
        assert metrics["avg_shared_layer_distance"] == 0.0

    def test_synchronized_shared_layers(self):
        """Test with perfectly synchronized shared layers."""
        computer = SharedLayerSyncMetric()
        shared_layer = torch.tensor([1.0, 2.0, 3.0])
        context = {
            "models": {
                "cluster_0": {
                    "shared.layer1": shared_layer,
                    "private.layer1": torch.tensor([100.0]),
                },
                "cluster_1": {
                    "shared.layer1": shared_layer,
                    "private.layer1": torch.tensor([200.0]),
                },
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        assert metrics["shared_layers_synchronized"] is True
        assert metrics["max_shared_layer_distance"] < 1e-6
        assert metrics["avg_shared_layer_distance"] < 1e-6

    def test_unsynchronized_shared_layers(self):
        """Test with unsynchronized shared layers."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {
                    "shared.layer1": torch.tensor([1.0, 2.0]),
                    "private.layer1": torch.tensor([100.0]),
                },
                "cluster_1": {
                    "shared.layer1": torch.tensor([2.0, 3.0]),  # Different!
                    "private.layer1": torch.tensor([200.0]),
                },
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        assert metrics["shared_layers_synchronized"] is False
        assert metrics["max_shared_layer_distance"] > 1e-6
        assert metrics["avg_shared_layer_distance"] > 0.0

    def test_pattern_matching(self):
        """Test that only layers matching patterns are checked."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {
                    "shared.layer1": torch.tensor([1.0]),
                    "private.layer1": torch.tensor([1.0]),
                },
                "cluster_1": {
                    "shared.layer1": torch.tensor([1.0]),
                    "private.layer1": torch.tensor([100.0]),  # Very different
                },
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        # Private layer difference should be ignored
        assert metrics["shared_layers_synchronized"] is True

    def test_multiple_patterns(self):
        """Test with multiple shared layer patterns."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {
                    "shared.layer1": torch.tensor([1.0]),
                    "policy.shared.layer1": torch.tensor([2.0]),
                    "private.layer1": torch.tensor([3.0]),
                },
                "cluster_1": {
                    "shared.layer1": torch.tensor([1.0]),
                    "policy.shared.layer1": torch.tensor([2.0]),
                    "private.layer1": torch.tensor([100.0]),
                },
            },
            "shared_layer_patterns": ["shared.*", ".*\\.shared.*"]
        }

        metrics = computer.compute(context)

        # Both shared patterns should be synchronized
        assert metrics["shared_layers_synchronized"] is True

    def test_no_shared_layers(self):
        """Test when no layers match the patterns."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {"private.layer1": torch.tensor([1.0])},
                "cluster_1": {"private.layer1": torch.tensor([100.0])},
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        # No shared layers = synchronized
        assert metrics["shared_layers_synchronized"] is True
        assert metrics["max_shared_layer_distance"] == 0.0
        assert metrics["avg_shared_layer_distance"] == 0.0

    def test_three_models_sync(self):
        """Test synchronization with three models."""
        computer = SharedLayerSyncMetric()
        shared_layer = torch.tensor([1.0, 2.0])
        context = {
            "models": {
                "cluster_0": {"shared.layer1": shared_layer},
                "cluster_1": {"shared.layer1": shared_layer},
                "cluster_2": {"shared.layer1": shared_layer},
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        # All three should be synchronized
        assert metrics["shared_layers_synchronized"] is True

    def test_sync_tolerance(self):
        """Test synchronization tolerance threshold."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {"shared.layer1": torch.tensor([1.0])},
                "cluster_1": {"shared.layer1": torch.tensor([1.0 + 1e-7])},  # Very small diff
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        # Within tolerance (< 1e-6)
        assert metrics["shared_layers_synchronized"] is True

    def test_with_non_tensor_values(self):
        """Test with non-tensor values (should convert)."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {"shared.layer1": [1.0, 2.0]},  # List
                "cluster_1": {"shared.layer1": [1.0, 2.0]},  # List
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        assert metrics["shared_layers_synchronized"] is True

    def test_max_vs_avg_distance(self):
        """Test that max and avg distances are computed correctly."""
        computer = SharedLayerSyncMetric()
        context = {
            "models": {
                "cluster_0": {
                    "shared.layer1": torch.tensor([0.0]),
                    "shared.layer2": torch.tensor([0.0]),
                },
                "cluster_1": {
                    "shared.layer1": torch.tensor([1.0]),
                    "shared.layer2": torch.tensor([2.0]),
                },
            },
            "shared_layer_patterns": ["shared.*"]
        }

        metrics = computer.compute(context)

        # Max should be 2.0, avg should be 1.5
        assert metrics["max_shared_layer_distance"] == 2.0
        assert metrics["avg_shared_layer_distance"] == 1.5
