# tests/integration/test_aggregation_pipeline.py
import pytest

@pytest.mark.asyncio
async def test_intra_cluster_aggregation():
    """Test that intra-cluster aggregation produces valid output."""
    from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model
    
    node_models = {
        'agg_001': generate_alphazero_model().state_dict(),
        'agg_002': generate_alphazero_model().state_dict(),
        'agg_003': generate_alphazero_model().state_dict(),
        'agg_004': generate_alphazero_model().state_dict()
    }
    # Create metrics
    node_metrics = {
        'agg_001': {'samples': 1000, 'loss': 0.5},
        'agg_002': {'samples': 1200, 'loss': 0.4},
        'agg_003': {'samples': 800, 'loss': 0.6},
        'agg_004': {'samples': 1000, 'loss': 0.5}
    }
    
    # Create aggregator
    aggregator = IntraClusterAggregator(
        framework='pytorch',
        weighting_strategy='samples'
    )
    
    # Get weights and aggregate
    weights = aggregator.get_aggregation_weights(node_metrics)
    aggregated_model, metrics = await aggregator.aggregate(
        node_models, weights, round_num=1
    )
    
    # Validate results
    assert aggregated_model is not None
    assert len(aggregated_model) > 0
    assert metrics.participant_count == 4
    assert metrics.total_samples == 4000
    assert metrics.aggregation_time > 0
    
    # Validate model structure matches input
    reference_keys = set(node_models['agg_001'].keys())
    aggregated_keys = set(aggregated_model.keys())
    assert reference_keys == aggregated_keys
    
@pytest.mark.asyncio
async def test_inter_cluster_aggregation():
    """Test that inter-cluster aggregation preserves cluster-specific layers."""
    
    from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model
    
    # Create cluster models with AlphaZero structure
    cluster_models = {
        'cluster_aggressive': generate_alphazero_model().state_dict(),
        'cluster_positional': generate_alphazero_model().state_dict()
    }
    
    # Store original cluster-specific layers for comparison
    # Use the actual key names from MockAlphaZeroModel
    original_policy_agg = cluster_models['cluster_aggressive']['policy_head.fc.weight']
    original_policy_pos = cluster_models['cluster_positional']['policy_head.fc.weight']
    original_value_agg = cluster_models['cluster_aggressive']['value_head.fc2.weight']
    original_value_pos = cluster_models['cluster_positional']['value_head.fc2.weight']
    
    # Create metrics
    cluster_metrics = {
        'cluster_aggressive': {'samples': 16000, 'loss': 0.3},
        'cluster_positional': {'samples': 16000, 'loss': 0.35}
    }
    
    # Create aggregator with clear layer separation
    aggregator = InterClusterAggregator(
        framework='pytorch',
        shared_layer_patterns=['input_conv.*', 'residual.0.*'],  # Share first residual block
        cluster_specific_patterns=['policy_head.*', 'value_head.*', 'residual.[1-9].*'],
        weighting_strategy='uniform'
    )
    
    # Get weights and aggregate
    weights = aggregator.get_aggregation_weights(cluster_metrics)
    updated_models, metrics = await aggregator.aggregate(
        cluster_models, weights, round_num=1
    )
    
    # Validate results
    assert len(updated_models) == 2
    assert 'cluster_aggressive' in updated_models
    assert 'cluster_positional' in updated_models
    
    # Critical test: Cluster-specific layers should be UNCHANGED
    assert updated_models['cluster_aggressive']['policy_head.fc.weight'] == original_policy_agg
    assert updated_models['cluster_positional']['policy_head.fc.weight'] == original_policy_pos
    assert updated_models['cluster_aggressive']['value_head.fc2.weight'] == original_value_agg
    assert updated_models['cluster_positional']['value_head.fc2.weight'] == original_value_pos
    
    # Critical test: Shared layers should be IDENTICAL across clusters
    shared_layer_agg = updated_models['cluster_aggressive']['input_conv.weight']
    shared_layer_pos = updated_models['cluster_positional']['input_conv.weight']
    assert shared_layer_agg == shared_layer_pos
    
    # Also check a residual layer
    shared_res_agg = updated_models['cluster_aggressive']['residual.0.conv1.weight']
    shared_res_pos = updated_models['cluster_positional']['residual.0.conv1.weight']
    assert shared_res_agg == shared_res_pos
    
    # Metrics validation
    assert metrics.participant_count == 2
    assert metrics.additional_metrics['shared_layer_count'] >= 1
    
@pytest.mark.asyncio
async def test_full_aggregation_pipeline_with_residual_0_shared():
    """Test complete aggregation workflow: nodes → clusters → selective sharing."""
    
    from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
    from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model
    from loguru import logger
        
    log = logger.bind(context="test_full_aggregation_pipeline")
    log.info("Testing full 3-tier aggregation pipeline")
    
    # Step 1: Generate node models (8 nodes total, 4 per cluster)
    # All nodes have identical architecture but different weights
    log.info("Step 1: Generating node models...")
    
    aggressive_nodes = {
        f'agg_{i:03d}': generate_alphazero_model(num_residual_blocks=10, seed=i).state_dict()
        for i in range(1, 5)
    }
    
    positional_nodes = {
        f'pos_{i:03d}': generate_alphazero_model(num_residual_blocks=10, seed=i+10).state_dict()
        for i in range(1, 5)
    }
    
    log.info(f"Generated {len(aggressive_nodes)} aggressive nodes")
    log.info(f"Generated {len(positional_nodes)} positional nodes")
    
    # Step 2: Intra-cluster aggregation (aggregate within each cluster)
    log.info("Step 2: Performing intra-cluster aggregation...")
    
    intra_agg = IntraClusterAggregator(
        framework='pytorch',
        weighting_strategy='samples'
    )
    
    # Create node metrics
    agg_metrics = {nid: {'samples': 1000, 'loss': 0.4} for nid in aggressive_nodes.keys()}
    pos_metrics = {nid: {'samples': 1000, 'loss': 0.35} for nid in positional_nodes.keys()}
    
    # Get weights and aggregate each cluster
    agg_weights = intra_agg.get_aggregation_weights(agg_metrics)
    pos_weights = intra_agg.get_aggregation_weights(pos_metrics)
    
    cluster_agg, metrics_agg = await intra_agg.aggregate(
        aggressive_nodes, agg_weights, round_num=1
    )
    cluster_pos, metrics_pos = await intra_agg.aggregate(
        positional_nodes, pos_weights, round_num=1
    )
    
    log.info(f"Aggressive cluster aggregation: {metrics_agg.participant_count} nodes, "
             f"{metrics_agg.total_samples} samples")
    log.info(f"Positional cluster aggregation: {metrics_pos.participant_count} nodes, "
             f"{metrics_pos.total_samples} samples")
    
    # Validate intra-cluster results
    assert cluster_agg is not None
    assert cluster_pos is not None
    assert metrics_agg.participant_count == 4
    assert metrics_pos.participant_count == 4
    assert metrics_agg.total_samples == 4000
    assert metrics_pos.total_samples == 4000
    
    # Store some cluster-specific layers before inter-cluster aggregation
    original_agg_policy = cluster_agg['policy_head.fc.weight']
    original_pos_policy = cluster_pos['policy_head.fc.weight']
    original_agg_value = cluster_agg['value_head.fc2.weight']
    original_pos_value = cluster_pos['value_head.fc2.weight']
    
    # Step 3: Inter-cluster selective aggregation (share only generic layers)
    log.info("Step 3: Performing inter-cluster selective aggregation...")
    
    inter_agg = InterClusterAggregator(
        framework='pytorch',
        shared_layer_patterns=[
            'input_conv.*',    # Share input encoding
            'residual.0.*'     # Share first residual block only
        ],
        cluster_specific_patterns=[
            'policy_head.*',   # Keep policy head cluster-specific
            'value_head.*',    # Keep value head cluster-specific
            'residual.[1-9].*' # Keep later residual blocks cluster-specific
        ],
        weighting_strategy='uniform'  # Equal weight for diversity preservation
    )
    
    cluster_models = {
        'cluster_aggressive': cluster_agg,
        'cluster_positional': cluster_pos
    }
    
    cluster_metrics = {
        'cluster_aggressive': {'samples': 4000, 'loss': 0.4},
        'cluster_positional': {'samples': 4000, 'loss': 0.35}
    }
    
    cluster_weights = inter_agg.get_aggregation_weights(cluster_metrics)
    final_models, metrics_inter = await inter_agg.aggregate(
        cluster_models, cluster_weights, round_num=1
    )
    
    log.info(f"Inter-cluster aggregation: {metrics_inter.participant_count} clusters")
    log.info(f"Shared {metrics_inter.additional_metrics['shared_layer_count']} layers")
    log.info(f"Preserved {metrics_inter.additional_metrics['cluster_specific_count']} "
             f"cluster-specific layers")
    
    # Validate complete pipeline results
    assert len(final_models) == 2
    assert 'cluster_aggressive' in final_models
    assert 'cluster_positional' in final_models
    
    # Each cluster should have complete model with all layers
    for cluster_id, model in final_models.items():
        assert 'input_conv.weight' in model
        assert 'residual.0.conv1.weight' in model
        assert 'residual.1.conv1.weight' in model
        assert 'residual.2.conv1.weight' in model
        assert 'residual.3.conv1.weight' in model
        assert 'residual.4.conv1.weight' in model
        assert 'residual.5.conv1.weight' in model
        assert 'residual.6.conv1.weight' in model
        assert 'residual.7.conv1.weight' in model
        assert 'residual.8.conv1.weight' in model
        assert 'residual.9.conv1.weight' in model
        assert 'policy_head.fc.weight' in model
        assert 'value_head.fc2.weight' in model
    
    # Critical validation: Cluster-specific layers should be UNCHANGED
    log.info("Validating cluster-specific layers are preserved...")
    assert final_models['cluster_aggressive']['policy_head.fc.weight'] == original_agg_policy
    assert final_models['cluster_positional']['policy_head.fc.weight'] == original_pos_policy
    assert final_models['cluster_aggressive']['value_head.fc2.weight'] == original_agg_value
    assert final_models['cluster_positional']['value_head.fc2.weight'] == original_pos_value
    
    # Critical validation: Shared layers should be IDENTICAL across clusters
    log.info("Validating shared layers are synchronized...")
    
    # Check input_conv is identical
    assert (final_models['cluster_aggressive']['input_conv.weight'] == 
            final_models['cluster_positional']['input_conv.weight'])
    assert (final_models['cluster_aggressive']['input_conv.bias'] == 
            final_models['cluster_positional']['input_conv.bias'])
    
    # Check residual.0 layers are identical
    assert (final_models['cluster_aggressive']['residual.0.conv1.weight'] == 
            final_models['cluster_positional']['residual.0.conv1.weight'])
    assert (final_models['cluster_aggressive']['residual.0.bn1.weight'] == 
            final_models['cluster_positional']['residual.0.bn1.weight'])
    
    # Critical validation: Non-shared residual layers should be DIFFERENT
    log.info("Validating non-shared layers remain different...")
    assert (final_models['cluster_aggressive']['residual.1.conv1.weight'] != 
            final_models['cluster_positional']['residual.1.conv1.weight'])
    assert (final_models['cluster_aggressive']['residual.2.conv1.weight'] != 
            final_models['cluster_positional']['residual.2.conv1.weight'])
    
    # Check that policy and value are different across clusters
    assert (final_models['cluster_aggressive']['policy_head.fc.weight'] != 
            final_models['cluster_positional']['policy_head.fc.weight'])
    assert (final_models['cluster_aggressive']['value_head.fc2.weight'] != 
            final_models['cluster_positional']['value_head.fc2.weight'])
    
    log.info("✓ Full aggregation pipeline test passed")
    log.info("✓ Diversity preservation verified")
    log.info("✓ Selective layer sharing verified")
    
@pytest.mark.asyncio
async def test_full_aggregation_pipeline_with_no_residual_shared():
    """Test complete aggregation workflow: nodes → clusters → selective sharing."""
    
    from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
    from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model
    from loguru import logger
        
    log = logger.bind(context="test_full_aggregation_pipeline")
    log.info("Testing full 3-tier aggregation pipeline")
    
    # Step 1: Generate node models (8 nodes total, 4 per cluster)
    # All nodes have identical architecture but different weights
    log.info("Step 1: Generating node models...")
    
    aggressive_nodes = {
        f'agg_{i:03d}': generate_alphazero_model(num_residual_blocks=10, seed=i).state_dict()
        for i in range(1, 5)
    }
    
    positional_nodes = {
        f'pos_{i:03d}': generate_alphazero_model(num_residual_blocks=10, seed=i+10).state_dict()
        for i in range(1, 5)
    }
    
    log.info(f"Generated {len(aggressive_nodes)} aggressive nodes")
    log.info(f"Generated {len(positional_nodes)} positional nodes")
    
    # Step 2: Intra-cluster aggregation (aggregate within each cluster)
    log.info("Step 2: Performing intra-cluster aggregation...")
    
    intra_agg = IntraClusterAggregator(
        framework='pytorch',
        weighting_strategy='samples'
    )
    
    # Create node metrics
    agg_metrics = {nid: {'samples': 1000, 'loss': 0.4} for nid in aggressive_nodes.keys()}
    pos_metrics = {nid: {'samples': 1000, 'loss': 0.35} for nid in positional_nodes.keys()}
    
    # Get weights and aggregate each cluster
    agg_weights = intra_agg.get_aggregation_weights(agg_metrics)
    pos_weights = intra_agg.get_aggregation_weights(pos_metrics)
    
    cluster_agg, metrics_agg = await intra_agg.aggregate(
        aggressive_nodes, agg_weights, round_num=1
    )
    cluster_pos, metrics_pos = await intra_agg.aggregate(
        positional_nodes, pos_weights, round_num=1
    )
    
    log.info(f"Aggressive cluster aggregation: {metrics_agg.participant_count} nodes, "
             f"{metrics_agg.total_samples} samples")
    log.info(f"Positional cluster aggregation: {metrics_pos.participant_count} nodes, "
             f"{metrics_pos.total_samples} samples")
    
    # Validate intra-cluster results
    assert cluster_agg is not None
    assert cluster_pos is not None
    assert metrics_agg.participant_count == 4
    assert metrics_pos.participant_count == 4
    assert metrics_agg.total_samples == 4000
    assert metrics_pos.total_samples == 4000
    
    # Store some cluster-specific layers before inter-cluster aggregation
    original_agg_policy = cluster_agg['policy_head.fc.weight']
    original_pos_policy = cluster_pos['policy_head.fc.weight']
    original_agg_value = cluster_agg['value_head.fc2.weight']
    original_pos_value = cluster_pos['value_head.fc2.weight']
    
    # Step 3: Inter-cluster selective aggregation (share only generic layers)
    log.info("Step 3: Performing inter-cluster selective aggregation...")
    
    inter_agg = InterClusterAggregator(
        framework='pytorch',
        shared_layer_patterns=[
            'input_conv.*',    # Share input encoding
        ],
        cluster_specific_patterns=[
            'policy_head.*',   # Keep policy head cluster-specific
            'value_head.*',    # Keep value head cluster-specific
            'residual.[0-9].*' # Keep later residual blocks cluster-specific
        ],
        weighting_strategy='uniform'  # Equal weight for diversity preservation
    )
    
    cluster_models = {
        'cluster_aggressive': cluster_agg,
        'cluster_positional': cluster_pos
    }
    
    cluster_metrics = {
        'cluster_aggressive': {'samples': 4000, 'loss': 0.4},
        'cluster_positional': {'samples': 4000, 'loss': 0.35}
    }
    
    cluster_weights = inter_agg.get_aggregation_weights(cluster_metrics)
    final_models, metrics_inter = await inter_agg.aggregate(
        cluster_models, cluster_weights, round_num=1
    )
    
    log.info(f"Inter-cluster aggregation: {metrics_inter.participant_count} clusters")
    log.info(f"Shared {metrics_inter.additional_metrics['shared_layer_count']} layers")
    log.info(f"Preserved {metrics_inter.additional_metrics['cluster_specific_count']} "
             f"cluster-specific layers")
    
    # Validate complete pipeline results
    assert len(final_models) == 2
    assert 'cluster_aggressive' in final_models
    assert 'cluster_positional' in final_models
    
    # Each cluster should have complete model with all layers
    for cluster_id, model in final_models.items():
        assert 'input_conv.weight' in model
        assert 'residual.0.conv1.weight' in model
        assert 'residual.1.conv1.weight' in model
        assert 'residual.2.conv1.weight' in model
        assert 'residual.3.conv1.weight' in model
        assert 'residual.4.conv1.weight' in model
        assert 'residual.5.conv1.weight' in model
        assert 'residual.6.conv1.weight' in model
        assert 'residual.7.conv1.weight' in model
        assert 'residual.8.conv1.weight' in model
        assert 'residual.9.conv1.weight' in model
        assert 'residual.10.conv1.weight' not in model
        assert 'policy_head.fc.weight' in model
        assert 'value_head.fc2.weight' in model
    
    # Critical validation: Cluster-specific layers should be UNCHANGED
    log.info("Validating cluster-specific layers are preserved...")
    assert final_models['cluster_aggressive']['policy_head.fc.weight'] == original_agg_policy
    assert final_models['cluster_positional']['policy_head.fc.weight'] == original_pos_policy
    assert final_models['cluster_aggressive']['value_head.fc2.weight'] == original_agg_value
    assert final_models['cluster_positional']['value_head.fc2.weight'] == original_pos_value
    
    # Critical validation: Shared layers should be IDENTICAL across clusters
    log.info("Validating shared layers are synchronized...")
    
    # Check input_conv is identical
    assert (final_models['cluster_aggressive']['input_conv.weight'] == 
            final_models['cluster_positional']['input_conv.weight'])
    assert (final_models['cluster_aggressive']['input_conv.bias'] == 
            final_models['cluster_positional']['input_conv.bias'])
    
    # Check residual.0 layers are not identical
    assert (final_models['cluster_aggressive']['residual.0.conv1.weight'] != 
            final_models['cluster_positional']['residual.0.conv1.weight'])
    assert (final_models['cluster_aggressive']['residual.0.bn1.weight'] != 
            final_models['cluster_positional']['residual.0.bn1.weight'])
    
    # Critical validation: Non-shared residual layers should be DIFFERENT
    log.info("Validating non-shared layers remain different...")
    assert (final_models['cluster_aggressive']['residual.1.conv1.weight'] != 
            final_models['cluster_positional']['residual.1.conv1.weight'])
    assert (final_models['cluster_aggressive']['residual.2.conv1.weight'] != 
            final_models['cluster_positional']['residual.2.conv1.weight'])
    
    # Check that policy and value are different across clusters
    assert (final_models['cluster_aggressive']['policy_head.fc.weight'] != 
            final_models['cluster_positional']['policy_head.fc.weight'])
    assert (final_models['cluster_aggressive']['value_head.fc2.weight'] != 
            final_models['cluster_positional']['value_head.fc2.weight'])
    
    log.info("✓ Full aggregation pipeline test passed")
    log.info("✓ Diversity preservation verified")
    log.info("✓ Selective layer sharing verified")
    
@pytest.mark.asyncio
async def test_expected_intra_cluster_aggregation():
    """Test intra-cluster aggregation produces expected averaged weights."""
    from server.aggregation.intra_cluster_aggregator import IntraClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model

    # Create two simple models with known weights for deterministic aggregation
    model1 = generate_alphazero_model(num_residual_blocks=2, seed=42)
    model2 = generate_alphazero_model(num_residual_blocks=2, seed=43)
    state1 = model1.state_dict()
    state2 = model2.state_dict()

    # Manually set a layer to known values for both models
    state1['input_conv.weight'] = [1.0, 2.0, 3.0]
    state2['input_conv.weight'] = [4.0, 5.0, 6.0]

    node_models = {
        'node1': state1,
        'node2': state2
    }
    node_metrics = {
        'node1': {'samples': 2, 'loss': 0.1},
        'node2': {'samples': 2, 'loss': 0.2}
    }

    aggregator = IntraClusterAggregator(
        framework='pytorch',
        weighting_strategy='samples'
    )
    weights = aggregator.get_aggregation_weights(node_metrics)
    aggregated_model, metrics = await aggregator.aggregate(node_models, weights, round_num=1)

    # The weights should be averaged equally (since samples are equal)
    expected = [(a + b) / 2 for a, b in zip(state1['input_conv.weight'], state2['input_conv.weight'])]
    assert aggregated_model['input_conv.weight'] == expected

    # Check metrics
    assert metrics.participant_count == 2
    assert metrics.total_samples == 4

    # Now test with different sample counts (weighted average)
    node_metrics = {
        'node1': {'samples': 1, 'loss': 0.1},
        'node2': {'samples': 3, 'loss': 0.2}
    }
    weights = aggregator.get_aggregation_weights(node_metrics)
    aggregated_model, metrics = await aggregator.aggregate(node_models, weights, round_num=2)

    # Weighted average: (1*[1,2,3] + 3*[4,5,6]) / 4 = ([1+12, 2+15, 3+18]/4) = [13/4, 17/4, 21/4]
    expected_weighted = [
        (1*1.0 + 3*4.0)/4,
        (1*2.0 + 3*5.0)/4,
        (1*3.0 + 3*6.0)/4
    ]
    assert aggregated_model['input_conv.weight'] == expected_weighted
    assert metrics.participant_count == 2
    assert metrics.total_samples == 4
    
pytest.mark.asyncio
async def test_expected_inter_cluster_aggregation():
    """Test inter-cluster aggregation with selective layer sharing."""
    from server.aggregation.inter_cluster_aggregator import InterClusterAggregator
    from tests.fixtures.model_utils import generate_alphazero_model

    # Create two cluster models with known weights for deterministic aggregation
    cluster1 = generate_alphazero_model(num_residual_blocks=2, seed=100)
    cluster2 = generate_alphazero_model(num_residual_blocks=2, seed=200)
    state1 = cluster1.state_dict()
    state2 = cluster2.state_dict()

    # Manually set a shared layer to known values for both clusters
    state1['input_conv.weight'] = [1.0, 2.0, 3.0]
    state2['input_conv.weight'] = [4.0, 5.0, 6.0]

    cluster_models = {
        'cluster1': state1,
        'cluster2': state2
    }
    cluster_metrics = {
        'cluster1': {'samples': 2, 'loss': 0.1},
        'cluster2': {'samples': 2, 'loss': 0.2}
    }

    aggregator = InterClusterAggregator(
        framework='pytorch',
        shared_layer_patterns=['input_conv.*'],
        cluster_specific_patterns=['policy_head.*', 'value_head.*', 'residual.[0-9].*'],
        weighting_strategy='uniform'
    )
    weights = aggregator.get_aggregation_weights(cluster_metrics)
    updated_models, metrics = await aggregator.aggregate(cluster_models, weights, round_num=1)

    # The shared layer should be averaged equally (since samples are equal)
    expected = [(a + b) / 2 for a, b in zip(state1['input_conv.weight'], state2['input_conv.weight'])]
    assert updated_models['cluster1']['input_conv.weight'] == expected
    assert updated_models['cluster2']['input_conv.weight'] == expected

    # Cluster-specific layers should remain unchanged
    assert updated_models['cluster1']['policy_head.fc.weight'] == state1['policy_head.fc.weight']
    assert updated_models['cluster2']['policy_head.fc.weight'] == state2['policy_head.fc.weight']

    # Check metrics
    assert metrics.participant_count == 2
    assert metrics.additional_metrics['shared_layer_count'] >= 1
    assert metrics.additional_metrics['cluster_specific_count'] >= 1
    
    # Now test with different sample counts (weighted average)
    # Create new aggregator with samples-based weighting
    aggregator_weighted = InterClusterAggregator(
        framework='pytorch',
        shared_layer_patterns=['input_conv.*'],
        cluster_specific_patterns=['policy_head.*', 'value_head.*', 'residual.[0-9].*'],
        weighting_strategy='samples'
    )

    cluster_metrics = {
        'cluster1': {'samples': 1, 'loss': 0.1},
        'cluster2': {'samples': 3, 'loss': 0.2}
    }
    weights = aggregator_weighted.get_aggregation_weights(cluster_metrics)
    updated_models, metrics = await aggregator_weighted.aggregate(cluster_models, weights, round_num=2)

    # Weighted average: (1*[1,2,3] + 3*[4,5,6]) / 4 = ([1+12, 2+15, 3+18]/4) = [13/4, 17/4, 21/4]
    expected_weighted = [
        (1*1.0 + 3*4.0)/4,
        (1*2.0 + 3*5.0)/4,
        (1*3.0 + 3*6.0)/4
    ]
    assert updated_models['cluster1']['input_conv.weight'] == expected_weighted
    assert updated_models['cluster2']['input_conv.weight'] == expected_weighted
    assert metrics.participant_count == 2
    assert metrics.additional_metrics['shared_layer_count'] >= 1
    assert metrics.additional_metrics['cluster_specific_count'] >= 1