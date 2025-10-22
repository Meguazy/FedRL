# Resume Training Guide

This guide explains how to resume training from a previous checkpoint **with proper data offset** to ensure you don't retrain on the same data.

## Quick Start

To resume training from round 30 with your existing models:

### 1. Configure Resume Training

Edit `chess-federated-learning/config/cluster_topology.yaml`:

```yaml
# Enable resume training and set the starting round
resume_training:
  enabled: true                     # Enable resume training
  starting_round: 30                # Round number to resume from

clusters:
  - id: "cluster_tactical"
    playstyle: "tactical"
    node_count: 4
    node_prefix: "agg"
    initial_model: "../storage/models/run_20251014_170903_e247af7e/cluster_tactical/round_0030.pt"

  - id: "cluster_positional"
    playstyle: "positional"
    node_count: 4
    node_prefix: "pos"
    initial_model: "../storage/models/run_20251014_170903_e247af7e/cluster_positional/round_0030.pt"
```

**IMPORTANT**: Setting `starting_round: 30` ensures the data sampling offset is correct. Without this, you would retrain on rounds 1-30 data again!

### 2. Update Games Per Round (Optional)

Edit `chess-federated-learning/config/server_config.yaml`:

```yaml
orchestrator_config:
  games_per_round: 200  # Increased from 100 to 200
  aggregation_threshold: 0.8
  timeout_seconds: 1200
```

### 3. Start Training

The server will automatically:
1. Load the checkpoint files you specified
2. Broadcast them to all nodes in each cluster
3. **Offset data sampling** so you use NEW data (games 12001+ instead of reusing games 1-12000)
4. Start training from round 1 with the loaded models and correct data offset

```bash
# Start the server
cd chess-federated-learning
uv run python server/main.py

# Start your clients as normal
# They will receive the initial models and offset before training begins
```

## How It Works

### Data Sampling Offset

This is the **critical feature** that makes resume training work correctly:

**Without offset (WRONG)**:
- Previous training: Rounds 1-30 used games 0-12000
- Resume training: Rounds 1-50 would use games 0-20000
- **Problem**: Rounds 1-30 of new training reuse games 0-12000 (duplicate training!)

**With offset (CORRECT)**:
- Previous training: Rounds 1-30 used games 0-12000
- Resume training: `starting_round: 30` sets offset to 30
- New round 1 uses data from position (1+30)*400 = 12400
- New round 2 uses data from position (2+30)*400 = 12800
- **Result**: All new training uses fresh, unseen games!

### Server Side

1. **Load Config**: Server reads `resume_training.starting_round` from cluster topology
2. **Load Models**: Server loads the `initial_model` checkpoint for each cluster
3. **Broadcast Models**: Before round 1, broadcasts initial models to all nodes (round 0 message)
4. **Send Offset**: Each START_TRAINING message includes `round_offset` in payload
5. **Continue Training**: Training proceeds from round 1, but data sampling is offset

### Client Side

1. **Receive Model**: Client receives initial model as `CLUSTER_MODEL` message (round 0)
2. **Receive Offset**: Client receives `round_offset` in START_TRAINING message
3. **Offset Calculation**: When sampling data, uses `effective_round = current_round + round_offset`
4. **Train on New Data**: Extracts games from the offset position, ensuring no duplication

## Configuration Details

### Cluster Topology Config

The `resume_training` section controls the data offset:

```yaml
resume_training:
  enabled: true                     # Must be true to enable offset
  starting_round: 30                # The round number you're resuming from
```

Each cluster must have an `initial_model` path:

```yaml
clusters:
  - id: "cluster_tactical"
    playstyle: "tactical"
    node_count: 4
    node_prefix: "agg"
    initial_model: "../storage/models/run_xyz/cluster_tactical/round_0030.pt"
```

**Notes:**
- `starting_round` MUST match the round number in your checkpoint filenames
- If `initial_model` is not set or file doesn't exist, that cluster starts with random initialization (but still uses the offset!)
- Paths are relative to the server working directory
- Each cluster can resume from different checkpoints, but all use the same `starting_round` offset

### Checkpoint File Format

The server supports two checkpoint formats:

1. **Model state dict only** (raw PyTorch state dict):
   ```python
   torch.save(model.state_dict(), "model.pt")
   ```

2. **Full checkpoint** (with additional metadata):
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'round': 30,
       'metrics': {...}
   }, "model.pt")
   ```

The server automatically detects the format and extracts the model state.

## Example: Your Current Setup

Based on your training:
- Run ID: `run_20251014_170903_e247af7e`
- Completed: 30 rounds with 100 games per round
- Total games used: 30 * 4 nodes * 100 games = 12,000 games
- Data positions used: 0 to 11,999

To resume for 50 more rounds with 200 games per round:

```yaml
# cluster_topology.yaml
resume_training:
  enabled: true
  starting_round: 30  # Critical! This ensures offset to position 12,000+

clusters:
  - id: "cluster_tactical"
    playstyle: "tactical"
    node_count: 4
    node_prefix: "agg"
    initial_model: "../storage/models/run_20251014_170903_e247af7e/cluster_tactical/round_0030.pt"

  - id: "cluster_positional"
    playstyle: "positional"
    node_count: 4
    node_prefix: "pos"
    initial_model: "../storage/models/run_20251014_170903_e247af7e/cluster_positional/round_0030.pt"
```

```yaml
# server_config.yaml
orchestrator_config:
  games_per_round: 200  # Increased from 100
```

**Data Usage Verification**:
- Previous training (rounds 1-30, 100 games): Used positions 0-11,999
- New training (rounds 1-50, 200 games):
  - Round 1: Starts at (1+30)*4*200 = position 24,800
  - Round 50: Starts at (50+30)*4*200 = position 64,000
  - **No overlap!** ✓

## Troubleshooting

### Model File Not Found

```
ERROR: Initial model file not found: ../storage/models/.../round_0030.pt
```

**Fix:** Check the file path is correct relative to where you run the server. Use absolute paths if needed:
```yaml
initial_model: "/home/fra/Uni/Thesis/main_repo/FedRL/storage/models/run_xyz/cluster_tactical/round_0030.pt"
```

### Model Load Error

```
ERROR: Failed to load initial model for cluster_tactical: ...
```

**Fix:**
- Verify the checkpoint file is valid (try loading it manually with `torch.load()`)
- Check that the model architecture matches
- If the error persists, the cluster will fall back to random initialization

### Wrong Model Architecture

If you've changed the model architecture since saving the checkpoint:
- The load will fail
- The cluster will start with random initialization
- You'll see a warning in the logs

## Advanced: Different Checkpoints Per Cluster

You can resume each cluster from different rounds:

```yaml
clusters:
  - id: "cluster_tactical"
    initial_model: "../storage/models/run_xyz/cluster_tactical/round_0030.pt"

  - id: "cluster_positional"
    initial_model: "../storage/models/run_xyz/cluster_positional/round_0025.pt"  # Different round!
```

This is useful for:
- Testing different starting points
- Replacing a poorly-performing cluster model
- Experimenting with mixed training stages

## Best Practices

1. **Backup Before Resuming**: Keep a copy of your checkpoint files
2. **Increase Games Gradually**: Don't jump from 100 to 1000 games per round
3. **Monitor First Round**: Check that models load successfully in server logs
4. **Evaluate Periodically**: Use the evaluator to track ELO progress every 10 rounds
5. **Save Checkpoints Regularly**: Keep checkpoints from multiple rounds in case you need to revert

## What's Next?

After resuming training for 50+ more rounds with 200 games per round:

```bash
# Evaluate your improved model
cd chess-federated-learning
uv run python evaluator.py \
    --model ../storage/models/<new_run_id>/cluster_tactical/round_0050.pt \
    --auto \
    --games 20 \
    --output eval_round80.json

# Compare with your round 30 model
cat results.json | grep estimated_elo  # Round 30
cat eval_round80.json | grep estimated_elo  # Round 80 (30 + 50)
```

Expected improvement with 50 more rounds and 2x games per round:
- **Current (Round 30, 100 games/round)**: ~800 ELO (loses to ELO 800)
- **After 50 more (Round 80, 200 games/round)**: ~1000-1200 ELO (competitive with novice players)
- **After 100 more rounds**: ~1200-1400 ELO (intermediate player)

Good luck with your training! 🎯
