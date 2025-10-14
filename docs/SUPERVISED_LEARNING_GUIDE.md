# Supervised Learning Guide

This guide explains how to use supervised learning to bootstrap AlphaZero networks from high-quality chess game databases.

## Overview

The supervised learning phase trains the AlphaZero neural network on human games before transitioning to self-play reinforcement learning. This provides:

1. **Better initialization**: Network learns reasonable moves from expert games
2. **Faster convergence**: Reduces training time compared to learning from scratch
3. **Playstyle specialization**: Trains cluster-specific policies (tactical vs positional)

## Prerequisites

### 1. Install Dependencies

All dependencies are managed with `uv`. Make sure you're in the project root:

```bash
cd /home/fra/Uni/Thesis/main_repo/FedRL
uv sync
```

### 2. Download Chess Game Databases

Download PGN databases from Lichess or other sources. Recommended:

```bash
# Create databases directory
mkdir -p databases

# Download Lichess standard rated games (example for January 2024)
# Visit: https://database.lichess.org/
# Download: lichess_db_standard_rated_2024-01.pgn.zst

# Place in databases directory
mv lichess_db_standard_rated_2024-01.pgn.zst databases/
```

**Supported formats:**
- `.pgn` - Uncompressed PGN
- `.pgn.gz` - Gzip compressed
- `.pgn.zst` - Zstandard compressed (recommended, best compression)

## Configuration

### Step 1: Generate Node Configurations

Use the `generate_node_configs.py` script to create supervised learning configurations:

```bash
uv run python chess-federated-learning/scripts/generate_node_configs.py \
  --topology chess-federated-learning/config/cluster_topology.yaml \
  --trainer-type supervised \
  --pgn-database ./databases/lichess_db_standard_rated_2024-01.pgn.zst \
  --output-dir chess-federated-learning/config/nodes
```

**Parameters:**
- `--topology`: Path to cluster topology file
- `--trainer-type`: Must be `supervised` for supervised learning
- `--pgn-database`: Path to PGN database file
- `--output-dir`: Where to save generated configs (default: `config/nodes`)
- `--server-host`: FL server hostname (default: `localhost`)
- `--server-port`: FL server port (default: `8765`)

### Step 2: Verify Generated Configurations

Check one of the generated configs to ensure it has the supervised section:

```bash
cat chess-federated-learning/config/nodes/agg_001.yaml
```

Expected output:

```yaml
node_id: agg_001
cluster_id: cluster_tactical
server_host: localhost
server_port: 8765
trainer_type: supervised
auto_reconnect: true
training:
  games_per_round: 100
  batch_size: 32
  learning_rate: 0.001
  exploration_factor: 1.0
  max_game_length: 200
  save_games: true
  playstyle: tactical
storage:
  enabled: true
  base_path: ./chess-federated-learning/storage
  save_models: true
  save_metrics: true
logging:
  level: INFO
  file: ./logs/agg_001.log
  format: text
supervised:
  pgn_database_path: ./databases/lichess_db_standard_rated_2024-01.pgn.zst
  min_rating: 2000
  skip_opening_moves: 10
  skip_endgame_pieces: 6
  sample_rate: 1.0
```

### Step 3: Customize Supervised Settings (Optional)

You can manually edit the config files to customize supervised learning parameters:

**`supervised` section parameters:**
- `pgn_database_path`: Path to PGN database
- `min_rating`: Minimum player rating for games (default: 2000)
- `skip_opening_moves`: Skip first N moves (default: 10, too formulaic)
- `skip_endgame_pieces`: Skip positions with < N pieces (default: 6)
- `sample_rate`: Extract every Nth position (1.0 = all, 0.5 = every other)

**`training` section parameters:**
- `games_per_round`: Number of games to extract per training round
- `batch_size`: Training batch size (default: 32, increase for GPUs)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `playstyle`: Filter games by playstyle (`tactical` or `positional`)

## Running Supervised Learning

### Option 1: Start All Nodes

Launch all configured nodes in parallel:

```bash
uv run python chess-federated-learning/scripts/start_all_nodes.py \
  --config-dir chess-federated-learning/config/nodes
```

This will:
1. Load all node configurations from the directory
2. Create supervised trainers for each node
3. Configure PGN database paths
4. Start all nodes in separate threads

### Option 2: Start Single Node

Test with a single node first:

```bash
uv run python chess-federated-learning/scripts/start_node.py \
  --config chess-federated-learning/config/nodes/agg_001.yaml
```

### Option 3: Start Specific Nodes

Launch only specific nodes:

```bash
uv run python chess-federated-learning/scripts/start_all_nodes.py \
  --config-dir chess-federated-learning/config/nodes \
  --nodes agg_001,agg_002,pos_001
```

## Training Pipeline

### What Happens During Training

Each training round, the supervised trainer:

1. **Extracts samples** from PGN database:
   - Filters games by playstyle (tactical/positional)
   - Filters games by rating (≥2000 ELO)
   - Skips opening moves (first 10 moves)
   - Skips endgame positions (<6 pieces)
   - Uses offset for diversity (different games each round)

2. **Encodes positions**:
   - Board → 119-plane tensor (96 pieces+history, 23 metadata)
   - Move → action index (0-4671, using 8×8×73 encoding)

3. **Trains neural network**:
   - Policy head: Cross-entropy loss for move prediction
   - Value head: MSE loss for game outcome prediction
   - Combined loss = policy_loss + value_loss

4. **Returns updated model**:
   - Serialized using PyTorchSerializer
   - Sent to aggregation server
   - Aggregated with other cluster nodes

### Monitoring Training

Check the logs to monitor progress:

```bash
tail -f chess-federated-learning/logs/agg_001.log
```

Expected log output:

```
11:30:15 | INFO     | Starting supervised training
11:30:15 | INFO     | Extracting samples from database (offset=0)
11:30:18 | SUCCESS  | Extracted 2500 training samples
11:30:18 | INFO     | Created dataloader with 78 batches
11:30:20 | INFO     | Batch 10/78: loss=2.3456, policy=1.8234, value=0.5222
11:30:22 | INFO     | Batch 20/78: loss=2.1234, policy=1.6512, value=0.4722
...
11:30:45 | SUCCESS  | Training complete: loss=1.8456
```

## Cluster-Specific Training

### Tactical Cluster

Tactical nodes train on:
- Sicilian Defense (B20-B99)
- King's Gambit, Italian Game (C30-C99)
- Ruy Lopez (C60-C99)
- Other sharp openings

Configuration:
```yaml
playstyle: tactical
supervised:
  pgn_database_path: ./databases/lichess_db_standard_rated_2024-01.pgn.zst
  min_rating: 2000
```

### Positional Cluster

Positional nodes train on:
- Queen's Gambit (D00-D69)
- Indian Defenses (E00-E99)
- English Opening (A10-A39)
- Other closed openings

Configuration:
```yaml
playstyle: positional
supervised:
  pgn_database_path: ./databases/lichess_db_standard_rated_2024-01.pgn.zst
  min_rating: 2000
```

## Federated Learning Aggregation

### Layer Aggregation Strategy

The AlphaZero network uses the correct naming convention for federated learning:

**SHARED layers** (aggregated across clusters):
- `input_conv.*` - Initial convolution
- `input_bn.*` - Initial batch normalization
- `residual.{i}.*` - All residual blocks

**CLUSTER-SPECIFIC layers** (kept separate):
- `policy_head.*` - Move prediction (playstyle-specific)
- `value_head.*` - Position evaluation (playstyle-specific)

This allows:
- Shared feature extraction across playstyles
- Specialized move selection per cluster
- Hierarchical aggregation (cluster → global)

## Troubleshooting

### Issue: "PGN database not found"

**Solution:** Verify the database path in your config:

```bash
ls -lh databases/
# Ensure the file exists and path is correct in config
```

### Issue: "No training samples extracted"

**Possible causes:**
1. **Rating filter too high**: Lower `min_rating` in config
2. **Wrong playstyle**: Check ECO codes in database match playstyle
3. **Database too small**: Use larger database or lower `games_per_round`

**Solution:** Check sample extraction:

```bash
uv run python chess-federated-learning/data/sample_extractor.py \
  databases/lichess_db_standard_rated_2024-01.pgn.zst
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in config:

```yaml
training:
  batch_size: 16  # Reduce from 32
```

Or use CPU:

```python
# In trainer_supervised.py, modify __init__:
self.device = torch.device("cpu")
```

### Issue: "Import errors"

**Solution:** Ensure you're using `uv run`:

```bash
# Wrong
python chess-federated-learning/scripts/start_node.py

# Correct
uv run python chess-federated-learning/scripts/start_node.py
```

## Performance Tips

### 1. Use Multiple Workers

For large databases, extract samples in parallel:

```python
# In trainer_supervised.py, modify DataLoader:
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # Use multiple workers
    pin_memory=True
)
```

### 2. Use GPU

Ensure PyTorch detects your GPU:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Optimize Batch Size

- **CPU**: batch_size = 32-64
- **GPU (8GB)**: batch_size = 128-256
- **GPU (16GB+)**: batch_size = 512+

### 4. Cache Samples

For repeated training on same games, cache extracted samples:

```python
# TODO: Implement sample caching in future version
```

## Next Steps

### 1. Monitor Training Progress

Track loss and metrics over multiple rounds:

```bash
# View aggregation server logs
tail -f chess-federated-learning/logs/server.log

# Check saved models
ls -lh storage/models/
```

### 2. Transition to Self-Play

Once supervised training converges (loss < 0.5):

1. Regenerate configs with `--trainer-type alphazero`
2. Use supervised model as initialization
3. Start self-play reinforcement learning

### 3. Evaluate Performance

Test the trained model:

```python
from client.trainer.trainer_supervised import SupervisedTrainer

# Load trained model
trainer = SupervisedTrainer(...)
metrics = await trainer.evaluate(model_state, num_games=100)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
print(f"Value loss: {metrics['value_loss']:.4f}")
```

## References

- **AlphaZero Paper**: [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- **Lichess Database**: https://database.lichess.org/
- **Move Encoding**: See `chess-federated-learning/client/trainer/models/move_encoding.md`
- **Data Pipeline**: See `chess-federated-learning/data/` modules
