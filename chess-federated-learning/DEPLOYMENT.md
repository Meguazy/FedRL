# Deployment Guide

This guide explains how to deploy and run the federated learning chess engine framework.

## Prerequisites

1. Python 3.8+ with required dependencies installed
2. Server configuration file: `config/server_config.yaml`
3. Cluster topology file: `config/cluster_topology.yaml`

## Quick Start

### 1. Generate Node Configurations

Generate configuration files for all nodes defined in the cluster topology:

```bash
# Generate configs from topology
python scripts/generate_node_configs.py --topology config/cluster_topology.yaml

# This creates config/nodes/agg_001.yaml, agg_002.yaml, ..., pos_001.yaml, etc.
```

**Options:**
- `--topology`: Path to cluster topology YAML (default: `config/cluster_topology.yaml`)
- `--output-dir`: Directory to save configs (default: `config/nodes`)
- `--server-host`: FL server hostname (default: `localhost`)
- `--server-port`: FL server port (default: `8765`)
- `--trainer-type`: Trainer type - `dummy`, `supervised`, or `alphazero` (default: `dummy`)

### 2. Start the Server

```bash
python chess-federated-learning/server/main.py \
    --config config/server_config.yaml \
    --cluster-config config/cluster_topology.yaml
```

The server will:
- Load cluster topology
- Start WebSocket server on port 8765
- Wait for nodes to register
- Run automated training rounds

### 3. Launch All Nodes

Launch all nodes in parallel using asyncio:

```bash
# Launch from topology (auto-generates configs)
python scripts/start_all_nodes.py --topology config/cluster_topology.yaml

# Or launch from pre-generated config directory
python scripts/start_all_nodes.py --config-dir config/nodes
```

**Common options:**
- `--limit N`: Launch only first N nodes (useful for testing)
- `--clusters cluster1,cluster2`: Launch only specific clusters
- `--delay 0.2`: Stagger startup delay in seconds (default: 0.1)
- `--verbose`: Enable debug logging

## Deployment Scenarios

### Testing with 4 Nodes

```bash
# Generate configs for 4 nodes (2 per cluster)
# First, edit config/cluster_topology.yaml:
#   node_count: 2  # for each cluster

python scripts/generate_node_configs.py
python scripts/start_all_nodes.py --topology config/cluster_topology.yaml --limit 4
```

### Production with 64 Nodes

```bash
# Edit config/cluster_topology.yaml:
#   node_count: 32  # for each cluster (64 total)

python scripts/generate_node_configs.py

# Launch all nodes
python scripts/start_all_nodes.py --topology config/cluster_topology.yaml
```

### Launching Specific Nodes

```bash
# Launch only specific nodes
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --nodes agg_001,agg_002,pos_001,pos_002
```

### Launching One Cluster

```bash
# Launch only aggressive cluster
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --clusters cluster_aggressive
```

## Architecture

### Node Execution Model

All nodes run as **parallel asyncio tasks** within a single Python process:

```
Main Process (start_all_nodes.py)
├── Node 1 (async task) ──> WebSocket ──> Server
├── Node 2 (async task) ──> WebSocket ──> Server
├── Node 3 (async task) ──> WebSocket ──> Server
└── Node N (async task) ──> WebSocket ──> Server
```

Benefits:
- **True parallelism**: All nodes run concurrently
- **Efficient**: Shared event loop, minimal overhead
- **Scalable**: Can run 64+ nodes on modern hardware
- **Graceful shutdown**: Signal handlers for clean termination

### Training Workflow

```
1. Server starts → loads topology → waits for nodes
2. Nodes register → connect to server
3. Server broadcasts START_TRAINING
4. Nodes train locally in parallel
5. Nodes send MODEL_UPDATE to server
6. Server aggregates (intra-cluster, then inter-cluster)
7. Server broadcasts CLUSTER_MODEL back to nodes
8. Repeat steps 3-7 for N rounds
```

## Configuration Files

### Cluster Topology (`config/cluster_topology.yaml`)

```yaml
clusters:
  - id: "cluster_aggressive"
    playstyle: "aggressive"
    node_count: 32
    node_prefix: "agg"

  - id: "cluster_positional"
    playstyle: "positional"
    node_count: 32
    node_prefix: "pos"
```

### Node Configuration (auto-generated)

```yaml
node_id: "agg_001"
cluster_id: "cluster_aggressive"
server_host: "localhost"
server_port: 8765
trainer_type: "dummy"
auto_reconnect: true

training:
  games_per_round: 100
  batch_size: 32
  learning_rate: 0.001
  playstyle: "aggressive"

storage:
  enabled: true
  base_path: "./storage"

logging:
  level: "INFO"
  file: "./logs/agg_001.log"
```

## Monitoring

### Server Logs

The server provides real-time status:
```
Connected nodes: 64/64
Round 1/100 progress: 47/64 nodes completed
Intra-cluster aggregation complete
Inter-cluster selective aggregation complete
Broadcasting updated models...
```

### Node Logs

Each node logs to console and file:
```
./logs/agg_001.log
./logs/agg_002.log
...
```

### Graceful Shutdown

Press `Ctrl+C` to gracefully shutdown all nodes:
```
Signal received, initiating shutdown...
Shutting down 64 nodes...
All nodes shut down
```

## Troubleshooting

### Issue: "No node configurations to launch"

**Solution**: Generate configs first or check topology file:
```bash
python scripts/generate_node_configs.py --topology config/cluster_topology.yaml
```

### Issue: Nodes can't connect to server

**Solution**: Verify server is running and check host/port:
```bash
# Check if server is listening
netstat -an | grep 8765

# Launch with explicit server address
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --server-host 192.168.1.100
```

### Issue: Too many nodes for available resources

**Solution**: Use `--limit` to launch fewer nodes:
```bash
python scripts/start_all_nodes.py --topology config/cluster_topology.yaml --limit 8
```

## Advanced Usage

### Custom Trainer Type

Generate configs with a different trainer:
```bash
python scripts/generate_node_configs.py \
    --trainer-type alphazero
```

### Remote Server Deployment

1. Start server on remote machine
2. Launch nodes with remote host:
```bash
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --server-host 192.168.1.100 \
    --server-port 8765
```

### Distributed Deployment

Launch subsets of nodes on different machines:

**Machine 1:**
```bash
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --clusters cluster_aggressive \
    --server-host <server-ip>
```

**Machine 2:**
```bash
python scripts/start_all_nodes.py \
    --topology config/cluster_topology.yaml \
    --clusters cluster_positional \
    --server-host <server-ip>
```

## Performance Considerations

- **Memory**: Each node requires ~500MB RAM (with model)
- **CPU**: Each node uses 1 thread (64 nodes = 64 threads)
- **Network**: WebSocket connections are lightweight
- **Startup**: Stagger delay prevents connection storms (0.1s × nodes)

### Recommended Hardware

| Nodes | CPU Cores | RAM | Storage |
|-------|-----------|-----|---------|
| 8     | 4+        | 8GB | 50GB    |
| 16    | 8+        | 16GB| 100GB   |
| 32    | 16+       | 32GB| 200GB   |
| 64    | 32+       | 64GB| 500GB   |

## Next Steps

After successful deployment:

1. Monitor training progress via server logs
2. Check experiment metrics in `storage/metrics/`
3. Analyze model checkpoints in `storage/models/`
4. Review game data in `storage/chess_data/`

For more details, see:
- [Implementation Report](fedrl_report_v4.md)
- [Demo README](DEMO_README.md)
