# Storage Directory

This directory contains all experiment data for the federated learning chess project.

## Structure

- `experiments/` - Experiment configurations and metadata
- `metrics/` - Training metrics (JSONL format)
- `models/` - Model checkpoints (PyTorch .pt files)
- `chess_data/` - Chess-specific data (PGNs, playstyle analysis)
- `analysis/` - Jupyter notebooks and generated reports
- `cache/` - Temporary query cache
- `.metadata/` - Storage system metadata

## Usage

Data is organized by run_id:
```
storage/
├── metrics/{run_id}/events.jsonl.gz
├── models/{run_id}/cluster_tactical/round_0001.pt
└── chess_data/{run_id}/games/...
```

## Note

This directory is gitignored. Backup important experiments separately.
