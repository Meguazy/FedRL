#!/bin/bash
# Preprocess database into fast cache files

echo "========================================"
echo "Database Cache Preprocessor"
echo "========================================"
echo ""
echo "This will preprocess your PGN database into cache files."
echo "First run: Takes ~30-60 minutes (one time)"
echo "Subsequent training: INSTANT sample access!"
echo ""

# Check if database exists
DB_PATH="chess-federated-learning/data/databases/lichess_db_standard_rated_2024-01.pgn.zst"
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    echo "Please download it first from https://database.lichess.org/"
    exit 1
fi

echo "Database found: $DB_PATH"
echo "Cache directory: chess-federated-learning/data/cache/"
echo ""
echo "Starting preprocessing..."
echo ""

# Run preprocessor
uv run python -m chess-federated-learning.data.database_preprocessor \
    --input "$DB_PATH" \
    --output chess-federated-learning/data/cache \
    --min-rating 2000 \
    --playstyle positional

echo ""
echo "========================================"
echo "Preprocessing complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start the server: uv run python chess-federated-learning/server/main.py"
echo "  2. Start the nodes: uv run python chess-federated-learning/scripts/start_all_nodes.py --config-dir chess-federated-learning/config/nodes"
echo ""
echo "Extraction will now be INSTANT (no more waiting!)"
echo ""
