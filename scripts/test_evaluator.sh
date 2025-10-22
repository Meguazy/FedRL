#!/bin/bash
# Quick test of the evaluator

echo "Testing Chess Model Evaluator"
echo "=============================="
echo

# Check if Stockfish is installed
if ! command -v stockfish &> /dev/null; then
    echo "❌ Stockfish not found. Install with: sudo apt install stockfish"
    exit 1
fi
echo "✓ Stockfish found: $(which stockfish)"

# Check if model exists
if [ ! -f "storage/models/cluster_tactical_round_5.pt" ]; then
    echo "⚠ No trained model found at storage/models/cluster_tactical_round_5.pt"
    echo "  Please train your model first or specify a different model path"
    exit 1
fi
echo "✓ Model file found"

# Check python-chess
echo -n "✓ Checking python-chess... "
uv run python -c "import chess; print('OK')" 2>&1 | grep -q "OK" && echo "✓" || echo "❌ Failed"

echo
echo "Running quick test (2 games)..."
echo

cd chess-federated-learning
uv run python evaluator.py \
    --model ../storage/models/cluster_tactical_round_5.pt \
    --elo 1000 \
    --games 2 \
    --quiet

echo
echo "✓ Test complete! Use --games 20 --auto for full evaluation"
