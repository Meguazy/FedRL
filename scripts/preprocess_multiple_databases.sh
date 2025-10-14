#!/bin/bash
# Preprocess multiple databases into one combined cache

echo "========================================"
echo "Multi-Database Cache Preprocessor"
echo "========================================"
echo ""

DB_DIR="chess-federated-learning/data/databases"
CACHE_DIR="chess-federated-learning/data/cache"

# Find all .zst databases
DATABASES=$(find "$DB_DIR" -name "lichess_db_standard_rated_*.pgn.zst" | sort)
DB_COUNT=$(echo "$DATABASES" | wc -l)

if [ $DB_COUNT -eq 0 ]; then
    echo "ERROR: No databases found in $DB_DIR"
    exit 1
fi

echo "Found $DB_COUNT database(s):"
echo "$DATABASES"
echo ""
echo "This will combine all databases into one cache."
echo "Estimated time: $((DB_COUNT * 45)) minutes"
echo ""
read -p "Continue? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborted"
    exit 0
fi

echo ""
echo "Processing databases..."
echo ""

# Process each database
for DB_PATH in $DATABASES; do
    DB_NAME=$(basename "$DB_PATH")
    echo "========================================"
    echo "Processing: $DB_NAME"
    echo "========================================"
    
    uv run python -m chess-federated-learning.data.database_preprocessor \
        --input "$DB_PATH" \
        --output "$CACHE_DIR" \
        --min-rating 2000 \
        --playstyle both \
        --append
    
    echo ""
done

echo "========================================"
echo "All databases processed!"
echo "========================================"
echo ""
echo "Cache directory: $CACHE_DIR"
echo "Total samples available:"
ls -lh "$CACHE_DIR"/*.pkl
echo ""
echo "Training can now use samples from all databases!"
echo ""
