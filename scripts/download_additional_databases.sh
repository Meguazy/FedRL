#!/bin/bash
# Download additional Lichess databases for more training diversity

echo "========================================"
echo "Lichess Database Downloader"
echo "========================================"
echo ""
echo "Your current single database is sufficient for 10,000+ rounds!"
echo "But if you want more variety, here are options:"
echo ""

DB_DIR="chess-federated-learning/data/databases"
mkdir -p "$DB_DIR"

echo "Available Lichess Databases (2024):"
echo "1. January 2024   - lichess_db_standard_rated_2024-01.pgn.zst (31GB)"
echo "2. February 2024  - lichess_db_standard_rated_2024-02.pgn.zst (28GB)"
echo "3. March 2024     - lichess_db_standard_rated_2024-03.pgn.zst (33GB)"
echo "4. April 2024     - lichess_db_standard_rated_2024-04.pgn.zst (31GB)"
echo "5. May 2024       - lichess_db_standard_rated_2024-05.pgn.zst (34GB)"
echo ""
echo "Each database contains ~80-100 million games"
echo "After filtering: ~4-5 million per playstyle"
echo ""

read -p "Download additional month? (1-5, or 'n' to skip): " choice

case $choice in
    1)
        MONTH="2024-01"
        ;;
    2)
        MONTH="2024-02"
        ;;
    3)
        MONTH="2024-03"
        ;;
    4)
        MONTH="2024-04"
        ;;
    5)
        MONTH="2024-05"
        ;;
    n|N)
        echo "Skipping download. Your current database is sufficient!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

FILE="lichess_db_standard_rated_${MONTH}.pgn.zst"
URL="https://database.lichess.org/standard/${FILE}"

echo ""
echo "Downloading $FILE..."
echo "This will take 30-60 minutes depending on your connection"
echo ""

cd "$DB_DIR"
wget "$URL"

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "To use multiple databases, update your node configs:"
echo ""
echo "# In chess-federated-learning/config/nodes/agg_001.yaml"
echo "supervised:"
echo "  pgn_database_path: .chess-federated-learning/databases/${FILE}"
echo ""
echo "Or preprocess all databases into one combined cache:"
echo "./scripts/preprocess_multiple_databases.sh"
echo ""
