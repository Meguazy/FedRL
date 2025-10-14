#!/bin/bash
#
# Preprocess tactical games from Lichess database into Redis cache.
#
# This script indexes games with tactical playstyle (aggressive openings, sharp positions)
# into Redis for fast loading during training.
#
# Usage:
#   ./scripts/preprocess_tactical_database.sh [database_file] [min_rating]
#
# Examples:
#   ./scripts/preprocess_tactical_database.sh                                # Use defaults
#   ./scripts/preprocess_tactical_database.sh lichess_db_2024-01.pgn.zst    # Specific database
#   ./scripts/preprocess_tactical_database.sh lichess_db_2024-01.pgn.zst 2200  # Custom rating
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_DATABASE="chess-federated-learning/data/databases/lichess_db_standard_rated_2024-01.pgn.zst"
DEFAULT_MIN_RATING=2000
PLAYSTYLE="tactical"

# Parse arguments
DATABASE="${1:-$DEFAULT_DATABASE}"
MIN_RATING="${2:-$DEFAULT_MIN_RATING}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}    TACTICAL GAMES PREPROCESSING${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "Database:    ${GREEN}${DATABASE}${NC}"
echo -e "Playstyle:   ${GREEN}${PLAYSTYLE}${NC}"
echo -e "Min Rating:  ${GREEN}${MIN_RATING}${NC}"
echo ""

# Check if database file exists
if [ ! -f "$DATABASE" ]; then
    echo -e "${RED}✗ Error: Database file not found: ${DATABASE}${NC}"
    echo ""
    echo "Available databases:"
    ls -lh chess-federated-learning/data/databases/*.pgn.zst 2>/dev/null || echo "  No databases found"
    echo ""
    exit 1
fi

# Check if Redis is running
echo -e "${YELLOW}Checking Redis connection...${NC}"
if docker compose ps redis 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis is not running${NC}"
    echo ""
    echo "Please start Redis first:"
    echo -e "  ${BLUE}docker compose up -d redis${NC}"
    echo ""
    exit 1
fi

# Run indexing script
echo ""
echo -e "${YELLOW}Starting indexing process...${NC}"
echo -e "${YELLOW}This may take several minutes for large databases.${NC}"
echo ""

uv run scripts/index_games_to_redis.py \
    --action index \
    --input "$DATABASE" \
    --playstyle "$PLAYSTYLE" \
    --min-rating "$MIN_RATING" \
    --redis-port 6381

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================${NC}"
    echo -e "${GREEN}    ✓ TACTICAL PREPROCESSING COMPLETE${NC}"
    echo -e "${GREEN}=================================================${NC}"
    echo ""
    
    # Show cache statistics
    echo -e "${BLUE}Cache Statistics:${NC}"
    uv run scripts/index_games_to_redis.py --action stats --redis-port 6381
    
else
    echo ""
    echo -e "${RED}=================================================${NC}"
    echo -e "${RED}    ✗ PREPROCESSING FAILED${NC}"
    echo -e "${RED}=================================================${NC}"
    echo ""
    exit $EXIT_CODE
fi
