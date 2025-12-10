#!/bin/bash
# Launch script for P1: Share Early Layers Experiment
# Shares input and early residual blocks (0-5) while keeping mid/late layers independent

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}P1: Share Early Layers Experiment${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Configuration
SERVER_CONFIG="chess-federated-learning/config/server_config_P1.yaml"
CLUSTER_TOPOLOGY="chess-federated-learning/config/cluster_topology.yaml"
EXPERIMENT_NAME="P1_share_early"
NUM_ROUNDS=500

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Project root: ${PROJECT_ROOT}${NC}"
echo -e "${GREEN}Server config: ${SERVER_CONFIG}${NC}"
echo -e "${GREEN}Cluster topology: ${CLUSTER_TOPOLOGY}${NC}"
echo -e "${GREEN}Experiment name: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}Number of rounds: ${NUM_ROUNDS}${NC}"
echo -e "${GREEN}Games per node per round: 400${NC}"
echo -e "${GREEN}Total games per round: 3200 (1600 tactical + 1600 positional)${NC}"
echo -e "${YELLOW}Shared layers: Input + early res_blocks (0-5)${NC}"
echo -e "${YELLOW}Cluster-specific: Mid/late res_blocks (6-18) + heads${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv first:${NC}"
    echo -e "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${BLUE}Using uv for Python environment...${NC}"

# Check if Redis is running
echo -e "${BLUE}Checking Redis status...${NC}"
if ! docker ps | grep -q chess-redis-cache; then
    echo -e "${YELLOW}Redis is not running. Starting Redis...${NC}"
    docker-compose up -d redis
    echo -e "${GREEN}Waiting for Redis to be ready...${NC}"
    sleep 3
else
    echo -e "${GREEN}Redis is already running${NC}"
fi

# Create required directories with proper permissions
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p logs

# Fix storage directory permissions if needed
if [ ! -w "storage" ]; then
    echo -e "${YELLOW}Storage directory requires sudo permissions. Fixing...${NC}"
    sudo chown -R $USER:$USER storage
    sudo chmod -R 755 storage
fi

# Create storage subdirectories
mkdir -p storage/metrics storage/models storage/checkpoints storage/experiments

# Kill any existing processes
echo -e "${BLUE}Checking for existing processes...${NC}"
if pgrep -f 'chess-federated-learning' > /dev/null; then
    echo -e "${YELLOW}Killing existing chess-federated-learning processes...${NC}"
    pkill -f 'chess-federated-learning'
    sleep 2
fi

# Check if port 8765 is in use and kill the process
if lsof -ti:8765 > /dev/null 2>&1; then
    echo -e "${YELLOW}Port 8765 is in use, killing process...${NC}"
    lsof -ti:8765 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo -e "${GREEN}Ready to start fresh processes${NC}"

# Step 1: Start the server in the background
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 1: Starting Federated Learning Server${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Copy P1 config to server_config.yaml temporarily
cp "$SERVER_CONFIG" chess-federated-learning/config/server_config.yaml.backup 2>/dev/null || true
cp "$SERVER_CONFIG" chess-federated-learning/config/server_config.yaml

# Start server with automated training using uv
uv run chess-federated-learning/server/run_training.py \
    --rounds $NUM_ROUNDS \
    --experiment "$EXPERIMENT_NAME" \
    --server-config "$SERVER_CONFIG" \
    --cluster-config "$CLUSTER_TOPOLOGY" \
    > logs/server_P1.log 2>&1 &
SERVER_PID=$!

echo -e "${GREEN}Server started with PID: ${SERVER_PID}${NC}"
echo -e "${YELLOW}Server logs: logs/server_P1.log${NC}"
echo ""

# Wait for server to start
echo -e "${BLUE}Waiting for server to initialize (10 seconds)...${NC}"
sleep 10

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: Server failed to start. Check logs/server_P1.log${NC}"
    exit 1
fi

echo -e "${GREEN}Server is running${NC}"
echo ""

# Step 2: Start all client nodes
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Step 2: Starting All Client Nodes (8 total)${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Start all nodes from topology using uv
uv run chess-federated-learning/scripts/start_all_nodes.py \
    --topology "$CLUSTER_TOPOLOGY" \
    --server-host localhost \
    --server-port 8765 \
    --delay 0.5 \
    > logs/nodes_P1.log 2>&1 &
NODES_PID=$!

echo -e "${GREEN}Nodes started with PID: ${NODES_PID}${NC}"
echo -e "${YELLOW}Nodes logs: logs/nodes_P1.log${NC}"
echo ""

# Wait for nodes to connect
echo -e "${BLUE}Waiting for nodes to connect (20 seconds)...${NC}"
sleep 20

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}P1 Experiment Setup Complete!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Server PID: ${SERVER_PID}${NC}"
echo -e "${GREEN}Nodes PID: ${NODES_PID}${NC}"
echo ""
echo -e "${YELLOW}Experiment Configuration:${NC}"
echo -e "  - Experiment: P1 (Share Early Layers)"
echo -e "  - Rounds: 500"
echo -e "  - Games per node per round: 400"
echo -e "  - Total nodes: 8 (4 tactical + 4 positional)"
echo -e "  - Shared layers: input + res_blocks 0-5 (6 blocks)"
echo -e "  - Cluster-specific layers: res_blocks 6-18 + heads (13 blocks + 2 heads)"
echo -e "  - Intra-cluster aggregation: YES"
echo -e "  - Inter-cluster aggregation: YES (early layers only)"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  Server: tail -f logs/server_P1.log"
echo -e "  Nodes:  tail -f logs/nodes_P1.log"
echo ""
echo -e "${GREEN}Training will start automatically after nodes connect (30 seconds)${NC}"
echo ""
echo -e "${YELLOW}To stop the experiment:${NC}"
echo -e "  kill ${SERVER_PID} ${NODES_PID}"
echo -e "  Or use: pkill -f 'python chess-federated-learning'"
echo ""
echo -e "${BLUE}================================================================${NC}"

# Keep script running and forward logs
echo -e "${BLUE}Monitoring server logs... (Press Ctrl+C to detach)${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    kill $NODES_PID 2>/dev/null || true
    # Restore original config
    mv chess-federated-learning/config/server_config.yaml.backup chess-federated-learning/config/server_config.yaml 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT

# Tail the server log
tail -f logs/server_P1.log
