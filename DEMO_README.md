# Federated Learning Demo Scripts

This directory contains demo scripts that showcase the server-client communication system for federated learning.

## Files

- **`server_demo.py`** - Interactive server with menu for managing connections and sending messages
- **`client_demo.py`** - Interactive client with menu for connecting and sending messages
- **`DEMO_README.md`** - This instructions file

## Prerequisites

Make sure you have the required dependencies:
- Python 3.7+
- `websockets` library
- `loguru` library

Install dependencies:
```bash
pip install websockets loguru
```

## Usage

### 1. Start the Server

```bash
python server_demo.py
```

The server will:
- Start listening on `localhost:8765`
- Display a menu for server operations
- Show connected clients and handle messages

### 2. Start Client(s)

In separate terminals, run:
```bash
python client_demo.py
```

You'll be prompted for:
- Node ID (default: demo_node_001)
- Cluster ID (default: demo_cluster)
- Server host (default: localhost)
- Server port (default: 8765)

### 3. Demo Scenarios

#### Server Menu Options:
1. **Show connected nodes** - View all connected clients
2. **Broadcast start training** - Send training command to all clients
3. **Send cluster model** - Distribute aggregated model to a cluster
4. **Send message to specific node** - Target individual clients
5. **Start training for cluster** - Begin training for specific cluster
6. **Disconnect node** - Manually disconnect a client
7. **Show server stats** - Display server statistics
8. **Stop server** - Shutdown the server

#### Client Menu Options:
1. **Send model update** - Submit trained model weights to server
2. **Send metrics** - Report training metrics (loss, samples)
3. **Send heartbeat** - Manual connection health check
4. **Show client status** - Display client state information
5. **Show connection stats** - View connection statistics
6. **Simulate training cycle** - Complete training simulation
7. **Send custom message** - Send custom protocol messages
8. **Disconnect from server** - Gracefully disconnect

## Example Workflow

1. Start server with `python server_demo.py`
2. Start 2-3 clients with `python client_demo.py` in separate terminals
3. Use different node IDs (e.g., node_001, node_002, node_003)
4. On server: Choose option 1 to see connected nodes
5. On server: Choose option 2 to broadcast start training
6. On clients: Choose option 6 to simulate training and send model update
7. On server: Choose option 3 to send cluster model back to clients

## Message Types Supported

- **REGISTER** - Client registration
- **REGISTER_ACK** - Registration acknowledgment
- **START_TRAINING** - Begin training round
- **MODEL_UPDATE** - Submit trained model weights
- **CLUSTER_MODEL** - Distribute aggregated model
- **METRICS** - Report training metrics
- **HEARTBEAT** - Connection health check
- **ERROR** - Error notifications
- **DISCONNECT** - Graceful disconnection

## Notes

- The demos simulate training with dummy data
- Real model weights would be much larger
- Multiple clients can connect simultaneously
- Server handles concurrent connections
- All communication uses WebSocket protocol
- Messages follow the defined FL protocol structure

## Troubleshooting

**Connection refused**: Make sure server is running before starting clients

**Import errors**: Ensure you're running from the correct directory with proper Python path

**Port in use**: Server default port 8765 might be occupied - check server output

**Client won't connect**: Verify server host/port settings match between server and client