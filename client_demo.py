#!/usr/bin/env python3

import asyncio
import sys
import os
from typing import Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), 'chess-federated-learning'))

from client.communication.client_socket import FederatedLearningClient, ClientState
from server.communication.protocol import Message, MessageType, MessageFactory


class ClientDemo:
    def __init__(self, node_id: str, cluster_id: str, server_host: str = "localhost", server_port: int = 8765):
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.client = FederatedLearningClient(
            node_id=node_id,
            cluster_id=cluster_id,
            server_host=server_host,
            server_port=server_port
        )
        self.is_running = False

    async def start_client(self):
        print(f"🚀 Starting Federated Learning Client: {self.node_id}")
        print(f"📡 Connecting to server at {self.client.server_host}:{self.client.server_port}")
        print(f"🏷️  Node ID: {self.node_id}")
        print(f"🎯 Cluster: {self.cluster_id}")

        # Set up message handlers
        self.client.set_message_handler(MessageType.START_TRAINING, self._handle_start_training)
        self.client.set_message_handler(MessageType.CLUSTER_MODEL, self._handle_cluster_model)
        self.client.set_message_handler(MessageType.ERROR, self._handle_error)
        self.client.set_message_handler(MessageType.DISCONNECT, self._handle_disconnect)

        self.is_running = True

        # Start client in background
        client_task = asyncio.create_task(self.client.start())
        menu_task = asyncio.create_task(self._menu_loop())

        try:
            # Use wait instead of gather to handle task completion better
            done, pending = await asyncio.wait(
                [client_task, menu_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except KeyboardInterrupt:
            print("\n🛑 Client shutdown requested...")
        finally:
            self.is_running = False
            print("🛑 Stopping client...")
            await self.client.stop()
            print("✅ Client stopped")

    async def _handle_start_training(self, message: Message):
        games_per_round = message.payload.get('games_per_round', 100)
        round_num = message.round_num

        print(f"\n🏃 Training started! Round {round_num}, {games_per_round} games")
        print("🎮 Simulating training...")

        # Simulate training delay
        await asyncio.sleep(2)

        # Simulate training results
        dummy_model = {
            "layer1.weight": [0.2, 0.3, 0.4],
            "layer1.bias": [0.1, 0.2],
            "trained": True
        }
        samples = 50 + (round_num * 10)  # Simulate increasing samples
        loss = max(0.1, 1.0 - (round_num * 0.1))  # Simulate decreasing loss

        print(f"✅ Training complete! Loss: {loss:.4f}, Samples: {samples}")

    async def _handle_cluster_model(self, message: Message):
        round_num = message.round_num
        model_state = message.payload.get('model_state', {})

        print(f"\n📥 Received cluster model for round {round_num}")
        print(f"🧠 Model keys: {list(model_state.keys())}")
        print("📊 Model updated!")

    async def _handle_error(self, message: Message):
        error_msg = message.payload.get('message', 'Unknown error')
        error_code = message.payload.get('error_code', 0)

        print(f"\n❌ Server error [{error_code}]: {error_msg}")

    async def _handle_disconnect(self, message: Message):
        reason = message.payload.get('reason', 'Server disconnect')

        print(f"\n🚪 Disconnection requested by server: {reason}")

    async def _menu_loop(self):
        # Wait for client to connect
        while self.is_running and not self.client.is_connected():
            await asyncio.sleep(0.5)

        if not self.is_running:
            return

        print(f"\n✅ Connected to server as {self.node_id}")

        while self.is_running:
            # Check if still connected
            if not self.client.is_connected() and self.client.state not in [ClientState.CONNECTING, ClientState.REGISTERING]:
                print("❌ Lost connection to server")
                if not self.client.auto_reconnect:
                    break
                else:
                    print("🔄 Waiting for reconnection...")
                    await asyncio.sleep(1)
                    continue

            print("\n" + "="*50)
            print(f"🎛️  CLIENT MENU - {self.node_id}")
            print("="*50)
            print("1. 📊 Send model update")
            print("2. 📈 Send metrics")
            print("3. 💗 Send heartbeat")
            print("4. ℹ️  Show client status")
            print("5. 📊 Show connection stats")
            print("6. 🔄 Simulate training cycle")
            print("7. 📤 Send custom message")
            print("8. 🚪 Disconnect from server")
            print("="*50)

            try:
                choice = await self._get_input("Enter your choice (1-8): ")

                if choice == "1":
                    await self._send_model_update()
                elif choice == "2":
                    await self._send_metrics()
                elif choice == "3":
                    await self._send_heartbeat()
                elif choice == "4":
                    await self._show_client_status()
                elif choice == "5":
                    await self._show_connection_stats()
                elif choice == "6":
                    await self._simulate_training_cycle()
                elif choice == "7":
                    await self._send_custom_message()
                elif choice == "8":
                    print("🚪 Disconnecting from server...")
                    await self._disconnect_from_server()
                    self.is_running = False
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                print("\n🛑 Stopping client...")
                self.is_running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(2)

    async def _get_input(self, prompt: str) -> str:
        """Non-blocking input for async context"""
        print(prompt, end='', flush=True)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sys.stdin.readline)
        return result.strip()

    async def _send_model_update(self):
        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num) if round_num else 1

        samples = await self._get_input("Enter sample count (default 100): ")
        samples = int(samples) if samples else 100

        loss = await self._get_input("Enter loss value (default 0.5): ")
        loss = float(loss) if loss else 0.5

        # Create dummy model state
        model_state = {
            "layer1.weight": [0.1 + round_num * 0.1, 0.2, 0.3],
            "layer1.bias": [0.0, 0.1],
            "round": round_num,
            "node_id": self.node_id
        }

        print("📤 Sending model update...")
        await self.client.send_model_update(model_state, samples, loss, round_num)
        print("✅ Model update sent!")

    async def _send_metrics(self):
        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num.strip()) if round_num.strip() else 1

        loss = await self._get_input("Enter loss value (default 0.3): ")
        loss = float(loss.strip()) if loss.strip() else 0.3

        samples = await self._get_input("Enter sample count (default 75): ")
        samples = int(samples.strip()) if samples.strip() else 75

        metrics = {
            "loss": loss,
            "samples": samples,
            "accuracy": 1.0 - loss,  # Simple inverse relationship
            "node_id": self.node_id
        }

        print("📈 Sending metrics...")
        await self.client.send_metrics(metrics, round_num)
        print("✅ Metrics sent!")

    async def _send_heartbeat(self):
        heartbeat_msg = MessageFactory.create_heartbeat(
            node_id=self.node_id,
            cluster_id=self.cluster_id
        )

        print("💗 Sending heartbeat...")
        await self.client._send_message(heartbeat_msg)
        print("✅ Heartbeat sent!")

    async def _show_client_status(self):
        print(f"\nℹ️  Client Status:")
        print(f"Node ID: {self.client.node_id}")
        print(f"Cluster ID: {self.client.cluster_id}")
        print(f"State: {self.client.state.value}")
        print(f"Connected: {self.client.is_connected()}")
        print(f"Training: {self.client.is_training()}")
        print(f"Current round: {self.client.get_current_round() or 'N/A'}")
        print(f"Server URL: {self.client.server_url}")
        print(f"Auto-reconnect: {self.client.auto_reconnect}")
        print(f"Heartbeat interval: {self.client.heartbeat_interval}s")

    async def _show_connection_stats(self):
        stats = self.client.get_stats()
        print(f"\n📊 Connection Statistics:")
        print(f"Connection attempts: {stats['connection_attempts']}")
        print(f"Successful connections: {stats['successful_connections']}")
        print(f"Total uptime: {stats['total_uptime']:.1f}s")
        print(f"Messages sent: {stats['messages_sent']}")
        print(f"Messages received: {stats['messages_received']}")
        print(f"Reconnections: {stats['reconnections']}")

        if stats['total_uptime'] > 0:
            msg_rate = (stats['messages_sent'] + stats['messages_received']) / stats['total_uptime']
            print(f"Average message rate: {msg_rate:.2f} msg/s")

    async def _simulate_training_cycle(self):
        print("🎮 Simulating complete training cycle...")

        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num.strip()) if round_num.strip() else 1

        # Step 1: Simulate receiving training command (already handled by server)
        print("1️⃣ Waiting for training command from server...")
        print("   (In real scenario, server would send START_TRAINING)")

        await asyncio.sleep(1)

        # Step 2: Simulate training
        print("2️⃣ Starting local training...")
        training_time = 3  # seconds
        for i in range(training_time):
            progress = (i + 1) / training_time * 100
            print(f"   Training progress: {progress:.0f}%")
            await asyncio.sleep(1)

        # Step 3: Generate training results
        samples = 80 + round_num * 20
        loss = max(0.05, 0.8 - round_num * 0.1)

        print("3️⃣ Training completed!")
        print(f"   Samples trained: {samples}")
        print(f"   Final loss: {loss:.4f}")

        # Step 4: Send model update
        model_state = {
            "layer1.weight": [0.15 + round_num * 0.05, 0.25, 0.35],
            "layer2.weight": [0.4, 0.5, 0.6],
            "layer1.bias": [0.05, 0.15],
            "layer2.bias": [0.1],
            "trained_samples": samples,
            "round": round_num,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time()
        }

        print("4️⃣ Sending model update to server...")
        await self.client.send_model_update(model_state, samples, loss, round_num)

        print("✅ Training cycle complete!")
        print("🔄 Ready for next round or aggregation")

    async def _send_custom_message(self):
        print("\nCustom message types:")
        print("1. Error message")
        print("2. Disconnect message")
        print("3. Raw message")

        choice = await self._get_input("Select message type: ")

        if choice.strip() == "1":
            error_msg = await self._get_input("Enter error message: ")
            error_code = await self._get_input("Enter error code (default 1001): ")
            error_code = int(error_code.strip()) if error_code.strip() else 1001

            message = MessageFactory.create_error(
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                error_msg=error_msg.strip(),
                error_code=error_code
            )

        elif choice.strip() == "2":
            reason = await self._get_input("Enter disconnect reason: ")
            message = MessageFactory.create_disconnect(
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                reason=reason.strip()
            )

        elif choice.strip() == "3":
            print("⚠️  Raw message - be careful with format!")
            msg_type = await self._get_input("Message type: ")
            payload_str = await self._get_input("Payload (JSON string): ")

            try:
                import json
                payload = json.loads(payload_str) if payload_str.strip() else {}
            except json.JSONDecodeError:
                print("❌ Invalid JSON payload")
                return

            message = Message(
                type=msg_type.strip(),
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                payload=payload,
                timestamp=asyncio.get_event_loop().time()
            )

        else:
            print("❌ Invalid choice.")
            return

        print("📤 Sending custom message...")
        await self.client._send_message(message)
        print("✅ Custom message sent!")

    async def _disconnect_from_server(self):
        """Properly disconnect from the server"""
        try:
            # Send disconnect message first
            disconnect_msg = MessageFactory.create_disconnect(
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                reason="User requested disconnect"
            )

            print("📤 Sending disconnect message...")
            await self.client._send_message(disconnect_msg)
            print("✅ Disconnect message sent")

            # Give server time to process
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"⚠️  Warning: Could not send disconnect message: {e}")

        # Stop the client (this will close the connection)
        print("🔌 Closing connection...")


async def main():
    print("🎯 Federated Learning Client Demo")
    print("This script demonstrates the FL client with an interactive menu.")
    print("\n📋 Available clusters and valid node IDs:")
    print("Cluster 'cluster_aggressive' (aggressive playstyle):")
    print("  Valid node IDs: agg_001, agg_002, agg_003, agg_004")
    print("Cluster 'cluster_positional' (positional playstyle):")
    print("  Valid node IDs: pos_001, pos_002, pos_003, pos_004")

    # Get client configuration
    node_id = input("\nEnter node ID (e.g., agg_001 or pos_001): ").strip()
    if not node_id:
        node_id = "agg_001"  # Default to first aggressive node

    # Auto-determine cluster_id based on node_id prefix
    if node_id.startswith("agg_"):
        cluster_id = "cluster_aggressive"
    elif node_id.startswith("pos_"):
        cluster_id = "cluster_positional"
    else:
        print("⚠️  Warning: Node ID doesn't match expected patterns (agg_XXX or pos_XXX)")
        cluster_id = input("Enter cluster ID (cluster_aggressive or cluster_positional): ").strip()
        cluster_id = cluster_id if cluster_id else "cluster_aggressive"

    server_host = input("Enter server host (default: localhost): ").strip()
    server_host = server_host if server_host else "localhost"

    server_port = input("Enter server port (default: 8765): ").strip()
    server_port = int(server_port) if server_port else 8765

    print(f"\n🔧 Configuration:")
    print(f"Node ID: {node_id}")
    print(f"Cluster ID: {cluster_id}")
    print(f"Server: {server_host}:{server_port}")

    if cluster_id == "cluster_aggressive":
        print(f"🗡️  Playstyle: Aggressive (tactical, attacking chess)")
    elif cluster_id == "cluster_positional":
        print(f"🏰 Playstyle: Positional (strategic, long-term planning)")

    demo = ClientDemo(node_id, cluster_id, server_host, server_port)

    try:
        await demo.start_client()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Client error: {e}")


if __name__ == "__main__":
    asyncio.run(main())