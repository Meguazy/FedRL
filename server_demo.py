#!/usr/bin/env python3

import asyncio
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'chess-federated-learning'))

from server.communication.server_socket import FederatedLearningServer
from server.communication.protocol import Message, MessageType, MessageFactory


class ServerDemo:
    def __init__(self):
        # Path to cluster configuration
        cluster_config_path = "chess-federated-learning/config/cluster_topology.yaml"
        self.server = FederatedLearningServer(
            host="localhost",
            port=8765,
            cluster_config_path=cluster_config_path
        )
        self.is_running = False

    async def start_server(self):
        print("🚀 Starting Federated Learning Server...")
        print(f"📡 Server will listen on {self.server.host}:{self.server.port}")

        # Set up message handlers
        self.server.set_message_handler(MessageType.MODEL_UPDATE, self._handle_model_update)
        self.server.set_message_handler(MessageType.METRICS, self._handle_metrics)

        self.is_running = True

        # Start server in background
        server_task = asyncio.create_task(self.server.start_server())
        menu_task = asyncio.create_task(self._menu_loop())

        try:
            await asyncio.gather(server_task, menu_task)
        except KeyboardInterrupt:
            print("\n🛑 Server shutdown requested...")
        finally:
            self.is_running = False
            await self.server.stop_server()

    async def _handle_model_update(self, node_id: str, message: Message):
        samples = message.payload.get("samples", 0)
        loss = message.payload.get("loss", 0.0)
        round_num = message.round_num
        print(f"📊 Model update from {node_id} (round {round_num}): {samples} samples, loss={loss:.4f}")

    async def _handle_metrics(self, node_id: str, message: Message):
        loss = message.payload.get("loss", 0.0)
        samples = message.payload.get("samples", 0)
        round_num = message.round_num

        # Update the node's current round
        if node_id in self.server.connected_nodes:
            self.server.connected_nodes[node_id].current_round = round_num

        print(f"📈 Metrics from {node_id} (round {round_num}): loss={loss:.4f}, samples={samples}")

    async def _menu_loop(self):
        await asyncio.sleep(2)  # Give server time to start

        while self.is_running:
            print("\n" + "="*50)
            print("🎛️  FEDERATED LEARNING SERVER MENU")
            print("="*50)
            print("1. 📋 Show connected nodes")
            print("2. 🔔 Broadcast start training")
            print("3. 📤 Send cluster model")
            print("4. 💌 Send message to specific node")
            print("5. 🏃 Start training round for cluster")
            print("6. ❌ Disconnect node")
            print("7. 📊 Show server stats")
            print("8. 🛑 Stop server")
            print("="*50)

            try:
                choice = await self._get_input("Enter your choice (1-8): ")

                if choice == "1":
                    await self._show_connected_nodes()
                elif choice == "2":
                    await self._broadcast_start_training()
                elif choice == "3":
                    await self._send_cluster_model()
                elif choice == "4":
                    await self._send_message_to_node()
                elif choice == "5":
                    await self._start_training_for_cluster()
                elif choice == "6":
                    await self._disconnect_node()
                elif choice == "7":
                    await self._show_server_stats()
                elif choice == "8":
                    print("🛑 Stopping server...")
                    self.is_running = False
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
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

    async def _show_connected_nodes(self):
        nodes = self.server.get_connected_nodes()

        if not nodes:
            print("📭 No nodes currently connected.")
            return

        print(f"\n📋 Connected Nodes ({len(nodes)}):")
        print("-" * 60)
        for node_id, node in nodes.items():
            uptime = time.time() - node.registration_time
            print(f"🔹 {node_id}")
            print(f"   Cluster: {node.cluster_id}")
            print(f"   State: {node.state.value}")
            print(f"   Uptime: {uptime:.1f}s")
            print(f"   Round: {node.current_round or 'N/A'}")
            print()

    async def _broadcast_start_training(self):
        games_per_round = await self._get_input("Enter games per round (default 100): ")
        games_per_round = int(games_per_round) if games_per_round else 100

        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num) if round_num else 1

        # Create start training message
        message = MessageFactory.create_start_training(
            node_id="server",
            cluster_id="all",
            games_per_round=games_per_round,
            round_num=round_num
        )

        await self.server.broadcast_to_all(message)

        # Update current round for all connected nodes
        for node in self.server.connected_nodes.values():
            node.current_round = round_num

        print(f"📡 Broadcasted START_TRAINING: round {round_num}, {games_per_round} games")

    async def _send_cluster_model(self):
        cluster_id = await self._get_input("Enter cluster ID: ")

        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num) if round_num else 1

        # Create dummy model state
        model_state = {
            "layer1.weight": [0.1, 0.2, 0.3],
            "layer1.bias": [0.0, 0.1]
        }

        message = MessageFactory.create_cluster_model(
            node_id="server",
            cluster_id=cluster_id,
            model_state=model_state,
            round_num=round_num
        )

        await self.server.broadcast_to_cluster(cluster_id, message)
        print(f"📤 Sent CLUSTER_MODEL to cluster '{cluster_id}' for round {round_num}")

    async def _send_message_to_node(self):
        nodes = self.server.get_connected_nodes()

        if not nodes:
            print("📭 No nodes connected.")
            return

        print("\nAvailable nodes:")
        for i, node_id in enumerate(nodes.keys(), 1):
            print(f"{i}. {node_id}")

        choice = await self._get_input("Select node number: ")
        try:
            node_index = int(choice.strip()) - 1
            node_id = list(nodes.keys())[node_index]
            node = nodes[node_id]
        except (ValueError, IndexError):
            print("❌ Invalid node selection.")
            return

        print("\nMessage types:")
        print("1. Error message")
        print("2. Start training")
        print("3. Disconnect")

        msg_choice = await self._get_input("Select message type: ")

        if msg_choice.strip() == "1":
            error_msg = await self._get_input("Enter error message: ")
            message = MessageFactory.create_error(
                node_id=node_id,
                cluster_id=node.cluster_id,
                error_msg=error_msg.strip()
            )
        elif msg_choice.strip() == "2":
            games = await self._get_input("Games per round (default 50): ")
            games = int(games.strip()) if games.strip() else 50
            round_num = await self._get_input("Round number (default 1): ")
            round_num = int(round_num) if round_num else 1

            message = MessageFactory.create_start_training(
                node_id=node_id,
                cluster_id=node.cluster_id,
                games_per_round=games,
                round_num=round_num
            )
        elif msg_choice.strip() == "3":
            reason = await self._get_input("Disconnect reason (default 'manual'): ")
            reason = reason.strip() or "manual"
            message = MessageFactory.create_disconnect(
                node_id=node_id,
                cluster_id=node.cluster_id,
                reason=reason
            )
        else:
            print("❌ Invalid message type.")
            return

        await self.server._send_message_to_node(node_id, message)
        print(f"📤 Message sent to {node_id}")

    async def _start_training_for_cluster(self):
        cluster_id = await self._get_input("Enter cluster ID: ")
        cluster_id = cluster_id.strip()

        cluster_nodes = self.server.get_cluster_nodes(cluster_id)
        if not cluster_nodes:
            print(f"❌ No nodes found in cluster '{cluster_id}'")
            return

        games_per_round = await self._get_input("Enter games per round (default 100): ")
        games_per_round = int(games_per_round.strip()) if games_per_round.strip() else 100

        round_num = await self._get_input("Enter round number (default 1): ")
        round_num = int(round_num) if round_num else 1

        message = MessageFactory.create_start_training(
            node_id="server",
            cluster_id=cluster_id,
            games_per_round=games_per_round,
            round_num=round_num
        )

        await self.server.broadcast_to_cluster(cluster_id, message)

        # Update current round for nodes in this cluster
        for node_id in cluster_nodes:
            if node_id in self.server.connected_nodes:
                self.server.connected_nodes[node_id].current_round = round_num

        print(f"🏃 Started training for cluster '{cluster_id}': {len(cluster_nodes)} nodes")

    async def _disconnect_node(self):
        nodes = self.server.get_connected_nodes()

        if not nodes:
            print("📭 No nodes connected.")
            return

        print("\nConnected nodes:")
        for i, node_id in enumerate(nodes.keys(), 1):
            print(f"{i}. {node_id}")

        choice = await self._get_input("Select node to disconnect: ")
        try:
            node_index = int(choice.strip()) - 1
            node_id = list(nodes.keys())[node_index]
        except (ValueError, IndexError):
            print("❌ Invalid node selection.")
            return

        reason = await self._get_input("Disconnect reason (default 'manual'): ")
        reason = reason.strip() or "manual"

        await self.server._disconnect_node(node_id, reason)
        print(f"❌ Disconnected node {node_id}")

    async def _show_server_stats(self):
        stats = self.server.get_server_statistics()

        print(f"\n📊 Server Statistics:")
        print(f"Connected nodes: {stats['connections']['total_connected_nodes']}")
        print(f"Active clusters: {stats['connections']['total_active_clusters']}")
        print(f"Messages handled: {stats['server']['total_messages_handled']}")
        print(f"Current round: {stats['server']['current_round']}")
        print(f"Server uptime: {stats['server']['uptime_seconds']:.1f}s")

        print(f"\n🏗️  Cluster Manager Statistics:")
        cm_stats = stats['cluster_manager']
        print(f"Total clusters: {cm_stats['cluster_count']}")
        print(f"Total expected nodes: {cm_stats['total_expected_nodes']}")
        print(f"Total registered nodes: {cm_stats['total_registered_nodes']}")
        print(f"Active nodes: {cm_stats['total_active_nodes']}")
        print(f"Ready clusters: {cm_stats['ready_clusters']}")
        print(f"Uptime: {cm_stats['uptime_seconds']:.1f}s")

        print(f"\n🏷️  Cluster Readiness:")
        cluster_readiness = stats['cluster_readiness']
        for cluster_id, readiness in cluster_readiness.items():
            status = "✅" if readiness['is_ready'] else "❌"
            ratio = readiness.get('readiness_ratio', 0.0)
            active = readiness.get('active_nodes', 0)
            expected = readiness.get('expected_nodes', 0)
            playstyle = readiness.get('playstyle', 'unknown')
            print(f"  {status} {cluster_id} ({playstyle}): {active}/{expected} nodes ({ratio:.1%})")

        print(f"\n🔗 Connected Clusters:")
        for cluster_id in self.server.connections_by_cluster:
            node_count = len(self.server.connections_by_cluster[cluster_id])
            print(f"  {cluster_id}: {node_count} connected nodes")


async def main():
    print("🎯 Federated Learning Server Demo")
    print("This script demonstrates the FL server with an interactive menu.")
    print("Clients can connect and you can send various messages through the menu.\n")

    demo = ServerDemo()

    try:
        await demo.start_server()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())