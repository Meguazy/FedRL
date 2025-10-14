# Quick Start: Play Chess Against Your Trained Models! ♟️

Play against the models you trained using the federated learning system.

## Prerequisites

✅ PyQt6 (already installed)
✅ Trained model checkpoints in `chess-federated-learning/storage/models/`

## Launch the Game

```bash
cd /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning

# Launch the GUI
uv run python play_chess.py
```

## If Running on WSL

The GUI needs a display server. See detailed instructions:
[chess-federated-learning/gui/SETUP_WSL.md](chess-federated-learning/gui/SETUP_WSL.md)

**Quick WSL setup:**

```bash
# 1. Install X server dependencies
sudo apt install -y libxcb-cursor0 libxcb-xinerama0

# 2. Set display (if using VcXsrv on Windows)
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# 3. Launch the game
uv run python play_chess.py
```

## How to Play

### 1. Load Your Model

Click **"Load Model Checkpoint"** and select a model file from:
- `storage/models/cluster_tactical_round_X.pt`
- `storage/models/cluster_positional_round_X.pt`
- `storage/experiments/<run_id>/checkpoints/`

The GUI will load your trained AlphaZero network.

### 2. Choose Your Color

Select **White** or **Black** from the dropdown.

### 3. Make Moves

- **Click a piece** to select it (legal moves highlight in green)
- **Click a square** to move there
- The AI will respond automatically

### 4. Game Features

- **New Game**: Start over
- **Undo Move**: Take back moves
- **Flip Board**: Rotate the view
- **Move History**: See all moves

## Model Selection Tips

### Cluster-Specific Models

- **Tactical models** (`cluster_tactical_*`): Prefer aggressive, tactical play
- **Positional models** (`cluster_positional_*`): Prefer strategic, positional play

### Training Rounds

- **Early rounds** (1-3): Weaker, may make mistakes
- **Mid rounds** (4-10): Developing understanding
- **Late rounds** (10+): Best performance

### Aggregated Models

- **Inter-cluster aggregated**: Best overall performance
- Combines knowledge from all playstyles

## Example Game Session

```bash
# Launch GUI
cd /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning
uv run python play_chess.py

# In the GUI:
# 1. Click "Load Model Checkpoint"
# 2. Navigate to: storage/models/cluster_tactical_round_5.pt
# 3. Select "Black" (play as Black)
# 4. Click "New Game"
# 5. AI (White) makes first move
# 6. Make your move by clicking pieces
```

## Troubleshooting

### "Could not load Qt platform plugin"

You're on WSL without a display. See [SETUP_WSL.md](chess-federated-learning/gui/SETUP_WSL.md)

### "Failed to load model"

Make sure the model file:
- Exists in the storage directory
- Is a PyTorch checkpoint file
- Was created by your training runs

### AI Makes Weak Moves

- Load a model from a later training round
- Use an inter-cluster aggregated model
- Models need sufficient training data

### GUI Doesn't Open

```bash
# Test if PyQt6 works
uv run python -c "from PyQt6.QtWidgets import QApplication; print('OK')"

# If that fails, you need X server setup (WSL only issue)
```

## Features

✨ **Interactive Board**: Click-to-move interface
🤖 **AI Opponent**: Your trained AlphaZero models
📊 **Move History**: Track the entire game
♟️ **Legal Moves**: Highlighted in green
🔄 **Undo**: Take back mistakes
🎨 **Board Flip**: See from both sides
✅ **Game Detection**: Auto-detect checkmate/stalemate

## Advanced: Compare Models

Want to see which model is stronger?

1. Play a game against tactical model
2. Note the game result
3. Play against positional model
4. Compare performance!

Or create a model vs model script (future feature).

## Keyboard Controls

Currently mouse-only. Future versions may add:
- Keyboard move input (e2e4 notation)
- Arrow key navigation

## Need Help?

- GUI Documentation: [gui/README.md](chess-federated-learning/gui/README.md)
- WSL Setup: [gui/SETUP_WSL.md](chess-federated-learning/gui/SETUP_WSL.md)
- Report issues: GitHub repository

---

**Enjoy playing against your own AI! 🎮♟️**
