# Chess GUI - Play Against Your FL Models

A graphical chess application that lets you play against models trained using the federated learning system.

## Features

- **Interactive Chess Board**: Click pieces to select, click destination to move
- **Model Selection**: Load any saved model checkpoint from your training runs
- **Move Validation**: Only legal moves are allowed
- **Move Highlighting**: See available moves for selected pieces
- **Color Selection**: Play as White or Black
- **Move History**: Track all moves in the game
- **Game State Detection**: Automatic detection of check, checkmate, stalemate
- **Undo Functionality**: Take back moves if needed
- **Board Flipping**: Flip the board to see from opponent's perspective

## Installation

Make sure you have PyQt6 installed:

```bash
pip install PyQt6
```

All other dependencies should already be installed from the main project.

## Usage

### Launch the GUI

```bash
# From the chess-federated-learning directory
python play_chess.py

# Or directly
python gui/chess_game.py

# Or with uv
uv run python play_chess.py
```

### Playing a Game

1. **Load a Model** (optional):
   - Click "Load Model Checkpoint"
   - Navigate to `storage/models/` or wherever your checkpoints are saved
   - Select a model file (e.g., `cluster_tactical_round_5.pt`)
   - The model will be loaded and ready to play

2. **Choose Your Color**:
   - Select "White" or "Black" from the dropdown
   - White plays first

3. **Make Moves**:
   - Click on a piece to select it (legal moves will be highlighted in green)
   - Click on a destination square to move
   - If playing against AI, it will automatically make its move

4. **Game Controls**:
   - **New Game**: Start a fresh game
   - **Undo Move**: Take back the last move (or two moves if playing vs AI)
   - **Flip Board**: Rotate the board 180 degrees

## Model Checkpoint Locations

The GUI can load models from:
- `storage/models/` - Models saved during training
- `storage/experiments/<run_id>/checkpoints/` - Experiment-specific checkpoints

Supported file formats:
- PyTorch checkpoint files (`.pt`, `.pth`)
- Files with serialized model state (from federated learning training)

## Keyboard Shortcuts

Currently, the GUI uses mouse interaction only. Future versions may add:
- Arrow keys for board navigation
- Keyboard move input (algebraic notation)

## Troubleshooting

### "No module named PyQt6"
Install PyQt6:
```bash
pip install PyQt6
```

### "Failed to load model"
Make sure the model file:
- Is a valid PyTorch checkpoint
- Contains a `model_state` key or is a direct state_dict
- Was saved with a compatible version of the AlphaZeroNet architecture

### AI Makes Illegal Moves
This shouldn't happen due to move filtering, but if it does:
- The model may not be well-trained
- Try loading a model from a later training round

### Board Not Displaying Correctly
- Make sure you have proper font support for Unicode chess pieces
- Try resizing the window

## Technical Details

### Architecture

- **ChessBoardWidget**: Custom PyQt6 widget for rendering the chess board
- **ChessAI**: Wrapper around AlphaZeroNet that converts board positions to moves
- **ChessGameWindow**: Main application window with controls

### Model Integration

The GUI uses:
- `BoardEncoder`: Converts board positions to 119-plane tensor representations
- `MoveEncoder`: Converts between chess moves and action indices (0-4671)
- `AlphaZeroNet`: Your trained policy/value network

### Performance

- Models run on CPU by default (GPU if available)
- Move generation is typically instant (<100ms)
- Board rendering uses native PyQt6 painting

## Future Enhancements

Potential improvements:
- [ ] MCTS integration for stronger play
- [ ] Adjustable AI difficulty (temperature, search depth)
- [ ] Game analysis and engine evaluation
- [ ] Save/load games (PGN format)
- [ ] Multiple model comparison
- [ ] Opening book integration
- [ ] Time controls

## License

Same as the main federated learning project.
