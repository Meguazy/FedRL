"""
Chess GUI Application - Play against trained federated learning models.

This application provides a graphical interface to play chess against
models trained using the federated learning system.

Features:
- Interactive chess board with drag-and-drop moves
- Model selection from checkpoint files
- Move highlighting and legal move validation
- Game state display (check, checkmate, stalemate)
- Move history
- Undo functionality
"""

import sys
import torch
import chess
import chess.svg
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
    QComboBox, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen, QFont
from PyQt6.QtSvg import QSvgRenderer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.trainer.models.alphazero_net import AlphaZeroNet
from data.board_encoder import BoardEncoder
from data.move_encoder import MoveEncoder
from common.model_serialization import PyTorchSerializer


class ChessBoardWidget(QWidget):
    """
    Widget that displays an interactive chess board.
    """

    square_clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.flipped = False

        # Board dimensions
        self.square_size = 70
        self.board_size = self.square_size * 8

        self.setFixedSize(self.board_size, self.board_size)
        self.setMouseTracking(True)

        # Colors
        self.light_square_color = QColor(240, 217, 181)
        self.dark_square_color = QColor(181, 136, 99)
        self.selected_square_color = QColor(255, 255, 0, 100)
        self.legal_move_color = QColor(0, 255, 0, 80)
        self.last_move_color = QColor(255, 255, 0, 60)

    def set_board(self, board: chess.Board):
        """Update the board state."""
        self.board = board
        self.selected_square = None
        self.legal_moves = []
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse clicks on the board."""
        x = event.position().x()
        y = event.position().y()

        col = int(x // self.square_size)
        row = int(y // self.square_size)

        if self.flipped:
            row = 7 - row
            col = 7 - col

        if 0 <= row < 8 and 0 <= col < 8:
            square = chess.square(col, 7 - row)
            self.handle_square_click(square)

    def handle_square_click(self, square: int):
        """Handle a click on a chess square."""
        # If no square selected, select this square if it has a piece
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                # Get legal moves from this square
                self.legal_moves = [
                    move for move in self.board.legal_moves
                    if move.from_square == square
                ]
                self.update()
        else:
            # Try to make a move
            move = chess.Move(self.selected_square, square)

            # Check for promotion
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(square) == 7) or \
                   (piece.color == chess.BLACK and chess.square_rank(square) == 0):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self.last_move = move
                self.board.push(move)
                self.square_clicked.emit(move.from_square, move.to_square)

            # Deselect
            self.selected_square = None
            self.legal_moves = []
            self.update()

    def paintEvent(self, event):
        """Draw the chess board and pieces."""
        painter = QPainter(self)

        # Draw squares
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size

                # Determine square color
                is_light = (row + col) % 2 == 0
                color = self.light_square_color if is_light else self.dark_square_color
                painter.fillRect(x, y, self.square_size, self.square_size, color)

                # Highlight last move
                if self.last_move:
                    display_row = row if not self.flipped else 7 - row
                    display_col = col if not self.flipped else 7 - col
                    square = chess.square(display_col, 7 - display_row)

                    if square in (self.last_move.from_square, self.last_move.to_square):
                        painter.fillRect(x, y, self.square_size, self.square_size, self.last_move_color)

                # Highlight selected square
                if self.selected_square is not None:
                    display_row = row if not self.flipped else 7 - row
                    display_col = col if not self.flipped else 7 - col
                    square = chess.square(display_col, 7 - display_row)

                    if square == self.selected_square:
                        painter.fillRect(x, y, self.square_size, self.square_size, self.selected_square_color)

                    # Highlight legal moves
                    for move in self.legal_moves:
                        if move.to_square == square:
                            painter.fillRect(x, y, self.square_size, self.square_size, self.legal_move_color)

        # Draw pieces using Unicode characters
        font = QFont("Arial", 48)
        painter.setFont(font)

        piece_symbols = {
            (chess.PAWN, chess.WHITE): '♙',
            (chess.KNIGHT, chess.WHITE): '♘',
            (chess.BISHOP, chess.WHITE): '♗',
            (chess.ROOK, chess.WHITE): '♖',
            (chess.QUEEN, chess.WHITE): '♕',
            (chess.KING, chess.WHITE): '♔',
            (chess.PAWN, chess.BLACK): '♟',
            (chess.KNIGHT, chess.BLACK): '♞',
            (chess.BISHOP, chess.BLACK): '♝',
            (chess.ROOK, chess.BLACK): '♜',
            (chess.QUEEN, chess.BLACK): '♛',
            (chess.KING, chess.BLACK): '♚',
        }

        for row in range(8):
            for col in range(8):
                display_row = row if not self.flipped else 7 - row
                display_col = col if not self.flipped else 7 - col
                square = chess.square(display_col, 7 - display_row)

                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece_symbols.get((piece.piece_type, piece.color), '')
                    x = col * self.square_size + self.square_size // 2 - 20
                    y = row * self.square_size + self.square_size // 2 + 20

                    # Add shadow for better visibility
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(x + 2, y + 2, symbol)

                    # Draw piece
                    painter.setPen(QColor(255, 255, 255) if piece.color == chess.WHITE else QColor(0, 0, 0))
                    painter.drawText(x, y, symbol)


class ChessAI:
    """
    Chess AI using a trained AlphaZero model.
    """

    def __init__(self, model: AlphaZeroNet, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.model.to(device)
        self.model.eval()

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the best move for the current board position.

        Args:
            board: Current board state

        Returns:
            Best move or None if no legal moves
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Encode board
        board_tensor = self.board_encoder.encode(board, history=[])
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

        # Get model prediction
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)[0]

        # Find best legal move
        best_move = None
        best_prob = -1

        for move in legal_moves:
            move_index = self.move_encoder.encode(move, board)
            prob = policy_probs[move_index].item()

            if prob > best_prob:
                best_prob = prob
                best_move = move

        return best_move


class ChessGameWindow(QMainWindow):
    """
    Main window for the chess game application.
    """

    def __init__(self):
        super().__init__()

        self.board = chess.Board()
        self.ai = None
        self.player_color = chess.WHITE
        self.thinking = False

        self.serializer = PyTorchSerializer(compression=True, encoding='base64')

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Chess vs FL Model")
        self.setGeometry(100, 100, 900, 700)

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QHBoxLayout(main_widget)

        # Left side: Chess board
        board_layout = QVBoxLayout()
        self.chess_board = ChessBoardWidget()
        self.chess_board.square_clicked.connect(self.on_move_made)
        board_layout.addWidget(self.chess_board)

        # Board controls
        board_controls = QHBoxLayout()
        self.flip_button = QPushButton("Flip Board")
        self.flip_button.clicked.connect(self.flip_board)
        board_controls.addWidget(self.flip_button)
        board_layout.addLayout(board_controls)

        main_layout.addLayout(board_layout)

        # Right side: Controls and info
        right_layout = QVBoxLayout()

        # Model selection group
        model_group = QGroupBox("AI Model")
        model_layout = QVBoxLayout()

        self.model_label = QLabel("No model loaded")
        model_layout.addWidget(self.model_label)

        self.load_model_button = QPushButton("Load Model Checkpoint")
        self.load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_button)

        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)

        # Game controls group
        game_group = QGroupBox("Game Controls")
        game_layout = QVBoxLayout()

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Play as:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["White", "Black"])
        self.color_combo.currentIndexChanged.connect(self.change_player_color)
        color_layout.addWidget(self.color_combo)
        game_layout.addLayout(color_layout)

        self.new_game_button = QPushButton("New Game")
        self.new_game_button.clicked.connect(self.new_game)
        game_layout.addWidget(self.new_game_button)

        self.undo_button = QPushButton("Undo Move")
        self.undo_button.clicked.connect(self.undo_move)
        game_layout.addWidget(self.undo_button)

        game_group.setLayout(game_layout)
        right_layout.addWidget(game_group)

        # Game status
        status_group = QGroupBox("Game Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("White to move")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # Move history
        history_group = QGroupBox("Move History")
        history_layout = QVBoxLayout()

        self.move_history = QTextEdit()
        self.move_history.setReadOnly(True)
        self.move_history.setMaximumHeight(200)
        history_layout.addWidget(self.move_history)

        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)

        right_layout.addStretch()

        main_layout.addLayout(right_layout)

        # Initial status update
        self.update_status()

    def load_model(self):
        """Load a model from a checkpoint file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            str(Path("./storage/models")),
            "All Files (*)"
        )

        if not file_path:
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(file_path, map_location='cpu')

            # Extract model state
            if 'model_state' in checkpoint:
                model_state = checkpoint['model_state']
            else:
                model_state = checkpoint

            # Check if model state is serialized
            if isinstance(model_state, dict) and 'serialized_data' in model_state:
                model_state = self.serializer.deserialize(model_state['serialized_data'])

            # Create model and load state
            model = AlphaZeroNet()
            model.load_state_dict(model_state)

            # Create AI
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ai = ChessAI(model, device=device)

            self.model_label.setText(f"Model loaded: {Path(file_path).name}")
            QMessageBox.information(self, "Success", f"Model loaded successfully!\nDevice: {device}")

            # If AI plays first, make a move
            if self.player_color == chess.BLACK and self.board.turn == chess.WHITE:
                QTimer.singleShot(500, self.make_ai_move)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

    def flip_board(self):
        """Flip the board orientation."""
        self.chess_board.flipped = not self.chess_board.flipped
        self.chess_board.update()

    def change_player_color(self, index):
        """Change the player's color."""
        self.player_color = chess.WHITE if index == 0 else chess.BLACK
        self.new_game()

    def new_game(self):
        """Start a new game."""
        self.board = chess.Board()
        self.chess_board.set_board(self.board)
        self.move_history.clear()
        self.update_status()

        # If AI plays first, make a move
        if self.player_color == chess.BLACK and self.ai:
            QTimer.singleShot(500, self.make_ai_move)

    def undo_move(self):
        """Undo the last move (or two moves if playing against AI)."""
        if not self.board.move_stack:
            return

        # Undo player move
        self.board.pop()

        # Undo AI move if present
        if self.board.move_stack and self.ai:
            self.board.pop()

        self.chess_board.set_board(self.board)
        self.update_move_history()
        self.update_status()

    def on_move_made(self, from_square, to_square):
        """Handle a move being made on the board."""
        self.chess_board.set_board(self.board)
        self.update_move_history()
        self.update_status()

        # Check if game is over
        if self.board.is_game_over():
            self.show_game_over()
            return

        # If it's AI's turn, make a move
        if self.ai and self.board.turn != self.player_color:
            QTimer.singleShot(500, self.make_ai_move)

    def make_ai_move(self):
        """Make the AI's move."""
        if not self.ai or self.thinking:
            return

        self.thinking = True
        self.status_label.setText("AI is thinking...")
        QApplication.processEvents()

        try:
            move = self.ai.get_move(self.board)
            if move:
                self.board.push(move)
                self.chess_board.last_move = move
                self.chess_board.set_board(self.board)
                self.update_move_history()
        except Exception as e:
            QMessageBox.warning(self, "AI Error", f"AI failed to make a move:\n{str(e)}")

        self.thinking = False
        self.update_status()

        # Check if game is over
        if self.board.is_game_over():
            self.show_game_over()

    def update_status(self):
        """Update the game status label."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.setText(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.status_label.setText("Stalemate!")
        elif self.board.is_insufficient_material():
            self.status_label.setText("Draw - Insufficient material")
        elif self.board.is_check():
            color = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.setText(f"{color} is in check!")
        else:
            color = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.setText(f"{color} to move")

    def update_move_history(self):
        """Update the move history display."""
        moves = []
        board_copy = chess.Board()

        for i, move in enumerate(self.board.move_stack):
            if i % 2 == 0:
                moves.append(f"{i // 2 + 1}. {board_copy.san(move)}")
            else:
                moves[-1] += f" {board_copy.san(move)}"
            board_copy.push(move)

        self.move_history.setText("\n".join(moves))

    def show_game_over(self):
        """Show game over dialog."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            result = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            result = "Game drawn by stalemate"
        elif self.board.is_insufficient_material():
            result = "Game drawn - Insufficient material"
        elif self.board.is_fifty_moves():
            result = "Game drawn - Fifty move rule"
        elif self.board.is_repetition():
            result = "Game drawn - Threefold repetition"
        else:
            result = "Game over"

        QMessageBox.information(self, "Game Over", result)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set style
    app.setStyle('Fusion')

    window = ChessGameWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
