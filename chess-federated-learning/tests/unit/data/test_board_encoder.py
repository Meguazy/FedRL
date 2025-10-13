"""
Unit tests for board_encoder.py

Tests the BoardEncoder class to ensure correct encoding of chess positions
into 119-plane tensor representation following AlphaZero specification.
"""

import pytest
import numpy as np
import chess
import sys
from pathlib import Path

# Add parent directory to path
chess_fl_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chess_fl_dir))

from data.board_encoder import BoardEncoder


class TestBoardEncoder:
    """Test suite for BoardEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create a BoardEncoder instance for testing."""
        return BoardEncoder(history_length=8)

    @pytest.fixture
    def start_board(self):
        """Create a starting position board."""
        return chess.Board()

    def test_encoder_initialization(self):
        """Test encoder initializes with correct parameters."""
        encoder = BoardEncoder(history_length=8)
        assert encoder.history_length == 8

        encoder_custom = BoardEncoder(history_length=4)
        assert encoder_custom.history_length == 4

    def test_encode_output_shape(self, encoder, start_board):
        """Test that encoded tensor has correct shape (119, 8, 8)."""
        tensor = encoder.encode(start_board)
        assert tensor.shape == (119, 8, 8)

    def test_encode_output_dtype(self, encoder, start_board):
        """Test that encoded tensor is float32."""
        tensor = encoder.encode(start_board)
        assert tensor.dtype == np.float32

    def test_encode_output_range(self, encoder, start_board):
        """Test that all values are in [0, 1] range."""
        tensor = encoder.encode(start_board)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    # Piece plane tests
    def test_piece_planes_starting_position(self, encoder: BoardEncoder, start_board: chess.Board):
        """Test piece encoding for starting position."""
        tensor = encoder.encode(start_board)

        # White pawns (plane 0) - should have 8 pawns
        assert tensor[0].sum() == 8.0

        # White knights (plane 1) - should have 2 knights
        assert tensor[1].sum() == 2.0

        # White bishops (plane 2) - should have 2 bishops
        assert tensor[2].sum() == 2.0

        # White rooks (plane 3) - should have 2 rooks
        assert tensor[3].sum() == 2.0

        # White queen (plane 4) - should have 1 queen
        assert tensor[4].sum() == 1.0

        # White king (plane 5) - should have 1 king
        assert tensor[5].sum() == 1.0

        # Black pawns (plane 6) - should have 8 pawns
        assert tensor[6].sum() == 8.0

        # Black pieces follow same pattern
        assert tensor[7].sum() == 2.0  # knights
        assert tensor[8].sum() == 2.0  # bishops
        assert tensor[9].sum() == 2.0  # rooks
        assert tensor[10].sum() == 1.0  # queen
        assert tensor[11].sum() == 1.0  # king

    def test_piece_planes_after_moves(self, encoder):
        """Test piece encoding after some moves."""
        board = chess.Board()
        board.push_san("e4")  # Pawn moves
        board.push_san("e5")
        board.push_san("Nf3")  # Knight moves

        tensor = encoder.encode(board)

        # White pawns should still be 8 (one moved but still counted)
        assert tensor[0].sum() == 8.0

        # Black pawns should still be 8
        assert tensor[6].sum() == 8.0

    def test_piece_planes_after_capture(self, encoder):
        """Test piece encoding after a capture."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("d5")
        board.push_san("exd5")  # Pawn captures pawn

        tensor = encoder.encode(board)

        # White should have 8 pawns
        assert tensor[0].sum() == 8.0

        # Black should have 7 pawns (one captured)
        assert tensor[6].sum() == 7.0

    def test_piece_position_encoding(self, encoder):
        """Test that pieces are encoded at correct squares."""
        board = chess.Board()
        tensor = encoder.encode(board)

        # White king should be on e1 (square 4)
        # e1 = rank 0, file 4
        assert tensor[5, 0, 4] == 1.0

        # Black king should be on e8 (square 60)
        # e8 = rank 7, file 4
        assert tensor[11, 7, 4] == 1.0

        # White rook on a1 (square 0)
        # a1 = rank 0, file 0
        assert tensor[3, 0, 0] == 1.0

    # History plane tests
    def test_history_planes_no_history(self, encoder, start_board):
        """Test that without history, current position is repeated."""
        tensor = encoder.encode(start_board, history=None)

        # Current position (planes 0-11)
        current = tensor[0:12]

        # Position 1 move ago (planes 12-23) should be identical
        history_1 = tensor[12:24]

        assert np.allclose(current, history_1)

    def test_history_planes_with_history(self, encoder):
        """Test encoding with actual game history."""
        board = chess.Board()
        history = []

        # Make some moves and save history
        board.push_san("e4")
        history.insert(0, chess.Board())  # Starting position

        board.push_san("e5")
        prev = chess.Board()
        prev.push_san("e4")
        history.insert(0, prev.copy())

        tensor = encoder.encode(board, history=history)

        # Current and history should be different
        current = tensor[0:12]
        history_1 = tensor[12:24]

        assert not np.allclose(current, history_1)

    def test_history_planes_padding(self, encoder):
        """Test that insufficient history is padded with current position."""
        board = chess.Board()
        board.push_san("e4")

        # Provide only 2 history positions (need 7 for full 8-position history)
        short_history = [chess.Board()]  # Only starting position

        tensor = encoder.encode(board, history=short_history)

        # Should not raise error and should have correct shape
        assert tensor.shape == (119, 8, 8)

    # Repetition tests
    def test_repetition_encoding_no_repetition(self, encoder, start_board):
        """Test repetition planes when there's no repetition."""
        tensor = encoder.encode(start_board)

        # Planes 96-97 should be all zeros (no repetition)
        assert tensor[96].sum() == 0.0
        assert tensor[97].sum() == 0.0

    # Castling rights tests
    def test_castling_rights_starting_position(self, encoder, start_board):
        """Test castling rights encoding at start."""
        tensor = encoder.encode(start_board)

        # All castling rights should be available
        assert tensor[98, 0, 0] == 1.0  # White kingside
        assert tensor[99, 0, 0] == 1.0  # White queenside
        assert tensor[100, 0, 0] == 1.0  # Black kingside
        assert tensor[101, 0, 0] == 1.0  # Black queenside

    def test_castling_rights_after_king_move(self, encoder):
        """Test castling rights lost after king moves."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Ke2")  # White king moves, loses castling

        tensor = encoder.encode(board)

        # White should lose both castling rights
        assert tensor[98, 0, 0] == 0.0  # White kingside
        assert tensor[99, 0, 0] == 0.0  # White queenside

        # Black should still have castling rights
        assert tensor[100, 0, 0] == 1.0  # Black kingside
        assert tensor[101, 0, 0] == 1.0  # Black queenside

    def test_castling_rights_after_rook_move(self, encoder):
        """Test castling rights lost after rook moves."""
        board = chess.Board()
        board.push_san("h4")  # Pawn move
        board.push_san("h5")
        board.push_san("Rh3")  # White h-rook moves

        tensor = encoder.encode(board)

        # White should lose kingside castling
        assert tensor[98, 0, 0] == 0.0  # White kingside lost
        assert tensor[99, 0, 0] == 1.0  # White queenside still available

    # Color to move tests
    def test_color_to_move_white(self, encoder, start_board):
        """Test color plane when white to move."""
        tensor = encoder.encode(start_board)

        # Plane 102 should be all 1s (white to move)
        assert tensor[102].sum() == 64.0
        assert tensor[102, 0, 0] == 1.0

    def test_color_to_move_black(self, encoder):
        """Test color plane when black to move."""
        board = chess.Board()
        board.push_san("e4")  # White moves, now black's turn

        tensor = encoder.encode(board)

        # Plane 102 should be all 0s (black to move)
        assert tensor[102].sum() == 0.0
        assert tensor[102, 0, 0] == 0.0

    # Move count tests
    def test_move_count_starting_position(self, encoder, start_board):
        """Test move count at start (move 1)."""
        tensor = encoder.encode(start_board)

        # Plane 103 should have normalized move count
        # Move 1 / 100 = 0.01
        expected = 1.0 / 100.0
        assert np.allclose(tensor[103, 0, 0], expected)

    def test_move_count_after_moves(self, encoder):
        """Test move count increases correctly."""
        board = chess.Board()

        # Make 10 moves (5 full moves)
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "d6", "Bg5", "f6"]
        for move in moves:
            board.push_san(move)

        tensor = encoder.encode(board)

        # Should be at move 6
        expected = 6.0 / 100.0
        assert np.allclose(tensor[103, 0, 0], expected, atol=0.01)

    # No-progress count tests
    def test_no_progress_starting_position(self, encoder, start_board):
        """Test no-progress planes at start (0 halfmoves)."""
        tensor = encoder.encode(start_board)

        # Planes 104-118 should all be zeros (no halfmoves yet)
        for i in range(104, 119):
            assert tensor[i].sum() == 0.0

    def test_no_progress_after_pawn_move(self, encoder):
        """Test no-progress resets after pawn move."""
        board = chess.Board()
        board.push_san("e4")  # Pawn move resets counter

        tensor = encoder.encode(board)

        # Should still be 0 (just reset)
        for i in range(104, 119):
            assert tensor[i].sum() == 0.0

    def test_no_progress_thermometer_encoding(self, encoder):
        """Test thermometer encoding for no-progress count."""
        board = chess.Board()

        # Make non-pawn, non-capture moves to increase halfmove clock
        board.push_san("e4")  # Resets to 0
        board.push_san("Nf6")  # +1
        board.push_san("Nf3")  # +2
        board.push_san("Nc6")  # +3
        board.push_san("Nc3")  # +4

        tensor = encoder.encode(board)

        # Halfmove clock should be 4
        # 4 // 4 = 1, so plane 104 should be filled
        assert tensor[104].sum() == 64.0
        # Plane 105 should be empty
        assert tensor[105].sum() == 0.0

    # Batch encoding tests
    def test_batch_encode_shape(self, encoder):
        """Test batch encoding produces correct shape."""
        boards = [chess.Board() for _ in range(4)]

        batch = encoder.encode_batch(boards)

        assert batch.shape == (4, 119, 8, 8)

    def test_batch_encode_different_positions(self, encoder):
        """Test batch encoding with different positions."""
        boards = [chess.Board() for _ in range(3)]
        boards[0].push_san("e4")
        boards[1].push_san("d4")
        boards[2].push_san("Nf3")

        batch = encoder.encode_batch(boards)

        # Each should be different
        assert not np.allclose(batch[0], batch[1])
        assert not np.allclose(batch[1], batch[2])

    def test_batch_encode_with_histories(self, encoder):
        """Test batch encoding with history for each board."""
        boards = [chess.Board(), chess.Board()]
        boards[0].push_san("e4")
        boards[1].push_san("d4")

        histories = [[chess.Board()], [chess.Board()]]

        batch = encoder.encode_batch(boards, histories)

        assert batch.shape == (2, 119, 8, 8)

    # Edge case tests
    def test_empty_board(self, encoder):
        """Test encoding a board with no pieces (edge case)."""
        board = chess.Board()
        board.clear()  # Remove all pieces

        tensor = encoder.encode(board)

        # Should not crash and should have correct shape
        assert tensor.shape == (119, 8, 8)
        # Piece planes should be empty
        assert tensor[0:12].sum() == 0.0

    def test_many_captures(self, encoder):
        """Test encoding after many captures."""
        board = chess.Board()
        # Play a game with captures
        moves = ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+"]
        for move in moves:
            board.push_san(move)

        tensor = encoder.encode(board)

        assert tensor.shape == (119, 8, 8)
        # Should have fewer pieces than start
        total_pieces = tensor[0:12].sum()
        assert total_pieces < 32.0  # Less than starting 32 pieces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
