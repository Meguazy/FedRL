"""
Unit tests for move_encoder.py

Tests the MoveEncoder class to ensure correct encoding and decoding of chess moves
using the AlphaZero 8×8×73 move encoding scheme (4672 total actions).
"""

import pytest
import chess
import sys
from pathlib import Path

# Add parent directory to path
chess_fl_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chess_fl_dir))

from data.move_encoder import MoveEncoder


class TestMoveEncoder:
    """Test suite for MoveEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create a MoveEncoder instance for testing."""
        return MoveEncoder()

    @pytest.fixture
    def start_board(self):
        """Create a starting position board."""
        return chess.Board()

    # Initialization tests
    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder is not None
        assert len(encoder.QUEEN_DIRECTIONS) == 8
        assert len(encoder.KNIGHT_MOVES) == 8
        assert len(encoder.UNDERPROMO_DIRECTIONS_WHITE) == 3
        assert len(encoder.UNDERPROMO_DIRECTIONS_BLACK) == 3

    def test_queen_directions_definition(self, encoder):
        """Test that queen directions cover all 8 compass directions."""
        expected_directions = [
            (-1, 0),   # North
            (-1, 1),   # NorthEast
            (0, 1),    # East
            (1, 1),    # SouthEast
            (1, 0),    # South
            (1, -1),   # SouthWest
            (0, -1),   # West
            (-1, -1),  # NorthWest
        ]
        assert encoder.QUEEN_DIRECTIONS == expected_directions

    def test_knight_moves_definition(self, encoder):
        """Test that knight moves are correctly defined."""
        expected_moves = [
            (-2, 1),   # NNE
            (-1, 2),   # ENE
            (1, 2),    # ESE
            (2, 1),    # SSE
            (2, -1),   # SSW
            (1, -2),   # WSW
            (-1, -2),  # WNW
            (-2, -1),  # NNW
        ]
        assert encoder.KNIGHT_MOVES == expected_moves

    # Basic encoding tests
    def test_encode_pawn_move_single_step(self, encoder, start_board):
        """Test encoding a single-step pawn move."""
        move = chess.Move.from_uci("e2e3")
        index = encoder.encode(move, start_board)
        
        # Should be a valid index
        assert 0 <= index < 4672
        
        # Decode and verify
        decoded = encoder.decode(index, start_board)
        assert decoded == move

    def test_encode_pawn_move_double_step(self, encoder, start_board):
        """Test encoding a double-step pawn move."""
        move = chess.Move.from_uci("e2e4")
        index = encoder.encode(move, start_board)
        
        assert 0 <= index < 4672
        decoded = encoder.decode(index, start_board)
        assert decoded == move

    def test_encode_knight_move(self, encoder, start_board):
        """Test encoding a knight move."""
        move = chess.Move.from_uci("g1f3")
        index = encoder.encode(move, start_board)
        
        assert 0 <= index < 4672
        decoded = encoder.decode(index, start_board)
        assert decoded == move

    def test_encode_all_starting_moves(self, encoder, start_board):
        """Test encoding all legal moves from starting position."""
        for move in start_board.legal_moves:
            index = encoder.encode(move, start_board)
            assert 0 <= index < 4672
            decoded = encoder.decode(index, start_board)
            assert decoded == move

    # Queen-style move tests
    def test_encode_queen_move_north(self, encoder):
        """Test encoding a north (vertical up) move."""
        board = chess.Board("8/8/8/8/8/8/R7/8 w - - 0 1")
        move = chess.Move.from_uci("a2a4")  # Rook north 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_south(self, encoder):
        """Test encoding a south (vertical down) move."""
        board = chess.Board("R7/8/8/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("a8a6")  # Rook south 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_east(self, encoder):
        """Test encoding an east (horizontal right) move."""
        board = chess.Board("8/8/8/8/8/8/R7/8 w - - 0 1")
        move = chess.Move.from_uci("a2c2")  # Rook east 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_west(self, encoder):
        """Test encoding a west (horizontal left) move."""
        board = chess.Board("8/8/8/8/8/8/7R/8 w - - 0 1")
        move = chess.Move.from_uci("h2f2")  # Rook west 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_diagonal_ne(self, encoder):
        """Test encoding a northeast diagonal move."""
        board = chess.Board("8/8/8/8/8/8/B7/8 w - - 0 1")
        move = chess.Move.from_uci("a2c4")  # Bishop NE 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_diagonal_se(self, encoder):
        """Test encoding a southeast diagonal move."""
        board = chess.Board("B7/8/8/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("a8c6")  # Bishop SE 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_diagonal_sw(self, encoder):
        """Test encoding a southwest diagonal move."""
        board = chess.Board("7B/8/8/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("h8f6")  # Bishop SW 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_diagonal_nw(self, encoder):
        """Test encoding a northwest diagonal move."""
        board = chess.Board("8/8/8/8/8/8/7B/8 w - - 0 1")
        move = chess.Move.from_uci("h2f4")  # Bishop NW 2 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_queen_move_max_distance(self, encoder):
        """Test encoding queen-style moves with maximum distance (7 squares)."""
        board = chess.Board("Q7/8/8/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci("a8a1")  # Queen south 7 squares
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    def test_encode_king_move(self, encoder):
        """Test encoding a single-step king move (queen-style, distance 1)."""
        board = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e1e2")  # King north 1 square
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move

    # Knight move tests
    def test_encode_all_knight_moves(self, encoder):
        """Test encoding all 8 possible knight move types."""
        # Place knight in center of board
        board = chess.Board("8/8/8/3N4/8/8/8/8 w - - 0 1")
        knight_square = chess.D5
        
        # All possible knight moves from d5
        knight_targets = [
            chess.F6,  # NNE
            chess.F4,  # ENE (actually ESE from d5)
            chess.E3,  # ESE (actually SSE from d5)
            chess.C3,  # SSE (actually SSW from d5)
            chess.B4,  # SSW (actually WSW from d5)
            chess.B6,  # WSW (actually WNW from d5)
            chess.C7,  # WNW (actually NNW from d5)
            chess.E7,  # NNW (actually NNE from d5)
        ]
        
        for target in knight_targets:
            move = chess.Move(knight_square, target)
            if move in board.legal_moves:
                index = encoder.encode(move, board)
                assert 0 <= index < 4672
                decoded = encoder.decode(index, board)
                assert decoded == move

    def test_knight_move_from_g1(self, encoder, start_board):
        """Test knight move from starting position."""
        move = chess.Move.from_uci("g1f3")
        index = encoder.encode(move, start_board)
        
        # Verify it's encoded as a knight move (plane 56-63)
        plane = index % 73
        assert 56 <= plane < 64
        
        decoded = encoder.decode(index, start_board)
        assert decoded == move

    def test_knight_move_from_b1(self, encoder, start_board):
        """Test different knight starting square."""
        move = chess.Move.from_uci("b1c3")
        index = encoder.encode(move, start_board)
        
        plane = index % 73
        assert 56 <= plane < 64
        
        decoded = encoder.decode(index, start_board)
        assert decoded == move

    # Promotion tests
    def test_encode_queen_promotion_forward(self, encoder):
        """Test encoding a forward queen promotion."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci("a7a8q")
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        # Decoder should add queen promotion for pawn reaching last rank
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square
        assert decoded.promotion == chess.QUEEN

    def test_encode_queen_promotion_capture_left(self, encoder):
        """Test encoding a left-diagonal queen promotion with capture."""
        board = chess.Board("1n6/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci("a7b8q")
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square
        assert decoded.promotion == chess.QUEEN

    def test_encode_knight_underpromotion_forward(self, encoder):
        """Test encoding a forward knight underpromotion."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci("a7a8n")
        index = encoder.encode(move, board)
        
        # Verify it's encoded as underpromotion (plane 64-72)
        plane = index % 73
        assert 64 <= plane < 73
        
        decoded = encoder.decode(index, board)
        assert decoded == move
        assert decoded.promotion == chess.KNIGHT

    def test_encode_bishop_underpromotion(self, encoder):
        """Test encoding a bishop underpromotion."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci("a7a8b")
        index = encoder.encode(move, board)
        
        plane = index % 73
        assert 64 <= plane < 73
        
        decoded = encoder.decode(index, board)
        assert decoded == move
        assert decoded.promotion == chess.BISHOP

    def test_encode_rook_underpromotion(self, encoder):
        """Test encoding a rook underpromotion."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci("a7a8r")
        index = encoder.encode(move, board)
        
        plane = index % 73
        assert 64 <= plane < 73
        
        decoded = encoder.decode(index, board)
        assert decoded == move
        assert decoded.promotion == chess.ROOK

    def test_encode_black_queen_promotion(self, encoder):
        """Test encoding a black pawn queen promotion."""
        board = chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")
        move = chess.Move.from_uci("a2a1q")
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        assert decoded == move
        assert decoded.promotion == chess.QUEEN

    def test_encode_black_knight_underpromotion(self, encoder):
        """Test encoding a black pawn knight underpromotion."""
        board = chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")
        move = chess.Move.from_uci("a2a1n")
        index = encoder.encode(move, board)
        
        plane = index % 73
        assert 64 <= plane < 73
        
        decoded = encoder.decode(index, board)
        assert decoded == move
        assert decoded.promotion == chess.KNIGHT

    def test_encode_all_promotion_types(self, encoder):
        """Test encoding all promotion piece types (q, n, b, r)."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        
        promotion_pieces = [
            (chess.QUEEN, 'q'),
            (chess.KNIGHT, 'n'),
            (chess.BISHOP, 'b'),
            (chess.ROOK, 'r'),
        ]
        
        for piece_type, uci_char in promotion_pieces:
            move = chess.Move.from_uci(f"a7a8{uci_char}")
            index = encoder.encode(move, board)
            decoded = encoder.decode(index, board)
            assert decoded == move
            assert decoded.promotion == piece_type

    def test_encode_underpromotion_all_directions(self, encoder):
        """Test underpromotions in all three directions (left, forward, right)."""
        # White pawn ready to promote with pieces to capture
        board = chess.Board("1n1n4/1P6/8/8/8/8/8/K6k w - - 0 1")
        
        moves = [
            chess.Move.from_uci("b7a8n"),  # Left diagonal
            chess.Move.from_uci("b7b8n"),  # Forward
            chess.Move.from_uci("b7c8n"),  # Right diagonal
        ]
        
        for move in moves:
            if move in board.legal_moves:
                index = encoder.encode(move, board)
                plane = index % 73
                assert 64 <= plane < 73
                decoded = encoder.decode(index, board)
                assert decoded == move

    # Roundtrip encoding tests
    def test_roundtrip_all_legal_moves_starting_position(self, encoder, start_board):
        """Test encode-decode roundtrip for all starting moves."""
        for move in start_board.legal_moves:
            index = encoder.encode(move, start_board)
            decoded = encoder.decode(index, start_board)
            assert decoded == move, f"Failed roundtrip for move {move.uci()}"

    def test_roundtrip_mid_game_position(self, encoder):
        """Test roundtrip encoding for a mid-game position."""
        board = chess.Board()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"]
        for san_move in moves:
            board.push_san(san_move)
        
        for move in board.legal_moves:
            index = encoder.encode(move, board)
            decoded = encoder.decode(index, board)
            assert decoded == move

    def test_roundtrip_complex_position(self, encoder):
        """Test roundtrip with a complex tactical position."""
        # Position with many piece types and move options
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        
        for move in board.legal_moves:
            index = encoder.encode(move, board)
            decoded = encoder.decode(index, board)
            assert decoded == move

    # Plane index tests
    def test_queen_move_plane_range(self, encoder):
        """Test that queen-style moves produce plane indices 0-55."""
        board = chess.Board("R7/8/8/8/8/8/8/8 w - - 0 1")
        
        # Test various distances and directions
        test_moves = [
            "a8a1",  # 7 squares south
            "a8h8",  # 7 squares east
            "a8h1",  # 7 squares southeast diagonal
        ]
        
        for uci in test_moves:
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert 0 <= plane < 56, f"Queen move {uci} should be in plane 0-55, got {plane}"

    def test_knight_move_plane_range(self, encoder):
        """Test that knight moves produce plane indices 56-63."""
        board = chess.Board("8/8/8/3N4/8/8/8/8 w - - 0 1")
        
        for move in board.legal_moves:
            index = encoder.encode(move, board)
            plane = index % 73
            assert 56 <= plane < 64, f"Knight move should be in plane 56-63, got {plane}"

    def test_underpromotion_plane_range(self, encoder):
        """Test that underpromotions produce plane indices 64-72."""
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        
        underpromo_moves = [
            chess.Move.from_uci("a7a8n"),
            chess.Move.from_uci("a7a8b"),
            chess.Move.from_uci("a7a8r"),
        ]
        
        for move in underpromo_moves:
            index = encoder.encode(move, board)
            plane = index % 73
            assert 64 <= plane < 73, f"Underpromotion should be in plane 64-72, got {plane}"

    # From-square encoding tests
    def test_from_square_encoding(self, encoder, start_board):
        """Test that from-square is correctly encoded in the index."""
        move = chess.Move.from_uci("e2e4")
        index = encoder.encode(move, start_board)
        
        # Extract from_square from index
        from_row = index // (8 * 73)
        from_col = (index % (8 * 73)) // 73
        from_square = from_row * 8 + from_col
        
        assert from_square == move.from_square

    def test_from_square_all_squares(self, encoder):
        """Test encoding moves from all 64 board squares."""
        # Create positions with pieces on each square and test encoding
        for square in range(64):
            row, col = divmod(square, 8)
            # Place a queen on the square (can move in all directions)
            board = chess.Board(f"8/8/8/8/8/8/8/8 w - - 0 1")
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            
            # Test at least one legal move from this square
            legal_moves = [m for m in board.legal_moves if m.from_square == square]
            if legal_moves:
                move = legal_moves[0]
                index = encoder.encode(move, board)
                from_row_decoded = index // (8 * 73)
                from_col_decoded = (index % (8 * 73)) // 73
                assert from_row_decoded == row
                assert from_col_decoded == col

    # Index range tests
    def test_all_indices_in_valid_range(self, encoder, start_board):
        """Test that all encoded moves produce valid indices."""
        for move in start_board.legal_moves:
            index = encoder.encode(move, start_board)
            assert 0 <= index < 4672

    def test_index_calculation_formula(self, encoder):
        """Test the index calculation formula: from_row*8*73 + from_col*73 + plane."""
        board = chess.Board("8/8/8/8/8/8/R7/8 w - - 0 1")
        move = chess.Move.from_uci("a2a4")  # From a2 to a4
        
        index = encoder.encode(move, board)
        
        # Manual calculation
        from_row, from_col = 1, 0  # a2 = row 1, col 0
        # a2->a4: row 1->3, delta (+2,0) = South direction (index 4), distance 2
        # South 2 squares: direction 4 (South), distance 2
        plane = 4 * 7 + (2 - 1)  # = 29
        expected_index = from_row * 8 * 73 + from_col * 73 + plane
        
        assert index == expected_index

    def test_index_calculation_formula_knight_move(self, encoder):
        """Test index formula for a knight move."""
        board = chess.Board("8/8/8/8/8/8/8/N6k w - - 0 1")
        move = chess.Move.from_uci("a1c2")  # Knight move ENE
        
        index = encoder.encode(move, board)
        
        # Manual calculation
        from_row, from_col = 0, 0  # a1 = row 0, col 0
        # a1->c2: row 0->1, col 0->2, delta (1, 2) = ENE knight move (index 2)
        plane = 56 + 2  # ENE is index 2 in KNIGHT_MOVES
        expected_index = from_row * 8 * 73 + from_col * 73 + plane
        
        assert index == expected_index
        assert plane == index % 73

    def test_index_calculation_formula_diagonal_move(self, encoder):
        """Test index formula for a diagonal move."""
        board = chess.Board("8/8/8/8/8/8/8/B6k w - - 0 1")
        move = chess.Move.from_uci("a1d4")  # 3 squares NE
        
        index = encoder.encode(move, board)
        
        # Manual calculation
        from_row, from_col = 0, 0  # a1 = row 0, col 0
        # a1->d4: row 0->3, col 0->3, delta (+3,+3) = SE direction (index 3), distance 3
        # SE 3 squares: direction 3, distance 3
        plane = 3 * 7 + (3 - 1)  # = 23
        expected_index = from_row * 8 * 73 + from_col * 73 + plane
        
        assert index == expected_index

    def test_index_calculation_formula_different_from_square(self, encoder):
        """Test index formula varies correctly with different from_square."""
        board = chess.Board("8/8/8/8/8/8/R7/R6k w - - 0 1")
        
        # Same move type (1 square south) from different squares
        move1 = chess.Move.from_uci("a1a2")  # from row 0
        move2 = chess.Move.from_uci("a2a3")  # from row 1
        
        index1 = encoder.encode(move1, board)
        index2 = encoder.encode(move2, board)
        
        # Both should have same plane (South 1 square = direction 4, distance 1)
        plane = 4 * 7 + 0  # = 28
        assert index1 % 73 == plane
        assert index2 % 73 == plane
        
        # But different indices due to different from_square
        expected1 = 0 * 8 * 73 + 0 * 73 + plane  # from a1
        expected2 = 1 * 8 * 73 + 0 * 73 + plane  # from a2
        
        assert index1 == expected1
        assert index2 == expected2
        assert index2 - index1 == 8 * 73  # Difference is exactly one row

    # Error handling tests
    def test_encode_invalid_move_zero_distance(self, encoder):
        """Test that encoding a zero-distance move raises ValueError."""
        board = chess.Board("8/8/8/8/8/8/R7/8 w - - 0 1")
        # Create an invalid move (same from and to square)
        move = chess.Move(chess.A2, chess.A2)
        
        with pytest.raises(ValueError, match="zero distance"):
            encoder.encode(move, board)

    def test_decode_invalid_index_too_low(self, encoder, start_board):
        """Test that decoding index < 0 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode(-1, start_board)

    def test_decode_invalid_index_too_high(self, encoder, start_board):
        """Test that decoding index >= 4672 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode(4672, start_board)

    def test_decode_invalid_index_way_too_high(self, encoder, start_board):
        """Test that decoding very large index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode(10000, start_board)

    # Special move tests
    def test_encode_castling_kingside(self, encoder, start_board):
        """Test encoding kingside castling."""
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1g1")  # Kingside castle
        
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        
        # Should encode as a 2-square king move to the east
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square

    def test_encode_castling_queenside(self, encoder):
        """Test encoding queenside castling."""
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1c1")  # Queenside castle
        
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        
        # Should encode as a 2-square king move to the west
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square

    def test_encode_en_passant(self, encoder):
        """Test encoding en passant capture."""
        # Set up en passant position
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")
        move = chess.Move.from_uci("e5d6")  # En passant capture
        
        index = encoder.encode(move, board)
        decoded = encoder.decode(index, board)
        
        assert decoded == move

    # Encoding consistency tests
    def test_same_move_different_positions_same_encoding(self, encoder):
        """Test that the same move from/to squares encodes to same index regardless of board state."""
        move = chess.Move.from_uci("e2e4")
        
        board1 = chess.Board()
        board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        index1 = encoder.encode(move, board1)
        index2 = encoder.encode(move, board2)
        
        # Same move should produce same index
        assert index1 == index2

    def test_different_from_squares_different_indices(self, encoder):
        """Test that moves from different squares produce different indices."""
        board = chess.Board()
        
        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("d2d4")
        
        index1 = encoder.encode(move1, board)
        index2 = encoder.encode(move2, board)
        
        assert index1 != index2

    # Edge case board positions
    def test_encode_from_corner_a1(self, encoder):
        """Test encoding moves from corner square a1."""
        board = chess.Board("8/8/8/8/8/8/8/R6k w - - 0 1")
        
        for move in board.legal_moves:
            if move.from_square == chess.A1:
                index = encoder.encode(move, board)
                decoded = encoder.decode(index, board)
                assert decoded == move

    def test_encode_from_corner_h8(self, encoder):
        """Test encoding moves from corner square h8."""
        board = chess.Board("7r/8/8/8/8/8/8/K7 b - - 0 1")
        
        for move in board.legal_moves:
            if move.from_square == chess.H8:
                index = encoder.encode(move, board)
                decoded = encoder.decode(index, board)
                assert decoded == move

    def test_encode_center_square(self, encoder):
        """Test encoding moves from center squares d4/e4."""
        board = chess.Board("8/8/8/8/3Q4/8/8/K6k w - - 0 1")
        
        for move in board.legal_moves:
            if move.from_square in [chess.D4, chess.E4]:
                index = encoder.encode(move, board)
                decoded = encoder.decode(index, board)
                assert decoded == move

    # Comprehensive coverage test
    def test_action_space_coverage(self, encoder):
        """Test that we can encode moves to various parts of the action space."""
        # Collect a variety of encoded indices
        indices = set()
        
        # Starting position
        board = chess.Board()
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Mid-game position
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Position with queen in center
        board = chess.Board("8/8/8/3Q4/8/8/8/K6k w - - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Position with knight in center
        board = chess.Board("8/8/8/3N4/8/8/8/K6k w - - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Position with promotions
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Position with rook in corner (tests long-range moves)
        board = chess.Board("8/8/8/8/8/8/8/R6k w - - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Position with bishop in corner (tests diagonal moves)
        board = chess.Board("8/8/8/8/8/8/8/B6k w - - 0 1")
        for move in board.legal_moves:
            indices.add(encoder.encode(move, board))
        
        # Should have reasonable coverage of action space
        # With these diverse positions, we expect at least 100 unique actions
        assert len(indices) >= 100, f"Only covered {len(indices)} unique actions"

    # Specific plane calculation tests
    def test_queen_plane_calculation_all_directions(self, encoder):
        """Test plane calculation for queen moves in all 8 directions."""
        test_cases = [
            ((-1, 0), 1, 0),   # North, 1 square: direction 0, plane 0
            ((-1, 0), 7, 6),   # North, 7 squares: direction 0, plane 6
            ((-1, 1), 1, 7),   # NorthEast, 1 square: direction 1, plane 7
            ((0, 1), 1, 14),   # East, 1 square: direction 2, plane 14
            ((1, 1), 1, 21),   # SouthEast, 1 square: direction 3, plane 21
            ((1, 0), 1, 28),   # South, 1 square: direction 4, plane 28
            ((1, -1), 1, 35),  # SouthWest, 1 square: direction 5, plane 35
            ((0, -1), 1, 42),  # West, 1 square: direction 6, plane 42
            ((-1, -1), 1, 49), # NorthWest, 1 square: direction 7, plane 49
        ]
        
        for delta, distance, expected_plane in test_cases:
            plane = encoder._encode_queen_move(delta[0] * distance, delta[1] * distance)
            assert plane == expected_plane, f"Failed for delta {delta}, distance {distance}"

    # Additional comprehensive encoding tests
    def test_all_queen_directions_all_distances(self, encoder):
        """Test all 8 queen directions with all 7 distances systematically."""
        board = chess.Board("8/8/8/3Q4/8/8/8/K6k w - - 0 1")
        
        # Test each direction with each distance
        # NOTE: Direction naming follows AlphaZero's convention (standard array indexing):
        #   - Row increases (0→7) = "South" (moves toward rank 8, visually "up")
        #   - Row decreases (7→0) = "North" (moves toward rank 1, visually "down")
        # This is counterintuitive but matches the paper and ensures compatibility.
        test_moves = [
            # North (direction 0, row decreases)
            ("d5d4", 0), ("d5d3", 1), ("d5d1", 3),  # distances 1, 2, 4
            # NorthEast (direction 1, row decreases, col increases)
            ("d5e4", 7), ("d5f3", 8), ("d5h1", 10),  # distances 1, 2, 4
            # East (direction 2)
            ("d5e5", 14), ("d5f5", 15), ("d5h5", 17),  # distances 1, 2, 4
            # SouthEast (direction 3, row increases, col increases)
            ("d5e6", 21), ("d5f7", 22), ("d5g8", 23),  # distances 1, 2, 3
            # South (direction 4, row increases)
            ("d5d6", 28), ("d5d7", 29), ("d5d8", 30),  # distances 1, 2, 3
            # SouthWest (direction 5, row increases, col decreases)
            ("d5c6", 35), ("d5b7", 36), ("d5a8", 37),  # distances 1, 2, 3
            # West (direction 6)
            ("d5c5", 42), ("d5b5", 43), ("d5a5", 44),  # distances 1, 2, 3
            # NorthWest (direction 7, row decreases, col decreases)
            ("d5c4", 49), ("d5b3", 50), ("d5a2", 51),  # distances 1, 2, 3
        ]
        
        for uci, expected_plane in test_moves:
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"Move {uci}: expected plane {expected_plane}, got {plane}"
            
            # Verify roundtrip
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for {uci}"

    def test_all_knight_moves_from_center(self, encoder):
        """Test all 8 knight moves from a central position."""
        board = chess.Board("8/8/8/3N4/8/8/8/K6k w - - 0 1")
        
        # All 8 knight moves from d5 (row 4, col 3) with correct planes
        knight_moves = [
            ("d5e3", 56),  # delta (-2,  1) → NNE → plane 56
            ("d5f4", 57),  # delta (-1,  2) → ENE → plane 57
            ("d5f6", 58),  # delta ( 1,  2) → ESE → plane 58
            ("d5e7", 59),  # delta ( 2,  1) → SSE → plane 59
            ("d5c7", 60),  # delta ( 2, -1) → SSW → plane 60
            ("d5b6", 61),  # delta ( 1, -2) → WSW → plane 61
            ("d5b4", 62),  # delta (-1, -2) → WNW → plane 62
            ("d5c3", 63),  # delta (-2, -1) → NNW → plane 63
        ]
        
        for uci, expected_plane in knight_moves:
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"Knight move {uci}: expected plane {expected_plane}, got {plane}"
            
            # Verify roundtrip
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for knight move {uci}"

    def test_all_underpromotion_types_and_directions(self, encoder):
        """Test all 9 underpromotion combinations (3 pieces × 3 directions)."""
        # White underpromotions from a7
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        
        white_underpromos = [
            ("a7a8n", 65),  # Knight forward
            ("a7b8n", 66),  # Knight right-capture
            ("a7a8b", 68),  # Bishop forward
            ("a7b8b", 69),  # Bishop right-capture
            ("a7a8r", 71),  # Rook forward
            ("a7b8r", 72),  # Rook right-capture
        ]
        
        for uci, expected_plane in white_underpromos:
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"White underpromo {uci}: expected plane {expected_plane}, got {plane}"
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for {uci}"
        
        # Black underpromotions from a2
        board = chess.Board("K7/8/8/8/8/8/p7/7k b - - 0 1")
        
        black_underpromos = [
            ("a2b1n", 64),  # Knight left-capture (from black's perspective, delta (-1, 1))
            ("a2a1n", 65),  # Knight forward (delta (-1, 0))
            ("a2b1b", 67),  # Bishop left-capture
            ("a2a1b", 68),  # Bishop forward
            ("a2b1r", 70),  # Rook left-capture
            ("a2a1r", 71),  # Rook forward
        ]
        
        for uci, expected_plane in black_underpromos:
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"Black underpromo {uci}: expected plane {expected_plane}, got {plane}"
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for {uci}"

    def test_boundary_squares_encoding(self, encoder):
        """Test moves from all corner and edge squares."""
        # Test corner squares
        corner_tests = [
            ("8/8/8/8/8/8/8/R6k w - - 0 1", "a1a2", 0, 0),    # a1 corner
            ("8/8/8/8/8/8/8/7R w - - 0 1", "h1h2", 0, 7),    # h1 corner
            ("R7/8/8/8/8/8/8/7k w - - 0 1", "a8a7", 7, 0),    # a8 corner
            ("7R/8/8/8/8/8/8/k7 w - - 0 1", "h8h7", 7, 7),    # h8 corner
        ]
        
        for fen, uci, expected_row, expected_col in corner_tests:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            
            # Verify from_square encoding
            from_row = index // (8 * 73)
            from_col = (index % (8 * 73)) // 73
            assert from_row == expected_row, f"Move {uci}: expected row {expected_row}, got {from_row}"
            assert from_col == expected_col, f"Move {uci}: expected col {expected_col}, got {from_col}"
            
            # Verify roundtrip
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for corner move {uci}"

    def test_maximum_distance_moves(self, encoder):
        """Test moves with maximum distance (7 squares) in each direction."""
        max_distance_tests = [
            # Rook from a1 to a8 (7 squares south)
            ("8/8/8/8/8/8/8/R6k w - - 0 1", "a1a8", 34),  # South, distance 7
            # Rook from a1 to h1 (7 squares east)
            ("8/8/8/8/8/8/8/R6k w - - 0 1", "a1h1", 20),  # East, distance 7
            # Bishop from a1 to h8 (7 squares SE)
            ("8/8/8/8/8/8/8/B6k w - - 0 1", "a1h8", 27),  # SouthEast, distance 7
            # Bishop from h1 to a8 (7 squares SW)
            ("8/8/8/8/8/8/8/7B w - - 0 1", "h1a8", 41),  # SouthWest, distance 7
        ]
        
        for fen, uci, expected_plane in max_distance_tests:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"Max distance move {uci}: expected plane {expected_plane}, got {plane}"
            
            decoded = encoder.decode(index, board)
            assert decoded == move, f"Roundtrip failed for max distance move {uci}"

    def test_pawn_special_moves(self, encoder):
        """Test pawn-specific moves: single step, double step, captures."""
        # Single pawn move
        board = chess.Board()
        move = chess.Move.from_uci("e2e3")
        index = encoder.encode(move, board)
        plane = index % 73
        assert plane == 28, f"Pawn single step: expected plane 28, got {plane}"
        
        # Double pawn move
        move = chess.Move.from_uci("e2e4")
        index = encoder.encode(move, board)
        plane = index % 73
        assert plane == 29, f"Pawn double step: expected plane 29, got {plane}"
        
        # Pawn capture
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        move = chess.Move.from_uci("e4d5")
        index = encoder.encode(move, board)
        plane = index % 73
        assert plane == 35, f"Pawn capture: expected plane 35 (SW), got {plane}"

    def test_queen_promotion_all_directions(self, encoder):
        """Test queen promotions in all three directions (left, forward, right)."""
        # White queen promotions
        board = chess.Board("3k4/P7/8/8/8/8/8/K7 w - - 0 1")
        
        promotions = [
            ("a7a8q", 28),   # Forward queen promotion (South 1 square)
        ]
        
        # With capture left
        board = chess.Board("1k6/P7/8/8/8/8/8/K7 w - - 0 1")
        move = chess.Move.from_uci("a7b8q")
        index = encoder.encode(move, board)
        plane = index % 73
        assert plane == 21, f"Queen promotion capture: expected plane 21 (SE), got {plane}"
        
        for uci, expected_plane in promotions:
            board = chess.Board("3k4/P7/8/8/8/8/8/K7 w - - 0 1")
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            plane = index % 73
            assert plane == expected_plane, f"Queen promotion {uci}: expected plane {expected_plane}, got {plane}"

    def test_encoding_consistency_across_positions(self, encoder):
        """Test that the same relative move gets the same plane from different positions."""
        # Test "move 1 square south" from different starting squares
        positions_and_moves = [
            ("8/8/8/8/8/8/R7/K6k w - - 0 1", "a2a3"),  # from a2
            ("8/8/8/8/8/R7/8/K6k w - - 0 1", "a3a4"),  # from a3
            ("8/8/8/8/R7/8/8/K6k w - - 0 1", "a4a5"),  # from a4
            ("8/8/8/R7/8/8/8/K6k w - - 0 1", "a5a6"),  # from a5
        ]
        
        planes = []
        for fen, uci in positions_and_moves:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            index = encoder.encode(move, board)
            planes.append(index % 73)
        
        # All should have the same plane (South, 1 square = plane 28)
        assert all(p == planes[0] for p in planes), f"Planes inconsistent: {planes}"
        assert planes[0] == 28, f"Expected plane 28 for 1 square south, got {planes[0]}"

    def test_decode_all_plane_types(self, encoder):
        """Test decoding works for all plane types (queen, knight, underpromotion)."""
        # Queen-style plane
        board = chess.Board("8/8/8/8/8/8/R7/K6k w - - 0 1")
        index = 1 * 584 + 0 * 73 + 28  # a2, plane 28 (South 1)
        decoded = encoder.decode(index, board)
        assert decoded == chess.Move.from_uci("a2a3")
        
        # Knight plane
        board = chess.Board("8/8/8/8/8/8/8/N6k w - - 0 1")
        index = 0 * 584 + 0 * 73 + 58  # a1, plane 58 (knight ESE)
        decoded = encoder.decode(index, board)
        assert decoded == chess.Move.from_uci("a1c2")
        
        # Underpromotion plane
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        index = 6 * 584 + 0 * 73 + 65  # a7, plane 65 (knight forward underpromo)
        decoded = encoder.decode(index, board)
        assert decoded == chess.Move.from_uci("a7a8n")

    def test_index_uniqueness(self, encoder):
        """Test that different moves always produce different indices."""
        board = chess.Board("8/8/8/3Q4/8/8/8/K6k w - - 0 1")
        
        indices = {}
        for move in board.legal_moves:
            index = encoder.encode(move, board)
            uci = move.uci()
            
            if index in indices:
                pytest.fail(f"Index collision! Moves {uci} and {indices[index]} both encode to {index}")
            indices[index] = uci
        
        # All indices should be unique
        assert len(indices) == len(list(board.legal_moves))

    def test_plane_ranges_boundaries(self, encoder):
        """Test that planes stay within their designated ranges."""
        board = chess.Board()
        
        queen_planes = []
        knight_planes = []
        
        # Collect planes from starting position
        for move in board.legal_moves:
            index = encoder.encode(move, board)
            plane = index % 73
            
            from_square = move.from_square
            to_square = move.to_square
            from_row, from_col = divmod(from_square, 8)
            to_row, to_col = divmod(to_square, 8)
            delta = (to_row - from_row, to_col - from_col)
            
            if delta in encoder.KNIGHT_MOVES:
                knight_planes.append(plane)
            else:
                queen_planes.append(plane)
        
        # Verify ranges
        if queen_planes:
            assert all(0 <= p < 56 for p in queen_planes), f"Queen plane out of range: {queen_planes}"
        if knight_planes:
            assert all(56 <= p < 64 for p in knight_planes), f"Knight plane out of range: {knight_planes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
