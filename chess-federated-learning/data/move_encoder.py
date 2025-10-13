"""
Move encoder for converting chess moves to AlphaZero action indices.

This module implements the 8x8x73 move encoding scheme (4672 total moves) used in AlphaZero:
- 56 planes: Queen-style moves (8 directions x 7 distances)
- 8 planes: Knight moves
- 9 planes: Underpromotions (3 directions x 3 piece types)

For detailed encoding documentation, see:
chess-federated-learning/client/trainer/models/move_encoding.md
"""
import chess
from typing import Optional, Tuple
from loguru import logger


class MoveEncoder:
    """
    Encodes chess moves to action indices (0-4671) and vice versa.

    The encoding follows AlphaZero's 8x8x73 scheme:
    - From-square: 64 possibilities (8x8 board)
    - Move plane: 73 possibilities (56 queen-style + 8 knight + 9 underpromotion)
    - Total: 64 x 73 = 4672 possible actions

    Example:
        >>> encoder = MoveEncoder()
        >>> move = chess.Move.from_uci("e2e4")
        >>> index = encoder.encode(move)
        >>> decoded_move = encoder.decode(index)
        >>> assert move == decoded_move
    """

    # Direction vectors for queen-style moves (N, NE, E, SE, S, SW, W, NW)
    # NOTE: Direction naming follows AlphaZero's array indexing convention:
    #   Row 0 = rank 1 (white's back rank), Row 7 = rank 8 (black's back rank)
    #   "South" = increasing row (toward rank 8), "North" = decreasing row (toward rank 1)
    #   This is counterintuitive but standard in the literature.
    QUEEN_DIRECTIONS = [
        (-1, 0),   # North: row decreases (toward rank 1)
        (-1, 1),   # NorthEast
        (0, 1),    # East
        (1, 1),    # SouthEast
        (1, 0),    # South: row increases (toward rank 8)
        (1, -1),   # SouthWest
        (0, -1),   # West
        (-1, -1),  # NorthWest
    ]

    # Knight move offsets (NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW)
    KNIGHT_MOVES = [
        (-2, 1),   # NNE: 2 North, 1 East
        (-1, 2),   # ENE: 2 East, 1 North
        (1, 2),    # ESE: 2 East, 1 South
        (2, 1),    # SSE: 2 South, 1 East
        (2, -1),   # SSW: 2 South, 1 West
        (1, -2),   # WSW: 2 West, 1 South
        (-1, -2),  # WNW: 2 West, 1 North
        (-2, -1),  # NNW: 2 North, 1 West
    ]

    # Underpromotion direction offsets (Left-diagonal, Forward, Right-diagonal)
    # For white pawns (promoting from row 6 to row 7, row increases)
    UNDERPROMO_DIRECTIONS_WHITE = [
        (1, -1),   # Left-diagonal (towards h-file, capturing left from white's perspective)
        (1, 0),    # Forward (straight ahead)
        (1, 1),    # Right-diagonal (towards a-file, capturing right from white's perspective)
    ]

    # For black pawns (promoting from row 1 to row 0, row decreases)
    UNDERPROMO_DIRECTIONS_BLACK = [
        (-1, 1),   # Left-diagonal (towards a-file, capturing left from black's perspective)
        (-1, 0),   # Forward (straight ahead)
        (-1, -1),  # Right-diagonal (towards h-file, capturing right from black's perspective)
    ]

    def __init__(self):
        """Initialize the move encoder."""
        pass

    def encode(self, move: chess.Move, board: Optional[chess.Board] = None) -> int:
        """
        Encode a chess move to an action index (0-4671).

        Args:
            move: The chess move to encode
            board: Optional board for context (needed to determine promotion piece color)

        Returns:
            Action index in range [0, 4671]

        Raises:
            ValueError: If the move cannot be encoded
        """
        from_square = move.from_square
        to_square = move.to_square

        # Get from/to coordinates
        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)

        # Calculate movement delta
        delta_row = to_row - from_row
        delta_col = to_col - from_col

        # Handle promotions
        if move.promotion:
            # Queen promotion uses queen-style moves
            if move.promotion == chess.QUEEN:
                plane = self._encode_queen_move(delta_row, delta_col)
            else:
                # Underpromotion (knight, bishop, rook)
                plane = self._encode_underpromotion(move, from_row, board)
        # Handle knight moves
        elif (delta_row, delta_col) in self.KNIGHT_MOVES:
            plane = self._encode_knight_move(delta_row, delta_col)
        # Handle queen-style moves (rook, bishop, queen, king, pawn)
        else:
            plane = self._encode_queen_move(delta_row, delta_col)

        # Convert to flat index
        index = from_row * 8 * 73 + from_col * 73 + plane
        return index

    def decode(self, index: int, board: chess.Board) -> chess.Move:
        """
        Decode an action index to a chess move.

        Args:
            index: Action index in range [0, 4671]
            board: Current board position (needed to determine exact move)

        Returns:
            The decoded chess move

        Raises:
            ValueError: If the index is invalid or doesn't correspond to a legal move
        """
        if not 0 <= index < 4672:
            raise ValueError(f"Index {index} out of range [0, 4671]")

        # Decode to (from_row, from_col, plane)
        from_row = index // (8 * 73)
        from_col = (index % (8 * 73)) // 73
        plane = index % 73

        from_square = from_row * 8 + from_col

        # Decode plane to move
        if plane < 56:  # Queen-style move
            to_row, to_col = self._decode_queen_move(from_row, from_col, plane)
        elif plane < 64:  # Knight move
            to_row, to_col = self._decode_knight_move(from_row, from_col, plane)
        else:  # Underpromotion
            to_row, to_col, promotion = self._decode_underpromotion(from_row, from_col, plane, board)
            to_square = to_row * 8 + to_col
            return chess.Move(from_square, to_square, promotion=promotion)

        to_square = to_row * 8 + to_col

        # Check if this is a promotion (pawn reaching last rank)
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_row == 7) or \
               (piece.color == chess.BLACK and to_row == 0):
                # Queen promotion (underpromotions handled above)
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return chess.Move(from_square, to_square)

    def _encode_queen_move(self, delta_row: int, delta_col: int) -> int:
        """Encode a queen-style move to a plane index (0-55)."""
        # Determine direction and distance
        distance = max(abs(delta_row), abs(delta_col))

        if distance == 0:
            raise ValueError("Invalid move: zero distance")

        # Normalize direction
        dir_row = 0 if delta_row == 0 else delta_row // abs(delta_row)
        dir_col = 0 if delta_col == 0 else delta_col // abs(delta_col)

        # Find direction index
        try:
            direction_idx = self.QUEEN_DIRECTIONS.index((dir_row, dir_col))
        except ValueError:
            raise ValueError(f"Invalid queen-style direction: ({dir_row}, {dir_col})")

        # Plane = direction * 7 + (distance - 1)
        plane = direction_idx * 7 + (distance - 1)
        return plane

    def _encode_knight_move(self, delta_row: int, delta_col: int) -> int:
        """Encode a knight move to a plane index (56-63)."""
        try:
            knight_idx = self.KNIGHT_MOVES.index((delta_row, delta_col))
        except ValueError:
            raise ValueError(f"Invalid knight move: ({delta_row}, {delta_col})")

        return 56 + knight_idx

    def _encode_underpromotion(self, move: chess.Move, from_row: int,
                               board: Optional[chess.Board]) -> int:
        """Encode an underpromotion to a plane index (64-72)."""
        from_square = move.from_square
        to_square = move.to_square

        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)

        delta_row = to_row - from_row
        delta_col = to_col - from_col

        # Determine piece color based on which rank they're promoting to
        # White pawns promote to row 7 (rank 8), black pawns to row 0 (rank 1)
        if to_row == 7:  # White pawn promotion (moving to rank 8)
            directions = self.UNDERPROMO_DIRECTIONS_WHITE
        else:  # to_row == 0, Black pawn promotion (moving to rank 1)
            directions = self.UNDERPROMO_DIRECTIONS_BLACK

        # Find direction index
        try:
            direction_idx = directions.index((delta_row, delta_col))
        except ValueError:
            raise ValueError(f"Invalid underpromotion direction: ({delta_row}, {delta_col})")

        # Map promotion piece to index
        piece_map = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
        }

        if move.promotion not in piece_map:
            raise ValueError(f"Invalid underpromotion piece: {move.promotion}")

        piece_idx = piece_map[move.promotion]

        # Plane = 64 + piece_type * 3 + direction
        plane = 64 + piece_idx * 3 + direction_idx
        return plane

    def _decode_queen_move(self, from_row: int, from_col: int, plane: int) -> Tuple[int, int]:
        """Decode a queen-style move plane to target square."""
        direction_idx = plane // 7
        distance = (plane % 7) + 1

        dir_row, dir_col = self.QUEEN_DIRECTIONS[direction_idx]

        to_row = from_row + dir_row * distance
        to_col = from_col + dir_col * distance

        return to_row, to_col

    def _decode_knight_move(self, from_row: int, from_col: int, plane: int) -> Tuple[int, int]:
        """Decode a knight move plane to target square."""
        knight_idx = plane - 56
        delta_row, delta_col = self.KNIGHT_MOVES[knight_idx]

        to_row = from_row + delta_row
        to_col = from_col + delta_col

        return to_row, to_col

    def _decode_underpromotion(self, from_row: int, from_col: int, plane: int,
                               board: chess.Board) -> Tuple[int, int, int]:
        """Decode an underpromotion plane to target square and promotion piece."""
        p = plane - 64
        piece_idx = p // 3
        direction_idx = p % 3

        # Map piece index to piece type
        piece_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion_piece = piece_types[piece_idx]

        # Determine direction based on from_row position
        # White pawns promote from row 6 to row 7 (rank 7 to rank 8)
        # Black pawns promote from row 1 to row 0 (rank 2 to rank 1)
        if from_row == 6:  # White pawn promotion
            directions = self.UNDERPROMO_DIRECTIONS_WHITE
        else:  # from_row == 1, Black pawn promotion
            directions = self.UNDERPROMO_DIRECTIONS_BLACK

        delta_row, delta_col = directions[direction_idx]

        to_row = from_row + delta_row
        to_col = from_col + delta_col

        return to_row, to_col, promotion_piece


# if __name__ == "__main__":
#     import sys

#     log = logger.bind(module="MoveEncoder.__main__")

#     log.info("="*70)
#     log.info("MOVE ENCODER TEST")
#     log.info("="*70)

#     encoder = MoveEncoder()
#     board = chess.Board()

#     # Test 1: Encode and decode all legal moves from starting position
#     log.info("")
#     log.info("[TEST 1] Encoding all legal moves from starting position")
#     log.info("-" * 70)

#     legal_moves = list(board.legal_moves)
#     log.info(f"Found {len(legal_moves)} legal moves")

#     for i, move in enumerate(legal_moves[:5], 1):  # Show first 5
#         index = encoder.encode(move, board)
#         decoded = encoder.decode(index, board)

#         match = "✓" if decoded == move else "✗"
#         log.info(f"  {i}. {move.uci():6s} → index {index:4d} → {decoded.uci():6s} {match}")

#     # Test 2: Test specific move types
#     log.info("")
#     log.info("[TEST 2] Testing specific move types")
#     log.info("-" * 70)

#     test_cases = [
#         ("e2e4", "Pawn advance 2 squares"),
#         ("g1f3", "Knight move"),
#     ]

#     for uci, description in test_cases:
#         move = chess.Move.from_uci(uci)
#         if move in board.legal_moves:
#             index = encoder.encode(move, board)
#             decoded = encoder.decode(index, board)
#             match = "✓" if decoded == move else "✗"
#             log.info(f"  {description:25s}: {uci} → {index:4d} → {decoded.uci()} {match}")

#     # Test 3: Test promotion
#     log.info("")
#     log.info("[TEST 3] Testing pawn promotions")
#     log.info("-" * 70)

#     # Set up a position with pawn ready to promote
#     board_promo = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
#     log.info(f"Position: {board_promo.fen()}")

#     promo_moves = list(board_promo.legal_moves)
#     log.info(f"Legal moves: {[m.uci() for m in promo_moves]}")

#     for move in promo_moves:
#         index = encoder.encode(move, board_promo)
#         decoded = encoder.decode(index, board_promo)
#         match = "✓" if decoded == move else "✗"
#         promo_piece = chess.piece_name(move.promotion) if move.promotion else "none"
#         log.info(f"  {move.uci():6s} (→{promo_piece:6s}) → index {index:4d} → {decoded.uci():6s} {match}")

#     # Test 4: Encoding space coverage
#     log.info("")
#     log.info("[TEST 4] Encoding space coverage")
#     log.info("-" * 70)

#     # Count moves by type
#     queen_moves = sum(1 for m in legal_moves if encoder.encode(m, board) % 73 < 56)
#     knight_moves = sum(1 for m in legal_moves if 56 <= encoder.encode(m, board) % 73 < 64)

#     log.info(f"  Queen-style moves: {queen_moves}")
#     log.info(f"  Knight moves: {knight_moves}")
#     log.info(f"  Total action space: 4672 (8x8x73)")
#     log.info(f"  Legal moves in starting position: {len(legal_moves)}")
#     log.info(f"  Sparsity: {100 * (1 - len(legal_moves)/4672):.1f}% of moves are illegal")

#     log.info("")
#     log.info("="*70)
#     log.success("TEST COMPLETE")
#     log.info("="*70)
