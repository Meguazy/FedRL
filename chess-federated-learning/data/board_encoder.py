"""
Board encoder for converting chess positions to tensor representation.

This module converts chess.Board objects into the 119-plane tensor representation
used by AlphaZero for neural network input.

Input representation (119 planes, 8x8 each):
- Planes 0-95: Piece positions (6 piece types x 2 colors x 8 history positions)
- Planes 96-97: Repetition counters (1 and 2+ repetitions)
- Planes 98-101: Castling rights (white kingside, white queenside, black kingside, black queenside)
- Plane 102: Side to move (1 if white to move, 0 if black)
- Plane 103: Move count (normalized by 100)
- Planes 104-118: No-progress count (50-move rule counter, 15 planes)

Total: 119 planes x 8 x 8 = 7,616 values per position
"""
import chess
import numpy as np
from typing import List, Optional
from loguru import logger


class BoardEncoder:
    """
    Encodes chess board positions into 119-plane tensor representation.

    This follows the AlphaZero paper's input representation for chess.

    Example:
        >>> encoder = BoardEncoder()
        >>> board = chess.Board()
        >>> tensor = encoder.encode(board)
        >>> print(tensor.shape)
        (119, 8, 8)
    """

    # Piece type mapping
    PIECE_TYPES = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING
    ]
    
    def __init__(self, history_length: int = 8):
        """Initializes the BoardEncoder."""
        log = logger.bind(context="BoardEncoder.__init__")
        self.history_length = history_length
        self.num_planes = 119  # Total number of planes in the representation
        
        log.info(f"Initialized BoardEncoder with history_length={history_length}, num_planes={self.num_planes}")
        
    def encode(self, board: chess.Board, history: Optional[List[chess.Board]] = None) -> np.ndarray:
        """
        Encode a chess board into 119-plane tensor.

        Args:
            board: Current chess board position
            history: List of previous board positions (most recent first)
                    If None, current position is repeated for all history planes

        Returns:
            numpy array of shape (119, 8, 8) with dtype float32

        Example:
            >>> encoder = BoardEncoder()
            >>> board = chess.Board()
            >>> tensor = encoder.encode(board)
            >>> print(tensor.shape, tensor.dtype)
            (119, 8, 8) float32
        """
        planes = []
        log = logger.bind(context="BoardEncoder.encode")
        log.debug("Encoding board position...")
        
        # Planes 0-95: Piece positions with history
        piece_planes = self._encode_piece_planes(board, history)
        planes.append(piece_planes) # Shape (96, 8, 8)
        
        # Planes 96-97: Repetition counters
        repetition_planes = self._encode_repetition_planes(board)
        planes.append(repetition_planes) # Shape (2, 8, 8)
        
        # Planes 98-101: Castling rights
        castling_planes = self._encode_castling_planes(board)
        planes.append(castling_planes) # Shape (4, 8, 8)
        
        # Plane 102: Side to move
        side_to_move_plane = self._encode_color(board)
        planes.append(side_to_move_plane) # Shape (1, 8, 8)
        
        # Plane 103: Move count
        move_count_plane = self._encode_move_count(board)
        planes.append(move_count_plane) # Shape (1, 8, 8)

        # Planes 104-118: No-progress count
        no_progress_planes = self._encode_no_progress_count(board)
        planes.append(no_progress_planes) # Shape (15, 8, 8)

        # Stack all planes into a single tensor
        tensor = np.concatenate(planes, axis=0) # Shape (119, 8, 8)
        log.debug(f"Encoded tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        return tensor.astype(np.float32)
    
    def _encode_piece_planes(self, board: chess.Board, history: Optional[List[chess.Board]] = None) -> np.ndarray:
        """
        Encode piece positions with history (96 planes).

        Planes 0-11: Current position (6 piece types × 2 colors)
        Planes 12-23: Position 1 move ago
        Planes 24-35: Position 2 moves ago
        ...
        Planes 84-95: Position 7 moves ago

        Args:
            board: Current board
            history: Previous board positions

        Returns:
            Array of shape (96, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_piece_planes")
        log.trace("Encoding piece planes with history...")
        planes = np.zeros((96, 8, 8), dtype=np.float32)
        
        # Prepare history (current + previous positions)
        if history is None:
            history = []
            
        boards = [board] + history
        boards = boards[:self.history_length]  # Limit to history_length
        
        # Pad with current position if history is short
        while len(boards) < self.history_length:
            boards.append(board)
            
        # Encode each historical position
        for i, hist_board in enumerate(boards):
            offset = i * 12  # 12 planes per position (6 pieces x 2 colors)
            
            # Encode white pieces
            for piece_idx, piece_type in enumerate(self.PIECE_TYPES):
                plane_idx = offset + piece_idx
                squares = hist_board.pieces(piece_type, chess.WHITE)
                for square in squares:
                    rank, file = divmod(square, 8)
                    planes[plane_idx, rank, file] = 1.0
                    
            # Encode black pieces
            for piece_idx, piece_type in enumerate(self.PIECE_TYPES):
                plane_idx = offset + 6 + piece_idx
                squares = hist_board.pieces(piece_type, chess.BLACK)
                for square in squares:
                    rank, file = divmod(square, 8)
                    planes[plane_idx, rank, file] = 1.0
            
        log.trace("All piece planes encoded.")
        return planes
    
    def _encode_repetition_planes(self, board: chess.Board) -> np.ndarray:
        """
        Encode repetition counters (2 planes).

        Plane 96: 1 repetition
        Plane 97: 2 or more repetitions

        Args:
            board: Current board

        Returns:
            Array of shape (2, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_repetition_planes")
        log.trace("Encoding repetition planes...")
        planes = np.zeros((2, 8, 8), dtype=np.float32)
        
        # Check if current position is a repetition
        if board.is_repetition(2):
            planes[0, :, :] = 1.0  # 1 repetition
        if board.is_repetition(3):
            planes[1, :, :] = 1.0  # 2 or more repetitions
            
        return planes
    
    def _encode_castling_planes(self, board: chess.Board) -> np.ndarray:
        """
        Encode castling rights (4 planes).

        Plane 0: White kingside castling
        Plane 1: White queenside castling
        Plane 2: Black kingside castling
        Plane 3: Black queenside castling

        Args:
            board: Chess board

        Returns:
            Array of shape (4, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_castling_planes")
        log.trace("Encoding castling rights...")
        planes = np.zeros((4, 8, 8), dtype=np.float32)
        
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[0, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[1, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[2, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[3, :, :] = 1.0
            
        return planes
    
    def _encode_color(self, board: chess.Board) -> np.ndarray:
        """
        Encode side to move (1 plane).

        1.0 if white to move, 0.0 if black to move

        Args:
            board: Chess board

        Returns:
            Array of shape (1, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_color")
        log.trace("Encoding side to move...")
        plane = np.zeros((1, 8, 8), dtype=np.float32)

        if board.turn == chess.WHITE:
            plane[0, :, :] = 1.0
        else:
            plane[0, :, :] = 0.0

        return plane

    
    def _encode_move_count(self, board: chess.Board) -> np.ndarray:
        """
        Encode move count (1 plane).

        Normalized by dividing by 100 to keep values in reasonable range.

        Args:
            board: Chess board

        Returns:
            Array of shape (1, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_move_count")
        log.trace("Encoding move count...")
        plane = np.zeros((1, 8, 8), dtype=np.float32)

        # Normalize move count
        move_count = board.fullmove_number
        normalized = min(move_count / 100.0, 1.0)  # Cap at 1.0
        plane[0, :, :] = normalized

        return plane

    def _encode_no_progress_count(self, board: chess.Board) -> np.ndarray:
        """
        Encode no-progress count for 50-move rule (15 planes).

        Uses thermometer encoding: first N planes are 1.0 where N = halfmove_clock // 4

        Args:
            board: Chess board

        Returns:
            Array of shape (15, 8, 8)
        """
        log = logger.bind(context="BoardEncoder._encode_no_progress_count")
        log.trace("Encoding no-progress count planes...")
        planes = np.zeros((15, 8, 8), dtype=np.float32)

        # Halfmove clock counts plies since last capture or pawn move
        halfmoves = board.halfmove_clock

        # Thermometer encoding: fill first N planes
        # Divide by 4 to spread across 15 planes (0-60 halfmoves -> 0-15 planes)
        num_filled = min(halfmoves // 4, 15)

        for i in range(num_filled):
            planes[i, :, :] = 1.0
            
        return planes

    def encode_batch(self, boards: List[chess.Board], histories: Optional[List[List[chess.Board]]] = None) -> np.ndarray:
        """
        Encode a batch of boards.

        Args:
            boards: List of chess boards
            histories: List of history lists (one per board)

        Returns:
            Array of shape (batch_size, 119, 8, 8)

        Example:
            >>> encoder = BoardEncoder()
            >>> boards = [chess.Board() for _ in range(32)]
            >>> batch = encoder.encode_batch(boards)
            >>> print(batch.shape)
            (32, 119, 8, 8)
        """
        if histories is None:
            histories = [None] * len(boards)

        encoded = [self.encode(board, history) for board, history in zip(boards, histories)]
        return np.stack(encoded, axis=0)

# if __name__ == "__main__":
#     log = logger.bind(module="BoardEncoder.__main__")

#     log.info("="*70)
#     log.info("BOARD ENCODER TEST")
#     log.info("="*70)

#     encoder = BoardEncoder(history_length=8)

#     # Test 1: Encode starting position
#     log.info("\n[TEST 1] Encoding starting position")
#     log.info("-" * 70)

#     board = chess.Board()
#     log.debug(f"Initial board state:\n{board}")
#     tensor = encoder.encode(board)
#     log.debug(f"Encoded tensor:\n{tensor}")

#     log.success(f"Encoded tensor shape: {tensor.shape}")
#     log.info(f"Tensor dtype: {tensor.dtype}")
#     log.info(f"Tensor min/max: {tensor.min():.3f} / {tensor.max():.3f}")

#     # Check piece planes
#     log.info("\nChecking piece planes (first position):")
#     log.info(f"  Plane 0 (white pawns): {tensor[0].sum():.0f} pieces")
#     log.info(f"  Plane 1 (white knights): {tensor[1].sum():.0f} pieces")
#     log.info(f"  Plane 5 (white king): {tensor[5].sum():.0f} pieces")
#     log.info(f"  Plane 6 (black pawns): {tensor[6].sum():.0f} pieces")

#     # Check metadata planes
#     log.info("\nChecking metadata planes:")
#     log.info(f"  Plane 98 (white K-side castle): {tensor[98, 0, 0]:.0f}")
#     log.info(f"  Plane 102 (color to move): {tensor[102, 0, 0]:.0f} (1=white)")
#     log.info(f"  Plane 103 (move count): {tensor[103, 0, 0]:.3f}")

#     # Test 2: Encode position after some moves
#     log.info("\n[TEST 2] Encoding position after moves")
#     log.info("-" * 70)

#     board.push_san("e4")
#     board.push_san("e5")
#     board.push_san("Nf3")

#     tensor2 = encoder.encode(board)

#     log.success(f"Encoded tensor shape: {tensor2.shape}")
#     log.info(f"Move count: {tensor2[103, 0, 0]:.3f}")
#     log.info(f"Color to move: {tensor2[102, 0, 0]:.0f} (0=black)")

#     # Test 3: Batch encoding
#     log.info("\n[TEST 3] Batch encoding")
#     log.info("-" * 70)

#     boards = [chess.Board() for _ in range(4)]
#     boards[1].push_san("e4")
#     boards[2].push_san("d4")
#     boards[3].push_san("Nf3")

#     batch_tensor = encoder.encode_batch(boards)

#     log.success(f"Batch tensor shape: {batch_tensor.shape}")
#     log.info(f"Batch dtype: {batch_tensor.dtype}")
#     log.info(f"Memory size: {batch_tensor.nbytes / 1024:.1f} KB")

#     # Test 4: Encoding with history
#     log.info("\n[TEST 4] Encoding with history")
#     log.info("-" * 70)

#     # Create a game with history
#     board_with_history = chess.Board()
#     history_boards = []

#     # Play some moves and save history
#     moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3"]
    
#     for move in moves:
#         # Save current position before making move
#         history_boards.insert(0, board_with_history.copy())  # Insert at front (most recent first)
#         board_with_history.push_san(move)
    
#     # Keep only last 7 positions (we encode current + 7 history = 8 total)
#     history_boards = history_boards[:7]

#     log.info(f"Current position after: {' '.join(moves)}")
#     log.info(f"History length: {len(history_boards)} positions")

#     # Encode with history
#     tensor_with_history = encoder.encode(board_with_history, history=history_boards)

#     log.success(f"Encoded tensor with history shape: {tensor_with_history.shape}")
    
#     # Compare current position planes vs 1 move ago planes
#     log.info("\nComparing piece planes:")
#     log.info(f"  Current position (plane 0, white pawns): {tensor_with_history[0].sum():.0f} pieces")
#     log.info(f"  1 move ago (plane 12, white pawns): {tensor_with_history[12].sum():.0f} pieces")
#     log.info(f"  2 moves ago (plane 24, white pawns): {tensor_with_history[24].sum():.0f} pieces")
    
#     # Encode WITHOUT history (should repeat current position)
#     tensor_no_history = encoder.encode(board_with_history)
    
#     log.info("\nWithout history (repeats current position):")
#     log.info(f"  Current position (plane 0): {tensor_no_history[0].sum():.0f} pieces")
#     log.info(f"  'History' plane 12 (repeated): {tensor_no_history[12].sum():.0f} pieces")
#     log.info(f"  'History' plane 24 (repeated): {tensor_no_history[24].sum():.0f} pieces")
    
#     # Verify they're different
#     planes_diff = np.abs(tensor_with_history[12:24] - tensor_no_history[12:24]).sum()
#     log.info(f"\nDifference between with/without history: {planes_diff:.0f}")
    
#     if planes_diff > 0:
#         log.success("History encoding works correctly!")
#     else:
#         log.warning("History planes are identical (unexpected)")
    
#     log.info("\n" + "="*70)
#     log.success("TEST COMPLETE")
#     log.info("="*70)
