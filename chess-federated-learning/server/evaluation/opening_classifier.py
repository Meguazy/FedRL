"""
Opening classifier for chess games.

This module classifies chess openings by analyzing the first moves and
matching them against known opening patterns to determine ECO codes.

Uses a simple move-sequence based approach for common openings.
"""

import chess
import chess.pgn
from typing import Optional, Tuple
from io import StringIO


class OpeningClassifier:
    """
    Classify chess openings based on move sequences.

    This is a simplified classifier that identifies common openings
    based on the first few moves.
    """

    # Opening patterns: sequence of moves -> (ECO, name)
    OPENING_PATTERNS = {
        # Sicilian Defense
        ("e4", "c5"): ("B20", "Sicilian Defense"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"): ("B70", "Sicilian Dragon"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"): ("B90", "Sicilian Najdorf"),

        # Ruy Lopez
        ("e4", "e5", "Nf3", "Nc6", "Bb5"): ("C60", "Ruy Lopez"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Nxe4"): ("C80", "Ruy Lopez, Marshall Attack"),

        # Italian Game
        ("e4", "e5", "Nf3", "Nc6", "Bc4"): ("C50", "Italian Game"),

        # French Defense
        ("e4", "e6"): ("C00", "French Defense"),
        ("e4", "e6", "d4", "d5"): ("C10", "French Defense, Classical"),

        # Caro-Kann Defense
        ("e4", "c6"): ("B10", "Caro-Kann Defense"),
        ("e4", "c6", "d4", "d5"): ("B12", "Caro-Kann Defense, Classical"),

        # Scandinavian Defense
        ("e4", "d5"): ("B01", "Scandinavian Defense"),

        # Alekhine's Defense
        ("e4", "Nf6"): ("B02", "Alekhine's Defense"),

        # Pirc Defense
        ("e4", "d6"): ("B00", "Pirc Defense"),

        # Queen's Gambit
        ("d4", "d5", "c4"): ("D06", "Queen's Gambit"),
        ("d4", "d5", "c4", "e6"): ("D30", "Queen's Gambit Declined"),
        ("d4", "d5", "c4", "c6"): ("D10", "Slav Defense"),
        ("d4", "d5", "c4", "dxc4"): ("D20", "Queen's Gambit Accepted"),

        # Indian Defenses
        ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"): ("E20", "Nimzo-Indian Defense"),
        ("d4", "Nf6", "c4", "g6"): ("E60", "King's Indian Defense"),
        ("d4", "Nf6", "c4", "e6", "Nf3", "b6"): ("E12", "Queen's Indian Defense"),

        # English Opening
        ("c4",): ("A10", "English Opening"),
        ("c4", "e5"): ("A20", "English Opening, Reversed Sicilian"),
        ("c4", "Nf6"): ("A15", "English Opening, Anglo-Indian"),

        # Reti Opening
        ("Nf3",): ("A04", "Reti Opening"),
        ("Nf3", "d5", "c4"): ("A09", "Reti Opening, Advance Variation"),

        # King's Indian Attack
        ("Nf3", "Nf6", "g3"): ("A07", "King's Indian Attack"),

        # London System
        ("d4", "d5", "Bf4"): ("D00", "London System"),
        ("d4", "Nf6", "Bf4"): ("D02", "London System vs Nf6"),
    }

    @staticmethod
    def classify_from_pgn(pgn_string: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify opening from PGN string.

        Args:
            pgn_string: PGN string of the game

        Returns:
            Tuple of (ECO code, opening name) or (None, None) if not classified
        """
        try:
            pgn_io = StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)

            if game is None:
                return None, None

            # Check if ECO is already in headers
            eco = game.headers.get("ECO")
            opening = game.headers.get("Opening")
            if eco and opening:
                return eco, opening

            # Extract move sequence
            moves = []
            board = game.board()
            for move in game.mainline_moves():
                san_move = board.san(move)
                moves.append(san_move)
                board.push(move)

                # Only analyze first 10 moves (20 plies)
                if len(moves) >= 20:
                    break

            # Try to match against patterns (longest match first)
            best_match = (None, None)
            best_match_length = 0

            for pattern, (eco, name) in OpeningClassifier.OPENING_PATTERNS.items():
                pattern_length = len(pattern)
                if pattern_length > best_match_length and pattern_length <= len(moves):
                    # Check if pattern matches
                    if tuple(moves[:pattern_length]) == pattern:
                        best_match = (eco, name)
                        best_match_length = pattern_length

            return best_match

        except Exception:
            return None, None

    @staticmethod
    def classify_from_moves(moves: list) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify opening from list of SAN moves.

        Args:
            moves: List of SAN move strings

        Returns:
            Tuple of (ECO code, opening name) or (None, None) if not classified
        """
        # Try to match against patterns (longest match first)
        best_match = (None, None)
        best_match_length = 0

        for pattern, (eco, name) in OpeningClassifier.OPENING_PATTERNS.items():
            pattern_length = len(pattern)
            if pattern_length > best_match_length and pattern_length <= len(moves):
                # Check if pattern matches
                if tuple(moves[:pattern_length]) == pattern:
                    best_match = (eco, name)
                    best_match_length = pattern_length

        return best_match
