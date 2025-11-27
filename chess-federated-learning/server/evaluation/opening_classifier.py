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
        # ==================== 1. e4 Openings ====================

        # Sicilian Defense (B20-B99)
        ("e4", "c5"): ("B20", "Sicilian Defense"),
        ("e4", "c5", "Nf3"): ("B20", "Sicilian Defense, Open"),
        ("e4", "c5", "Nf3", "d6"): ("B50", "Sicilian Defense"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"): ("B50", "Sicilian Defense, Delayed Alapin"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"): ("B70", "Sicilian Dragon"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"): ("B70", "Sicilian Dragon"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"): ("B90", "Sicilian Najdorf"),
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6"): ("B80", "Sicilian Scheveningen"),
        ("e4", "c5", "Nf3", "Nc6"): ("B30", "Sicilian Defense, Old Sicilian"),
        ("e4", "c5", "Nf3", "e6"): ("B40", "Sicilian Defense, French Variation"),
        ("e4", "c5", "c3"): ("B22", "Sicilian Defense, Alapin Variation"),
        ("e4", "c5", "Nc3"): ("B23", "Sicilian Defense, Closed"),

        # Ruy Lopez / Spanish Opening (C60-C99)
        ("e4", "e5", "Nf3", "Nc6", "Bb5"): ("C60", "Ruy Lopez"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"): ("C70", "Ruy Lopez, Morphy Defense"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"): ("C70", "Ruy Lopez, Morphy Defense"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"): ("C70", "Ruy Lopez, Morphy Defense"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"): ("C80", "Ruy Lopez, Open"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Nxe4"): ("C80", "Ruy Lopez, Open, Marshall Attack"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"): ("C90", "Ruy Lopez, Closed"),
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"): ("C65", "Ruy Lopez, Berlin Defense"),

        # Italian Game (C50-C59)
        ("e4", "e5", "Nf3", "Nc6", "Bc4"): ("C50", "Italian Game"),
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"): ("C50", "Italian Game, Giuoco Piano"),
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"): ("C55", "Italian Game, Two Knights Defense"),
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5"): ("C57", "Italian Game, Two Knights Defense, Fried Liver Attack"),

        # Scotch Game (C44-C45)
        ("e4", "e5", "Nf3", "Nc6", "d4"): ("C44", "Scotch Game"),
        ("e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4"): ("C45", "Scotch Game"),

        # Four Knights Game (C46-C49)
        ("e4", "e5", "Nf3", "Nc6", "Nc3"): ("C46", "Four Knights Game"),
        ("e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"): ("C46", "Four Knights Game"),

        # Petrov Defense (C42-C43)
        ("e4", "e5", "Nf3", "Nf6"): ("C42", "Petrov Defense"),
        ("e4", "e5", "Nf3", "Nf6", "Nxe5"): ("C42", "Petrov Defense"),

        # King's Gambit (C30-C39)
        ("e4", "e5", "f4"): ("C30", "King's Gambit"),
        ("e4", "e5", "f4", "exf4"): ("C33", "King's Gambit Accepted"),

        # French Defense (C00-C19)
        ("e4", "e6"): ("C00", "French Defense"),
        ("e4", "e6", "d4"): ("C00", "French Defense"),
        ("e4", "e6", "d4", "d5"): ("C10", "French Defense"),
        ("e4", "e6", "d4", "d5", "Nc3"): ("C15", "French Defense, Winawer Variation"),
        ("e4", "e6", "d4", "d5", "Nc3", "Bb4"): ("C15", "French Defense, Winawer Variation"),
        ("e4", "e6", "d4", "d5", "Nd2"): ("C03", "French Defense, Tarrasch Variation"),
        ("e4", "e6", "d4", "d5", "exd5"): ("C01", "French Defense, Exchange Variation"),

        # Caro-Kann Defense (B10-B19)
        ("e4", "c6"): ("B10", "Caro-Kann Defense"),
        ("e4", "c6", "d4"): ("B10", "Caro-Kann Defense"),
        ("e4", "c6", "d4", "d5"): ("B12", "Caro-Kann Defense"),
        ("e4", "c6", "d4", "d5", "Nc3"): ("B12", "Caro-Kann Defense, Advance Variation"),
        ("e4", "c6", "d4", "d5", "Nc3", "dxe4"): ("B13", "Caro-Kann Defense, Exchange Variation"),
        ("e4", "c6", "d4", "d5", "e5"): ("B12", "Caro-Kann Defense, Advance Variation"),

        # Scandinavian Defense (B01)
        ("e4", "d5"): ("B01", "Scandinavian Defense"),
        ("e4", "d5", "exd5"): ("B01", "Scandinavian Defense"),
        ("e4", "d5", "exd5", "Qxd5"): ("B01", "Scandinavian Defense, Main Line"),
        ("e4", "d5", "exd5", "Nf6"): ("B01", "Scandinavian Defense, Modern Variation"),

        # Alekhine's Defense (B02-B05)
        ("e4", "Nf6"): ("B02", "Alekhine's Defense"),
        ("e4", "Nf6", "e5"): ("B02", "Alekhine's Defense"),
        ("e4", "Nf6", "e5", "Nd5"): ("B02", "Alekhine's Defense"),
        ("e4", "Nf6", "e5", "Nd5", "d4"): ("B03", "Alekhine's Defense, Four Pawns Attack"),

        # Pirc Defense (B07-B09)
        ("e4", "d6"): ("B00", "Pirc Defense"),
        ("e4", "d6", "d4", "Nf6"): ("B07", "Pirc Defense"),
        ("e4", "d6", "d4", "Nf6", "Nc3", "g6"): ("B07", "Pirc Defense"),

        # Modern Defense (B06)
        ("e4", "g6"): ("B06", "Modern Defense"),

        # Owen's Defense (B00)
        ("e4", "b6"): ("B00", "Owen's Defense"),

        # Nimzowitsch Defense (B00)
        ("e4", "Nc6"): ("B00", "Nimzowitsch Defense"),

        # ==================== 2. d4 Openings ====================

        # Queen's Gambit (D06-D69)
        ("d4", "d5", "c4"): ("D06", "Queen's Gambit"),
        ("d4", "d5", "c4", "e6"): ("D30", "Queen's Gambit Declined"),
        ("d4", "d5", "c4", "e6", "Nc3"): ("D30", "Queen's Gambit Declined"),
        ("d4", "d5", "c4", "e6", "Nc3", "Nf6"): ("D30", "Queen's Gambit Declined, Orthodox Defense"),
        ("d4", "d5", "c4", "e6", "Nf3", "Nf6"): ("D30", "Queen's Gambit Declined"),
        ("d4", "d5", "c4", "c6"): ("D10", "Slav Defense"),
        ("d4", "d5", "c4", "c6", "Nf3"): ("D10", "Slav Defense"),
        ("d4", "d5", "c4", "c6", "Nf3", "Nf6"): ("D11", "Slav Defense"),
        ("d4", "d5", "c4", "dxc4"): ("D20", "Queen's Gambit Accepted"),
        ("d4", "d5", "c4", "dxc4", "Nf3"): ("D20", "Queen's Gambit Accepted"),
        ("d4", "d5", "c4", "Nf6"): ("D30", "Queen's Gambit Declined"),

        # Indian Defenses

        # King's Indian Defense (E60-E99)
        ("d4", "Nf6", "c4", "g6"): ("E60", "King's Indian Defense"),
        ("d4", "Nf6", "c4", "g6", "Nc3"): ("E60", "King's Indian Defense"),
        ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7"): ("E60", "King's Indian Defense"),
        ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4"): ("E70", "King's Indian Defense, Normal Variation"),
        ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"): ("E70", "King's Indian Defense"),

        # Nimzo-Indian Defense (E20-E59)
        ("d4", "Nf6", "c4", "e6"): ("E00", "Catalan Opening"),
        ("d4", "Nf6", "c4", "e6", "Nc3"): ("E20", "Nimzo-Indian Defense"),
        ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"): ("E20", "Nimzo-Indian Defense"),
        ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Qc2"): ("E32", "Nimzo-Indian Defense, Classical Variation"),
        ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"): ("E40", "Nimzo-Indian Defense, Rubinstein Variation"),

        # Queen's Indian Defense (E12-E19)
        ("d4", "Nf6", "c4", "e6", "Nf3"): ("E00", "Catalan Opening"),
        ("d4", "Nf6", "c4", "e6", "Nf3", "b6"): ("E12", "Queen's Indian Defense"),
        ("d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3"): ("E15", "Queen's Indian Defense, Fianchetto Variation"),

        # Grunfeld Defense (D70-D99)
        ("d4", "Nf6", "c4", "g6", "Nc3", "d5"): ("D70", "Grunfeld Defense"),
        ("d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5"): ("D70", "Grunfeld Defense, Exchange Variation"),

        # Benoni Defense (A60-A79)
        ("d4", "Nf6", "c4", "c5"): ("A60", "Benoni Defense"),
        ("d4", "Nf6", "c4", "c5", "d5"): ("A60", "Benoni Defense, Modern Variation"),
        ("d4", "c5"): ("A40", "Benoni Defense"),

        # Bogo-Indian Defense (E11)
        ("d4", "Nf6", "c4", "e6", "Nf3", "Bb4+"): ("E11", "Bogo-Indian Defense"),

        # Dutch Defense (A80-A99)
        ("d4", "f5"): ("A80", "Dutch Defense"),
        ("d4", "f5", "g3"): ("A80", "Dutch Defense, Fianchetto Variation"),
        ("d4", "f5", "c4"): ("A80", "Dutch Defense"),

        # London System (D00-D02)
        ("d4", "d5", "Bf4"): ("D00", "London System"),
        ("d4", "Nf6", "Bf4"): ("D02", "London System"),
        ("d4", "d5", "Nf3", "Nf6", "Bf4"): ("D00", "London System"),

        # Torre Attack (D03)
        ("d4", "Nf6", "Nf3", "e6", "Bg5"): ("D03", "Torre Attack"),
        ("d4", "d5", "Nf3", "Nf6", "Bg5"): ("D03", "Torre Attack"),

        # Trompowsky Attack (A45)
        ("d4", "Nf6", "Bg5"): ("A45", "Trompowsky Attack"),

        # Colle System (D05)
        ("d4", "d5", "Nf3", "Nf6", "e3"): ("D05", "Colle System"),

        # Stonewall Attack
        ("d4", "d5", "e3", "Nf6", "Bd3"): ("D00", "Stonewall Attack"),

        # ==================== 3. Other First Moves ====================

        # English Opening (A10-A39)
        ("c4",): ("A10", "English Opening"),
        ("c4", "e5"): ("A20", "English Opening, Reversed Sicilian"),
        ("c4", "e5", "Nc3"): ("A20", "English Opening, Reversed Sicilian"),
        ("c4", "Nf6"): ("A15", "English Opening, Anglo-Indian"),
        ("c4", "Nf6", "Nc3"): ("A15", "English Opening, Anglo-Indian"),
        ("c4", "c5"): ("A25", "English Opening, Symmetrical Variation"),
        ("c4", "c5", "Nc3"): ("A25", "English Opening, Symmetrical Variation"),
        ("c4", "e6"): ("A13", "English Opening"),
        ("c4", "c6"): ("A11", "English Opening, Caro-Kann Defensive System"),
        ("c4", "g6"): ("A10", "English Opening, Anglo-Scandinavian Defense"),

        # Reti Opening (A04-A09)
        ("Nf3",): ("A04", "Reti Opening"),
        ("Nf3", "d5"): ("A04", "Reti Opening"),
        ("Nf3", "d5", "c4"): ("A09", "Reti Opening, Advance Variation"),
        ("Nf3", "d5", "g3"): ("A05", "Reti Opening, King's Indian Attack"),
        ("Nf3", "Nf6"): ("A04", "Reti Opening"),
        ("Nf3", "Nf6", "c4"): ("A09", "Reti Opening"),
        ("Nf3", "Nf6", "g3"): ("A07", "King's Indian Attack"),
        ("Nf3", "c5"): ("A04", "Reti Opening"),

        # King's Indian Attack (A07-A08)
        ("Nf3", "Nf6", "g3"): ("A07", "King's Indian Attack"),
        ("Nf3", "d5", "g3", "c5"): ("A07", "King's Indian Attack"),

        # Bird's Opening (A02-A03)
        ("f4",): ("A02", "Bird's Opening"),
        ("f4", "d5"): ("A02", "Bird's Opening"),

        # Larsen's Opening (A01)
        ("b3",): ("A01", "Nimzowitsch-Larsen Attack"),

        # Grob's Attack (A00)
        ("g4",): ("A00", "Grob's Attack"),

        # Polish Opening / Sokolsky Opening (A00)
        ("b4",): ("A00", "Polish Opening"),

        # Hungarian Opening (A00)
        ("g3",): ("A00", "Hungarian Opening"),

        # Anderssen's Opening (A00)
        ("a3",): ("A00", "Anderssen's Opening"),
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
