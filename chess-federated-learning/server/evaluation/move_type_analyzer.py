"""
Move Type Distribution Analyzer for Chess Games.

This module analyzes the types of moves made in chess games to quantify
tactical vs positional playing style through move classification.

Move Types:
    - Captures: Moves that take opponent pieces
    - Checks: Moves that give check to opponent king
    - Pawn Advances: Non-capture pawn moves
    - Piece Development: Knight/Bishop moves in opening (plies 1-20)
    - Castling: Kingside or queenside castling
    - Quiet Moves: Non-capture, non-check moves (positional)
    - Aggressive Moves: Captures + Checks (tactical)

Expected Patterns:
    - Tactical cluster: Higher % of captures, checks, aggressive moves
    - Positional cluster: Higher % of quiet moves, pawn advances
"""

import chess
import chess.pgn
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional
from loguru import logger


@dataclass
class MoveTypeStats:
    """Statistics for move types in a game."""
    # Counts
    total_moves: int = 0
    captures: int = 0
    checks: int = 0
    pawn_advances: int = 0
    piece_development: int = 0  # N/B moves in opening
    castling: int = 0
    quiet_moves: int = 0

    # Derived (computed)
    aggressive_moves: int = 0  # captures + checks

    # By color
    white_captures: int = 0
    white_checks: int = 0
    white_pawn_advances: int = 0
    white_quiet_moves: int = 0
    black_captures: int = 0
    black_checks: int = 0
    black_pawn_advances: int = 0
    black_quiet_moves: int = 0

    def compute_derived(self):
        """Compute derived statistics."""
        self.aggressive_moves = self.captures + self.checks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with percentages."""
        self.compute_derived()

        total = max(self.total_moves, 1)  # Avoid division by zero

        return {
            "counts": {
                "total_moves": self.total_moves,
                "captures": self.captures,
                "checks": self.checks,
                "pawn_advances": self.pawn_advances,
                "piece_development": self.piece_development,
                "castling": self.castling,
                "quiet_moves": self.quiet_moves,
                "aggressive_moves": self.aggressive_moves
            },
            "percentages": {
                "captures_pct": round(self.captures / total * 100, 2),
                "checks_pct": round(self.checks / total * 100, 2),
                "pawn_advances_pct": round(self.pawn_advances / total * 100, 2),
                "piece_development_pct": round(self.piece_development / total * 100, 2),
                "castling_pct": round(self.castling / total * 100, 2),
                "quiet_moves_pct": round(self.quiet_moves / total * 100, 2),
                "aggressive_pct": round(self.aggressive_moves / total * 100, 2)
            },
            "by_color": {
                "white": {
                    "captures": self.white_captures,
                    "checks": self.white_checks,
                    "pawn_advances": self.white_pawn_advances,
                    "quiet_moves": self.white_quiet_moves
                },
                "black": {
                    "captures": self.black_captures,
                    "checks": self.black_checks,
                    "pawn_advances": self.black_pawn_advances,
                    "quiet_moves": self.black_quiet_moves
                }
            }
        }


@dataclass
class ClusterMoveTypeMetrics:
    """Aggregated move type metrics for a cluster."""
    cluster_id: str
    games_analyzed: int = 0

    # Aggregated counts
    total_moves: int = 0
    total_captures: int = 0
    total_checks: int = 0
    total_pawn_advances: int = 0
    total_quiet_moves: int = 0
    total_aggressive: int = 0

    # Per-game averages (computed)
    avg_captures_per_game: float = 0.0
    avg_checks_per_game: float = 0.0
    avg_aggressive_per_game: float = 0.0

    def add_game_stats(self, stats: MoveTypeStats):
        """Add statistics from a single game."""
        self.games_analyzed += 1
        self.total_moves += stats.total_moves
        self.total_captures += stats.captures
        self.total_checks += stats.checks
        self.total_pawn_advances += stats.pawn_advances
        self.total_quiet_moves += stats.quiet_moves
        stats.compute_derived()
        self.total_aggressive += stats.aggressive_moves

    def compute_averages(self):
        """Compute per-game averages."""
        if self.games_analyzed > 0:
            self.avg_captures_per_game = self.total_captures / self.games_analyzed
            self.avg_checks_per_game = self.total_checks / self.games_analyzed
            self.avg_aggressive_per_game = self.total_aggressive / self.games_analyzed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.compute_averages()

        total = max(self.total_moves, 1)

        return {
            "cluster_id": self.cluster_id,
            "games_analyzed": self.games_analyzed,
            "totals": {
                "total_moves": self.total_moves,
                "captures": self.total_captures,
                "checks": self.total_checks,
                "pawn_advances": self.total_pawn_advances,
                "quiet_moves": self.total_quiet_moves,
                "aggressive_moves": self.total_aggressive
            },
            "percentages": {
                "captures_pct": round(self.total_captures / total * 100, 2),
                "checks_pct": round(self.total_checks / total * 100, 2),
                "pawn_advances_pct": round(self.total_pawn_advances / total * 100, 2),
                "quiet_moves_pct": round(self.total_quiet_moves / total * 100, 2),
                "aggressive_pct": round(self.total_aggressive / total * 100, 2)
            },
            "averages_per_game": {
                "avg_captures": round(self.avg_captures_per_game, 2),
                "avg_checks": round(self.avg_checks_per_game, 2),
                "avg_aggressive": round(self.avg_aggressive_per_game, 2)
            }
        }


class MoveTypeAnalyzer:
    """
    Analyze chess games for move type distribution.

    Classifies each move into categories to understand
    tactical vs positional playing patterns.
    """

    # Opening phase (for piece development tracking)
    OPENING_PLY_LIMIT = 20

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze_game(self, pgn_string: str) -> MoveTypeStats:
        """
        Analyze a single game for move type distribution.

        Args:
            pgn_string: PGN string of the game

        Returns:
            MoveTypeStats with move classifications
        """
        stats = MoveTypeStats()

        # Parse PGN
        pgn_io = StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            logger.warning("Invalid PGN string, returning empty stats")
            return stats

        # Play through the game
        board = game.board()
        ply = 0

        for move in game.mainline_moves():
            ply += 1
            is_white = board.turn == chess.WHITE

            # Classify the move
            move_types = self._classify_move(board, move, ply)

            # Update stats
            stats.total_moves += 1

            if move_types["is_capture"]:
                stats.captures += 1
                if is_white:
                    stats.white_captures += 1
                else:
                    stats.black_captures += 1

            if move_types["is_check"]:
                stats.checks += 1
                if is_white:
                    stats.white_checks += 1
                else:
                    stats.black_checks += 1

            if move_types["is_pawn_advance"]:
                stats.pawn_advances += 1
                if is_white:
                    stats.white_pawn_advances += 1
                else:
                    stats.black_pawn_advances += 1

            if move_types["is_piece_development"]:
                stats.piece_development += 1

            if move_types["is_castling"]:
                stats.castling += 1

            if move_types["is_quiet"]:
                stats.quiet_moves += 1
                if is_white:
                    stats.white_quiet_moves += 1
                else:
                    stats.black_quiet_moves += 1

            # Make the move
            board.push(move)

        stats.compute_derived()
        return stats

    def _classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        ply: int
    ) -> Dict[str, bool]:
        """
        Classify a move into types.

        Args:
            board: Board state before the move
            move: The move to classify
            ply: Current ply number

        Returns:
            Dictionary of move type flags
        """
        result = {
            "is_capture": False,
            "is_check": False,
            "is_pawn_advance": False,
            "is_piece_development": False,
            "is_castling": False,
            "is_quiet": False
        }

        piece = board.piece_at(move.from_square)
        if piece is None:
            return result

        # Check for capture
        result["is_capture"] = board.is_capture(move)

        # Check if move gives check
        board_copy = board.copy()
        board_copy.push(move)
        result["is_check"] = board_copy.is_check()

        # Check for pawn advance (non-capture)
        if piece.piece_type == chess.PAWN and not result["is_capture"]:
            result["is_pawn_advance"] = True

        # Check for piece development (N/B in opening)
        if ply <= self.OPENING_PLY_LIMIT:
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                result["is_piece_development"] = True

        # Check for castling
        if board.is_castling(move):
            result["is_castling"] = True

        # Quiet move: not capture, not check
        if not result["is_capture"] and not result["is_check"]:
            result["is_quiet"] = True

        return result

    def analyze_games(
        self,
        pgn_strings: List[str],
        cluster_id: str
    ) -> ClusterMoveTypeMetrics:
        """
        Analyze multiple games for a cluster.

        Args:
            pgn_strings: List of PGN strings
            cluster_id: Cluster identifier

        Returns:
            ClusterMoveTypeMetrics with aggregated statistics
        """
        cluster_metrics = ClusterMoveTypeMetrics(cluster_id=cluster_id)

        for pgn in pgn_strings:
            try:
                game_stats = self.analyze_game(pgn)
                cluster_metrics.add_game_stats(game_stats)
            except Exception as e:
                logger.warning(f"Failed to analyze game: {e}")
                continue

        cluster_metrics.compute_averages()

        logger.info(
            f"Analyzed {cluster_metrics.games_analyzed} games for {cluster_id}: "
            f"captures={cluster_metrics.total_captures}, "
            f"checks={cluster_metrics.total_checks}, "
            f"aggressive_pct={cluster_metrics.total_aggressive / max(cluster_metrics.total_moves, 1) * 100:.1f}%"
        )

        return cluster_metrics


def compute_move_type_comparison(
    cluster_metrics: Dict[str, ClusterMoveTypeMetrics]
) -> Dict[str, Any]:
    """
    Compare move type distributions between clusters.

    Args:
        cluster_metrics: Dict of cluster_id -> ClusterMoveTypeMetrics

    Returns:
        Comparison metrics including divergence
    """
    if len(cluster_metrics) < 2:
        return {"error": "Need at least 2 clusters for comparison"}

    cluster_ids = list(cluster_metrics.keys())

    # Get percentages for each cluster
    cluster_pcts = {}
    for cid, metrics in cluster_metrics.items():
        metrics.compute_averages()
        total = max(metrics.total_moves, 1)
        cluster_pcts[cid] = {
            "captures_pct": metrics.total_captures / total * 100,
            "checks_pct": metrics.total_checks / total * 100,
            "pawn_advances_pct": metrics.total_pawn_advances / total * 100,
            "quiet_moves_pct": metrics.total_quiet_moves / total * 100,
            "aggressive_pct": metrics.total_aggressive / total * 100
        }

    # Compute differences between first two clusters
    c1, c2 = cluster_ids[0], cluster_ids[1]

    differences = {
        "clusters_compared": [c1, c2],
        "differences": {
            "captures_diff": round(cluster_pcts[c1]["captures_pct"] - cluster_pcts[c2]["captures_pct"], 2),
            "checks_diff": round(cluster_pcts[c1]["checks_pct"] - cluster_pcts[c2]["checks_pct"], 2),
            "pawn_advances_diff": round(cluster_pcts[c1]["pawn_advances_pct"] - cluster_pcts[c2]["pawn_advances_pct"], 2),
            "quiet_moves_diff": round(cluster_pcts[c1]["quiet_moves_pct"] - cluster_pcts[c2]["quiet_moves_pct"], 2),
            "aggressive_diff": round(cluster_pcts[c1]["aggressive_pct"] - cluster_pcts[c2]["aggressive_pct"], 2)
        },
        "interpretation": {
            f"{c1}_more_tactical": cluster_pcts[c1]["aggressive_pct"] > cluster_pcts[c2]["aggressive_pct"],
            "tactical_cluster": c1 if cluster_pcts[c1]["aggressive_pct"] > cluster_pcts[c2]["aggressive_pct"] else c2,
            "positional_cluster": c2 if cluster_pcts[c1]["aggressive_pct"] > cluster_pcts[c2]["aggressive_pct"] else c1
        }
    }

    return differences
