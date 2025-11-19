"""
Playstyle metrics calculator for chess games.

This module implements tactical vs. positional playstyle analysis based on
the methodology from novachess.ai. It analyzes chess games to classify
playing styles on a spectrum from "Very Positional" to "Very Tactical".

Key Metrics:
    - Attacked Material: Total value of capturable pieces
    - Legal Moves: Number of legal moves available (tracked per phase)
    - Material Captured: Total material actually captured
    - Center Control: Pieces controlling center squares (d4, d5, e4, e5)
    - Tactical Score: Weighted combination of metrics (0.0 = positional, 1.0 = tactical)
    - Delta (Tipping Points): Difference between best and 2nd best move (sparse sampling)
    - Pawn Structure: Average pawn rank, isolated/doubled pawns
    - Move Diversity: Unique destination squares targeted

Analysis Range:
    - Positions analyzed: Plies 12-50 (moves 6-25) for core metrics
    - Legal moves tracked: All game (by phase: opening 1-12, middlegame 13-40, endgame 41+)
    - Captures tracked: Plies 1-50 (moves 1-25)
    - Delta analysis: Every 3rd middlegame position (plies 15-40)
    - Positions in check: Skipped for core metrics

Classification:
    - Very Tactical: > 0.70
    - Tactical: 0.65-0.70
    - Balanced: 0.60-0.65
    - Positional: 0.50-0.60
    - Very Positional: < 0.50
"""

import chess
import chess.pgn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO
from loguru import logger
from collections import Counter


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King is not capturable in normal play
}

# Center squares for center control metric
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]

# Game phases by ply
OPENING_PLIES = (1, 12)
MIDDLEGAME_PLIES = (13, 40)
ENDGAME_PLIES = (41, 999)


@dataclass
class PositionMetrics:
    """Metrics for a single board position."""
    ply: int
    attacked_material_white: int = 0
    attacked_material_black: int = 0
    legal_moves_white: int = 0
    legal_moves_black: int = 0
    center_control_white: int = 0
    center_control_black: int = 0
    is_check: bool = False
    fen: Optional[str] = None


@dataclass
class DeltaMetric:
    """Delta (tipping point) metric for a position."""
    ply: int
    delta_white: Optional[float] = None  # Centipawn difference between best and 2nd best
    delta_black: Optional[float] = None
    best_move_cp: Optional[float] = None  # Centipawn evaluation of best move


@dataclass
class PawnStructureMetrics:
    """Pawn structure metrics for a position."""
    ply: int
    avg_pawn_rank_white: float = 0.0  # Average rank of white pawns (1-8)
    avg_pawn_rank_black: float = 0.0  # Average rank of black pawns (1-8)
    isolated_pawns_white: int = 0
    isolated_pawns_black: int = 0
    doubled_pawns_white: int = 0
    doubled_pawns_black: int = 0


@dataclass
class CaptureEvent:
    """Record of a capture in the game."""
    ply: int
    piece_value: int
    capturing_side: str  # "white" or "black"


@dataclass
class GameMetrics:
    """Raw metrics collected from a single game."""
    white_cluster: str
    black_cluster: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str
    total_plies: int
    pgn: str

    # Collected metrics
    position_metrics: List[PositionMetrics] = field(default_factory=list)
    captures: List[CaptureEvent] = field(default_factory=list)

    # New enhanced metrics
    delta_metrics: List[DeltaMetric] = field(default_factory=list)
    pawn_structure_metrics: List[PawnStructureMetrics] = field(default_factory=list)

    # Move diversity tracking (per color)
    unique_destinations_white: set = field(default_factory=set)
    unique_destinations_black: set = field(default_factory=set)

    # Opening analysis
    eco_code: Optional[str] = None
    opening_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "white_cluster": self.white_cluster,
            "black_cluster": self.black_cluster,
            "result": self.result,
            "termination": self.termination,
            "total_plies": self.total_plies,
            "pgn": self.pgn,
            "eco_code": self.eco_code,
            "opening_name": self.opening_name,
            "position_metrics": [
                {
                    "ply": pm.ply,
                    "attacked_material_white": pm.attacked_material_white,
                    "attacked_material_black": pm.attacked_material_black,
                    "legal_moves_white": pm.legal_moves_white,
                    "legal_moves_black": pm.legal_moves_black,
                    "center_control_white": pm.center_control_white,
                    "center_control_black": pm.center_control_black,
                    "is_check": pm.is_check,
                    "fen": pm.fen
                }
                for pm in self.position_metrics
            ],
            "captures": [
                {
                    "ply": ce.ply,
                    "piece_value": ce.piece_value,
                    "capturing_side": ce.capturing_side
                }
                for ce in self.captures
            ],
            "delta_metrics": [
                {
                    "ply": dm.ply,
                    "delta_white": dm.delta_white,
                    "delta_black": dm.delta_black,
                    "best_move_cp": dm.best_move_cp
                }
                for dm in self.delta_metrics
            ],
            "pawn_structure_metrics": [
                {
                    "ply": psm.ply,
                    "avg_pawn_rank_white": round(psm.avg_pawn_rank_white, 2),
                    "avg_pawn_rank_black": round(psm.avg_pawn_rank_black, 2),
                    "isolated_pawns_white": psm.isolated_pawns_white,
                    "isolated_pawns_black": psm.isolated_pawns_black,
                    "doubled_pawns_white": psm.doubled_pawns_white,
                    "doubled_pawns_black": psm.doubled_pawns_black
                }
                for psm in self.pawn_structure_metrics
            ],
            "move_diversity": {
                "unique_destinations_white": len(self.unique_destinations_white),
                "unique_destinations_black": len(self.unique_destinations_black)
            }
        }


@dataclass
class PlayerComputedMetrics:
    """Computed metrics for one player in a game."""
    # Raw aggregates
    total_attacked_material: float = 0.0
    total_legal_moves: int = 0
    total_captures: int = 0
    avg_center_control: float = 0.0
    positions_analyzed: int = 0

    # Legal moves per phase
    legal_moves_opening: int = 0
    legal_moves_middlegame: int = 0
    legal_moves_endgame: int = 0
    avg_legal_moves_opening: float = 0.0
    avg_legal_moves_middlegame: float = 0.0
    avg_legal_moves_endgame: float = 0.0

    # Delta metrics (tipping points)
    avg_delta: float = 0.0
    max_delta: float = 0.0
    min_delta: float = 0.0
    delta_samples: int = 0

    # Pawn structure
    avg_pawn_rank: float = 0.0
    avg_isolated_pawns: float = 0.0
    avg_doubled_pawns: float = 0.0

    # Move diversity
    unique_move_destinations: int = 0
    move_diversity_ratio: float = 0.0  # unique destinations / total moves

    # Normalized metrics
    attacks_metric: float = 0.0
    moves_metric: float = 0.0
    material_metric: float = 0.0

    # Final score
    tactical_score: float = 0.0
    classification: str = "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "total_attacked_material": self.total_attacked_material,
            "total_legal_moves": self.total_legal_moves,
            "total_captures": self.total_captures,
            "avg_center_control": self.avg_center_control,
            "positions_analyzed": self.positions_analyzed,
            "legal_moves_opening": self.legal_moves_opening,
            "legal_moves_middlegame": self.legal_moves_middlegame,
            "legal_moves_endgame": self.legal_moves_endgame,
            "avg_legal_moves_opening": round(self.avg_legal_moves_opening, 2),
            "avg_legal_moves_middlegame": round(self.avg_legal_moves_middlegame, 2),
            "avg_legal_moves_endgame": round(self.avg_legal_moves_endgame, 2),
            "avg_delta": round(self.avg_delta, 2),
            "max_delta": round(self.max_delta, 2),
            "min_delta": round(self.min_delta, 2),
            "delta_samples": self.delta_samples,
            "avg_pawn_rank": round(self.avg_pawn_rank, 2),
            "avg_isolated_pawns": round(self.avg_isolated_pawns, 2),
            "avg_doubled_pawns": round(self.avg_doubled_pawns, 2),
            "unique_move_destinations": self.unique_move_destinations,
            "move_diversity_ratio": round(self.move_diversity_ratio, 3),
            "attacks_metric": self.attacks_metric,
            "moves_metric": self.moves_metric,
            "material_metric": self.material_metric,
            "tactical_score": self.tactical_score,
            "classification": self.classification
        }


@dataclass
class ComputedGameMetrics:
    """Computed playstyle metrics for a game."""
    white_metrics: PlayerComputedMetrics
    black_metrics: PlayerComputedMetrics
    game_characteristics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "white_metrics": self.white_metrics.to_dict(),
            "black_metrics": self.black_metrics.to_dict(),
            "game_characteristics": self.game_characteristics
        }


class GameAnalyzer:
    """
    Analyze individual chess games for playstyle metrics.

    This class plays through a game and collects position-by-position
    metrics for playstyle analysis.
    """

    def __init__(
        self,
        skip_check_positions: bool = True,
        enable_delta_analysis: bool = True,
        delta_sampling_rate: int = 3,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 12
    ):
        """
        Initialize the analyzer.

        Args:
            skip_check_positions: If True, skip positions where either player is in check
            enable_delta_analysis: If True, compute delta (tipping point) metrics
            delta_sampling_rate: Analyze every Nth position (3 = every 3rd position)
            stockfish_path: Path to Stockfish executable (None = auto-detect)
            stockfish_depth: Search depth for Stockfish analysis (12-15 recommended for speed)
        """
        self.skip_check_positions = skip_check_positions
        self.enable_delta_analysis = enable_delta_analysis
        self.delta_sampling_rate = delta_sampling_rate
        self.stockfish_depth = stockfish_depth

        # Initialize Stockfish if delta analysis is enabled
        self.stockfish = None
        if self.enable_delta_analysis:
            try:
                import chess.engine
                if stockfish_path:
                    self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                else:
                    # Try common paths
                    for path in ["/usr/local/bin/stockfish", "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish", "stockfish"]:
                        try:
                            self.stockfish = chess.engine.SimpleEngine.popen_uci(path)
                            break
                        except Exception:
                            continue

                if self.stockfish:
                    logger.info(f"Stockfish initialized for delta analysis (depth {stockfish_depth})")
                else:
                    logger.warning("Could not initialize Stockfish - delta analysis will be disabled")
                    self.enable_delta_analysis = False
            except ImportError:
                logger.warning("chess.engine not available - delta analysis will be disabled")
                self.enable_delta_analysis = False

    def __del__(self):
        """Clean up Stockfish engine."""
        if self.stockfish:
            try:
                self.stockfish.quit()
            except Exception:
                pass

    def analyze_game(
        self,
        pgn_string: str,
        white_cluster: str,
        black_cluster: str
    ) -> GameMetrics:
        """
        Analyze a game from PGN string.

        Args:
            pgn_string: PGN string of the game
            white_cluster: Cluster ID of white player
            black_cluster: Cluster ID of black player

        Returns:
            GameMetrics with collected position and capture data
        """
        log = logger.bind(context="GameAnalyzer.analyze_game")

        # Parse PGN
        pgn_io = StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            raise ValueError("Invalid PGN string")

        # Get game result and metadata
        result = game.headers.get("Result", "*")
        termination = game.headers.get("Termination", "unknown")

        # Extract opening information
        eco_code = game.headers.get("ECO")
        opening_name = game.headers.get("Opening")

        # If not in headers, try to classify from moves
        if not eco_code or not opening_name:
            from server.evaluation.opening_classifier import OpeningClassifier
            classified_eco, classified_name = OpeningClassifier.classify_from_pgn(pgn_string)
            if classified_eco:
                eco_code = classified_eco
                opening_name = classified_name

        # Initialize metrics
        metrics = GameMetrics(
            white_cluster=white_cluster,
            black_cluster=black_cluster,
            result=result,
            termination=termination,
            total_plies=0,
            pgn=pgn_string,
            eco_code=eco_code,
            opening_name=opening_name
        )

        # Play through the game
        board = game.board()
        ply = 0
        delta_counter = 0  # For sparse sampling

        for move in game.mainline_moves():
            ply += 1
            move_before = board.turn  # Track whose move it was

            # Track move diversity (destination squares)
            if move_before == chess.WHITE:
                metrics.unique_destinations_white.add(move.to_square)
            else:
                metrics.unique_destinations_black.add(move.to_square)

            # Track captures (plies 1-50)
            if ply <= 50 and board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
                    capturing_side = "white" if move_before == chess.WHITE else "black"
                    metrics.captures.append(CaptureEvent(
                        ply=ply,
                        piece_value=piece_value,
                        capturing_side=capturing_side
                    ))

            # Make the move
            board.push(move)

            # Analyze position (plies 12-50 for core metrics)
            if 12 <= ply <= 50:
                # Skip positions in check if configured
                if self.skip_check_positions and board.is_check():
                    continue

                position_metrics = self._analyze_position(board, ply)
                metrics.position_metrics.append(position_metrics)

            # Delta analysis (sparse sampling in middlegame: plies 15-40)
            if (self.enable_delta_analysis and
                MIDDLEGAME_PLIES[0] <= ply <= MIDDLEGAME_PLIES[1] and
                delta_counter % self.delta_sampling_rate == 0):
                try:
                    delta_metric = self._analyze_delta(board, ply)
                    if delta_metric:
                        metrics.delta_metrics.append(delta_metric)
                except Exception as e:
                    log.debug(f"Delta analysis failed at ply {ply}: {e}")

            delta_counter += 1

            # Pawn structure analysis (sample every 5 plies after opening)
            if ply > OPENING_PLIES[1] and ply % 5 == 0:
                pawn_metrics = self._analyze_pawn_structure(board, ply)
                metrics.pawn_structure_metrics.append(pawn_metrics)

        metrics.total_plies = ply

        log.debug(f"Analyzed game: {ply} plies, {len(metrics.position_metrics)} positions, "
                 f"{len(metrics.captures)} captures, {len(metrics.delta_metrics)} delta samples, "
                 f"{len(metrics.pawn_structure_metrics)} pawn structure samples")

        return metrics

    def _analyze_position(self, board: chess.Board, ply: int) -> PositionMetrics:
        """
        Analyze a single position.

        Args:
            board: Current board state
            ply: Current ply number

        Returns:
            PositionMetrics for this position
        """
        metrics = PositionMetrics(ply=ply, is_check=board.is_check())

        # Store FEN for reference
        metrics.fen = board.fen()

        # Calculate metrics for current position
        # Note: We need to analyze from both perspectives

        # Analyze for the side to move
        current_turn = board.turn

        # Count legal moves for current player
        legal_moves_count = len(list(board.legal_moves))

        # Calculate attacked material for current player
        attacked_material = self._calculate_attacked_material(board)

        # Calculate center control for both sides
        center_control_white, center_control_black = self._calculate_center_control(board)

        # Store metrics based on whose turn it is
        if current_turn == chess.WHITE:
            metrics.legal_moves_white = legal_moves_count
            metrics.attacked_material_white = attacked_material
        else:
            metrics.legal_moves_black = legal_moves_count
            metrics.attacked_material_black = attacked_material

        metrics.center_control_white = center_control_white
        metrics.center_control_black = center_control_black

        return metrics

    def _calculate_attacked_material(self, board: chess.Board) -> int:
        """
        Calculate total value of opponent pieces that can be captured.

        Args:
            board: Current board state

        Returns:
            Sum of capturable piece values
        """
        attacked_value = 0

        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    attacked_value += PIECE_VALUES.get(captured_piece.piece_type, 0)

        return attacked_value

    def _calculate_center_control(self, board: chess.Board) -> Tuple[int, int]:
        """
        Calculate center control for both sides.

        Center squares: d4, d5, e4, e5
        Control = number of pieces attacking/occupying these squares

        Args:
            board: Current board state

        Returns:
            Tuple of (white_control, black_control)
        """
        white_control = 0
        black_control = 0

        for square in CENTER_SQUARES:
            # Count attackers for each center square
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))

            white_control += white_attackers
            black_control += black_attackers

        return white_control, black_control

    def _analyze_delta(self, board: chess.Board, ply: int) -> Optional[DeltaMetric]:
        """
        Analyze position to compute delta (tipping point) metric.

        Delta = difference between best move evaluation and 2nd best move evaluation.
        High delta indicates forcing/tactical positions, low delta indicates flexible positions.

        Args:
            board: Current board state
            ply: Current ply number

        Returns:
            DeltaMetric or None if analysis fails
        """
        if not self.stockfish:
            return None

        try:
            import chess.engine

            # Analyze with multi-PV to get top 2 moves
            info = self.stockfish.analyse(
                board,
                chess.engine.Limit(depth=self.stockfish_depth),
                multipv=2
            )

            if len(info) < 2:
                # Not enough moves available
                return None

            # Extract scores for top 2 moves
            score1 = info[0]["score"].white()
            score2 = info[1]["score"].white()

            # Convert to centipawns (handle mate scores)
            if score1.is_mate():
                cp1 = 10000 if score1.mate() > 0 else -10000
            else:
                cp1 = score1.score()

            if score2.is_mate():
                cp2 = 10000 if score2.mate() > 0 else -10000
            else:
                cp2 = score2.score()

            # Calculate delta
            delta = abs(cp1 - cp2) / 100.0  # Convert to pawns

            # Store based on whose turn it is
            metric = DeltaMetric(ply=ply, best_move_cp=cp1 / 100.0)
            if board.turn == chess.WHITE:
                metric.delta_white = delta
            else:
                metric.delta_black = delta

            return metric

        except Exception as e:
            logger.debug(f"Delta analysis failed: {e}")
            return None

    def _analyze_pawn_structure(self, board: chess.Board, ply: int) -> PawnStructureMetrics:
        """
        Analyze pawn structure metrics.

        Args:
            board: Current board state
            ply: Current ply number

        Returns:
            PawnStructureMetrics
        """
        metrics = PawnStructureMetrics(ply=ply)

        # Get pawn positions
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        # Average pawn rank
        if white_pawns:
            metrics.avg_pawn_rank_white = sum(chess.square_rank(sq) + 1 for sq in white_pawns) / len(white_pawns)

        if black_pawns:
            metrics.avg_pawn_rank_black = sum(chess.square_rank(sq) + 1 for sq in black_pawns) / len(black_pawns)

        # Isolated and doubled pawns
        for color, pawns, is_white in [(chess.WHITE, white_pawns, True), (chess.BLACK, black_pawns, False)]:
            # Track pawns by file
            file_counts = {}
            for sq in pawns:
                file = chess.square_file(sq)
                file_counts[file] = file_counts.get(file, 0) + 1

            # Count doubled pawns (more than 1 pawn on same file)
            doubled = sum(count - 1 for count in file_counts.values() if count > 1)

            # Count isolated pawns (no friendly pawns on adjacent files)
            isolated = 0
            for file in file_counts.keys():
                has_neighbor = False
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file <= 7 and adj_file in file_counts:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    isolated += file_counts[file]

            if is_white:
                metrics.doubled_pawns_white = doubled
                metrics.isolated_pawns_white = isolated
            else:
                metrics.doubled_pawns_black = doubled
                metrics.isolated_pawns_black = isolated

        return metrics


class PlaystyleMetricsCalculator:
    """
    Calculate playstyle metrics from game data.

    This class takes raw game metrics and computes normalized metrics,
    tactical scores, and classifications.
    """

    @staticmethod
    def classify_tactical_score(score: float) -> str:
        """
        Classify a tactical score into a category.

        Args:
            score: Tactical score (0.0 to 1.0+)

        Returns:
            Classification string
        """
        if score > 0.70:
            return "Very Tactical"
        elif score >= 0.65:
            return "Tactical"
        elif score >= 0.60:
            return "Balanced"
        elif score >= 0.50:
            return "Positional"
        else:
            return "Very Positional"

    @staticmethod
    def compute_game_metrics(game_metrics: GameMetrics) -> ComputedGameMetrics:
        """
        Compute playstyle metrics from raw game metrics.

        Args:
            game_metrics: Raw metrics from game analysis

        Returns:
            ComputedGameMetrics with normalized metrics and scores
        """
        log = logger.bind(context="PlaystyleMetricsCalculator.compute_game_metrics")

        # Separate metrics by color
        white_metrics = PlayerComputedMetrics()
        black_metrics = PlayerComputedMetrics()

        # Count positions analyzed
        positions_count = len(game_metrics.position_metrics)
        white_metrics.positions_analyzed = positions_count
        black_metrics.positions_analyzed = positions_count

        if positions_count == 0:
            log.warning("No positions to analyze (game too short or all positions in check)")
            return ComputedGameMetrics(
                white_metrics=white_metrics,
                black_metrics=black_metrics,
                game_characteristics={"error": "no_positions_analyzed"}
            )

        # Track phase counters for averaging
        opening_count_white = opening_count_black = 0
        middlegame_count_white = middlegame_count_black = 0
        endgame_count_white = endgame_count_black = 0

        # Aggregate raw metrics (including per-phase legal moves)
        for pm in game_metrics.position_metrics:
            white_metrics.total_attacked_material += pm.attacked_material_white
            white_metrics.total_legal_moves += pm.legal_moves_white
            black_metrics.total_attacked_material += pm.attacked_material_black
            black_metrics.total_legal_moves += pm.legal_moves_black

            white_metrics.avg_center_control += pm.center_control_white
            black_metrics.avg_center_control += pm.center_control_black

            # Track legal moves by phase
            ply = pm.ply
            if OPENING_PLIES[0] <= ply <= OPENING_PLIES[1]:
                if pm.legal_moves_white > 0:
                    white_metrics.legal_moves_opening += pm.legal_moves_white
                    opening_count_white += 1
                if pm.legal_moves_black > 0:
                    black_metrics.legal_moves_opening += pm.legal_moves_black
                    opening_count_black += 1
            elif MIDDLEGAME_PLIES[0] <= ply <= MIDDLEGAME_PLIES[1]:
                if pm.legal_moves_white > 0:
                    white_metrics.legal_moves_middlegame += pm.legal_moves_white
                    middlegame_count_white += 1
                if pm.legal_moves_black > 0:
                    black_metrics.legal_moves_middlegame += pm.legal_moves_black
                    middlegame_count_black += 1
            elif ply >= ENDGAME_PLIES[0]:
                if pm.legal_moves_white > 0:
                    white_metrics.legal_moves_endgame += pm.legal_moves_white
                    endgame_count_white += 1
                if pm.legal_moves_black > 0:
                    black_metrics.legal_moves_endgame += pm.legal_moves_black
                    endgame_count_black += 1

        # Calculate average center control
        white_metrics.avg_center_control /= positions_count
        black_metrics.avg_center_control /= positions_count

        # Calculate average legal moves per phase
        if opening_count_white > 0:
            white_metrics.avg_legal_moves_opening = white_metrics.legal_moves_opening / opening_count_white
        if middlegame_count_white > 0:
            white_metrics.avg_legal_moves_middlegame = white_metrics.legal_moves_middlegame / middlegame_count_white
        if endgame_count_white > 0:
            white_metrics.avg_legal_moves_endgame = white_metrics.legal_moves_endgame / endgame_count_white

        if opening_count_black > 0:
            black_metrics.avg_legal_moves_opening = black_metrics.legal_moves_opening / opening_count_black
        if middlegame_count_black > 0:
            black_metrics.avg_legal_moves_middlegame = black_metrics.legal_moves_middlegame / middlegame_count_black
        if endgame_count_black > 0:
            black_metrics.avg_legal_moves_endgame = black_metrics.legal_moves_endgame / endgame_count_black

        # Count captures through ply 50
        for capture in game_metrics.captures:
            if capture.capturing_side == "white":
                white_metrics.total_captures += capture.piece_value
            else:
                black_metrics.total_captures += capture.piece_value

        # Aggregate delta metrics
        white_deltas = []
        black_deltas = []
        for dm in game_metrics.delta_metrics:
            if dm.delta_white is not None:
                white_deltas.append(dm.delta_white)
            if dm.delta_black is not None:
                black_deltas.append(dm.delta_black)

        if white_deltas:
            white_metrics.avg_delta = sum(white_deltas) / len(white_deltas)
            white_metrics.max_delta = max(white_deltas)
            white_metrics.min_delta = min(white_deltas)
            white_metrics.delta_samples = len(white_deltas)

        if black_deltas:
            black_metrics.avg_delta = sum(black_deltas) / len(black_deltas)
            black_metrics.max_delta = max(black_deltas)
            black_metrics.min_delta = min(black_deltas)
            black_metrics.delta_samples = len(black_deltas)

        # Aggregate pawn structure metrics
        if game_metrics.pawn_structure_metrics:
            white_pawn_ranks = []
            black_pawn_ranks = []
            white_isolated = []
            black_isolated = []
            white_doubled = []
            black_doubled = []

            for psm in game_metrics.pawn_structure_metrics:
                if psm.avg_pawn_rank_white > 0:
                    white_pawn_ranks.append(psm.avg_pawn_rank_white)
                if psm.avg_pawn_rank_black > 0:
                    black_pawn_ranks.append(psm.avg_pawn_rank_black)
                white_isolated.append(psm.isolated_pawns_white)
                black_isolated.append(psm.isolated_pawns_black)
                white_doubled.append(psm.doubled_pawns_white)
                black_doubled.append(psm.doubled_pawns_black)

            if white_pawn_ranks:
                white_metrics.avg_pawn_rank = sum(white_pawn_ranks) / len(white_pawn_ranks)
            if black_pawn_ranks:
                black_metrics.avg_pawn_rank = sum(black_pawn_ranks) / len(black_pawn_ranks)

            if white_isolated:
                white_metrics.avg_isolated_pawns = sum(white_isolated) / len(white_isolated)
            if black_isolated:
                black_metrics.avg_isolated_pawns = sum(black_isolated) / len(black_isolated)

            if white_doubled:
                white_metrics.avg_doubled_pawns = sum(white_doubled) / len(white_doubled)
            if black_doubled:
                black_metrics.avg_doubled_pawns = sum(black_doubled) / len(black_doubled)

        # Move diversity
        white_metrics.unique_move_destinations = len(game_metrics.unique_destinations_white)
        black_metrics.unique_move_destinations = len(game_metrics.unique_destinations_black)

        # Calculate move diversity ratio (unique destinations / total moves in game)
        total_moves = game_metrics.total_plies / 2  # Approximate moves per side
        if total_moves > 0:
            white_metrics.move_diversity_ratio = white_metrics.unique_move_destinations / total_moves
            black_metrics.move_diversity_ratio = black_metrics.unique_move_destinations / total_moves

        # Compute normalized metrics for white
        PlaystyleMetricsCalculator._compute_normalized_metrics(white_metrics)

        # Compute normalized metrics for black
        PlaystyleMetricsCalculator._compute_normalized_metrics(black_metrics)

        # Game characteristics
        avg_tactical_score = (white_metrics.tactical_score + black_metrics.tactical_score) / 2
        tactical_imbalance = abs(white_metrics.tactical_score - black_metrics.tactical_score)

        game_characteristics = {
            "avg_tactical_score": round(avg_tactical_score, 3),
            "tactical_imbalance": round(tactical_imbalance, 3),
            "game_style": PlaystyleMetricsCalculator.classify_tactical_score(avg_tactical_score)
        }

        log.debug(f"Computed metrics: White={white_metrics.tactical_score:.3f} ({white_metrics.classification}), "
                 f"Black={black_metrics.tactical_score:.3f} ({black_metrics.classification})")

        return ComputedGameMetrics(
            white_metrics=white_metrics,
            black_metrics=black_metrics,
            game_characteristics=game_characteristics
        )

    @staticmethod
    def _compute_normalized_metrics(metrics: PlayerComputedMetrics) -> None:
        """
        Compute normalized metrics and tactical score for a player.

        Modifies the metrics object in place.

        Args:
            metrics: PlayerComputedMetrics to compute
        """
        # Attacks metric: sum of attacked material / 39
        metrics.attacks_metric = metrics.total_attacked_material / 39.0

        # Moves metric: sum of legal moves / 40, capped at 1.0
        metrics.moves_metric = min(1.0, metrics.total_legal_moves / 40.0)

        # Material metric: sum of captures / 20, capped at 1.0
        metrics.material_metric = min(1.0, metrics.total_captures / 20.0)

        # Tactical score: average of metrics
        if metrics.material_metric > 0:
            # With captures: average of all three metrics
            metrics.tactical_score = (
                metrics.attacks_metric +
                metrics.moves_metric +
                metrics.material_metric
            ) / 3.0
        else:
            # Without captures: average of attacks and moves only
            metrics.tactical_score = (
                metrics.attacks_metric +
                metrics.moves_metric
            ) / 2.0

        # Classify
        metrics.classification = PlaystyleMetricsCalculator.classify_tactical_score(
            metrics.tactical_score
        )

        # Round for cleaner output
        metrics.tactical_score = round(metrics.tactical_score, 3)
        metrics.attacks_metric = round(metrics.attacks_metric, 3)
        metrics.moves_metric = round(metrics.moves_metric, 3)
        metrics.material_metric = round(metrics.material_metric, 3)
        metrics.avg_center_control = round(metrics.avg_center_control, 2)
