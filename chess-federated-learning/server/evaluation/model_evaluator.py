"""
Model evaluator for playstyle analysis and ELO estimation against Stockfish.

This module orchestrates games between cluster models and Stockfish to evaluate
playstyle characteristics and estimate absolute ELO ratings.

Key Features:
    - Load cluster models from state dicts
    - Play games against Stockfish at various ELO levels
    - Collect game PGNs and metadata
    - Aggregate playstyle metrics across multiple games
    - Estimate absolute ELO ratings for each cluster
    - Run evaluations in parallel for multiple clusters
"""

import asyncio
import chess
import chess.engine
import chess.pgn
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from loguru import logger

# Import from existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.trainer.models.alphazero_net import AlphaZeroNet
from data.board_encoder import BoardEncoder
from data.move_encoder import MoveEncoder
from server.evaluation.playstyle_metrics import (
    GameAnalyzer,
    PlaystyleMetricsCalculator,
    GameMetrics,
    ComputedGameMetrics,
    PlayerComputedMetrics
)
from server.evaluation.move_type_analyzer import (
    MoveTypeAnalyzer,
    ClusterMoveTypeMetrics
)


@dataclass
class MatchResult:
    """Results of games against a specific Stockfish ELO level."""
    opponent: str
    opponent_elo: int
    games_played: int
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def score(self) -> float:
        """Total score (wins + 0.5*draws)."""
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        return (self.score / self.games_played * 100) if self.games_played > 0 else 0


@dataclass
class ClusterEvaluationMetrics:
    """Aggregated metrics for a cluster across multiple games."""
    cluster_id: str
    games_analyzed: int = 0
    games_as_white: int = 0
    games_as_black: int = 0

    # Aggregated raw metrics
    avg_attacked_material: float = 0.0
    avg_legal_moves: float = 0.0
    avg_captures: float = 0.0
    avg_center_control: float = 0.0

    # Aggregated normalized metrics
    avg_attacks_metric: float = 0.0
    avg_moves_metric: float = 0.0
    avg_material_metric: float = 0.0

    # Tactical score statistics
    tactical_score: float = 0.0
    tactical_score_std: float = 0.0
    tactical_score_min: float = float('inf')
    tactical_score_max: float = float('-inf')
    classification: str = "Unknown"

    # Classification distribution
    classification_distribution: Dict[str, int] = field(default_factory=lambda: {
        "Very Tactical": 0,
        "Tactical": 0,
        "Balanced": 0,
        "Positional": 0,
        "Very Positional": 0
    })

    # Game outcomes
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0

    # ELO estimation
    estimated_elo: int = 1500
    elo_confidence: int = 500

    # Match results by opponent
    match_results: List[MatchResult] = field(default_factory=list)

    # Opening analysis
    opening_frequency: Dict[str, int] = field(default_factory=dict)  # ECO code -> count
    top_openings: List[Tuple[str, str, int]] = field(default_factory=list)  # (ECO, name, count)

    # Move type distribution
    move_type_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "games_analyzed": self.games_analyzed,
            "games_as_white": self.games_as_white,
            "games_as_black": self.games_as_black,
            "avg_attacked_material": round(self.avg_attacked_material, 2),
            "avg_legal_moves": round(self.avg_legal_moves, 2),
            "avg_captures": round(self.avg_captures, 2),
            "avg_center_control": round(self.avg_center_control, 2),
            "avg_attacks_metric": round(self.avg_attacks_metric, 3),
            "avg_moves_metric": round(self.avg_moves_metric, 3),
            "avg_material_metric": round(self.avg_material_metric, 3),
            "tactical_score": round(self.tactical_score, 3),
            "tactical_score_std": round(self.tactical_score_std, 3),
            "tactical_score_min": round(self.tactical_score_min, 3) if self.tactical_score_min != float('inf') else 0.0,
            "tactical_score_max": round(self.tactical_score_max, 3) if self.tactical_score_max != float('-inf') else 0.0,
            "classification": self.classification,
            "classification_distribution": self.classification_distribution,
            "win_rate": round(self.win_rate, 3),
            "draw_rate": round(self.draw_rate, 3),
            "loss_rate": round(self.loss_rate, 3),
            "estimated_elo": self.estimated_elo,
            "elo_confidence": self.elo_confidence,
            "match_results": [
                {
                    "opponent": mr.opponent,
                    "opponent_elo": mr.opponent_elo,
                    "games_played": mr.games_played,
                    "wins": mr.wins,
                    "draws": mr.draws,
                    "losses": mr.losses,
                    "score": round(mr.score, 1),
                    "win_rate": round(mr.win_rate, 1)
                }
                for mr in self.match_results
            ],
            "opening_frequency": self.opening_frequency,
            "top_openings": [
                {"eco": eco, "name": name, "count": count}
                for eco, name, count in self.top_openings
            ],
            "move_type_metrics": self.move_type_metrics
        }


class ELOEstimator:
    """Estimate ELO rating based on match results against Stockfish."""

    @staticmethod
    def expected_score(elo_a: int, elo_b: int) -> float:
        """Calculate expected score for player A vs player B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    @staticmethod
    def estimate_elo(results: List[MatchResult], initial_elo: int = 1500) -> Tuple[int, int]:
        """
        Estimate ELO using iterative approximation.

        Args:
            results: List of match results against different Stockfish levels
            initial_elo: Starting ELO estimate

        Returns:
            Tuple of (estimated_elo, confidence_range)
        """
        if not results:
            return initial_elo, 500

        # Simple approach: iterate to find ELO that minimizes error
        best_elo = initial_elo
        min_error = float('inf')

        for test_elo in range(800, 2400, 25):
            total_error = 0

            for match in results:
                expected = ELOEstimator.expected_score(test_elo, match.opponent_elo)
                actual = match.score / match.games_played
                total_error += (expected - actual) ** 2

            if total_error < min_error:
                min_error = total_error
                best_elo = test_elo

        # Estimate confidence based on number of games
        total_games = sum(m.games_played for m in results)
        confidence_range = max(50, 400 - total_games * 10)

        return best_elo, confidence_range


class ChessAIPlayer:
    """AI player using trained AlphaZero model for evaluation games."""

    def __init__(
        self,
        model: AlphaZeroNet,
        cluster_id: str,
        device: str = 'cpu'
    ):
        """
        Initialize AI player.

        Args:
            model: Trained AlphaZero model
            cluster_id: ID of the cluster this model belongs to
            device: Device to run model on
        """
        self.model = model
        self.cluster_id = cluster_id
        self.device = device
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.board_history = []

        self.model.to(device)
        self.model.eval()

    def get_move(self, board: chess.Board, time_limit: Optional[float] = None) -> chess.Move:
        """
        Get best move for current position.

        Args:
            board: Current board state
            time_limit: Not used, for compatibility

        Returns:
            Best move according to model
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Get last 7 board positions for history
        history = self.board_history[-7:] if self.board_history else []

        # Encode board with history
        board_tensor = self.board_encoder.encode(board, history=history)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

        # Get model prediction
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)[0]

        # Find best legal move
        best_move = None
        best_prob = -1

        for move in legal_moves:
            try:
                move_index = self.move_encoder.encode(move, board)
                prob = policy_probs[move_index].item()

                if prob > best_prob:
                    best_prob = prob
                    best_move = move
            except:
                continue

        # Update history with current board state
        self.board_history.append(board.copy())

        return best_move if best_move else legal_moves[0]

    def reset(self):
        """Reset board history for a new game."""
        self.board_history = []


class StockfishPlayer:
    """Stockfish engine player."""

    def __init__(self, path: str, elo: int):
        """
        Initialize Stockfish player.

        Args:
            path: Path to Stockfish binary
            elo: Target ELO strength
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.elo = elo
        self.depth_limit = None

        # Configure Stockfish
        if elo < 1320:
            # Use skill level for weak play
            skill = max(0, min(20, int((elo - 800) / 100)))
            if elo < 900:
                depth = 1
            elif elo < 1100:
                depth = 2
            else:
                depth = 3

            self.engine.configure({"Skill Level": skill})
            self.depth_limit = depth
        else:
            # Use ELO limiting for stronger opponents
            try:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            except chess.engine.EngineError:
                # Clamp to valid range
                if "UCI_Elo" in self.engine.options:
                    min_elo = self.engine.options["UCI_Elo"].min
                    max_elo = self.engine.options["UCI_Elo"].max
                    clamped_elo = max(min_elo, min(max_elo, elo))
                    self.elo = clamped_elo
                    self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": clamped_elo})

    def get_move(self, board: chess.Board, time_limit: float = 0.1) -> chess.Move:
        """Get best move from Stockfish."""
        if self.depth_limit:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth_limit, time=time_limit))
        else:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def close(self):
        """Close the engine."""
        self.engine.quit()


class ModelEvaluator:
    """
    Orchestrate evaluation games against Stockfish.

    This class loads models, plays games against Stockfish at various ELO levels,
    analyzes the games for playstyle metrics, and estimates ELO ratings.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        device: str = 'cpu',
        skip_check_positions: bool = True,
        enable_delta_analysis: bool = True,
        delta_sampling_rate: int = 3,
        stockfish_depth: int = 12
    ):
        """
        Initialize evaluator.

        Args:
            stockfish_path: Path to Stockfish binary (auto-detected if None)
            device: Device to run models on
            skip_check_positions: Skip positions in check for analysis
            enable_delta_analysis: Enable delta (tipping point) metric analysis
            delta_sampling_rate: Analyze every Nth middlegame position for delta
            stockfish_depth: Search depth for Stockfish delta analysis
        """
        # Auto-detect Stockfish if not specified
        if stockfish_path is None:
            import shutil
            stockfish_path = shutil.which('stockfish')
            if stockfish_path is None:
                raise FileNotFoundError(
                    "Stockfish not found. Install with: sudo apt install stockfish"
                )

        self.stockfish_path = stockfish_path
        self.device = device
        self.game_analyzer = GameAnalyzer(
            skip_check_positions=skip_check_positions,
            enable_delta_analysis=enable_delta_analysis,
            delta_sampling_rate=delta_sampling_rate,
            stockfish_path=stockfish_path,
            stockfish_depth=stockfish_depth
        )
        self.logger = logger.bind(context="ModelEvaluator")

    def load_model_from_state(
        self,
        model_state: Dict[str, Any],
        cluster_id: str
    ) -> ChessAIPlayer:
        """
        Load a model from state dict.

        Args:
            model_state: PyTorch state dict
            cluster_id: ID of the cluster

        Returns:
            ChessAIPlayer instance
        """
        model = AlphaZeroNet()
        model.load_state_dict(model_state)
        player = ChessAIPlayer(model, cluster_id, device=self.device)

        self.logger.debug(f"Loaded model for {cluster_id}")
        return player

    def play_game(
        self,
        white_player,
        black_player,
        max_moves: int = 200,
        time_per_move: float = 0.1
    ) -> Tuple[str, str, str]:
        """
        Play a game between two players.

        Args:
            white_player: Player with white pieces (ChessAIPlayer or StockfishPlayer)
            black_player: Player with black pieces
            max_moves: Maximum moves before draw
            time_per_move: Time limit per move

        Returns:
            Tuple of (result, termination, pgn_string)
        """
        # Reset AI players for new game
        if isinstance(white_player, ChessAIPlayer):
            white_player.reset()
        if isinstance(black_player, ChessAIPlayer):
            black_player.reset()

        board = chess.Board()
        game = chess.pgn.Game()
        node = game

        # Set headers
        game.headers["Event"] = "Model Evaluation vs Stockfish"
        white_name = white_player.cluster_id if isinstance(white_player, ChessAIPlayer) else f"Stockfish-{white_player.elo}"
        black_name = black_player.cluster_id if isinstance(black_player, ChessAIPlayer) else f"Stockfish-{black_player.elo}"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            current_player = white_player if board.turn == chess.WHITE else black_player

            try:
                move = current_player.get_move(board, time_per_move)
                board.push(move)
                node = node.add_variation(move)
                move_count += 1

            except Exception as e:
                self.logger.error(f"Error getting move: {e}")
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                termination = f"error: {e}"
                game.headers["Result"] = result
                game.headers["Termination"] = termination
                return result, termination, str(game)

        # Determine result
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "checkmate"
        elif board.is_stalemate():
            result = "1/2-1/2"
            termination = "stalemate"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
            termination = "insufficient material"
        elif board.can_claim_fifty_moves():
            result = "1/2-1/2"
            termination = "fifty-move rule"
        elif board.can_claim_threefold_repetition():
            result = "1/2-1/2"
            termination = "threefold repetition"
        elif move_count >= max_moves:
            result = "1/2-1/2"
            termination = "max moves reached"
        else:
            result = "1/2-1/2"
            termination = "unknown"

        game.headers["Result"] = result
        game.headers["Termination"] = termination

        # Note: ECO code will be classified later using chess-openings library
        # or by analyzing the opening moves against a database

        return result, termination, str(game)

    async def evaluate_cluster(
        self,
        cluster_id: str,
        model_state: Dict[str, Any],
        num_games: int = 10,
        stockfish_elo_levels: Optional[List[int]] = None,
        time_per_move: float = 0.1
    ) -> ClusterEvaluationMetrics:
        """
        Evaluate a single cluster model against Stockfish.

        Args:
            cluster_id: Cluster ID
            model_state: Model state dict
            num_games: Number of games per Stockfish ELO level
            stockfish_elo_levels: List of Stockfish ELO levels to test against
            time_per_move: Time per move in seconds

        Returns:
            ClusterEvaluationMetrics with all results
        """
        if stockfish_elo_levels is None:
            # Default: test at 3 different levels around expected strength
            stockfish_elo_levels = [1000, 1200, 1400]

        self.logger.info(f"Evaluating {cluster_id}: {num_games} games x {len(stockfish_elo_levels)} ELO levels")

        # Load model
        ai_player = self.load_model_from_state(model_state, cluster_id)

        # Store results
        game_results = []
        computed_metrics_list = []
        match_results = []

        # Play against each Stockfish ELO level
        for stockfish_elo in stockfish_elo_levels:
            self.logger.info(f"{cluster_id} vs Stockfish-{stockfish_elo}: {num_games} games")

            stockfish = StockfishPlayer(self.stockfish_path, stockfish_elo)
            match = MatchResult(
                opponent=f"Stockfish-{stockfish_elo}",
                opponent_elo=stockfish_elo,
                games_played=num_games
            )

            for game_num in range(num_games):
                # Alternate colors
                ai_is_white = (game_num % 2 == 0)
                white = ai_player if ai_is_white else stockfish
                black = stockfish if ai_is_white else ai_player

                # Play game
                result, termination, pgn = self.play_game(white, black, time_per_move=time_per_move)

                # Update match result
                if ai_is_white:
                    if result == "1-0":
                        match.wins += 1
                    elif result == "1/2-1/2":
                        match.draws += 1
                    else:
                        match.losses += 1
                else:
                    if result == "0-1":
                        match.wins += 1
                    elif result == "1/2-1/2":
                        match.draws += 1
                    else:
                        match.losses += 1

                # Analyze game for playstyle metrics
                white_id = cluster_id if ai_is_white else f"Stockfish-{stockfish_elo}"
                black_id = f"Stockfish-{stockfish_elo}" if ai_is_white else cluster_id

                game_metrics = self.game_analyzer.analyze_game(pgn, white_id, black_id)
                game_results.append(game_metrics)

                # Compute playstyle metrics
                computed = PlaystyleMetricsCalculator.compute_game_metrics(game_metrics)
                computed_metrics_list.append((computed, ai_is_white))

                self.logger.debug(f"Game {game_num + 1}/{num_games}: {result} - "
                                f"AI tactical score: {computed.white_metrics.tactical_score if ai_is_white else computed.black_metrics.tactical_score:.3f}")

            stockfish.close()
            match_results.append(match)
            self.logger.info(f"{cluster_id} vs Stockfish-{stockfish_elo}: "
                           f"{match.wins}W-{match.draws}D-{match.losses}L ({match.win_rate:.1f}%)")

        # Estimate ELO
        estimated_elo, elo_confidence = ELOEstimator.estimate_elo(match_results)

        # Aggregate metrics
        cluster_metrics = self._aggregate_metrics(
            cluster_id, game_results, computed_metrics_list, match_results,
            estimated_elo, elo_confidence
        )

        self.logger.info(f"{cluster_id} evaluation complete: "
                        f"ELO={estimated_elo}±{elo_confidence}, "
                        f"Tactical Score={cluster_metrics.tactical_score:.3f} ({cluster_metrics.classification})")

        return cluster_metrics

    async def evaluate_models(
        self,
        cluster_models: Dict[str, Dict[str, Any]],
        num_games: int = 10,
        stockfish_elo_levels: Optional[List[int]] = None,
        time_per_move: float = 0.1
    ) -> Dict[str, Any]:
        """
        Evaluate multiple cluster models in parallel.

        Args:
            cluster_models: Dict of cluster_id -> model_state_dict
            num_games: Number of games per Stockfish ELO level
            stockfish_elo_levels: List of Stockfish ELO levels
            time_per_move: Time per move in seconds

        Returns:
            Dict with evaluation results for all clusters
        """
        self.logger.info(f"Starting parallel evaluation of {len(cluster_models)} clusters")

        # Run evaluations in parallel
        tasks = []
        for cluster_id, model_state in cluster_models.items():
            task = self.evaluate_cluster(
                cluster_id, model_state, num_games,
                stockfish_elo_levels, time_per_move
            )
            tasks.append(task)

        # Wait for all evaluations
        cluster_metrics_list = await asyncio.gather(*tasks)

        # Create results dict
        cluster_metrics = {
            cm.cluster_id: cm
            for cm in cluster_metrics_list
        }

        # Create summary
        summary = self._create_summary(cluster_metrics)

        return {
            "cluster_metrics": {cid: cm.to_dict() for cid, cm in cluster_metrics.items()},
            "summary": summary
        }

    def _aggregate_metrics(
        self,
        cluster_id: str,
        game_results: List[GameMetrics],
        computed_metrics_list: List[Tuple[ComputedGameMetrics, bool]],
        match_results: List[MatchResult],
        estimated_elo: int,
        elo_confidence: int
    ) -> ClusterEvaluationMetrics:
        """Aggregate metrics for a cluster."""
        cm = ClusterEvaluationMetrics(cluster_id=cluster_id)
        cm.estimated_elo = estimated_elo
        cm.elo_confidence = elo_confidence
        cm.match_results = match_results

        tactical_scores = []
        opening_counts = {}  # ECO code -> (opening_name, count)

        for game_result, (computed, ai_is_white) in zip(game_results, computed_metrics_list):
            # Get AI player metrics
            player_metrics = computed.white_metrics if ai_is_white else computed.black_metrics

            # Track opening statistics
            if game_result.eco_code:
                eco = game_result.eco_code
                name = game_result.opening_name or "Unknown"
                if eco not in opening_counts:
                    opening_counts[eco] = [name, 0]
                opening_counts[eco][1] += 1

            cm.games_analyzed += 1
            if ai_is_white:
                cm.games_as_white += 1
            else:
                cm.games_as_black += 1

            # Aggregate raw metrics
            cm.avg_attacked_material += player_metrics.total_attacked_material
            cm.avg_legal_moves += player_metrics.total_legal_moves
            cm.avg_captures += player_metrics.total_captures
            cm.avg_center_control += player_metrics.avg_center_control

            # Aggregate normalized metrics
            cm.avg_attacks_metric += player_metrics.attacks_metric
            cm.avg_moves_metric += player_metrics.moves_metric
            cm.avg_material_metric += player_metrics.material_metric

            # Tactical score
            cm.tactical_score += player_metrics.tactical_score
            tactical_scores.append(player_metrics.tactical_score)

            if player_metrics.tactical_score < cm.tactical_score_min:
                cm.tactical_score_min = player_metrics.tactical_score
            if player_metrics.tactical_score > cm.tactical_score_max:
                cm.tactical_score_max = player_metrics.tactical_score

            # Classification distribution
            cm.classification_distribution[player_metrics.classification] += 1

            # Game outcomes
            if ai_is_white:
                if game_result.result == "1-0":
                    cm.wins += 1
                elif game_result.result == "1/2-1/2":
                    cm.draws += 1
                else:
                    cm.losses += 1
            else:
                if game_result.result == "0-1":
                    cm.wins += 1
                elif game_result.result == "1/2-1/2":
                    cm.draws += 1
                else:
                    cm.losses += 1

        # Calculate averages
        if cm.games_analyzed > 0:
            cm.avg_attacked_material /= cm.games_analyzed
            cm.avg_legal_moves /= cm.games_analyzed
            cm.avg_captures /= cm.games_analyzed
            cm.avg_center_control /= cm.games_analyzed
            cm.avg_attacks_metric /= cm.games_analyzed
            cm.avg_moves_metric /= cm.games_analyzed
            cm.avg_material_metric /= cm.games_analyzed
            cm.tactical_score /= cm.games_analyzed

            # Calculate std
            cm.tactical_score_std = float(np.std(tactical_scores))

            # Classification
            cm.classification = PlaystyleMetricsCalculator.classify_tactical_score(cm.tactical_score)

            # Rates
            total_games = cm.wins + cm.draws + cm.losses
            cm.win_rate = cm.wins / total_games
            cm.draw_rate = cm.draws / total_games
            cm.loss_rate = cm.losses / total_games

            # Aggregate opening statistics
            cm.opening_frequency = {eco: count for eco, (name, count) in opening_counts.items()}

            # Top 10 openings
            sorted_openings = sorted(opening_counts.items(), key=lambda x: x[1][1], reverse=True)
            cm.top_openings = [(eco, name, count) for eco, (name, count) in sorted_openings[:10]]

            # Compute move type metrics from PGN strings
            move_type_analyzer = MoveTypeAnalyzer()
            pgn_strings = [gr.pgn for gr in game_results]
            move_type_cluster_metrics = move_type_analyzer.analyze_games(pgn_strings, cluster_id)
            cm.move_type_metrics = move_type_cluster_metrics.to_dict()

        return cm

    def _create_summary(self, cluster_metrics: Dict[str, ClusterEvaluationMetrics]) -> Dict[str, Any]:
        """Create evaluation summary."""
        # ELO rankings
        elo_rankings = []
        for cluster_id, cm in cluster_metrics.items():
            elo_rankings.append({
                "cluster_id": cluster_id,
                "estimated_elo": cm.estimated_elo,
                "elo_confidence": cm.elo_confidence,
                "tactical_score": round(cm.tactical_score, 3),
                "classification": cm.classification,
                "overall_winrate": round(cm.win_rate, 3)
            })

        elo_rankings.sort(key=lambda x: x["estimated_elo"], reverse=True)

        # Tactical rankings
        tactical_rankings = sorted(elo_rankings, key=lambda x: x["tactical_score"], reverse=True)

        # Playstyle divergence
        if len(cluster_metrics) >= 2:
            scores = [cm.tactical_score for cm in cluster_metrics.values()]
            divergence = float(np.std(scores))
        else:
            divergence = 0.0

        # ELO spread
        elos = [cm.estimated_elo for cm in cluster_metrics.values()]
        elo_spread = max(elos) - min(elos) if len(elos) > 1 else 0

        # Move type comparison
        move_type_comparison = None
        if len(cluster_metrics) >= 2:
            from server.evaluation.move_type_analyzer import compute_move_type_comparison, ClusterMoveTypeMetrics
            # Reconstruct ClusterMoveTypeMetrics from dicts
            cluster_move_metrics = {}
            for cid, cm in cluster_metrics.items():
                if cm.move_type_metrics:
                    mtm = ClusterMoveTypeMetrics(cluster_id=cid)
                    mtm.games_analyzed = cm.move_type_metrics.get("games_analyzed", 0)
                    totals = cm.move_type_metrics.get("totals", {})
                    mtm.total_moves = totals.get("total_moves", 0)
                    mtm.total_captures = totals.get("captures", 0)
                    mtm.total_checks = totals.get("checks", 0)
                    mtm.total_pawn_advances = totals.get("pawn_advances", 0)
                    mtm.total_quiet_moves = totals.get("quiet_moves", 0)
                    mtm.total_aggressive = totals.get("aggressive_moves", 0)
                    cluster_move_metrics[cid] = mtm

            if len(cluster_move_metrics) >= 2:
                move_type_comparison = compute_move_type_comparison(cluster_move_metrics)

        return {
            "total_clusters_evaluated": len(cluster_metrics),
            "elo_rankings": elo_rankings,
            "tactical_rankings": tactical_rankings,
            "playstyle_divergence": round(divergence, 3),
            "elo_spread": elo_spread,
            "move_type_comparison": move_type_comparison
        }
