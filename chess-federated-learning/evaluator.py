#!/usr/bin/env python3
"""
Chess Model Evaluator - Test models against Stockfish to estimate ELO.

This script plays your trained AlphaZero models against Stockfish at various
strength levels to estimate the model's ELO rating.

Usage:
    python evaluator.py --model path/to/model.pt --games 20 --stockfish-path /usr/bin/stockfish

Features:
    - Play against multiple Stockfish ELO levels
    - Automatic ELO estimation using Bayesian updating
    - Detailed game statistics and analysis
    - PGN export of games
    - Progress tracking
"""

import argparse
import sys
import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import loguru

log = loguru.logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from client.trainer.models.alphazero_net import AlphaZeroNet
from data.board_encoder import BoardEncoder
from data.move_encoder import MoveEncoder
from common.model_serialization import PyTorchSerializer


@dataclass
class GameResult:
    """Result of a single game."""
    white_player: str
    black_player: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: List[str]
    termination: str
    duration: float

    def get_score_for(self, player: str) -> float:
        """Get score (0, 0.5, or 1) for a player."""
        if self.result == "1/2-1/2":
            return 0.5
        if player == self.white_player:
            return 1.0 if self.result == "1-0" else 0.0
        else:
            return 1.0 if self.result == "0-1" else 0.0


@dataclass
class MatchResult:
    """Results of a match against a specific opponent."""
    opponent: str
    opponent_elo: int
    games_played: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: List[GameResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Total score (wins + 0.5*draws)."""
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        return (self.score / self.games_played * 100) if self.games_played > 0 else 0


class ChessAI:
    """AI player using trained AlphaZero model."""

    def __init__(self, model: AlphaZeroNet, name: str = "AlphaZero", device: str = 'cpu'):
        self.model = model
        self.name = name
        self.device = device
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.board_history = []  # Track board history for encoding

        self.model.to(device)
        self.model.eval()

    def get_move(self, board: chess.Board, time_limit: Optional[float] = None) -> chess.Move:
        """
        Get best move for current position.

        Args:
            board: Current board state
            time_limit: Time limit in seconds (not used, for compatibility)

        Returns:
            Best move
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Get last 7 board positions for history (AlphaZero uses 8 time steps total)
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
        
        # Debug: Check if model is producing sensible outputs
        top_5_indices = torch.topk(policy_probs, k=min(5, len(policy_probs))).indices
        
        for move in legal_moves:
            try:
                move_index = self.move_encoder.encode(move, board)
                prob = policy_probs[move_index].item()

                if prob > best_prob:
                    best_prob = prob
                    best_move = move
            except:
                # If encoding fails, skip this move
                continue

        # After selecting move, update history with current board state
        # (This will be the previous position for the next move)
        self.board_history.append(board.copy())

        return best_move if best_move else legal_moves[0]


class StockfishPlayer:
    """Stockfish engine player."""

    def __init__(self, path: str, elo: int, name: Optional[str] = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.elo = elo
        self.name = name or f"Stockfish-{elo}"
        self.depth_limit = None

        # For very weak play (below Stockfish's minimum ELO), use skill level
        if elo < 1320:
            # Map ELO to skill level (0-20) and depth
            # ELO 800 = skill 0, depth 1 (very weak)
            # ELO 1200 = skill 4, depth 3 (weak)
            skill = max(0, min(20, int((elo - 800) / 100)))

            # Limit search depth for weaker play
            if elo < 900:
                depth = 1  # Very weak (beginner)
            elif elo < 1100:
                depth = 2  # Weak (novice)
            else:
                depth = 3  # Moderate (intermediate)

            self.engine.configure({"Skill Level": skill})
            self.depth_limit = depth
            self.name = f"Stockfish-ELO{elo}~"
            print(f"  Using skill level {skill}, depth {depth} (approx. ELO {elo})")
        else:
            # Use ELO limiting for stronger opponents
            try:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            except chess.engine.EngineError:
                # Extract minimum ELO from error
                if "UCI_Elo" in self.engine.options:
                    min_elo = self.engine.options["UCI_Elo"].min
                    max_elo = self.engine.options["UCI_Elo"].max
                    print(f"  Warning: Stockfish ELO range is {min_elo}-{max_elo}")

                    # Clamp to valid range
                    clamped_elo = max(min_elo, min(max_elo, elo))
                    print(f"  Adjusting requested ELO {elo} to {clamped_elo}")
                    self.elo = clamped_elo
                    self.name = f"Stockfish-{clamped_elo}"
                    self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": clamped_elo})
                else:
                    raise

    def get_move(self, board: chess.Board, time_limit: float = 0.1) -> chess.Move:
        """
        Get best move from Stockfish.

        Args:
            board: Current board state
            time_limit: Time limit in seconds

        Returns:
            Best move
        """
        # Use depth limit for weak play, time limit for stronger play
        if hasattr(self, 'depth_limit') and self.depth_limit:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth_limit, time=time_limit))
        else:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def close(self):
        """Close the engine."""
        self.engine.quit()

    def __del__(self):
        """Cleanup."""
        try:
            self.close()
        except:
            pass


class GamePlayer:
    """Plays a game between two players."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def play_game(
        self,
        white_player,
        black_player,
        max_moves: int = 200,
        time_per_move: float = 0.1
    ) -> GameResult:
        """
        Play a complete game.

        Args:
            white_player: Player with white pieces
            black_player: Player with black pieces
            max_moves: Maximum moves before draw
            time_per_move: Time limit per move

        Returns:
            GameResult
        """
        board = chess.Board()
        moves = []
        start_time = time.time()

        # Reset board history for AI players at the start of each game
        if hasattr(white_player, 'board_history'):
            white_player.board_history = []
        if hasattr(black_player, 'board_history'):
            black_player.board_history = []

        if self.verbose:
            log.info(f"\n{white_player.name} (White) vs {black_player.name} (Black)")

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            # Get current player
            current_player = white_player if board.turn == chess.WHITE else black_player

            try:
                # Get move
                move = current_player.get_move(board, time_per_move)

                # Play move
                san_move = board.san(move)
                board.push(move)
                moves.append(san_move)
                move_count += 1

                if self.verbose and move_count % 10 == 0:
                    log.info(f"  Move {move_count}: {san_move}")

            except Exception as e:
                log.info(f"Error getting move: {e}")
                # Game ends in error
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                termination = f"error: {e}"
                break
        else:
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

        duration = time.time() - start_time

        if self.verbose:
            log.info(f"  Result: {result} ({termination}) in {move_count} moves ({duration:.1f}s)")

        return GameResult(
            white_player=white_player.name,
            black_player=black_player.name,
            result=result,
            moves=moves,
            termination=termination,
            duration=duration
        )


class ELOEstimator:
    """Estimate ELO rating based on match results."""

    @staticmethod
    def expected_score(elo_a: int, elo_b: int) -> float:
        """Calculate expected score for player A vs player B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    @staticmethod
    def estimate_elo(results: List[MatchResult], initial_elo: int = 1500) -> Tuple[int, int]:
        """
        Estimate ELO using Bayesian updating.

        Args:
            results: List of match results
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


class ModelEvaluator:
    """Main evaluator class."""

    def __init__(
        self,
        model_path: str,
        stockfish_path: str = None,
        device: str = None,
        verbose: bool = True
    ):
        # Auto-detect Stockfish if not specified
        if stockfish_path is None:
            import shutil
            stockfish_path = shutil.which('stockfish')
            if stockfish_path is None:
                raise FileNotFoundError(
                    "Stockfish not found. Install with: sudo apt install stockfish\n"
                    "Or specify path with --stockfish /path/to/stockfish"
                )

        self.stockfish_path = stockfish_path
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Load model
        log.info(f"Loading model from {self.model_path}...")
        self.model = self._load_model()
        self.ai = ChessAI(self.model, name="AlphaZero-FL", device=self.device)

        log.info(f"Model loaded successfully on {self.device}")

        # Game player
        self.game_player = GamePlayer(verbose=verbose)

        # Results storage
        self.results: List[MatchResult] = []

    def _load_model(self) -> AlphaZeroNet:
        """Load model from checkpoint."""
        serializer = PyTorchSerializer(compression=True, encoding='base64')

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Extract model state
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint

        # Check if serialized
        if isinstance(model_state, dict) and 'serialized_data' in model_state:
            model_state = serializer.deserialize(model_state['serialized_data'])

        # Create and load model
        model = AlphaZeroNet()
        model.load_state_dict(model_state)

        return model

    def play_match(
        self,
        opponent_elo: int,
        num_games: int,
        alternate_colors: bool = True,
        time_per_move: float = 0.1
    ) -> MatchResult:
        """
        Play a match against Stockfish.

        Args:
            opponent_elo: Stockfish ELO level
            num_games: Number of games to play
            alternate_colors: Alternate colors each game
            time_per_move: Time per move in seconds

        Returns:
            MatchResult
        """
        log.info(f"\n{'='*60}")
        log.info(f"Match: AlphaZero-FL vs Stockfish (ELO {opponent_elo})")
        log.info(f"Games: {num_games}, Time per move: {time_per_move}s")
        log.info(f"{'='*60}")

        # Create Stockfish player
        stockfish = StockfishPlayer(self.stockfish_path, opponent_elo)

        # Create match result
        match = MatchResult(
            opponent=stockfish.name,
            opponent_elo=opponent_elo,
            games_played=num_games
        )

        # Play games
        for i in range(num_games):
            # Alternate colors or play as white
            ai_is_white = (i % 2 == 0) if alternate_colors else True

            white = self.ai if ai_is_white else stockfish
            black = stockfish if ai_is_white else self.ai

            log.info(f"\nGame {i+1}/{num_games}:")
            game = self.game_player.play_game(white, black, time_per_move=time_per_move)

            # Record result
            match.games.append(game)

            # Update stats
            score = game.get_score_for(self.ai.name)
            if score == 1.0:
                match.wins += 1
            elif score == 0.5:
                match.draws += 1
            else:
                match.losses += 1

        # Close Stockfish
        stockfish.close()

        # log.info summary
        log.info(f"\n{'-'*60}")
        log.info(f"Match Result: {match.wins}W-{match.draws}D-{match.losses}L")
        log.info(f"Score: {match.score}/{num_games} ({match.win_rate:.1f}%)")
        log.info(f"{'-'*60}")

        self.results.append(match)
        return match

    def auto_evaluate(
        self,
        num_games_per_level: int = 10,
        start_elo: int = 1000,
        max_iterations: int = 5
    ) -> Dict:
        """
        Automatically evaluate model by playing at different ELO levels.

        Args:
            num_games_per_level: Games per ELO level
            start_elo: Starting ELO guess (default 1000 for weak models)
            max_iterations: Maximum refinement iterations

        Returns:
            Evaluation summary dict
        """
        print(f"\n{'#'*60}")
        print(f"# Automatic ELO Evaluation")
        print(f"# Starting ELO: {start_elo}")
        print(f"# Games per level: {num_games_per_level}")
        print(f"{'#'*60}\n")

        # Test at different ELO levels (supports weak play with skill level)
        test_elos = [
            max(800, start_elo - 200),  # Don't go below 800
            start_elo,
            start_elo + 200,
        ]

        for elo in test_elos:
            self.play_match(elo, num_games_per_level, time_per_move=0.1)

        # Estimate ELO
        estimated_elo, confidence = ELOEstimator.estimate_elo(self.results, start_elo)

        log.info(f"\n{'='*60}")
        log.info(f"ESTIMATED ELO: {estimated_elo} ± {confidence}")
        log.info(f"{'='*60}\n")

        return {
            'estimated_elo': estimated_elo,
            'confidence_range': confidence,
            'matches': self.results,
            'timestamp': datetime.now().isoformat()
        }

    def save_results(self, output_path: str):
        """Save evaluation results to JSON."""
        output = {
            'model_path': str(self.model_path),
            'device': self.device,
            'timestamp': datetime.now().isoformat(),
            'matches': []
        }

        for match in self.results:
            output['matches'].append({
                'opponent': match.opponent,
                'opponent_elo': match.opponent_elo,
                'games_played': match.games_played,
                'wins': match.wins,
                'draws': match.draws,
                'losses': match.losses,
                'score': match.score,
                'win_rate': match.win_rate
            })

        # Estimate ELO
        estimated_elo, confidence = ELOEstimator.estimate_elo(self.results)
        output['estimated_elo'] = estimated_elo
        output['confidence_range'] = confidence

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        log.info(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess model against Stockfish")
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--stockfish', help='Path to Stockfish binary (auto-detected if not specified)')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--elo', type=int, help='Specific Stockfish ELO to test against')
    parser.add_argument('--start-elo', type=int, default=1000, help='Starting ELO for auto evaluation (use 800-1200 for weak models)')
    parser.add_argument('--auto', action='store_true', help='Auto-evaluate across multiple ELO levels')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to run model on')
    parser.add_argument('--time', type=float, default=0.1, help='Time per move in seconds')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')

    args = parser.parse_args()

    # Disable debug logging if quiet mode
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.INFO)
        # Also disable loguru if it's being used
        try:
            from loguru import logger
            logger.remove()  # Remove default handler
            logger.add(sys.stderr, level="INFO")  # Set to INFO or higher
        except ImportError:
            pass

    # Create evaluator (will auto-detect Stockfish if not specified)
    try:
        evaluator = ModelEvaluator(
            args.model,
            args.stockfish,
            device=args.device,
            verbose=True
        )
        if not args.quiet:
            log.info(f"Using Stockfish at: {evaluator.stockfish_path}")
    except FileNotFoundError as e:
        log.info(f"Error: {e}")
        sys.exit(1)

    # Run evaluation
    if args.auto:
        results = evaluator.auto_evaluate(
            num_games_per_level=args.games,
            start_elo=args.start_elo
        )
    elif args.elo:
        evaluator.play_match(args.elo, args.games, time_per_move=args.time)
    else:
        log.info("Error: Specify --elo <rating> or --auto")
        sys.exit(1)

    # Save results
    if args.output:
        evaluator.save_results(args.output)

    log.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
