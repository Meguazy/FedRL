"""
Sample extractor for converting chess games into training samples.

This module extracts training samples from chess games by:
1. Loading games from database (with diversity)
2. Extracting each position as a separate sample
3. Preparing data for supervised learning

Each game position becomes one training sample with:
- Input: Board state (to be encoded by board_encoder)
- Target policy: Move that was played (to be encoded by move_encoder)
- Target value: Game outcome from player's perspective
"""
import chess
import chess.pgn
import random

from typing import List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass
from loguru import logger

from .game_loader import GameLoader, GameFilter
from .eco_classifier import PlaystyleType


@dataclass
class TrainingSample:
    """
    A single training sample extracted from a game position.

    Attributes:
        board: Chess board at this position
        move_played: The move that was played in this position
        game_outcome: Game result from current player's perspective (+1, 0, -1)
        move_number: Which move in the game this is
        eco_code: ECO opening code of the game
        playstyle: Tactical or positional classification
    """
    board: chess.Board
    move_played: chess.Move
    game_outcome: float
    move_number: int
    eco_code: str
    playstyle: PlaystyleType
    
    
@dataclass
class ExtractionConfig:
    """
    Configuration for sample extraction.

    Attributes:
        skip_opening_moves: Skip first N moves (too formulaic)
        skip_endgame_moves: Skip positions with < N pieces (simplified endgames)
        sample_rate: Extract every Nth position (1.0 = all positions)
        max_positions_per_game: Maximum positions to extract per game
        shuffle_games: Randomize game order for diversity
    """
    skip_opening_moves: int = 10
    skip_endgame_moves: int = 5
    sample_rate: float = 1.0
    max_positions_per_game: Optional[int] = None
    shuffle_games: bool = True
    
class SampleExtractor:
    """
    Extracts training samples from chess games.

    Example:
        >>> extractor = SampleExtractor("lichess_db_2024-01.pgn.zst")
        >>>
        >>> # Extract 100 tactical games for training
        >>> samples = extractor.extract_samples(
        ...     num_games=100,
        ...     playstyle="tactical",
        ...     min_rating=2000
        ... )
    """

    def __init__(self, pgn_path: str, extraction_config: Optional[ExtractionConfig] = None):
        """
        Initialize sample extractor.

        Args:
            pgn_path: Path to PGN database file
            extraction_config: Configuration for sample extraction
        """
        log = logger.bind(module="SampleExtractor.__init__")
        log.info(f"Loading games from {pgn_path}")
        
        self.pgn_path = pgn_path
        self.game_loader = GameLoader(pgn_path)
        self.config = extraction_config or ExtractionConfig()
        
        log.info(f"Sample extractor initialized with path: {pgn_path}")
        
    def extract_samples(
        self,
        num_games: int,
        playstyle: Optional[PlaystyleType] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        offset: int = 0
    ) -> List[TrainingSample]:
        """
        Extract training samples from games.

        Args:
            num_games: Number of games to extract samples from
            playstyle: Filter by playstyle ("tactical", "positional", or None)
            min_rating: Minimum player rating to filter games
            max_rating: Maximum player rating to filter games
            offset: Skip first N games (for pagination)
        
        Returns:
            List of extracted training samples
        """
        log = logger.bind(module="SampleExtractor.extract_samples")
        log.info(f"Extracting samples from {num_games} games with playstyle={playstyle}, "
                 f"rating_range=({min_rating}, {max_rating}), offset={offset}")
        
        game_filter = GameFilter(
            min_rating=min_rating,
            max_rating=max_rating,
            playstyle=playstyle,
            max_games=num_games + offset
        )
        
        # Load games
        games = list(self.game_loader.load_games(game_filter))
        
        # Apply offset to get different games each round
        if offset > 0:
            games = games[offset:offset + num_games]
        else:
            games = games[:num_games]
            
        # Shuffle games for diversity
        if self.config.shuffle_games:
            random.shuffle(games)
            
        # Extract samples from each game
        all_samples = []
        for i, game in enumerate(games):
            log.info(f"Extracting samples from game {i+1}/{len(games)}")
            try:
                samples = self._extract_samples_from_game(game)
                all_samples.extend(samples)
                
                if i % 10 == 0:
                    log.info(f"Extracted {len(all_samples)} samples so far")
                    
            except Exception as e:
                log.warning(f"Error extracting samples from game {i+1}: {e}")
                continue
            
        log.success(f"Extraction complete. Total samples extracted: {len(all_samples)}")
        return all_samples
    
    def _extract_samples_from_game(self, game: chess.pgn.Game) -> List[TrainingSample]:
        """
        Extract samples from a single game.

        Args:
            game: A chess.pgn.Game object
        
        Returns:
            List of TrainingSample objects extracted from the game
        """
        samples = []
        
        # Get game info
        headers = game.headers
        eco_code = headers.get("ECO", "")
        playstyle = self.game_loader.eco_classifier.classify(eco_code)
        result = headers.get("Result", "*")
        
        # Parse game outcome
        outcome_white, outcome_black = self._parse_game_result(result)
        
        # Traverse game moves
        board = game.board()
        move_number = 0
        
        for node in game.mainline():
            move_number += 1
            move = node.move
            
            # Skip opening moves
            if move_number <= self.config.skip_opening_moves:
                board.push(move)
                continue
            
            # Skip endgame positions
            if self._count_pieces(board) <= self.config.skip_endgame_moves:
                board.push(move)
                continue
            
            # Apply sampling rate
            if self.config.sample_rate < 1.0:
               if random.random() > self.config.sample_rate:
                   board.push(move)
                   continue
               
            # Determine game outcome from current player's perspective
            current_player = board.turn
            game_outcome = outcome_white if current_player == chess.WHITE else outcome_black

            # Create sample with board state BEFORE move is played
            # (this is the position where the player needs to choose the move)
            sample = TrainingSample(
                board=board.copy(),
                move_played=move,
                game_outcome=game_outcome,
                move_number=move_number,
                eco_code=eco_code,
                playstyle=playstyle
            )
            samples.append(sample)

            # Now push the move to advance the board
            board.push(move)

            # Limit positions per game
            if self.config.max_positions_per_game and len(samples) >= self.config.max_positions_per_game:
                break

        return samples
    
    def _parse_game_result(self, result: str) -> Tuple[float, float]:
        """Parse game result to outcomes for white and black."""
        if result == "1-0":
            return (1.0, -1.0)
        elif result == "0-1":
            return (-1.0, 1.0)
        elif result == "1/2-1/2":
            return (0.0, 0.0)
        else:
            return (0.0, 0.0)

    def _count_pieces(self, board: chess.Board) -> int:
        """Count total pieces on the board."""
        return len(board.piece_map())

    def get_statistics(self, samples: List[TrainingSample]) -> dict:
        """Get statistics about extracted samples."""
        if not samples:
            return {"total_samples": 0}

        # Count outcomes
        wins = sum(1 for s in samples if s.game_outcome > 0)
        draws = sum(1 for s in samples if s.game_outcome == 0)
        losses = sum(1 for s in samples if s.game_outcome < 0)

        # Count playstyles
        tactical = sum(1 for s in samples if s.playstyle == "tactical")
        positional = sum(1 for s in samples if s.playstyle == "positional")

        # Move statistics
        move_numbers = [s.move_number for s in samples]

        total = len(samples)
        return {
            "total_samples": total,
            "outcomes": {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_ratio": wins / total if total > 0 else 0.0,
                "draw_ratio": draws / total if total > 0 else 0.0,
                "loss_ratio": losses / total if total > 0 else 0.0,
            },
            "playstyles": {
                "tactical": tactical,
                "positional": positional,
            },
            "move_stats": {
                "avg_move": sum(move_numbers) / len(move_numbers),
                "min_move": min(move_numbers),
                "max_move": max(move_numbers),
            }
        }


# if __name__ == "__main__":
#     import sys

#     log = logger.bind(module="SampleExtractor.__main__")

#     if len(sys.argv) < 2:
#         log.error("Missing PGN file path argument")
#         print("Usage: python sample_extractor.py <path_to_pgn_file>")
#         print("\nExample:")
#         print("  python sample_extractor.py databases/lichess_db_standard_rated_2024-01.pgn.zst")
#         sys.exit(1)

#     pgn_path = sys.argv[1]

#     log.info("="*70)
#     log.info("SAMPLE EXTRACTOR TEST")
#     log.info("="*70)

#     # Create extractor with config
#     config = ExtractionConfig(
#         skip_opening_moves=10,
#         skip_endgame_moves=6,
#         sample_rate=1.0,
#         shuffle_games=True
#     )

#     extractor = SampleExtractor(pgn_path, config)

#     # Test 1: Extract tactical samples
#     log.info("")
#     log.info("[TEST 1] Extracting TACTICAL samples")
#     log.info("-" * 70)

#     tactical_samples = extractor.extract_samples(
#         num_games=100,
#         playstyle="tactical",
#         min_rating=2000,
#         offset=0
#     )

#     log.success(f"Extracted {len(tactical_samples)} tactical samples from 5 games")

#     # Show first 3 samples
#     log.info("First 3 samples:")
#     for i, sample in enumerate(tactical_samples[:3], 1):
#         log.info(f"  {i}. Move {sample.move_number}: {sample.move_played.uci()}")
#         log.info(f"     Outcome: {sample.game_outcome:+.1f} | Playstyle: {sample.playstyle}")
#         log.info(f"     ECO: {sample.eco_code}")
#         log.info(f"     FEN: {sample.board.fen()[:60]}...")

#     # Show statistics
#     log.info("")
#     log.info("-" * 70)
#     log.info("TACTICAL STATISTICS")
#     log.info("-" * 70)
#     stats = extractor.get_statistics(tactical_samples)
#     log.info(f"Total samples: {stats['total_samples']}")
#     log.info(f"Outcomes: {stats['outcomes']['wins']} wins, "
#           f"{stats['outcomes']['draws']} draws, "
#           f"{stats['outcomes']['losses']} losses")
#     log.info(f"Playstyles: {stats['playstyles']['tactical']} tactical, "
#           f"{stats['playstyles']['positional']} positional")
#     log.info(f"Average move: {stats['move_stats']['avg_move']:.1f}")
#     log.info(f"Move range: {stats['move_stats']['min_move']}-{stats['move_stats']['max_move']}")

#     # Test 2: Extract positional samples
#     log.info("")
#     log.info("="*70)
#     log.info("[TEST 2] Extracting POSITIONAL samples")
#     log.info("-" * 70)

#     positional_samples = extractor.extract_samples(
#         num_games=100,
#         playstyle="positional",
#         min_rating=2000,
#         offset=0
#     )

#     log.success(f"Extracted {len(positional_samples)} positional samples from 5 games")

#     # Show statistics
#     log.info("")
#     log.info("-" * 70)
#     log.info("POSITIONAL STATISTICS")
#     log.info("-" * 70)
#     stats = extractor.get_statistics(positional_samples)
#     log.info(f"Total samples: {stats['total_samples']}")
#     log.info(f"Outcomes: {stats['outcomes']['wins']} wins, "
#           f"{stats['outcomes']['draws']} draws, "
#           f"{stats['outcomes']['losses']} losses")
#     log.info(f"Playstyles: {stats['playstyles']['tactical']} tactical, "
#           f"{stats['playstyles']['positional']} positional")
#     log.info(f"Average move: {stats['move_stats']['avg_move']:.1f}")
#     log.info(f"Move range: {stats['move_stats']['min_move']}-{stats['move_stats']['max_move']}")

#     # Test 3: Test offset (different games)
#     log.info("")
#     log.info("="*70)
#     log.info("[TEST 3] Testing offset for different games")
#     log.info("-" * 70)

#     samples_round1 = extractor.extract_samples(
#         num_games=100,
#         playstyle="tactical",
#         min_rating=2000,
#         offset=0
#     )

#     samples_round2 = extractor.extract_samples(
#         num_games=100,
#         playstyle="tactical",
#         min_rating=2000,
#         offset=100  # Skip first 100 games
#     )

#     log.info(f"Round 1 (offset=0): {len(samples_round1)} samples")
#     log.info(f"Round 2 (offset=100): {len(samples_round2)} samples")
#     log.info("Offset ensures different games are used each round!")

#     log.info("")
#     log.info("="*70)
#     log.success("TEST COMPLETE")
#     log.info("="*70)
