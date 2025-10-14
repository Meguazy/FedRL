"""
Game loader module for extracting chess games from PGN databases.

This module handles loading games from Lichess database files (or other PGN sources),
filtering by rating, ECO code, and playstyle, and preparing them for training.
"""

import chess.pgn
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import io

from .eco_classifier import ECOClassifier, PlaystyleType


@dataclass
class GameFilter:
    """
    Filter criteria for loading games.

    Attributes:
        min_rating: Minimum average player rating
        max_rating: Maximum average player rating (None = no limit)
        playstyle: Filter by playstyle ('tactical' or 'positional')
        time_control: Filter by time control (e.g., 'blitz', 'rapid', 'classical')
        result: Filter by result ('1-0', '0-1', '1/2-1/2', or None for all)
        max_games: Maximum number of games to load (None = all)
    """
    min_rating: int = 1800
    max_rating: Optional[int] = None
    playstyle: Optional[PlaystyleType] = None
    time_control: Optional[str] = None
    result: Optional[str] = None
    max_games: Optional[int] = None


class GameLoader:
    """
    Loader for chess games from PGN databases.

    This class handles reading PGN files, filtering games by various criteria,
    and providing an iterator over matching games.

    Example:
        >>> loader = GameLoader("lichess_db_standard_rated_2024-01.pgn")
        >>> filter = GameFilter(min_rating=2000, playstyle="tactical", max_games=1000)
        >>> for game in loader.load_games(filter):
        ...     print(game.headers["White"], "vs", game.headers["Black"])
    """

    def __init__(self, pgn_path: str):
        """
        Initialize the game loader.

        Args:
            pgn_path: Path to PGN file or compressed PGN file (.pgn, .pgn.zst, .pgn.gz)
        """
        log = logger.bind(component="GameLoader.__init__")
        self.pgn_path = Path(pgn_path)
        self.eco_classifier = ECOClassifier()

        if not self.pgn_path.exists():
            log.error(f"PGN file not found: {pgn_path}")
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        log.info(f"GameLoader initialized with: {self.pgn_path}")

    def load_games(self, game_filter: Optional[GameFilter] = None, offset: int = 0) -> Iterator[chess.pgn.Game]:
        """
        Load games from PGN file with optional filtering.

        Args:
            game_filter: Filter criteria for games (None = load all)
            offset: Skip first N matching games (for pagination/diversity)

        Yields:
            chess.pgn.Game objects that match the filter

        Example:
            >>> loader = GameLoader("games.pgn")
            >>> filter = GameFilter(min_rating=2000, playstyle="tactical")
            >>> games = list(loader.load_games(filter, offset=1000))
            >>> print(f"Loaded {len(games)} tactical games starting from offset 1000")
        """
        log = logger.bind(component="GameLoader.load_games")
        if game_filter is None:
            game_filter = GameFilter()

        log.info(f"Loading games with filter: {game_filter}, offset: {offset}")

        games_loaded = 0
        games_filtered = 0
        games_skipped = 0

        # Open PGN file (handle compressed formats)
        pgn_file = self._open_pgn_file()
        log.info(f"Opened PGN file: {pgn_file}")

        try:
            while True:
                game = chess.pgn.read_game(pgn_file)

                if game is None:
                    break  # End of file

                games_filtered += 1

                # Apply filters
                if not self._passes_filter(game, game_filter):
                    continue

                # Skip games if we haven't reached offset yet
                if games_skipped < offset:
                    games_skipped += 1
                    # Log skip progress every 100 games
                    if games_skipped % 100 == 0:
                        logger.debug(f"Skipping to offset: {games_skipped}/{offset}")
                    continue

                games_loaded += 1
                yield game

                # Check max games limit
                if game_filter.max_games and games_loaded >= game_filter.max_games:
                    logger.info(f"Reached max_games limit: {game_filter.max_games}")
                    break

                # Log progress every 1000 games checked
                if games_filtered % 1000 == 0:
                    logger.debug(f"Checked {games_filtered} games, skipped {games_skipped}, loaded {games_loaded}")

        finally:
            pgn_file.close()

        logger.success(f"Loaded {games_loaded} games (checked {games_filtered} total, skipped {games_skipped} for offset)")

    def _open_pgn_file(self):
        """
        Open PGN file, handling compressed formats.

        Returns:
            File handle for reading PGN data
        """
        suffix = self.pgn_path.suffix.lower()

        if suffix == '.pgn':
            return open(self.pgn_path, 'r', encoding='utf-8', errors='ignore')

        elif suffix == '.zst':
            import zstandard as zstd
            import io
            dctx = zstd.ZstdDecompressor()
            f = open(self.pgn_path, 'rb')  # Don't use 'with' here!
            reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            return text_stream

        elif suffix == '.gz':
            import gzip
            return gzip.open(self.pgn_path, 'rt', encoding='utf-8', errors='ignore')

        else:
            raise ValueError(f"Unsupported file format: {suffix}")


    def _passes_filter(self, game: chess.pgn.Game, game_filter: GameFilter) -> bool:
        """
        Check if a game passes the filter criteria.

        Args:
            game: Chess game to check
            game_filter: Filter criteria

        Returns:
            True if game passes all filters, False otherwise
        """
        headers = game.headers

        # Filter by rating
        if game_filter.min_rating or game_filter.max_rating:
            avg_rating = self._get_average_rating(headers)
            if avg_rating is None:
                return False
            if game_filter.min_rating and avg_rating < game_filter.min_rating:
                return False
            if game_filter.max_rating and avg_rating > game_filter.max_rating:
                return False

        # Filter by playstyle (using ECO code)
        if game_filter.playstyle:
            eco = headers.get("ECO", "")
            if not eco:
                return False
            game_playstyle = self.eco_classifier.classify(eco)
            if game_playstyle != game_filter.playstyle:
                return False

        # Filter by time control
        if game_filter.time_control:
            time_control = headers.get("TimeControl", "")
            if not self._matches_time_control(time_control, game_filter.time_control):
                return False

        # Filter by result
        if game_filter.result:
            result = headers.get("Result", "*")
            if result != game_filter.result:
                return False

        return True

    def _get_average_rating(self, headers: Dict[str, str]) -> Optional[int]:
        """
        Get average rating of both players.

        Args:
            headers: Game headers dictionary

        Returns:
            Average rating or None if ratings not available
        """
        try:
            white_rating = int(headers.get("WhiteElo", 0))
            black_rating = int(headers.get("BlackElo", 0))

            if white_rating == 0 or black_rating == 0:
                return None

            return (white_rating + black_rating) // 2
        except (ValueError, TypeError):
            return None

    def _matches_time_control(self, game_tc: str, filter_tc: str) -> bool:
        """
        Check if game time control matches filter.

        Args:
            game_tc: Game time control string (e.g., "600+0", "180+2")
            filter_tc: Filter time control category (e.g., "blitz", "rapid", "classical")

        Returns:
            True if time control matches
        """
        if not game_tc or game_tc == "-":
            return False

        # Parse time control (format: "base+increment")
        try:
            parts = game_tc.split("+")
            base_time = int(parts[0])

            # Classify time control
            # Blitz: < 10 minutes
            # Rapid: 10-60 minutes
            # Classical: > 60 minutes
            if filter_tc.lower() == "blitz":
                return base_time < 600
            elif filter_tc.lower() == "rapid":
                return 600 <= base_time <= 3600
            elif filter_tc.lower() == "classical":
                return base_time > 3600
            else:
                return True  # Unknown filter, pass through

        except (ValueError, IndexError):
            return False

    def get_game_info(self, game: chess.pgn.Game) -> Dict[str, Any]:
        """
        Extract useful information from a game.

        Args:
            game: Chess game

        Returns:
            Dictionary with game information
        """
        headers = game.headers

        return {
            "white": headers.get("White", "?"),
            "black": headers.get("Black", "?"),
            "white_elo": headers.get("WhiteElo", "?"),
            "black_elo": headers.get("BlackElo", "?"),
            "result": headers.get("Result", "*"),
            "eco": headers.get("ECO", ""),
            "opening": headers.get("Opening", ""),
            "time_control": headers.get("TimeControl", "-"),
            "date": headers.get("Date", "????.??.??"),
            "playstyle": self.eco_classifier.classify(headers.get("ECO", "")),
        }


