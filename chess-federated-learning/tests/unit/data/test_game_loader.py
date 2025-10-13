"""
Unit tests for game_loader.py

Tests the GameLoader class to ensure correct loading and filtering of chess games
from PGN database files.
"""

import pytest
import chess.pgn
import io
import sys
from pathlib import Path

# Add parent directory to path
chess_fl_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chess_fl_dir))

from data.game_loader import GameLoader, GameFilter


# Sample PGN data for testing
SAMPLE_PGN = """[Event "Test Game 1"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2180"]
[ECO "B90"]
[Opening "Sicilian Najdorf"]
[TimeControl "600+0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 1-0

[Event "Test Game 2"]
[Site "Test"]
[Date "2024.01.02"]
[Round "2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2120"]
[ECO "D63"]
[Opening "Queen's Gambit Declined"]
[TimeControl "180+2"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 0-1

[Event "Test Game 3"]
[Site "Test"]
[Date "2024.01.03"]
[Round "3"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]
[WhiteElo "1800"]
[BlackElo "1820"]
[ECO "B70"]
[Opening "Sicilian Dragon"]
[TimeControl "900+10"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 1/2-1/2
"""


class TestGameFilter:
    """Test suite for GameFilter dataclass."""

    def test_game_filter_defaults(self):
        """Test GameFilter initializes with default values."""
        filter = GameFilter()
        assert filter.min_rating == 1800
        assert filter.max_rating is None
        assert filter.playstyle is None
        assert filter.time_control is None
        assert filter.result is None
        assert filter.max_games is None

    def test_game_filter_custom_values(self):
        """Test GameFilter with custom values."""
        filter = GameFilter(
            min_rating=2000,
            max_rating=2500,
            playstyle="tactical",
            time_control="blitz",
            result="1-0",
            max_games=100
        )
        assert filter.min_rating == 2000
        assert filter.max_rating == 2500
        assert filter.playstyle == "tactical"
        assert filter.time_control == "blitz"
        assert filter.result == "1-0"
        assert filter.max_games == 100


class TestGameLoader:
    """Test suite for GameLoader class."""

    @pytest.fixture
    def temp_pgn_file(self, tmp_path):
        """Create a temporary PGN file for testing."""
        pgn_file = tmp_path / "test_games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        return str(pgn_file)

    @pytest.fixture
    def loader(self, temp_pgn_file):
        """Create a GameLoader instance with test PGN file."""
        return GameLoader(temp_pgn_file)

    # Initialization tests
    def test_loader_initialization(self, loader):
        """Test that GameLoader initializes correctly."""
        assert loader is not None
        assert loader.pgn_path.exists()
        assert loader.eco_classifier is not None

    def test_loader_initialization_missing_file(self):
        """Test that initialization fails with missing file."""
        with pytest.raises(FileNotFoundError):
            GameLoader("/nonexistent/path/games.pgn")

    # Basic loading tests
    def test_load_all_games(self, loader):
        """Test loading all games without filters."""
        games = list(loader.load_games())
        assert len(games) == 3

    def test_load_games_returns_chess_game_objects(self, loader):
        """Test that loaded games are chess.pgn.Game objects."""
        games = list(loader.load_games())
        for game in games:
            assert isinstance(game, chess.pgn.Game)

    def test_load_games_preserves_headers(self, loader):
        """Test that game headers are preserved."""
        games = list(loader.load_games())

        assert games[0].headers["White"] == "Player1"
        assert games[0].headers["ECO"] == "B90"
        assert games[1].headers["White"] == "Player3"
        assert games[1].headers["ECO"] == "D63"

    # Rating filter tests
    def test_filter_by_min_rating(self, loader):
        """Test filtering by minimum rating."""
        filter = GameFilter(min_rating=2000)
        games = list(loader.load_games(filter))

        # Should only get games 1 and 2 (avg ratings 2190 and 2110)
        # Game 3 has avg rating 1810, should be filtered out
        assert len(games) == 2

    def test_filter_by_max_rating(self, loader):
        """Test filtering by maximum rating."""
        filter = GameFilter(min_rating=0, max_rating=2000)
        games = list(loader.load_games(filter))

        # Should only get game 3 (avg rating 1810)
        assert len(games) == 1
        assert games[0].headers["White"] == "Player5"

    def test_filter_by_rating_range(self, loader):
        """Test filtering by rating range."""
        filter = GameFilter(min_rating=2050, max_rating=2150)
        games = list(loader.load_games(filter))

        # Should only get game 2 (avg rating 2110)
        assert len(games) == 1
        assert games[0].headers["White"] == "Player3"

    # Playstyle filter tests
    def test_filter_by_tactical_playstyle(self, loader):
        """Test filtering by tactical playstyle."""
        filter = GameFilter(playstyle="tactical", min_rating=0)
        games = list(loader.load_games(filter))

        # Should get games 1 and 3 (B90 and B70 are tactical)
        assert len(games) == 2
        eco_codes = [game.headers["ECO"] for game in games]
        assert "B90" in eco_codes
        assert "B70" in eco_codes

    def test_filter_by_positional_playstyle(self, loader):
        """Test filtering by positional playstyle."""
        filter = GameFilter(playstyle="positional", min_rating=0)
        games = list(loader.load_games(filter))

        # Should get game 2 (D63 is positional)
        assert len(games) == 1
        assert games[0].headers["ECO"] == "D63"

    # Time control filter tests
    def test_filter_by_blitz_time_control(self, loader):
        """Test filtering by blitz time control."""
        filter = GameFilter(time_control="blitz", min_rating=0)
        games = list(loader.load_games(filter))

        # Game 2 has 180+2 (blitz: < 10 minutes)
        assert len(games) == 1
        assert games[0].headers["White"] == "Player3"

    def test_filter_by_rapid_time_control(self, loader):
        """Test filtering by rapid time control."""
        filter = GameFilter(time_control="rapid", min_rating=0)
        games = list(loader.load_games(filter))

        # Games 1 and 3 have 600+0 and 900+10 (rapid: 10-60 minutes)
        assert len(games) == 2

    # Result filter tests
    def test_filter_by_white_win(self, loader):
        """Test filtering by white win."""
        filter = GameFilter(result="1-0", min_rating=0)
        games = list(loader.load_games(filter))

        assert len(games) == 1
        assert games[0].headers["Result"] == "1-0"

    def test_filter_by_black_win(self, loader):
        """Test filtering by black win."""
        filter = GameFilter(result="0-1", min_rating=0)
        games = list(loader.load_games(filter))

        assert len(games) == 1
        assert games[0].headers["Result"] == "0-1"

    def test_filter_by_draw(self, loader):
        """Test filtering by draw."""
        filter = GameFilter(result="1/2-1/2", min_rating=0)
        games = list(loader.load_games(filter))

        assert len(games) == 1
        assert games[0].headers["Result"] == "1/2-1/2"

    # Max games filter tests
    def test_filter_max_games(self, loader):
        """Test limiting number of games loaded."""
        filter = GameFilter(max_games=2, min_rating=0)
        games = list(loader.load_games(filter))

        assert len(games) == 2

    def test_filter_max_games_larger_than_available(self, loader):
        """Test max_games larger than available games."""
        filter = GameFilter(max_games=10, min_rating=0)
        games = list(loader.load_games(filter))

        # Should get all 3 games
        assert len(games) == 3

    # Combined filter tests
    def test_multiple_filters_combined(self, loader):
        """Test combining multiple filters."""
        filter = GameFilter(
            min_rating=2000,
            playstyle="tactical",
            max_games=1
        )
        games = list(loader.load_games(filter))

        # Should get 1 tactical game with rating >= 2000
        assert len(games) == 1
        assert games[0].headers["ECO"] in ["B90", "B70"]

    # Helper method tests
    def test_get_average_rating(self, loader):
        """Test _get_average_rating method."""
        headers = {"WhiteElo": "2200", "BlackElo": "2180"}
        avg = loader._get_average_rating(headers)
        assert avg == 2190

    def test_get_average_rating_missing_elo(self, loader):
        """Test _get_average_rating with missing ELO."""
        headers = {"WhiteElo": "2200"}
        avg = loader._get_average_rating(headers)
        assert avg is None

    def test_get_average_rating_invalid_elo(self, loader):
        """Test _get_average_rating with invalid ELO."""
        headers = {"WhiteElo": "invalid", "BlackElo": "2180"}
        avg = loader._get_average_rating(headers)
        assert avg is None

    def test_matches_time_control_blitz(self, loader):
        """Test time control matching for blitz."""
        assert loader._matches_time_control("180+2", "blitz") is True
        assert loader._matches_time_control("600+0", "blitz") is False

    def test_matches_time_control_rapid(self, loader):
        """Test time control matching for rapid."""
        assert loader._matches_time_control("600+0", "rapid") is True
        assert loader._matches_time_control("1800+0", "rapid") is True
        assert loader._matches_time_control("180+2", "rapid") is False

    def test_matches_time_control_classical(self, loader):
        """Test time control matching for classical."""
        assert loader._matches_time_control("3600+0", "classical") is False
        assert loader._matches_time_control("7200+30", "classical") is True

    def test_matches_time_control_invalid(self, loader):
        """Test time control matching with invalid format."""
        assert loader._matches_time_control("-", "blitz") is False
        assert loader._matches_time_control("invalid", "blitz") is False

    # get_game_info tests
    def test_get_game_info(self, loader):
        """Test getting game information."""
        games = list(loader.load_games(GameFilter(min_rating=0)))
        info = loader.get_game_info(games[0])

        assert info["white"] == "Player1"
        assert info["black"] == "Player2"
        assert info["white_elo"] == "2200"
        assert info["black_elo"] == "2180"
        assert info["result"] == "1-0"
        assert info["eco"] == "B90"
        assert info["opening"] == "Sicilian Najdorf"
        assert info["time_control"] == "600+0"
        assert info["playstyle"] == "tactical"

    def test_get_game_info_missing_headers(self, loader):
        """Test get_game_info with missing headers."""
        # Create a minimal game
        game = chess.pgn.Game()
        info = loader.get_game_info(game)

        assert info["white"] == "?"
        assert info["white_elo"] == "?"
        assert info["result"] == "*"
        assert info["eco"] == ""

    # Edge cases
    def test_empty_pgn_file(self, tmp_path):
        """Test loading from empty PGN file."""
        empty_pgn = tmp_path / "empty.pgn"
        empty_pgn.write_text("")

        loader = GameLoader(str(empty_pgn))
        games = list(loader.load_games())

        assert len(games) == 0

    def test_games_with_missing_eco(self, tmp_path):
        """Test games without ECO codes are handled correctly."""
        pgn_no_eco = """[Event "No ECO"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. e4 e5 1-0
"""
        pgn_file = tmp_path / "no_eco.pgn"
        pgn_file.write_text(pgn_no_eco)

        loader = GameLoader(str(pgn_file))
        filter = GameFilter(playstyle="tactical", min_rating=0)
        games = list(loader.load_games(filter))

        # Should filter out game without ECO code
        assert len(games) == 0

    def test_filter_no_games_match(self, loader):
        """Test filter that matches no games."""
        filter = GameFilter(min_rating=3000)  # No games have 3000+ rating
        games = list(loader.load_games(filter))

        assert len(games) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
