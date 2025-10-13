"""
Unit tests for sample_extractor.py

Tests the SampleExtractor class to ensure correct extraction of training samples
from chess games.
"""

import pytest
import chess
import chess.pgn
import sys
from pathlib import Path

# Add parent directory to path
chess_fl_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chess_fl_dir))

from data.sample_extractor import SampleExtractor, TrainingSample, ExtractionConfig


# Sample PGN for testing
SAMPLE_PGN = """[Event "Test Game 1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2180"]
[ECO "B90"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Qb6
8. Qd2 Qxb2 9. Rb1 Qa3 10. e5 dxe5 11. fxe5 Nfd7 12. Bc4 Bb4 13. Rb3 Qa5
14. O-O Bxc3 15. Qxc3 Qxc3 16. Rxc3 1-0

[Event "Test Game 2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2120"]
[ECO "D63"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6
8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. O-O Nxc3 12. Rxc3 e5 0-1
"""


class TestTrainingSample:
    """Test suite for TrainingSample dataclass."""

    def test_training_sample_creation(self):
        """Test creating a TrainingSample."""
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")

        sample = TrainingSample(
            board=board,
            move_played=move,
            game_outcome=1.0,
            move_number=1,
            eco_code="B90",
            playstyle="tactical"
        )

        assert sample.board == board
        assert sample.move_played == move
        assert sample.game_outcome == 1.0
        assert sample.move_number == 1
        assert sample.eco_code == "B90"
        assert sample.playstyle == "tactical"


class TestExtractionConfig:
    """Test suite for ExtractionConfig dataclass."""

    def test_extraction_config_defaults(self):
        """Test ExtractionConfig default values."""
        config = ExtractionConfig()

        assert config.skip_opening_moves == 10
        assert config.skip_endgame_moves == 5
        assert config.sample_rate == 1.0
        assert config.max_positions_per_game is None
        assert config.shuffle_games is True

    def test_extraction_config_custom_values(self):
        """Test ExtractionConfig with custom values."""
        config = ExtractionConfig(
            skip_opening_moves=15,
            skip_endgame_moves=8,
            sample_rate=0.5,
            max_positions_per_game=20,
            shuffle_games=False
        )

        assert config.skip_opening_moves == 15
        assert config.skip_endgame_moves == 8
        assert config.sample_rate == 0.5
        assert config.max_positions_per_game == 20
        assert config.shuffle_games is False


class TestSampleExtractor:
    """Test suite for SampleExtractor class."""

    @pytest.fixture
    def temp_pgn_file(self, tmp_path):
        """Create a temporary PGN file for testing."""
        pgn_file = tmp_path / "test_games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        return str(pgn_file)

    @pytest.fixture
    def extractor(self, temp_pgn_file):
        """Create a SampleExtractor instance."""
        config = ExtractionConfig(
            skip_opening_moves=5,  # Shorter for testing
            skip_endgame_moves=4,
            sample_rate=1.0,
            shuffle_games=False  # Deterministic for testing
        )
        return SampleExtractor(temp_pgn_file, config)

    # Initialization tests
    def test_extractor_initialization(self, extractor):
        """Test SampleExtractor initializes correctly."""
        assert extractor is not None
        assert extractor.config is not None
        assert extractor.game_loader is not None

    def test_extractor_initialization_with_default_config(self, temp_pgn_file):
        """Test initialization with default config."""
        extractor = SampleExtractor(temp_pgn_file)
        assert extractor.config.skip_opening_moves == 10
        assert extractor.config.skip_endgame_moves == 5

    # Sample extraction tests
    def test_extract_samples_basic(self, extractor):
        """Test basic sample extraction."""
        samples = extractor.extract_samples(
            num_games=1,
            playstyle=None,
            min_rating=0
        )

        assert len(samples) > 0
        assert all(isinstance(s, TrainingSample) for s in samples)

    def test_extract_samples_skips_opening_moves(self, extractor):
        """Test that opening moves are skipped."""
        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        # All samples should have move_number > skip_opening_moves
        assert all(s.move_number > extractor.config.skip_opening_moves for s in samples)

    def test_extract_samples_filters_by_playstyle(self, extractor):
        """Test filtering samples by playstyle."""
        tactical_samples = extractor.extract_samples(
            num_games=1,
            playstyle="tactical",
            min_rating=0
        )

        # Should get tactical game (B90)
        assert len(tactical_samples) > 0
        assert all(s.playstyle == "tactical" for s in tactical_samples)

    def test_extract_samples_filters_by_rating(self, extractor):
        """Test filtering by minimum rating."""
        samples = extractor.extract_samples(
            num_games=2,
            min_rating=2150  # Should filter out game 2
        )

        # Should only get game 1
        assert len(samples) > 0
        eco_codes = set(s.eco_code for s in samples)
        assert "B90" in eco_codes
        assert "D63" not in eco_codes

    def test_extract_samples_with_offset(self, extractor):
        """Test using offset to skip games."""
        # Get first game
        samples1 = extractor.extract_samples(
            num_games=1,
            min_rating=0,
            offset=0
        )

        # Get second game (skip first)
        samples2 = extractor.extract_samples(
            num_games=1,
            min_rating=0,
            offset=1
        )

        # Should be different games
        eco1 = samples1[0].eco_code if samples1 else None
        eco2 = samples2[0].eco_code if samples2 else None

        assert eco1 != eco2

    # Game outcome tests
    def test_game_outcome_white_win(self, extractor):
        """Test game outcome encoding for white win."""
        samples = extractor.extract_samples(
            num_games=1,
            playstyle="tactical",
            min_rating=0
        )

        # Game 1 is 1-0 (white win)
        # Board is stored BEFORE move is played
        # White moves: board.turn == WHITE, outcome should be +1.0
        # Black moves: board.turn == BLACK, outcome should be -1.0
        white_moves = [s for s in samples if s.board.turn == chess.WHITE]
        black_moves = [s for s in samples if s.board.turn == chess.BLACK]

        if white_moves:
            # White wins, so white moves have +1.0 outcome
            assert all(s.game_outcome == 1.0 for s in white_moves)

        if black_moves:
            # White wins, so black moves have -1.0 outcome
            assert all(s.game_outcome == -1.0 for s in black_moves)

    def test_game_outcome_black_win(self, extractor):
        """Test game outcome encoding for black win."""
        samples = extractor.extract_samples(
            num_games=1,
            playstyle="positional",
            min_rating=0
        )

        # Game 2 is 0-1 (black win)
        # Board is stored BEFORE move is played
        # White moves: board.turn == WHITE, outcome should be -1.0
        # Black moves: board.turn == BLACK, outcome should be +1.0
        white_moves = [s for s in samples if s.board.turn == chess.WHITE]
        black_moves = [s for s in samples if s.board.turn == chess.BLACK]

        if white_moves:
            # Black wins, so white moves have -1.0 outcome
            assert all(s.game_outcome == -1.0 for s in white_moves)

        if black_moves:
            # Black wins, so black moves have +1.0 outcome
            assert all(s.game_outcome == 1.0 for s in black_moves)

    # Board state tests
    def test_sample_boards_are_copies(self, extractor):
        """Test that sample boards are independent copies."""
        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        if len(samples) >= 2:
            # Store original FEN
            original_fen_0 = samples[0].board.fen()
            original_fen_1 = samples[1].board.fen()

            # Modify one board by pushing any legal move
            legal_move = list(samples[0].board.legal_moves)[0]
            samples[0].board.push(legal_move)

            # First board should be different from its original state
            assert samples[0].board.fen() != original_fen_0

            # Second board should be unchanged
            assert samples[1].board.fen() == original_fen_1

    def test_sample_move_played_is_valid(self, extractor):
        """Test that move_played is legal in the position."""
        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        for sample in samples:
            # Move should be legal in the position
            assert sample.move_played in sample.board.legal_moves

    # Position filtering tests
    def test_skip_endgame_positions(self, temp_pgn_file):
        """Test that endgame positions are skipped."""
        config = ExtractionConfig(
            skip_opening_moves=0,
            skip_endgame_moves=10,  # Skip positions with < 10 pieces
            sample_rate=1.0
        )
        extractor = SampleExtractor(temp_pgn_file, config)

        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        # All samples should have >= 10 pieces
        for sample in samples:
            piece_count = len(sample.board.piece_map())
            assert piece_count >= config.skip_endgame_moves

    def test_sample_rate_reduces_samples(self, temp_pgn_file):
        """Test that sample_rate < 1.0 reduces number of samples."""
        # Full sampling
        config_full = ExtractionConfig(skip_opening_moves=5, sample_rate=1.0)
        extractor_full = SampleExtractor(temp_pgn_file, config_full)
        samples_full = extractor_full.extract_samples(num_games=1, min_rating=0)

        # Half sampling
        config_half = ExtractionConfig(skip_opening_moves=5, sample_rate=0.5)
        extractor_half = SampleExtractor(temp_pgn_file, config_half)
        samples_half = extractor_half.extract_samples(num_games=1, min_rating=0)

        # Half sampling should produce fewer samples (approximately half)
        assert len(samples_half) < len(samples_full)

    def test_max_positions_per_game_limit(self, temp_pgn_file):
        """Test max_positions_per_game limits samples."""
        config = ExtractionConfig(
            skip_opening_moves=0,
            max_positions_per_game=5
        )
        extractor = SampleExtractor(temp_pgn_file, config)

        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        # Should have at most 5 samples per game
        assert len(samples) <= 5

    # Helper method tests
    def test_parse_game_result_white_win(self, extractor):
        """Test parsing white win result."""
        white_outcome, black_outcome = extractor._parse_game_result("1-0")
        assert white_outcome == 1.0
        assert black_outcome == -1.0

    def test_parse_game_result_black_win(self, extractor):
        """Test parsing black win result."""
        white_outcome, black_outcome = extractor._parse_game_result("0-1")
        assert white_outcome == -1.0
        assert black_outcome == 1.0

    def test_parse_game_result_draw(self, extractor):
        """Test parsing draw result."""
        white_outcome, black_outcome = extractor._parse_game_result("1/2-1/2")
        assert white_outcome == 0.0
        assert black_outcome == 0.0

    def test_parse_game_result_unknown(self, extractor):
        """Test parsing unknown result."""
        white_outcome, black_outcome = extractor._parse_game_result("*")
        assert white_outcome == 0.0
        assert black_outcome == 0.0

    def test_count_pieces(self, extractor):
        """Test piece counting."""
        board = chess.Board()
        count = extractor._count_pieces(board)
        assert count == 32  # Starting position has 32 pieces

        board.clear()
        count_empty = extractor._count_pieces(board)
        assert count_empty == 0

    # Statistics tests
    def test_get_statistics(self, extractor):
        """Test getting extraction statistics."""
        samples = extractor.extract_samples(
            num_games=2,
            min_rating=0
        )

        stats = extractor.get_statistics(samples)

        assert "total_samples" in stats
        assert "outcomes" in stats
        assert "playstyles" in stats
        assert "move_stats" in stats
        assert stats["total_samples"] == len(samples)

    def test_get_statistics_empty_samples(self, extractor):
        """Test statistics with empty sample list."""
        stats = extractor.get_statistics([])
        assert stats["total_samples"] == 0

    def test_get_statistics_outcome_ratios(self, extractor):
        """Test outcome ratio calculations in statistics."""
        samples = extractor.extract_samples(
            num_games=2,
            min_rating=0
        )

        stats = extractor.get_statistics(samples)

        # Ratios should sum to 1.0
        win_ratio = stats["outcomes"]["win_ratio"]
        draw_ratio = stats["outcomes"]["draw_ratio"]
        loss_ratio = stats["outcomes"]["loss_ratio"]

        assert pytest.approx(win_ratio + draw_ratio + loss_ratio, 0.01) == 1.0

    def test_get_statistics_move_stats(self, extractor):
        """Test move statistics calculations."""
        samples = extractor.extract_samples(
            num_games=1,
            min_rating=0
        )

        stats = extractor.get_statistics(samples)

        assert stats["move_stats"]["min_move"] > 0
        assert stats["move_stats"]["max_move"] >= stats["move_stats"]["min_move"]
        assert stats["move_stats"]["avg_move"] > 0

    # Edge cases
    def test_extract_from_short_game(self, tmp_path):
        """Test extracting from very short game."""
        short_pgn = """[Event "Short Game"]
[White "P1"]
[Black "P2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]
[ECO "B00"]

1. e4 e5 2. Qh5 1-0
"""
        pgn_file = tmp_path / "short.pgn"
        pgn_file.write_text(short_pgn)

        config = ExtractionConfig(skip_opening_moves=5)
        extractor = SampleExtractor(str(pgn_file), config)

        samples = extractor.extract_samples(num_games=1, min_rating=0)

        # Very short game might have 0 samples after skipping opening
        assert len(samples) >= 0

    def test_extract_no_games_match_filter(self, extractor):
        """Test extraction when no games match filter."""
        samples = extractor.extract_samples(
            num_games=10,
            min_rating=3000  # No games have this rating
        )

        assert len(samples) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
