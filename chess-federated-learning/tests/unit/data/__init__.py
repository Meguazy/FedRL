"""
Unit tests for the data module.

This package contains unit tests for:
- board_encoder: Chess board to 119-plane tensor encoding
- eco_classifier: ECO code classification (tactical vs positional)
- game_loader: PGN game loading and filtering
- sample_extractor: Training sample extraction from games
"""

__all__ = [
    "test_board_encoder",
    "test_eco_classifier",
    "test_game_loader",
    "test_sample_extractor",
]
