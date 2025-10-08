"""
Data module for chess game extraction and processing.

This module handles:
- Downloading and extracting games from chess databases (Lichess, etc.)
- Filtering games by ECO code, rating, time control
- Converting games to training samples (board states, moves, outcomes)
- Distributing games across clusters (tactical vs positional)
"""

from .game_loader import GameLoader
from .eco_classifier import ECOClassifier, classify_game_by_eco

__all__ = [
    "GameLoader",
    "ECOClassifier",
    "classify_game_by_eco"
]
