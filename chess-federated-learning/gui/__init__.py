"""
GUI Package for Chess Game Application.

This package provides a graphical user interface for playing chess
against trained federated learning models.
"""

from .chess_game import ChessGameWindow, ChessAI, ChessBoardWidget

__all__ = ['ChessGameWindow', 'ChessAI', 'ChessBoardWidget']
