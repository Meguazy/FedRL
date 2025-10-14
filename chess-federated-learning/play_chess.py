#!/usr/bin/env python3
"""
Chess Game Launcher - Play against your trained FL models!

Usage:
    python play_chess.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.chess_game import main

if __name__ == "__main__":
    main()
