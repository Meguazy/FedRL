# Chess Game Data Directory

This directory contains modules for extracting and processing chess games from online databases.

## ⚠️ Important: Database Files Not Included

Chess database files are **NOT included in git** due to their size (several GB each). You need to download them separately.

## Downloading Chess Databases

### Option 1: Lichess Database (Recommended)

Lichess provides free monthly archives of all games played on their platform:

**Download location**: https://database.lichess.org/

**Recommended files**:
- `lichess_db_standard_rated_YYYY-MM.pgn.zst` - All rated standard games for a month
- Files are ~2-5 GB compressed, ~15-30 GB uncompressed

**Example download**:
```bash
# Download to data directory
cd chess-federated-learning/data/

# Download January 2024 games (example)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# Or use curl
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
```

**File format**: `.pgn.zst` (Zstandard compressed PGN)

### Option 2: Smaller Test Database

For testing, you can use a smaller PGN file:

```bash
# Download a single Lichess game export (much smaller)
wget https://lichess.org/game/export/<game-id>
```

Or create your own PGN from personal games.

## Required Dependencies

To read compressed Lichess databases:

```bash
pip install zstandard
# or
uv pip install zstandard
```

## Directory Structure

```
data/
├── README.md                    # This file
├── __init__.py                  # Module initialization
├── eco_classifier.py            # ECO code classifier (tactical/positional)
├── game_loader.py               # Load games from PGN files
├── databases/                   # Downloaded databases (gitignored)
│   └── lichess_db_*.pgn.zst
├── processed/                   # Processed training data (gitignored)
└── cache/                       # Cached data (gitignored)
```

## Usage Examples

### 1. Load Tactical Games

```python
from data.game_loader import GameLoader, GameFilter

# Load games
loader = GameLoader("data/databases/lichess_db_standard_rated_2024-01.pgn.zst")

# Filter for tactical games (rating > 2000)
filter = GameFilter(
    min_rating=2000,
    playstyle="tactical",
    max_games=1000
)

# Iterate over games
for game in loader.load_games(filter):
    print(f"Loaded: {game.headers['White']} vs {game.headers['Black']}")
```

### 2. Load Positional Games

```python
filter = GameFilter(
    min_rating=2200,
    playstyle="positional",
    time_control="classical",
    max_games=500
)

for game in loader.load_games(filter):
    # Process game...
    pass
```

### 3. Classify Games by ECO

```python
from data.eco_classifier import classify_game_by_eco

eco_code = "B90"  # Sicilian Najdorf
playstyle = classify_game_by_eco(eco_code)
print(f"{eco_code} is {playstyle}")  # Output: "B90 is tactical"
```

## Filtering Options

The `GameFilter` class supports:

| Filter | Description | Example |
|--------|-------------|---------|
| `min_rating` | Minimum average player rating | `2000` |
| `max_rating` | Maximum average player rating | `2500` |
| `playstyle` | "tactical" or "positional" | `"tactical"` |
| `time_control` | "blitz", "rapid", "classical" | `"rapid"` |
| `result` | "1-0", "0-1", "1/2-1/2" | `"1-0"` |
| `max_games` | Maximum games to load | `1000` |

## ECO Classification

Games are classified as **tactical** or **positional** based on their ECO opening code:

### Tactical Openings (168 codes)
- Sicilian Defense (B20-B99)
- King's Gambit (C30-C39)
- Italian Game (C50-C59)
- King's Indian Attack (E60-E99)
- And more...

### Positional Openings (196 codes)
- Queen's Gambit Declined (D30-D69)
- Caro-Kann Defense (B10-B19)
- Nimzo-Indian Defense (E20-E59)
- English Opening (A10-A39)
- And more...

See [eco_classifier.py](eco_classifier.py) for the complete list.

## Performance Notes

- **Compressed files**: The loader can read `.zst`, `.gz`, and `.bz2` compressed PGN files directly
- **Memory efficient**: Uses iterators, doesn't load entire database into memory
- **Fast filtering**: ECO-based filtering is very fast (O(1) lookup)
- **Large databases**: Can handle multi-gigabyte files efficiently

## Database Statistics

Typical Lichess monthly database (standard rated):
- **Size**: 2-5 GB compressed, 15-30 GB uncompressed
- **Games**: 80-120 million games per month
- **Rating range**: 800-3000+ ELO
- **Time controls**: All (bullet, blitz, rapid, classical)

For supervised learning, recommended to use:
- **Rating filter**: 2000+ for high-quality games
- **Sample size**: 10K-1M games depending on compute
- **Balance**: Equal tactical and positional games per cluster

## Troubleshooting

### "File not found" error
Make sure you've downloaded the database to the correct location:
```bash
ls -lh data/databases/
```

### "zstandard not installed" error
Install the zstandard library:
```bash
pip install zstandard
```

### Slow loading
- Use compressed files (much faster than extracting first)
- Set `max_games` limit for testing
- Consider using a smaller time range (one month vs. full year)

### Out of memory
- The loader uses iterators, so memory usage should be minimal
- Process games one at a time instead of loading all into a list
- Reduce `max_games` for initial testing

## Next Steps

After setting up the data loader:
1. Download a Lichess database
2. Test the game loader with a small `max_games` limit
3. Implement board encoding (119-plane representation)
4. Implement move encoding (4672 action space)
5. Create training dataset class
6. Integrate with supervised trainer
