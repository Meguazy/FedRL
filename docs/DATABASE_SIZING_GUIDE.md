# Database Sizing Guide

## How Many Games Do You Need?

### Quick Calculator

**Formula:** `Games Needed = Nodes × Games_per_round × Rounds × Diversity_Factor`

For your setup:
- **8 nodes** (4 tactical + 4 positional)
- **100 games per node per round**
- **100 rounds**
- **Diversity factor: 1.0** (each node gets unique games)

### Calculation

**Per Round:**
- Tactical cluster: 4 nodes × 100 games = **400 games/round**
- Positional cluster: 4 nodes × 100 games = **400 games/round**
- Total: **800 games/round**

**For 100 Rounds:**
- Tactical: 400 × 100 = **40,000 games**
- Positional: 400 × 100 = **40,000 games**
- **Total: 80,000 filtered games needed**

**With safety margin (×2 for filtering):**
- Need ~160,000 raw games to get 80,000 after filtering

## Your Current Database

**January 2024 Lichess** (`lichess_db_standard_rated_2024-01.pgn.zst`):

| Metric | Count |
|--------|-------|
| Total games | ~80-100 million |
| Rating ≥ 2000 | ~8-10 million |
| Tactical (after ECO filter) | ~4-5 million |
| Positional (after ECO filter) | ~4-5 million |

### Verdict: ✅ MORE THAN ENOUGH!

Your single database can support:
- **100 rounds**: Uses 40k/4M = **1% of available games**
- **1,000 rounds**: Uses 400k/4M = **10% of available games**
- **10,000 rounds**: Uses 4M/4M = **100% of available games**

**You can train for 10,000+ rounds without running out of data!**

## When to Download More Databases

### Scenarios Where You Might Want More

1. **Extreme long training** (>5,000 rounds)
2. **More diversity** (want samples from different time periods)
3. **Different player pools** (different months = different players)
4. **Redundancy** (backup if one database has issues)

### Available Databases

Lichess publishes monthly databases: https://database.lichess.org/

**2024 Databases:**

| Month | File | Size | Games |
|-------|------|------|-------|
| January | `lichess_db_standard_rated_2024-01.pgn.zst` | 31GB | ~95M |
| February | `lichess_db_standard_rated_2024-02.pgn.zst` | 28GB | ~85M |
| March | `lichess_db_standard_rated_2024-03.pgn.zst` | 33GB | ~100M |
| April | `lichess_db_standard_rated_2024-04.pgn.zst` | 31GB | ~95M |
| May | `lichess_db_standard_rated_2024-05.pgn.zst` | 34GB | ~105M |

**Recommendation for 100 rounds:** Use what you have! ✅

**Recommendation for 1,000+ rounds:** Download 2-3 additional months for variety

## How to Use Multiple Databases

### Option 1: Separate Preprocessing (Simple)

Process each database separately and rotate between them:

```bash
# Download additional month
./scripts/download_additional_databases.sh

# Preprocess into separate cache
uv run python -m chess-federated-learning.data.database_preprocessor \
    --input databases/lichess_db_standard_rated_2024-02.pgn.zst \
    --output data/cache_feb2024 \
    --min-rating 2000 \
    --playstyle both

# Update node configs to alternate
# Round 0-99: Use cache_jan2024
# Round 100-199: Use cache_feb2024
```

### Option 2: Combined Cache (Better)

Merge all databases into one mega-cache:

```bash
# Download multiple months
./scripts/download_additional_databases.sh

# Combine into one cache
./scripts/preprocess_multiple_databases.sh
```

This creates a single cache with samples from all databases mixed together.

### Option 3: Dynamic Database Selection

Modify `trainer_supervised.py` to rotate databases:

```python
# In SupervisedTrainer
database_files = [
    "databases/lichess_db_standard_rated_2024-01.pgn.zst",
    "databases/lichess_db_standard_rated_2024-02.pgn.zst",
    "databases/lichess_db_standard_rated_2024-03.pgn.zst",
]

# Rotate every N rounds
database_index = (current_round // 100) % len(database_files)
self.pgn_database_path = database_files[database_index]
```

## Storage Requirements

### Single Database (Current)

| Item | Size |
|------|------|
| Raw database (.zst) | 31 GB |
| Preprocessed cache | ~10 GB |
| **Total** | **41 GB** |

### Multiple Databases (3 months)

| Item | Size |
|------|------|
| Raw databases (.zst) | 93 GB |
| Preprocessed cache | ~30 GB |
| **Total** | **123 GB** |

### Recommendation

**For 100 rounds:** Keep current setup (41 GB)

**For 1,000+ rounds:** Add 1-2 more months (82 GB total)

## Data Diversity Analysis

### With 1 Database (4M games per playstyle)

- **Round 1-100**: Fresh games, no repeats
- **Round 101-200**: Fresh games, no repeats
- **Round 1,000+**: Fresh games, no repeats
- **Round 10,000**: Starting to exhaust unique games

### With 3 Databases (12M games per playstyle)

- Can train for **30,000 rounds** without repeating games
- Samples from different time periods (variety in meta)
- Different player pools (variety in style)

## Practical Recommendations

### For Your Use Case (100 rounds)

```bash
# 1. Use current database - it's perfect!
./scripts/preprocess_database.sh

# 2. Start training
# You'll use only 1% of available games
```

### For Extended Training (1,000 rounds)

```bash
# 1. Download one additional month for variety
./scripts/download_additional_databases.sh
# Choose option 2 (February) or 3 (March)

# 2. Preprocess both into combined cache
./scripts/preprocess_multiple_databases.sh

# 3. Train with 8M games available
# You'll use only 10% of available games
```

### For Research/Production (10,000+ rounds)

```bash
# 1. Download 3-6 months
./scripts/download_additional_databases.sh  # Repeat for each month

# 2. Combine all into mega-cache
./scripts/preprocess_multiple_databases.sh

# 3. Train for years without running out
# 6 months = ~24M games per playstyle
```

## Monitoring Data Usage

Add to your training logs:

```python
# In trainer_supervised.py
total_samples_available = len(cache_loader.get_samples(playstyle))
samples_used_so_far = current_round * samples_per_round
usage_percentage = (samples_used_so_far / total_samples_available) * 100

log.info(f"Data usage: {samples_used_so_far:,}/{total_samples_available:,} "
         f"({usage_percentage:.1f}%)")
```

This way you'll know when you're running low and need to add more data.

## Summary

### Current Status
- ✅ **You have 4-5M games per playstyle**
- ✅ **You need only 40k for 100 rounds**
- ✅ **You can train 10,000+ rounds with current database**

### Recommendation
**Use what you have!** Your single January 2024 database is more than sufficient for 100 rounds. Focus on:
1. Preprocessing your current database
2. Starting training
3. Monitoring results

Only download additional databases if:
- You plan to train for >5,000 rounds
- You want maximum diversity
- You have extra storage space

**Bottom line:** You're good to go with what you have! 🚀
