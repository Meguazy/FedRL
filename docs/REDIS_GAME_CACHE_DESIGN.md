# Redis Game Cache Architecture (REVISED)

## Core Insight: Cache Games, Not Samples!

### Why This Is Better

**Original Idea:** Cache extracted samples in Redis
- ❌ Huge storage: All positions from all games
- ❌ Inflexible: Can't change extraction config
- ❌ Still memory-heavy: 2.8GB

**Better Idea:** Cache games in Redis
- ✅ Small storage: Just the games themselves (~2KB each)
- ✅ Flexible: Extract samples on-the-fly with any config
- ✅ Tiny memory: ~400MB for 20K games
- ✅ Same speed: No decompression overhead

---

## Data Flow

### ONE-TIME INDEXING
```
┌─────────────────────────────┐
│ Compressed PGN Database     │
│ (tactical.pgn.zst - 500MB)  │
└──────────────┬──────────────┘
               │
               ↓ Read once, decompress
┌──────────────────────────────┐
│ index_games_to_redis.py      │
│ - GameLoader.load_games()    │
│ - Filter by rating/playstyle │
│ - Convert game → PGN string  │
└──────────────┬───────────────┘
               │
               ↓ Store
┌──────────────────────────────┐
│ Redis (400MB)                │
│ chess:game:tactical:0        │
│ chess:game:tactical:1        │
│ ...                          │
│ chess:game:tactical:9999     │
└──────────────────────────────┘
```

### TRAINING (EVERY ROUND)
```
┌──────────────────────────────┐
│ Trainer calls:               │
│ SampleExtractor.extract()    │
└──────────────┬───────────────┘
               │
               ↓ Calls
┌──────────────────────────────┐
│ GameLoader.load_games()      │
│ ┌──────────────────────────┐ │
│ │ Check Redis?             │ │
│ │  YES → Fetch from Redis  │ │ ← FAST! No decompression
│ │  NO  → Read from .zst    │ │ ← Slower, but still works
│ └──────────────────────────┘ │
└──────────────┬───────────────┘
               │
               ↓ Returns game iterator
┌──────────────────────────────┐
│ SampleExtractor              │
│ For each game:               │
│  - Extract positions         │
│  - Create TrainingSamples    │
└──────────────┬───────────────┘
               │
               ↓ Returns samples
┌──────────────────────────────┐
│ Trainer uses samples         │
│ for training batch           │
└──────────────────────────────┘
```

---

## Key Design Decisions

### 1. GameLoader Abstraction
```python
# GameLoader hides Redis completely!
# Callers don't need to know about Redis

# In trainer (NO CHANGES NEEDED):
extractor = SampleExtractor("database.pgn.zst")
samples = extractor.extract_samples(playstyle='tactical', offset=100)

# Inside GameLoader (NEW):
def load_games(self, game_filter, offset):
    if self.redis_cache and self.redis_cache.has_games(playstyle):
        # Fast path: games from Redis
        for game in self.redis_cache.get_games(playstyle, offset, count):
            yield game
    else:
        # Slow path: decompress file
        for game in self._load_from_file():
            yield game
```

**Result:** Transparent caching! No trainer code changes!

### 2. PGN String Storage
```python
# Each game stored as PGN string:
redis.set('chess:game:tactical:0', """
[Event "Rated Blitz game"]
[Site "https://lichess.org/abc123"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2450"]
[BlackElo "2420"]
[ECO "C50"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 ...
""")

# When retrieving:
pgn_string = redis.get('chess:game:tactical:0')
game = chess.pgn.read_game(io.StringIO(pgn_string))
```

**Benefits:**
- Standard format
- Human-readable (debug-friendly)
- Small (~2KB per game)
- Easy to parse back to chess.pgn.Game

### 3. Memory Calculation
```
Storage per game:
- Headers: ~500 bytes (Event, Site, Players, Ratings, ECO, etc.)
- Moves: ~1000-1500 bytes (40 moves × ~30 bytes)
- Total: ~2KB per game

Total for 20,000 games:
- Raw: 20,000 × 2KB = 40MB
- Redis overhead: ~10x = 400MB (safe estimate)
```

**Comparison:**
- Samples: 20,000 games × 70 samples × ~2KB = **2.8GB**
- Games: 20,000 games × 2KB = **40MB raw, 400MB with overhead**
- **7x smaller!**

---

## Implementation Steps (REVISED)

### Step 1: Docker Compose
```yaml
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 1gb --maxmemory-policy noeviction
    ports:
      - "6379:6379"
    volumes:
      - ./storage/redis-data:/data
```
Only need 1GB (not 4GB!) since we're storing games not samples.

### Step 2: RedisGameCache Class
```python
class RedisGameCache:
    def store_game(self, playstyle: str, index: int, pgn: str):
        """Store a game as PGN string."""
        key = f"chess:game:{playstyle}:{index}"
        self.redis.set(key, pgn)
    
    def get_games(self, playstyle: str, offset: int, count: int) -> Iterator[chess.pgn.Game]:
        """Fetch games and yield as chess.pgn.Game objects."""
        pipeline = self.redis.pipeline()
        for i in range(offset, offset + count):
            pipeline.get(f"chess:game:{playstyle}:{i}")
        
        pgn_strings = pipeline.execute()
        
        for pgn_str in pgn_strings:
            if pgn_str:
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                yield game
```

### Step 3: Indexing Script
```python
# scripts/index_games_to_redis.py

cache = RedisGameCache()
loader = GameLoader(pgn_path, use_redis=False)  # Force file reading

game_filter = GameFilter(min_rating=2000, playstyle='tactical')
index = 0

for game in loader.load_games(game_filter):
    # Convert game to PGN string
    exporter = chess.pgn.StringExporter()
    pgn_string = game.accept(exporter)
    
    # Store in Redis
    cache.store_game('tactical', index, pgn_string)
    index += 1
    
    if index % 100 == 0:
        print(f"Indexed {index} games...")
```

### Step 4: Update GameLoader
```python
class GameLoader:
    def __init__(self, pgn_path: str, use_redis: bool = True):
        self.pgn_path = Path(pgn_path)
        self.redis_cache = RedisGameCache() if use_redis else None
    
    def load_games(self, game_filter=None, offset=0):
        # Try Redis first
        if self.redis_cache and game_filter and game_filter.playstyle:
            total = self.redis_cache.get_total_games(game_filter.playstyle)
            if total > 0:
                yield from self.redis_cache.get_games(
                    game_filter.playstyle,
                    offset,
                    game_filter.max_games or 9999
                )
                return
        
        # Fall back to file
        yield from self._load_from_file(game_filter, offset)
```

### Step 5: No Trainer Changes!
The trainer already does:
```python
extractor = SampleExtractor(pgn_path)
samples = extractor.extract_samples(playstyle='tactical')
```

SampleExtractor calls `GameLoader.load_games()`, which now transparently uses Redis. **Zero changes needed!**

---

## Performance Impact

### Before (Current)
```
Node starts training round
    ↓
Open tactical.pgn.zst (500MB compressed)
    ↓
Decompress with zstandard (CPU intensive)
    ↓
Read through file to offset (I/O intensive)
    ↓
Extract 100 games worth of samples
    ↓
TIME: 30s - 3min per round
```

### After (With Redis Game Cache)
```
Node starts training round
    ↓
Fetch 100 games from Redis (network call)
    ↓
Parse PGN strings to chess.pgn.Game objects
    ↓
Extract samples from games
    ↓
TIME: <5s per round (6x-36x faster!)
```

---

## Memory Usage

### Redis Server
```
Games: 400MB (20,000 games)
Overhead: Redis process ~100MB
Total: ~500MB
```

### Each Training Node
```
Before: ~50MB (no cache)
After:  ~50MB (fetches games on demand)
No change! But much faster!
```

---

## Why This Is Brilliant

1. **Minimal Storage:** Games are tiny vs samples
2. **Maximum Flexibility:** Can change extraction config anytime
3. **No Trainer Changes:** GameLoader abstraction handles everything
4. **Graceful Fallback:** Works without Redis (just slower)
5. **Shared Benefit:** All 8 nodes share one cache
6. **Fast Indexing:** Just copying PGN strings, no extraction
7. **Easy Debugging:** Can inspect games in Redis directly

---

## Migration Path

1. ✅ Keep current system working (no Redis)
2. ✅ Add Redis as optional optimization
3. ✅ Index games to Redis when convenient
4. ✅ Nodes automatically use Redis if available
5. ✅ Can always fall back to files

**Zero risk, pure upside!**

---

## Approve This Plan?

- [ ] Store **games** in Redis (not samples) ✓
- [ ] Use GameLoader abstraction for transparency ✓
- [ ] PGN string format ✓
- [ ] No trainer changes needed ✓
- [ ] ~400MB Redis memory (not 2.8GB) ✓
- [ ] Graceful fallback to files ✓

**Ready to implement?** 🚀
