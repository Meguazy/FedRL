# Redis Cache Migration Plan

## Overview

**Goal:** Replace local pickle file caching with Redis-based shared cache to solve memory issues.

**Current Problem:**
- Each of 8 nodes loads 1.4GB pickle file → 11.2GB total memory
- Causes OOM crashes when all nodes start simultaneously

**Solution:**
- One Redis instance stores all samples → 2.8GB total memory
- Nodes fetch only what they need → ~50MB per node
- **80% memory reduction**

---

## Architecture

### Current Flow (File-based)
```
PGN Database (compressed .zst)
    ↓
GameLoader.load_games() → Decompresses on-the-fly → Iterates games
    ↓
SampleExtractor.extract_samples() → Extracts positions from games
    ↓
Trainer uses samples
```

**Problems:**
- ❌ Decompress .zst file every training round (30s - 3min per node)
- ❌ File I/O overhead on compressed archives
- ❌ No caching = repeated work

### New Flow (Redis Game Cache)
```
ONE-TIME INDEXING:
PGN Database (compressed .zst)
    ↓
index_games_to_redis.py → Streams games → Stores as PGN strings in Redis
    ↓
Redis (shared, ~400MB for games)

TRAINING (every round):
Redis → GameLoader.load_games() → Returns cached games (no decompression!)
    ↓
SampleExtractor.extract_samples() → Extracts positions (same as before)
    ↓
Trainer uses samples
```

**Benefits:**
- ✅ No decompression overhead during training
- ✅ Shared game cache across all nodes
- ✅ Much smaller storage (games << samples)
- ✅ Flexible: Change extraction config without reindexing
- ✅ Transparent: GameLoader abstraction hides Redis
- ✅ Automatic persistence (Redis RDB/AOF)

---

## Implementation Steps

### Step 1: Docker Compose with Redis
**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: chess-redis-cache
    command: redis-server --maxmemory 4gb --maxmemory-policy noeviction --save 60 1000 --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - ./storage/redis-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
```

**Why:** 
- Separate microservice, easy to manage
- Persistent storage in `./storage/redis-data`
- Health checks ensure availability
- 4GB max memory (room for both tactical + positional)

---

### Step 2: RedisGameCache Adapter Class
**File:** `chess-federated-learning/data/redis_game_cache.py`

**Interface:**
```python
class RedisGameCache:
    def __init__(self, host='localhost', port=6379)
    def store_game(self, playstyle: str, index: int, game_pgn: str)
    def get_games(self, playstyle: str, offset: int, count: int) -> Iterator[chess.pgn.Game]
    def get_total_games(self, playstyle: str) -> int
    def clear_cache(self, playstyle: Optional[str] = None)
    def get_stats() -> Dict
```

**Redis Keys Schema:**
```
chess:game:tactical:0          → "[Event ...]\n[White ...]...\n1. e4 e5 ..."  (PGN string)
chess:game:tactical:1          → "[Event ...]\n[White ...]...\n1. d4 d5 ..."
...
chess:game:positional:0        → "[Event ...]\n..."
chess:meta:tactical:count      → 10000 (total games)
chess:meta:positional:count    → 8500
```

**Why:**
- Store **games** not samples (much smaller!)
- PGN string is compact and standard format
- SampleExtractor can extract with any config
- ~400MB for 20K games vs 2.8GB for all samples (7x smaller!)

---

### Step 3: Redis Game Indexing Script
**File:** `scripts/index_games_to_redis.py`

**Functionality:**
```bash
# Index tactical database into Redis
python scripts/index_games_to_redis.py \
    --input chess-federated-learning/data/databases/tactical.pgn.zst \
    --playstyle tactical \
    --min-rating 2000

# Index positional database
python scripts/index_games_to_redis.py \
    --input chess-federated-learning/data/databases/positional.pgn.zst \
    --playstyle positional \
    --min-rating 2000

# Check status
python scripts/index_games_to_redis.py --action stats

# Clear cache
python scripts/index_games_to_redis.py --action clear --playstyle tactical
```

**Process:**
1. Use `GameLoader.load_games()` to iterate through compressed PGN (streaming)
2. For each game, convert `chess.pgn.Game` → PGN string with `str(game)`
3. Store in Redis with key `chess:game:{playstyle}:{index}`
4. Update counter `chess:meta:{playstyle}:count`
5. Progress bar + checkpoint system for resume

**Why:**
- Reuses existing GameLoader filtering logic
- No need for SampleExtractor during indexing!
- Stores compact PGN strings (~2KB per game)
- Streaming approach (low memory)
- One-time preprocessing step

---

### Step 4: Remove Local Pickle Cache Code

**Files to Clean:**

1. **`database_preprocessor.py`**
   - ❌ Remove: `CachedSampleLoader` class (entire class)
   - ❌ Remove: `preprocess_by_playstyle()` method
   - ❌ Remove: All pickle file saving logic
   - ✅ Keep: Only if there's generic extraction logic (but likely not needed)

2. **`trainer_supervised.py`**
   - ❌ Remove: `from data.database_preprocessor import CachedSampleLoader`
   - ❌ Remove: `self.use_cache` flag
   - ❌ Remove: `self.cache_loader` attribute
   - ❌ Remove: Cache detection logic in `__init__`
   - ❌ Remove: Pickle cache loading branch in `_extract_samples()`

3. **Delete obsolete files:**
   - ❌ `chess-federated-learning/data/cache/*.pkl` (old pickle files)
   - ❌ `chess-federated-learning/data/cache/*.json` (progress files)

**Why:**
- Clean slate, no confusion
- Single source of truth (Redis)
- Reduce code complexity

---

### Step 4: Update GameLoader to Support Redis

**File:** `chess-federated-learning/data/game_loader.py`

**Changes in `__init__`:**
```python
def __init__(self, pgn_path: str, use_redis: bool = True):
    self.pgn_path = Path(pgn_path)
    self.eco_classifier = ECOClassifier()
    
    # Try to connect to Redis game cache
    self.redis_cache = None
    if use_redis:
        try:
            from .redis_game_cache import RedisGameCache
            self.redis_cache = RedisGameCache()
            log.info("Redis game cache available")
        except Exception as e:
            log.debug(f"Redis not available: {e}")
```

**Changes in `load_games()`:**
```python
def load_games(self, game_filter: Optional[GameFilter] = None, offset: int = 0):
    # Try Redis first (fast - no decompression!)
    if self.redis_cache and game_filter:
        playstyle = game_filter.playstyle
        if playstyle and self.redis_cache.get_total_games(playstyle) > 0:
            log.info(f"Loading games from Redis cache (playstyle={playstyle})")
            yield from self._load_from_redis(game_filter, offset)
            return
    
    # Fallback: Load from compressed PGN file (slower)
    log.info(f"Loading games from PGN file (no Redis cache)")
    yield from self._load_from_file(game_filter, offset)

def _load_from_redis(self, game_filter, offset):
    # Fetch games from Redis and yield them
    count = game_filter.max_games or 999999
    for game in self.redis_cache.get_games(game_filter.playstyle, offset, count):
        # Apply any additional filters (rating, etc.)
        if self._passes_filter(game, game_filter):
            yield game

def _load_from_file(self, game_filter, offset):
    # Original logic: open PGN file, decompress, iterate
    # (existing code)
```

**Why:**
- **Transparent**: Callers don't know if using Redis or file
- **Automatic fallback**: Works without Redis
- **No trainer changes needed!** SampleExtractor just calls GameLoader.load_games()

---

### Step 6: Redis Cache Management CLI

**File:** `scripts/index_to_redis.py` (extended)

**Commands:**
```bash
# Index databases
python scripts/index_to_redis.py --action index --playstyle tactical --input path/to/db.pgn.zst

# Show statistics
python scripts/index_to_redis.py --action stats
# Output:
# TACTICAL:  89,234 samples (1.2GB in Redis)
# POSITIONAL: 76,891 samples (1.1GB in Redis)
# REDIS:     2.3GB used / 4.0GB max

# Clear cache
python scripts/index_to_redis.py --action clear --playstyle tactical

# Verify integrity
python scripts/index_to_redis.py --action verify --playstyle tactical
# Checks: All indices sequential, no missing samples, deserializes correctly
```

**Why:**
- Easy maintenance
- Debug and monitor cache health
- Convenient CLI for operations

---

### Step 7: Update Preprocessing Scripts

**Files:**
- `scripts/preprocess_tactical_database.sh`
- `scripts/preprocess_positional_database.sh`

**Changes:**
```bash
# OLD (pickle-based):
cd chess-federated-learning && uv run python -m data.database_preprocessor \
    --input ../data/databases/tactical.pgn.zst \
    --output data/cache \
    --min-rating 2000

# NEW (Redis-based):
uv run python scripts/index_to_redis.py \
    --action index \
    --input chess-federated-learning/data/databases/tactical.pgn.zst \
    --playstyle tactical \
    --min-rating 2000
```

**Why:**
- Keep familiar workflow
- Scripts remain entry point for preprocessing
- Just changes backend from pickle to Redis

---

### Step 8: Create Redis Setup Documentation

**File:** `REDIS_SETUP.md`

**Contents:**
1. **Quick Start**
   - `docker-compose up -d redis`
   - `python scripts/index_to_redis.py --action index --playstyle all`
   - Start training

2. **Architecture Diagram**
   - Visual flow of data

3. **Memory Requirements**
   - Redis: ~2.8GB
   - Per node: ~50MB
   - Total system: ~3.2GB (vs 11.2GB before)

4. **Indexing Workflow**
   - Step-by-step guide
   - Time estimates
   - Troubleshooting

5. **Performance Benchmarks**
   - Pickle vs Redis comparison
   - Access speed tests

6. **Maintenance**
   - Backup/restore
   - Clearing cache
   - Monitoring

**Why:**
- Complete reference for users
- Reduces support questions
- Clear migration path

---

### Step 9: Add Redis Health Check to Startup

**File:** `chess-federated-learning/server/main.py` (or startup script)

**Add at startup:**
```python
def check_redis_cache():
    """Check if Redis cache is available and populated."""
    try:
        from data.redis_cache import RedisCache
        cache = RedisCache()
        stats = cache.get_stats()
        
        if stats.get('tactical', {}).get('count', 0) > 0:
            logger.success("✓ Redis cache available (tactical samples loaded)")
        else:
            logger.warning("⚠ Redis available but no tactical samples")
            logger.info("  Run: python scripts/index_to_redis.py --action index")
        
        if stats.get('positional', {}).get('count', 0) > 0:
            logger.success("✓ Redis cache available (positional samples loaded)")
        else:
            logger.warning("⚠ Redis available but no positional samples")
            logger.info("  Run: python scripts/index_to_redis.py --action index")
            
    except Exception as e:
        logger.warning(f"⚠ Redis cache not available: {e}")
        logger.info("  Nodes will use slower direct extraction")
        logger.info("  To enable cache: docker-compose up -d redis")

# In main():
check_redis_cache()
```

**Why:**
- Early detection of cache issues
- Helpful messages guide user
- System still works without Redis

---

## Migration Workflow

### For New Users:
```bash
# 1. Start Redis
docker-compose up -d redis

# 2. Index databases
./scripts/preprocess_tactical_database.sh
./scripts/preprocess_positional_database.sh

# 3. Start training
uv run python chess-federated-learning/server/main.py &
uv run python chess-federated-learning/scripts/start_all_nodes.py \
    --config-dir chess-federated-learning/config/nodes
```

### For Existing Users (with pickle cache):
```bash
# 1. Start Redis
docker-compose up -d redis

# 2. Index from existing databases (pickle files ignored)
./scripts/preprocess_tactical_database.sh
./scripts/preprocess_positional_database.sh

# 3. (Optional) Delete old pickle files
rm -rf chess-federated-learning/data/cache/*.pkl

# 4. Start training (automatically uses Redis)
# ... same as before ...
```

---

## Testing Plan

### Unit Tests
- `test_redis_cache.py`: Test RedisCache class methods
- `test_index_to_redis.py`: Test indexing script

### Integration Tests
- Index small PGN → verify Redis contents
- Train with Redis cache → verify samples correct
- Test fallback when Redis unavailable

### Performance Tests
- Measure indexing speed (games/sec)
- Measure retrieval speed (samples/sec)
- Memory usage per node (should be ~50MB)

---

## Rollback Plan

If Redis causes issues:

1. **Keep SampleExtractor fallback** in trainer
   - System works without Redis (just slower)
   - No training disruption

2. **Don't delete pickle preprocessing code immediately**
   - Mark as deprecated but keep for one release
   - Allows reverting if needed

3. **Document old pickle workflow**
   - In case emergency rollback needed

---

## Expected Benefits

| Metric | Before (File) | After (Redis Games) | Improvement |
|--------|---------------|---------------------|-------------|
| Cache Size | N/A (decompress each time) | ~400MB (games only) | **Persistent cache** |
| Extraction Time | 30s-3min per round | <5s per round | **36x-600x faster** |
| Redis Memory | N/A | ~400MB total | **Very efficient** |
| Per-Node Memory | ~50MB | ~50MB | **Same** |
| Flexibility | N/A | Can change extraction config | **Better** |
| Setup Complexity | None | Medium | Trade-off |
| Multi-node Support | All decompress | Shared cache | **Much better** |

**Key Insight:** We cache **games** (small) not **samples** (huge), giving us speed without memory bloat!

---

## Questions to Answer Before Starting

1. **Redis Persistence:** AOF, RDB, or both?
   - **Recommendation:** Both (RDB every 60s, AOF everysec)

2. **Game Storage Format:** PGN string or binary?
   - **Recommendation:** PGN string (standard, readable, ~2KB per game)

3. **Batch Size:** How many games to fetch per Redis call?
   - **Recommendation:** 100 games per pipeline call

4. **Indexing Time:** How long to index full databases?
   - **Estimate:** ~5-10 min for 10K games (just storing PGN strings, fast!)

5. **Redis Version:** Which Docker image?
   - **Recommendation:** redis:7-alpine (latest stable, small image)

6. **Memory estimate:** How much Redis memory for games?
   - **Calculation:** 20,000 games × 2KB average = ~40MB (plus overhead = ~400MB safe)

---

## Approval Checklist

- [ ] Docker Compose configuration looks good
- [ ] RedisCache class interface makes sense
- [ ] Indexing script workflow is clear
- [ ] Okay to delete all pickle cache code
- [ ] Trainer fallback logic is acceptable
- [ ] CLI commands are intuitive
- [ ] Documentation plan is sufficient
- [ ] Health check approach works
- [ ] Migration workflow is smooth

**Ready to proceed?** Please review and approve! 🚀
