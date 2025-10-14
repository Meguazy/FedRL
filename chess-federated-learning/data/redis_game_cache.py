"""
Redis-based game cache for fast access to chess games.

This module provides a Redis-backed cache system that stores chess games
as PGN strings, allowing multiple training nodes to access games without
repeatedly decompressing large database files.

Architecture:
- Games stored as PGN strings in Redis
- Each game has a key: chess:game:{playstyle}:{index}
- Metadata tracks total game count per playstyle
- All nodes share the same Redis instance

Benefits:
- No decompression overhead during training
- Shared cache across all nodes (~400MB for 20K games)
- Much smaller than caching extracted samples
- Flexible: Can change extraction config without reindexing

Usage:
    # Index games into Redis (one time)
    cache = RedisGameCache(port=6381)
    for i, game in enumerate(games):
        pgn_string = str(game)
        cache.store_game('tactical', i, pgn_string)
    
    # Retrieve games during training
    cache = RedisGameCache(port=6381)
    for game in cache.get_games('tactical', offset=100, count=50):
        # Use game for training
        samples = extractor.extract_samples_from_game(game)
"""

import redis
import chess.pgn
import io
from typing import Iterator, Optional, Dict
from loguru import logger


class RedisGameCache:
    """Cache chess games in Redis as PGN strings."""
    
    # Redis key prefixes
    GAME_KEY_PREFIX = "chess:game"
    META_KEY_PREFIX = "chess:meta"
    
    def __init__(self, host: str = "localhost", port: int = 6381, db: int = 0):
        """
        Initialize Redis game cache.
        
        Args:
            host: Redis server host (default: localhost)
            port: Redis server port (default: 6381 - chess Redis instance)
            db: Redis database number (default: 0)
        """
        self.host = host
        self.port = port
        self.db = db
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,  # Return strings not bytes
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.debug(f"Connected to Redis at {host}:{port}")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
            logger.error("Make sure Redis is running: docker-compose up -d redis")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise
    
    def _get_game_key(self, playstyle: str, index: int) -> str:
        """Get Redis key for a specific game."""
        return f"{self.GAME_KEY_PREFIX}:{playstyle}:{index}"
    
    def _get_count_key(self, playstyle: str) -> str:
        """Get Redis key for game count."""
        return f"{self.META_KEY_PREFIX}:{playstyle}:count"
    
    def store_game(self, playstyle: str, index: int, pgn_string: str) -> bool:
        """
        Store a game as PGN string in Redis.
        
        Args:
            playstyle: 'tactical' or 'positional'
            index: Game index (0-based)
            pgn_string: PGN representation of the game
            
        Returns:
            True if stored successfully
        """
        try:
            key = self._get_game_key(playstyle, index)
            self.redis_client.set(key, pgn_string)
            
            # Update count if this is a new game
            count_key = self._get_count_key(playstyle)
            current_count = self.redis_client.get(count_key)
            
            if current_count is None or int(current_count) <= index:
                self.redis_client.set(count_key, index + 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store game {playstyle}:{index}: {e}")
            return False
    
    def store_games_batch(self, playstyle: str, games: list[tuple[int, str]], batch_size: int = 1000) -> int:
        """
        Store multiple games in batches for efficiency.
        
        Args:
            playstyle: 'tactical' or 'positional'
            games: List of (index, pgn_string) tuples
            batch_size: Number of games per pipeline batch
            
        Returns:
            Number of games stored successfully
        """
        stored = 0
        total = len(games)
        
        try:
            for i in range(0, total, batch_size):
                batch = games[i:i + batch_size]
                pipeline = self.redis_client.pipeline()
                
                for index, pgn_string in batch:
                    key = self._get_game_key(playstyle, index)
                    pipeline.set(key, pgn_string)
                
                pipeline.execute()
                stored += len(batch)
                
                if stored % 5000 == 0 or stored == total:
                    logger.info(f"Stored {stored}/{total} {playstyle} games in Redis")
            
            # Update total count
            max_index = max(idx for idx, _ in games)
            count_key = self._get_count_key(playstyle)
            self.redis_client.set(count_key, max_index + 1)
            
            logger.success(f"Successfully stored {stored} {playstyle} games")
            return stored
            
        except Exception as e:
            logger.error(f"Failed to store games batch: {e}")
            return stored
    
    def get_game(self, playstyle: str, index: int) -> Optional[chess.pgn.Game]:
        """
        Retrieve a single game from Redis.
        
        Args:
            playstyle: 'tactical' or 'positional'
            index: Game index
            
        Returns:
            chess.pgn.Game object or None if not found
        """
        try:
            key = self._get_game_key(playstyle, index)
            pgn_string = self.redis_client.get(key)
            
            if pgn_string is None:
                return None
            
            # Parse PGN string back to chess.pgn.Game
            game = chess.pgn.read_game(io.StringIO(pgn_string))
            return game
            
        except Exception as e:
            logger.error(f"Failed to retrieve game {playstyle}:{index}: {e}")
            return None
    
    def get_games(self, playstyle: str, offset: int = 0, count: int = 100) -> Iterator[chess.pgn.Game]:
        """
        Retrieve multiple games from Redis.
        
        This method fetches games in batches using Redis pipelines
        for efficiency, then yields them as chess.pgn.Game objects.
        
        Args:
            playstyle: 'tactical' or 'positional'
            offset: Starting game index
            count: Number of games to retrieve
            
        Yields:
            chess.pgn.Game objects
        """
        try:
            total_games = self.get_total_games(playstyle)
            
            if total_games == 0:
                logger.warning(f"No {playstyle} games in Redis cache")
                return
            
            # Clamp count to available games
            end_index = min(offset + count, total_games)
            actual_count = end_index - offset
            
            if actual_count <= 0:
                logger.warning(f"Offset {offset} beyond available games ({total_games})")
                return
            
            logger.debug(f"Fetching {actual_count} {playstyle} games from Redis (offset={offset})")
            
            # Fetch in batches for efficiency
            batch_size = 100
            for batch_start in range(offset, end_index, batch_size):
                batch_end = min(batch_start + batch_size, end_index)
                
                # Use pipeline to fetch batch
                pipeline = self.redis_client.pipeline()
                for i in range(batch_start, batch_end):
                    key = self._get_game_key(playstyle, i)
                    pipeline.get(key)
                
                pgn_strings = pipeline.execute()
                
                # Parse and yield games
                for pgn_string in pgn_strings:
                    if pgn_string:
                        try:
                            game = chess.pgn.read_game(io.StringIO(pgn_string))
                            if game:
                                yield game
                        except Exception as e:
                            logger.warning(f"Failed to parse game: {e}")
                            continue
            
            logger.debug(f"Finished fetching {playstyle} games from Redis")
            
        except Exception as e:
            logger.error(f"Failed to retrieve games: {e}")
            return
    
    def get_total_games(self, playstyle: str) -> int:
        """
        Get total number of games stored for a playstyle.
        
        Args:
            playstyle: 'tactical' or 'positional'
            
        Returns:
            Number of games (0 if none stored)
        """
        try:
            count_key = self._get_count_key(playstyle)
            count = self.redis_client.get(count_key)
            return int(count) if count else 0
            
        except Exception as e:
            logger.error(f"Failed to get game count: {e}")
            return 0
    
    def clear_cache(self, playstyle: Optional[str] = None) -> int:
        """
        Clear cached games from Redis.
        
        Args:
            playstyle: Specific playstyle to clear, or None for all
            
        Returns:
            Number of keys deleted
        """
        try:
            if playstyle:
                pattern = f"{self.GAME_KEY_PREFIX}:{playstyle}:*"
                logger.info(f"Clearing {playstyle} games from Redis...")
            else:
                pattern = f"{self.GAME_KEY_PREFIX}:*"
                logger.info("Clearing all games from Redis...")
            
            deleted = 0
            cursor = 0
            
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor, 
                    match=pattern, 
                    count=1000
                )
                
                if keys:
                    deleted += self.redis_client.delete(*keys)
                
                if cursor == 0:
                    break
            
            # Clear metadata
            if playstyle:
                meta_key = self._get_count_key(playstyle)
                self.redis_client.delete(meta_key)
            else:
                for ps in ['tactical', 'positional']:
                    meta_key = self._get_count_key(ps)
                    self.redis_client.delete(meta_key)
            
            logger.success(f"Deleted {deleted} keys from Redis")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            stats = {
                'tactical': {
                    'count': self.get_total_games('tactical')
                },
                'positional': {
                    'count': self.get_total_games('positional')
                }
            }
            
            # Get Redis memory info
            info = self.redis_client.info('memory')
            stats['redis'] = {
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'used_memory_peak_human': info.get('used_memory_peak_human', 'N/A'),
                'maxmemory_human': info.get('maxmemory_human', 'N/A'),
            }
            
            # Get total games
            total = stats['tactical']['count'] + stats['positional']['count']
            stats['total_games'] = total
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def verify_integrity(self, playstyle: str, sample_size: int = 100) -> Dict:
        """
        Verify cache integrity by sampling games.
        
        Args:
            playstyle: 'tactical' or 'positional'
            sample_size: Number of games to check
            
        Returns:
            Dictionary with verification results
        """
        try:
            total = self.get_total_games(playstyle)
            
            if total == 0:
                return {
                    'success': False,
                    'error': 'No games in cache'
                }
            
            # Sample indices
            import random
            sample_size = min(sample_size, total)
            indices = random.sample(range(total), sample_size)
            
            valid = 0
            invalid = 0
            missing = 0
            
            for i in indices:
                game = self.get_game(playstyle, i)
                
                if game is None:
                    missing += 1
                elif game.end().board().is_valid():
                    valid += 1
                else:
                    invalid += 1
            
            return {
                'success': True,
                'playstyle': playstyle,
                'total_games': total,
                'sampled': sample_size,
                'valid': valid,
                'invalid': invalid,
                'missing': missing,
                'integrity': f"{(valid/sample_size)*100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.debug("Closed Redis connection")
        except:
            pass
