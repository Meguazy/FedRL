"""
Redis-based puzzle cache for fast access to Lichess puzzles.

This module provides a Redis-backed cache system that stores puzzles
as JSON objects, allowing multiple training nodes to access puzzles without
repeatedly decompressing the large puzzle database file.

Architecture:
- Puzzles stored as JSON in Redis
- Each puzzle has a key: chess:puzzle:{index}
- Metadata tracks total puzzle count
- All nodes share the same Redis instance

Benefits:
- No decompression overhead during training
- Shared cache across all nodes
- Fast random access by index
- Efficient offset-based sampling for nodes

Usage:
    # Index puzzles into Redis (one time)
    cache = RedisPuzzleCache(port=6381)
    for i, puzzle_data in enumerate(puzzles):
        cache.store_puzzle(i, puzzle_data)
    
    # Retrieve puzzles during training
    cache = RedisPuzzleCache(port=6381)
    puzzles = cache.get_puzzles(offset=1000, count=500)
    for puzzle in puzzles:
        # Use puzzle for training
        ...
"""

import redis
import json
import random
from typing import List, Dict, Optional, Any
from loguru import logger


class RedisPuzzleCache:
    """Cache Lichess puzzles in Redis as JSON objects."""
    
    # Redis key prefixes
    PUZZLE_KEY_PREFIX = "chess:puzzle"
    META_KEY_PREFIX = "chess:meta:puzzle"
    
    def __init__(self, host: str = "localhost", port: int = 6381, db: int = 0):
        """
        Initialize Redis puzzle cache.
        
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
    
    def _get_puzzle_key(self, index: int) -> str:
        """Get Redis key for a specific puzzle."""
        return f"{self.PUZZLE_KEY_PREFIX}:{index}"
    
    def _get_count_key(self) -> str:
        """Get Redis key for puzzle count."""
        return f"{self.META_KEY_PREFIX}:count"
    
    def store_puzzle(self, index: int, puzzle_data: Dict[str, Any]) -> bool:
        """
        Store a puzzle as JSON in Redis.
        
        Args:
            index: Puzzle index (0-based)
            puzzle_data: Dictionary containing puzzle data
                Required keys: puzzle_id, fen, moves, rating, themes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._get_puzzle_key(index)
            json_data = json.dumps(puzzle_data)
            
            # Store puzzle
            self.redis_client.set(key, json_data)
            
            # Update count
            self.redis_client.set(self._get_count_key(), index + 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store puzzle {index}: {e}")
            return False
    
    def store_puzzles_batch(self, puzzles: List[tuple]) -> int:
        """
        Store multiple puzzles in a single pipeline for efficiency.
        
        Args:
            puzzles: List of (index, puzzle_data) tuples
            
        Returns:
            Number of puzzles successfully stored
        """
        if not puzzles:
            return 0
        
        try:
            pipe = self.redis_client.pipeline()
            
            for index, puzzle_data in puzzles:
                key = self._get_puzzle_key(index)
                json_data = json.dumps(puzzle_data)
                pipe.set(key, json_data)
            
            # Update count to highest index + 1
            max_index = max(idx for idx, _ in puzzles)
            pipe.set(self._get_count_key(), max_index + 1)
            
            pipe.execute()
            return len(puzzles)
            
        except Exception as e:
            logger.error(f"Failed to store puzzle batch: {e}")
            return 0
    
    def get_puzzle(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a puzzle by index.
        
        Args:
            index: Puzzle index (0-based)
            
        Returns:
            Puzzle data dictionary or None if not found
        """
        try:
            key = self._get_puzzle_key(index)
            json_data = self.redis_client.get(key)
            
            if json_data is None:
                return None
            
            return json.loads(json_data)
            
        except Exception as e:
            logger.error(f"Failed to get puzzle {index}: {e}")
            return None
    
    def get_puzzles(self, offset: int = 0, count: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve multiple puzzles starting from offset.
        
        Args:
            offset: Starting index (0-based)
            count: Number of puzzles to retrieve
            
        Returns:
            List of puzzle data dictionaries
        """
        puzzles = []
        
        try:
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for i in range(offset, offset + count):
                key = self._get_puzzle_key(i)
                pipe.get(key)
            
            results = pipe.execute()
            
            for json_data in results:
                if json_data is not None:
                    puzzles.append(json.loads(json_data))
            
            return puzzles
            
        except Exception as e:
            logger.error(f"Failed to get puzzles batch: {e}")
            return []
    
    def get_total_puzzles(self) -> int:
        """
        Get total number of puzzles in cache.
        
        Returns:
            Total puzzle count
        """
        try:
            count = self.redis_client.get(self._get_count_key())
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Failed to get puzzle count: {e}")
            return 0
    
    def clear_cache(self) -> int:
        """
        Clear all puzzles from Redis.
        
        Returns:
            Number of keys deleted
        """
        try:
            # Get all puzzle keys
            pattern = f"{self.PUZZLE_KEY_PREFIX}:*"
            keys = list(self.redis_client.scan_iter(match=pattern, count=1000))
            
            # Add metadata key
            keys.append(self._get_count_key())
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Deleted {deleted} puzzle keys from Redis")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear puzzle cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_puzzles = self.get_total_puzzles()
            
            # Get Redis info
            redis_info = self.redis_client.info('memory')
            
            return {
                'total_puzzles': total_puzzles,
                'redis': {
                    'used_memory_human': redis_info.get('used_memory_human', 'N/A'),
                    'used_memory_peak_human': redis_info.get('used_memory_peak_human', 'N/A'),
                    'maxmemory_human': redis_info.get('maxmemory_human', 'unlimited')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def verify_integrity(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Verify cache integrity by sampling puzzles.
        
        Args:
            sample_size: Number of puzzles to sample
            
        Returns:
            Dictionary with verification results
        """
        try:
            total_puzzles = self.get_total_puzzles()
            
            if total_puzzles == 0:
                return {
                    'success': True,
                    'total_puzzles': 0,
                    'sampled': 0,
                    'valid': 0,
                    'invalid': 0,
                    'missing': 0,
                    'integrity': '100%'
                }
            
            # Sample random indices
            sample_size = min(sample_size, total_puzzles)
            indices = random.sample(range(total_puzzles), sample_size)
            
            valid = 0
            invalid = 0
            missing = 0
            
            for idx in indices:
                puzzle = self.get_puzzle(idx)
                
                if puzzle is None:
                    missing += 1
                elif self._validate_puzzle(puzzle):
                    valid += 1
                else:
                    invalid += 1
            
            integrity = f"{(valid / sample_size * 100):.1f}%" if sample_size > 0 else "N/A"
            
            return {
                'success': True,
                'total_puzzles': total_puzzles,
                'sampled': sample_size,
                'valid': valid,
                'invalid': invalid,
                'missing': missing,
                'integrity': integrity
            }
            
        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_puzzle(self, puzzle: Dict[str, Any]) -> bool:
        """Validate puzzle data structure."""
        required_keys = ['puzzle_id', 'fen', 'moves', 'rating', 'themes']
        
        # Check all required keys present
        if not all(k in puzzle for k in required_keys):
            return False
        
        # Check data types
        if not isinstance(puzzle['moves'], list) or len(puzzle['moves']) == 0:
            return False
        if not isinstance(puzzle['rating'], int):
            return False
        if not isinstance(puzzle['themes'], list):
            return False
        
        return True
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.debug("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
