#!/usr/bin/env python3
"""
Index chess games from PGN databases into Redis cache.

This script reads compressed PGN databases, filters games by rating and playstyle,
and stores them in Redis as PGN strings for fast access during training.

Usage:
    # Index tactical games
    python scripts/index_games_to_redis.py \\
        --input chess-federated-learning/data/databases/tactical.pgn.zst \\
        --playstyle tactical \\
        --min-rating 2000

    # Index positional games
    python scripts/index_games_to_redis.py \\
        --input chess-federated-learning/data/databases/positional.pgn.zst \\
        --playstyle positional \\
        --min-rating 2000

    # Check cache stats
    python scripts/index_games_to_redis.py --action stats

    # Clear cache
    python scripts/index_games_to_redis.py --action clear --playstyle tactical

    # Verify cache integrity
    python scripts/index_games_to_redis.py --action verify --playstyle tactical
"""

import sys
import argparse
import chess.pgn
from pathlib import Path
from loguru import logger

# Add parent directory to path to import from chess-federated-learning
sys.path.insert(0, str(Path(__file__).parent.parent / "chess-federated-learning"))

from data.redis_game_cache import RedisGameCache
from data.game_loader import GameLoader, GameFilter


def index_games(
    input_path: str,
    playstyle: str,
    min_rating: int,
    max_games: int = None,
    redis_host: str = "localhost",
    redis_port: int = 6381,
    batch_size: int = 1000
):
    """
    Index games from PGN file into Redis.
    
    Args:
        input_path: Path to PGN database file
        playstyle: 'tactical' or 'positional'
        min_rating: Minimum player rating
        max_games: Maximum games to index (None = all)
        redis_host: Redis server host
        redis_port: Redis server port
        batch_size: Batch size for storing
    """
    logger.info("="*70)
    logger.info("CHESS GAME INDEXING TO REDIS")
    logger.info("="*70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Playstyle: {playstyle}")
    logger.info(f"Min Rating: {min_rating}")
    logger.info(f"Max Games: {max_games or 'unlimited'}")
    logger.info(f"Redis: {redis_host}:{redis_port}")
    logger.info("="*70)
    
    # Initialize Redis cache
    try:
        cache = RedisGameCache(host=redis_host, port=redis_port)
        logger.success("✓ Connected to Redis")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Redis: {e}")
        logger.error("Make sure Redis is running: docker-compose up -d redis")
        return False
    
    # Check if games already exist
    existing_count = cache.get_total_games(playstyle)
    if existing_count > 0:
        logger.warning(f"Redis already contains {existing_count} {playstyle} games")
        response = input("Clear existing cache and reindex? (y/n): ").strip().lower()
        if response == 'y':
            logger.info("Clearing existing cache...")
            cache.clear_cache(playstyle)
            logger.success("Cache cleared")
        else:
            logger.info("Keeping existing cache, appending new games...")
            # Note: This will start indexing from existing_count
    
    # Initialize game loader (disable Redis to force file reading)
    try:
        loader = GameLoader(input_path, use_redis=False)
        logger.success(f"✓ Opened PGN database: {input_path}")
    except Exception as e:
        logger.error(f"✗ Failed to open database: {e}")
        return False
    
    # Set up game filter
    game_filter = GameFilter(
        min_rating=min_rating,
        playstyle=playstyle,
        max_games=max_games
    )
    
    # Stream games and store in batches
    logger.info("="*70)
    logger.info("STREAMING GAMES FROM DATABASE")
    logger.info("This may take a while for large databases...")
    logger.info("Progress will be logged every 1000 games")
    logger.info("="*70)
    
    games_batch = []
    total_indexed = 0
    total_checked = 0
    
    try:
        for game in loader.load_games(game_filter, offset=0):
            total_checked += 1
            
            # Convert game to PGN string
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            pgn_string = game.accept(exporter)
            
            # Add to batch
            game_index = existing_count + total_indexed
            games_batch.append((game_index, pgn_string))
            total_indexed += 1
            
            # Store batch when full
            if len(games_batch) >= batch_size:
                stored = cache.store_games_batch(playstyle, games_batch, batch_size=batch_size)
                
                if stored != len(games_batch):
                    logger.warning(f"Only stored {stored}/{len(games_batch)} games in batch")
                
                games_batch = []
                
                # Log progress
                logger.info(f"Progress: {total_indexed:,} games indexed ({total_checked:,} checked)")
            
            # Check if reached max_games limit
            if max_games and total_indexed >= max_games:
                logger.info(f"Reached max_games limit: {max_games}")
                break
        
        # Store remaining games in final batch
        if games_batch:
            stored = cache.store_games_batch(playstyle, games_batch, batch_size=len(games_batch))
            if stored != len(games_batch):
                logger.warning(f"Only stored {stored}/{len(games_batch)} games in final batch")
        
        logger.info("="*70)
        logger.success("INDEXING COMPLETE")
        logger.info(f"Games checked: {total_checked:,}")
        logger.info(f"Games indexed: {total_indexed:,}")
        logger.info(f"Total in Redis: {cache.get_total_games(playstyle):,}")
        logger.info("="*70)
        
        # Show Redis stats
        stats = cache.get_stats()
        logger.info("\nRedis Cache Stats:")
        logger.info(f"  Tactical games: {stats.get('tactical', {}).get('count', 0):,}")
        logger.info(f"  Positional games: {stats.get('positional', {}).get('count', 0):,}")
        logger.info(f"  Total games: {stats.get('total_games', 0):,}")
        logger.info(f"  Memory used: {stats.get('redis', {}).get('used_memory_human', 'N/A')}")
        logger.info(f"  Memory limit: {stats.get('redis', {}).get('maxmemory_human', 'N/A')}")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n\nIndexing interrupted by user!")
        logger.info(f"Games indexed so far: {total_indexed:,}")
        logger.info("You can resume indexing or use what's already cached")
        return False
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        cache.close()


def show_stats(redis_host: str = "localhost", redis_port: int = 6381):
    """Show Redis cache statistics."""
    try:
        cache = RedisGameCache(host=redis_host, port=redis_port)
        stats = cache.get_stats()
        
        print("\n" + "="*70)
        print("REDIS CACHE STATISTICS")
        print("="*70)
        print(f"\nTACTICAL GAMES:  {stats.get('tactical', {}).get('count', 0):,}")
        print(f"POSITIONAL GAMES: {stats.get('positional', {}).get('count', 0):,}")
        print(f"TOTAL GAMES:      {stats.get('total_games', 0):,}")
        print(f"\nREDIS MEMORY:")
        print(f"  Used:  {stats.get('redis', {}).get('used_memory_human', 'N/A')}")
        print(f"  Peak:  {stats.get('redis', {}).get('used_memory_peak_human', 'N/A')}")
        print(f"  Limit: {stats.get('redis', {}).get('maxmemory_human', 'N/A')}")
        print("="*70 + "\n")
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return False


def clear_cache(playstyle: str = None, redis_host: str = "localhost", redis_port: int = 6381):
    """Clear Redis cache."""
    try:
        cache = RedisGameCache(host=redis_host, port=redis_port)
        
        if playstyle:
            logger.info(f"Clearing {playstyle} games from Redis...")
        else:
            logger.info("Clearing ALL games from Redis...")
        
        response = input("Are you sure? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Cancelled")
            return False
        
        deleted = cache.clear_cache(playstyle)
        logger.success(f"Deleted {deleted:,} keys from Redis")
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def verify_cache(playstyle: str, sample_size: int = 100, redis_host: str = "localhost", redis_port: int = 6381):
    """Verify cache integrity."""
    try:
        cache = RedisGameCache(host=redis_host, port=redis_port)
        
        logger.info(f"Verifying {playstyle} cache integrity...")
        logger.info(f"Sampling {sample_size} games...")
        
        result = cache.verify_integrity(playstyle, sample_size)
        
        if not result.get('success'):
            logger.error(f"Verification failed: {result.get('error', 'Unknown error')}")
            return False
        
        print("\n" + "="*70)
        print("CACHE INTEGRITY CHECK")
        print("="*70)
        print(f"Playstyle:    {result['playstyle']}")
        print(f"Total games:  {result['total_games']:,}")
        print(f"Sampled:      {result['sampled']}")
        print(f"Valid:        {result['valid']}")
        print(f"Invalid:      {result['invalid']}")
        print(f"Missing:      {result['missing']}")
        print(f"Integrity:    {result['integrity']}")
        print("="*70 + "\n")
        
        if result['missing'] > 0:
            logger.warning(f"Found {result['missing']} missing games!")
        if result['invalid'] > 0:
            logger.warning(f"Found {result['invalid']} invalid games!")
        
        if result['valid'] == result['sampled']:
            logger.success("✓ Cache integrity verified - all sampled games valid")
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify cache: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index chess games from PGN into Redis cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index tactical games
  python scripts/index_games_to_redis.py \\
      --input data/databases/tactical.pgn.zst \\
      --playstyle tactical \\
      --min-rating 2000

  # Show cache statistics
  python scripts/index_games_to_redis.py --action stats

  # Clear tactical cache
  python scripts/index_games_to_redis.py --action clear --playstyle tactical

  # Verify cache integrity
  python scripts/index_games_to_redis.py --action verify --playstyle tactical
        """
    )
    
    parser.add_argument(
        '--action',
        choices=['index', 'stats', 'clear', 'verify'],
        default='index',
        help='Action to perform (default: index)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input PGN database (required for index action)'
    )
    parser.add_argument(
        '--playstyle',
        choices=['tactical', 'positional'],
        help='Playstyle to index (required for index, clear, verify actions)'
    )
    parser.add_argument(
        '--min-rating',
        type=int,
        default=2000,
        help='Minimum player rating (default: 2000)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        help='Maximum games to index (default: unlimited)'
    )
    parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    parser.add_argument(
        '--redis-port',
        type=int,
        default=6381,
        help='Redis port (default: 6381)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for storing games (default: 1000)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Sample size for verification (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on action
    if args.action == 'index':
        if not args.input:
            parser.error("--input is required for index action")
        if not args.playstyle:
            parser.error("--playstyle is required for index action")
        
        success = index_games(
            input_path=args.input,
            playstyle=args.playstyle,
            min_rating=args.min_rating,
            max_games=args.max_games,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            batch_size=args.batch_size
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'stats':
        success = show_stats(
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'clear':
        success = clear_cache(
            playstyle=args.playstyle,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'verify':
        if not args.playstyle:
            parser.error("--playstyle is required for verify action")
        
        success = verify_cache(
            playstyle=args.playstyle,
            sample_size=args.sample_size,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
