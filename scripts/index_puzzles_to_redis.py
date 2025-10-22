#!/usr/bin/env python3
"""
Index Lichess puzzles into Redis cache.

This script reads the compressed Lichess puzzle database and stores puzzles
in Redis for fast access during training.

Usage:
    # Index all puzzles
    python scripts/index_puzzles_to_redis.py \
        --input chess-federated-learning/data/databases/lichess_puzzles.csv.zst

    # Check cache stats
    python scripts/index_puzzles_to_redis.py --action stats

    # Clear cache
    python scripts/index_puzzles_to_redis.py --action clear

    # Verify cache integrity
    python scripts/index_puzzles_to_redis.py --action verify
"""

import sys
import argparse
import csv
import json
import subprocess
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "chess-federated-learning"))

from data.redis_puzzle_cache import RedisPuzzleCache


def index_puzzles(
    input_path: str,
    min_rating: int = 0,
    max_rating: int = 5000,
    max_puzzles: int = None,
    redis_host: str = "localhost",
    redis_port: int = 6381,
    batch_size: int = 1000
):
    """
    Index puzzles from CSV file into Redis.
    
    Args:
        input_path: Path to puzzle CSV file (.csv.zst)
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
        max_puzzles: Maximum puzzles to index (None = all)
        redis_host: Redis server host
        redis_port: Redis server port
        batch_size: Batch size for storing
    """
    logger.info("="*70)
    logger.info("LICHESS PUZZLE INDEXING TO REDIS")
    logger.info("="*70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Rating range: {min_rating}-{max_rating}")
    logger.info(f"Max Puzzles: {max_puzzles or 'unlimited'}")
    logger.info(f"Redis: {redis_host}:{redis_port}")
    logger.info("="*70)
    
    # Initialize Redis cache
    try:
        cache = RedisPuzzleCache(host=redis_host, port=redis_port)
        logger.success("✓ Connected to Redis")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Redis: {e}")
        logger.error("Make sure Redis is running: docker-compose up -d redis")
        return False
    
    # Check if puzzles already exist
    existing_count = cache.get_total_puzzles()
    if existing_count > 0:
        logger.warning(f"Redis already contains {existing_count:,} puzzles")
        response = input("Clear existing cache and reindex? (y/n): ").strip().lower()
        if response == 'y':
            logger.info("Clearing existing cache...")
            cache.clear_cache()
            logger.success("Cache cleared")
        else:
            logger.info("Aborting to avoid duplicates")
            return False
    
    # Open puzzle database
    path = Path(input_path)
    if not path.exists():
        logger.error(f"File not found: {input_path}")
        return False
    
    logger.info("="*70)
    logger.info("STREAMING PUZZLES FROM DATABASE")
    logger.info("This may take a few minutes for 5.4M puzzles...")
    logger.info("Progress will be logged every 10,000 puzzles")
    logger.info("="*70)
    
    puzzles_batch = []
    total_indexed = 0
    total_checked = 0
    
    try:
        # Open compressed file
        if path.suffix == '.zst':
            cmd = ['unzstd', '-c', str(path)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            reader = csv.DictReader(proc.stdout)
        else:
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
        
        for row in reader:
            total_checked += 1
            
            try:
                rating = int(row['Rating'])
                
                # Filter by rating
                if rating < min_rating or rating > max_rating:
                    continue
                
                # Create puzzle dict
                puzzle_data = {
                    'puzzle_id': row['PuzzleId'],
                    'fen': row['FEN'],
                    'moves': row['Moves'].split(),
                    'rating': rating,
                    'themes': row['Themes'].split()
                }
                
                # Add to batch
                puzzles_batch.append((total_indexed, puzzle_data))
                total_indexed += 1
                
                # Store batch when full
                if len(puzzles_batch) >= batch_size:
                    stored = cache.store_puzzles_batch(puzzles_batch)
                    
                    if stored != len(puzzles_batch):
                        logger.warning(f"Only stored {stored}/{len(puzzles_batch)} puzzles in batch")
                    
                    puzzles_batch = []
                    
                    # Log progress every 10k puzzles
                    if total_indexed % 10000 == 0:
                        logger.info(f"Progress: {total_indexed:,} puzzles indexed ({total_checked:,} checked)")
                
                # Check if reached max_puzzles limit
                if max_puzzles and total_indexed >= max_puzzles:
                    logger.info(f"Reached max_puzzles limit: {max_puzzles:,}")
                    break
                    
            except (ValueError, KeyError) as e:
                # Skip malformed puzzles
                continue
        
        # Store remaining puzzles in final batch
        if puzzles_batch:
            stored = cache.store_puzzles_batch(puzzles_batch)
            if stored != len(puzzles_batch):
                logger.warning(f"Only stored {stored}/{len(puzzles_batch)} puzzles in final batch")
        
        logger.info("="*70)
        logger.success("INDEXING COMPLETE")
        logger.info(f"Puzzles checked: {total_checked:,}")
        logger.info(f"Puzzles indexed: {total_indexed:,}")
        logger.info(f"Total in Redis: {cache.get_total_puzzles():,}")
        logger.info("="*70)
        
        # Show Redis stats
        stats = cache.get_stats()
        logger.info("\nRedis Cache Stats:")
        logger.info(f"  Total puzzles: {stats.get('total_puzzles', 0):,}")
        logger.info(f"  Memory used: {stats.get('redis', {}).get('used_memory_human', 'N/A')}")
        logger.info(f"  Memory limit: {stats.get('redis', {}).get('maxmemory_human', 'N/A')}")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n\nIndexing interrupted by user!")
        logger.info(f"Puzzles indexed so far: {total_indexed:,}")
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
        cache = RedisPuzzleCache(host=redis_host, port=redis_port)
        stats = cache.get_stats()
        
        print("\n" + "="*70)
        print("REDIS PUZZLE CACHE STATISTICS")
        print("="*70)
        print(f"\nTOTAL PUZZLES: {stats.get('total_puzzles', 0):,}")
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


def clear_cache(redis_host: str = "localhost", redis_port: int = 6381):
    """Clear Redis puzzle cache."""
    try:
        cache = RedisPuzzleCache(host=redis_host, port=redis_port)
        
        logger.info("Clearing ALL puzzles from Redis...")
        response = input("Are you sure? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Cancelled")
            return False
        
        deleted = cache.clear_cache()
        logger.success(f"Deleted {deleted:,} keys from Redis")
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def verify_cache(sample_size: int = 100, redis_host: str = "localhost", redis_port: int = 6381):
    """Verify cache integrity."""
    try:
        cache = RedisPuzzleCache(host=redis_host, port=redis_port)
        
        logger.info(f"Verifying puzzle cache integrity...")
        logger.info(f"Sampling {sample_size} puzzles...")
        
        result = cache.verify_integrity(sample_size)
        
        if not result.get('success'):
            logger.error(f"Verification failed: {result.get('error', 'Unknown error')}")
            return False
        
        print("\n" + "="*70)
        print("CACHE INTEGRITY CHECK")
        print("="*70)
        print(f"Total puzzles: {result['total_puzzles']:,}")
        print(f"Sampled:       {result['sampled']}")
        print(f"Valid:         {result['valid']}")
        print(f"Invalid:       {result['invalid']}")
        print(f"Missing:       {result['missing']}")
        print(f"Integrity:     {result['integrity']}")
        print("="*70 + "\n")
        
        if result['missing'] > 0:
            logger.warning(f"Found {result['missing']} missing puzzles!")
        if result['invalid'] > 0:
            logger.warning(f"Found {result['invalid']} invalid puzzles!")
        
        if result['valid'] == result['sampled']:
            logger.success("✓ Cache integrity verified - all sampled puzzles valid")
        
        cache.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify cache: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index Lichess puzzles into Redis cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all puzzles
  python scripts/index_puzzles_to_redis.py \\
      --input chess-federated-learning/data/databases/lichess_puzzles.csv.zst

  # Index puzzles in rating range
  python scripts/index_puzzles_to_redis.py \\
      --input chess-federated-learning/data/databases/lichess_puzzles.csv.zst \\
      --min-rating 1600 --max-rating 2400

  # Show cache statistics
  python scripts/index_puzzles_to_redis.py --action stats

  # Clear cache
  python scripts/index_puzzles_to_redis.py --action clear

  # Verify cache integrity
  python scripts/index_puzzles_to_redis.py --action verify
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
        help='Path to input puzzle CSV (required for index action)'
    )
    parser.add_argument(
        '--min-rating',
        type=int,
        default=0,
        help='Minimum puzzle rating (default: 0)'
    )
    parser.add_argument(
        '--max-rating',
        type=int,
        default=5000,
        help='Maximum puzzle rating (default: 5000)'
    )
    parser.add_argument(
        '--max-puzzles',
        type=int,
        help='Maximum puzzles to index (default: unlimited)'
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
        help='Batch size for storing puzzles (default: 1000)'
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
        
        success = index_puzzles(
            input_path=args.input,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            max_puzzles=args.max_puzzles,
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
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'verify':
        success = verify_cache(
            sample_size=args.sample_size,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
