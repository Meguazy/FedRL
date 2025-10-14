"""
Database preprocessor for creating fast-access cache files.

This script pre-processes large compressed PGN databases into separate
cache files for each playstyle, enabling instant random access during training.

Usage:
    python -m data.database_preprocessor \
        --input databases/lichess_db_standard_rated_2024-01.pgn.zst \
        --output data/cache/ \
        --min-rating 2000
"""

import pickle
import argparse
from pathlib import Path
from typing import List, Dict
from loguru import logger

from .game_loader import GameLoader, GameFilter
from .sample_extractor import SampleExtractor, ExtractionConfig, TrainingSample


class DatabasePreprocessor:
    """Pre-process PGN databases into fast-access cache files."""
    
    def __init__(self, pgn_path: str, cache_dir: str, min_rating: int = 2000):
        """
        Initialize preprocessor.
        
        Args:
            pgn_path: Path to source PGN database
            cache_dir: Directory to store cache files
            min_rating: Minimum rating filter
        """
        self.pgn_path = pgn_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_rating = min_rating
        
        logger.info(f"Initialized preprocessor: {pgn_path} -> {cache_dir}")
    
    def preprocess_by_playstyle(self, playstyle: str, chunk_size: int = 10000):
        """
        Extract and cache games for a specific playstyle.
        
        Args:
            playstyle: 'tactical' or 'positional'
            chunk_size: Number of samples per cache chunk
        """
        logger.info(f"Preprocessing {playstyle} games...")
        
        # Create sample extractor
        extraction_config = ExtractionConfig(
            skip_opening_moves=10,
            skip_endgame_moves=6,
            sample_rate=1.0,
            shuffle_games=False  # Keep order for consistent chunks
        )
        extractor = SampleExtractor(self.pgn_path, extraction_config)
        
        # Extract ALL games of this playstyle
        logger.info(f"Extracting all {playstyle} games with rating >= {self.min_rating}...")
        samples = extractor.extract_samples(
            num_games=None,  # Extract all available games
            playstyle=playstyle,
            min_rating=self.min_rating,
            offset=0
        )
        
        logger.success(f"Extracted {len(samples)} samples for {playstyle}")
        
        # Save in chunks for efficient loading
        cache_file = self.cache_dir / f"{playstyle}_samples.pkl"
        logger.info(f"Saving to {cache_file}...")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.success(f"Saved {len(samples)} samples to {cache_file}")
        
        # Save metadata
        metadata = {
            'playstyle': playstyle,
            'num_samples': len(samples),
            'min_rating': self.min_rating,
            'source_database': str(self.pgn_path)
        }
        
        metadata_file = self.cache_dir / f"{playstyle}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Metadata saved to {metadata_file}")
        
        return len(samples)
    
    def preprocess_all(self):
        """Preprocess both tactical and positional playstyles."""
        logger.info("="*70)
        logger.info("DATABASE PREPROCESSING")
        logger.info("="*70)
        
        tactical_count = self.preprocess_by_playstyle('tactical')
        positional_count = self.preprocess_by_playstyle('positional')
        
        logger.info("="*70)
        logger.success("PREPROCESSING COMPLETE")
        logger.info(f"  Tactical: {tactical_count:,} samples")
        logger.info(f"  Positional: {positional_count:,} samples")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info("="*70)


class CachedSampleLoader:
    """Fast sample loader using preprocessed cache files."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize cached loader.
        
        Args:
            cache_dir: Directory containing cache files
        """
        self.cache_dir = Path(cache_dir)
        self._cache = {}
        self._metadata = {}
        
        # Load metadata
        for playstyle in ['tactical', 'positional']:
            metadata_file = self.cache_dir / f"{playstyle}_metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self._metadata[playstyle] = pickle.load(f)
                logger.info(f"Loaded metadata for {playstyle}: "
                          f"{self._metadata[playstyle]['num_samples']:,} samples")
    
    def load_samples(self, playstyle: str, num_samples: int, offset: int = 0) -> List[TrainingSample]:
        """
        Load samples from cache with instant random access.
        
        Args:
            playstyle: 'tactical' or 'positional'
            num_samples: Number of samples to load
            offset: Starting offset in the cache
            
        Returns:
            List of training samples
        """
        # Load cache if not already loaded
        if playstyle not in self._cache:
            cache_file = self.cache_dir / f"{playstyle}_samples.pkl"
            logger.info(f"Loading {playstyle} cache from {cache_file}...")
            
            with open(cache_file, 'rb') as f:
                self._cache[playstyle] = pickle.load(f)
            
            logger.success(f"Loaded {len(self._cache[playstyle]):,} samples into memory")
        
        # Return requested slice (instant!)
        samples = self._cache[playstyle]
        end_offset = min(offset + num_samples, len(samples))
        
        return samples[offset:end_offset]
    
    def get_metadata(self, playstyle: str) -> Dict:
        """Get metadata for a playstyle."""
        return self._metadata.get(playstyle, {})


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess PGN database into cache files")
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input PGN database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='chess-federated-learning/data/cache',
        help='Output directory for cache files'
    )
    parser.add_argument(
        '--min-rating',
        type=int,
        default=2000,
        help='Minimum player rating (default: 2000)'
    )
    parser.add_argument(
        '--playstyle',
        type=str,
        choices=['tactical', 'positional', 'both'],
        default='both',
        help='Which playstyle to preprocess (default: both)'
    )
    
    args = parser.parse_args()
    
    preprocessor = DatabasePreprocessor(args.input, args.output, args.min_rating)
    
    if args.playstyle == 'both':
        preprocessor.preprocess_all()
    else:
        preprocessor.preprocess_by_playstyle(args.playstyle)


if __name__ == '__main__':
    main()
