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
import json
import argparse
import chess
import chess.pgn
from pathlib import Path
from typing import List, Dict, Optional
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
    
    def _load_progress(self, playstyle: str) -> Dict:
        """Load progress tracker from JSON file."""
        progress_file = self.cache_dir / f"{playstyle}_progress.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Found progress file: {progress_file}")
                logger.info(f"  Games processed: {progress.get('games_processed', 0)}")
                logger.info(f"  Samples extracted: {progress.get('num_samples', 0)}")
                logger.info(f"  Last checkpoint: {progress.get('checkpoint', 0)}")
                return progress
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        
        return {
            'playstyle': playstyle,
            'games_processed': 0,
            'num_samples': 0,
            'checkpoint': 0,
            'source_database': str(self.pgn_path),
            'min_rating': self.min_rating
        }
    
    def _save_progress(self, playstyle: str, progress: Dict):
        """Save progress tracker to JSON file."""
        progress_file = self.cache_dir / f"{playstyle}_progress.json"
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _extract_samples_streaming(
        self, 
        extractor: SampleExtractor,
        game_iterator,
        num_games: int,
        playstyle: str,
        min_rating: int,
        current_position: int
    ) -> List[TrainingSample]:
        """
        Extract samples from an open file handle without reopening.
        
        This method reads games sequentially from the current position
        without reopening the file or seeking.
        """
        samples = []
        games_extracted = 0
        games_checked = 0
        
        while games_extracted < num_games:
            game = chess.pgn.read_game(game_iterator)
            
            if game is None:
                # End of file
                break
            
            games_checked += 1
            
            # Apply filters
            game_filter = GameFilter(min_rating=min_rating, playstyle=playstyle)
            if not extractor.game_loader._passes_filter(game, game_filter):
                continue
            
            # Extract samples from this game
            game_samples = extractor._extract_samples_from_game(game)
            samples.extend(game_samples)
            games_extracted += 1
            
            # Log progress
            if games_extracted % 10 == 0:
                logger.debug(f"Extracted {games_extracted}/{num_games} games, {len(samples)} samples so far")
        
        logger.info(f"Checked {games_checked} games, extracted {games_extracted} matching games, {len(samples)} total samples")
        return samples
    
    def preprocess_by_playstyle(self, playstyle: str, checkpoint_games: int = 1000):
        """
        Extract and cache games for a specific playstyle with incremental checkpoints.
        
        This method saves progress at regular intervals, so you can stop it anytime
        and still have usable cached data. Each checkpoint overwrites the previous one.
        Progress is tracked in a JSON file for automatic resume capability.
        
        Args:
            playstyle: 'tactical' or 'positional'
            checkpoint_games: Number of games to process before saving checkpoint (default: 1000)
        """
        logger.info(f"Preprocessing {playstyle} games...")
        
        cache_file = self.cache_dir / f"{playstyle}_samples.pkl"
        metadata_file = self.cache_dir / f"{playstyle}_metadata.pkl"
        
        # Load progress tracker (automatic resume)
        progress = self._load_progress(playstyle)
        games_processed = progress['games_processed']
        
        # Load existing samples if any
        existing_samples = []
        if cache_file.exists() and games_processed > 0:
            logger.info(f"Loading existing cache from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    existing_samples = pickle.load(f)
                logger.success(f"Loaded {len(existing_samples)} existing samples")
                logger.info(f"Resuming from game {games_processed}")
            except Exception as e:
                logger.error(f"Could not load existing cache: {e}")
                logger.info("Starting fresh...")
                existing_samples = []
                games_processed = 0
                progress['games_processed'] = 0
                progress['num_samples'] = 0
                progress['checkpoint'] = 0
        elif games_processed > 0:
            logger.warning(f"Progress file indicates {games_processed} games processed but cache not found")
            logger.info("Starting fresh...")
            games_processed = 0
            progress['games_processed'] = 0
            progress['num_samples'] = 0
            progress['checkpoint'] = 0
        
        # Create sample extractor (reused across checkpoints for efficiency)
        extraction_config = ExtractionConfig(
            skip_opening_moves=10,
            skip_endgame_moves=6,
            sample_rate=1.0,
            shuffle_games=False  # Keep order for consistent chunks
        )
        extractor = SampleExtractor(self.pgn_path, extraction_config)
        
        # Open game loader ONCE and keep file handle open
        # This avoids reopening and seeking through the compressed file repeatedly
        logger.info("Opening PGN file for streaming extraction...")
        game_iterator = extractor.game_loader._open_pgn_file()
        current_position = 0  # Track where we are in the file
        
        # Process in chunks with checkpoints
        logger.info(f"="*70)
        logger.info(f"Starting incremental extraction (checkpoint every {checkpoint_games} games)")
        logger.info(f"You can stop at any time with Ctrl+C - progress will be saved!")
        logger.info(f"Re-run the script to automatically resume from last checkpoint")
        logger.info(f"="*70)
        
        all_samples = existing_samples.copy()
        checkpoint_num = progress['checkpoint'] + 1
        
        try:
            # If resuming, skip to the correct position in the file
            if games_processed > 0:
                logger.info(f"Resuming: skipping {games_processed} already-processed games...")
                logger.info(f"This may take a few minutes for the first resume...")
                skipped = 0
                while current_position < games_processed:
                    game = chess.pgn.read_game(game_iterator)
                    if game is None:
                        logger.warning("Reached end of file while skipping - no more games!")
                        break
                    
                    # Check if this game matches our filters
                    if extractor.game_loader._passes_filter(
                        game, 
                        GameFilter(min_rating=self.min_rating, playstyle=playstyle)
                    ):
                        current_position += 1
                        if current_position % 100 == 0:
                            logger.debug(f"Resume skip progress: {current_position}/{games_processed}")
                
                logger.success(f"Resume complete! Now at position {current_position}")
            
            while True:
                # Extract next chunk of games from current file position
                logger.info(f"\n[Checkpoint {checkpoint_num}] Extracting games {games_processed} to {games_processed + checkpoint_games}...")
                
                chunk_samples = self._extract_samples_streaming(
                    extractor=extractor,
                    game_iterator=game_iterator,
                    num_games=checkpoint_games,
                    playstyle=playstyle,
                    min_rating=self.min_rating,
                    current_position=current_position
                )
                
                if not chunk_samples:
                    logger.info("No more games found - extraction complete!")
                    break
                
                current_position += checkpoint_games
                
                # Add to accumulated samples
                all_samples.extend(chunk_samples)
                games_processed += checkpoint_games
                
                logger.success(f"[Checkpoint {checkpoint_num}] Extracted {len(chunk_samples)} samples from this chunk")
                logger.info(f"[Checkpoint {checkpoint_num}] Total samples so far: {len(all_samples):,}")
                
                # Save checkpoint (overwrite)
                logger.info(f"[Checkpoint {checkpoint_num}] Saving checkpoint to {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update progress tracker
                progress['games_processed'] = games_processed
                progress['num_samples'] = len(all_samples)
                progress['checkpoint'] = checkpoint_num
                self._save_progress(playstyle, progress)
                
                # Update metadata
                metadata = {
                    'playstyle': playstyle,
                    'num_samples': len(all_samples),
                    'games_processed': games_processed,
                    'min_rating': self.min_rating,
                    'source_database': str(self.pgn_path),
                    'checkpoint': checkpoint_num
                }
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                
                logger.success(f"[Checkpoint {checkpoint_num}] ✓ Checkpoint saved! Safe to stop now.")
                checkpoint_num += 1
                
        except KeyboardInterrupt:
            logger.warning("\n\nInterrupted by user!")
            logger.info(f"Progress saved: {len(all_samples):,} samples in cache")
            logger.info(f"You can resume later or use what's cached for training")
            return len(all_samples)
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            logger.info(f"Last checkpoint saved: {len(all_samples):,} samples")
            raise
        finally:
            # Close the file handle
            try:
                game_iterator.close()
                logger.debug("Closed PGN file handle")
            except:
                pass
        
        logger.success(f"\n{'='*70}")
        logger.success(f"EXTRACTION COMPLETE for {playstyle}")
        logger.success(f"Final count: {len(all_samples):,} samples")
        logger.success(f"Games processed: {games_processed}")
        logger.success(f"{'='*70}")
        
        return len(all_samples)
    
    def preprocess_all(self, checkpoint_games: int = 1000):
        """Preprocess both tactical and positional playstyles."""
        logger.info("="*70)
        logger.info("DATABASE PREPROCESSING")
        logger.info("="*70)
        
        tactical_count = self.preprocess_by_playstyle('tactical', checkpoint_games=checkpoint_games)
        positional_count = self.preprocess_by_playstyle('positional', checkpoint_games=checkpoint_games)
        
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
            
            try:
                with open(cache_file, 'rb') as f:
                    self._cache[playstyle] = pickle.load(f)
                
                logger.success(f"Loaded {len(self._cache[playstyle]):,} samples into memory")
            except ModuleNotFoundError as e:
                logger.error(f"Module import error while loading cache: {e}")
                logger.error(f"This usually means the cache was created with a different module structure")
                logger.error(f"Try regenerating the cache with: ./scripts/preprocess_{playstyle}_database.sh --reset")
                raise
        
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
    parser.add_argument(
        '--checkpoint-games',
        type=int,
        default=1000,
        help='Number of games to process before saving checkpoint (default: 1000)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start from scratch (deletes existing cache and progress)'
    )
    
    args = parser.parse_args()
    
    preprocessor = DatabasePreprocessor(args.input, args.output, args.min_rating)
    
    # Reset if requested
    if args.reset:
        logger.info("RESET requested - deleting existing cache and progress files...")
        cache_dir = Path(args.output)
        
        if args.playstyle == 'both':
            playstyles = ['tactical', 'positional']
        else:
            playstyles = [args.playstyle]
        
        for playstyle in playstyles:
            for suffix in ['samples.pkl', 'metadata.pkl', 'progress.json']:
                file_path = cache_dir / f"{playstyle}_{suffix}"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted: {file_path}")
        
        logger.success("Reset complete - starting fresh")
    
    if args.playstyle == 'both':
        preprocessor.preprocess_all(checkpoint_games=args.checkpoint_games)
    else:
        preprocessor.preprocess_by_playstyle(args.playstyle, checkpoint_games=args.checkpoint_games)


if __name__ == '__main__':
    main()
