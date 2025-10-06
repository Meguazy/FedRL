import time
from typing import Any, Dict

from loguru import logger
from .trainer_interface import TrainerInterface, TrainingResult


class DummyTrainer(TrainerInterface):
    """
    Dummy trainer for testing and development.
    
    This trainer simulates training by:
    - Making small random modifications to model weights
    - Generating synthetic metrics
    - Completing quickly for fast iteration
    
    Use this for:
    - Testing the federated learning pipeline
    - Validating storage and aggregation
    - Development without chess engine
    """
    
    async def train(self, initial_model_state: Dict[str, Any]) -> TrainingResult:
        """
        Simulate training with dummy data.
        
        Args:
            initial_model_state: Starting model state
        
        Returns:
            TrainingResult with slightly modified model and synthetic metrics
        """
        log = logger.bind(context=f"DummyTrainer.{self.node_id}")
        log.info(f"Starting dummy training for {self.config.games_per_round} games")
        
        start_time = time.time()
        
        # Simulate training delay
        import asyncio
        await asyncio.sleep(0.5)  # 500ms simulated training
        
        # Make small random changes to model weights
        import random
        updated_model = {}
        for key, value in initial_model_state.items():
            if isinstance(value, list):
                if isinstance(value[0], list):  # 2D
                    updated_model[key] = [
                        [v + random.uniform(-0.01, 0.01) for v in row]
                        for row in value
                    ]
                else:  # 1D
                    updated_model[key] = [v + random.uniform(-0.01, 0.01) for v in value]
            else:
                updated_model[key] = value + random.uniform(-0.01, 0.01)
        
        # Generate synthetic metrics
        training_time = time.time() - start_time
        samples = self.config.games_per_round * 50  # ~50 positions per game
        loss = max(0.1, random.uniform(0.3, 0.7) - len(self.training_history) * 0.02)
        
        result = TrainingResult(
            model_state=updated_model,
            samples=samples,
            loss=loss,
            games_played=self.config.games_per_round,
            training_time=training_time,
            metrics={
                "accuracy": min(0.95, 0.6 + len(self.training_history) * 0.02),
                "win_rate": random.uniform(0.4, 0.6),
                "avg_game_length": random.randint(30, 80),
            },
            success=True
        )
        
        # Update statistics
        self.training_history.append(result)
        self.total_games_played += result.games_played
        self.total_training_time += result.training_time
        self.current_model_state = updated_model
        
        log.info(f"Dummy training complete: loss={loss:.4f}, samples={samples}")
        return result
    
    async def evaluate(self, model_state: Dict[str, Any], 
                      num_games: int = 10) -> Dict[str, Any]:
        """
        Simulate evaluation with dummy metrics.
        
        Args:
            model_state: Model to evaluate
            num_games: Number of games to simulate
        
        Returns:
            Dict with synthetic evaluation metrics
        """
        log = logger.bind(context=f"DummyTrainer.{self.node_id}")
        log.info(f"Evaluating model with {num_games} dummy games")
        
        import asyncio
        import random
        
        await asyncio.sleep(0.2)  # Simulate evaluation time
        
        return {
            "games_played": num_games,
            "win_rate": random.uniform(0.45, 0.55),
            "avg_loss": random.uniform(0.2, 0.4),
            "avg_game_length": random.randint(40, 70),
            "evaluation_time": 0.2,
        }




