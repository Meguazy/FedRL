import time
import random
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

    def _create_mock_alphazero_model(self, num_residual_blocks: int = 3) -> Dict[str, Any]:
        """
        Create a mock AlphaZero model structure for testing.

        Mimics the structure of an AlphaZero neural network with:
        - Input convolution (shared layers)
        - Residual blocks (shared layers)
        - Policy head (cluster-specific)
        - Value head (cluster-specific)

        Args:
            num_residual_blocks: Number of residual blocks (default: 3)

        Returns:
            Dictionary with model layer parameters
        """
        layers = {}

        # Input convolution (SHARED - will be aggregated across clusters)
        layers["input_conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(64)] for _ in range(256)
        ]
        layers["input_conv.bias"] = [random.uniform(-1, 1) for _ in range(256)]

        # Residual blocks (SHARED - will be aggregated across clusters)
        for i in range(num_residual_blocks):
            # Conv1
            layers[f"residual.{i}.conv1.weight"] = [
                [random.uniform(-1, 1) for _ in range(256)] for _ in range(256)
            ]
            layers[f"residual.{i}.conv1.bias"] = [random.uniform(-1, 1) for _ in range(256)]

            # Batch norm
            layers[f"residual.{i}.bn1.weight"] = [random.uniform(-1, 1) for _ in range(256)]
            layers[f"residual.{i}.bn1.bias"] = [random.uniform(-1, 1) for _ in range(256)]

            # Conv2
            layers[f"residual.{i}.conv2.weight"] = [
                [random.uniform(-1, 1) for _ in range(256)] for _ in range(256)
            ]
            layers[f"residual.{i}.conv2.bias"] = [random.uniform(-1, 1) for _ in range(256)]

            # Batch norm 2
            layers[f"residual.{i}.bn2.weight"] = [random.uniform(-1, 1) for _ in range(256)]
            layers[f"residual.{i}.bn2.bias"] = [random.uniform(-1, 1) for _ in range(256)]

        # Policy head (CLUSTER-SPECIFIC - unique per playstyle)
        layers["policy_head.conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(32)
        ]
        layers["policy_head.conv.bias"] = [random.uniform(-1, 1) for _ in range(32)]
        layers["policy_head.fc.weight"] = [
            [random.uniform(-1, 1) for _ in range(2048)] for _ in range(1968)  # 1968 possible moves
        ]
        layers["policy_head.fc.bias"] = [random.uniform(-1, 1) for _ in range(1968)]

        # Value head (CLUSTER-SPECIFIC - unique per playstyle)
        layers["value_head.conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(1)
        ]
        layers["value_head.conv.bias"] = [random.uniform(-1, 1)]
        layers["value_head.fc1.weight"] = [
            [random.uniform(-1, 1) for _ in range(64)] for _ in range(256)
        ]
        layers["value_head.fc1.bias"] = [random.uniform(-1, 1) for _ in range(256)]
        layers["value_head.fc2.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(1)
        ]
        layers["value_head.fc2.bias"] = [random.uniform(-1, 1)]

        return layers

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

        # Initialize model if empty (first round)
        if not initial_model_state:
            log.info("Initializing empty model with mock AlphaZero structure")
            initial_model_state = self._create_mock_alphazero_model(num_residual_blocks=3)

        # Make small random changes to model weights
        updated_model = {}
        for key, value in initial_model_state.items():
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):  # 2D
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
        self._add_to_history(result)
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




