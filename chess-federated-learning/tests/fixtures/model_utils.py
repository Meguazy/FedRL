import random
from typing import Any, Dict
from loguru import logger


class MockAlphaZeroModel:
    """
    Mock AlphaZero-style model for testing inter-cluster aggregation.
    
    Mimics the structure of an AlphaZero neural network with:
    - Input convolution
    - Residual blocks
    - Policy head
    - Value head
    """
    
    def __init__(self, num_residual_blocks: int = 3):
        """
        Initialize mock AlphaZero model.
        
        Args:
            num_residual_blocks: Number of residual blocks (default: 3 for testing)
        """
        log = logger.bind(context="MockAlphaZeroModel.__init__")
        
        self.layers = {}
        
        # Input convolution (board encoding)
        self.layers["input_conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(64)] for _ in range(256)
        ]
        self.layers["input_conv.bias"] = [random.uniform(-1, 1) for _ in range(256)]
        
        # Residual blocks
        for i in range(num_residual_blocks):
            # Conv1
            self.layers[f"residual.{i}.conv1.weight"] = [
                [random.uniform(-1, 1) for _ in range(256)] for _ in range(256)
            ]
            self.layers[f"residual.{i}.conv1.bias"] = [random.uniform(-1, 1) for _ in range(256)]
            
            # Batch norm
            self.layers[f"residual.{i}.bn1.weight"] = [random.uniform(-1, 1) for _ in range(256)]
            self.layers[f"residual.{i}.bn1.bias"] = [random.uniform(-1, 1) for _ in range(256)]
            
            # Conv2
            self.layers[f"residual.{i}.conv2.weight"] = [
                [random.uniform(-1, 1) for _ in range(256)] for _ in range(256)
            ]
            self.layers[f"residual.{i}.conv2.bias"] = [random.uniform(-1, 1) for _ in range(256)]
            
            # Batch norm 2
            self.layers[f"residual.{i}.bn2.weight"] = [random.uniform(-1, 1) for _ in range(256)]
            self.layers[f"residual.{i}.bn2.bias"] = [random.uniform(-1, 1) for _ in range(256)]
        
        # Policy head
        self.layers["policy_head.conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(32)
        ]
        self.layers["policy_head.conv.bias"] = [random.uniform(-1, 1) for _ in range(32)]
        self.layers["policy_head.fc.weight"] = [
            [random.uniform(-1, 1) for _ in range(2048)] for _ in range(1968)  # 1968 possible moves in chess
        ]
        self.layers["policy_head.fc.bias"] = [random.uniform(-1, 1) for _ in range(1968)]
        
        # Value head
        self.layers["value_head.conv.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(1)
        ]
        self.layers["value_head.conv.bias"] = [random.uniform(-1, 1)]
        self.layers["value_head.fc1.weight"] = [
            [random.uniform(-1, 1) for _ in range(64)] for _ in range(256)
        ]
        self.layers["value_head.fc1.bias"] = [random.uniform(-1, 1) for _ in range(256)]
        self.layers["value_head.fc2.weight"] = [
            [random.uniform(-1, 1) for _ in range(256)] for _ in range(1)
        ]
        self.layers["value_head.fc2.bias"] = [random.uniform(-1, 1)]
        
        log.debug(f"Created MockAlphaZeroModel with {len(self.layers)} parameters")
    
    def state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary."""
        return self.layers.copy()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dictionary."""
        self.layers = state_dict.copy()
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        total = 0
        for key, value in self.layers.items():
            if isinstance(value, list):
                if isinstance(value[0], list):
                    total += len(value) * len(value[0])
                else:
                    total += len(value)
        return total


def generate_alphazero_model(num_residual_blocks: int = 10, seed: int = None) -> MockAlphaZeroModel:
    """
    Generate a mock AlphaZero model for testing.
    
    Args:
        num_residual_blocks: Number of residual blocks
        seed: Random seed for reproducibility
    
    Returns:
        MockAlphaZeroModel instance
    """
    log = logger.bind(context="generate_alphazero_model")
    
    if seed is not None:
        random.seed(seed)
    
    model = MockAlphaZeroModel(num_residual_blocks)
    log.info(f"Generated AlphaZero model with {num_residual_blocks} residual blocks, "
             f"{model.get_parameter_count()} parameters")
    
    return model