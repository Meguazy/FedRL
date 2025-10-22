from loguru import logger
from .trainer_dummy import DummyTrainer
from .trainer_supervised import SupervisedTrainer
from .trainer_puzzle import PuzzleTrainer
from .trainer_interface import TrainerInterface, TrainingConfig


def create_trainer(trainer_type: str, node_id: str, cluster_id: str,
                  config: TrainingConfig) -> TrainerInterface:
    """
    Factory function to create trainers.
    
    Args:
        trainer_type: Type of trainer ("dummy", "supervised", "puzzle", "alphazero")
        node_id: Node identifier
        cluster_id: Cluster identifier
        config: Training configuration
    
    Returns:
        TrainerInterface implementation
    
    Raises:
        ValueError: If trainer type is unknown
    """
    log = logger.bind(context="create_trainer")
    log.info(f"Creating {trainer_type} trainer for node {node_id}")
    
    if trainer_type == "dummy":
        return DummyTrainer(node_id, cluster_id, config)
    elif trainer_type == "supervised":
        return SupervisedTrainer(node_id, cluster_id, config)
    elif trainer_type == "puzzle":
        # Default puzzle database path (can be overridden later in start_node.py)
        puzzle_database_path = config.additional_params.get(
            'puzzle_database_path',
            'chess-federated-learning/data/databases/lichess_puzzles.csv.zst'
        )
        redis_host = config.additional_params.get('redis_host', 'localhost')
        redis_port = config.additional_params.get('redis_port', 6381)
        device = config.additional_params.get('device', 'cpu')
        
        return PuzzleTrainer(
            node_id=node_id,
            config=config,
            puzzle_database_path=puzzle_database_path,
            device=device,
            redis_host=redis_host,
            redis_port=redis_port
        )
    elif trainer_type == "alphazero":
        # TODO: Implement AlphaZeroTrainer
        raise NotImplementedError("AlphaZeroTrainer not yet implemented")
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
