from loguru import logger
from client.trainer.trainer_dummy import DummyTrainer
from client.trainer.trainer_interface import TrainerInterface, TrainingConfig


def create_trainer(trainer_type: str, node_id: str, cluster_id: str,
                  config: TrainingConfig) -> TrainerInterface:
    """
    Factory function to create trainers.
    
    Args:
        trainer_type: Type of trainer ("dummy", "supervised", "alphazero")
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
        # TODO: Implement SupervisedTrainer
        raise NotImplementedError("SupervisedTrainer not yet implemented")
    elif trainer_type == "alphazero":
        # TODO: Implement AlphaZeroTrainer
        raise NotImplementedError("AlphaZeroTrainer not yet implemented")
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
