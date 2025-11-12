"""
Model configuration for Parrot with Alzheimer's LLM.
Defines default hyperparameters for the transformer architecture.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for GPT model architecture."""
    
    # Model architecture
    vocab_size: int = 100  # Will be set from tokenizer
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 256
    ff_dim: int = 512  # 4 * embed_dim
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        
        if self.ff_dim is None:
            self.ff_dim = 4 * self.embed_dim
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        }


# Predefined configurations
TINY_CONFIG = ModelConfig(
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    max_seq_len=128,
    ff_dim=256,
    dropout=0.1
)

SMALL_CONFIG = ModelConfig(
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=256,
    ff_dim=512,
    dropout=0.1
)

MEDIUM_CONFIG = ModelConfig(
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=512,
    ff_dim=1024,
    dropout=0.1
)


def get_config(size: str = 'small') -> ModelConfig:
    """
    Get predefined model configuration.
    
    Args:
        size: Model size ('tiny', 'small', or 'medium')
        
    Returns:
        ModelConfig instance
    """
    configs = {
        'tiny': TINY_CONFIG,
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG
    }
    
    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")
    
    return configs[size]