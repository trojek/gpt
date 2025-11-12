"""
IMPROVED Model Configuration for Better Factual Knowledge
Use this instead of the default config.py

NOTE: This file works with both character-level and subword tokenization.
No changes needed for subword tokenization.
"""

from dataclasses import dataclass


@dataclass
class ImprovedModelConfig:
    """Improved configuration for better factual knowledge."""
    
    # Model architecture - LARGER for better memory
    vocab_size: int = 100  # Will be set from tokenizer
    embed_dim: int = 256   # DOUBLED from 128
    num_heads: int = 8     # DOUBLED from 4
    num_layers: int = 6    # Increased from 4
    max_seq_len: int = 512 # DOUBLED from 256
    ff_dim: int = 1024     # 4 * embed_dim
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


# Predefined configurations for different hardware

# For M1 (8GB) - Balanced
BALANCED_CONFIG = ImprovedModelConfig(
    embed_dim=192,
    num_heads=6,
    num_layers=6,
    max_seq_len=384,
    ff_dim=768,
    dropout=0.1
)

# For M1 Pro/Max (16-32GB) - Better quality
QUALITY_CONFIG = ImprovedModelConfig(
    embed_dim=256,
    num_heads=8,
    num_layers=8,
    max_seq_len=512,
    ff_dim=1024,
    dropout=0.1
)

# For M1 Ultra / High-end GPU (64GB+) - Best quality
BEST_CONFIG = ImprovedModelConfig(
    embed_dim=384,
    num_heads=12,
    num_layers=10,
    max_seq_len=768,
    ff_dim=1536,
    dropout=0.1
)


def get_improved_config(hardware: str = 'balanced') -> ImprovedModelConfig:
    """
    Get improved model configuration based on hardware.
    
    Args:
        hardware: 'balanced' (M1 8GB), 'quality' (M1 Pro/Max), 'best' (M1 Ultra/GPU)
        
    Returns:
        ImprovedModelConfig instance
    """
    configs = {
        'balanced': BALANCED_CONFIG,
        'quality': QUALITY_CONFIG,
        'best': BEST_CONFIG
    }
    
    if hardware not in configs:
        raise ValueError(f"Unknown config: {hardware}. Choose from {list(configs.keys())}")
    
    return configs[hardware]


# Comparison with original
"""
ORIGINAL (Poor Results):
- embed_dim: 128
- num_heads: 4  
- num_layers: 4
- max_seq_len: 256
- Parameters: ~2.8M

IMPROVED BALANCED (Much Better):
- embed_dim: 192
- num_heads: 6
- num_layers: 6  
- max_seq_len: 384
- Parameters: ~8M (3x more capacity)

IMPROVED QUALITY (Best for most):
- embed_dim: 256
- num_heads: 8
- num_layers: 8
- max_seq_len: 512
- Parameters: ~18M (6x more capacity for facts!)
"""