#!/usr/bin/env python3
"""
Model Configuration
Defines model architectures for different use cases and hardware.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for GPT model architecture."""
    
    vocab_size: int = 100  # Will be set from tokenizer
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 256
    ff_dim: Optional[int] = None
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Validate embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        
        # Set ff_dim to 4x embed_dim if not specified
        if self.ff_dim is None:
            self.ff_dim = 4 * self.embed_dim
    
    def to_dict(self):
        """Convert config to dictionary for model initialization."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout
        }


# Predefined configurations for different hardware/use cases

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

LARGE_CONFIG = ModelConfig(
    embed_dim=384,
    num_heads=12,
    num_layers=8,
    max_seq_len=512,
    ff_dim=1536,
    dropout=0.1
)


def get_config(name: str = 'small') -> ModelConfig:
    """
    Get predefined model configuration by name.
    
    Args:
        name: Configuration name ('tiny', 'small', 'medium', 'large')
        
    Returns:
        ModelConfig instance
        
    Raises:
        ValueError: If configuration name is unknown
    """
    configs = {
        'tiny': TINY_CONFIG,
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG,
        'large': LARGE_CONFIG
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    
    return configs[name]


def print_config_comparison():
    """Print comparison of all configurations."""
    print("Available Configurations:")
    print("="*70)
    print(f"{'Config':<10} {'Params':<12} {'Embed':<8} {'Layers':<8} {'Heads':<8} {'Context':<10}")
    print("-"*70)
    
    for name in ['tiny', 'small', 'medium', 'large']:
        config = get_config(name)
        # Rough parameter estimate (will be accurate after vocab_size is set)
        params = estimate_params(config, vocab_size=5000)
        print(f"{name:<10} ~{params/1e6:.1f}M{'':<7} {config.embed_dim:<8} "
              f"{config.num_layers:<8} {config.num_heads:<8} {config.max_seq_len:<10}")
    
    print("="*70)
    print("\nRecommendations:")
    print("  tiny   - Quick experiments (2-3 hours)")
    print("  small  - Standard training (3-4 hours)")  
    print("  medium - Better quality (8-12 hours)")
    print("  large  - Best quality (16-24 hours, requires 16GB+ RAM)")


def estimate_params(config: ModelConfig, vocab_size: int) -> int:
    """
    Estimate total parameters for a configuration.
    
    Args:
        config: Model configuration
        vocab_size: Vocabulary size
        
    Returns:
        Estimated parameter count
    """
    # Token embedding
    token_emb = vocab_size * config.embed_dim
    
    # Position embedding
    pos_emb = config.max_seq_len * config.embed_dim
    
    # Transformer blocks
    # Each block has: attention (4 weight matrices) + FF (2 weight matrices) + 4 layer norms
    attn_params = 4 * config.embed_dim * config.embed_dim  # Q, K, V, output projection
    ff_params = config.embed_dim * config.ff_dim + config.ff_dim * config.embed_dim
    ln_params = 4 * config.embed_dim  # 2 layer norms per block, each has 2 params per dim
    
    block_params = (attn_params + ff_params + ln_params) * config.num_layers
    
    # Output head
    head_params = config.embed_dim * vocab_size
    
    # Final layer norm
    final_ln = 2 * config.embed_dim
    
    total = token_emb + pos_emb + block_params + head_params + final_ln
    return int(total)


if __name__ == '__main__':
    print_config_comparison()