#!/usr/bin/env python3
"""
Create and inspect the GPT model architecture.
Displays model structure and parameter counts.
"""

import argparse
import sys
import torch
from pathlib import Path

from model import GPTModel
from config import get_config, print_config_comparison
from tokenizer import SubwordTokenizer
from constants import VOCAB_FILE


def print_model_info(model: GPTModel):
    """Print detailed model information."""
    print("Model Architecture:")
    print(f"  Vocabulary size: {model.vocab_size:,}")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Number of layers: {len(model.blocks)}")
    print(f"  Number of heads: {model.blocks[0].attn.num_heads}")
    print(f"  Head dimension: {model.blocks[0].attn.head_dim}")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"  Feed-forward dimension: {model.blocks[0].ff.fc1.out_features}")
    print()
    
    # Count parameters by component
    print("Parameters by component:")
    
    # Embeddings
    emb_params = sum(p.numel() for p in model.token_embedding.parameters())
    emb_params += sum(p.numel() for p in model.position_embedding.parameters())
    print(f"  Embeddings: {emb_params:,}")
    
    # Transformer blocks
    block_params = sum(p.numel() for p in model.blocks.parameters())
    print(f"  Transformer blocks: {block_params:,}")
    
    # Output head
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"  Output head: {head_params:,}")
    
    # Layer norm
    ln_params = sum(p.numel() for p in model.ln_f.parameters())
    print(f"  Final layer norm: {ln_params:,}")
    
    total_params = model.get_num_params()
    print(f"  Total: {total_params:,}")
    print()
    
    # Memory estimate
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Memory estimates:")
    print(f"  Model (float32): {param_size_mb:.2f} MB")
    print(f"  Model (float16): {param_size_mb/2:.2f} MB")
    print(f"  + Activations (approx): {param_size_mb * 3:.2f} MB")
    print(f"  Total (training): ~{param_size_mb * 4:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create and inspect GPT model')
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['tiny', 'small', 'medium', 'large'],
        help='Model configuration size'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configurations and exit'
    )
    
    args = parser.parse_args()
    
    # List configs and exit
    if args.list_configs:
        print_config_comparison()
        return
    
    print("="*60)
    print("GPT MODEL ARCHITECTURE INSPECTOR")
    print("="*60)
    print()
    
    # Check if vocab file exists
    if not VOCAB_FILE.exists():
        print(f"Error: Vocabulary file not found at {VOCAB_FILE}")
        print("Please run tokenization first:")
        print("  python tokenize.py")
        print()
        print("Or list available configurations:")
        print("  python create_model.py --list-configs")
        sys.exit(1)
    
    # Load vocabulary size
    print(f"Loading BPE tokenizer from: {VOCAB_FILE}")
    try:
        tokenizer = SubwordTokenizer.load(str(VOCAB_FILE))
        vocab_size = tokenizer.vocab_size
        print(f"✓ Vocabulary size: {vocab_size}")
        print(f"✓ Merge rules: {len(tokenizer.merges)}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        sys.exit(1)
    print()
    
    # Get configuration
    print(f"Creating model with '{args.config}' configuration...")
    config = get_config(args.config)
    config.vocab_size = vocab_size
    
    print(f"Configuration details:")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  FF dim: {config.ff_dim}")
    print(f"  Dropout: {config.dropout}")
    print()
    
    # Create model
    model = GPTModel(**config.to_dict())
    print("✓ Model created successfully")
    print()
    
    # Print model info
    print_model_info(model)
    
    # Test forward pass
    print("Testing forward pass...")
    test_input = torch.randint(0, vocab_size, (2, 10))
    with torch.no_grad():
        logits, _ = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print("  ✓ Forward pass successful")
    print()
    
    print("="*60)
    print("Model is ready for training!")
    print()
    print("Next steps:")
    print(f"  python train.py --config {args.config}")
    print()
    print("To see all available configurations:")
    print("  python create_model.py --list-configs")
    print("  python train.py --list-configs")
    print("="*60)


if __name__ == '__main__':
    main()