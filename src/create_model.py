#!/usr/bin/env python3
"""
Create and inspect the GPT model architecture.
Run this after tokenization to see your model structure. NOT ESSENTIAL
"""

import torch
from pathlib import Path
from model import GPTModel
from config import get_config
from tokenizer import SubwordTokenizer  # ← CHANGED


def load_vocab_size(vocab_path: str = 'data/tokenizer.json') -> int:  # ← CHANGED
    """Load vocabulary size from tokenizer."""
    tokenizer = SubwordTokenizer.load(vocab_path)  # ← CHANGED
    return tokenizer.vocab_size  # ← CHANGED


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
    
    total_params = model.get_num_params()
    print(f"  Total: {total_params:,}")
    print()
    
    # Memory estimate
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Memory (float32): {param_size_mb:.2f} MB")
    print(f"Memory (float16): {param_size_mb/2:.2f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and inspect GPT model')
    parser.add_argument(
        '--vocab-path',
        type=str,
        default='data/tokenizer.json',  # ← CHANGED
        help='Path to tokenizer file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['tiny', 'small', 'medium'],
        help='Model configuration size'
    )
    
    args = parser.parse_args()
    
    # Check if vocab file exists
    if not Path(args.vocab_path).exists():
        print(f"Error: Tokenizer file not found at {args.vocab_path}")
        print("Please run tokenization first: python tokenize.py")
        return
    
    print("="*60)
    print("PARROT WITH ALZHEIMER'S - MODEL ARCHITECTURE")
    print("="*60)
    print()
    
    # Load vocabulary size
    print(f"Loading tokenizer from: {args.vocab_path}")
    vocab_size = load_vocab_size(args.vocab_path)
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Create model
    print(f"Creating model with '{args.config}' configuration...")
    config = get_config(args.config)
    config.vocab_size = vocab_size
    
    model = GPTModel(**config.to_dict())
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
    print("Next step: python train.py")
    print("="*60)


if __name__ == '__main__':
    main()