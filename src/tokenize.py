#!/usr/bin/env python3
"""
Tokenization script using Byte Pair Encoding (BPE).
1. Trains BPE tokenizer on training data
2. Tokenizes train and validation files
"""

import argparse
import numpy as np
from pathlib import Path
from tokenizer import SubwordTokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Train BPE tokenizer and tokenize text files'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default='data/train.txt',
        help='Path to training text file'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default='data/val.txt',
        help='Path to validation text file'
    )
    parser.add_argument(
        '--vocab-file',
        type=str,
        default='data/vocab.json',
        help='Path to save vocabulary file'
    )
    parser.add_argument(
        '--train-tokens',
        type=str,
        default='data/train_tokens.npy',
        help='Path to save training tokens'
    )
    parser.add_argument(
        '--val-tokens',
        type=str,
        default='data/val_tokens.npy',
        help='Path to save validation tokens'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=5000,
        help='Target vocabulary size for BPE'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("BPE TOKENIZATION")
    print("="*70)
    print()
    
    # Check if input files exist
    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    
    if not train_path.exists():
        print(f"Error: Training file not found at {args.train_file}")
        print("Please run preprocessing first:")
        print("  python preprocess_data.py --input data/raw_wikipedia.txt")
        return
    
    if not val_path.exists():
        print(f"Error: Validation file not found at {args.val_file}")
        print("Please run preprocessing first:")
        print("  python preprocess_data.py --input data/raw_wikipedia.txt")
        return
    
    # Step 1: Load training data
    print(f"Loading training data from: {args.train_file}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    print(f"Training text size: {len(train_text):,} characters")
    print()
    
    # Step 2: Train BPE tokenizer
    print(f"Training BPE tokenizer (target vocab_size={args.vocab_size})...")
    print("This may take a few minutes...")
    print()
    
    tokenizer = SubwordTokenizer.train(
        text=train_text,
        vocab_size=args.vocab_size,
        min_frequency=2
    )
    
    print()
    print(f"✓ Tokenizer trained successfully")
    print(f"  Final vocabulary size: {tokenizer.vocab_size}")
    print(f"  Number of merge rules: {len(tokenizer.merges)}")
    print()
    
    # Save vocabulary
    vocab_path = Path(args.vocab_file)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.vocab_file)
    print(f"✓ Vocabulary saved to: {args.vocab_file}")
    print()
    
    # Step 3: Tokenize training data
    print("Tokenizing training data...")
    train_tokens = tokenizer.encode(train_text)
    print(f"Training tokens: {len(train_tokens):,}")
    
    # Convert to numpy array
    train_tokens_array = np.array(train_tokens, dtype=np.uint16)
    
    train_tokens_path = Path(args.train_tokens)
    train_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.train_tokens, train_tokens_array)
    
    tokens_size_mb = train_tokens_array.nbytes / (1024 * 1024)
    print(f"✓ Training tokens saved to: {args.train_tokens}")
    print(f"  Size: {tokens_size_mb:.2f} MB")
    print()
    
    # Step 4: Tokenize validation data
    print("Tokenizing validation data...")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    
    val_tokens = tokenizer.encode(val_text)
    print(f"Validation tokens: {len(val_tokens):,}")
    
    # Convert to numpy array
    val_tokens_array = np.array(val_tokens, dtype=np.uint16)
    
    val_tokens_path = Path(args.val_tokens)
    val_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.val_tokens, val_tokens_array)
    
    val_size_mb = val_tokens_array.nbytes / (1024 * 1024)
    print(f"✓ Validation tokens saved to: {args.val_tokens}")
    print(f"  Size: {val_size_mb:.2f} MB")
    print()
    
    # Statistics
    print("="*70)
    print("TOKENIZATION COMPLETE")
    print("="*70)
    print()
    print("Output files:")
    print(f"  Vocabulary: {args.vocab_file} ({tokenizer.vocab_size} tokens)")
    print(f"  Training:   {args.train_tokens} ({len(train_tokens):,} tokens)")
    print(f"  Validation: {args.val_tokens} ({len(val_tokens):,} tokens)")
    print()
    
    # Compression ratio
    char_to_token_ratio = len(train_text) / len(train_tokens)
    print(f"Compression ratio: {char_to_token_ratio:.2f}x")
    print(f"  (1 token ≈ {char_to_token_ratio:.2f} characters)")
    print()
    
    print("Next step:")
    print("  python train.py --config small")
    print()


if __name__ == '__main__':
    main()