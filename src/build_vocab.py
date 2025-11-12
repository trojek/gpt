#!/usr/bin/env python3
"""
Tokenization Script for Parrot with Alzheimer's LLM
Builds subword vocabulary using BPE and encodes text to token IDs.
"""

import argparse
import numpy as np
from pathlib import Path

from tokenizer import SubwordTokenizer


def build_vocabulary(train_path: Path, vocab_size: int = 20000, min_freq: int = 2) -> SubwordTokenizer:
    """
    Build vocabulary from training data using BPE.
    
    Args:
        train_path: Path to training text file
        vocab_size: Target vocabulary size
        min_freq: Minimum token frequency
        
    Returns:
        SubwordTokenizer instance
    """
    print(f"Reading training data from: {train_path}")
    
    # Check file size
    file_size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    print(f"Training BPE tokenizer with vocab_size={vocab_size:,}...")
    print("This may take a few minutes...")
    tokenizer = SubwordTokenizer.from_text(str(train_path), vocab_size=vocab_size, min_frequency=min_freq)
    
    print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")
    
    return tokenizer


def encode_file(tokenizer: SubwordTokenizer, file_path: Path) -> np.ndarray:
    """
    Encode a text file to token IDs.
    
    Args:
        tokenizer: SubwordTokenizer instance
        file_path: Path to text file
        
    Returns:
        Numpy array of token IDs
    """
    print(f"Encoding: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    token_ids = tokenizer.encode(text)
    
    print(f"  Encoded {len(text):,} chars -> {len(token_ids):,} tokens")
    print(f"  Compression ratio: {len(text) / len(token_ids):.2f} chars/token")
    
    return np.array(token_ids, dtype=np.uint16)


def save_vocabulary(tokenizer: SubwordTokenizer, output_path: Path):
    """
    Save vocabulary to JSON file.
    
    Args:
        tokenizer: SubwordTokenizer instance
        output_path: Path to save vocabulary
    """
    tokenizer.save(str(output_path))
    print(f"Vocabulary saved to: {output_path}")


def get_array_size_mb(array: np.ndarray) -> float:
    """Calculate numpy array size in megabytes."""
    return array.nbytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description='Build subword vocabulary and tokenize text data using BPE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--train',
        type=str,
        default='data/train.txt',
        help='Path to training text file'
    )
    parser.add_argument(
        '--val',
        type=str,
        default='data/val.txt',
        help='Path to validation text file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save tokenized files'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=20000,
        help='Target vocabulary size (recommended: 15000-30000)'
    )
    parser.add_argument(
        '--min-freq',
        type=int,
        default=2,
        help='Minimum token frequency to include in vocabulary'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    train_path = Path(args.train)
    val_path = Path(args.val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / 'tokenizer.json' 
    train_tokens_path = output_dir / 'train_tokens.npy'
    val_tokens_path = output_dir / 'val_tokens.npy'
    
    # Validate input files
    if not train_path.exists():
        print(f"Error: Training file '{train_path}' not found")
        return
    
    if not val_path.exists():
        print(f"Error: Validation file '{val_path}' not found")
        return
    
    print("="*60)
    print("TOKENIZATION - SUBWORD LEVEL (BPE)")
    print("="*60)
    print()
    
    # Build vocabulary from training data
    tokenizer = build_vocabulary(train_path, vocab_size=args.vocab_size, min_freq=args.min_freq)
    print()
    
    # Display vocabulary info
    print("Tokenization method: Byte-Pair Encoding (BPE)")
    print("This is similar to GPT-2 and will:")
    print("  - Create meaningful subword units")
    print("  - Enable ~3-4x better context than character-level")
    print("  - Allow model to learn semantic relationships")
    print()
    
    # Encode training data
    train_tokens = encode_file(tokenizer, train_path)
    print()
    
    # Encode validation data
    val_tokens = encode_file(tokenizer, val_path)
    print()
    
    # Save vocabulary
    save_vocabulary(tokenizer, vocab_path)
    print()
    
    # Save tokenized data
    print(f"Saving training tokens to: {train_tokens_path}")
    np.save(train_tokens_path, train_tokens)
    
    print(f"Saving validation tokens to: {val_tokens_path}")
    np.save(val_tokens_path, val_tokens)
    print()
    
    # Print summary
    print("="*60)
    print("TOKENIZATION COMPLETE")
    print("="*60)
    print(f"Vocabulary:")
    print(f"  - Size: {tokenizer.vocab_size:,} tokens")
    print(f"  - File: {vocab_path}")
    print(f"\nTraining tokens:")
    print(f"  - Count: {len(train_tokens):,}")
    print(f"  - Size: {get_array_size_mb(train_tokens):.2f} MB")
    print(f"  - File: {train_tokens_path}")
    print(f"\nValidation tokens:")
    print(f"  - Count: {len(val_tokens):,}")
    print(f"  - Size: {get_array_size_mb(val_tokens):.2f} MB")
    print(f"  - File: {val_tokens_path}")
    print()
    
    # Test encode/decode
    test_text = "The capital of France is Paris. It is a beautiful city."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print("Tokenizer test:")
    print(f"  Original:  {test_text}")
    print(f"  Tokens:    {len(encoded)} tokens (vs {len(test_text)} characters)")
    print(f"  Token IDs: {encoded[:10]}..." if len(encoded) > 10 else f"  Token IDs: {encoded}")
    print(f"  Decoded:   {decoded}")
    print(f"  Status:    {'✓ PASS' if test_text == decoded else '✗ FAIL'}")
    print()
    
    print("IMPORTANT NOTES:")
    print("  - You need to update your training script to load 'tokenizer.json'")
    print("    instead of 'vocab.json'")
    print(f"  - Your effective context is now ~{int(512 * 0.7)} words (was ~100 words)")
    print("  - Expected improvement: Much better fact learning and coherence")


if __name__ == '__main__':
    main()