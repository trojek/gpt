#!/usr/bin/env python3
"""
Text generation script.
Loads trained model and generates text from prompts.
"""

import argparse
import sys
import torch

from model import GPTModel
from tokenizer import SubwordTokenizer
from constants import VOCAB_FILE, CHECKPOINT_DIR


def generate_text(
    model: GPTModel,
    tokenizer: SubwordTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = torch.device('cpu')
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device to run on
        
    Returns:
        Generated text including prompt
    """
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_tokens = generated_ids[0].cpu().numpy().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained GPT model')
    
    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(CHECKPOINT_DIR / 'model_best.pt'),
        help='Path to model checkpoint'
    )
    
    # Generation arguments
    parser.add_argument(
        '--prompt',
        type=str,
        default='The capital of France is',
        help='Starting prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0.1=conservative, 1.5=creative)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TEXT GENERATION")
    print("="*70)
    print()
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print()
    
    # Load tokenizer
    print(f"Loading BPE tokenizer from: {VOCAB_FILE}")
    try:
        tokenizer = SubwordTokenizer.load(str(VOCAB_FILE))
        print(f"✓ Tokenizer loaded")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Merge rules: {len(tokenizer.merges)}")
    except FileNotFoundError:
        print(f"Error: Tokenizer not found at {VOCAB_FILE}")
        print("Please run tokenization first:")
        print("  python tokenize.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    print()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print(f"✓ Checkpoint loaded (step: {checkpoint.get('step', 'unknown')})")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first:")
        print("  python train.py --config small")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print(f"✓ Config loaded from checkpoint")
    else:
        print("Warning: No config found in checkpoint, using default 'small' config")
        from config import get_config
        config = get_config('small')
        config.vocab_size = tokenizer.vocab_size
        config_dict = config.to_dict()
    
    # Validate vocab size
    checkpoint_vocab_size = checkpoint.get('vocab_size', None)
    if checkpoint_vocab_size and checkpoint_vocab_size != tokenizer.vocab_size:
        print(f"\n⚠️  Warning: Vocab size mismatch!")
        print(f"  Checkpoint vocab_size: {checkpoint_vocab_size}")
        print(f"  Tokenizer vocab_size: {tokenizer.vocab_size}")
        print(f"  This may cause incorrect generation.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print()
    
    # Create model
    print("Creating model...")
    model = GPTModel(**config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded ({model.get_num_params():,} parameters)")
    print()
    
    # Display settings
    print("Generation settings:")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print()
    
    # Generate
    print("Generating...")
    print("="*70)
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(generated_text)
    print("="*70)
    print()
    print("Generation complete!")
    print()
    print("Tips:")
    print("  - Lower temperature (0.3-0.5) for more conservative output")
    print("  - Higher temperature (1.0-1.5) for more creative output")
    print("  - Lower top-k (10-20) for more focused output")
    print("  - Higher top-k (50-100) for more diverse output")


if __name__ == '__main__':
    main()