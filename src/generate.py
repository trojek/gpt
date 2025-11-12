#!/usr/bin/env python3
"""
Text Generation Script - Step 6
Generate text from a trained model.
"""

import argparse
import json
import torch
from pathlib import Path

from tokenizer import SubwordTokenizer 
from model import GPTModel


def load_model(checkpoint_path: str, vocab_size: int, device: torch.device) -> GPTModel:
    """Load trained model from checkpoint."""
    # Create model with same config as training
    model = GPTModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=10,
        max_seq_len=512
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_text(
    model: GPTModel,
    tokenizer: SubwordTokenizer, 
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.1,
    top_k: int = 40,
    device: torch.device = torch.device('cpu')
) -> str:
    """
    Generate text from prompt.
    
    Args:
        model: Trained GPT model
        tokenizer: Subword tokenizer
        prompt: Starting text
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k tokens
        device: Device to run on
        
    Returns:
        Generated text (prompt + new tokens)
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_tokens = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description='Generate text from trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Starting text prompt')
    
    # Optional arguments
    parser.add_argument('--vocab', type=str, default='data/tokenizer.json',
                        help='Path to tokenizer file')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature (0.1=conservative, 1.5=creative)')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Sample from top k tokens')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.vocab).exists():
        print(f"Error: Tokenizer not found: {args.vocab}")
        return
    
    # Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("="*60)
    print("TEXT GENERATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SubwordTokenizer.load(args.vocab) 
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, tokenizer.vocab_size, device)
    print(f"Model loaded ({model.get_num_params():,} parameters)")
    print()
    
    # Generate
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print("="*60)
    print()
    
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(generated)
    print()
    print("="*60)
    print(f"Generated {args.max_tokens} tokens")
    print("="*60)


if __name__ == '__main__':
    main()