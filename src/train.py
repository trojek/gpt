#!/usr/bin/env python3
"""
Training Script for GPT Model
Trains a transformer language model on tokenized text data.
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import GPTModel
from config import get_config, print_config_comparison


class TextDataset(Dataset):
    """Dataset for loading tokenized text data."""
    
    def __init__(self, tokens: np.ndarray, context_length: int):
        self.tokens = tokens
        self.context_length = context_length
    
    def __len__(self) -> int:
        return len(self.tokens) - self.context_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx:idx + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Calculate learning rate with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20
) -> float:
    """Estimate loss on validation set."""
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break
        
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def train(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int = 15000,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 500,
    save_interval: int = 1500,
    log_interval: int = 100,
    checkpoint_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("logs")
):
    """Train the model."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )
    
    model.train()
    train_losses = []
    start_time = time.time()
    
    # Open log file
    log_file = log_dir / "training_log.csv"
    with open(log_file, 'w') as f:
        f.write("step,train_loss,val_loss,lr,time,perplexity\n")
    
    print("="*70)
    print("TRAINING START")
    print("="*70)
    print(f"Device: {device}")
    print(f"Max steps: {max_steps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup steps: {warmup_steps:,}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Context length: {model.max_seq_len}")
    print("="*70)
    print()
    
    # Training loop
    step = 0
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    while step < max_steps:
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Get learning rate
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        train_losses.append(loss.item())
        
        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = np.mean(train_losses[-log_interval:])
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * train_loader.batch_size * x.size(1) / elapsed
            
            print(f"Step {step+1:5d}/{max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")
        
        # Evaluation
        if (step + 1) % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, max_batches=20)
            perplexity = np.exp(val_loss)
            elapsed = time.time() - start_time
            
            # Check if this is the best model so far
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_marker = " 🌟 BEST"
            else:
                best_marker = ""
            
            print(f"\n{'='*70}")
            print(f"Step {step+1:5d} | Val Loss: {val_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f}{best_marker}")
            print(f"{'='*70}\n")
            
            # Save to log
            with open(log_file, 'a') as f:
                f.write(f"{step+1},{avg_loss:.6f},{val_loss:.6f},{lr:.8f},{elapsed:.2f},{perplexity:.4f}\n")
            
            # Save best model
            if is_best:
                best_path = checkpoint_dir / "model_best.pt"
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'perplexity': perplexity,
                }, best_path)
                print(f"✅ Best model saved: {best_path}")
        
        # Save checkpoint
        if (step + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"model_step_{step+1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        step += 1
    
    # Final checkpoint
    final_path = checkpoint_dir / f"model_final_step_{max_steps}.pt"
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final checkpoint: {final_path}")
    print(f"Best model: checkpoints/model_best.pt (Perplexity: {np.exp(best_val_loss):.2f})")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Log file: {log_file}")
    print()
    print("Next: python generate.py --checkpoint checkpoints/model_best.pt --prompt 'Your text'")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train GPT model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default='data/train_tokens.npy',
                        help='Path to training tokens')
    parser.add_argument('--val-data', type=str, default='data/val_tokens.npy',
                        help='Path to validation tokens')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.json',
                        help='Path to vocabulary file')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large'],
                        help='Model configuration')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max-steps', type=int, default=15000,
                        help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Warmup steps')
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    
    # Debug option
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configurations and exit')
    
    args = parser.parse_args()
    
    # List configs and exit
    if args.list_configs:
        print_config_comparison()
        return
    
    # Load tokenizer to get vocab size
    print("Loading tokenizer...")
    try:
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer.load(args.vocab_path)
        vocab_size = tokenizer.vocab_size
        print(f"✓ Character tokenizer loaded")
    except:
        try:
            from tokenizer import SubwordTokenizer
            tokenizer = SubwordTokenizer.load(args.vocab_path)
            vocab_size = tokenizer.vocab_size
            print(f"✓ Subword tokenizer loaded")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Make sure you've run tokenization first: python tokenize.py")
            return
    
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Load data
    print("Loading training data...")
    train_tokens = np.load(args.train_data)
    print(f"Training tokens: {len(train_tokens):,}")
    
    print("Loading validation data...")
    val_tokens = np.load(args.val_data)
    print(f"Validation tokens: {len(val_tokens):,}")
    print()
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")
    print()
    
    # Get model configuration
    print(f"Creating model with '{args.config}' configuration...")
    config = get_config(args.config)
    config.vocab_size = vocab_size
    
    # Create model
    model = GPTModel(**config.to_dict())
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Model size: {model.get_num_params() * 4 / 1024 / 1024:.2f} MB")
    print()
    
    # Create datasets
    context_length = config.max_seq_len
    train_dataset = TextDataset(train_tokens, context_length)
    val_dataset = TextDataset(val_tokens, context_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=Path("checkpoints"),
        log_dir=Path("logs")
    )


if __name__ == '__main__':
    main()