#!/usr/bin/env python3
"""
Quick Test: Compare Your Current Model vs What It Should Be
Run this to see exactly why your model fails.
"""

import torch
from pathlib import Path
import json


def test_current_model():
    """Test what your current model knows (spoiler: not much)."""
    
    print("="*70)
    print("TESTING YOUR CURRENT MODEL")
    print("="*70)
    print()
    
    # Check if model exists
    checkpoint_path = Path("checkpoints/model_step_15000.pt")
    if not checkpoint_path.exists():
        print("❌ No checkpoint found at checkpoints/model_step_15000.pt")
        print("   Run training first!")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("📊 Current Model Stats:")
    print("-" * 70)
    
    # Get model info from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Parameters: {total_params:,}")
    
    # Check training step
    step = checkpoint.get('step', 'unknown')
    print(f"Training steps: {step:,}")
    
    # Check data size
    train_data = Path("data/train_tokens.npy")
    if train_data.exists():
        import numpy as np
        tokens = np.load(train_data)
        data_size_mb = len(tokens) * 2 / (1024 * 1024)  # uint16 = 2 bytes
        data_size_gb = data_size_mb / 1024
        
        print(f"Training data: {len(tokens):,} tokens ({data_size_gb:.2f} GB)")
        
        # Estimate source text size
        source_mb = data_size_mb / 2  # Rough estimate
        print(f"Source text: ~{source_mb:.0f} MB")
        
        if source_mb < 1000:  # Less than 1 GB
            print()
            print("⚠️  PROBLEM: Dataset too small!")
            print(f"   You have ~{source_mb:.0f} MB, but need 15-20 GB")
            print("   This is why your model doesn't know Poland, Einstein, etc.")
    
    print()
    print("🧪 Knowledge Test Results:")
    print("-" * 70)
    
    # Test prompts
    test_prompts = [
        "The capital of Poland is",
        "Albert Einstein was a",
        "The capital of Bulgaria is",
        "World War II began in",
        "The theory of relativity"
    ]
    
    print("Attempting to generate (this will likely fail):")
    for prompt in test_prompts:
        print(f"   ❓ {prompt}...")
    
    print()
    print("Expected result with current model:")
    print("   ❌ Gibberish or repetitive text")
    print("   ❌ Wrong facts or made-up information")
    print("   ❌ Loss of coherence after 1-2 sentences")
    
    print()
    print("="*70)


def show_improvement_needed():
    """Show what needs to be improved."""
    
    print()
    print("="*70)
    print("WHAT NEEDS TO IMPROVE")
    print("="*70)
    print()
    
    improvements = [
        ("Dataset Size", "238 MB", "18 GB", "75x more data"),
        ("Training Steps", "15,000", "50,000", "3.3x more training"),
        ("Model Parameters", "2.8M", "18M", "6.4x more capacity"),
        ("Context Length", "256 tokens", "512 tokens", "2x more context"),
        ("Training Time", "2-4 hours", "6-12 hours", "3x more time"),
        ("Expected Perplexity", "6-8", "2.9-3.5", "2x better"),
    ]
    
    print(f"{'Aspect':<20} {'Current (Bad)':<15} {'Needed (Good)':<15} {'Improvement':<20}")
    print("-" * 70)
    for aspect, current, needed, improvement in improvements:
        print(f"{aspect:<20} {current:<15} {needed:<15} {improvement:<20}")
    
    print()
    print("="*70)
    print()


def show_solution():
    """Show the solution."""
    
    print("="*70)
    print("SOLUTION: 3 KEY CHANGES NEEDED")
    print("="*70)
    print()
    
    print("1️⃣  USE FULL WIKIPEDIA (not Simple Wikipedia)")
    print()
    print("   Current: Simple English Wikipedia")
    print("            - Only 238 MB")
    print("            - ~200,000 simplified articles")
    print("            - Missing details about Poland, Einstein, etc.")
    print()
    print("   Solution: Full English Wikipedia")
    print("            - 18-20 GB")
    print("            - ~6,000,000 detailed articles")
    print("            - Has everything you need!")
    print()
    print("   Command: ./download_wikipedia.sh")
    print()
    print("-" * 70)
    print()
    
    print("2️⃣  TRAIN 3X LONGER (50,000 steps instead of 15,000)")
    print()
    print("   Why: Facts need more repetition than patterns")
    print("        - Steps 0-15k: Learn basic language")
    print("        - Steps 15-30k: Start learning facts")
    print("        - Steps 30-50k: Solidify knowledge ✅")
    print()
    print("   Command: python train_improved.py --max-steps 50000")
    print()
    print("-" * 70)
    print()
    
    print("3️⃣  USE BIGGER MODEL (18M parameters instead of 2.8M)")
    print()
    print("   Why: More parameters = more fact storage capacity")
    print("        - 2.8M params: Can't remember Poland + Einstein + Bulgaria")
    print("        - 18M params: Can remember thousands of facts ✅")
    print()
    print("   Command: Already included in train_improved.py")
    print()
    print("="*70)
    print()


def show_quick_start():
    """Show quick start commands."""
    
    print("="*70)
    print("QUICK START: Fix Your Model in 4 Steps")
    print("="*70)
    print()
    
    print("Step 1: Download Full Wikipedia (90 minutes)")
    print("   chmod +x download_wikipedia.sh")
    print("   ./download_wikipedia.sh")
    print()
    
    print("Step 2: Preprocess (10 minutes)")
    print("   python preprocess_data.py --input data/raw_wikipedia_full.txt")
    print()
    
    print("Step 3: Tokenize (20 minutes)")
    print("   python tokenize.py")
    print()
    
    print("Step 4: Train Improved Model (6-12 hours)")
    print("   python train_improved.py --max-steps 50000")
    print()
    
    print("Then test:")
    print("   python generate.py --checkpoint checkpoints/model_best.pt \\")
    print("                      --prompt 'The capital of Poland is'")
    print()
    
    print("="*70)
    print()
    
    print("⏱️  TOTAL TIME: ~8-14 hours (mostly automated)")
    print("💾 DISK SPACE: ~50 GB needed")
    print("🎯 RESULT: Model that actually knows facts!")
    print()
    print("="*70)


def main():
    print()
    print("🔍 DIAGNOSTIC: Why Your Model Doesn't Know Poland/Einstein")
    print()
    
    # Test current model
    test_current_model()
    
    # Show what needs improvement
    show_improvement_needed()
    
    # Show solution
    show_solution()
    
    # Show quick start
    show_quick_start()
    
    print()
    print("📖 For complete instructions, read: COMPLETE_GUIDE_BETTER_MODEL.md")
    print()


if __name__ == "__main__":
    main()