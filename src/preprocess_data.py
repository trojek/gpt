#!/usr/bin/env python3
"""
Data Preprocessing Script for Parrot with Alzheimer's LLM
FIXED VERSION: Removes Wikipedia extraction artifacts like <doc> tags
"""

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Tuple


def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Clean and normalize text data.
    Removes Wikipedia extraction artifacts from WikiExtractor output.
    
    Args:
        text: Raw input text
        lowercase: Whether to convert text to lowercase
        
    Returns:
        Cleaned text string
    """
    print("Cleaning Wikipedia extraction artifacts...")
    
    # Remove <doc> opening tags with all their attributes
    # Example: <doc id="1077157" url="https://..." title="Red Sea crisis">
    text = re.sub(r'<doc\s+[^>]*>', '', text)
    
    # Remove closing </doc> tags
    text = re.sub(r'</doc>', '', text)
    
    # Remove standalone URLs (lines that are just URLs)
    text = re.sub(r'^https?://[^\s]+$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines and lines with just whitespace
    text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)
    
    print("Removing duplicate titles...")
    # Remove duplicate title lines (WikiExtractor puts title twice)
    # The title appears once as article heading, then again as first line
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            cleaned_lines.append('')
            continue
        
        # Skip if this line is identical to previous line (duplicate title)
        if prev_line and line_stripped == prev_line:
            continue
        
        cleaned_lines.append(line)
        prev_line = line_stripped
    
    text = '\n'.join(cleaned_lines)
    
    print("Normalizing Unicode...")
    # Normalize Unicode characters (handle accents, special symbols)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    print("Cleaning whitespace...")
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    print("Removing short lines (noise)...")
    # Remove lines that are too short (likely noise or artifacts)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) == 0 or len(line.strip()) >= 10]
    text = '\n'.join(lines)
    
    # Optional: convert to lowercase
    if lowercase:
        print("Converting to lowercase...")
        text = text.lower()
    
    return text.strip()


def split_data(text: str, train_ratio: float = 0.9) -> Tuple[str, str]:
    """
    Split text into training and validation sets.
    
    Args:
        text: Full text to split
        train_ratio: Proportion of data for training (default: 0.9)
        
    Returns:
        Tuple of (train_text, val_text)
    """
    split_idx = int(len(text) * train_ratio)
    
    # Split at nearest newline to avoid cutting mid-sentence
    split_idx = text.rfind('\n', 0, split_idx)
    
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    return train_text, val_text


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def show_sample(text: str, num_lines: int = 10):
    """Show a sample of the cleaned text."""
    lines = text.split('\n')[:num_lines]
    print("\n" + "="*60)
    print("SAMPLE OF CLEANED TEXT (first 10 lines):")
    print("="*60)
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}: {line[:80]}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw text data for LLM training (removes Wikipedia artifacts)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw input text file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Directory to save processed files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of data for training (rest is validation)'
    )
    parser.add_argument(
        '--lowercase',
        action='store_true',
        help='Convert all text to lowercase'
    )
    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Show sample of cleaned text'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.txt'
    val_path = output_dir / 'val.txt'
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return
    
    print("="*60)
    print("PREPROCESSING - FIXED FOR WIKIPEDIA ARTIFACTS")
    print("="*60)
    print(f"Reading raw data from: {input_path}")
    print(f"Input file size: {get_file_size_mb(input_path):.2f} MB\n")
    
    # Load raw text
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"Raw text length: {len(raw_text):,} characters")
    
    # Count artifacts before cleaning
    doc_tags = len(re.findall(r'<doc\s+[^>]*>', raw_text))
    print(f"Found {doc_tags} <doc> tags to remove")
    
    # Clean text
    print("\n" + "="*60)
    print("CLEANING TEXT")
    print("="*60)
    clean = clean_text(raw_text, lowercase=args.lowercase)
    
    print("\n" + "="*60)
    print(f"Clean text length: {len(clean):,} characters")
    print(f"Removed: {len(raw_text) - len(clean):,} characters ({100*(len(raw_text)-len(clean))/len(raw_text):.1f}%)")
    print("="*60 + "\n")
    
    # Show sample if requested
    if args.show_sample:
        show_sample(clean)
    
    # Split into train/val
    print(f"Splitting data (train ratio: {args.train_ratio})...")
    train_text, val_text = split_data(clean, args.train_ratio)
    
    # Save processed files
    print(f"\nSaving training data to: {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    print(f"Saving validation data to: {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Train set:")
    print(f"  - Size: {get_file_size_mb(train_path):.2f} MB")
    print(f"  - Characters: {len(train_text):,}")
    print(f"\nValidation set:")
    print(f"  - Size: {get_file_size_mb(val_path):.2f} MB")
    print(f"  - Characters: {len(val_text):,}")
    print(f"\nTotal: {get_file_size_mb(train_path) + get_file_size_mb(val_path):.2f} MB")
    print("\n✅ Data is now clean and ready for tokenization!")
    print("\nNext step: python tokenize.py")
    print("="*60)


if __name__ == '__main__':
    main()