#!/usr/bin/env python3
"""
Subword tokenizer using Byte Pair Encoding (BPE).
Similar to tokenization used in GPT models.
"""

import json
from typing import Dict, List
from collections import Counter
import re


class SubwordTokenizer:
    """
    Subword tokenizer using Byte Pair Encoding (BPE).
    More efficient than character-level tokenization.
    """
    
    def __init__(self, vocab: Dict[str, int], merges: List[tuple]):
        """
        Initialize tokenizer with vocabulary and merge rules.
        
        Args:
            vocab: Dictionary mapping tokens to IDs
            merges: List of merge rules (tuple of token pairs)
        """
        self.vocab = vocab
        self.merges = merges
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        
        # Create merge dictionary for faster lookup
        self.merge_dict = {pair: i for i, pair in enumerate(merges)}
    
    @staticmethod
    def train(text: str, vocab_size: int = 5000, min_frequency: int = 2) -> 'SubwordTokenizer':
        """
        Train BPE tokenizer on text.
        
        Args:
            text: Training text
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a merge
            
        Returns:
            Trained SubwordTokenizer
        """
        print(f"Training BPE tokenizer with target vocab_size={vocab_size}...")
        
        # Split text into words (simple whitespace + punctuation splitting)
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Initialize vocabulary with individual characters
        chars = set()
        for word in words:
            chars.update(list(word))
        
        vocab = {char: i for i, char in enumerate(sorted(chars))}
        next_id = len(vocab)
        
        # Convert words to character sequences
        word_freqs = Counter(words)
        splits = {word: list(word) for word in word_freqs.keys()}
        
        merges = []
        
        print(f"Starting vocabulary size: {len(vocab)}")
        print(f"Target vocabulary size: {vocab_size}")
        
        # Perform merges until we reach target vocab size
        while len(vocab) < vocab_size:
            # Count all adjacent pairs
            pairs = Counter()
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < min_frequency:
                break
            
            # Merge the best pair
            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = next_id
            next_id += 1
            
            # Update splits
            for word in word_freqs.keys():
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split
            
            if len(vocab) % 100 == 0:
                print(f"  Vocabulary size: {len(vocab)}/{vocab_size}")
        
        print(f"✓ Training complete. Final vocabulary size: {len(vocab)}")
        print(f"✓ Number of merges: {len(merges)}")
        
        return SubwordTokenizer(vocab, merges)
    
    def _split_word(self, word: str) -> List[str]:
        """Split word into tokens using learned merges."""
        tokens = list(word)
        
        while len(tokens) > 1:
            # Find the pair with lowest merge index
            pairs = [(i, (tokens[i], tokens[i + 1])) 
                    for i in range(len(tokens) - 1)]
            
            # Get pairs that exist in merge dict
            valid_pairs = [(i, pair) for i, pair in pairs if pair in self.merge_dict]
            
            if not valid_pairs:
                break
            
            # Find pair with lowest merge index (merged earliest in training)
            i, pair = min(valid_pairs, key=lambda x: self.merge_dict[x[1]])
            
            # Merge the pair
            new_tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i + 2:]
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Split text into words
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Tokenize each word
        token_ids = []
        for word in words:
            tokens = self._split_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Fallback: use first char if token not in vocab
                    if len(token) > 0 and token[0] in self.vocab:
                        token_ids.append(self.vocab[token[0]])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = [self.id_to_token.get(id, '') for id in token_ids]
        text = ''.join(tokens)
        return text
    
    def save(self, path: str):
        """
        Save tokenizer to JSON file.
        
        Args:
            path: Path to save tokenizer (typically 'data/vocab.json')
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'type': 'subword',
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load(path: str) -> 'SubwordTokenizer':
        """
        Load tokenizer from JSON file.
        
        Args:
            path: Path to tokenizer file (typically 'data/vocab.json')
            
        Returns:
            SubwordTokenizer instance
            
        Raises:
            FileNotFoundError: If tokenizer file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tokenizer file not found at {path}\n"
                f"Please run tokenization first: python tokenize.py"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid tokenizer file format: {e}")
        
        # Validate format
        if 'vocab' not in data or 'merges' not in data:
            raise ValueError("Tokenizer file missing required fields")
        
        vocab = data['vocab']
        merges = [tuple(merge) for merge in data['merges']]
        
        return SubwordTokenizer(vocab, merges)
    
    def __repr__(self) -> str:
        return f"SubwordTokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"