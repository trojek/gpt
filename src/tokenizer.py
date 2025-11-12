"""
Subword tokenizer for Parrot with Alzheimer's LLM using Byte-Pair Encoding (BPE).
Can be imported and used in training/inference scripts.
"""

import json
from pathlib import Path
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


class SubwordTokenizer:
    """
    Subword tokenizer using Byte-Pair Encoding (BPE).
    
    This is similar to GPT-2's tokenization and will:
    - Create meaningful subword units (not just characters)
    - Allow for much better context (512 tokens ≈ 350-400 words)
    - Enable the model to learn semantic relationships
    
    Example:
        # Train from text
        tokenizer = SubwordTokenizer.from_text('data/train.txt', vocab_size=20000)
        
        # Encode text
        ids = tokenizer.encode("Hello, world!")
        
        # Decode back
        text = tokenizer.decode(ids)
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize tokenizer with a trained tokenizer.
        
        Args:
            tokenizer: Hugging Face Tokenizer instance
        """
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(ids)
    
    @classmethod
    def from_text(cls, text_path: str, vocab_size: int = 20000, min_frequency: int = 2) -> 'SubwordTokenizer':
        """
        Build tokenizer from text by training BPE.
        
        Args:
            text_path: Path to training text file
            vocab_size: Target vocabulary size (default: 20000)
            min_frequency: Minimum frequency for tokens (default: 2)
            
        Returns:
            SubwordTokenizer instance
        """
        # Initialize a tokenizer with BPE model
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Use byte-level pre-tokenization (like GPT-2)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
            show_progress=True
        )
        
        # Train the tokenizer
        tokenizer.train(files=[text_path], trainer=trainer)
        
        return cls(tokenizer)
    
    @classmethod
    def load(cls, vocab_path: str) -> 'SubwordTokenizer':
        """
        Load tokenizer from saved vocabulary file.
        
        Args:
            vocab_path: Path to tokenizer.json file
            
        Returns:
            SubwordTokenizer instance
        """
        tokenizer = Tokenizer.from_file(vocab_path)
        return cls(tokenizer)
    
    def save(self, vocab_path: str):
        """
        Save tokenizer to JSON file.
        
        Args:
            vocab_path: Path to save tokenizer.json
        """
        self.tokenizer.save(vocab_path)
    
    def __repr__(self) -> str:
        return f"SubwordTokenizer(vocab_size={self.vocab_size})"


# Keep backward compatibility - alias for old code
CharTokenizer = SubwordTokenizer