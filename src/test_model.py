#!/usr/bin/env python3
"""
Test script for model architecture.
Verifies that all components work correctly.
"""

import torch
from model import GPTModel, MultiHeadAttention, FeedForward, TransformerBlock
from config import ModelConfig, get_config


def test_multihead_attention():
    """Test multi-head attention component."""
    print("Testing MultiHeadAttention...")
    
    attn = MultiHeadAttention(embed_dim=128, num_heads=4, dropout=0.1)
    x = torch.randn(2, 10, 128)
    
    output = attn(x)
    assert output.shape == (2, 10, 128), f"Expected shape (2, 10, 128), got {output.shape}"
    
    print("  ✓ MultiHeadAttention working correctly")


def test_feedforward():
    """Test feed-forward component."""
    print("Testing FeedForward...")
    
    ff = FeedForward(embed_dim=128, ff_dim=512, dropout=0.1)
    x = torch.randn(2, 10, 128)
    
    output = ff(x)
    assert output.shape == (2, 10, 128), f"Expected shape (2, 10, 128), got {output.shape}"
    
    print("  ✓ FeedForward working correctly")


def test_transformer_block():
    """Test transformer block component."""
    print("Testing TransformerBlock...")
    
    block = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512, dropout=0.1)
    x = torch.randn(2, 10, 128)
    
    output = block(x)
    assert output.shape == (2, 10, 128), f"Expected shape (2, 10, 128), got {output.shape}"
    
    print("  ✓ TransformerBlock working correctly")


def test_gpt_model():
    """Test complete GPT model."""
    print("Testing GPTModel...")
    
    model = GPTModel(vocab_size=100, embed_dim=128, num_heads=4, num_layers=4)
    
    # Test forward pass without targets
    input_ids = torch.randint(0, 100, (2, 10))
    logits, loss = model(input_ids)
    
    assert logits.shape == (2, 10, 100), f"Expected shape (2, 10, 100), got {logits.shape}"
    assert loss is None, "Loss should be None when no targets provided"
    
    # Test forward pass with targets
    target_ids = torch.randint(0, 100, (2, 10))
    logits, loss = model(input_ids, targets=target_ids)
    
    assert loss is not None, "Loss should not be None when targets provided"
    assert isinstance(loss.item(), float), "Loss should be a scalar"
    
    print(f"  ✓ GPTModel working correctly")
    print(f"  ✓ Forward pass successful (loss: {loss.item():.4f})")


def test_generation():
    """Test text generation."""
    print("Testing text generation...")
    
    model = GPTModel(vocab_size=100, embed_dim=128, num_heads=4, num_layers=4)
    
    # Generate text
    input_ids = torch.randint(0, 100, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=10, temperature=1.0, top_k=40)
    
    assert generated.shape == (1, 15), f"Expected shape (1, 15), got {generated.shape}"
    
    print(f"  ✓ Generation working correctly")
    print(f"  ✓ Generated {generated.shape[1]} tokens")


def test_model_configs():
    """Test predefined model configurations."""
    print("Testing model configurations...")
    
    configs = ['tiny', 'small', 'medium', 'large']
    
    for config_name in configs:
        config = get_config(config_name)
        config.vocab_size = 100
        
        # Verify config values
        assert config.embed_dim % config.num_heads == 0, \
            f"embed_dim must be divisible by num_heads in {config_name}"
        
        # Create model with config
        model = GPTModel(**config.to_dict())
        n_params = model.get_num_params()
        
        print(f"  ✓ {config_name.capitalize():6s} config: {n_params:>8,} parameters "
              f"(layers={config.num_layers}, heads={config.num_heads})")


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Valid config
    try:
        config = ModelConfig(embed_dim=128, num_heads=4)
        print("  ✓ Valid config accepted")
    except:
        raise AssertionError("Valid config was rejected")
    
    # Invalid config (embed_dim not divisible by num_heads)
    try:
        config = ModelConfig(embed_dim=100, num_heads=3)
        raise AssertionError("Invalid config was accepted")
    except AssertionError as e:
        if "divisible" in str(e):
            print("  ✓ Invalid config rejected")
        else:
            raise


def test_save_load():
    """Test saving and loading model."""
    print("Testing save/load...")
    
    # Create and save model
    config = get_config('tiny')
    config.vocab_size = 100
    model1 = GPTModel(**config.to_dict())
    torch.save(model1.state_dict(), '/tmp/test_model.pt')
    
    # Load model
    model2 = GPTModel(**config.to_dict())
    model2.load_state_dict(torch.load('/tmp/test_model.pt'))
    
    # Verify weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights don't match saved weights"
    
    print("  ✓ Save/load working correctly")


def test_config_to_dict():
    """Test config to_dict method."""
    print("Testing config.to_dict()...")
    
    config = get_config('small')
    config.vocab_size = 100
    
    config_dict = config.to_dict()
    
    # Verify all required keys are present
    required_keys = ['vocab_size', 'embed_dim', 'num_heads', 'num_layers', 
                     'max_seq_len', 'ff_dim', 'dropout']
    for key in required_keys:
        assert key in config_dict, f"Missing key: {key}"
    
    # Verify model can be created from dict
    model = GPTModel(**config_dict)
    assert model.vocab_size == 100
    assert model.embed_dim == config.embed_dim
    
    print("  ✓ config.to_dict() working correctly")


def main():
    """Run all tests."""
    print("="*60)
    print("MODEL ARCHITECTURE TESTS")
    print("="*60)
    print()
    
    torch.manual_seed(42)
    
    try:
        test_multihead_attention()
        print()
        
        test_feedforward()
        print()
        
        test_transformer_block()
        print()
        
        test_gpt_model()
        print()
        
        test_generation()
        print()
        
        test_model_configs()
        print()
        
        test_config_validation()
        print()
        
        test_config_to_dict()
        print()
        
        test_save_load()
        print()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        print("Your model architecture is working correctly!")
        print("Next steps:")
        print("  1. python tokenize.py              # Prepare data")
        print("  2. python train.py --config small  # Train model")
        print("  3. python generate.py              # Generate text")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()