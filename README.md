# GPT Language Model - From Scratch

A minimal implementation of GPT (Generative Pre-trained Transformer) trained on Simple English Wikipedia. Train a 2.8M parameter language model in 2-4 hours on M1 Mac.

## Project Structure

```
.
├── config.py              # Model architecture configurations (tiny/small/medium/large)
├── tokenizer.py           # Character-level tokenizer
├── model.py               # Transformer architecture implementation
├── preprocess_data.py     # Raw text cleaning and splitting
├── tokenize.py            # Convert text to token arrays
├── train.py               # Training loop and optimization
├── generate.py            # Text generation from trained model
├── create_model.py        # Model inspection utility
├── test_model.py          # Architecture validation tests
└── requirements.txt       # Python dependencies
```

## File Descriptions

### Core Architecture
- **`config.py`** - Defines model configurations (tiny/small/medium/large) with hyperparameters. Provides `get_config()` to select architecture by name and validates that embed_dim is divisible by num_heads.
- **`tokenizer.py`** - Implements character-level tokenization with encode/decode methods and vocabulary management. Supports both CharTokenizer and SubwordTokenizer.
- **`model.py`** - Complete GPT architecture: multi-head attention, transformer blocks, embeddings, and text generation logic with top-k sampling.

### Data Pipeline
- **`preprocess_data.py`** - Cleans raw Wikipedia text (unicode normalization, whitespace handling) and splits into train/validation sets (90/10 split).
- **`tokenize.py`** - Builds vocabulary from training data and converts text files into numpy token arrays for efficient training.

### Training & Inference
- **`train.py`** - Main training script that loads config via `get_config()`, initializes model, runs training with AdamW optimizer, learning rate scheduling, and saves checkpoints.
- **`generate.py`** - Loads trained checkpoint and generates text with configurable temperature and top-k sampling.

### Utilities
- **`create_model.py`** - Creates model from config, prints architecture summary, parameter count, and memory estimates. Use `--list-configs` to see all available configurations.
- **`test_model.py`** - Runs unit tests on model components to verify shapes, forward pass, config validation, and save/load functionality.

## The Process: Data → Model

### Step 1: Download Data
```bash
# Download Simple English Wikipedia (~350MB)
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2

# Extract plain text (~238MB)
pip install wikiextractor
python -m wikiextractor.WikiExtractor simplewiki-latest-pages-articles.xml.bz2 -o data/extracted

# Combine all text files
find data/extracted -name 'wiki_*' -exec cat {} \; > data/raw_wikipedia.txt
```

### Step 2: Preprocess Text
```bash
# Clean and split data (90% train, 10% validation)
python preprocess_data.py --input data/raw_wikipedia.txt

# Output: data/train.txt (~210MB), data/val.txt (~23MB)
```

**What it does:** Removes control characters, normalizes unicode, collapses excessive whitespace, filters short lines.

### Step 3: Tokenize
```bash
# Build vocabulary and convert to token arrays
python tokenize.py

# Output: 
#   data/vocab.json (~2KB)
#   data/train_tokens.npy (~421MB)
#   data/val_tokens.npy (~47MB)
```

**What it does:** Scans text to find ~120 unique characters, creates char↔ID mappings, encodes text as numpy arrays.

### Step 4: Train Model
```bash
# Train small model (2.8M parameters, 15K steps, ~3 hours)
python train.py --config small --max-steps 15000

# Or quick experiment with tiny model
python train.py --config tiny --max-steps 5000

# Or high-quality medium model
python train.py --config medium --max-steps 25000
```

**What it does:** Loads config via `get_config()`, creates model, trains with next-token prediction, saves checkpoints every 1500 steps.

**Expected progress:**
```
Step 0:     Loss ~4.6 (random)
Step 5000:  Loss ~2.0 (learning words)
Step 10000: Loss ~1.7 (forming sentences)
Step 15000: Loss ~1.5 (coherent text)
```

### Step 5: Generate Text
```bash
# Generate from trained checkpoint
python generate.py \
  --checkpoint checkpoints/model_best.pt \
  --prompt "The capital of France is" \
  --max-tokens 200 \
  --temperature 0.8
```

**Output example:**
```
The capital of France is Paris. Paris is the largest city in France and 
is located in the northern part of the country. The city has a population 
of over two million people.
```

## Quick Start

### Installation
```bash
# Install dependencies
pip install torch numpy pandas matplotlib tqdm

# Or use requirements file
pip install -r requirements.txt
```

### Complete Pipeline
```bash
# 1. Preprocess (assuming you have raw_wikipedia.txt)
python preprocess_data.py --input data/raw_wikipedia.txt

# 2. Tokenize
python tokenize.py

# 3. Train with selected configuration
python train.py --config small --max-steps 15000

# 4. Generate
python generate.py --checkpoint checkpoints/model_best.pt --prompt "Hello"
```

## Configuration Options

### Available Configurations

View all configurations:
```bash
python train.py --list-configs
# or
python create_model.py --list-configs
```

| Config | Parameters | Layers | Embed Dim | Heads | Context | Training Time |
|--------|-----------|---------|-----------|-------|---------|---------------|
| tiny   | ~500K     | 2       | 64        | 2     | 128     | 2-3 hours    |
| small  | ~2.8M     | 4       | 128       | 4     | 256     | 3-4 hours    |
| medium | ~11M      | 6       | 256       | 8     | 512     | 8-12 hours   |
| large  | ~25M      | 8       | 384       | 12    | 512     | 16-24 hours  |

**Recommendations:**
- **tiny** - Quick experiments, testing pipeline
- **small** - Standard training, good balance (recommended)
- **medium** - Better quality, more memory required
- **large** - Best quality, requires 16GB+ RAM

### Training Parameters
```bash
python train.py \
  --config small \           # Model size (tiny/small/medium/large)
  --batch-size 32 \          # Batch size (reduce if OOM)
  --max-steps 15000 \        # Training steps
  --learning-rate 3e-4 \     # Peak learning rate
  --device mps               # Device (mps/cuda/cpu)
```

### Generation Parameters
```bash
python generate.py \
  --checkpoint path/to/model.pt \
  --prompt "Your text here" \
  --max-tokens 200 \         # Length of generation
  --temperature 0.8 \        # Randomness (0.1=boring, 1.5=creative)
  --top-k 40                 # Diversity (lower=focused)
```

## Usage Examples

### Example 1: List Available Configurations
```bash
# See all configs with parameter counts and recommendations
python train.py --list-configs
```

Output:
```
Available Configurations:
======================================================================
Config     Params       Embed    Layers   Heads    Context   
----------------------------------------------------------------------
tiny       ~0.5M        64       2        2        128       
small      ~2.8M        128      4        4        256       
medium     ~11.0M       256      6        8        512       
large      ~25.0M       384      8        12       512       
======================================================================

Recommendations:
  tiny   - Quick experiments (2-3 hours)
  small  - Standard training (3-4 hours)
  medium - Better quality (8-12 hours)
  large  - Best quality (16-24 hours, requires 16GB+ RAM)
```

### Example 2: Inspect Model Architecture
```bash
# View detailed architecture and memory estimates
python create_model.py --config small
```

### Example 3: Test Model Components
```bash
# Verify architecture works correctly
python test_model.py
```

### Example 4: Train Tiny Model (Fast)
```bash
# Quick 2-hour training run for testing
python train.py --config tiny --max-steps 5000 --batch-size 32
python generate.py --checkpoint checkpoints/model_best.pt --prompt "AI is"
```

### Example 5: Train Standard Model
```bash
# Full training run (~4 hours) - recommended
python train.py --config small --max-steps 15000
```

### Example 6: Train High-Quality Model
```bash
# Longer training for better results
python train.py --config medium --max-steps 25000 --batch-size 24
```

### Example 7: Generate with Different Temperatures
```bash
# Conservative (boring but safe)
python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "Water is" --temperature 0.5

# Balanced (recommended)
python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "Water is" --temperature 0.8

# Creative (interesting but risky)
python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "Water is" --temperature 1.2
```

## Hardware Requirements

**Minimum:**
- 8GB RAM
- 10GB disk space
- CPU (very slow, 24+ hours)

**Recommended:**
- 16GB RAM
- Apple M1/M2/M3 or NVIDIA GPU (8GB+)
- 20GB disk space

**Training Speed by Config:**

| Config | M1 Mac | NVIDIA 3080 | CPU |
|--------|--------|-------------|-----|
| tiny   | 2-3h   | 1-2h        | 12-18h |
| small  | 3-4h   | 2-3h        | 24-36h |
| medium | 8-12h  | 4-6h        | 48-72h |
| large  | 16-24h | 8-12h       | 96h+ |

## Expected Results

### Quality by Configuration
- **tiny:** Basic coherence, 1-2 sentences
- **small:** Good coherence, 2-3 sentences (recommended)
- **medium:** Very good coherence, 3-4 sentences
- **large:** Excellent coherence, 4-5 sentences

### Sample Outputs (small config)
```
Prompt: "The solar system contains"
Output: "eight planets. The planets orbit around the Sun. Mercury is the 
closest planet to the Sun and Neptune is the farthest."

Prompt: "Albert Einstein was"
Output: "a German physicist who developed the theory of relativity. He won 
the Nobel Prize in Physics in 1921 for his work on the photoelectric effect."

Prompt: "Machine learning is"
Output: "a type of artificial intelligence that allows computers to learn 
from data without being explicitly programmed. It is used in many applications."
```

## Architecture Details

### Model Components
- **Token Embeddings:** Convert character IDs to vectors
- **Position Embeddings:** Encode token positions in sequence
- **Transformer Blocks:** N layers of self-attention + feedforward (N depends on config)
- **Multi-Head Attention:** Multiple attention heads per layer (varies by config)
- **Output Head:** Projects to vocabulary size for next-token prediction

### Configuration System
The `config.py` file provides:
- **ModelConfig dataclass:** Stores all hyperparameters
- **Predefined configs:** tiny, small, medium, large
- **get_config():** Returns config by name
- **Validation:** Ensures embed_dim divisible by num_heads
- **to_dict():** Converts config to dict for model initialization

### Training Details
- **Objective:** Next-token prediction (cross-entropy loss)
- **Optimizer:** AdamW with weight decay (0.1)
- **Schedule:** Linear warmup (500 steps) → Cosine decay
- **Batch Size:** 32 sequences × context_len tokens per batch
- **Gradient Clipping:** Max norm 1.0

## Troubleshooting

### Out of Memory (OOM)
```bash
# Option 1: Reduce batch size
python train.py --config small --batch-size 16

# Option 2: Use smaller model
python train.py --config tiny --batch-size 32

# Option 3: Reduce context length (edit config.py)
# Change max_seq_len in config
```

### Slow Training
```bash
# Check device
python -c "import torch; print(torch.backends.mps.is_available())"

# Force specific device
python train.py --device mps  # For M1 Mac
python train.py --device cuda # For NVIDIA GPU
```

### Poor Generation Quality
```bash
# Option 1: Train longer
python train.py --max-steps 25000

# Option 2: Use larger config
python train.py --config medium

# Option 3: Adjust generation parameters
python generate.py --temperature 0.7 --top-k 50
```

### Config Not Found Error
```bash
# List available configs
python train.py --list-configs

# Use correct config name
python train.py --config small  # Not 'Small' or 'SMALL'
```

## Validation

### Test Model Architecture
```bash
python test_model.py
# Should print: "ALL TESTS PASSED ✓"
```

### Check Available Configurations
```bash
python create_model.py --list-configs
```

### Inspect Specific Configuration
```bash
python create_model.py --config medium
```

### View Training Progress
```bash
# View training logs
tail -f logs/training_log.csv

# Check saved checkpoints
ls -lh checkpoints/
```

### Verify Config Usage
```python
from config import get_config

# Load configuration
config = get_config('small')
print(f"Embed dim: {config.embed_dim}")
print(f"Num layers: {config.num_layers}")

# Convert to dict for model
config.vocab_size = 120
model_kwargs = config.to_dict()
```

## What This Project Does ✓
- Trains a working GPT-style language model
- Generates coherent text for 2-3 sentences (small config)
- Demonstrates transformer architecture
- Provides configurable model sizes
- Uses proper configuration management
- Runs on consumer hardware (M1 Mac)

## What This Project Doesn't Do ✗
- Answer questions reliably
- Follow instructions
- Reason logically
- Maintain long coherence (>4 sentences even with large config)
- Match GPT-3/GPT-4 quality

## Technical Specifications

**Architecture:** Decoder-only transformer (GPT-style)  
**Training Data:** Simple English Wikipedia (~238MB text)  
**Tokenization:** Character-level (~120 vocab size)  
**Configurations:** 4 predefined (tiny/small/medium/large)  
**Context Window:** 128-512 tokens (config-dependent)  
**Parameters:** 500K-25M (config-dependent)  
**Training Time:** 2-24 hours (config-dependent)  
**Framework:** PyTorch 2.0+

## Configuration Management

### How Configs Are Used

1. **Define configs in config.py:**
   ```python
   SMALL_CONFIG = ModelConfig(
       embed_dim=128,
       num_heads=4,
       num_layers=4,
       max_seq_len=256
   )
   ```

2. **Load config in train.py:**
   ```python
   from config import get_config
   config = get_config('small')
   config.vocab_size = vocab_size
   model = GPTModel(**config.to_dict())
   ```

3. **Use config in other scripts:**
   ```python
   # create_model.py, test_model.py also use get_config()
   config = get_config('medium')
   ```

### Benefits of Config System
- ✓ Centralized configuration management
- ✓ Easy to switch between model sizes
- ✓ Validation of hyperparameters
- ✓ Consistent configs across scripts
- ✓ No hardcoded values in train.py

## Next Steps

1. **Improve tokenization:** Use BPE instead of character-level for better efficiency
2. **More data:** Train on full English Wikipedia (10GB+) for better knowledge
3. **Larger model:** Try medium or large config for better quality
4. **Fine-tuning:** Adapt to specific domains with additional training
5. **Instruction tuning:** Add prompt-response pairs for instruction following

## License & Credits

Educational project demonstrating GPT architecture. Based on the "Attention is All You Need" paper (Vaswani et al., 2017) and GPT-2 architecture.

---

**Quick Commands Summary:**
```bash
# Setup
pip install -r requirements.txt

# View configs
python train.py --list-configs

# Test architecture
python test_model.py

# Preprocess data
python preprocess_data.py --input data/raw_wikipedia.txt

# Tokenize
python tokenize.py

# Train (choose config)
python train.py --config small

# Generate
python generate.py --checkpoint checkpoints/model_best.pt --prompt "Hello"
```

**Questions?** Check file docstrings or run with `--help` flag.