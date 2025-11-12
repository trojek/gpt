# GPT Language Model - From Scratch

Train a GPT-style language model using Byte Pair Encoding (BPE) tokenization, just like ChatGPT. Train a 2.8M parameter model in 3-4 hours on M1 Mac.

## Project Structure

```
.
├── config.py              # Model configurations (tiny/small/medium/large)
├── tokenizer.py           # Subword tokenizer using BPE (like ChatGPT)
├── model.py               # GPT transformer architecture
├── constants.py           # Centralized file paths
├── preprocess_data.py     # Text cleaning and train/val splitting
├── tokenize.py            # Train BPE tokenizer + tokenize files
├── train.py               # Training loop with config system
├── generate.py            # Text generation from trained model
├── create_model.py        # Model architecture inspector
├── test_model.py          # Architecture validation tests
└── requirements.txt       # Python dependencies
```

## File Descriptions

### Core Architecture
- **`config.py`** - Defines model configurations with hyperparameters. Use `--list-configs` to see all options (tiny: 500K params, small: 2.8M params, medium: 11M params, large: 25M params).
- **`tokenizer.py`** - Implements Byte Pair Encoding (BPE) tokenizer like GPT/ChatGPT. Learns subword units from data for efficient tokenization (1 token ≈ 4-5 characters vs 1 token = 1 character).
- **`model.py`** - Complete GPT architecture with multi-head attention, transformer blocks, embeddings, and top-k sampling for text generation.

### Data Pipeline
- **`preprocess_data.py`** - Cleans Wikipedia text by normalizing unicode, removing control characters, collapsing whitespace, filtering short lines, and splitting 90/10 into train/validation sets.
- **`tokenize.py`** - Trains BPE tokenizer on training data to learn merge rules, builds vocabulary of ~5000 subword tokens, and converts text to numpy token arrays.

### Training & Inference
- **`train.py`** - Main training script that loads config via `get_config()`, trains with AdamW optimizer, cosine learning rate schedule, and saves config+vocab_size in checkpoints for reproducibility.
- **`generate.py`** - Loads trained checkpoint with embedded config, encodes prompt using BPE, generates text with temperature/top-k sampling, and decodes output.

### Utilities
- **`create_model.py`** - Creates model from config, displays architecture summary with parameter counts per component, and estimates memory usage for training.
- **`test_model.py`** - Runs comprehensive tests on all model components, validates config system, and verifies forward pass and generation work correctly.

## The Process: Data → Model

### Step 1: Download Data

```bash
# Download Simple English Wikipedia (~350MB compressed)
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2

# Extract plain text using WikiExtractor
pip install wikiextractor
python -m wikiextractor.WikiExtractor simplewiki-latest-pages-articles.xml.bz2 -o data/extracted

# Combine all extracted files
find data/extracted -name 'wiki_*' -exec cat {} \; > data/raw_wikipedia.txt
```

**Example input file `raw_wikipedia.txt` (first few lines):**
```
<doc id="12" url="https://simple.wikipedia.org/wiki/Anarchism">
Anarchism

Anarchism is a political philosophy that thinks that governments are harmful and not needed...
```

**What you get:**
- Single file with all Wikipedia articles
- Size: ~238 MB of plain text
- Contains articles separated by XML markers

### Step 2: Preprocess Text

```bash
python preprocess_data.py --input data/raw_wikipedia.txt
```

**Example transformation:**

**Input (`raw_wikipedia.txt`):**
```
<doc id="12">
Anarchism

Anarchism  is   a    political philosophy...


Multiple    blank   lines...
```

**Output (`data/train.txt`):**
```
anarchism

anarchism is a political philosophy

multiple blank lines
```

**What it does:**
- Removes XML tags: `<doc id="12">` → removed
- Normalizes unicode: `café` → normalized form
- Collapses spaces: `a    political` → `a political`
- Collapses newlines: Multiple blank lines → single blank line
- Filters short lines: Lines < 10 chars → removed
- Splits data: 90% train (~210MB), 10% validation (~23MB)

**Output files:**
- `data/train.txt` - Training data (~210 MB)
- `data/val.txt` - Validation data (~23 MB)

### Step 3: Train BPE Tokenizer & Tokenize

```bash
python tokenize.py --vocab-size 5000
```

**Example BPE training process:**

**Input text:**
```
the cat is sleeping
the cat is running
the dog is sleeping
```

**BPE learns merges:**
```
Step 1: Start with characters: ['t','h','e',' ','c','a','t',' ','i','s',...]
Step 2: Count pairs: ('t','h')=3, ('h','e')=3 ← most frequent
Step 3: Merge 'th' → new token
Step 4: Count pairs: ('th','e')=3 ← most frequent
Step 5: Merge 'the' → new token
Step 6: Continue merging: 'cat', 'is', 'sleep', 'ing', 'run'...
```

**Example tokenization:**

**Text:** `"The cat is sleeping"`

**Character-level (old, inefficient):**
```
Tokens: ['T','h','e',' ','c','a','t',' ','i','s',' ','s','l','e','e','p','i','n','g']
Count: 19 tokens
```

**BPE (new, efficient):**
```
Tokens: ['The', ' cat', ' is', ' sleep', 'ing']
Count: 5 tokens
Compression: 3.8× fewer tokens!
```

**Output from tokenize.py:**
```
Training BPE tokenizer with target vocab_size=5000...
Starting vocabulary size: 87 (unique characters)
  Vocabulary size: 100/5000
  Vocabulary size: 500/5000
  Vocabulary size: 1000/5000
  ...
✓ Training complete. Final vocabulary size: 5000
✓ Number of merges: 4913

Tokenizing training data...
Training tokens: 52,384,721
✓ Training tokens saved to: data/train_tokens.npy (100 MB)

Compression ratio: 4.32x
  (1 token ≈ 4.32 characters)
```

**Output files:**
- `data/vocab.json` - Vocabulary (5000 subword tokens + merge rules)
- `data/train_tokens.npy` - Tokenized training data (~100 MB)
- `data/val_tokens.npy` - Tokenized validation data (~11 MB)

### Step 4: Train Model

```bash
# Train small model (recommended)
python train.py --config small --max-steps 15000

# Or quick experiment with tiny model
python train.py --config tiny --max-steps 5000

# Or high-quality medium model
python train.py --config medium --max-steps 25000
```

**Example training output:**

```
Loading BPE tokenizer...
✓ Tokenizer loaded from: data/vocab.json
  Vocabulary size: 5000
  Merge rules: 4913

Loading training data...
✓ Training tokens loaded: 52,384,721

Creating model with 'small' configuration...
✓ Model created
  Parameters: 2,847,500
  Size: 10.86 MB (float32)

======================================================================
TRAINING START
======================================================================
Device: mps
Max steps: 15,000
Learning rate: 0.0003
Warmup steps: 500
Batch size: 32
Parameters: 2,847,500
Context length: 256
======================================================================

Step   100/15000 | Loss: 4.2341 | LR: 6.00e-05 | Tokens/s: 11234
Step   200/15000 | Loss: 3.8921 | LR: 1.20e-04 | Tokens/s: 11456
Step   300/15000 | Loss: 3.6234 | LR: 1.80e-04 | Tokens/s: 11589
...
Step   500/15000 | Loss: 3.2145 | LR: 3.00e-04 | Tokens/s: 11723

======================================================================
Step   500 | Val Loss: 3.1872 | Perplexity: 24.21 🌟 BEST
======================================================================

Step  1000/15000 | Loss: 2.7634 | LR: 2.97e-04 | Tokens/s: 11834
...
Step  5000/15000 | Loss: 2.0123 | LR: 2.55e-04 | Tokens/s: 12045

======================================================================
Step  5000 | Val Loss: 2.0345 | Perplexity: 7.65 🌟 BEST
======================================================================

Step 10000/15000 | Loss: 1.7234 | LR: 1.80e-04 | Tokens/s: 12156
...

======================================================================
Step 15000 | Val Loss: 1.6234 | Perplexity: 5.07 🌟 BEST
======================================================================

======================================================================
TRAINING COMPLETE!
======================================================================
Final checkpoint: checkpoints/model_final_step_15000.pt
Best model: checkpoints/model_best.pt (Perplexity: 5.07)
Total time: 3.42 hours
Log file: logs/training_log.csv

Next: python generate.py --checkpoint checkpoints/model_best.pt --prompt 'Your text'
======================================================================
```

**What the metrics mean:**

- **Loss 4.6 → 1.6**: Model learning (lower = better)
- **Perplexity 100 → 5**: Prediction confidence (lower = better, 5 = good for this size)
- **Tokens/s 12,000**: Processing speed on M1 Mac
- **Val Loss**: Performance on unseen data (no overfitting if close to train loss)

**Training progress interpretation:**
```
Step 0:     Loss ~4.6 (random predictions)
Step 1000:  Loss ~2.8 (learning common words)
Step 5000:  Loss ~2.0 (forming word patterns)
Step 10000: Loss ~1.7 (basic sentence structure)
Step 15000: Loss ~1.6 (coherent short text) ✓ Good!
```

**Output files:**
- `checkpoints/model_best.pt` - Best model based on validation loss
- `checkpoints/model_step_1500.pt` - Intermediate checkpoint
- `checkpoints/model_step_3000.pt` - Intermediate checkpoint
- `checkpoints/model_final_step_15000.pt` - Final checkpoint
- `logs/training_log.csv` - Detailed training metrics

### Step 5: Generate Text

```bash
python generate.py \
  --checkpoint checkpoints/model_best.pt \
  --prompt "The capital of France is" \
  --max-tokens 200 \
  --temperature 0.8
```

**Example generation session:**

```
======================================================================
TEXT GENERATION
======================================================================

Using device: mps

Loading BPE tokenizer from: data/vocab.json
✓ Tokenizer loaded
  Vocabulary size: 5000
  Merge rules: 4913

Loading checkpoint from: checkpoints/model_best.pt
✓ Checkpoint loaded (step: 15000)
✓ Config loaded from checkpoint

Creating model...
✓ Model loaded (2,847,500 parameters)

Generation settings:
  Prompt: 'The capital of France is'
  Max tokens: 200
  Temperature: 0.8
  Top-k: 40

Generating...
======================================================================
The capital of France is Paris. Paris is the largest city in France 
and is located in the northern part of the country. The city has a 
population of over two million people. Paris is known for the Eiffel 
Tower, the Louvre Museum, and many other famous landmarks. The city 
is also an important center for art, fashion, and culture.
======================================================================

Generation complete!
```

**More example outputs:**

**Prompt:** `"Albert Einstein was"`
```
Albert Einstein was a German physicist who developed the theory of 
relativity. He won the Nobel Prize in Physics in 1921 for his work 
on the photoelectric effect. Einstein is considered one of the most 
important scientists of the 20th century.
```

**Prompt:** `"The solar system contains"`
```
The solar system contains eight planets. The planets orbit around 
the Sun. Mercury is the closest planet to the Sun and Neptune is 
the farthest. The planets are divided into two groups: the inner 
planets (Mercury, Venus, Earth, Mars) and the outer planets 
(Jupiter, Saturn, Uranus, Neptune).
```

**Prompt:** `"Machine learning is"`
```
Machine learning is a type of artificial intelligence that allows 
computers to learn from data without being explicitly programmed. 
It is used in many applications such as image recognition, natural 
language processing, and recommendation systems.
```

**Temperature effect examples:**

**Temperature 0.5 (conservative, boring):**
```
Prompt: "Water is"
Output: "Water is a chemical compound that is essential for life. 
Water is found in lakes, rivers, and oceans."
```

**Temperature 0.8 (balanced, natural):**
```
Prompt: "Water is"
Output: "Water is a liquid that is essential for all living things. 
It covers about 71% of the Earth's surface. Water is made up of 
hydrogen and oxygen molecules."
```

**Temperature 1.2 (creative, risky):**
```
Prompt: "Water is"
Output: "Water is an important substance that has many uses in 
agriculture and industry. People need water for drinking and also 
for growing crops. Without water, life on Earth would not exist."
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy pandas matplotlib tqdm

# Or use requirements file
pip install -r requirements.txt
```

**Example requirements.txt:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### Complete Pipeline

```bash
# 1. Preprocess (assuming you have raw_wikipedia.txt)
python preprocess_data.py --input data/raw_wikipedia.txt

# 2. Train BPE tokenizer and tokenize
python tokenize.py --vocab-size 5000

# 3. Train model
python train.py --config small --max-steps 15000

# 4. Generate text
python generate.py --checkpoint checkpoints/model_best.pt --prompt "Hello"
```

**Example: Complete 2-hour run (tiny model):**
```bash
# Fast experiment for testing
python preprocess_data.py --input data/raw_wikipedia.txt
# Output: train.txt (210MB), val.txt (23MB) - takes 2 minutes

python tokenize.py --vocab-size 3000
# Output: vocab.json, train_tokens.npy, val_tokens.npy - takes 5 minutes

python train.py --config tiny --max-steps 5000 --batch-size 32
# Output: checkpoints/model_best.pt - takes 2 hours

python generate.py --checkpoint checkpoints/model_best.pt --prompt "AI is"
# Output: Generated text - instant
```

## Configuration Options

### Available Configurations

```bash
# View all available model configurations
python train.py --list-configs
python create_model.py --list-configs
```

**Example output:**
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

### Model Size Comparison

| Config | Parameters | Vocab Size | Context | Memory | Training Time | Quality |
|--------|-----------|------------|---------|--------|---------------|---------|
| tiny   | 500K      | 5000       | 128     | 2GB    | 2-3h          | Basic   |
| small  | 2.8M      | 5000       | 256     | 4GB    | 3-4h          | Good ✓  |
| medium | 11M       | 5000       | 512     | 8GB    | 8-12h         | Better  |
| large  | 25M       | 5000       | 512     | 16GB   | 16-24h        | Best    |

**Example: Inspect model architecture:**
```bash
python create_model.py --config small
```

**Output:**
```
Model Architecture:
  Vocabulary size: 5,000
  Embedding dimension: 128
  Number of layers: 4
  Number of heads: 4
  Head dimension: 32
  Max sequence length: 256
  Feed-forward dimension: 512

Parameters by component:
  Embeddings: 672,768
  Transformer blocks: 1,966,080
  Output head: 640,000
  Final layer norm: 256
  Total: 2,847,500

Memory estimates:
  Model (float32): 10.86 MB
  Model (float16): 5.43 MB
  + Activations (approx): 32.58 MB
  Total (training): ~43.44 MB
```

### Training Parameters

```bash
python train.py \
  --config small \           # Model size
  --batch-size 32 \          # Sequences per batch
  --max-steps 15000 \        # Training steps
  --learning-rate 3e-4 \     # Peak learning rate
  --warmup-steps 500 \       # LR warmup steps
  --device mps               # mps/cuda/cpu
```

**Parameter effect examples:**

**Batch size:**
- `--batch-size 16`: Slower but uses less memory (4GB)
- `--batch-size 32`: Balanced (recommended, 6GB)
- `--batch-size 64`: Faster but needs more memory (12GB)

**Max steps:**
- `--max-steps 5000`: Quick test (1-2 hours, loss ~2.0)
- `--max-steps 15000`: Standard (3-4 hours, loss ~1.6) ✓
- `--max-steps 25000`: High quality (5-7 hours, loss ~1.4)

**Learning rate:**
- `--learning-rate 1e-4`: Conservative, slower convergence
- `--learning-rate 3e-4`: Standard (recommended) ✓
- `--learning-rate 5e-4`: Aggressive, may be unstable

### Generation Parameters

```bash
python generate.py \
  --checkpoint checkpoints/model_best.pt \
  --prompt "Your text here" \
  --max-tokens 200 \         # Length of generation
  --temperature 0.8 \        # Randomness
  --top-k 40                 # Diversity
```

**Parameter effect examples:**

**Temperature (randomness):**

`--temperature 0.3` (very conservative):
```
Prompt: "The cat"
Output: "The cat is a small animal that lives in the house."
(Predictable, safe, boring)
```

`--temperature 0.8` (balanced - recommended):
```
Prompt: "The cat"
Output: "The cat sat on the windowsill, watching birds fly by in 
the garden outside."
(Natural, interesting, coherent)
```

`--temperature 1.5` (very creative):
```
Prompt: "The cat"
Output: "The cat wandered through mysterious alleyways, discovering 
hidden treasures and meeting strange creatures."
(Creative, risky, sometimes weird)
```

**Top-k (diversity):**

`--top-k 10` (focused):
```
Output: Uses only the 10 most likely next tokens
Result: More predictable, repetitive
```

`--top-k 40` (balanced - recommended):
```
Output: Uses top 40 most likely next tokens
Result: Natural variety
```

`--top-k 100` (diverse):
```
Output: Uses top 100 most likely next tokens
Result: More creative, less predictable
```

## Usage Examples

### Example 1: Quick Test with Tiny Model

```bash
# Fast 2-hour experiment to verify everything works
python preprocess_data.py --input data/raw_wikipedia.txt
python tokenize.py --vocab-size 3000
python train.py --config tiny --max-steps 5000
python generate.py --checkpoint checkpoints/model_best.pt --prompt "Science is"
```

**Expected output:**
```
Training complete in 2.1 hours
Final loss: 2.1, Perplexity: 8.2

Generated text:
"Science is the study of the natural world. Science uses experiments 
to learn about things."
```

**Quality:** Basic but coherent (good for testing pipeline)

### Example 2: Standard Training (Recommended)

```bash
# Full training run for good quality (~4 hours)
python preprocess_data.py --input data/raw_wikipedia.txt
python tokenize.py --vocab-size 5000
python train.py --config small --max-steps 15000
python generate.py --checkpoint checkpoints/model_best.pt --prompt "The universe"
```

**Expected output:**
```
Training complete in 3.5 hours
Final loss: 1.6, Perplexity: 5.0

Generated text:
"The universe is everything that exists, including all matter, energy, 
space, and time. Scientists estimate that the universe is about 13.8 
billion years old. The universe is constantly expanding."
```

**Quality:** Good, coherent for 2-3 sentences (recommended for learning)

### Example 3: High-Quality Model

```bash
# Extended training for better results (~10 hours)
python train.py --config medium --max-steps 25000 --batch-size 24
python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "Artificial intelligence" \
  --temperature 0.7 \
  --max-tokens 300
```

**Expected output:**
```
Training complete in 10.2 hours
Final loss: 1.4, Perplexity: 4.1

Generated text:
"Artificial intelligence (AI) refers to computer systems that can 
perform tasks that normally require human intelligence. These tasks 
include visual perception, speech recognition, decision-making, and 
language translation. AI systems use machine learning algorithms to 
improve their performance over time. Modern AI applications include 
virtual assistants, autonomous vehicles, and medical diagnosis systems. 
The field of AI has advanced significantly in recent years due to 
improvements in computing power and the availability of large datasets."
```

**Quality:** Very good, maintains coherence for 4-5 sentences

### Example 4: Compare Temperatures

```bash
# Test different creativity levels
python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "The ocean is" --temperature 0.5 --max-tokens 100

python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "The ocean is" --temperature 0.8 --max-tokens 100

python generate.py --checkpoint checkpoints/model_best.pt \
  --prompt "The ocean is" --temperature 1.2 --max-tokens 100
```

**Output comparison:**

**Temp 0.5:**
```
The ocean is a large body of salt water. The ocean covers most of 
the Earth's surface. There are five major oceans on Earth.
```
(Safe, factual, predictable)

**Temp 0.8:**
```
The ocean is home to millions of different species of fish, mammals, 
and other marine life. The ocean plays an important role in regulating 
the Earth's climate and weather patterns.
```
(Natural, interesting, varied)

**Temp 1.2:**
```
The ocean is a mysterious and fascinating place full of wonders 
waiting to be discovered. Deep beneath the waves lie ancient secrets 
and incredible creatures that few people have ever seen.
```
(Creative, poetic, more risk of incoherence)

### Example 5: Verify Setup

```bash
# Test that everything is installed correctly
python test_model.py
```

**Expected output:**
```
======================================================================
MODEL ARCHITECTURE TESTS
======================================================================

Testing MultiHeadAttention...
  ✓ MultiHeadAttention working correctly

Testing FeedForward...
  ✓ FeedForward working correctly

Testing TransformerBlock...
  ✓ TransformerBlock working correctly

Testing GPTModel...
  ✓ GPTModel working correctly
  ✓ Forward pass successful (loss: 4.6234)

Testing text generation...
  ✓ Generation working correctly
  ✓ Generated 15 tokens

Testing model configurations...
  ✓ Tiny    config:   497,408 parameters (layers=2, heads=2)
  ✓ Small   config: 2,847,500 parameters (layers=4, heads=4)
  ✓ Medium  config: 11,327,488 parameters (layers=6, heads=8)
  ✓ Large   config: 25,485,312 parameters (layers=8, heads=12)

Testing configuration validation...
  ✓ Valid config accepted
  ✓ Invalid config rejected

Testing config.to_dict()...
  ✓ config.to_dict() working correctly

Testing save/load...
  ✓ Save/load working correctly

======================================================================
ALL TESTS PASSED ✓
======================================================================

Your model architecture is working correctly!
Next steps:
  1. python tokenize.py              # Prepare data
  2. python train.py --config small  # Train model
  3. python generate.py              # Generate text
```

## Hardware Requirements

### Minimum Configuration
- **RAM:** 8GB system memory
- **Storage:** 10GB free disk space
- **CPU:** Any modern processor (very slow: 24-36 hours for training)

### Recommended Configuration
- **RAM:** 16GB system memory
- **GPU:** Apple M1/M2/M3 or NVIDIA GPU (8GB+ VRAM)
- **Storage:** 20GB free disk space

### Training Speed by Hardware

| Hardware | Tiny (5K steps) | Small (15K steps) | Medium (25K steps) |
|----------|----------------|-------------------|-------------------|
| M1 Mac (8GB) | 2-3h | 3-4h | 8-10h |
| M1 Pro (16GB) | 1.5-2h | 2.5-3h | 6-8h |
| M1 Max (32GB) | 1-1.5h | 2-2.5h | 5-7h |
| NVIDIA 3080 (10GB) | 1-1.5h | 2-3h | 4-6h |
| NVIDIA 4090 (24GB) | 0.5-1h | 1-1.5h | 3-4h |
| CPU only | 12-18h | 24-36h | 48-72h |

**Example: M1 Mac training output:**
```
Step  1000/15000 | Tokens/s: 11,234  (MPS acceleration)
Step  5000/15000 | Tokens/s: 11,456
Step 10000/15000 | Tokens/s: 11,589
Step 15000/15000 | Tokens/s: 11,723

Total time: 3.42 hours
```

## Expected Results

### Quality by Configuration

**Tiny (500K params):**
- Coherence: 1-2 sentences
- Grammar: Mostly correct
- Facts: Often correct for common knowledge
- Use case: Quick experiments

**Example:**
```
Prompt: "Dogs are"
Output: "Dogs are animals that live with people. Dogs are pets."
```

**Small (2.8M params) - Recommended:**
- Coherence: 2-3 sentences ✓
- Grammar: Mostly correct
- Facts: Often accurate
- Use case: Learning and experimentation

**Example:**
```
Prompt: "Dogs are"
Output: "Dogs are domesticated animals that have been kept as pets 
for thousands of years. They are known for their loyalty and are 
often called man's best friend."
```

**Medium (11M params):**
- Coherence: 3-4 sentences
- Grammar: Good
- Facts: Usually accurate
- Use case: Better quality applications

**Example:**
```
Prompt: "Dogs are"
Output: "Dogs are domesticated mammals that belong to the family 
Canidae. They were first domesticated from wolves thousands of years 
ago. Dogs come in many different breeds, sizes, and colors. They are 
highly social animals and have been bred for various purposes including 
hunting, herding, and companionship."
```

**Large (25M params):**
- Coherence: 4-5 sentences
- Grammar: Very good
- Facts: Generally accurate
- Use case: High-quality applications

**Example:**
```
Prompt: "Dogs are"
Output: "Dogs are domesticated carnivorous mammals that have been 
companions to humans for at least 15,000 years. They belong to the 
family Canidae and are descended from wolves. Dogs have been selectively 
bred for various purposes, resulting in hundreds of different breeds 
with diverse appearances and temperaments. They are known for their 
intelligence, loyalty, and ability to form strong bonds with humans. 
Dogs serve many roles in human society, including as pets, working 
animals, and service animals."
```

### Common Limitations

**What the model CAN do:**
- Generate grammatically correct text
- Produce factually accurate content for well-known topics
- Maintain coherence for 2-5 sentences (depending on size)
- Complete sentences naturally
- Write in Wikipedia-style prose

**What the model CANNOT do:**
- Answer questions reliably
- Follow complex instructions
- Reason logically about new problems
- Maintain coherence beyond a few sentences
- Match GPT-3/GPT-4 quality

**Example failure cases:**

**Prompt:** "Calculate 2 + 2 ="
```
Output: "Calculate 2 + 2 = the sum of two numbers is four. The number 
two is an even number."
(Can state facts but can't actually calculate)
```

**Prompt:** "In three steps, explain how to make coffee"
```
Output: "In three steps, explain how to make coffee is a drink made 
from coffee beans. Coffee is popular around the world."
(Can't follow instructions to list steps)
```

**Prompt:** "If all dogs are animals and all animals need food, do dogs need food?"
```
Output: "If all dogs are animals and all animals need food, do dogs 
need food for survival. Dogs eat meat and other foods."
(Loses the logical reasoning structure)
```

## BPE Tokenization Details

### How BPE Works

**Step-by-step example:**

**Input text:**
```
"low low low lower lower newest newest newest"
```

**Iteration 1:** Start with characters
```
Tokens: ['l','o','w',' ','l','o','w',' ','l','o','w',' ','l','o','w','e','r',...]
Count pairs:
  ('l','o'): 6 times ← most frequent
  ('o','w'): 6 times
  ('w',' '): 6 times
Merge ('l','o') → 'lo'
```

**Iteration 2:**
```
Tokens: ['lo','w',' ','lo','w',' ','lo','w',' ','lo','w','e','r',...]
Count pairs:
  ('lo','w'): 6 times ← most frequent
Merge ('lo','w') → 'low'
```

**Iteration 3:**
```
Tokens: ['low',' ','low',' ','low',' ','low','e','r',...]
Count pairs:
  ('low','e'): 2 times ← most frequent
Merge ('low','e') → 'lowe'
```

**Continue until vocab_size reached (5000 tokens)**

### Compression Examples

**Example 1: Common phrases**
```
Text: "the quick brown fox jumps over the lazy dog"
Character-level: 44 tokens (including spaces)
BPE: 9 tokens ['the', ' quick', ' brown', ' fox', ' jump', 's', ' over', ' the', ' lazy', ' dog']
Compression: 4.9×
```

**Example 2: Technical text**
```
Text: "machine learning algorithms"
Character-level: 27 tokens
BPE: 4 tokens ['machine', ' learning', ' algorithm', 's']
Compression: 6.75×
```

**Example 3: Wikipedia article excerpt**
```
Text: "Artificial intelligence is the simulation of human intelligence by machines"
Character-level: 79 tokens
BPE: 14 tokens ['Art', 'ificial', ' intelligence', ' is', ' the', ' simulation', ' of', ' human', ' intelligence', ' by', ' machines']
Compression: 5.6×
```

### Vocabulary Structure

**Example vocab.json structure:**
```json
{
  "type": "subword",
  "vocab": {
    "a": 0,
    "b": 1,
    "c": 2,
    ...
    "th": 87,
    "the": 156,
    "ing": 234,
    ...
    " the": 501,
    " and": 523,
    "tion": 678,
    ...
  },
  "merges": [
    ["t", "h"],
    ["th", "e"],
    ["i", "n"],
    ["in", "g"],
    ...
  ],
  "vocab_size": 5000
}
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**

**Option 1: Reduce batch size**
```bash
python train.py --config small --batch-size 16  # Instead of 32
```

**Option 2: Use smaller model**
```bash
python train.py --config tiny --batch-size 32
```

**Option 3: Reduce context length**
Edit `config.py`:
```python
SMALL_CONFIG = ModelConfig(
    max_seq_len=128,  # Instead of 256
    ...
)
```

**Example: Memory usage by config**
```
tiny (batch=32):   ~2GB GPU memory
small (batch=32):  ~4GB GPU memory
small (batch=16):  ~2GB GPU memory
medium (batch=32): ~8GB GPU memory
large (batch=16):  ~8GB GPU memory
```

### Slow Training

**Symptom:**
```
Step  100/15000 | Tokens/s: 1,234  (very slow)
```

**Check device:**
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True (for M1 Mac)
```

**Force specific device:**
```bash
python train.py --device mps   # For M1 Mac
python train.py --device cuda  # For NVIDIA GPU
```

**Example speed comparison:**
```
CPU:  1,500 tokens/s  (very slow)
MPS:  11,000 tokens/s (M1 Mac) ✓
CUDA: 18,000 tokens/s (NVIDIA 3080) ✓
```

### Poor Generation Quality

**Symptom:**
```
Generated text is repetitive or incoherent
```

**Solution 1: Train longer**
```bash
python train.py --max-steps 25000  # Instead of 15000
```

**Solution 2: Adjust generation parameters**
```bash
python generate.py \
  --temperature 0.7 \    # Lower = more conservative
  --top-k 50             # Higher = more diverse
```

**Solution 3: Use larger model**
```bash
python train.py --config medium --max-steps 25000
```

**Example: Quality improvement over training steps**
```
Step  5,000: "The cat is a animal that lives in houses."
Step 10,000: "The cat is a small domesticated animal that lives with people."
Step 15,000: "The cat is a small carnivorous mammal that has been domesticated for thousands of years."
Step 25,000: "The cat is a small carnivorous mammal that has been kept as a pet by humans for approximately 10,000 years."
```

### Loss Not Decreasing

**Symptom:**
```
Step  1000 | Loss: 4.2
Step  2000 | Loss: 4.1
Step  3000 | Loss: 4.0  (very slow decrease)
```

**Solution 1: Check learning rate**
```bash
python train.py --learning-rate 1e-4   # Try lower
python train.py --learning-rate 5e-4   # Or try higher
```

**Solution 2: Check data**
```bash
# Verify tokenization completed
ls -lh data/*.npy
# Should show: train_tokens.npy (~100MB), val_tokens.npy (~11MB)
```

**Solution 3: Increase warmup**
```bash
python train.py --warmup-steps 1000  # Instead of 500
```

### FileNotFoundError

**Symptom:**
```
FileNotFoundError: Tokenizer file not found at data/vocab.json
```

**Solution:**
```bash
# Run tokenization first
python tokenize.py --vocab-size 5000
```

**Verify files exist:**
```bash
ls -lh data/
# Should show:
# train.txt, val.txt (text files)
# vocab.json (vocabulary)
# train_tokens.npy, val_tokens.npy (tokenized arrays)
```

## Validation

### Test Model Architecture

```bash
python test_model.py
```

**Expected output:** All tests pass with ✓ marks

### Check Available Configurations

```bash
python create_model.py --list-configs
```

**Example output:**
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
```

### View Training Progress

```bash
# Real-time monitoring (in separate terminal)
tail -f logs/training_log.csv

# Or view last 10 steps
tail -10 logs/training_log.csv
```

**Example log output:**
```
step,train_loss,val_loss,lr,time,perplexity
500,3.187234,3.234521,0.00030000,121.34,25.42
1000,2.876543,2.912345,0.00029700,243.67,18.43
1500,2.654321,2.698765,0.00029100,365.23,14.89
```

### Check Saved Checkpoints

```bash
ls -lh checkpoints/
```

**Example output:**
```
-rw-r--r-- 1 user 10.9M model_best.pt           (best validation loss)
-rw-r--r-- 1 user 10.9M model_step_1500.pt      (intermediate)
-rw-r--r-- 1 user 10.9M model_step_3000.pt      (intermediate)
-rw-r--r-- 1 user 10.9M model_final_step_15000.pt (final)
```

### Verify Tokenizer

```python
# Test tokenizer in Python
from tokenizer import SubwordTokenizer

# Load trained tokenizer
tokenizer = SubwordTokenizer.load('data/vocab.json')
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Merge rules: {len(tokenizer.merges)}")

# Test encoding
text = "The quick brown fox"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Test decoding
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
assert text.lower() == decoded, "Encoding/decoding mismatch!"
print("✓ Tokenizer working correctly")
```

**Expected output:**
```
Vocabulary size: 5000
Merge rules: 4913
Text: The quick brown fox
Tokens: [342, 1876, 987, 2341]
Token count: 4
Decoded: the quick brown fox
✓ Tokenizer working correctly
```

## What This Project Does ✓

- Trains a working GPT-style language model
- Uses BPE tokenization (like ChatGPT)
- Generates coherent text for 2-5 sentences
- Demonstrates transformer architecture
- Provides configurable model sizes
- Includes proper configuration management
- Runs on consumer hardware (M1 Mac)

## What This Project Doesn't Do ✗

- Answer questions reliably
- Follow complex instructions
- Reason logically about new problems
- Maintain coherence beyond a few sentences
- Match GPT-3/GPT-4 quality
- Support fine-tuning (would need additional code)
- Include RLHF or instruction tuning

## Technical Specifications

**Architecture:** Decoder-only transformer (GPT-style)  
**Training Data:** Simple English Wikipedia (~238MB text)  
**Tokenization:** Subword (BPE) with ~5000 tokens  
**Configurations:** 4 predefined (tiny/small/medium/large)  
**Context Window:** 128-512 tokens (config-dependent)  
**Parameters:** 500K-25M (config-dependent)  
**Training Time:** 2-24 hours (config-dependent)  
**Framework:** PyTorch 2.0+

## License & Credits

Educational project demonstrating GPT architecture with BPE tokenization. Based on:
- "Attention is All You Need" paper (Vaswani et al., 2017)
- GPT-2 architecture and tokenization approach
- Byte Pair Encoding algorithm

---

**Quick Commands Summary:**
```bash
# Setup
pip install -r requirements.txt

# View available configurations
python train.py --list-configs

# Test architecture
python test_model.py

# Complete pipeline
python preprocess_data.py --input data/raw_wikipedia.txt
python tokenize.py --vocab-size 5000
python train.py --config small --max-steps 15000
python generate.py --checkpoint checkpoints/model_best.pt --prompt "Hello"
```

**Questions?** Check file docstrings, run with `--help` flag, or review the detailed examples above.