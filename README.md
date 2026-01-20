# GPT Language Model - From Scratch

A minimal implementation of GPT (Generative Pre-trained Transformer) trained on Simple English Wikipedia. Uses **Byte Pair Encoding (BPE)** tokenization like GPT/ChatGPT. Train a 2.8M parameter model in 2-4 hours on M1 Mac.

## Project Structure

```
.
├── main.py                # Unified CLI entry point
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── logs/                  # Training logs
├── checkpoints/           # Model checkpoints
├── data/                  # Data directory
└── src/                   # Source code
    ├── config.py          # Model configurations
    ├── model.py           # Transformer architecture
    ├── train.py           # Training loop
    ├── generate.py        # Text generation
    ├── tokenizer.py       # BPE Tokenizer implementation
    ├── tokenize.py        # Tokenizer training script
    ├── preprocess_data.py # Data preprocessing
    ├── test_model.py      # Tests
    └── constants.py       # File path constants
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Complete Pipeline

```bash
# 1. Download & Extract Data (Manual Step)
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
python -m wikiextractor.WikiExtractor simplewiki-latest-pages-articles.xml.bz2 -o data/extracted
find data/extracted -name 'wiki_*' -exec cat {} \; > data/raw_wikipedia.txt

# 2. Preprocess
python main.py preprocess --input data/raw_wikipedia.txt

# 3. Train BPE tokenizer and tokenize
python main.py tokenize --vocab-size 5000

# 4. Train with selected configuration
python main.py train --config small --max-steps 15000

# 5. Generate
python main.py generate --checkpoint checkpoints/model_best.pt --prompt "Hello"
```

## Unified CLI Usage

The project uses a unified CLI `main.py` to manage all operations.

```bash
python main.py <command> [args]
```

Available commands:
- `train`: Train the model
- `generate`: Generate text
- `tokenize`: Train tokenizer and encode data
- `preprocess`: Clean and split raw text
- `test`: Run architecture tests
- `create`: Inspect model configurations

### Training

```bash
python main.py train \
  --config small \
  --batch-size 32 \
  --max-steps 15000 \
  --device mps
```

### Generation

```bash
python main.py generate \
  --checkpoint checkpoints/model_best.pt \
  --prompt "The capital of France is" \
  --temperature 0.8
```

### Configuration Options

View all available model configurations:

```bash
python main.py create --list-configs
```

| Config | Parameters | Layers | Embed Dim | Heads | Context |
|--------|-----------|---------|-----------|-------|---------|
| tiny   | ~500K     | 2       | 64        | 2     | 128     |
| small  | ~2.8M     | 4       | 128       | 4     | 256     |
| medium | ~11M      | 6       | 256       | 8     | 512     |
| large  | ~25M      | 8       | 384       | 12    | 512     |

## The Process: Data → Model

### Step 1: Data Preparation
Raw Wikipedia XML is extracted to plain text using `wikiextractor`.
`python main.py preprocess` cleans this text and splits it into training and validation sets.

### Step 2: Tokenization
`python main.py tokenize` learns a BPE vocabulary (default 5000 tokens) and converts the text into efficient numpy arrays (`train_tokens.npy`, `val_tokens.npy`).

### Step 3: Training
`python main.py train` trains the Transformer model to predict the next token. Checkpoints are saved to `checkpoints/`.

### Step 4: Inference
`python main.py generate` loads a checkpoint and autoregressively generates new text based on a prompt.

## Hardware Requirements

**Minimum:** 8GB RAM, CPU (slow)
**Recommended:** 16GB RAM, Apple M1/M2/M3 or NVIDIA GPU

## License & Credits

Educational project based on "Attention is All You Need" (Vaswani et al.) and GPT-2 (Radford et al.).