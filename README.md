# Parrot with Alzheimer's 🦜
### A Hands-On Guide to Building Your First LLM from Scratch

---

## Project Overview

**Goal:** Build a tiny language model trained on Simple English Wikipedia (~235MB plain text) to understand the complete LLM development pipeline.

**Expected Result:** A text generator that produces Wikipedia-style sentences with basic coherence for 2-3 sentences before losing the plot.

**Time Investment:** 15-20 hours over one weekend  
**Hardware:** GPU recommended (GTX 1060 or better), CPU training possible but slow (1-2 days)

---

## Step 1: Data Collection

**Objective:** Obtain raw training data from Wikipedia. The model learns language patterns by analyzing large amounts of text, so we need a substantial corpus of well-written, diverse content.

**Tools:**
- `wget` for downloading
- `wikiextractor` or `mwparserfromhell` for XML parsing
- Python `requests` library (alternative)

**Process:**
1. Download Wikipedia dump (Simple English Wikipedia recommended)
2. Download size: ~350MB compressed (.bz2)
3. Unpacked XML size: ~1.52 GB
4. Extract plain text from XML (removes markup, templates, etc.)
5. Final plain text: ~238 MB (approximately 200,000 articles)

**Input:** Wikipedia dump URLs  
**Output:** `data/raw_wikipedia.txt` (~238 MB plain text file)

**Commands:**
```bash
mkdir data

# Download compressed Wikipedia dump
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2

# Extract
bzip2 -d simplewiki-latest-pages-articles.xml.bz2

# Install WikiExtractor
pip install wikiextractor

# Convert XML to plain text
wikiextractor simplewiki-latest-pages-articles.xml -o extracted --no-templates

# Combine all extracted files into one
find extracted -name 'wiki_*' -exec cat {} \; > data/raw_wikipedia.txt
```

**Success Criteria:** You have a single ~238 MB text file with readable Wikipedia content, no XML tags or wiki markup

---

## Step 2: Data Preprocessing

**Objective:** Clean and normalize text for training. Removing noise and inconsistencies ensures the model learns from high-quality patterns rather than formatting artifacts or errors.

**Tools:**
- Python standard library (`re`, `unicodedata`)
- Custom cleaning script

**Process:**
1. Remove special characters and excessive whitespace
2. Normalize Unicode (handle accents, symbols)
3. Convert to lowercase (optional, simplifies vocab)
4. Split into train (90%) and validation (10%) sets
5. Remove very short lines (< 10 characters)

**Input:** `data/raw_wikipedia.txt` (~238 MB)  
**Output:** 
- `data/train.txt` (~210 MB)
- `data/val.txt` (~23 MB)

**Key Considerations:**
- Keep punctuation (helps model learn sentence structure)
- Preserve numbers (useful for facts)
- Don't remove common special characters like quotes

**Estimated Processing Time:** 1-2 minutes

---

## Step 3: Tokenization

**Objective:** Convert text into numerical tokens the model can process. Neural networks operate on numbers, not text, so we transform each character into a unique ID that the model can mathematically process.

**Tools:**
- Custom Python tokenizer class
- Or `tiktoken` library (OpenAI's tokenizer)

**Approach:** Character-level tokenization (simplest)
- Each unique character becomes a token
- Vocabulary size: ~50-100 tokens
- Alternative: Use BPE (Byte-Pair Encoding) for subword tokens

**Process:**
1. Scan entire dataset to build vocabulary
2. Create char → ID and ID → char mappings
3. Encode all text into token IDs
4. Save as numpy arrays for fast loading

**Input:** `data/train.txt`, `data/val.txt`  
**Output:**
- `data/vocab.json` (character to ID mapping, ~50 entries)
- `data/train_tokens.npy` (numpy array of integers)
- `data/val_tokens.npy` (numpy array of integers)
- `tokenizer.py` (encode/decode functions)

**Example Vocabulary:**
```json
{
  "a": 0, "b": 1, "c": 2, ..., " ": 26, ".": 27, "!": 28
}
```

**Code Structure:**
```python
class CharTokenizer:
    def __init__(self, vocab):
        self.char_to_id = vocab
        self.id_to_char = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        return [self.char_to_id[c] for c in text]
    
    def decode(self, ids):
        return ''.join([self.id_to_char[i] for i in ids])
```

---

## Step 4: Model Architecture

**Objective:** Define the transformer neural network. The transformer architecture uses attention mechanisms to learn which parts of the input are relevant for predicting the next token.

**Tools:** PyTorch (`torch.nn`, `torch.nn.functional`)

**Architecture Specifications:**
- **Type:** Decoder-only Transformer (GPT-style)
- **Layers:** 4 transformer blocks
- **Embedding Dimension:** 128
- **Attention Heads:** 4 (32 dims per head)
- **Context Length:** 256 tokens
- **Feedforward Dimension:** 512 (4x embedding dim)
- **Vocabulary Size:** ~50-100 (from tokenizer)
- **Total Parameters:** ~2-3 million

**Input:** Model configuration dictionary  
**Output:** `model.py` containing:
- `TransformerBlock` class
- `GPTModel` class
- `generate()` function

**Key Components:**
1. Token embedding layer
2. Positional embedding layer
3. Multi-head self-attention
4. Feedforward network
5. Layer normalization
6. Dropout for regularization

**File Structure:**
```
model.py
├── class MultiHeadAttention
├── class FeedForward
├── class TransformerBlock
└── class GPTModel
```

---

## Step 5: Training Loop

**Objective:** Train the model to predict next characters. By repeatedly guessing the next character and correcting its mistakes, the model gradually learns the statistical patterns of language.

**Tools:**
- PyTorch training utilities
- `torch.optim.AdamW` optimizer
- `torch.utils.data.DataLoader`

**Training Configuration:**
- **Learning Rate:** 3e-4
- **Batch Size:** 32
- **Context Window:** 256 tokens
- **Max Steps:** 12,000-15,000 iterations (adjusted for 210MB dataset)
- **Warmup Steps:** 500
- **Weight Decay:** 0.1
- **Gradient Clipping:** 1.0

**Process:**
1. Initialize model with random weights
2. Load batches of token sequences
3. Forward pass: predict next token
4. Calculate cross-entropy loss
5. Backward pass: compute gradients
6. Update weights with optimizer
7. Log loss every 100 steps
8. Save checkpoint every 1,500 steps

**Input:**
- `data/train_tokens.npy`
- `model.py`
- Config file or hyperparameters

**Output:**
- `checkpoints/model_step_15000.pt` (trained weights)
- `logs/training_log.csv` (step, loss, learning_rate)

**Training Time:**
- **CPU:** 24-36 hours (possible but slow)
- **GPU (GTX 1060):** 4-6 hours
- **GPU (RTX 3070/3080):** 1.5-2.5 hours
- **GPU (RTX 4090):** 30-60 minutes

**What to Watch:**
- Loss should decrease from ~4.0 to ~1.5-2.0
- If loss doesn't decrease after 1000 steps, check learning rate
- Save multiple checkpoints (early ones might be more coherent!)
- Training typically converges around 10,000-12,000 steps

---

## Step 6: Text Generation

**Objective:** Use trained model to generate text from prompts. The model applies its learned patterns to predict and sample the most likely next characters, building coherent text one token at a time.

**Tools:**
- PyTorch inference mode
- Sampling strategies (temperature, top-k, top-p)

**Process:**
1. Load trained model checkpoint
2. Encode input prompt to tokens
3. Generate tokens one at a time
4. Apply sampling strategy to pick next token
5. Decode tokens back to text

**Input:**
- `checkpoints/model_step_15000.pt`
- `data/vocab.json`
- Prompt text: "The theory of relativity"

**Output:** Generated text continuation

**Sampling Parameters:**
- **Temperature:** 0.8 (lower = safer, higher = creative)
- **Top-k:** 40 (consider only top 40 tokens)
- **Max Length:** 100 tokens
- **Repetition Penalty:** 1.2 (discourage loops)

**Example Script:**
```python
python generate.py \
  --checkpoint checkpoints/model_step_15000.pt \
  --prompt "Albert Einstein was" \
  --length 50 \
  --temperature 0.7
```

**Interface Options:**
1. Command-line script (simplest)
2. Jupyter notebook (interactive)
3. Gradio web UI (user-friendly)

---

## Step 7: Evaluation & Metrics

**Objective:** Measure model performance quantitatively. Metrics like perplexity tell us how well the model predicts text it hasn't seen, helping us understand if training was successful.

**Tools:**
- PyTorch evaluation mode
- Custom metric calculations

**Metrics to Calculate:**
1. **Validation Loss:** How well model predicts held-out data
2. **Perplexity:** exp(loss) - lower is better (target: 4-8)
3. **Sample Quality:** Manual inspection of generated text

**Process:**
1. Load validation data
2. Run model in eval mode (no dropout)
3. Calculate average loss across all validation batches
4. Convert to perplexity
5. Generate 10-20 samples for qualitative review

**Input:**
- `checkpoints/model_step_15000.pt`
- `data/val_tokens.npy`

**Output:**
- `results/evaluation_metrics.txt`
- `results/sample_generations.txt`

**Benchmarks:**
- **Good:** Perplexity < 6, coherent for 2-3 sentences
- **Okay:** Perplexity 6-10, somewhat coherent
- **Poor:** Perplexity > 10, mostly gibberish

**Note:** With ~210MB of training data (Simple English Wikipedia), expect decent results but not as polished as models trained on larger datasets.

---

## Project Structure

```
parrot-with-alzheimers/
├── data/
│   ├── raw_wikipedia.txt
│   ├── train.txt
│   ├── val.txt
│   ├── vocab.json
│   ├── train_tokens.npy
│   └── val_tokens.npy
├── src/
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── checkpoints/
│   └── model_step_15000.pt
├── logs/
│   └── training_log.csv
├── results/
│   ├── evaluation_metrics.txt
│   └── sample_generations.txt
└── README.md
```

---

## Expected Results

**What Your Model Will Do:**
- Complete simple factual sentences
- Generate Wikipedia-style prose for 2-3 sentences
- Learn grammar and punctuation reasonably well
- Associate related concepts (e.g., "Einstein" → "physics" → "relativity")
- Use simpler vocabulary (reflecting Simple English Wikipedia's style)

**What It Won't Do:**
- Answer complex questions reliably
- Follow multi-step instructions
- Maintain coherence beyond 2-3 sentences
- Reason logically or understand nuanced context
- Avoid hallucinating facts (especially for less common topics)

**Sample Output:**
```
Prompt: "The capital of France is"
Output: "Paris. The city is located in the northern part of France and 
is the largest city in the country. Paris is known for the Eiffel Tower 
and many museums. The city has a population of over two million people..."
```

**Performance Notes:** 
- Simple English Wikipedia uses simpler language, so outputs will be more accessible
- ~210MB is a good learning dataset - enough to see real language patterns
- Smaller vocabulary compared to full Wikipedia means faster training
- Quality is educational but won't match commercial models

---

## Next Steps & Experiments

1. **Scale Up:** Train on full English Wikipedia (10-50GB) for significant quality improvements
2. **Add BPE Tokenization:** Smaller vocab, better efficiency, handles rare words better
3. **Increase Model Size:** 8-12 layers, 256-512 dims (requires better GPU)
4. **Fine-tune:** Train on specific domain (poetry, code, medical text, etc.)
5. **Add Instruction Following:** Collect prompt-response pairs for instruction tuning
6. **Implement Web Interface:** Deploy with Flask or Gradio for easy sharing
7. **Try Other Languages:** Train on non-English Wikipedia for multilingual models
8. **Compare Datasets:** Try training on different corpora (books, news, code) to see how data affects style

---

## Learning Outcomes

By completing this project, you'll understand:
- How transformers process sequential data
- Why scaling laws matter (more data + bigger model = better results)
- The tokenization-to-generation pipeline
- Why even billion-parameter models still hallucinate
- That "intelligence" emerges from pattern matching at scale

**Remember:** Your Parrot with Alzheimer's is terrible at language, but you've now built the same architecture that powers GPT-4, just 100,000x smaller!
