# GPT Language Model — From Scratch

An educational implementation of a GPT-style large language model built entirely from scratch with PyTorch. The project covers pretraining, weight loading from OpenAI's GPT-2, classification fine-tuning, and instruction-following fine-tuning.

---

## Project Overview

This repository implements the core components of the GPT architecture step by step:

- **Causal multi-head self-attention** with scaled dot-product and causal masking
- **Transformer blocks** with pre-norm (LayerNorm → Attention → Residual, LayerNorm → FFN → Residual)
- **Token + positional embeddings** (learned)
- **Full GPT-2 Small config** (~124M parameters equivalent): 12 layers, 12 heads, 768-dim embeddings, 50 257-token vocabulary
- **Text generation** with greedy decoding, temperature sampling, and top-k filtering
- **Fine-tuning** for sentiment classification and instruction following

---

## Installation

### Prerequisites

- Python 3.10+
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd LLM

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch will use MPS automatically when available.
> **Intel macOS:** requirements.txt pins specific torch and tensorflow versions for compatibility.
> **CUDA:** If a CUDA GPU is detected, PyTorch will use it automatically — no extra configuration needed.

---

## Running the Models

### Full GPT-2 Small — pretraining from scratch

```bash
python LLM.py
```

- Downloads the Genesis text dataset automatically (~300 KB) on first run
- Trains a 12-layer / 12-head / 768-dim GPT on a 90/10 train-validation split
- Prints train/val loss every 5 steps and generates a sample text after each epoch
- Default: 10 epochs, batch size 2, context length 256, learning rate 5e-4

**Configuration** (edit at the bottom of `LLM.py`):

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # original GPT-2 uses 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1,
}
```

### Nano GPT — lightweight model on Dante's Inferno

```bash
python nano_gpt.py
```

- Trains a smaller model (6 layers, 6 heads, 384-dim) on `inferno.txt`
- 5 000 training iterations, batch size 64, context length 256
- Generates 500 tokens at the end of training

---

## Jupyter Notebooks

Start the notebook server:

```bash
jupyter lab
```

| Notebook | Description |
|----------|-------------|
| `chatgpt.ipynb` | Interactive experiments with the ChatGPT API |
| `classifier_fine_tuning.ipynb` | Fine-tune GPT for sentiment classification using `train/validation/test.csv` |
| `instruction_file_tuning.ipynb` | Instruction-following fine-tuning on `instruction-data.json` |

---

## Architecture

```
Input token IDs  (B, T)
       │
       ├── Token Embedding      nn.Embedding(50257, 768)
       ├── Position Embedding   nn.Embedding(T, 768)
       │
       └─► x  (B, T, 768)
              │
         ┌───┴─── × 12 TransformerBlock ────────┐
         │  LayerNorm → MultiHeadAttention       │
         │  (12 heads, head_dim=64)  + residual  │
         │  LayerNorm → FeedForward              │
         │  (768 → 3072 → 768, GELU) + residual  │
         └───────────────────────────────────────┘
              │
         LayerNorm  (B, T, 768)
              │
         Linear head  (B, T, 50257)  ← logits
```

**Key implementation details:**
- **Causal mask**: upper-triangular mask registered as a buffer; future tokens are filled with `-inf` before softmax.
- **GELU**: tanh approximation `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))`.
- **Weight loading**: `loader.py` maps OpenAI's TensorFlow GPT-2 checkpoint tensors to the PyTorch model, including transposing convolutional weights to linear weights.
- **Nano GPT difference**: uses ReLU instead of GELU, character-level-compatible tokenization via tiktoken, and explicit per-head attention (`Head` + `MultiHeadAttention` with `ModuleList`).

---

## Dataset Files

| File | Description |
|------|-------------|
| `inferno.txt` | Dante's Inferno (706 lines) — training corpus for nano_gpt |
| `commedia.txt` | Full Divina Commedia (14 752 lines) |
| `train.csv` / `validation.csv` / `test.csv` | Sentiment classification dataset (Label + Text columns) |
| `instruction-data.json` | 5 500 instruction-response pairs for instruction tuning |
| `instruction-data-with-response.json` | Subset of instruction data including model responses |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.2.2` | Neural network framework |
| `tiktoken >= 0.5.1` | GPT-2 tokenizer (BPE, 50 257-token vocab) |
| `tensorflow >= 2.16.2` | Loading pre-trained OpenAI GPT-2 checkpoints |
| `jupyterlab >= 4.0` | Notebook interface |
| `matplotlib >= 3.7.1` | Loss curve visualization |
| `pandas >= 2.2.1` | CSV dataset handling |
| `tqdm >= 4.66.1` | Progress bars |
| `numpy >= 1.26` | Numerical utilities |

---

## GitHub Actions

Two automated Claude-powered workflows run on every pull request:

- **Claude PR Assistant** (`claude.yml`): responds to `@claude` mentions in PR/issue comments
- **Claude Code Review** (`claude-code-review.yml`): automatically reviews every PR when opened or updated

Both workflows require an `ANTHROPIC_API_KEY` secret configured in the repository settings.
