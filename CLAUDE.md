# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

A virtual environment is expected at `.venv/` (gitignored).

## Running

```bash
# Train the full GPT-2 Small model (downloads Genesis text dataset automatically)
python LLM.py

# Train the lightweight GPT on Dante's Inferno
python nano_gpt.py

# Launch notebooks
jupyter lab
```

## Architecture

This is an educational PyTorch implementation of GPT-style language models built from scratch.

### Core modules

| File | Purpose |
|------|---------|
| `LLM.py` | Main GPT-2 Small training pipeline (124M-param config). Defines all model classes, training loop, and text generation. Entry point for pretraining. |
| `gpt_model.py` | Standalone architecture module — same classes as `LLM.py` but extracted for reuse in notebooks. |
| `gpt_utils.py` | Advanced generation (temperature, top-k), loss utilities, and data loading helpers built on `tiktoken`. |
| `loader.py` | Loads pre-trained OpenAI GPT-2 TensorFlow checkpoint weights into the PyTorch model (`load_weights_into_gpt`). |
| `nano_gpt.py` | Minimal 6-layer/6-head GPT (384-dim) trained on `inferno.txt`. Self-contained, no external deps beyond torch. |

### Model configuration

The standard GPT config used across modules:
```python
GPT_CONFIG = {
    "vocab_size": 50257,    # GPT-2 tokenizer
    "context_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

### Notebooks

- `chatgpt.ipynb` — Interactive ChatGPT API experiments
- `classifier_fine_tuning.ipynb` — Fine-tuning for sentiment classification on `train/validation/test.csv`
- `instruction_file_tuning.ipynb` — Instruction-following fine-tuning on `instruction-data.json`

### Data

- `train.csv` / `validation.csv` / `test.csv` — Sentiment classification dataset (Label + Text columns)
- `instruction-data.json` — 5,500 instruction-response pairs for instruction tuning
- `commedia.txt` / `inferno.txt` — Literary pretraining text (Dante)

## GitHub Actions

Two Claude-powered workflows run automatically:
- **claude.yml** — Responds to `@claude` mentions in PRs and issues
- **claude-code-review.yml** — Auto-reviews every PR on open/sync

Both require the `ANTHROPIC_API_KEY` secret to be set in the repository.
