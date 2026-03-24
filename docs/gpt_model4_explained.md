# `gpt_model4.py` — Line-by-Line Technical Reference

A complete walkthrough of the GPT-4 style language model implementation: every class, every function, every line.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Imports](#2-imports-lines-15)
3. [RMSNorm](#3-rmsnorm-lines-1525)
4. [RoPE — Rotary Position Embedding](#4-rope--rotary-position-embedding-lines-3882)
   - 4.1 [`precompute_rope_freqs`](#41-precompute_rope_freqs-lines-3856)
   - 4.2 [`apply_rope`](#42-apply_rope-lines-5982)
5. [GroupedQueryAttention](#5-groupedqueryattention-lines-96164)
6. [FeedForwardSwiGLU](#6-feedforwardswiglu-lines-180196)
7. [TransformerBlock4](#7-transformerblock4-lines-204237)
8. [GPT4Model](#8-gpt4model-lines-246270)
9. [GPT4\_CONFIG\_SMALL](#9-gpt4_config_small-lines-277287)
10. [Architecture Comparison Table](#10-architecture-comparison-gpt-2-vs-gpt-4-style)
11. [Full Data Flow Diagram](#11-full-data-flow-diagram)

---

## 1. Overview

`gpt_model4.py` implements a **GPT-4 style** autoregressive language model in PyTorch from scratch. It follows the same overall blueprint as GPT-2 (stack of transformer blocks with a causal mask, trained to predict the next token), but replaces four internal components with techniques used in modern large language models (LLaMA, Mistral, PaLM, GPT-NeoX):

| What changes | GPT-2 | GPT-4 style |
|---|---|---|
| Normalisation | `LayerNorm` | `RMSNorm` |
| Position encoding | Learned absolute `pos_emb` | `RoPE` inside each attention head |
| Attention | Multi-Head Attention (MHA) | Grouped Query Attention (GQA) |
| Feed-Forward activation | GELU, 2 layers | SwiGLU, 3 layers |

The file is self-contained and defines exactly six public symbols:

- `RMSNorm` — normalisation layer
- `precompute_rope_freqs` — builds the RoPE cos/sin tables
- `apply_rope` — applies the rotary rotation to Q or K tensors
- `GroupedQueryAttention` — GQA + RoPE attention block
- `FeedForwardSwiGLU` — SwiGLU feed-forward block
- `TransformerBlock4` — one complete transformer layer
- `GPT4Model` — the full model
- `GPT4_CONFIG_SMALL` — a ready-to-use configuration dictionary

---

## 2. Imports (lines 1–5)

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
```

| Import | Why it is needed |
|---|---|
| `math` | Used for `math.ceil` in `FeedForwardSwiGLU` to round the hidden dimension up to the nearest multiple of 256 |
| `torch` | Core PyTorch tensor library: `torch.arange`, `torch.outer`, `torch.cos`, `torch.sqrt`, `torch.inf`, etc. |
| `torch.nn as nn` | Provides `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Dropout`, `nn.Parameter`, `nn.Sequential` — the building blocks of every layer |
| `torch.nn.functional as F` | Stateless functions. Used here exclusively for `F.silu(x)` (the SiLU / Swish activation inside SwiGLU) |

---

## 3. RMSNorm (lines 15–25)

### What is RMSNorm and why it replaces LayerNorm

Standard **LayerNorm** normalises each token's embedding vector by computing both the mean µ and the variance σ²:

```
LayerNorm(x) = (x - µ) / sqrt(σ² + ε) * γ + β
```

**RMSNorm** (Zhang & Sennrich, 2019) removes the mean-centring step entirely. It only divides by the Root Mean Square:

```
RMS(x)    = sqrt( mean(x²) + ε )
RMSNorm(x) = x / RMS(x) * γ
```

Why bother? Two reasons:
1. **Speed**: computing µ requires an extra pass over the vector; removing it cuts normalisation cost by ~30%.
2. **Quality**: empirically, the mean subtraction contributes very little to training stability; the re-scaling by γ is what matters. LLaMA, PaLM, and GPT-NeoX all use RMSNorm.

Note there is also **no β (shift) parameter** — RMSNorm has only γ (scale), not γ and β as LayerNorm does.

### `__init__(self, emb_dim, eps=1e-5)`

```python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
```

- **`emb_dim`** — the size of the last dimension C (e.g. 768). This is the dimension that gets normalised.
- **`self.eps = 1e-5`** — added inside the square root to prevent division by zero when x is an all-zero vector. Never learnable, never updated.
- **`self.scale = nn.Parameter(torch.ones(emb_dim))`** — the learnable γ vector, shape `(C,)`, initialised to all-ones. Wrapping in `nn.Parameter` registers it with the PyTorch optimiser so it is updated during training. It is initialised to ones so the layer starts as identity: `x / RMS(x) * 1 ≈ x` at init.

### `forward(self, x)` — step by step

```python
def forward(self, x):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    norm_x = x / rms
    return self.scale * norm_x
```

Input `x` shape: **(B, T, C)** — batch × tokens × embedding.

**Line 1: `rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)`**

Broken into sub-operations:
- `x.pow(2)` — squares every element. Shape stays **(B, T, C)**.
- `.mean(dim=-1, keepdim=True)` — averages across the last dimension (C), producing the mean of x². `keepdim=True` keeps that dimension as 1 so broadcasting works later. Shape: **(B, T, 1)**.
- `+ self.eps` — adds 1e-5 for numerical safety. Shape: **(B, T, 1)**.
- `torch.sqrt(...)` — element-wise square root. Shape: **(B, T, 1)** — one RMS value per token.

**Line 2: `norm_x = x / rms`**

Divides `x` of shape (B, T, C) by `rms` of shape (B, T, 1). PyTorch broadcasts the 1 across C automatically — each token's C-dimensional vector is divided by its own scalar RMS. Output shape: **(B, T, C)**.

**Line 3: `return self.scale * norm_x`**

`self.scale` has shape `(C,)`. PyTorch broadcasts it over (B, T, C) by treating it as `(1, 1, C)`. This multiplies each embedding dimension independently by its learned γ. Output shape: **(B, T, C)**.

---

## 4. RoPE — Rotary Position Embedding (lines 38–82)

### The core idea

In GPT-2, position is injected by *adding* a learned vector to the token embedding at the very start:

```
x = tok_emb(token_id) + pos_emb(position)   # GPT-2 style
```

This works but has a key limitation: the model sees positions as absolute values (position 0, 1, 2, …). If you train on sequences up to length 256 and then try to run on length 512, the position vectors are completely out-of-distribution.

**RoPE** (Su et al., 2021) takes a different approach: instead of adding a position vector to the token, it *rotates* the query and key vectors inside each attention head by an angle proportional to the token's position. Because attention is driven by the dot product **Q · Kᵀ**, and because a rotation of Q by position p and K by position q produces a dot product that depends only on **(p − q)**, the model naturally learns to attend based on *relative* distances. Long-range generalisation is dramatically better.

No new trainable parameters are introduced — the cos/sin tables are computed from a fixed formula.

---

### 4.1 `precompute_rope_freqs` (lines 38–56)

```python
def precompute_rope_freqs(head_dim, context_length, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin
```

**Purpose**: build two look-up tables, `cos` and `sin`, each of shape `(context_length, head_dim // 2)`. They are computed once at model initialisation and reused on every forward pass.

**Parameters**:
- `head_dim` — number of dimensions per attention head, e.g. 64 (= emb_dim / n_heads = 768 / 12).
- `context_length` — maximum sequence length, e.g. 1024. Tables are pre-built to this size.
- `base=10000` — the frequency base θ₀. A higher value (e.g. 500000 in LLaMA 3) stretches the frequency spectrum, enabling longer context.

**Line: `theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))`**

This implements the RoPE frequency formula:  `θᵢ = 1 / base^(2i / D)` for i = 0, 1, …, D/2 − 1.

- `torch.arange(0, head_dim, 2)` — produces `[0, 2, 4, …, head_dim−2]`, i.e. the even indices. Shape: `(head_dim//2,)`.
- `/ head_dim` — normalises to `[0/D, 2/D, 4/D, …]`, values in `[0, 1)`.
- `base ** (...)` — raises 10000 to each of those fractions. Since the fractions grow from 0 to nearly 1, the powers range from `10000⁰ = 1` to `10000^(≈1) = 10000`.
- `1.0 / (...)` — inverts: the first pair of dimensions gets frequency `1/1 = 1.0` (high frequency, fast oscillation), the last pair gets `1/10000 = 0.0001` (very low frequency, slow oscillation).

Result `theta`, shape `(head_dim//2,)`: a spectrum of frequencies, one per dimension pair.

**Line: `positions = torch.arange(context_length, dtype=torch.float32)`**

Integer token positions 0, 1, 2, …, context_length − 1. Shape: `(context_length,)`.

**Line: `freqs = torch.outer(positions, theta)`**

The outer product multiplies every position by every frequency:  `freqs[t, i] = t × θᵢ`.
Shape: `(context_length, head_dim//2)`.

This is the angle by which dimension pair `i` of the query/key at position `t` will be rotated.

**Lines: `cos = torch.cos(freqs)` / `sin = torch.sin(freqs)`**

Element-wise trigonometric functions. Both have shape `(context_length, head_dim//2)`.

These tables are returned and stored as non-trainable `register_buffer` entries inside `GroupedQueryAttention`, so they:
- Move automatically to GPU with `.to(device)` or `.cuda()`
- Are saved in checkpoints alongside the model weights
- Are not updated by the optimiser

---

### 4.2 `apply_rope` (lines 59–82)

```python
def apply_rope(x, cos, sin):
    _B, _H, T, D = x.shape
    half = D // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([
        x1 * c - x2 * s,
        x2 * c + x1 * s,
    ], dim=-1)

    return rotated
```

**Purpose**: given a query or key tensor of shape `(B, H, T, D)`, rotate each token's head vector by the angle corresponding to its position.

**Arguments**:
- `x` — query or key tensor, shape `(B, H, T, D)` where B=batch, H=heads, T=seq length, D=head_dim.
- `cos`, `sin` — the pre-computed tables from `precompute_rope_freqs`, shape `(context_length, D//2)`.

**Line: `_B, _H, T, D = x.shape`**

Unpacks the four dimensions. `T` is the *actual* sequence length for this batch (may be less than `context_length`); `D` is `head_dim`. B and H are not used directly and are prefixed with `_` to signal they are intentionally unused.

**Line: `half = D // 2`**

RoPE splits each head dimension into two equal halves. Each pair `(x[i], x[i+half])` is treated as a 2D point and rotated together. For head_dim=64, `half=32`.

**Lines: `x1 = x[..., :half]` / `x2 = x[..., half:]`**

`...` means "all leading dimensions" (B, H, T). `x1` is the first half of the embedding per head, `x2` the second. Both have shape `(B, H, T, D//2)`.

**Lines: `c = cos[:T].unsqueeze(0).unsqueeze(0)` / `s = sin[:T].unsqueeze(0).unsqueeze(0)`**

- `cos[:T]` — slice the first T rows from the pre-computed table. Shape: `(T, D//2)`.
- `.unsqueeze(0).unsqueeze(0)` — add two leading singleton dimensions → shape `(1, 1, T, D//2)`.

This allows broadcasting over the B and H dimensions automatically when multiplied with `x1` of shape `(B, H, T, D//2)`.

**The rotation formula**:

A 2D rotation of a point `(x1, x2)` by angle `θ` gives:
```
new_x1 = x1·cos(θ) − x2·sin(θ)
new_x2 = x2·cos(θ) + x1·sin(θ)
```

Each dimension pair `(x[i], x[i+half])` is a 2D point. The angle `θ` is `position × θᵢ` from the pre-computed table. The operation is applied element-wise across all `D//2` pairs simultaneously by the `* c` and `* s` multiplications.

**Line: `torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)`**

Concatenates the two rotated halves back along the last dimension → shape `(B, H, T, D)`. The overall shape is unchanged; only the values are rotated.

**Why this encodes relative position**: when you compute `Q @ Kᵀ` after applying RoPE, the dot product between Q at position p and K at position q ends up depending only on `(p − q)`, not on p and q individually. The model therefore learns *how far apart* two tokens are, not *where* they sit in the absolute sequence.

---

## 5. GroupedQueryAttention (lines 96–164)

### GQA vs standard Multi-Head Attention

In standard MHA there are H query heads, H key heads, and H value heads. For each query head there is exactly one dedicated key head and one value head.

In **Grouped Query Attention** (GQA, Ainslie et al. 2023), there are still H query heads, but only **G < H key/value heads**, where H is divisible by G. Every group of H/G consecutive query heads shares a single KV pair. The KV head is simply *repeated* to match the H query heads before computing the dot products.

Benefits:
- **KV-cache shrinks by H/G×** during autoregressive inference (the bottleneck in large models).
- **Attention compute** reduces similarly.
- **Quality loss is negligible**: in practice 1–2 perplexity points.

Used in: LLaMA 2 (G=8 for 70B), LLaMA 3, Mistral 7B (G=8), Gemma, Falcon.

### `__init__` — assertions, sizes, and submodules

```python
def __init__(self, d_in, d_out, context_length, dropout,
             n_heads, n_kv_heads, rope_base=10000, qkv_bias=False):
    super().__init__()

    assert d_out % n_heads == 0
    assert n_heads % n_kv_heads == 0

    self.d_out      = d_out
    self.n_heads    = n_heads
    self.n_kv_heads = n_kv_heads
    self.n_rep      = n_heads // n_kv_heads
    self.head_dim   = d_out // n_heads

    self.W_query  = nn.Linear(d_in, d_out,                    bias=qkv_bias)
    self.W_key    = nn.Linear(d_in, n_kv_heads * self.head_dim, bias=qkv_bias)
    self.W_value  = nn.Linear(d_in, n_kv_heads * self.head_dim, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out, bias=False)
    self.dropout  = nn.Dropout(dropout)

    self.register_buffer("mask",
        torch.triu(torch.ones(context_length, context_length), diagonal=1))

    cos, sin = precompute_rope_freqs(self.head_dim, context_length, rope_base)
    self.register_buffer("rope_cos", cos)
    self.register_buffer("rope_sin", sin)
```

**Assertions**:
- `d_out % n_heads == 0` — ensures `head_dim = d_out / n_heads` is an integer (no fractional dimensions).
- `n_heads % n_kv_heads == 0` — ensures each KV head serves a whole number of Q heads.

**Derived scalars**:
- `self.n_rep = n_heads // n_kv_heads` — how many query heads share each KV head. E.g. 12 Q heads / 4 KV heads = 3.
- `self.head_dim = d_out // n_heads` — dimensions per head. E.g. 768 / 12 = 64.

**Linear projections**:
- `W_query`: maps each token embedding `(B,T,C)` → `(B,T,H×D)` = `(B,T,768)`. Full H heads.
- `W_key`: maps `(B,T,C)` → `(B,T,G×D)` = `(B,T,256)` when G=4. **Smaller output than W_query** — this is the GQA saving.
- `W_value`: same shape as `W_key`.
- `out_proj`: recombines the H head outputs `(B,T,H×D)` → `(B,T,C)`. Always without bias.
- `dropout`: applied to the attention weight matrix after softmax.

**Causal mask buffer**:

```python
torch.triu(torch.ones(context_length, context_length), diagonal=1)
```

`torch.triu` keeps only elements on and above the k-th diagonal; `diagonal=1` means the main diagonal is zeroed and everything *above* it is 1. Shape `(T, T)`. A value of 1 at position `[i, j]` means token `i` is forbidden from attending to token `j` (because j > i, i.e. it is in the future). In `forward`, positions where the mask is 1 are filled with −∞ before softmax, making their attention weight effectively 0.

This is registered as a **buffer** (not a parameter): it moves to GPU with the model but is never updated by the optimiser.

**RoPE buffers**:

`precompute_rope_freqs` is called once here, in `__init__`. The returned `cos` and `sin` tensors are stored as buffers — they cost no gradient memory and are automatically moved to the correct device.

---

### `forward(x)` — step by step

```python
def forward(self, x):
    B, T, _C = x.shape

    queries = self.W_query(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
    keys    = self.W_key(x)  .view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    values  = self.W_value(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

    queries = apply_rope(queries, self.rope_cos, self.rope_sin)
    keys    = apply_rope(keys,    self.rope_cos, self.rope_sin)

    keys   = keys.repeat_interleave(self.n_rep, dim=1)
    values = values.repeat_interleave(self.n_rep, dim=1)

    attn_scores = queries @ keys.transpose(2, 3)

    mask_bool = self.mask.bool()[:T, :T]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.reshape(B, T, self.d_out)
    context_vec = self.out_proj(context_vec)

    return context_vec
```

**Step 1 — Read input shape**

`B, T, _C = x.shape` — B is batch size, T is the actual sequence length for this call (≤ context_length), C is embedding dim. `_C` is unused (it equals `d_in = d_out`).

**Step 2 — Project Q, K, V and reshape into heads**

```python
queries = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
```

- `self.W_query(x)`: linear layer maps `(B,T,C)` → `(B,T,H×D)` = `(B,T,768)`.
- `.view(B, T, n_heads, head_dim)`: reshape the last dimension into `(H, D)` = `(12, 64)`. Shape: `(B,T,12,64)`.
- `.transpose(1, 2)`: swap the T and H dimensions → `(B,H,T,D)` = `(B,12,T,64)`. This is the standard layout for batched attention (head dimension second).

For keys and values: the same steps, but `W_key` outputs only `G×D` features, so after reshape we get `(B,T,G,D)` → `(B,G,T,D)` = `(B,4,T,64)`.

**Step 3 — Apply RoPE to queries and keys**

```python
queries = apply_rope(queries, self.rope_cos, self.rope_sin)  # (B,H,T,D) unchanged
keys    = apply_rope(keys,    self.rope_cos, self.rope_sin)  # (B,G,T,D) unchanged
```

Each token's query/key head vector is rotated by an angle proportional to its position. Values are not rotated — RoPE only needs to affect the Q·Kᵀ dot product.

**Step 4 — Expand KV heads to match Q heads (GQA)**

```python
keys   = keys.repeat_interleave(self.n_rep, dim=1)    # (B,G,T,D) → (B,H,T,D)
values = values.repeat_interleave(self.n_rep, dim=1)  # (B,G,T,D) → (B,H,T,D)
```

`repeat_interleave(n, dim=1)` repeats each element along dimension 1 exactly n times *in place*. With n_rep=3 and G=4 KV heads, KV head 0 becomes heads 0,1,2 — KV head 1 becomes heads 3,4,5 — etc. After this, K and V have the same shape as Q: `(B,H,T,D)`.

**Step 5 — Scaled dot-product attention**

```python
attn_scores = queries @ keys.transpose(2, 3)
```

- `keys.transpose(2, 3)`: swap T and D → shape `(B,H,D,T)`.
- `queries @ keys.T`: batch matmul `(B,H,T,D) @ (B,H,D,T)` → `(B,H,T,T)`. Entry `[b,h,i,j]` is the dot product between query i and key j in head h of example b.

```python
mask_bool = self.mask.bool()[:T, :T]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

- The mask is trimmed to the actual sequence length T (the pre-built mask covers context_length but we only use the top-left T×T corner).
- Positions where `mask_bool` is True (i.e. j > i, future tokens) are replaced with −∞ *in-place*. The `_` suffix means the operation modifies `attn_scores` directly without allocating a new tensor.

```python
attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
```

- Division by √D (= √64 = 8) prevents the dot products from growing too large when D is high, which would push softmax into its saturated regions and produce near-zero gradients.
- `softmax(dim=-1)` normalises each row (one row per query token) so it sums to 1. Future positions have weight ≈ 0.

```python
attn_weights = self.dropout(attn_weights)
```

Randomly zero some attention weights during training. Regularises the attention distribution. Shape unchanged.

**Step 6 — Aggregate values and project**

```python
context_vec = (attn_weights @ values).transpose(1, 2)
```

- `attn_weights @ values`: `(B,H,T,T) @ (B,H,T,D)` → `(B,H,T,D)`. Each query position gets a weighted sum of value vectors.
- `.transpose(1, 2)`: swap H and T → `(B,T,H,D)`.

```python
context_vec = context_vec.reshape(B, T, self.d_out)
```

Flatten the last two dims: `(B,T,H,D)` → `(B,T,H×D)` = `(B,T,C)`. This is the "concatenate heads" operation.

```python
context_vec = self.out_proj(context_vec)
```

A final linear layer mixes information across all head outputs. Input/output: `(B,T,C)` → `(B,T,C)`.

---

## 6. FeedForwardSwiGLU (lines 180–196)

### The SwiGLU concept

The GPT-2 feed-forward block is a two-layer MLP:

```
output = W₂( GELU( W₁(x) ) )
```

with `hidden_dim = 4 × emb_dim` (e.g. 3072 for C=768).

**SwiGLU** (Shazeer, 2020) replaces this with a *gated* structure using three linear projections and no bias:

```
gate   = SiLU( W_gate(x) )      ← gating signal: which information to pass through
up     = W_up(x)                 ← content signal: what information is available
gated  = gate ⊙ up              ← element-wise product: gate controls up
output = W_down(gated)
```

The **SiLU** (Sigmoid Linear Unit) activation is defined as `SiLU(x) = x × sigmoid(x)`. It is a smooth, self-gated activation. Multiplied with the linear `up` projection, it creates a learnable gate that can suppress or amplify each hidden dimension independently.

Why three matrices instead of two? When you count parameters: two matrices at 4× each = 2×4C² = 8C². Three matrices at 8/3× each = 3×(8/3)C² = 8C². Same total parameter count, better quality.

### `__init__` — hidden dimension calculation

```python
hidden_dim = int(math.ceil((8 / 3 * emb_dim) / 256) * 256)
```

- `8 / 3 * emb_dim` — the "raw" target hidden dimension.
- `/ 256` then `math.ceil(...)` then `* 256` — round up to the nearest multiple of 256. This ensures the matrix dimensions are aligned to GPU memory boundaries (tensor cores prefer multiples of 8 or 64; 256 is a safe conservative choice).

Examples:
- `emb_dim=256`: raw = 682.7 → ceil(682.7/256)=3 → 3×256 = **768**
- `emb_dim=384`: raw = 1024 → ceil(1024/256)=4 → **1024**
- `emb_dim=768`: raw = 2048 → ceil(2048/256)=8 → **2048**

```python
self.gate = nn.Linear(emb_dim, hidden_dim, bias=False)
self.up   = nn.Linear(emb_dim, hidden_dim, bias=False)
self.down = nn.Linear(hidden_dim, emb_dim, bias=False)
```

All three layers have `bias=False`, following LLaMA convention. The `gate` and `up` projections expand from C to H_ff; `down` contracts back from H_ff to C.

### `forward(x)` — step by step

```python
def forward(self, x):
    gated = F.silu(self.gate(x)) * self.up(x)
    return self.down(gated)
```

**Line 1: `F.silu(self.gate(x)) * self.up(x)`**

- `self.gate(x)`: linear projection `(B,T,C)` → `(B,T,H_ff)`.
- `F.silu(...)`: applies SiLU element-wise. `SiLU(z) = z × σ(z)` where σ is the sigmoid function. Values near 0 are suppressed; large positive values pass through almost unchanged; large negative values are gated off. Output shape: `(B,T,H_ff)`.
- `self.up(x)`: separate linear projection `(B,T,C)` → `(B,T,H_ff)`. This is the "content" branch.
- `*`: element-wise multiplication. The SiLU-gated branch selects which dimensions of `up(x)` are emphasised. Shape: `(B,T,H_ff)`.

**Line 2: `self.down(gated)`**

Projects back from `(B,T,H_ff)` → `(B,T,C)`. This is the bottleneck-to-embedding projection.

---

## 7. TransformerBlock4 (lines 204–237)

### Architecture overview

Each `TransformerBlock4` is one complete transformer layer. The structure is:

```
x → [RMSNorm → GQA → dropout] → +x (residual)
  → [RMSNorm → SwiGLU → dropout] → +x (residual)
```

This is the **pre-norm** pattern: normalise *before* the sublayer, not after. Pre-norm is more stable during training and is used in all modern LLMs (GPT-2 already uses pre-norm). The residual connections allow gradients to flow directly through the full stack during backpropagation.

### `__init__` — all submodules

```python
self.att = GroupedQueryAttention(
    d_in=cfg["emb_dim"],
    d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    dropout=cfg["drop_rate"],
    n_heads=cfg["n_heads"],
    n_kv_heads=cfg["n_kv_heads"],
    rope_base=cfg.get("rope_base", 10000),
    qkv_bias=cfg.get("qkv_bias", False),
)
self.ff    = FeedForwardSwiGLU(cfg)
self.norm1 = RMSNorm(cfg["emb_dim"])
self.norm2 = RMSNorm(cfg["emb_dim"])
self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
```

- `self.att`: the GQA layer with RoPE. Reads all needed keys from `cfg`.
- `self.ff`: the SwiGLU FFN.
- `self.norm1`: applied to x before the attention block.
- `self.norm2`: applied to x before the FFN block.
- `self.drop_shortcut`: the same dropout instance is reused for both sublayers (saves memory; dropout is stateless at inference time since it is disabled in `model.eval()`).

### `forward(x)` — step by step

```python
def forward(self, x):
    # Attention sub-block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    # FFN sub-block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    return x
```

All shapes remain `(B, T, C)` throughout.

**Attention sub-block:**

1. `shortcut = x` — save the input before any transformation. This is the residual branch.
2. `x = self.norm1(x)` — RMSNorm before attention. The normalised version enters the attention layer, but the un-normalised `shortcut` is added back at the end.
3. `x = self.att(x)` — GQA + RoPE. Returns `(B,T,C)`.
4. `x = self.drop_shortcut(x)` — randomly zero activations during training.
5. `x = x + shortcut` — residual add. If the attention layer's parameters are all zero (at early training), `x` is unchanged. This guarantees gradient flow.

**FFN sub-block:**

6–10. Exact same pattern with `norm2` and `self.ff` instead of `norm1` and `self.att`.

11. `return x` — the enriched token representations, shape `(B, T, C)`.

---

## 8. GPT4Model (lines 246–270)

### Structure overview

`GPT4Model` is the top-level module that chains everything together:

```
token IDs → tok_emb → drop_emb → [block₀ → block₁ → … → block₁₁] → final_norm → out_head → logits
```

The most important difference from `GPTModel` (GPT-2): **there is no `pos_emb`**. The line:

```python
x = tok_embeds + pos_embeds   # GPT-2
```

disappears entirely. Position is encoded implicitly by the RoPE rotation inside each `GroupedQueryAttention`.

### `__init__` — all submodules

```python
def __init__(self, cfg):
    super().__init__()
    self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.drop_emb  = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock4(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = RMSNorm(cfg["emb_dim"])
    self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    self.context_length = cfg["context_length"]
```

**`self.tok_emb = nn.Embedding(vocab_size, emb_dim)`**

A look-up table of shape `(V, C)` = `(50257, 768)`. Given an integer token ID, it returns the corresponding row. During training, backpropagation updates only the rows that were looked up in each batch. At initialisation, rows are sampled from N(0, 1).

**`self.drop_emb = nn.Dropout(drop_rate)`**

Applied immediately after the embedding look-up. Randomly zeros entire embedding dimensions with probability `drop_rate`. Disabled during `model.eval()`.

**`self.trf_blocks = nn.Sequential(...)`**

`nn.Sequential` is a container that chains modules in order. The list comprehension `[TransformerBlock4(cfg) for _ in range(n_layers)]` creates 12 independent blocks (each with its own unique parameters). The `*` unpacks the list into positional arguments for `nn.Sequential`. In `forward`, calling `self.trf_blocks(x)` is equivalent to calling each block in order.

**`self.final_norm = RMSNorm(emb_dim)`**

A final RMSNorm is applied to the output of the last transformer block before the vocabulary projection. Without this, the output of the last transformer block may have a high or variable scale, which destabilises the logits.

**`self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)`**

Maps each token's hidden state `(C,)` → a score vector `(V,)` over the entire vocabulary. These scores are the raw **logits** — the probability of each next token is obtained by `softmax(logits)`.

`bias=False`: the bias would shift all 50257 scores uniformly and has no effect on relative rankings.

In many large models (GPT-2 included), `out_head.weight` is *weight-tied* to `tok_emb.weight` (they share the same matrix), halving the parameter count of the embedding+output layers. This implementation keeps them separate for clarity.

**`self.context_length = cfg["context_length"]`**

Stored as a plain Python integer attribute. External code (the training engine, the generation loop) needs to know the model's context window. In `GPTModel`, this was readable as `model.pos_emb.weight.shape[0]`; since `GPT4Model` has no `pos_emb`, this attribute is the equivalent.

### `forward(in_idx)` — step by step

```python
def forward(self, in_idx):
    tok_embeds = self.tok_emb(in_idx)
    x = self.drop_emb(tok_embeds)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
```

**Input** `in_idx` shape: `(B, T)` — integer token IDs. B is batch size, T is sequence length ≤ `context_length`.

**Line 1: `tok_embeds = self.tok_emb(in_idx)`**

Each integer in `in_idx` is replaced by its corresponding row in the embedding table. Output shape: `(B, T, C)` = `(B, T, 768)`.

**Line 2: `x = self.drop_emb(tok_embeds)`**

Embedding dropout. Shape unchanged: `(B, T, C)`.

**Line 3: `x = self.trf_blocks(x)`**

Passes x sequentially through all 12 `TransformerBlock4` modules. Each block preserves the shape `(B, T, C)`.

Inside each block, RoPE is applied to Q and K using the pre-computed cos/sin tables, which are automatically on the same device as x. No explicit position index is passed — the attention heads deduce position from the rotation.

**Line 4: `x = self.final_norm(x)`**

RMSNorm. Shape unchanged.

**Line 5: `logits = self.out_head(x)`**

Linear projection `(B, T, C)` → `(B, T, V)` = `(B, T, 50257)`. Each token position produces a 50257-dimensional score vector over the vocabulary.

**Line 6: `return logits`**

Raw unnormalised scores. During training, the cross-entropy loss function applies softmax internally. During generation, `softmax(logits[:, -1, :])` gives next-token probabilities.

---

## 9. `GPT4_CONFIG_SMALL` (lines 277–287)

```python
GPT4_CONFIG_SMALL = {
    "vocab_size":      50257,
    "context_length":  1024,
    "n_layers":        12,
    "n_heads":         12,
    "n_kv_heads":      4,
    "emb_dim":         768,
    "drop_rate":       0.1,
    "qkv_bias":        False,
    "rope_base":       10000,
}
```

| Key | Value | Meaning |
|---|---|---|
| `vocab_size` | 50257 | Number of tokens in the GPT-2 BPE tokenizer. Same as GPT-2 — reuses the same tokenizer for compatibility. |
| `context_length` | 1024 | Maximum sequence length. RoPE has better length generalisation than learned `pos_emb`, so this can be extended at inference time without retraining. |
| `n_layers` | 12 | Number of transformer blocks. Same as GPT-2 Small. |
| `n_heads` | 12 | Number of query attention heads. `head_dim = 768 / 12 = 64`. |
| `n_kv_heads` | 4 | Number of KV heads for GQA. `n_rep = 12 / 4 = 3` — each KV head is shared by 3 query heads. Reduces KV memory/compute by 3×. |
| `emb_dim` | 768 | Token embedding width. Same as GPT-2 Small (124M parameters). |
| `drop_rate` | 0.1 | 10% dropout on embeddings, attention weights, and residual branches. |
| `qkv_bias` | False | No bias terms in Q/K/V projections. Follows LLaMA design; saves a small number of parameters. |
| `rope_base` | 10000 | The θ₀ base for RoPE frequency computation. A higher value (e.g. 500000 in LLaMA 3) supports longer context. |

---

## 10. Architecture Comparison: GPT-2 vs GPT-4 style

| Component | GPT-2 (`gpt_model.py`) | GPT-4 style (`gpt_model4.py`) | Why changed |
|---|---|---|---|
| **Normalisation** | `LayerNorm` (mean + variance) | `RMSNorm` (RMS only, no mean) | ~30% cheaper; empirically equivalent quality |
| **Position encoding** | Learned `pos_emb`: `nn.Embedding(T, C)` added at input | `RoPE`: cos/sin rotation applied to Q and K inside each head | Encodes relative distance; better length generalisation; no trainable parameters |
| **Attention type** | Multi-Head Attention: H Q heads, H KV heads | Grouped Query Attention: H Q heads, G < H KV heads | Reduces KV-cache by H/G×; critical for long-context inference |
| **Q/K bias** | `qkv_bias=False` | `qkv_bias=False` | No change |
| **FFN activation** | `GELU` (Gaussian Error Linear Unit) | `SiLU` (Sigmoid Linear Unit) inside SwiGLU gate | More expressive gating; empirically better |
| **FFN structure** | 2 layers: `W₂(GELU(W₁(x)))` | 3 layers: `W_down(SiLU(W_gate(x)) ⊙ W_up(x))` | Gated structure; same parameter count with higher quality |
| **FFN hidden dim** | `4 × emb_dim` | `⌈8/3 × emb_dim⌉` rounded to 256 | Maintains ≈ same total parameters across 3 matrices |
| **Bias in FFN** | Yes (default `nn.Linear`) | No (`bias=False` everywhere) | LLaMA convention; saves parameters; negligible quality impact |
| **Positional parameters** | `T × C` = e.g. `256 × 768 = 196,608` parameters | 0 (RoPE is parameter-free) | Frees capacity for other parameters |
| **Output head bias** | No | No | No change |

---

## 11. Full Data Flow Diagram

```
INPUT
──────
in_idx : (B, T)    ← integer token IDs, e.g. shape (2, 256)

         │
         ▼ tok_emb: nn.Embedding(50257, 768)
         │
tok_embeds : (B, T, C)    = (2, 256, 768)

         │
         ▼ drop_emb: nn.Dropout(0.1)
         │
x : (B, T, C)             = (2, 256, 768)

         │
         ▼  ┌─────────────────────────────────────────────────┐
         │  │ TransformerBlock4  × n_layers (e.g. 12)         │
         │  │                                                  │
         │  │  shortcut = x                (B,T,C)            │
         │  │  x = RMSNorm(x)             (B,T,C)            │
         │  │  x = GroupedQueryAttention(x)                   │
         │  │   ├─ W_query(x)→(B,H,T,D)  H=12, D=64         │
         │  │   ├─ W_key(x)  →(B,G,T,D)  G=4,  D=64         │
         │  │   ├─ W_value(x)→(B,G,T,D)                      │
         │  │   ├─ apply_rope(Q, cos, sin) → (B,H,T,D)       │
         │  │   ├─ apply_rope(K, cos, sin) → (B,G,T,D)       │
         │  │   ├─ K.repeat_interleave(3) → (B,H,T,D)        │
         │  │   ├─ V.repeat_interleave(3) → (B,H,T,D)        │
         │  │   ├─ Q@Kᵀ → (B,H,T,T)  + causal mask          │
         │  │   ├─ softmax(·/√D)     → (B,H,T,T)             │
         │  │   ├─ dropout            → (B,H,T,T)             │
         │  │   └─ ·V → (B,H,T,D) → reshape → (B,T,C)        │
         │  │  x = dropout(x)         (B,T,C)                │
         │  │  x = x + shortcut       (B,T,C)  ← residual    │
         │  │                                                  │
         │  │  shortcut = x            (B,T,C)                │
         │  │  x = RMSNorm(x)         (B,T,C)                │
         │  │  x = FeedForwardSwiGLU(x)                       │
         │  │   ├─ SiLU(W_gate(x))   → (B,T,H_ff)            │
         │  │   ├─ W_up(x)           → (B,T,H_ff)            │
         │  │   ├─ gate ⊙ up         → (B,T,H_ff)            │
         │  │   └─ W_down(·)         → (B,T,C)               │
         │  │  x = dropout(x)         (B,T,C)                │
         │  │  x = x + shortcut       (B,T,C)  ← residual    │
         │  └─────────────────────────────────────────────────┘
         │
x : (B, T, C)             = (2, 256, 768)

         │
         ▼ final_norm: RMSNorm(768)
         │
x : (B, T, C)             = (2, 256, 768)

         │
         ▼ out_head: nn.Linear(768, 50257, bias=False)
         │
OUTPUT
──────
logits : (B, T, V)        = (2, 256, 50257)
         ← unnormalised score for each token at each position
         ← apply softmax(logits[:, -1, :]) for next-token probabilities
```

---

*Document generated for `gpt_model4.py` — GPT-4 style language model implementation.*
*Architecture references: RMSNorm (Zhang & Sennrich 2019), RoPE (Su et al. 2021), GQA (Ainslie et al. 2023), SwiGLU (Shazeer 2020).*
