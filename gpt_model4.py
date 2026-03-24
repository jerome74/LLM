import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm  (Root Mean Square Layer Normalisation)
# Replaces LayerNorm: removes the mean-centring step, keeping only the RMS
# scaling.  Cheaper to compute and empirically as good for language modelling.
# Paper: "Root Mean Square Layer Normalization", Zhang & Sennrich 2019.
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):  # RMSNorm: normalizza x per la sua radice-quadratica-media
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()  # Inizializzazione modulo
        self.eps = eps  # Epsilon per stabilità numerica; evita divisione per zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Parametro gamma apprendibile; shape=(C,)=(768,)

    def forward(self, x):
        # x shape: (B, T, C)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # RMS per token: sqrt(mean(x²)); shape=(B,T,1)
        norm_x = x / rms  # Normalizza dividendo per la RMS; shape=(B,T,C)
        return self.scale * norm_x  # Applica scaling affine gamma; shape=(B,T,C); nessun bias (β) come in GPT-4


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding  (RoPE)
# Instead of adding learned absolute position vectors at the input, RoPE
# rotates query and key vectors inside each attention head using position-
# dependent angles.  This encodes *relative* distance, which generalises
# better to longer sequences than absolute embeddings.
# Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
#        Su et al. 2021.  Used in LLaMA, GPT-NeoX, PaLM, etc.
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim, context_length, base=10000):
    """Precalcola le frequenze cos/sin per RoPE fino a context_length token.

    Ogni coppia di dimensioni (2i, 2i+1) di una testa viene ruotata di un angolo
    theta_i * position, dove theta_i = 1 / (base^(2i/D)).
    Restituisce cos e sin di shape (context_length, head_dim//2).
    """
    # theta: inversione di frequenza per ogni coppia di dim; shape=(head_dim//2,)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # freq decrescenti; es. 1/(10000^(0/64)), 1/(10000^(2/64)),...

    # Posizioni assolute 0..T-1; shape=(context_length,)
    positions = torch.arange(context_length, dtype=torch.float32)  # es. [0, 1, 2, ..., 255]

    # Prodotto esterno posizioni × theta: (T,) x (D/2,) -> (T, D/2)
    freqs = torch.outer(positions, theta)  # freqs[t, i] = t * theta_i; shape=(context_length, head_dim//2)

    cos = torch.cos(freqs)  # shape=(context_length, head_dim//2)
    sin = torch.sin(freqs)  # shape=(context_length, head_dim//2)
    return cos, sin  # Buffer registrati in GroupedQueryAttention


def apply_rope(x, cos, sin):
    """Applica RoPE a un tensore di query o key.

    x: (B, H, T, D)  — query o key di una testa
    cos, sin: (context_length, D//2) — buffer pre-calcolati
    Ritorna x ruotato, stessa shape (B, H, T, D).
    """
    _B, _H, T, D = x.shape  # Legge shape; T <= context_length; D = head_dim
    half = D // 2  # Metà delle dimensioni per testa; es. 32 se head_dim=64

    x1 = x[..., :half]   # Prima metà delle dim per testa: (_, _, T, D//2)
    x2 = x[..., half:]   # Seconda metà: (_, _, T, D//2)

    # Seleziona solo le T posizioni e broadcast su B, H
    c = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2) — broadcast su B e H
    s = sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)

    # Rotazione 2D: [x1, x2] * cos + [-x2, x1] * sin
    rotated = torch.cat([
        x1 * c - x2 * s,   # parte reale della rotazione; shape=(B,H,T,D//2)
        x2 * c + x1 * s,   # parte immaginaria; shape=(B,H,T,D//2)
    ], dim=-1)  # Ricongiungi le due metà; shape=(B,H,T,D) = shape originale

    return rotated  # Shape invariata (B, H, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Query Attention  (GQA)
# Standard Multi-Head Attention uses H query heads, H key heads, H value heads.
# GQA uses H query heads but only G < H key/value heads (G divides H evenly).
# Each KV head is shared among H/G query heads.  This reduces the KV-cache size
# and attention compute by a factor of H/G while barely hurting quality.
# Paper: "GQA: Training Generalised Multi-Query Transformer Models from
#         Multi-Head Checkpoints", Ainslie et al. 2023.
# Used in: LLaMA 2/3, Mistral, Gemma, etc.
# ─────────────────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):  # GQA: n_heads Q, n_kv_heads K/V con RoPE
    def __init__(self, d_in, d_out, context_length, dropout,
                 n_heads, n_kv_heads, rope_base=10000, qkv_bias=False):
        super().__init__()  # Inizializzazione modulo

        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"  # Controllo D/H intero
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"  # Controllo GQA valido

        self.d_out = d_out          # Dimensione uscita aggregata; es. C=768
        self.n_heads = n_heads      # Numero teste query; es. H=12
        self.n_kv_heads = n_kv_heads  # Numero teste K/V (< H per GQA); es. G=4
        self.n_rep = n_heads // n_kv_heads  # Query heads per KV head; es. 12//4=3
        self.head_dim = d_out // n_heads    # Dimensione per testa; es. D=64

        # Proiezioni senza bias (come LLaMA); nessun parametro extra
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Q: (B,T,C) -> (B,T,H*D)=(B,T,768)
        self.W_key   = nn.Linear(d_in, n_kv_heads * self.head_dim, bias=qkv_bias)  # K: (B,T,C) -> (B,T,G*D) es.(B,T,256)
        self.W_value = nn.Linear(d_in, n_kv_heads * self.head_dim, bias=qkv_bias)  # V: (B,T,C) -> (B,T,G*D)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)  # Output proiezione: (B,T,C) -> (B,T,C); senza bias
        self.dropout = nn.Dropout(dropout)  # Dropout pesi attenzione

        # Maschera causale: 1 sopra la diagonale principale (token futuri = -inf)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # shape=(T,T)=(context_length, context_length)

        # Frequenze RoPE pre-calcolate e registrate come buffer (non parametri)
        cos, sin = precompute_rope_freqs(self.head_dim, context_length, rope_base)
        self.register_buffer("rope_cos", cos)  # shape=(T, head_dim//2)
        self.register_buffer("rope_sin", sin)  # shape=(T, head_dim//2)

    def forward(self, x):
        B, T, _C = x.shape  # Input: (B,T,C) es. (2,256,768)

        # ── Proiezioni Q, K, V ──────────────────────────────────────────────
        queries = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: (B,T,C) -> (B,T,H,D) -> (B,H,T,D) = (2,12,256,64)

        keys = self.W_key(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # K: (B,T,C) -> (B,T,G,D) -> (B,G,T,D) = (2,4,256,64)

        values = self.W_value(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # V: (B,T,C) -> (B,T,G,D) -> (B,G,T,D) = (2,4,256,64)

        # ── Applica RoPE a queries e keys ────────────────────────────────────
        queries = apply_rope(queries, self.rope_cos, self.rope_sin)  # (B,H,T,D) — rotazione posizionale
        keys    = apply_rope(keys,    self.rope_cos, self.rope_sin)  # (B,G,T,D) — rotazione posizionale

        # ── Espandi K e V da G teste a H teste (GQA repeat) ─────────────────
        # repeat_interleave: (B,G,T,D) -> (B,H,T,D) ripetendo ogni KV head n_rep volte
        keys   = keys.repeat_interleave(self.n_rep, dim=1)    # (B,H,T,D)=(2,12,256,64)
        values = values.repeat_interleave(self.n_rep, dim=1)  # (B,H,T,D)=(2,12,256,64)

        # ── Scaled dot-product attention con maschera causale ────────────────
        attn_scores = queries @ keys.transpose(2, 3)  # (B,H,T,D)@(B,H,D,T) -> (B,H,T,T)=(2,12,256,256)

        mask_bool = self.mask.bool()[:T, :T]  # Maschera causale bool (T,T); 1=futuro da mascherare
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # Token futuri -> -inf (invariante di shape)

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)  # Softmax scalata; shape=(B,H,T,T)
        attn_weights = self.dropout(attn_weights)  # Dropout; shape invariata

        # ── Aggregazione e output ─────────────────────────────────────────────
        context_vec = (attn_weights @ values).transpose(1, 2)  # (B,H,T,T)@(B,H,T,D)->(B,H,T,D); transpose->(B,T,H,D)
        context_vec = context_vec.reshape(B, T, self.d_out)    # Concatena teste: (B,T,H*D)=(B,T,768)
        context_vec = self.out_proj(context_vec)                # Proiezione finale: (B,T,C)->(B,T,C)

        return context_vec  # shape=(B,T,C)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU  (Swish-Gated Linear Unit)
# Replaces the two-layer GELU MLP.  Uses THREE linear projections:
#   gate(x) = SiLU( W_gate(x) )   ← gating signal
#   up(x)   =       W_up(x)        ← content signal
#   output  = W_down( gate ⊙ up )  ← projected back to emb_dim
# The product of a sigmoid-like gate and a linear path is empirically better
# than GELU with 4×expansion.  The hidden dim uses 8/3×emb_dim (≈ same
# parameter count as a 4×GELU MLP when counting all three matrices).
# Paper: "GLU Variants Improve Transformer", Noam Shazeer 2020.
# Used in: LLaMA, PaLM, GPT-NeoX, etc.
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardSwiGLU(nn.Module):  # FFN SwiGLU: 3 linear, hidden_dim=round(8/3*C) al multiplo di 256
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        emb_dim = cfg["emb_dim"]  # Dimensione embedding C; es. 768

        # hidden_dim = ceil(8/3 * C) arrotondato al multiplo di 256
        # Es. emb_dim=256 -> raw=682 -> hidden_dim=768; emb_dim=768 -> raw=2048 -> hidden_dim=2048
        hidden_dim = int(math.ceil((8 / 3 * emb_dim) / 256) * 256)  # Multiplo di 256 per efficienza hardware

        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False)  # W_gate: (B,T,C) -> (B,T,H_ff); senza bias
        self.up   = nn.Linear(emb_dim, hidden_dim, bias=False)  # W_up:   (B,T,C) -> (B,T,H_ff); senza bias
        self.down = nn.Linear(hidden_dim, emb_dim, bias=False)  # W_down: (B,T,H_ff) -> (B,T,C); senza bias

    def forward(self, x):
        # Calcola gate con attivazione SiLU (≈ Swish) e moltiplica element-wise con up
        gated = F.silu(self.gate(x)) * self.up(x)  # (B,T,H_ff); gating: SiLU(W_gate(x)) ⊙ W_up(x)
        return self.down(gated)  # Proiezione di ritorno: (B,T,C)


# ─────────────────────────────────────────────────────────────────────────────
# TransformerBlock4  —  GPT-4 style: RMSNorm + GQA(RoPE) + SwiGLU
# Same pre-norm + residual structure as GPT-2, but every component is upgraded.
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock4(nn.Module):  # Blocco Transformer GPT-4 style: pre-RMSNorm + GQA + SwiGLU
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        self.att = GroupedQueryAttention(  # GQA con RoPE
            d_in=cfg["emb_dim"],                      # C=768
            d_out=cfg["emb_dim"],                     # C=768
            context_length=cfg["context_length"],     # T=256
            dropout=cfg["drop_rate"],                 # p=0.1
            n_heads=cfg["n_heads"],                   # H=12
            n_kv_heads=cfg["n_kv_heads"],             # G=4 (GQA)
            rope_base=cfg.get("rope_base", 10000),    # theta RoPE; 10000 default
            qkv_bias=cfg.get("qkv_bias", False),      # no bias
        )
        self.ff    = FeedForwardSwiGLU(cfg)             # FFN SwiGLU: (B,T,C)->(B,T,C)
        self.norm1 = RMSNorm(cfg["emb_dim"])            # Pre-norm prima di GQA
        self.norm2 = RMSNorm(cfg["emb_dim"])            # Pre-norm prima di FFN
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout sul ramo principale

    def forward(self, x):
        # ── Blocco Attenzione ────────────────────────────────────────────────
        shortcut = x                     # Salva residual; shape=(B,T,C)
        x = self.norm1(x)               # Pre-RMSNorm: (B,T,C)->(B,T,C)
        x = self.att(x)                 # GQA+RoPE: (B,T,C)->(B,T,C)
        x = self.drop_shortcut(x)       # Dropout; shape invariata
        x = x + shortcut                # Connessione residuale: (B,T,C)+(B,T,C)->(B,T,C)

        # ── Blocco Feed-Forward ──────────────────────────────────────────────
        shortcut = x                     # Nuovo residual; shape=(B,T,C)
        x = self.norm2(x)               # Pre-RMSNorm: (B,T,C)->(B,T,C)
        x = self.ff(x)                  # SwiGLU FFN: (B,T,C)->(B,T,C)
        x = self.drop_shortcut(x)       # Dropout; shape invariata
        x = x + shortcut                # Connessione residuale: (B,T,C)->(B,T,C)

        return x  # Feature aggiornate; shape=(B,T,C)


# ─────────────────────────────────────────────────────────────────────────────
# GPT4Model  —  full model (no learned positional embedding; RoPE is implicit)
# Key difference from GPTModel: tok_emb only, no pos_emb.  Position is encoded
# by rotating Q/K vectors inside each attention head (RoPE).
# ─────────────────────────────────────────────────────────────────────────────

class GPT4Model(nn.Module):  # Modello GPT-4 style: token emb + stack TransformerBlock4 + RMSNorm
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # Token embedding: (B,T)->(B,T,C); V=50257, C=768
        # Nessun pos_emb: la posizione è codificata da RoPE dentro ogni testa di attenzione
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout sugli embedding; shape invariata

        self.trf_blocks = nn.Sequential(  # Stack di n_layers TransformerBlock4
            *[TransformerBlock4(cfg) for _ in range(cfg["n_layers"])]  # es. 12 blocchi; input/output (B,T,C)
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])  # RMSNorm finale prima della testa: (B,T,C)->(B,T,C)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # Head vocab: (B,T,C)->(B,T,V)=(B,T,50257)

        # Memorizza context_length come attributo per accesso esterno (sostituisce pos_emb.weight.shape[0])
        self.context_length = cfg["context_length"]  # es. 256; usato dal loop di generazione

    def forward(self, in_idx):
        # in_idx shape: (B, T) es. (2, 256) — indici token interi
        tok_embeds = self.tok_emb(in_idx)   # Token embedding: (B,T)->(B,T,C)=(2,256,768)
        x = self.drop_emb(tok_embeds)       # Dropout: shape (B,T,C)
        x = self.trf_blocks(x)              # Stack TransformerBlock4: (B,T,C)->(B,T,C)
        x = self.final_norm(x)              # RMSNorm finale: shape (B,T,C)
        logits = self.out_head(x)           # Proiezione vocab: (B,T,V)=(2,256,50257)
        return logits                        # Logits non normalizzati per ogni token


# ─────────────────────────────────────────────────────────────────────────────
# Default GPT-4 style config  (ridimensionato per training educativo)
# ─────────────────────────────────────────────────────────────────────────────

GPT4_CONFIG_SMALL = {
    "vocab_size":      50257,   # GPT-2 tokenizer (stesso di GPT-2 per semplicità)
    "context_length":  1024,    # Finestra di contesto; RoPE scala meglio di pos_emb
    "n_layers":        12,      # Transformer blocks
    "n_heads":         12,      # Teste query
    "n_kv_heads":      4,       # Teste KV (GQA: 1 KV head per ogni 3 Q heads)
    "emb_dim":         768,     # Dimensione embedding
    "drop_rate":       0.1,     # Dropout
    "qkv_bias":        False,   # Nessun bias in QKV (come LLaMA)
    "rope_base":       10000,   # Theta RoPE (base frequenza)
}
