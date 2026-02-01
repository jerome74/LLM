import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):  # Definisce MHA (self-attention) stile GPT
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()  # Inizializza modulo PyTorch

        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"  # Controllo: C % H == 0 per avere head_dim intero

        self.d_out = d_out  # Memorizza dimensione di uscita aggregata; qui C=768
        self.num_heads = num_heads  # Memorizza H; qui 12
        self.head_dim = d_out // num_heads  # Calcola D=64 (768//12); dimensione per testa

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Proiezione Q: input (B,T,C) -> (B,T,C); peso shape=(C,C)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # Proiezione K: (B,T,C) -> (B,T,C)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Proiezione V: (B,T,C) -> (B,T,C)
        self.out_proj = nn.Linear(d_out, d_out)  # Output proiezione finale dopo concat heads: (B,T,C) -> (B,T,C)
        self.dropout = nn.Dropout(dropout)  # Dropout per pesi di attenzione; p=0.1 come da config
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # Registra maschera causale 2D (T,T) con 1 sopra diagonale; dtype=float; shape=(T,T)= (256,256)

    def forward(self, x):
        B, T, C = x.shape  # Legge shape input: x (B,T,C) es. (2,256,768)

        keys = self.W_key(x)  # Applica linea K; output shape=(B,T,C)=(2,256,768)
        queries = self.W_query(x)  # Applica linea Q; output shape=(B,T,C)
        values = self.W_value(x)  # Applica linea V; output shape=(B,T,C)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (B, T, d_out) -> (B, T, num_heads, head_dim)
        keys = keys.view(B, T, self.num_heads, self.head_dim)  # Reshape K: (2,256,12,64)
        values = values.view(B, T, self.num_heads, self.head_dim)  # Reshape V: (2,256,12,64)
        queries = queries.view(B, T, self.num_heads, self.head_dim)  # Reshape Q: (2,256,12,64)

        # Transpose: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        keys = keys.transpose(1, 2)  # K: (2,12,256,64)
        queries = queries.transpose(1, 2)  # Q: (2,12,256,64)
        values = values.transpose(1, 2)  # V: (2,12,256,64)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # MatMul per testa: (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T); shape=(2,12,256,256)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:T, :T]  # Converte a bool e taglia a T attuale; shape=(T,T)=(256,256)

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # Applica maschera causale: pos future -> -inf (broadcast su B,H); attn_scores shape resta (2,12,256,256)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # Softmax scalata su ultima dim T: (2,12,256,256); ogni riga somma a 1
        attn_weights = self.dropout(attn_weights)  # Applica dropout sui pesi; shape invariata (2,12,256,256)

        # Shape: (B, T, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # MatMul: (B,H,T,T)@(B,H,T,D)->(B,H,T,D) poi transpose->(B,T,H,D)=(2,256,12,64)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(B, T, self.d_out)  # Concatena teste: (B,T,H*D)->(B,T,C)=(2,256,768)
        context_vec = self.out_proj(context_vec)  # Proiezione finale: (B,T,C)->(B,T,C)=(2,256,768)

        return context_vec  # Restituisce tensore contesto per il blocco; shape=(B,T,C)


class LayerNorm(nn.Module):  # LayerNorm manuale (senza peso/bias di nn.LayerNorm)
    def __init__(self, emb_dim):
        super().__init__()  # Inizializzazione modulo
        self.eps = 1e-5  # Epsilon numerico per stabilità
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Parametro gamma (scaling); shape=(C,)=(768,)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Parametro beta (shift); shape=(C,)=(768,)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Media su dimensione embedding; shape=(B,T,1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Varianza su embedding; shape=(B,T,1)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # Normalizza; shape=(B,T,C)
        return self.scale * norm_x + self.shift  # Applica affine; broadcasting su (C,)->(B,T,C); ritorna shape=(B,T,C)


class GELU(nn.Module):  # Implementazione GELU (approssimazione tanh)
    def __init__(self):
        super().__init__()  # Inizializzazione modulo

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(  # Restituisce GELU(x); shape equal a x
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # Costante sqrt(2/pi); scalar
            (x + 0.044715 * torch.pow(x, 3))  # Approssimazione polinomiale; shape=x
        ))  # Output shape=(...) uguale alla shape di x (tipicamente (B,T,4*C) dentro FFN)


class FeedForward(nn.Module):  # MLP a 2 layer: C -> 4C -> C
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        self.layers = nn.Sequential(  # Sequenza di layer (MLP)
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Linear1: (B,T,C)->(B,T,4C)=(B,T,3072)
            GELU(),  # Attivazione GELU: mantiene shape (B,T,4C)
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Linear2: (B,T,4C)->(B,T,C)=(B,T,768)
        )

    def forward(self, x):
        return self.layers(x)  # Output shape=(B,T,C), stessa shape dell’input x


class TransformerBlock(nn.Module):  # Blocco Transformer pre-norm: Attenzione + FFN con shortcut
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        self.att = MultiHeadAttention(  # Istanzia MHA
            d_in=cfg["emb_dim"],  # C=768
            d_out=cfg["emb_dim"],  # C=768
            context_length=cfg["context_length"],  # T=256
            num_heads=cfg["n_heads"],  # H=12
            dropout=cfg["drop_rate"],  # p=0.1
            qkv_bias=cfg["qkv_bias"])  # False
        self.ff = FeedForward(cfg)  # Istanzia MLP: (B,T,C)->(B,T,C)
        self.norm1 = LayerNorm(cfg["emb_dim"])  # LayerNorm prima di MHA; output (B,T,C)
        self.norm2 = LayerNorm(cfg["emb_dim"])  # LayerNorm prima di FFN; output (B,T,C)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout sul ramo principale; non cambia shape

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x  # Salva input per residual; shape=(B,T,C)
        x = self.norm1(x)  # Pre-norm: (B,T,C)->(B,T,C)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]; (B,T,C)->(B,T,C)
        x = self.drop_shortcut(x)  # Applica dropout; shape invariata (B,T,C)
        x = x + shortcut  # Residual add: (B,T,C)+(B,T,C)->(B,T,C)

        # Shortcut connection for feed-forward block
        shortcut = x  # Nuovo residual; shape=(B,T,C)
        x = self.norm2(x)  # Pre-norm: (B,T,C)->(B,T,C)
        x = self.ff(x)  # MLP: (B,T,C)->(B,T,C)
        x = self.drop_shortcut(x)  # Dropout; shape invariata (B,T,C)
        x = x + shortcut  # Residual add: (B,T,C)->(B,T,C)

        return x  # Ritorna feature aggiornate: shape=(B,T,C)


class GPTModel(nn.Module):  # Modello GPT semplificato (stack di TransformerBlock)
    def __init__(self, cfg):
        super().__init__()  # Inizializzazione modulo
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # Lookup embedding: (B,T) -> (B,T,C) con V=50257, C=768
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # Positional embedding: (T)->(T,C) con T<=256
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout emb; shape invariata

        self.trf_blocks = nn.Sequential(  # Sequenza di n_layers TransformerBlock
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 12 blocchi; input/output (B,T,C)

        self.final_norm = LayerNorm(cfg["emb_dim"])  # LayerNorm finale: (B,T,C)->(B,T,C)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # Head: (B,T,C)->(B,T,V)=(B,T,50257)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # Legge shape input index: (B,T), es. (2,256)
        tok_embeds = self.tok_emb(in_idx)  # Token embedding: (B,T)->(B,T,C)=(2,256,768)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # Pos embed: (T)->(T,C)=(256,768)
        x = tok_embeds + pos_embeds  # Somma con broadcasting su B: (B,T,C)+(T,C)->(B,T,C)
        x = self.drop_emb(x)  # Dropout: shape (B,T,C)
        x = self.trf_blocks(x)  # Passa attraverso i blocchi Transformer: (B,T,C)->(B,T,C)
        x = self.final_norm(x)  # LayerNorm finale: shape (B,T,C)
        logits = self.out_head(x)  # Proiezione a vocab: (B,T,V)=(2,256,50257)
        return logits  # Restituisce logits non normalizzati per ogni token
