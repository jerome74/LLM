
import os  # Importa il modulo OS per operazioni su file/percorso; non crea tensori

import requests  # Libreria HTTP per scaricare il testo (dataset); non crea tensori

import torch  # Importa PyTorch (base di tensori/NN); definisce torch.Tensor e funzioni core

import tiktoken  # Tokenizer GPT-2 (encoding/decoding token IDs); usato per tokenizzare il testo

import torch.nn as nn  # Sotto-modulo di PyTorch per layer neurali (Linear, Embedding, ecc.)

from torch.utils.data import Dataset, DataLoader  # Strutture dati PyTorch per dataset e batch loader


class GPTDatasetV1(Dataset):  # Definisce un dataset custom per training GPT (token-level)
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # Lista che conterrà tensori di input; ogni item shape=(max_length,)
        self.target_ids = []  # Lista che conterrà tensori target (shiftati); ogni item shape=(max_length,)

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"&lt;|endoftext|&gt;"})  # Converte txt->lista di int; lunghezza L variabile

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):  # Scorre con finestra di lunghezza max_length, passo 'stride'; num_chunks ≈ (L-max_length)/stride
            input_chunk = token_ids[i:i + max_length]  # Slice di input; shape logica=(max_length,) (lista di int)
            target_chunk = token_ids[i + 1: i + max_length + 1]  # Slice target shiftata di +1; shape logica=(max_length,)
            self.input_ids.append(torch.tensor(input_chunk))  # Converte in tensor; shape=(max_length,), dtype=long
            self.target_ids.append(torch.tensor(target_chunk))  # Converte in tensor; shape=(max_length,), dtype=long

    def __len__(self):
        return len(self.input_ids)  # Numero di esempi nel dataset; intero ≈ num_chunks calcolati

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # Restituisce coppia (input,target); ciascuno shape=(max_length,)


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # Ottiene encoder GPT-2; mapping string->token IDs

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Istanzia dataset; ogni item ha (input,target) con shape=(max_length,)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)  # Crea loader; batch produce: input_batch shape=(B,T), target_batch shape=(B,T)

    return dataloader  # Restituisce DataLoader; iterabile su batch


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
        x = self.trf_blocks(x)  # Passa 12 blocchi: shape resta (B,T,C)
        x = self.final_norm(x)  # LayerNorm finale: shape (B,T,C)
        logits = self.out_head(x)  # Proiezione a vocab: (B,T,V)=(2,256,50257)
        return logits  # Restituisce logits non normalizzati per ogni token


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):  # Itera per generare max_new_tokens; loop di lunghezza 50 nell’uso sotto

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]  # Taglia contesto a ultimi 'context_size' token; shape=(B, min(T,context_size))
        # Get the predictions
        with torch.no_grad():  # Disattiva grad; inferenza
            logits = model(idx_cond)  # Forward: (B,t)->(B,t,V); t=idx_cond.shape[1]

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  # Prende ultimi logits per step corrente: shape=(B,V)=(B,50257)

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Argmax su vocab: shape=(B,1); contiene ID token scelto greedy

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Concatena al contesto: shape=(B, T+1) cresce di 1 ogni ciclo

    return idx  # Restituisce sequenza estesa con max_new_tokens in più; shape=(B, T+max_new_tokens)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, tok_k=None, eos_id=None):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():  # Disattiva grad; inferenza
            logits = model(idx_cond)  # Forward: (B,t)->(B,t,V); t=idx_cond.shape[1]
    logits = logits[:, -1, :]
    if tok_k is not None:
      top_k_logits, _ = torch.topk(logits, tok_k)
      min_val = top_k_logits[:,-1]
      logits = torch.where( condition=logits < min_val
                   , input=torch.tensor(float('-inf')).to(logits.device)
                   , other=logits )
    if temperature > 0:
      probs = torch.softmax(logits / temperature, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
    else:
      idx_next = torch.argmax(logits, dim=-1, keepdim=True)
      
    if eos_id == idx_next:
      break
    idx = torch.cat((idx, idx_next), dim=1)
  return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)  # Converte stringa->lista di token IDs; lunghezza variabile n
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension  # Converte in tensor long e aggiunge batch: shape=(1,n)
    return encoded_tensor  # Ritorna (B=1, T=n)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension  # Rimuove dimensione batch: shape=(T,)
    return tokenizer.decode(flat.tolist())  # Decodifica IDs->stringa; output: testo generato


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # Mappa a device (cpu/cuda); shape=(B,T) per entrambi
    logits = model(input_batch)  # Forward: (B,T)->(B,T,V)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # CE su classi V: logits (B*T,V), target (B*T,); scalare
    return loss  # Restituisce loss scalar tensor (shape=())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # Accumulatore float
    if len(data_loader) == 0:  # Se loader vuoto
        return float("nan")  # Restituisce NaN
    elif num_batches is None:
        num_batches = len(data_loader)  # Valuta tutte le batch disponibili
    else:
        num_batches = min(num_batches, len(data_loader))  # Limita a num_batches richieste
    for i, (input_batch, target_batch) in enumerate(data_loader):  # Itera sulle batch: input/target shape=(B,T)
        if i < num_batches:  # Se entro limite
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Calcola loss per batch; scalare
            total_loss += loss.item()  # Somma float
        else:
            break  # Esce dal loop
    return total_loss / num_batches  # Restituisce media loss (float)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # Modalità eval (dropout disattivato, BN eval)
    with torch.no_grad():  # No grad
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # Loss media su train (limita a eval_iter batch); float
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # Loss media su val; float
    model.train()  # Ritorna in train mode
    return train_loss, val_loss  # Ritorna tuple di float


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # Modalità eval
    context_size = model.pos_emb.weight.shape[0] # Determina context length dal pos_emb: T=256
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # Tokenizza contesto iniziale; shape=(1,n)
    with torch.no_grad():  # No grad
        token_ids = generate_text_simple(  # Genera 50 token aggiuntivi
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )  # Output token_ids shape=(1, n+50)
        decoded_text = token_ids_to_text(token_ids, tokenizer)  # Decodifica in stringa; lunghezza ≈ n+50 token (convertiti in testo)
        print(decoded_text.replace("\n", " "))  # Compact print format  # Stampa testo generato su una riga
    model.train()  # Ritorna in train mode


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # Re-definizione identica alla precedente; sovrascrive la funzione (stessa logica)

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Crea dataset; ogni sample shape=(T,)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)  # Batch shape: (B,T) per input e target

    return dataloader  # Restituisce DataLoader


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []  # Liste di float (progressi di training)
    tokens_seen = 0  # Contatore token visti (int)
    global_step = -1  # Step globale inizializzato a -1 (incrementato ogni batch)

    # Main training loop
    for epoch in range(num_epochs):  # Loop su epoche; num_epochs=10
        model.train()  # Set model to training mode  # Attiva dropout

        for input_batch, target_batch in train_loader:  # Itera su batch di train; input/target shape=(B,T) (es B=2, T=256)
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration  # Azzeramento grad
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Calcola CE loss; scalare
            loss.backward()  # Calculate loss gradients  # Backprop su parametri
            optimizer.step()  # Update model weights using loss gradients  # Aggiorna pesi (AdamW)
            tokens_seen += input_batch.numel()  # Aggiorna contatore: numel(B,T)=B*T (es 512 per batch B=2, T=256)
            global_step += 1  # Incrementa step globale di 1

            # Optional evaluation step
            if global_step % eval_freq == 0:  # Valutazione ogni 'eval_freq' step; qui ogni 5 batch
                train_loss, val_loss = evaluate_model(  # Calcola train/val loss (su eval_iter batch); float
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)  # Logga train loss
                val_losses.append(val_loss)  # Logga val loss
                track_tokens_seen.append(tokens_seen)  # Logga token visti finora
                print(f"Ep {epoch+1} (Step {global_step:06d}): "  # Stampa stato
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")  # Esempio output: "Ep 1 (Step 000005): Train loss 6.123, Val loss 6.210"

        # Print a sample text after each epoch
        generate_and_print_sample(  # Genera e stampa sample di testo (50 token) da start_context
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen  # Restituisce liste dei progressi + token visti


def main(gpt_config, settings):

    torch.manual_seed(123)  # Fissa il seed per riproducibilità (parametri iniziali, dropout deterministico dove applicabile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Seleziona device; tipicamente 'cuda' se disponibile, altrimenti 'cpu'

    ##############################
    # Download data if necessary
    ##############################

    file_path = "genesis.txt"  # Percorso locale del file dataset
    url = "https://people.sc.fsu.edu/~jburkardt/datasets/text/genesis.txt"  # URL sorgente testo (Libro della Genesi in inglese)

    if not os.path.exists(file_path):  # Se file non presente localmente
        response = requests.get(url, timeout=30)  # Scarica via HTTP con timeout 30s; response.status_code=200 atteso
        response.raise_for_status()  # Solleva eccezione se status non OK
        text_data = response.text  # Contenuto testo; stringa, lunghezza L caratteri (poi tokenizzata)
        with open(file_path, "w", encoding="utf-8") as file:  # Apre file in scrittura
            file.write(text_data)  # Scrive il testo
    else:
        with open(file_path, "r", encoding="utf-8") as file:  # Apre file esistente in lettura
            text_data = file.read()  # Legge contenuto in stringa

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)  # Istanzia modello GPT; parametri da gpt_config (V=50257, T=256, C=768, H=12, L=12)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes  # Sposta parametri su device (cuda/cpu); non cambia shape dei tensori runtime
    optimizer = torch.optim.AdamW(  # Crea ottimizzatore AdamW
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]  # lr=5e-4, weight_decay=0.1
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90  # 90% train, 10% validation
    split_idx = int(train_ratio * len(text_data))  # Split su caratteri; indice di split (int)

    train_loader = create_dataloader_v1(  # DataLoader train
        text_data[:split_idx],  # Sotto-stringa train
        batch_size=settings["batch_size"],  # B=2
        max_length=gpt_config["context_length"],  # T=256
        stride=gpt_config["context_length"],  # stride=T (chunk non sovrapposti)
        drop_last=True,  # Droppa ultimo batch se incompleto
        shuffle=True,  # Shuffling delle chunk
        num_workers=0  # Caricamento in main thread
    )

    val_loader = create_dataloader_v1(  # DataLoader validation
        text_data[split_idx:],  # Sotto-stringa val
        batch_size=settings["batch_size"],  # B=2
        max_length=gpt_config["context_length"],  # T=256
        stride=gpt_config["context_length"],  # stride=T
        drop_last=False,  # Mantiene ultimo batch anche se incompleto
        shuffle=False,  # Niente shuffle per validazione
        num_workers=0  # Main thread
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")  # Tokenizer per generazione/decoding; GPT-2

    train_losses, val_losses, tokens_seen = train_model_simple(  # Avvia training semplice; restituisce curve di loss e token visti
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,  # 10 epoche, eval ogni 5 step, 1 batch valutata
        start_context="And Cainan lived", tokenizer=tokenizer  # Prompt iniziale per sample; stampa dopo ogni epoca
    )

    return train_losses, val_losses, tokens_seen, model  # Restituisce risultati di training + modello addestrato


if __name__ == "__main__":  # Esecuzione script principale

    GPT_CONFIG_124M = {  # Config stile GPT-2 Small (≈124M parametri con context_length ridotto)
        "vocab_size": 50257,    # Vocabulary size  # V=50257
        "context_length": 256,  # Shortened context length (orig: 1024)  # T=256
        "emb_dim": 768,         # Embedding dimension  # C=768
        "n_heads": 12,          # Number of attention heads  # H=12
        "n_layers": 12,         # Number of layers  # L=12
        "drop_rate": 0.1,       # Dropout rate  # p=0.1
        "qkv_bias": False       # Query-key-value bias  # No bias nelle proiezioni Q/K/V
    }

    OTHER_SETTINGS = {  # Hyperparametri di training
        "learning_rate": 5e-4,  # LR dell’ottimizzatore
        "num_epochs": 10,       # Numero epoche
        "batch_size": 2,        # B=2
        "weight_decay": 0.1     # Regolarizzazione L2 tipo AdamW
    }

    ###########################
    # Initiate training
    ###########################

    #train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)  # Avvia pipeline: download/costruzione dataset/training; stampa log e sample per epoca

    model = GPTModel(GPT_CONFIG_124M)

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(model=model
                        , idx=text_to_token_ids("And Cainan lived", tokenizer)
                        , max_new_tokens=15
                        , context_size=GPT_CONFIG_124M["context_length"]
                        , tok_k=25
                        , temperature=1.4)
    print('Output text: ',token_ids_to_text(token_ids, tokenizer))