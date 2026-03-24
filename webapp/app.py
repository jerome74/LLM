"""Flask app — GPT Web Training Console."""

import json
import os
import queue
import sys
import threading

import psutil
import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint_manager import get_device, load_checkpoint, save_checkpoint
from state import state
from training_engine import (
    build_dataloaders,
    build_model_and_optimizer,
    count_parameters,
    start_training_thread,
)

app = Flask(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUILTIN_CORPORA = {
    "inferno.txt": os.path.join(DATA_DIR, "inferno.txt"),
    "commedia.txt": os.path.join(DATA_DIR, "commedia.txt"),
}


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    device_label = str(get_device()).upper()
    return render_template("index.html", device=device_label)


@app.route("/training")
def training():
    return render_template("training.html")


@app.route("/generate")
def generate_page():
    return render_template("generate.html")


# ---------------------------------------------------------------------------
# SSE — live loss stream
# ---------------------------------------------------------------------------


@app.route("/api/events")
def events():
    def event_stream():
        while True:
            try:
                msg = state.result_queue.get(timeout=1.0)

                if msg["type"] == "loss":
                    state.steps.append(msg["step"])
                    state.train_losses.append(msg["train_loss"])
                    state.val_losses.append(msg["val_loss"])
                    state.tokens_seen = msg["tokens_seen"]
                    state.current_epoch = msg["epoch"]
                    state.current_step = msg["step"]

                elif msg["type"] == "sample":
                    state.last_sample = msg["text"]
                    state.current_epoch = msg["epoch"]

                elif msg["type"] in ("done", "stopped", "error"):
                    state.training_active = False
                    yield f"data: {json.dumps(msg)}\n\n"
                    break

                yield f"data: {json.dumps(msg)}\n\n"

            except queue.Empty:
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Training control
# ---------------------------------------------------------------------------


@app.route("/api/train/start", methods=["POST"])
def train_start():
    if state.training_active:
        return jsonify({"error": "Training already running"}), 400

    data = request.get_json()

    model_type = data.get("model_type", "gpt2")  # "gpt2" or "gpt4"

    gpt_config = {
        "vocab_size": 50257,
        "context_length": int(data.get("context_length", 256)),
        "n_layers": int(data.get("n_layers", 4)),
        "n_heads": int(data.get("n_heads", 4)),
        "emb_dim": int(data.get("emb_dim", 256)),
        "drop_rate": float(data.get("drop_rate", 0.1)),
        "qkv_bias": False,
    }

    if model_type == "gpt4":
        gpt_config["n_kv_heads"] = int(data.get("n_kv_heads", max(1, gpt_config["n_heads"] // 3)))
        gpt_config["rope_base"] = int(data.get("rope_base", 10000))

    if gpt_config["emb_dim"] % gpt_config["n_heads"] != 0:
        return jsonify({"error": "emb_dim must be divisible by n_heads"}), 400

    if model_type == "gpt4" and gpt_config["n_heads"] % gpt_config["n_kv_heads"] != 0:
        return jsonify({"error": "n_heads must be divisible by n_kv_heads"}), 400

    # Corpus: inline text takes priority, then built-in file
    corpus_text = data.get("corpus_text")
    if not corpus_text:
        fname = data.get("corpus_filename", "")
        path = BUILTIN_CORPORA.get(fname)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                corpus_text = fh.read()

    if not corpus_text:
        return jsonify({"error": "Corpus not found or empty"}), 400

    model, optimizer, device = build_model_and_optimizer(
        gpt_config,
        learning_rate=float(data.get("learning_rate", 5e-4)),
        weight_decay=float(data.get("weight_decay", 0.1)),
        model_type=model_type,
    )
    train_loader, val_loader = build_dataloaders(
        corpus_text,
        batch_size=int(data.get("batch_size", 4)),
        context_length=gpt_config["context_length"],
        train_ratio=float(data.get("train_ratio", 0.9)),
    )

    state.model = model
    state.optimizer = optimizer
    state.device = device
    state.gpt_config = gpt_config
    state.result_queue = queue.Queue()
    state.stop_event = threading.Event()
    state.train_losses = []
    state.val_losses = []
    state.steps = []
    state.tokens_seen = 0
    state.current_epoch = 0
    state.current_step = 0
    state.last_sample = ""
    state.corpus_name = data.get("corpus_filename", "custom")
    state.model_type = model_type
    state.training_active = True

    state.training_thread = start_training_thread(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=int(data.get("num_epochs", 5)),
        eval_freq=int(data.get("eval_freq", 5)),
        result_queue=state.result_queue,
        stop_event=state.stop_event,
        start_context=data.get("start_context", "Once upon a time"),
    )

    return jsonify({"status": "started", "params": count_parameters(model), "device": str(device)})


@app.route("/api/train/stop", methods=["POST"])
def train_stop():
    if not state.training_active or state.stop_event is None:
        return jsonify({"error": "No active training"}), 400
    state.stop_event.set()
    state.training_active = False
    return jsonify({"status": "stop_requested"})


@app.route("/api/train/save", methods=["POST"])
def train_save():
    if state.model is None:
        return jsonify({"error": "No model loaded"}), 400
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = f"checkpoint_epoch{state.current_epoch}_step{state.current_step}.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)
    save_checkpoint(
        state.model, state.optimizer, state.gpt_config,
        state.current_epoch, state.current_step, path,
        model_type=state.model_type,
    )
    return jsonify({"status": "saved", "filename": filename})


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


@app.route("/api/corpus/builtin")
def corpus_builtin():
    available = {n: p for n, p in BUILTIN_CORPORA.items() if os.path.exists(p)}
    return jsonify(list(available.keys()))


@app.route("/api/corpus/stats", methods=["POST"])
def corpus_stats():
    import tiktoken
    data = request.get_json()
    text = data.get("text", "")
    # Allow fetching stats for a named built-in file
    if not text:
        fname = data.get("filename", "")
        path = BUILTIN_CORPORA.get(fname)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
    context_length = int(data.get("context_length", 256))
    train_ratio = float(data.get("train_ratio", 0.9))
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    n = len(tokens)
    split = int(train_ratio * n)
    return jsonify({
        "total_tokens": n,
        "train_tokens": split,
        "val_tokens": n - split,
        "train_batches_approx": max(1, split // context_length),
    })


@app.route("/api/corpus/upload", methods=["POST"])
def corpus_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    text = f.read().decode("utf-8", errors="replace")
    return jsonify({"filename": f.filename, "text": text, "length": len(text)})


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@app.route("/api/checkpoints")
def checkpoints():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
    return jsonify(files)


@app.route("/api/checkpoint/load", methods=["POST"])
def checkpoint_load():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    model, _opt_state, meta = load_checkpoint(path)
    model.eval()

    state.gen_model = model
    state.gen_config = meta["gpt_config"]
    state.gen_device = get_device()
    state.gen_model_type = meta.get("model_type", "gpt2")

    cfg = meta["gpt_config"]
    return jsonify({
        "status": "loaded",
        "epoch": meta["epoch"],
        "step": meta["step"],
        "gpt_config": cfg,
        "model_type": meta.get("model_type", "gpt2"),
        "params": sum(p.numel() for p in model.parameters()),
    })


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@app.route("/api/generate", methods=["POST"])
def generate_text():
    if state.gen_model is None:
        return jsonify({"error": "No model loaded. Load a checkpoint first."}), 400

    import tiktoken
    from LLM import generate, text_to_token_ids, token_ids_to_text

    data = request.get_json()
    prompt = data.get("prompt", "Once upon a time")
    max_new_tokens = int(data.get("max_new_tokens", 100))
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 50))

    from training_engine import get_context_size
    tokenizer = tiktoken.get_encoding("gpt2")
    context_size = get_context_size(state.gen_model, state.gen_config)
    encoded = text_to_token_ids(prompt, tokenizer).to(state.gen_device)

    with torch.no_grad():
        token_ids = generate(
            model=state.gen_model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            tok_k=top_k,
        )

    output = token_ids_to_text(token_ids, tokenizer)
    return jsonify({"text": output})


# ---------------------------------------------------------------------------
# System stats
# ---------------------------------------------------------------------------


@app.route("/api/system/stats")
def system_stats():
    # CPU
    cpu_pct = psutil.cpu_percent(interval=None)

    # RAM
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / (1024 ** 3)
    ram_total_gb = vm.total / (1024 ** 3)
    ram_pct = vm.percent

    # GPU
    gpu = {"available": False, "type": "CPU", "mem_used_mb": 0, "mem_total_mb": 0, "mem_pct": 0}

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        mem_used = torch.cuda.memory_allocated(dev)
        mem_total = props.total_memory
        gpu = {
            "available": True,
            "type": f"CUDA ({props.name})",
            "mem_used_mb": round(mem_used / (1024 ** 2), 1),
            "mem_total_mb": round(mem_total / (1024 ** 2), 1),
            "mem_pct": round(100 * mem_used / mem_total, 1) if mem_total else 0,
        }
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        mem_used = torch.mps.current_allocated_memory()
        gpu = {
            "available": True,
            "type": "MPS (Apple Silicon)",
            "mem_used_mb": round(mem_used / (1024 ** 2), 1),
            "mem_total_mb": None,   # MPS doesn't expose total VRAM
            "mem_pct": None,
        }

    return jsonify({
        "cpu_pct": cpu_pct,
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_pct": ram_pct,
        "gpu": gpu,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
