"""Background training thread with queue-based progress emission."""

import os
import sys
import threading

import tiktoken
import torch
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_model import GPTModel
from LLM import (
    GPTDatasetV1,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model_and_optimizer(gpt_config, learning_rate, weight_decay):
    device = get_device()
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return model, optimizer, device


def build_dataloaders(text, batch_size, context_length, train_ratio):
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    tokenizer = tiktoken.get_encoding("gpt2")
    stride = context_length

    train_dataset = GPTDatasetV1(train_text, tokenizer, context_length, stride)
    val_dataset = GPTDatasetV1(val_text, tokenizer, context_length, stride)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
    )
    return train_loader, val_loader


def _run_training(
    model, optimizer, train_loader, val_loader, device,
    num_epochs, eval_freq, result_queue, stop_event, start_context,
):
    """
    Wraps LLM.train_model_simple for use in a background thread.

    train_model_simple owns the inner training loop and returns all losses
    at the end of each epoch. Stop is only honoured between epochs.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    steps_per_epoch = len(train_loader)

    try:
        for epoch in range(num_epochs):
            if stop_event.is_set():
                result_queue.put({"type": "stopped", "epoch": epoch, "step": epoch * steps_per_epoch})
                return

            # Run one epoch via train_model_simple (eval_iter=1 keeps evaluation fast)
            train_losses, val_losses, track_tokens_seen = train_model_simple(
                model, train_loader, val_loader, optimizer, device,
                num_epochs=1,
                eval_freq=eval_freq,
                eval_iter=1,
                start_context=start_context,
                tokenizer=tokenizer,
            )

            # Flush all loss events collected during this epoch
            step_offset = epoch * steps_per_epoch
            for i, (tl, vl, tokens) in enumerate(
                zip(train_losses, val_losses, track_tokens_seen)
            ):
                result_queue.put({
                    "type": "loss",
                    "epoch": epoch + 1,
                    "step": step_offset + (i + 1) * eval_freq,
                    "tokens_seen": tokens,
                    "train_loss": float(tl),
                    "val_loss": float(vl),
                })

            # Generate a sample with temperature (richer than train_model_simple's greedy output)
            model.eval()
            context_size = model.pos_emb.weight.shape[0]
            encoded = text_to_token_ids(start_context, tokenizer).to(device)
            with torch.no_grad():
                token_ids = generate(
                    model=model,
                    idx=encoded,
                    max_new_tokens=50,
                    context_size=context_size,
                    temperature=1.0,
                    tok_k=25,
                )
            result_queue.put({
                "type": "sample",
                "epoch": epoch + 1,
                "text": token_ids_to_text(token_ids, tokenizer),
            })
            model.train()

        total_steps = num_epochs * steps_per_epoch
        result_queue.put({"type": "done", "step": total_steps, "tokens_seen": total_steps * train_loader.batch_size})

    except Exception as exc:
        result_queue.put({"type": "error", "message": str(exc)})


def start_training_thread(
    model, optimizer, train_loader, val_loader, device,
    num_epochs, eval_freq, result_queue, stop_event, start_context="Once upon a time",
):
    thread = threading.Thread(
        target=_run_training,
        args=(model, optimizer, train_loader, val_loader, device,
              num_epochs, eval_freq, result_queue, stop_event, start_context),
        daemon=True,
    )
    thread.start()
    return thread
