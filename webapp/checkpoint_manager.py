"""Checkpoint save/load utilities."""

import os
import sys

import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_model import GPTModel
from gpt_model4 import GPT4Model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, gpt_config, epoch, step, path, model_type="gpt2"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gpt_config": gpt_config,
            "model_type": model_type,
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def load_checkpoint(path, device=None):
    if device is None:
        device = get_device()
    data = torch.load(path, map_location=device)
    gpt_config = data["gpt_config"]
    model_type = data.get("model_type", "gpt2")  # backwards-compatible default
    if model_type == "gpt4":
        model = GPT4Model(gpt_config)
    else:
        model = GPTModel(gpt_config)
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    return model, data["optimizer_state_dict"], {
        "epoch": data["epoch"],
        "step": data["step"],
        "gpt_config": gpt_config,
        "model_type": model_type,
    }
