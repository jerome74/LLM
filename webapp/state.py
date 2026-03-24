"""Server-side training state singleton."""

import queue


class TrainingState:
    model = None
    optimizer = None
    device = None
    gpt_config = None
    result_queue = None
    stop_event = None
    training_thread = None
    training_active = False

    train_losses = []
    val_losses = []
    steps = []
    tokens_seen = 0
    current_epoch = 0
    current_step = 0
    last_sample = ""
    corpus_name = ""

    model_type = "gpt2"  # "gpt2" or "gpt4"

    # generation
    gen_model = None
    gen_config = None
    gen_device = None
    gen_model_type = "gpt2"


state = TrainingState()
