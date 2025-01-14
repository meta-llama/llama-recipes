def get_default_finetune_args():
    return {
        "learning_rate": 0.0003,
        "max_steps": 60,
        "early_stopping": False,
        "load_best_model_at_end": False,
        "peft_args": {"r_value": 32},
    }
