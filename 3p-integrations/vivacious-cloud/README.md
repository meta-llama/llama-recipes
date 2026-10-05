# Meta Llama 3.2 & 3.1 with Vivacious Cloud

This integration guide demonstrates how to calculate preflight CUDA memory requirements and deploy Meta Llama 3.2 and 3.1 fine-tuning and inference workflows across 12+ cloud GPU providers using [Vivacious Cloud](https://vivaciouscloud.com) and the open-source [`vivacious-cli`](https://github.com/Viavcious-cloud/vivacious-cli).

## Overview

Fine-tuning and deploying large generative models often results in Out-Of-Memory (CUDA OOM) failures due to underestimating optimizer state multipliers (e.g. 16 bytes/param for AdamW), dynamic KV-cache growth, or attention sequence padding.

Vivacious Cloud provides:
1. **Preflight VRAM Validation**: Calculate exact weights, activation memory, optimizer state overhead, and KV-cache headroom before launching training runs.
2. **Autonomous Multi-Cloud GPU Brokerage**: Provision spot or on-demand instances across AWS, Lambda Labs, RunPod, Vast.ai, and GCP without vendor lock-in.
3. **Failover Checkpoint Resilience**: Automatic warm migration if spot instances are preempted.

---

## Hardware & Preflight Matrix

| Llama Model | Workflow | Method | Context Window | Recommended GPU | Est. Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** | SFT / LoRA | LoRA (r=16) | 4,096 | 1x RTX 4090 / A10G | ~9.2 GB |
| **Llama 3.2 3B** | SFT / LoRA | LoRA (r=16) | 4,096 | 1x RTX 4090 / L4 | ~14.8 GB |
| **Llama 3.1 8B** | Instruct Tuning | QLoRA (4-bit) | 8,192 | 1x RTX 4090 / A100 40GB | ~19.5 GB |
| **Llama 3.1 8B** | Instruct Tuning | LoRA (16-bit) | 4,096 | 1x A100 80GB / H100 | ~34.2 GB |
| **Llama 3.1 70B** | vLLM Serving | AWQ / FP8 | 8,192 | 2x-4x A100 / H100 | ~78.0 GB |

---

## Quickstart

### 1. Preflight Memory Verification

Before spinning up compute, verify that your training hyperparameters fit on the target GPU architecture:

```bash
pip install vivacious-cli
```

Estimate exact memory allocations for Llama 3.2 3B:

```bash
vivacious estimate \
  --model meta-llama/Llama-3.2-3B \
  --seq-len 4096 \
  --batch-size 4 \
  --method lora \
  --rank 16
```

Output reports breakdown of:
- **Base model weights**: ~6.0 GB (bfloat16)
- **Trainable LoRA adapter weights**: ~0.12 GB
- **AdamW optimizer states**: ~0.48 GB
- **Activation checkpointed headroom**: ~4.1 GB
- **CUDA runtime overhead buffer**: ~1.5 GB
- **Verdict**: Fits comfortably on 24 GB GPU (Headroom: ~11.8 GB).

### 2. Launch Training via llama-recipes

Run your standard `llama-recipes` fine-tuning job on your provisioned instance:

```bash
python -m llama_recipes.finetuning \
  --use_peft \
  --peft_method lora \
  --quantization 4bit \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./lora-llama3-2-3b \
  --batch_size_training 4 \
  --context_length 4096
```

### 3. Serving with vLLM

Once trained, serve your LoRA adapter on vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --enable-lora \
  --lora-modules llama3-lora=./lora-llama3-2-3b \
  --gpu-memory-utilization 0.90
```

---

## Resources

- [Vivacious Cloud Official Platform](https://vivaciouscloud.com)
- [vivacious-cli GitHub Repository](https://github.com/Viavcious-cloud/vivacious-cli)
- [Free Browser Memory Calculator](https://vivaciouscloud.com/calculator)
