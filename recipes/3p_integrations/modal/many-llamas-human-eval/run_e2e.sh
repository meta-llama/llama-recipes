#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

command -v modal >/dev/null 2>&1 || { echo >&2 "modal command not found. Install modal first! Aborting."; exit 1; }

echo 'downloading LLaMA 3.2 3B Instruct model'
echo 'make sure to create a Secret called huggingface on Modal and accept the LLaMA 3.2 license'
modal run download.py

echo 'deploying vLLM inference server'
modal deploy inference.py

echo 'running HumanEval generation'
modal run generate.py --data-dir test --no-dry-run --n 1000 --subsample 100

echo 'running HumanEval evaluation'
modal run eval.py

echo 'generating graphs for pass@k and fail@k'
modal run plot.py