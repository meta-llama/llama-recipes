.PHONY: install install-dev test lint format clean

install:
	pip install -e .

install-dev:
	pip install -e '.[tests]'

test:
	python -m pytest src/tests/ -v

lint:
	python -m py_compile src/llama_cookbook/finetuning.py
	@echo "Lint passed"

format:
	black src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
