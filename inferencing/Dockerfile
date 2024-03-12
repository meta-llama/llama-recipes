FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL description="Contextual Conversation LLM for HelixAI Assistant"
WORKDIR /app
RUN apt update
RUN apt install python-is-python3 -y
COPY pyproject.toml pyproject.toml
RUN apt install vim -y
RUN apt install python3-distutils -y
RUN apt install python3-pip -y
RUN pip install poetry
RUN touch README.md
RUN poetry install
RUN poetry run huggingface-cli login --token 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
RUN mkdir /models
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
RUN cp /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so.11
ENV llama2_model_name=ct2_codellama-8bit-json-mkt-research-24-03-07
RUN poetry run huggingface-cli download HelixAI/$llama2_model_name --local-dir models/merged-codellama-ct2
COPY app.py app.py
COPY hl_mr_prompt.yaml hl_mr_prompt.yaml
CMD poetry run gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout=120 --bind 0.0.0.0:8082
