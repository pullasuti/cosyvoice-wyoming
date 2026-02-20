FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git sox libsox-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

RUN uv venv -p 3.10 && \
    . .venv/bin/activate && \
    uv pip install "setuptools<75" wheel && \
    uv pip install openai-whisper==20231117 --no-build-isolation && \
    uv pip install -r CosyVoice/requirements.txt && \
    uv pip install "huggingface_hub[hf_xet]" fastapi uvicorn python-multipart python-dotenv

RUN . .venv/bin/activate && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
                  local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')"

COPY .env.example .env
COPY server.py ./

ENV PORT=8020
EXPOSE 8020

CMD [".venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8020"]
