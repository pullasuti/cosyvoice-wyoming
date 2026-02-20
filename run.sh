#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "CosyVoice" ]; then
    echo ">>> Cloning CosyVoice..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
fi

if [ ! -d ".venv" ]; then
    echo ">>> Creating venv..."
    uv venv -p 3.10
fi

source .venv/bin/activate

echo ">>> Installing deps..."
uv pip install "setuptools<75" wheel
uv pip install openai-whisper==20231117 --no-build-isolation
uv pip install -r CosyVoice/requirements.txt --index-strategy unsafe-best-match
uv pip install "huggingface_hub[hf_xet]" fastapi uvicorn python-multipart python-dotenv

MODEL_DIR="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
if [ ! -f "$MODEL_DIR/llm.pt" ]; then
    echo ">>> Downloading model (~8GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='$MODEL_DIR')
"
fi

source .env 2>/dev/null || true
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo ">>> Starting server on port ${PORT:-8020}..."
uvicorn server:app --host 0.0.0.0 --port "${PORT:-8020}"
