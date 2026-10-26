#!/bin/bash
# CosyVoice Wyoming bridge launcher.
#
# First run: installs uv if missing, fetches the CosyVoice fork (submodule),
# builds the Python 3.10 venv and downloads the model (~8 GB). Later runs skip
# straight to starting the bridge.
#
# Configuration: optional .env next to this script (see .env.example).
# Extra arguments are passed through to bridge_direct.py, e.g.
#   ./run.sh --speed 1.1
#
#   SETUP_ONLY=1 ./run.sh   set everything up, don't start the bridge
#   REINSTALL=1 ./run.sh    reinstall Python dependencies
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL_ID="${MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
MODEL_DIR="${MODEL_DIR:-CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B}"
PROMPT_WAV="${PROMPT_WAV:-voices/nicoleheart.wav}"
VOICE="${VOICE:-nicoleheart}"
URI="${URI:-tcp://0.0.0.0:10214}"
USE_RL="${USE_RL:-1}"
FP16="${FP16:-1}"
TRT="${TRT:-1}"
PREBUFFER_SECONDS="${PREBUFFER_SECONDS:-1}"
MIN_MERGED_SECONDS="${MIN_MERGED_SECONDS:-2.5}"
DEFAULT_MOOD="${DEFAULT_MOOD-warm}"

# Only needed for setup; systemd's PATH usually lacks ~/.local/bin.
ensure_uv() {
    export PATH="${HOME:-/root}/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo ">>> Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

# CosyVoice is a submodule pinned to an instrumented fork; Matcha-TTS is a
# submodule of that.
if [ ! -f CosyVoice/third_party/Matcha-TTS/setup.py ]; then
    echo ">>> Fetching CosyVoice..."
    if [ -d .git ]; then
        git submodule update --init --recursive
    else
        git clone --recursive https://github.com/pullasuti/CosyVoice.git
    fi
fi

if [ ! -d .venv ]; then
    ensure_uv
    echo ">>> Creating venv..."
    uv venv -p 3.10
fi

if [ ! -f .venv/.deps-installed ] || [ "${REINSTALL:-0}" = 1 ]; then
    ensure_uv
    echo ">>> Installing dependencies..."
    uv pip install -p .venv "setuptools<75" wheel
    uv pip install -p .venv openai-whisper==20231117 --no-build-isolation
    uv pip install -p .venv -r CosyVoice/requirements.txt --index-strategy unsafe-best-match
    uv pip install -p .venv wyoming==1.10.2 "huggingface_hub[hf_xet]"
    touch .venv/.deps-installed
fi

# llm.pt may be a symlink to llm.rl.pt left by an older RL swap; either counts.
if [ ! -e "$MODEL_DIR/llm.pt" ] && [ ! -f "$MODEL_DIR/llm.rl.pt" ]; then
    echo ">>> Downloading $MODEL_ID (~8 GB)..."
    .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
"
fi

if [ ! -f "$PROMPT_WAV" ]; then
    echo "ERROR: reference voice $PROMPT_WAV not found." >&2
    echo "Put a clean 10-20 s mono WAV of the voice to clone there, or set" >&2
    echo "PROMPT_WAV in .env." >&2
    exit 1
fi

if [ "${SETUP_ONLY:-0}" = 1 ]; then
    echo ">>> Setup complete."
    exit 0
fi

ARGS=(
    --cosyvoice-repo CosyVoice
    --model-dir "$MODEL_DIR"
    --prompt-wav "$PROMPT_WAV"
    --voice "$VOICE"
    --uri "$URI"
    --prebuffer-seconds "$PREBUFFER_SECONDS"
    --min-merged-seconds "$MIN_MERGED_SECONDS"
    --default-mood "$DEFAULT_MOOD"
)
[ "$USE_RL" = 1 ] && ARGS+=(--use-rl)
[ "$FP16" = 1 ] && ARGS+=(--fp16)
[ "$TRT" = 1 ] && ARGS+=(--trt)

echo ">>> Starting Wyoming bridge on $URI..."
exec .venv/bin/python bridge_direct.py "${ARGS[@]}" "$@"
