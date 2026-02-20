# CosyVoice API

OpenAI-compatible text-to-speech API server with zero-shot voice cloning, powered by CosyVoice 3. Works well with Russian and English text, including mixed-language input.

## Quick Start

### Local (recommended for Apple Silicon / Metal)

On macOS with Apple Silicon, run locally to take advantage of Metal (MPS) GPU acceleration.
Docker cannot pass through the Metal GPU, so local execution is significantly faster.

```bash
./run.sh
```

On first run the script will:
- Install [uv](https://docs.astral.sh/uv/) if missing
- Clone the CosyVoice repository
- Install Python dependencies
- Download the model from HuggingFace (~8 GB)

Server starts on `http://localhost:8020`

### Docker

```bash
docker build -t cosyvoice-api .
docker run -p 8020:8020 --env-file .env cosyvoice-api
```

> **Note:** The Docker image is large (~15 GB) because it bundles the model weights.
> Docker runs on CPU only. For NVIDIA GPU support, use `--gpus all` with an NVIDIA runtime installed.

## Usage

### OpenAI-compatible endpoint

Drop-in replacement for `POST /v1/audio/speech`:

```bash
curl -s http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "default"}' \
  -o speech.wav
```

Works with the official OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="unused")
response = client.audio.speech.create(
    model="cosyvoice3-0.5b",
    voice="default",
    input="Hello, world!",
    response_format="wav",
)
response.stream_to_file("speech.wav")
```

### Extended endpoint with voice cloning

Upload a voice sample (WAV, 3-15s) for zero-shot cloning:

```bash
curl -s -X POST http://localhost:8020/tts \
  -F "text=Hello from a cloned voice!" \
  -F "voice=@sample.wav" \
  -o cloned.wav
```

### Emotion presets

```bash
curl -s http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "This is exciting!", "instructions": "Speak with a happy and excited tone"}' \
  -o happy.wav
```

Available presets via `/tts` endpoint: `happy`, `sad`, `angry`, `whisper`, `loud`, `slow`, `fast`, `robot`.

### List voices

```bash
curl http://localhost:8020/voices
```

### Record a voice sample (macOS)

```bash
bash record.sh my_voice
```

Records using `sox` (installed via Homebrew if missing) and saves to `voices/my_voice.wav`. Set `VOICE=my_voice` in `.env` to use as default. On Linux, install `sox` manually via your package manager.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable        | Default                                          | Description                        |
|-----------------|--------------------------------------------------|------------------------------------|
| `PORT`          | 8020                                             | Server port                        |
| `USE_RL`        | true                                             | Use RL-tuned model variant         |
| `COSYVOICE_DIR` | CosyVoice                                        | Path to CosyVoice repo            |
| `MODEL_DIR`     | CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B | Path to model weights             |
| `DEFAULT_VOICE` | CosyVoice/asset/zero_shot_prompt.wav             | Default voice sample               |
| `VOICE`         | default                                          | Active voice name                  |

## Inline Tags

CosyVoice supports inline tags in the text for expressive speech:

`[breath]`, `[laughter]`, `[sigh]`, `[cough]`, `[noise]`, `[lipsmack]`, `[mn]`

## Performance

- **Languages**: Russian + English (cross-lingual)
- **Voice cloning**: zero-shot from a 3-15s audio sample
- **Formats**: WAV, MP3
- **GPU**: Metal (MPS) on Apple Silicon, CUDA on NVIDIA, CPU fallback
- **Model**: Fun-CosyVoice3-0.5B-RL (~8 GB)

## Credits

Text-to-speech powered by [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by FunAudioLLM (Apache-2.0 license).
