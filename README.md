# CosyVoice Wyoming bridge

Streaming text-to-speech for Home Assistant, using
[CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice) with zero-shot voice
cloning, emotion tags and non-verbal sounds. The model runs in-process and
speaks the [Wyoming protocol](https://github.com/OHF-Voice/wyoming), so HA
connects to it directly.

## Quick start

```bash
./run.sh
```

The first run installs uv if needed, fetches the CosyVoice fork (git
submodule), builds a Python 3.10 venv, and downloads
`Fun-CosyVoice3-0.5B-2512` (~8 GB). With TensorRT on, the first start also
builds the flow-decoder engine, which takes several minutes and is cached
next to the model. Later runs start straight away.

The bridge listens on `tcp://0.0.0.0:10214` and accepts connections while
the model is still loading.

- `SETUP_ONLY=1 ./run.sh` sets everything up without starting the bridge.
- `REINSTALL=1 ./run.sh` reinstalls the Python dependencies.
- Extra arguments go to `bridge_direct.py` (`./run.sh --help` lists them).

Requires Linux with an NVIDIA GPU.

## Configuration

Copy `.env.example` to `.env` and edit. Everything is optional:

| Variable             | Default                                          | Description                                  |
|----------------------|--------------------------------------------------|----------------------------------------------|
| `PROMPT_WAV`         | `voices/nicoleheart.wav`                         | Reference voice (10-20 s clean mono WAV)     |
| `VOICE`              | `nicoleheart`                                    | Voice name shown in Home Assistant           |
| `URI`                | `tcp://0.0.0.0:10214`                            | Wyoming listen address                       |
| `MODEL_ID`           | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`           | HuggingFace model downloaded on first run    |
| `MODEL_DIR`          | `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` | Model location                              |
| `USE_RL`             | `1`                                              | Use the RL-tuned LLM weights                 |
| `FP16`               | `1`                                              | Half precision                               |
| `TRT`                | `1`                                              | TensorRT flow decoder                        |
| `PREBUFFER_SECONDS`  | `1`                                              | Audio buffered before playback starts        |
| `MIN_MERGED_SECONDS` | `2.5`                                            | Speech merged per model call after the first sentence |

`voices/` is git-ignored, so the reference WAV stays local.

## Tags

**Mood tags** set the delivery and stay in effect until the next mood tag.
They can be stacked:

`[warm]` `[happy]` `[excited]` `[confident]` `[serious]` `[sad]` `[angry]`
`[whisper]` `[loud]` `[slow]` `[fast]` `[robot]`

**Sound tags** are spoken by the model where they appear:

`[laughter]` `[breath]` `[sigh]` `[cough]` `[noise]` `[lipsmack]` `[mn]`

Curly braces and common variants also work, e.g. `(laughs)` becomes
`[laughter]`. Unknown tags are dropped and logged.

```
[warm] Good morning! [breath] The coffee is brewing already.
```

## Home Assistant

1. Add the **Wyoming Protocol** integration with this host and port 10214.
2. Pick the voice as the text-to-speech engine in your voice assistant.
3. Paste `ha_prompt.txt` into the conversation agent's instructions so the
   LLM uses the tags.

## Running as a service

`/etc/systemd/system/cosyvoice-wyoming.service` can run this bridge:

```ini
[Service]
WorkingDirectory=/root/cosyvoice-tts-api
ExecStart=/root/cosyvoice-tts-api/run.sh
Restart=always
RestartSec=5
```

## Files

- `bridge_direct.py`: the Wyoming server (sentence splitting, tag parsing,
  streaming playback).
- `run.sh`: setup and launcher.
- `test_bridge.py`: test client for a running bridge. It covers one-shot and
  streaming requests, the duplicate requests HA sends, and blank text
  (`.venv/bin/python test_bridge.py [--host HOST] [--port PORT]`).
- `ha_prompt.txt`: system prompt for the HA conversation agent.
- `CosyVoice/`: submodule pinned to a fork with segment-timing
  instrumentation and tuned CosyVoice 3 streaming parameters.

## Credits

Text-to-speech by [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) from
FunAudioLLM (Apache-2.0).
