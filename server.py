import sys, os, io, tempfile, glob
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

COSYVOICE_DIR = os.path.join(BASE_DIR, os.environ.get("COSYVOICE_DIR", "CosyVoice"))
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party/Matcha-TTS"))

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from cosyvoice.cli.cosyvoice import AutoModel


class ResponseFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"


class Mode(str, Enum):
    zero_shot = "zero_shot"
    cross_lingual = "cross_lingual"


class Emotion(str, Enum):
    none = ""
    happy = "happy"
    sad = "sad"
    angry = "angry"
    whisper = "whisper"
    loud = "loud"
    slow = "slow"
    fast = "fast"
    robot = "robot"


INSTRUCT_PRESETS = {
    "happy": "You are a helpful assistant. Speak with a happy and excited tone.<|endofprompt|>",
    "sad": "You are a helpful assistant. Speak with a sad and melancholic tone.<|endofprompt|>",
    "angry": "You are a helpful assistant. Speak with an angry tone.<|endofprompt|>",
    "whisper": "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
    "loud": "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
    "slow": "You are a helpful assistant. Speak as slowly as possible.<|endofprompt|>",
    "fast": "You are a helpful assistant. Speak as fast as possible.<|endofprompt|>",
    "robot": "You are a helpful assistant. Speak like a robot.<|endofprompt|>",
}

MIME = {"wav": "audio/wav", "mp3": "audio/mpeg"}

voices: dict[str, str] = {}

app = FastAPI(
    title="CosyVoice API",
    description="OpenAI-compatible TTS API with zero-shot voice cloning. "
                "Supports Russian & English. Uses Fun-CosyVoice3-0.5B-RL.",
    version="1.0.0",
)
model: AutoModel = None


def _load_voices():
    voices_dir = os.path.join(BASE_DIR, "voices")
    os.makedirs(voices_dir, exist_ok=True)

    default_voice = os.path.join(BASE_DIR, os.environ.get("DEFAULT_VOICE", "CosyVoice/asset/zero_shot_prompt.wav"))
    voices["default"] = default_voice

    for f in glob.glob(os.path.join(voices_dir, "*.wav")):
        name = os.path.splitext(os.path.basename(f))[0]
        voices[name] = f

    active = os.environ.get("VOICE", "default")
    if active not in voices:
        print(f"WARNING: VOICE={active} not found, falling back to 'default'")
        active = "default"
    voices["_active"] = voices[active]

    print(f"Voices loaded: {list(k for k in voices if k != '_active')}")
    print(f"Active voice: {active} -> {voices['_active']}")


def _get_voice(name: str = "") -> str:
    if not name or name == "default":
        return voices["_active"]
    if name in voices:
        return voices[name]
    raise HTTPException(400, f"Unknown voice '{name}'. Available: {list(k for k in voices if k != '_active')}")


@app.on_event("startup")
def startup():
    global model
    model_dir = os.path.join(BASE_DIR, os.environ.get("MODEL_DIR", "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))

    if os.environ.get("USE_RL", "true").lower() == "true":
        rl = os.path.join(model_dir, "llm.rl.pt")
        llm = os.path.join(model_dir, "llm.pt")
        bak = os.path.join(model_dir, "llm.base.pt")
        if os.path.exists(rl) and not os.path.exists(bak):
            os.rename(llm, bak)
            os.symlink("llm.rl.pt", llm)

    model = AutoModel(model_dir=model_dir)
    _load_voices()


class VoicesResponse(BaseModel):
    voices: list[str]
    active: str


@app.get("/voices", response_model=VoicesResponse, summary="List available voices")
def list_voices():
    names = [k for k in voices if k != "_active"]
    active = next((k for k, v in voices.items() if k != "_active" and v == voices["_active"]), "default")
    return VoicesResponse(voices=names, active=active)


# ── OpenAI-compatible endpoint ──────────────────────────────


class SpeechRequest(BaseModel):
    model: str = Field(
        default="cosyvoice3-0.5b",
        description="Model ID (ignored, always uses CosyVoice3-0.5B-RL). "
                    "Accepts any value for OpenAI client compatibility.",
    )
    input: str = Field(
        description="Text to synthesize (Russian / English). "
                    "Inline tags: [breath], [laughter], [sigh], [cough], [noise], [lipsmack], [mn]",
        json_schema_extra={"example": "Привет! [breath] Как дела?"},
    )
    voice: str = Field(
        default="default",
        description="Voice name. 'default' uses the active voice from VOICE env var. "
                    "Or specify a name matching a file in voices/ folder (without .wav).",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.wav,
        description="Output audio format: wav or mp3",
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Speech speed (0.25–4.0). Currently passed as instruction hint.",
    )
    instructions: str = Field(
        default="",
        description="Style instruction (OpenAI gpt-4o-mini-tts compatible). "
                    "Example: 'Speak with a cheerful tone' or 'Whisper softly'. "
                    "Can also use CosyVoice format: 'You are a helpful assistant. <instruction>.<|endofprompt|>'",
    )


def _build_instruct(instructions: str, speed: float) -> str:
    if not instructions and speed == 1.0:
        return ""
    parts = []
    if instructions:
        clean = instructions.strip().rstrip("<|endofprompt|>").strip()
        if clean.startswith("You are a helpful assistant."):
            parts.append(clean)
        else:
            parts.append(f"You are a helpful assistant. {clean}")
    if speed != 1.0:
        if speed < 0.7:
            parts.append("Speak as slowly as possible")
        elif speed < 1.0:
            parts.append("Speak a bit slower than normal")
        elif speed > 1.5:
            parts.append("Speak as fast as possible")
        elif speed > 1.0:
            parts.append("Speak a bit faster than normal")
    return ". ".join(parts) + ".<|endofprompt|>" if parts else ""


def _generate(text: str, prompt_wav: str, instruct_text: str = ""):
    if instruct_text:
        return model.inference_instruct2(text, instruct_text, prompt_wav, stream=False)
    return model.inference_cross_lingual(
        f"You are a helpful assistant.<|endofprompt|>{text}",
        prompt_wav, stream=False,
    )


def _encode(audio: torch.Tensor, fmt: ResponseFormat) -> tuple[bytes, str]:
    buf = io.BytesIO()
    torchaudio.save(buf, audio, model.sample_rate, format=fmt.value)
    return buf.getvalue(), MIME[fmt.value]


@app.post(
    "/v1/audio/speech",
    summary="OpenAI-compatible TTS",
    description="Drop-in replacement for OpenAI `POST /v1/audio/speech`. "
                "Returns audio in the requested format.",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}, "audio/mpeg": {}}}},
)
async def openai_speech(req: SpeechRequest):
    prompt_wav = _get_voice(req.voice)
    instruct_text = _build_instruct(req.instructions, req.speed)
    gen = _generate(req.input, prompt_wav, instruct_text)
    audio = torch.cat([c["tts_speech"] for c in gen], dim=1)
    data, mime = _encode(audio, req.response_format)
    ext = req.response_format.value
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f"attachment; filename=speech.{ext}"})


# ── Extended endpoint (voice cloning + emotions) ────────────


@app.post(
    "/tts",
    summary="Extended TTS with voice cloning",
    description="Upload a voice sample (WAV, 3-15s) for zero-shot cloning, "
                "or use a named voice from the voices/ folder. "
                "Supports emotion presets and inline tags.",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def tts(
    text: str = Form(
        description="Text to synthesize (Russian / English). "
                    "Inline tags: [breath], [laughter], [sigh], [cough], [noise], [lipsmack], [mn]",
    ),
    voice_name: str = Form(
        default="default",
        description="Named voice from voices/ folder (without .wav). "
                    "'default' uses the active voice set by VOICE env var.",
    ),
    mode: Mode = Form(
        default=Mode.cross_lingual,
        description="cross_lingual (default) — keeps voice timbre, speaks in text's language. "
                    "zero_shot — matches voice language exactly (use when voice and text are same language).",
    ),
    emotion: Emotion = Form(
        default=Emotion.none,
        description="Preset: happy, sad, angry, whisper, loud, slow, fast, robot",
    ),
    instruct: str = Form(
        default="",
        description="Custom style instruction. "
                    "Format: 'You are a helpful assistant. <instruction>.<|endofprompt|>'",
    ),
    response_format: ResponseFormat = Form(
        default=ResponseFormat.wav,
        description="Output format: wav or mp3",
    ),
    voice: Optional[UploadFile] = File(
        default=None,
        description="Voice sample WAV (3-15s clear speech) for cloning. "
                    "Overrides voice_name if provided.",
    ),
):
    prompt_wav = _get_voice(voice_name)
    tmp = None

    if voice:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(await voice.read())
        tmp.close()
        prompt_wav = tmp.name

    try:
        instruct_text = instruct or INSTRUCT_PRESETS.get(emotion.value, "")

        if instruct_text:
            gen = model.inference_instruct2(text, instruct_text, prompt_wav, stream=False)
        elif mode == Mode.cross_lingual:
            gen = model.inference_cross_lingual(
                f"You are a helpful assistant.<|endofprompt|>{text}",
                prompt_wav, stream=False,
            )
        else:
            gen = model.inference_zero_shot(
                text, "You are a helpful assistant.<|endofprompt|>", prompt_wav, stream=False,
            )

        audio = torch.cat([c["tts_speech"] for c in gen], dim=1)
        data, mime = _encode(audio, response_format)
        ext = response_format.value
        return Response(content=data, media_type=mime,
                        headers={"Content-Disposition": f"attachment; filename=speech.{ext}"})
    finally:
        if tmp:
            os.unlink(tmp.name)
