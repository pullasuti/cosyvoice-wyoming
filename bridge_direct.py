#!/usr/bin/env python3
"""CosyVoice Wyoming TTS bridge — DIRECT variant (no HTTP, no httpx).

Differences from the httpx bridge:

  * CosyVoice is imported and loaded in-process; there is no FastAPI /
    uvicorn middleman and no JSON serialisation per sentence.
  * Sample rate comes from ``model.sample_rate`` — one source of truth.
  * The zero-shot speaker embedding is computed ONCE at startup and
    cached (``add_zero_shot_spk`` / ``zero_shot_spk_id``) instead of
    being rebuilt on every request.
  * Model inference is blocking torch code, so it runs in a worker
    thread via ``asyncio.to_thread``. Calling it directly from the
    event loop would freeze the whole Wyoming server (including the
    reading of incoming SynthesizeChunk events) for the duration of
    every sentence.
  * A process-wide lock serialises inference, so two simultaneous HA
    requests can't collide on the GPU.

REQUIREMENTS / THINGS TO VERIFY AGAINST YOUR CHECKOUT
-----------------------------------------------------
This script must run inside the CosyVoice venv (it needs torch, numpy
and the CosyVoice package on sys.path). The exact inference signatures
have shifted between CosyVoice revisions, so check yours before
trusting the defaults:

    sed -n '1,140p' CosyVoice/cosyvoice/cli/cosyvoice.py

Specifically confirm:
  * the class name (``CosyVoice``) and constructor kwargs
  * ``inference_instruct2(tts_text, instruct_text, prompt_speech, ...)``
  * ``inference_zero_shot(tts_text, prompt_text, prompt_speech, ...)``
  * whether ``add_zero_shot_spk`` / ``zero_shot_spk_id`` exist
  * whether the generator yields dicts keyed ``"tts_speech"``

Anything that differs is a small edit inside ``SynthEngine._call`` —
that method is deliberately the only place that touches the model API.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag handling (unchanged from the httpx bridge)
# ---------------------------------------------------------------------------

TAG_MAP = {
    "warm": "Please speak in a warm and friendly voice.",
    "confident": "Please speak with a confident and steady voice.",
    "happy": "Speak with a happy and excited tone.",
    "sad": "Speak with a sad and melancholy tone.",
    "excited": "Speak with an excited and energetic tone.",
    "whisper": "Please say a sentence in a very soft voice.",
    "serious": "Please speak in a serious and calm manner.",
}

INLINE_TAGS = {"laughter", "breath", "sigh", "cough", "noise", "lipsmack", "mn"}

TAG_RE = re.compile(r"\s*[\[\{]([^\]\}]+)[\]\}]")
PAREN_RE = re.compile(r"\(\s*([a-zA-Z]+)\s*\)")

ALIASES = {
    "confidence": "confident",
    "excitement": "excited",
    "laughing": "laughter",
    "laughs": "laughter",
    "breaths": "breath",
    "sighs": "sigh",
    "coughs": "cough",
}

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}


def normalize(text: str) -> str:
    def fix(m: re.Match) -> str:
        key = m.group(1).lower()
        return f"[{key}]" if key in TAG_MAP or key in INLINE_TAGS else m.group(0)
    return PAREN_RE.sub(fix, text)


def parse_tags(text: str) -> tuple[str, Optional[str]]:
    text = normalize(text)
    tags: list[str] = []
    out: list[str] = []
    pos = 0

    for m in TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if m.start() > pos:
            out.append(text[pos:m.start()])

        key = ALIASES.get(tag.lower(), tag.lower())
        if key in TAG_MAP:
            tags.append(TAG_MAP[key])
        elif key in INLINE_TAGS:
            out.append(f"[{key}]")
        else:
            _LOGGER.warning("dropping unknown tag %r", tag)

        pos = m.end()

    if pos < len(text):
        out.append(text[pos:])

    clean = " ".join("".join(out).split())
    return clean, (" ".join(tags) if tags else None)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def to_pcm16(tensor: Any) -> bytes:
    """Convert a CosyVoice output tensor to little-endian signed 16-bit PCM.

    CosyVoice yields float waveforms in roughly [-1, 1]; the Wyoming
    client expects raw PCM16, so clip before scaling to avoid wraparound
    on the occasional overshoot.
    """
    arr = tensor.detach().to("cpu", torch.float32).numpy().reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# Model wrapper — the ONLY place that talks to CosyVoice
# ---------------------------------------------------------------------------

class SynthEngine:
    """Loads CosyVoice once and exposes a blocking synthesize() call."""

    def __init__(
        self,
        model_dir: str,
        prompt_wav: Optional[str],
        prompt_text: str,
        spk_id: Optional[str],
        mode: str,
        default_instruct: Optional[str],
        speed: float,
        fp16: bool,
    ) -> None:
        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.spk_id = spk_id
        self.mode = mode
        self.default_instruct = default_instruct
        self.speed = speed
        self.fp16 = fp16

        self.model: Any = None
        self.prompt_speech: Any = None
        self.sample_rate: int = 24000
        self._spk_registered = False
        # Serialises GPU work across concurrent utterances.
        self._lock = threading.Lock()

    # -- startup ------------------------------------------------------------

    def load(self) -> None:
        from cosyvoice.cli.cosyvoice import CosyVoice  # noqa: PLC0415
        from cosyvoice.utils.file_utils import load_wav  # noqa: PLC0415

        _LOGGER.info("loading model from %s", self.model_dir)
        t0 = time.monotonic()
        try:
            self.model = CosyVoice(
                self.model_dir, load_jit=False, load_trt=False, fp16=self.fp16
            )
        except TypeError:
            # Older/newer revisions may not accept the same kwargs.
            _LOGGER.warning("constructor kwargs rejected; falling back to positional")
            self.model = CosyVoice(self.model_dir)

        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        _LOGGER.info(
            "model ready in %.1fs (sample_rate=%d)", time.monotonic() - t0, self.sample_rate
        )

        if not self.prompt_wav:
            _LOGGER.warning("no --prompt-wav given; instruct/zero-shot modes will fail")
            return

        # Prompt features are extracted ONCE here rather than per request.
        self.prompt_speech = load_wav(self.prompt_wav, 16000)

        if self.spk_id:
            try:
                self.model.add_zero_shot_spk(
                    self.prompt_text, self.prompt_speech, self.spk_id
                )
                save = getattr(self.model, "save_spkinfo", None)
                if callable(save):
                    save()
                self._spk_registered = True
                _LOGGER.info("registered cached speaker %r", self.spk_id)
            except Exception:
                _LOGGER.exception(
                    "could not register speaker %r; falling back to per-call prompt",
                    self.spk_id,
                )

    # -- inference ----------------------------------------------------------

    def _call(self, text: str, instructions: Optional[str]) -> Iterator[Any]:
        """Dispatch to the right CosyVoice entry point.

        This is the single adapter between the bridge and the model API;
        adjust here if your checkout's signatures differ.
        """
        kwargs: dict[str, Any] = {"stream": True}
        if self.speed != 1.0:
            kwargs["speed"] = self.speed

        if self.mode == "instruct":
            instruct = instructions or self.default_instruct
            if instruct:
                return self.model.inference_instruct2(
                    text, instruct, self.prompt_speech, **kwargs
                )

        if self._spk_registered:
            # Cached speaker: no prompt text / speech needed per call.
            return self.model.inference_zero_shot(
                text, "", "", zero_shot_spk_id=self.spk_id, **kwargs
            )

        return self.model.inference_zero_shot(
            text, self.prompt_text, self.prompt_speech, **kwargs
        )

    def synthesize(self, text: str, instructions: Optional[str]) -> tuple[bytes, int]:
        """Blocking. Returns (pcm16 bytes, sample_rate)."""
        with self._lock:
            buf = bytearray()
            with torch.inference_mode():
                for out in self._call(text, instructions):
                    buf.extend(to_pcm16(out["tts_speech"]))
        return bytes(buf), self.sample_rate


# ---------------------------------------------------------------------------
# Wyoming handler
# ---------------------------------------------------------------------------

class BridgeHandler(AsyncEventHandler):
    def __init__(
        self,
        *args: Any,
        engine: SynthEngine,
        info: Info,
        voice: str,
        sample_rate: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.voice = voice
        self.sample_rate = sample_rate

        self._stash = ""            # legacy single-shot Synthesize text
        self._buf = ""              # sentence accumulation
        self._got_chunks = False
        self._audio_started = False
        self._stream_rate: Optional[int] = None
        self._instr: Optional[str] = None

        self._work_q: Optional[asyncio.Queue] = None
        self._play_q: Optional[asyncio.Queue] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

    # -- sentence splitter -------------------------------------------------

    def _take_sentences(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        pos = 0
        while (m := _TERM_RE.search(self._buf, pos)) is not None:
            word = _WORD_RE.search(self._buf[: m.start()])
            if word and word.group(0).lower() in _ABBREVIATIONS:
                pos = m.start() + 1
                continue
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            pos = 0
            if sentence:
                out.append(sentence)
        return out

    # -- pipeline lifecycle --------------------------------------------------

    async def _start_pipeline(self) -> None:
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        if self._fetch_task is None:
            return
        await self._work_q.put(None)
        await self._fetch_task          # forwards its own sentinel onward
        await self._play_q.put(None)
        await self._play_task
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers ---------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        """Synthesize sentences off the event loop, one at a time.

        ``asyncio.to_thread`` keeps the loop responsive while torch
        crunches, which is what lets SynthesizeChunk events keep being
        parsed (and therefore lets the LLM stay ahead of playback).
        """
        while (text := await self._work_q.get()) is not None:
            clean, instructions = parse_tags(text)
            if not clean:
                continue

            if instructions:
                self._instr = instructions

            _LOGGER.info("synthesizing: text=%r instr=%r", clean[:60], self._instr)

            try:
                audio, rate = await asyncio.to_thread(
                    self.engine.synthesize, clean, self._instr
                )
            except Exception:
                _LOGGER.exception("sentence synthesis failed: %r", clean[:60])
                continue

            await self._play_q.put((audio, rate))

    async def _play_worker(self) -> None:
        """Write synthesized audio to the Wyoming client, unpaced."""
        while (item := await self._play_q.get()) is not None:
            audio, rate = item
            if not audio:
                continue

            if not self._audio_started:
                self._stream_rate = rate
                await self.write_event(
                    AudioStart(rate=rate, width=2, channels=1).event()
                )
                self._audio_started = True
            elif rate != self._stream_rate:
                _LOGGER.warning(
                    "sample rate changed mid-stream (%s -> %s)", self._stream_rate, rate
                )

            for i in range(0, len(audio), 4096):
                await self.write_event(
                    AudioChunk(
                        audio=audio[i : i + 4096],
                        rate=rate,
                        width=2,
                        channels=1,
                    ).event()
                )

    # -- wyoming event handling ---------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            self._stash = ""
            self._buf = ""
            self._got_chunks = False
            self._audio_started = False
            self._stream_rate = None
            self._instr = None
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            for sentence in self._take_sentences(
                SynthesizeChunk.from_event(event).text
            ):
                self._work_q.put_nowait(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._work_q.put_nowait(tail)
            else:
                text = self._stash
                self._stash = ""
                if text.strip():
                    self._work_q.put_nowait(text.strip())

            await self._stop_pipeline()

            if self._audio_started:
                await self.write_event(AudioStop().event())
                self._audio_started = False
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            self._stash = Synthesize.from_event(event).text
            return True

        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CosyVoice Wyoming bridge (direct)")
    p.add_argument("--uri", default="tcp://0.0.0.0:10211", help="Wyoming server URI")
    p.add_argument(
        "--cosyvoice-repo",
        default="CosyVoice",
        help="Path to the CosyVoice checkout (added to sys.path)",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Model dir; defaults to <repo>/pretrained_models/Fun-CosyVoice3-0.5B",
    )
    p.add_argument("--prompt-wav", default=None, help="Reference speaker WAV")
    p.add_argument(
        "--prompt-text",
        default="",
        help="Transcript of the reference WAV (needed for zero-shot prompting)",
    )
    p.add_argument(
        "--spk-id",
        default="ha-default",
        help="Cache the speaker embedding under this id (empty to disable)",
    )
    p.add_argument(
        "--mode",
        choices=["instruct", "zero_shot"],
        default="instruct",
        help="instruct = use inference_instruct2 with tag instructions; "
        "zero_shot = plain cloning",
    )
    p.add_argument(
        "--instruct",
        default=None,
        help="Fallback instruction when a sentence carries no tag",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", help="Half precision on CUDA")
    p.add_argument("--voice", default="nicoleheart", help="Voice name reported to HA")
    p.add_argument("--sample-rate", type=int, default=24000, help="Reported rate (fallback)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    repo = Path(args.cosyvoice_repo).resolve()
    if not repo.is_dir():
        raise SystemExit(f"CosyVoice repo not found: {repo}")

    # CosyVoice is a repo, not a wheel: it and Matcha-TTS must be importable.
    for extra in (repo, repo / "third_party" / "Matcha-TTS"):
        if extra.is_dir():
            sys.path.insert(0, str(extra))

    model_dir = args.model_dir or str(repo / "pretrained_models" / "Fun-CosyVoice3-0.5B")

    engine = SynthEngine(
        model_dir=model_dir,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        spk_id=args.spk_id or None,
        mode=args.mode,
        default_instruct=args.instruct,
        speed=args.speed,
        fp16=args.fp16,
    )
    # Blocking load, deliberately before the server starts listening.
    await asyncio.to_thread(engine.load)

    attribution = Attribution(
        name="FunAudioLLM",
        url="https://github.com/FunAudioLLM/CosyVoice",
    )
    info = Info(
        tts=[
            TtsProgram(
                name="cosyvoice",
                attribution=attribution,
                installed=True,
                description="CosyVoice 3 (direct, in-process)",
                version="0.5B-RL",
                voices=[
                    TtsVoice(
                        name=args.voice,
                        attribution=attribution,
                        installed=True,
                        description=f"CosyVoice 3 cloned voice ({args.voice})",
                        version="0.5B-RL",
                        languages=["en"],
                    )
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("starting on %s (model=%s)", args.uri, model_dir)

    await server.run(
        partial(
            BridgeHandler,
            engine=engine,
            info=info,
            voice=args.voice,
            sample_rate=args.sample_rate,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
