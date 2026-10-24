#!/usr/bin/env python3
"""CosyVoice Wyoming TTS bridge — DIRECT variant (no HTTP, no httpx).

Corrected against the working server.py. Differences that mattered:

  * Class is ``AutoModel``, constructed as ``AutoModel(model_dir=...)``
    with NO extra kwargs (load_jit/load_trt/fp16 are not accepted here).
  * ``prompt_wav`` is passed as a FILE PATH, not a pre-loaded tensor —
    server.py never calls load_wav().
  * Instructions are wrapped in the CosyVoice3 envelope:
    "You are a helpful assistant. <instruction>.<|endofprompt|>"
  * Default mode is cross_lingual, which prefixes the TEXT with
    "You are a helpful assistant.<|endofprompt|>" and ignores prompt
    text entirely.
  * The RL checkpoint swap (llm.pt -> llm.rl.pt symlink) is replicated
    from server.py's startup(), guarded identically so it is idempotent.
  * Speaker caching (add_zero_shot_spk) is NOT used: server.py does not
    use it, so it is left out rather than guessed at.
  * Speed is expressed as an instruction hint, as server.py does — the
    inference methods take no ``speed`` kwarg.

Startup order: bind :10211 immediately, load the model in the
background, queue any sentences that arrive during the load.

Must run inside the CosyVoice venv (torch, numpy, wyoming, and the
CosyVoice repo on sys.path via --cosyvoice-repo).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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

ASSISTANT_PREFIX = "You are a helpful assistant."
EOP = "<|endofprompt|>"

# ---------------------------------------------------------------------------
# Tag handling — values are BARE instruction bodies, wrapped at call time.
# ---------------------------------------------------------------------------

TAG_MAP = {
    "warm": "Speak in a warm and friendly voice",
    "confident": "Speak with a confident and steady voice",
    "happy": "Speak with a happy and excited tone",
    "sad": "Speak with a sad and melancholic tone",
    "angry": "Speak with an angry tone",
    "excited": "Speak with an excited and energetic tone",
    "whisper": "Please say a sentence in a very soft voice",
    "serious": "Speak in a serious and calm manner",
    "loud": "Please say a sentence as loudly as possible",
    "slow": "Speak as slowly as possible",
    "fast": "Speak as fast as possible",
    "robot": "Speak like a robot",
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
    "melancholy": "sad",
    "melancholic": "sad",
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
    """Split text into (spoken text, instruction body or None)."""
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
    return clean, (". ".join(tags) if tags else None)


def wrap_instruct(body: str) -> str:
    """Apply the CosyVoice3 instruction envelope, mirroring server.py.

    server.py does ``". ".join(parts) + ".<|endofprompt|>"``; we strip a
    trailing period first so we don't emit ".." when the caller already
    punctuated the instruction.
    """
    body = body.strip()
    if body.endswith(EOP):
        body = body[: -len(EOP)].strip()
    if not body.startswith(ASSISTANT_PREFIX):
        body = f"{ASSISTANT_PREFIX} {body}"
    body = body.rstrip(".")
    return f"{body}.{EOP}"


def speed_hint(speed: float) -> Optional[str]:
    """Instruction-based speed control, matching server.py's thresholds."""
    if speed == 1.0:
        return None
    if speed < 0.7:
        return "Speak as slowly as possible"
    if speed < 1.0:
        return "Speak a bit slower than normal"
    if speed > 1.5:
        return "Speak as fast as possible"
    return "Speak a bit faster than normal"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def to_pcm16(tensor: Any) -> bytes:
    """CosyVoice float waveform [-1, 1] -> little-endian PCM16 bytes.

    Mirrors server.py's _to_pcm16 (clamp, squeeze batch dim, scale).
    """
    x = tensor.detach().cpu().float().clamp(-1.0, 1.0)
    if x.dim() == 2:
        x = x.squeeze(0)
    return (x * 32767.0).to(torch.int16).numpy().tobytes()


# ---------------------------------------------------------------------------
# Model wrapper — the ONLY place that talks to CosyVoice
# ---------------------------------------------------------------------------

class SynthEngine:
    """Loads CosyVoice once and exposes a blocking synthesize() call."""

    def __init__(
        self,
        model_dir: str,
        prompt_wav: str,
        prompt_text: str,
        mode: str,
        default_instruct: Optional[str],
        speed: float,
        use_rl: bool,
    ) -> None:
        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.mode = mode
        self.default_instruct = default_instruct
        self.speed = speed
        self.use_rl = use_rl

        self.model: Any = None
        self.sample_rate: int = 24000
        self._lock = threading.Lock()

    # -- startup ------------------------------------------------------------

    def _swap_rl_checkpoint(self) -> None:
        """Replicate server.py's startup() RL swap, idempotently.

        server.py renames llm.pt -> llm.base.pt and symlinks llm.rl.pt
        in its place. Guarded exactly as server.py guards it, so running
        this repeatedly is a no-op.
        """
        rl = os.path.join(self.model_dir, "llm.rl.pt")
        llm = os.path.join(self.model_dir, "llm.pt")
        bak = os.path.join(self.model_dir, "llm.base.pt")
        if os.path.exists(rl) and not os.path.exists(bak):
            _LOGGER.info("swapping in RL checkpoint (llm.pt -> llm.rl.pt)")
            os.rename(llm, bak)
            os.symlink("llm.rl.pt", llm)
        elif os.path.exists(bak):
            _LOGGER.info("RL checkpoint already swapped in")

    def load(self) -> None:
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

        if self.use_rl:
            try:
                self._swap_rl_checkpoint()
            except Exception:
                _LOGGER.exception("RL swap failed; continuing with current llm.pt")

        _LOGGER.info("loading model from %s", self.model_dir)
        t0 = time.monotonic()

        # Exactly the construction server.py uses: keyword only, no
        # load_jit / load_trt / fp16 — AutoModel does not take them.
        self.model = AutoModel(model_dir=self.model_dir)

        self.sample_rate = int(self.model.sample_rate)
        _LOGGER.info(
            "model ready in %.1fs (sample_rate=%d)",
            time.monotonic() - t0, self.sample_rate,
        )

        if not self.prompt_wav or not os.path.exists(self.prompt_wav):
            raise RuntimeError(f"prompt WAV not found: {self.prompt_wav!r}")

    # -- inference ----------------------------------------------------------

    def _call(self, text: str, instructions: Optional[str]) -> Iterator[Any]:
        """Dispatch mirroring server.py's _generate().

        All three entry points take the prompt WAV as a PATH.
        """
        instruct_body = instructions or self.default_instruct
        hint = speed_hint(self.speed)
        if hint:
            instruct_body = f"{instruct_body}. {hint}" if instruct_body else hint

        if instruct_body:
            return self.model.inference_instruct2(
                text, wrap_instruct(instruct_body), self.prompt_wav, stream=True
            )

        if self.mode == "zero_shot":
            prompt_text = self.prompt_text or f"{ASSISTANT_PREFIX}{EOP}"
            return self.model.inference_zero_shot(
                text, prompt_text, self.prompt_wav, stream=True
            )

        # Default: cross_lingual, prefix glued onto the TEXT (server.py
        # does the same — prompt text is unused in this mode).
        return self.model.inference_cross_lingual(
            f"{ASSISTANT_PREFIX}{EOP}{text}", self.prompt_wav, stream=True
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
        ready: asyncio.Event,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.voice = voice
        self.sample_rate = sample_rate
        self._ready = ready

        self._stash = ""
        self._buf = ""
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
        await self._fetch_task
        await self._play_q.put(None)
        await self._play_task
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers ---------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        """Synthesize sentences off the event loop, one at a time."""
        await self._ready.wait()

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
    p.add_argument(
        "--prompt-wav",
        required=True,
        help="Reference speaker WAV path (passed straight to the model)",
    )
    p.add_argument(
        "--prompt-text",
        default="",
        help="Transcript of the reference WAV (only used in zero_shot mode)",
    )
    p.add_argument(
        "--mode",
        choices=["cross_lingual", "zero_shot", "instruct"],
        default="cross_lingual",
        help="cross_lingual matches server.py's default; instruct forces "
        "inference_instruct2 for every sentence",
    )
    p.add_argument(
        "--instruct",
        default=None,
        help="Baseline instruction applied to every sentence (tag overrides it)",
    )
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = normal")
    p.add_argument(
        "--use-rl",
        action="store_true",
        help="Swap in llm.rl.pt as llm.pt, as server.py's startup() does",
    )
    p.add_argument("--voice", default="nicoleheart", help="Voice name reported to HA")
    p.add_argument("--sample-rate", type=int, default=24000, help="Fallback rate")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    repo = Path(args.cosyvoice_repo).resolve()
    if not repo.is_dir():
        raise SystemExit(f"CosyVoice repo not found: {repo}")

    for extra in (repo, repo / "third_party" / "Matcha-TTS"):
        if extra.is_dir():
            sys.path.insert(0, str(extra))

    model_dir = args.model_dir or str(repo / "pretrained_models" / "Fun-CosyVoice3-0.5B")

    engine = SynthEngine(
        model_dir=model_dir,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        mode=args.mode,
        default_instruct=args.instruct,
        speed=args.speed,
        use_rl=args.use_rl,
    )

    ready = asyncio.Event()

    async def _load_engine() -> None:
        try:
            await asyncio.to_thread(engine.load)
        except Exception:
            _LOGGER.critical("model load failed", exc_info=True)
            os._exit(1)
        _LOGGER.info("engine ready — synthesis enabled")
        ready.set()

    loader = asyncio.create_task(_load_engine())

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
    _LOGGER.info(
        "listening on %s (model=%s) — loading model in background",
        args.uri, model_dir,
    )

    await server.run(
        partial(
            BridgeHandler,
            engine=engine,
            info=info,
            voice=args.voice,
            sample_rate=args.sample_rate,
            ready=ready,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
