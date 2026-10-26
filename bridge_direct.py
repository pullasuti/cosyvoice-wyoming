#!/usr/bin/env python3
"""CosyVoice Wyoming TTS bridge — DIRECT variant (no HTTP, no httpx).

Loads CosyVoice3 in-process (AutoModel, path-based prompt wav, instruction
envelope) with anti-stutter and robustness measures:

  1. SENTENCE MERGING — sentences batched to --min-merged-seconds of
     estimated speech, amortizing the model's per-request warmup.
  2. PRE-BUFFERING (jitter buffer) — AudioStart deferred until
     --prebuffer-seconds is queued; afterwards every chunk is written
     immediately.
  3. UNIFORM RE-CHUNKING — PCM is re-emitted in fixed-size AudioChunks
     of --samples-per-chunk samples, with the short remainder carried
     across model yields (single flush at end of utterance).
  4. WYOMING ERROR EVENTS — synthesis failures are reported to the
     client via wyoming.error.Error so HA surfaces a failed TTS call
     with a reason instead of silent truncation. Remaining blocks of
     the utterance still run.
  5. GRACEFUL SHUTDOWN — SIGINT/SIGTERM cancel the server cleanly.
  6. RL CHECKPOINT RELOAD — uses cosyvoice.model.load(llm.rl.pt, ...)
     when available (no filesystem surgery); falls back to the
     rename-and-symlink swap if the revision lacks .load().
  7. fp16 — AutoModel(model_dir=..., fp16=True) halves compute on
     CUDA. Silently disabled by CosyVoice if no CUDA device is visible
     (startup log + post-load check surface that).

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
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
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

# Rough speech-rate estimate for merging decisions: ~2.8 words/second.
WORDS_PER_SECOND = 2.8

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
    """Apply the CosyVoice3 instruction envelope."""
    body = body.strip()
    if body.endswith(EOP):
        body = body[: -len(EOP)].strip()
    if not body.startswith(ASSISTANT_PREFIX):
        body = f"{ASSISTANT_PREFIX} {body}"
    body = body.rstrip(".")
    return f"{body}.{EOP}"


def speed_hint(speed: float) -> Optional[str]:
    """Instruction-based speed control."""
    if speed == 1.0:
        return None
    if speed < 0.7:
        return "Speak as slowly as possible"
    if speed < 1.0:
        return "Speak a bit slower than normal"
    if speed > 1.5:
        return "Speak as fast as possible"
    return "Speak a bit faster than normal"


def estimate_seconds(text: str) -> float:
    """Very rough spoken-duration estimate for merging decisions."""
    return len(text.split()) / WORDS_PER_SECOND


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def to_pcm16(tensor: Any) -> bytes:
    """CosyVoice float waveform [-1, 1] -> little-endian PCM16 bytes.

    Clamps, squeezes the batch dim and scales.
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
        fp16: bool,
        trt: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.mode = mode
        self.default_instruct = default_instruct
        self.speed = speed
        self.use_rl = use_rl
        self.fp16 = fp16
        self.trt = trt

        self.model: Any = None
        self.sample_rate: int = 24000
        self._lock = threading.Lock()

    # -- startup ------------------------------------------------------------

    def _swap_rl_checkpoint(self) -> None:
        """Fallback: swap llm.pt for llm.rl.pt via rename and symlink."""
        rl = os.path.join(self.model_dir, "llm.rl.pt")
        llm = os.path.join(self.model_dir, "llm.pt")
        bak = os.path.join(self.model_dir, "llm.base.pt")
        if os.path.exists(rl) and not os.path.exists(bak):
            _LOGGER.info("swapping in RL checkpoint (llm.pt -> llm.rl.pt)")
            os.rename(llm, bak)
            os.symlink("llm.rl.pt", llm)
        elif os.path.exists(bak):
            _LOGGER.info("RL checkpoint already swapped in")

    def _load_rl_weights(self) -> None:
        """Preferred: reload RL weights in memory via model.load().

        Cleaner than the file swap — no mutation of the model dir, no
        risk of a fresh download silently reverting to the base
        checkpoint. Falls back to the file swap if this CosyVoice
        revision does not expose .load().
        """
        llm_rl = os.path.join(self.model_dir, "llm.rl.pt")
        flow = os.path.join(self.model_dir, "flow.pt")
        hift = os.path.join(self.model_dir, "hift.pt")
        if not os.path.exists(llm_rl):
            _LOGGER.warning(
                "%s not found; keeping whatever llm.pt resolves to", llm_rl
            )
            return

        try:
            inner = self.model.model  # AutoModel -> CosyVoice3 -> .model
            load_fn = getattr(inner, "load", None)
            if not callable(load_fn):
                raise AttributeError("inner model has no .load()")
            load_fn(llm_rl, flow, hift)
            _LOGGER.info("reloaded LLM weights from %s (in-memory, no file swap)", llm_rl)
        except Exception:
            _LOGGER.warning(
                "model.load() RL reload unavailable (%s); "
                "falling back to file-swap", exc_info=True,
            )
            self._swap_rl_checkpoint()

    def _attach_trt(self) -> None:
        """Replace the flow decoder (DiT) with a TensorRT engine.

        Done AFTER the RL weight reload on purpose: model.load() reloads
        flow.pt too, which fails once the estimator is a TensorRT wrapper
        (and the file-swap fallback would silently leave the base weights).

        The first start builds the engine from flow.decoder.estimator.fp32.onnx
        (several minutes) and caches it next to the model files; later starts
        just load it. Any failure leaves the normal PyTorch flow in place.
        NOTE: the engine always uses full attention (CosyVoice's export has no
        'streaming' input), while PyTorch streaming uses chunked attention.
        The model is trained with both, but listen for differences.
        """
        inner = getattr(self.model, "model", None)
        load_trt = getattr(inner, "load_trt", None)
        if not callable(load_trt):
            _LOGGER.warning("this CosyVoice revision has no load_trt(); skipping TensorRT")
            return
        prec = "fp16" if self.fp16 else "fp32"
        plan = os.path.join(self.model_dir, f"flow.decoder.estimator.{prec}.mygpu.plan")
        onnx = os.path.join(self.model_dir, "flow.decoder.estimator.fp32.onnx")
        if not (os.path.exists(plan) and os.path.getsize(plan) > 0) \
                and not os.path.exists(onnx):
            _LOGGER.warning(
                "TensorRT: neither %s nor %s exists; keeping the PyTorch flow",
                os.path.basename(plan), os.path.basename(onnx))
            return
        had_plan = os.path.exists(plan) and os.path.getsize(plan) > 0
        if not had_plan:
            _LOGGER.info("TensorRT: building %s engine (first run only, this "
                         "can take several minutes) ...", prec)
        t0 = time.monotonic()
        original_estimator = inner.flow.decoder.estimator
        try:
            load_trt(plan, onnx, 1, self.fp16)
        except Exception:
            _LOGGER.exception("TensorRT setup failed; keeping the PyTorch flow")
            # load_trt deletes the PyTorch estimator before the engine is
            # deserialized, so put it back if that was the step that failed
            if not hasattr(inner.flow.decoder, "estimator"):
                inner.flow.decoder.estimator = original_estimator
            try:  # a half-written plan would be reused on the next start
                if not had_plan and os.path.exists(plan):
                    os.remove(plan)
            except OSError:
                pass
            return
        _LOGGER.info("TensorRT flow engine ready (%s) in %.1fs",
                     prec, time.monotonic() - t0)

    def _cache_prompt_features(self) -> None:
        """Memoize the reference-wav feature extraction in the frontend.

        frontend_instruct2() re-runs the speech tokenizer, the CAM++ speaker
        embedding and the mel extractor on the reference wav for EVERY text
        segment. The reference never changes, so compute each once.
        (add_zero_shot_spk is not an option here: with a cached speaker
        CosyVoice ignores the instruct text, which would silently drop the
        emotion instructions.) Tensors are cloned on return because the
        frontend modifies the length tensors in place.
        """
        frontend = getattr(self.model, "frontend", None)
        if frontend is None:
            _LOGGER.warning("no frontend found; prompt features not cached")
            return

        def clone(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.clone()
            if isinstance(v, tuple):
                return tuple(clone(x) for x in v)
            return v

        for name in ("_extract_speech_token", "_extract_spk_embedding",
                     "_extract_speech_feat"):
            fn = getattr(frontend, name, None)
            if fn is None:
                continue
            cache: dict = {}

            def wrapper(prompt_wav, _fn=fn, _cache=cache):
                if prompt_wav not in _cache:
                    _cache[prompt_wav] = _fn(prompt_wav)
                return clone(_cache[prompt_wav])

            setattr(frontend, name, wrapper)
        _LOGGER.info("reference-voice features will be cached after first use")

    def load(self) -> None:
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

        _LOGGER.info("loading model from %s (fp16=%s)", self.model_dir, self.fp16)
        t0 = time.monotonic()

        if self.fp16:
            try:
                self.model = AutoModel(model_dir=self.model_dir, fp16=True)
            except TypeError:
                _LOGGER.warning(
                    "this CosyVoice revision does not accept fp16; "
                    "loading in fp32"
                )
                self.model = AutoModel(model_dir=self.model_dir)
        else:
            self.model = AutoModel(model_dir=self.model_dir)

        if self.use_rl:
            self._load_rl_weights()

        # CosyVoice silently forces fp16 off when no CUDA device is
        # visible — surface that so a misconfigured GPU isn't mistaken
        # for a successful fp16 load.
        if self.fp16 and getattr(self.model, "fp16", None) is not True:
            _LOGGER.warning(
                "fp16 was requested but the model reports fp16=%s "
                "(no CUDA device visible?)", getattr(self.model, "fp16", None)
            )

        if self.trt:
            self._attach_trt()

        self._cache_prompt_features()

        self.sample_rate = int(self.model.sample_rate)
        _LOGGER.info(
            "model ready in %.1fs (sample_rate=%d)",
            time.monotonic() - t0, self.sample_rate,
        )

        if not self.prompt_wav or not os.path.exists(self.prompt_wav):
            raise RuntimeError(f"prompt WAV not found: {self.prompt_wav!r}")

    # -- inference ----------------------------------------------------------

    def _call(self, text: str, instructions: Optional[str]) -> Iterator[Any]:
        """Pick the CosyVoice inference mode for this block."""
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

        return self.model.inference_cross_lingual(
            f"{ASSISTANT_PREFIX}{EOP}{text}", self.prompt_wav, stream=True
        )

    def synthesize_iter(
        self, text: str, instructions: Optional[str]
    ) -> Iterator[tuple[bytes, int]]:
        """Blocking generator: yields (pcm16 chunk, sample_rate) as the model
        produces them. Run it in a worker thread and forward each chunk
        immediately; collecting them into a list first would hold back all
        audio until the whole block is finished."""
        with self._lock:
            with torch.inference_mode():
                for out in self._call(text, instructions):
                    yield to_pcm16(out["tts_speech"]), self.sample_rate


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
        min_merged_seconds: float,
        prebuffer_seconds: float,
        samples_per_chunk: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.voice = voice
        self.sample_rate = sample_rate
        self._ready = ready
        self._min_merged = min_merged_seconds
        self._prebuffer = prebuffer_seconds
        self._samples_per_chunk = samples_per_chunk

        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._stream_rate: Optional[int] = None
        self._instr: Optional[str] = None

        # Sentence merging accumulator
        self._pending_sentences: list[str] = []
        self._first_queued = False

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

    # -- sentence merging ----------------------------------------------------

    def _feed_sentence(self, sentence: str) -> None:
        """Accumulate sentences; queue a merged block once the estimated
        speech length exceeds the merge threshold (amortizes the model's
        per-request flow warmup). Hard cap of 6 sentences bounds onset
        latency for fragment-heavy replies."""
        self._pending_sentences.append(sentence)
        if not self._first_queued:
            # LATENCY-FIRST: don't wait to merge the opening sentence
            self._first_queued = True
            self._flush_pending()
            return
        est = sum(estimate_seconds(s) for s in self._pending_sentences)
        if est >= self._min_merged or len(self._pending_sentences) >= 6:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending_sentences:
            merged = " ".join(self._pending_sentences)
            self._pending_sentences = []
            self._work_q.put_nowait(merged)

    # -- pipeline lifecycle --------------------------------------------------

    async def _start_pipeline(self) -> None:
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        if self._fetch_task is None:
            return
        self._flush_pending()
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
        """Synthesize merged blocks off the event loop, one at a time.

        Failures are reported to the client as a Wyoming Error event so
        HA surfaces a diagnosable failure rather than silence — but the
        remaining blocks of the utterance still run.
        """
        await self._ready.wait()

        while (text := await self._work_q.get()) is not None:
            clean, instructions = parse_tags(text)
            if not clean:
                continue

            if instructions:
                self._instr = instructions

            _LOGGER.info("synthesizing: text=%r instr=%r", clean[:60], self._instr)

            try:
                await self._stream_block(clean, self._instr)
            except Exception as err:
                _LOGGER.exception("block synthesis failed: %r", clean[:60])
                try:
                    await self.write_event(
                        Error(
                            text=f"synthesis failed: {err}",
                            code=err.__class__.__name__,
                        ).event()
                    )
                except Exception:
                    pass  # connection may already be gone
                continue

    async def _stream_block(self, text: str, instructions: Optional[str]) -> None:
        """Run the model in a worker thread and hand every chunk to the play
        queue the moment it is produced (no waiting for the whole block)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for item in self.engine.synthesize_iter(text, instructions):
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except BaseException as err:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, err)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        error: Optional[BaseException] = None
        first = True
        t0 = time.monotonic()
        while (item := await q.get()) is not None:
            if isinstance(item, BaseException):
                error = item
                continue
            audio, rate = item
            if audio:
                if first:
                    _LOGGER.info("first audio chunk after %.2fs",
                                 time.monotonic() - t0)
                    first = False
                await self._play_q.put((audio, rate))
        await fut
        if error is not None:
            raise error

    async def _play_worker(self) -> None:
        """Write synthesized audio to the Wyoming client.

        Semantics:
          * JITTER BUFFER — before AudioStart, accumulate until
            --prebuffer-seconds is buffered (or the utterance ends);
            afterwards, drain continuously.
          * UNIFORM CHUNKING — audio is re-emitted in fixed
            --samples-per-chunk AudioChunks; the short remainder is
            carried across model yields and flushed once at the end of
            the utterance, so the player sees a constant cadence.
        """
        width = 2
        bytes_per_chunk = width * self._samples_per_chunk

        buf = bytearray()
        rate: Optional[int] = None
        started = False
        buffered_bytes = 0

        async def _maybe_emit_start() -> None:
            nonlocal started
            if not self._audio_started and buf:
                self._stream_rate = rate
                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=1).event()
                )
                self._audio_started = True
            started = True

        while (item := await self._play_q.get()) is not None:
            audio, r = item
            if not audio:
                continue

            if rate is None:
                rate = r
            elif r != rate:
                _LOGGER.warning(
                    "sample rate changed mid-stream (%s -> %s)", rate, r
                )

            buf.extend(audio)
            buffered_bytes += len(audio)

            if not started:
                prebuffer_bytes = int(
                    self._prebuffer * 2 * (rate or self.sample_rate)
                )
                if buffered_bytes < prebuffer_bytes:
                    continue
                await _maybe_emit_start()

            # Emit every complete fixed-size chunk now.
            while len(buf) >= bytes_per_chunk:
                await self.write_event(
                    AudioChunk(
                        audio=bytes(buf[:bytes_per_chunk]),
                        rate=rate,
                        width=width,
                        channels=1,
                    ).event()
                )
                del buf[:bytes_per_chunk]

        # End sentinel: short utterances that never reached the
        # prebuffer threshold are emitted here in full.
        if buf:
            await _maybe_emit_start()
            await self.write_event(
                AudioChunk(
                    audio=bytes(buf), rate=rate, width=width, channels=1
                ).event()
            )
            buf.clear()

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
            self._pending_sentences = []
            self._first_queued = False
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            for sentence in self._take_sentences(
                SynthesizeChunk.from_event(event).text
            ):
                self._feed_sentence(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._feed_sentence(tail)
            else:
                text = self._stash
                self._stash = ""
                if text.strip():
                    self._feed_sentence(text.strip())

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
        help="Mode for untagged text (tagged text always uses instruct)",
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
        help="Load RL weights (llm.rl.pt): via model.load() when available, "
        "falling back to a file swap",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Load the model in half precision (silently disabled by "
        "CosyVoice if no CUDA device is visible)",
    )
    p.add_argument(
        "--trt",
        action="store_true",
        help="Run the flow decoder through TensorRT (builds an engine on "
        "the first start). Falls back to PyTorch if anything fails.",
    )
    p.add_argument(
        "--min-merged-seconds",
        type=float,
        default=4.0,
        help="Batch sentences until the estimated speech length reaches "
        "this many seconds (0 disables merging). Amortizes the model's "
        "per-request flow warmup.",
    )
    p.add_argument(
        "--prebuffer-seconds",
        type=float,
        default=1.5,
        help="Hold playback until this much audio is queued, so the "
        "player never starves mid-utterance (0 disables).",
    )
    p.add_argument(
        "--samples-per-chunk",
        type=int,
        default=2048,
        help="Re-emit AudioChunks of this many samples (~85ms at 24kHz "
        "mono; remainder flushed at end of utterance).",
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
        fp16=args.fp16,
        trt=args.trt,
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
        "listening on %s (model=%s, fp16=%s) — loading model in background",
        args.uri, model_dir, args.fp16,
    )

    # Graceful shutdown: SIGINT/SIGTERM cancel the server task instead
    # of relying on default kill semantics.
    server_task = asyncio.create_task(
        server.run(
            partial(
                BridgeHandler,
                engine=engine,
                info=info,
                voice=args.voice,
                sample_rate=args.sample_rate,
                ready=ready,
                min_merged_seconds=args.min_merged_seconds,
                prebuffer_seconds=args.prebuffer_seconds,
                samples_per_chunk=args.samples_per_chunk,
            )
        )
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server_task.cancel)
    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("server stopped")


if __name__ == "__main__":
    asyncio.run(main())
