#!/usr/bin/env python3
"""CosyVoice Wyoming TTS bridge — streams to HA via Wyoming protocol.

Streams sentence-by-sentence: synthesis starts as soon as a sentence
boundary appears in the incoming SynthesizeChunk stream, while the LLM
is still generating the rest of the reply.

Pipeline layout (v2):

    LLM chunks ──► SynthesizeChunk ──► sentence queue ──► fetch worker ──► play queue ──► play worker ──► HA

Two decoupled workers means the fetch of sentence N+1 overlaps the
write-out of sentence N, hiding per-request model latency between
sentences.
"""

from __future__ import annotations

import time
import argparse
import asyncio
import logging
import re
from functools import partial
from typing import Optional

import httpx
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger(__name__)

# Tag → instruction mapping (preset-register phrasing that works)
TAG_MAP = {
    "warm": "Please speak in a warm and friendly voice.",
    "confident": "Please speak with a confident and steady voice.",
    "happy": "Speak with a happy and excited tone.",
    "sad": "Speak with a sad and melancholy tone.",
    "excited": "Speak with an excited and energetic tone.",
    "whisper": "Please say a sentence in a very soft voice.",
    "serious": "Please speak in a serious and calm manner.",
}

# Inline paralinguistic tags that pass through unchanged
INLINE_TAGS = {"laughter", "breath", "sigh", "cough", "noise", "lipsmack", "mn"}

TAG_RE = re.compile(r"\s*[\[\{]([^\]\}]+)[\]\}]")

# Pre-normalize: turn (tag) into [tag] ONLY for known vocabulary.
# Unknown parentheticals are prose ("(since Tuesday)") and stay untouched.
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

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# A terminator (optionally followed by closing quotes/brackets) plus
# whitespace. Requiring whitespace keeps decimals ("3.14") intact; text
# ending flush on a terminator is flushed when SynthesizeStop arrives.
_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")

# Word immediately before a terminator — used to reject abbreviations.
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

        key = ALIASES.get(tag.lower(), tag.lower())   # fold variants
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


class BridgeHandler(AsyncEventHandler):
    def __init__(self, *args, client: httpx.AsyncClient, info: Info,
                 voice: str, sample_rate: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.info = info
        self.voice = voice
        self.sample_rate = sample_rate
        # Legacy single-shot Synthesize event text
        self._stash = ""
        # Streaming utterance state
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._instr: Optional[str] = None
        # Pipeline: sentences in, fetched audio out. None = end sentinel.
        self._work_q: Optional[asyncio.Queue] = None
        self._play_q: Optional[asyncio.Queue] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

    # -- sentence splitter -------------------------------------------------

    def _take_sentences(self, text: str) -> list[str]:
        """Append text to the buffer and return any complete sentences."""
        self._buf += text
        out: list[str] = []
        pos = 0
        while (m := _TERM_RE.search(self._buf, pos)) is not None:
            word = _WORD_RE.search(self._buf[:m.start()])
            if word and word.group(0).lower() in _ABBREVIATIONS:
                # "e.g." and friends — not a sentence end, keep scanning
                pos = m.start() + 1
                continue
            sentence = self._buf[:m.end()].strip()
            self._buf = self._buf[m.end():]
            pos = 0
            if sentence:
                out.append(sentence)
        return out

    # -- pipeline lifecycle --------------------------------------------------

    async def _start_pipeline(self) -> None:
        """Spawn the fetch and play workers for this utterance."""
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        """Drain the pipeline: wait for all pending sentences to be
        fetched AND played before returning."""
        if self._fetch_task is None:
            return
        await self._work_q.put(None)          # fetch worker: no more input
        await self._fetch_task                 # ...which forwards its own
        await self._play_q.put(None)          # sentinel after the audio
        await self._play_task                  # play worker: all written
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers ---------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        """Pull sentences off the work queue and hit CosyVoice for each.

        Runs concurrently with the play worker, so sentence N+1's HTTP
        round-trip and model latency overlap sentence N's playback.
        """
        while (text := await self._work_q.get()) is not None:
            clean, instructions = parse_tags(text)
            if not clean:
                continue

            # Remember the last instruction tag for the rest of the
            # utterance so per-sentence requests keep consistent styling.
            if instructions:
                self._instr = instructions

            _LOGGER.info("synthesizing: text=%r instr=%r",
                         clean[:60], self._instr)

            payload = {
                "input": clean,
                "voice": self.voice,
                "stream": True,
            }
            if self._instr:
                payload["instructions"] = self._instr

            try:
                async with self.client.stream(
                        "POST", "/v1/audio/speech", json=payload) as resp:
                    resp.raise_for_status()

                    rate = int(resp.headers.get(
                        "X-Audio-Sample-Rate", self.sample_rate))
                    if rate != self.sample_rate:
                        _LOGGER.warning(
                            "server sample rate %d != configured %d",
                            rate, self.sample_rate)

                    # Read the whole sentence. Latency is hidden by the
                    # play worker, which is writing the previous sentence
                    # while we do this.
                    audio = bytearray()
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        audio.extend(chunk)

                await self._play_q.put((bytes(audio), rate))

            except Exception:
                # Log and continue: one failed sentence must not kill
                # the rest of the utterance.
                _LOGGER.exception("sentence synthesis failed: %r",
                                  clean[:60])

    async def _play_worker(self) -> None:
        """Write fetched audio to the Wyoming client.

        Reads full sentences from the play queue and emits them in
        large chunks (≈85 ms of audio each) so event-write overhead
        stays far away from real-time pace. NO artificial sleeps.
        """
        while (item := await self._play_q.get()) is not None:
            audio, rate = item
            if not audio:
                continue

            if not self._audio_started:
                await self.write_event(
                    AudioStart(rate=self.sample_rate, width=2, channels=1).event()
                )
                self._audio_started = True

            _t0 = time.monotonic()
            n = 0
            for i in range(0, len(audio), 4096):
                await self.write_event(
                    AudioChunk(
                        audio=audio[i:i + 4096],
                        rate=rate,
                        width=2,
                        channels=1,
                    ).event()
                )
                n += 1
                if n <= 2:
                    _LOGGER.info("wrote %d bytes / %d chunks in %.0f ms",
                                 len(audio), n, (time.monotonic() - _t0) * 1000)

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
            self._instr = None
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            # Queue each completed sentence; the fetch worker picks it
            # up immediately while the LLM keeps generating.
            for sentence in self._take_sentences(
                    SynthesizeChunk.from_event(event).text):
                self._work_q.put_nowait(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                # Flush anything left (final sentence often has no
                # trailing whitespace).
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._work_q.put_nowait(tail)
            else:
                # No chunk events: legacy single-shot Synthesize text.
                text = self._stash
                self._stash = ""
                if text.strip():
                    self._work_q.put_nowait(text.strip())

            # Wait for the pipeline to drain — every queued sentence
            # fetched AND written — before signaling completion.
            await self._stop_pipeline()

            if self._audio_started:
                await self.write_event(AudioStop().event())
                self._audio_started = False
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            # compat event: stash, never speak — the stop event will fire
            self._stash = Synthesize.from_event(event).text
            return True

        return True


async def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice Wyoming TTS bridge")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10211",
                        help="Wyoming server URI")
    parser.add_argument("--base-url", default="http://127.0.0.1:8020",
                        help="CosyVoice API base URL")
    parser.add_argument("--voice", default="nicoleheart",
                        help="Default voice name (must exist in CosyVoice voices/)")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="Audio sample rate in Hz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = httpx.AsyncClient(
        base_url=args.base_url,
        timeout=httpx.Timeout(120.0, connect=5.0)
    )

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
                description="CosyVoice 3 via streaming Wyoming bridge",
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
    _LOGGER.info("Starting on %s -> %s", args.uri, args.base_url)

    await server.run(
        partial(
            BridgeHandler,
            client=client,
            info=info,
            voice=args.voice,
            sample_rate=args.sample_rate,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
