"""Exercise the CosyVoice bridge's one-shot, streaming and duplicate paths.

Run against a running bridge:
    .venv/bin/python test_bridge.py [--host HOST] [--port PORT]

Expected: audio for every request except the late duplicate (TIMEOUT) and
the blank one-shot (empty audio-start/audio-stop pair).
"""
import argparse
import asyncio
import time

from wyoming.audio import AudioChunk
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

HOST = "127.0.0.1"
PORT = 10214


async def collect(c, until, timeout=30.0):
    """Read until an event of type `until`; return (first-audio s, audio s, types)."""
    t = time.monotonic()
    first, samples, types = None, 0, []
    try:
        while True:
            ev = await asyncio.wait_for(c.read_event(), timeout)
            if ev is None:
                break
            types.append(ev.type)
            if AudioChunk.is_type(ev.type):
                ch = AudioChunk.from_event(ev)
                samples += len(ch.audio) // 2
                first = first if first is not None else time.monotonic() - t
            if ev.type == until:
                break
    except asyncio.TimeoutError:
        types.append("TIMEOUT")
    return first, samples / 24000, types


def summary(types):
    out = []
    for t in types:
        if out and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return " ".join(f"{t}x{n}" if n > 1 else t for t, n in out)


async def oneshot(text):
    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(Synthesize(text=text).event())
        return await collect(c, "audio-stop")


async def stream_with_compat(text, late_dup=False):
    """Mimic HA: start, chunks, compat one-shot, stop (+ optional late dup)."""
    async with AsyncTcpClient(HOST, PORT) as c:
        await c.write_event(SynthesizeStart().event())
        half = len(text) // 2
        for part in (text[:half], text[half:]):
            await c.write_event(SynthesizeChunk(text=part).event())
        await c.write_event(Synthesize(text=text).event())
        await c.write_event(SynthesizeStop().event())
        res = await collect(c, "synthesize-stopped")
        if late_dup:
            await c.write_event(Synthesize(text=text).event())
            res = (res, await collect(c, "audio-stop", timeout=3.0))
        return res


async def main():
    first, dur, types = await oneshot("[warm] First request after restart.")
    print(f"warm-start one-shot: first {first:.2f}s, audio {dur:.1f}s | {summary(types)}")

    first, dur, types = await oneshot("[happy] Hello! [laughter] This is a one-shot request.")
    print(f"one-shot:            first {first:.2f}s, audio {dur:.1f}s | {summary(types)}")

    first, dur, types = await stream_with_compat("[warm] The lights are off. Good night!")
    print(f"stream+compat:       first {first:.2f}s, audio {dur:.1f}s | {summary(types)}")

    (s_first, s_dur, s_types), (_, d_dur, d_types) = await stream_with_compat(
        "[serious] The garage door is open.", late_dup=True)
    print(f"stream:              first {s_first:.2f}s, audio {s_dur:.1f}s | {summary(s_types)}")
    print(f"  late duplicate:    audio {d_dur:.1f}s | {summary(d_types)}")

    first, dur, types = await oneshot("   ")
    print(f"blank one-shot:      audio {dur:.1f}s | {summary(types)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    HOST, PORT = args.host, args.port
    asyncio.run(main())
