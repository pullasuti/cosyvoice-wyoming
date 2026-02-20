#!/bin/bash
#
# Record a voice sample for CosyVoice cloning.
#
# Usage:
#   bash record.sh                    # saves to voices/my_voice.wav
#   bash record.sh russian_female     # saves to voices/russian_female.wav
#
# Tips:
#   - Speak 5-10 seconds of natural speech
#   - Any language works (Russian, English, etc.)
#   - Quiet room, no background noise
#   - Press Ctrl+C to stop recording
#
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:-my_voice}"
OUT="$DIR/voices/${NAME}.wav"
mkdir -p "$DIR/voices"

echo "Recording voice sample: $NAME"
echo "Speak naturally for 5-10 seconds, then press Ctrl+C to stop."
echo ""
echo "Output: $OUT"
echo ""

sox -d -r 24000 -c 1 -b 16 "$OUT" 2>/dev/null || {
    echo ""
    echo "sox not found. Installing..."
    brew install sox
    sox -d -r 24000 -c 1 -b 16 "$OUT"
}

echo ""
DURATION=$(soxi -D "$OUT" 2>/dev/null || echo "?")
echo "Saved: $OUT (${DURATION}s)"
echo ""
echo "To use: set VOICE=$NAME in .env and restart server"
echo "Or via API: voice=$NAME / voice_name=$NAME"
