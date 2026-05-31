#!/usr/bin/env bash
# Generate synthetic audio fixtures for sanity-checking the analyzer.
# Requires ffmpeg (with libmp3lame and soxr). Run from anywhere:
#   ./testdata/make_fixtures.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "Generating fixtures in $(pwd) ..."

# Genuine full-band 44.1k/16-bit: white noise reaches ~22 kHz, no wall.
ffmpeg -v error -y -f lavfi -i "anoisesrc=d=20:color=white:r=44100:amplitude=0.5" \
    -ac 2 -sample_fmt s16 genuine_44k.flac

# Genuine hi-res 96k/24-bit: white noise truly reaching ~48 kHz.
ffmpeg -v error -y -f lavfi -i "anoisesrc=d=20:color=white:r=96000:amplitude=0.5" \
    -ac 2 -sample_fmt s32 genuine_96k.flac

# Fake lossless: encode to lossy then re-wrap as FLAC (leaves a brick wall).
ffmpeg -v error -y -f lavfi -i "anoisesrc=d=20:color=white:r=44100:amplitude=0.5" \
    -ac 2 -b:a 128k _tmp_128.mp3
ffmpeg -v error -y -i _tmp_128.mp3 -sample_fmt s16 fake_from_mp3_128.flac
ffmpeg -v error -y -f lavfi -i "anoisesrc=d=20:color=white:r=44100:amplitude=0.5" \
    -ac 2 -b:a 320k _tmp_320.mp3
ffmpeg -v error -y -i _tmp_320.mp3 -sample_fmt s16 fake_from_mp3_320.flac
rm -f _tmp_128.mp3 _tmp_320.mp3

# Upsampled: 44.1k content resampled up to 96k with a steep filter (hard wall
# at the old 22.05 kHz Nyquist) — the fake-hires signature.
ffmpeg -v error -y -i genuine_44k.flac \
    -af "aresample=96000:resampler=soxr:precision=28" -sample_fmt s32 upsampled_96k.flac

# Clipped: a loud 1 kHz tone driven well past full scale.
ffmpeg -v error -y -f lavfi -i "sine=frequency=1000:duration=15:sample_rate=44100" \
    -af "volume=30dB" -ac 2 -sample_fmt s16 clipped.flac

echo "Done:"
ls -1 *.flac
