#!/usr/bin/env bash
# Generate synthetic audio fixtures for sanity-checking the analyzer.
# Requires ffmpeg (with libmp3lame and soxr). Run from anywhere:
#   bash testdata/make_fixtures.sh
# Noise sources are seeded so the fixtures (and test expectations) are
# reproducible.
set -euo pipefail
cd "$(dirname "$0")"

echo "Generating fixtures in $(pwd) ..."

ff() { ffmpeg -v error -nostdin -y "$@"; }
# Lossy round trip: <src> <lame args...> <dst>. Leaves the lossy lowpass wall.
via_mp3() {
    local src=$1 dst=${!#}
    ff -i "$src" -c:a libmp3lame "${@:2:$#-2}" _tmp.mp3
    ff -i _tmp.mp3 -sample_fmt s16 "$dst"
    rm -f _tmp.mp3
}

WHITE="anoisesrc=d=20:color=white:r=44100:amplitude=0.5:seed=1"
# Music-like spectrum: pink noise (-3 dB/oct) with a treble rolloff above 12 kHz.
MUSIC="anoisesrc=d=20:color=pink:r=44100:amplitude=0.5:seed=7,lowpass=f=12000"
# Quiet, treble-light material: the lossy wall is shallow here.
QUIET="anoisesrc=d=20:color=pink:r=44100:amplitude=0.5:seed=7,lowpass=f=8000,volume=-25dB"

# Genuine full-band 44.1k/16-bit: white noise reaches ~22 kHz, no wall.
ff -f lavfi -i "$WHITE" -ac 2 -sample_fmt s16 genuine_44k.flac

# Genuine hi-res 96k/24-bit: white noise truly reaching ~48 kHz.
ff -f lavfi -i "anoisesrc=d=20:color=white:r=96000:amplitude=0.5:seed=2" \
    -ac 2 -sample_fmt s32 genuine_96k.flac

# Fake lossless: encode to lossy then re-wrap as FLAC (leaves a brick wall).
via_mp3 genuine_44k.flac -b:a 128k fake_from_mp3_128.flac
via_mp3 genuine_44k.flac -b:a 320k fake_from_mp3_320.flac

# Music-like genuine + transcodes.
ff -f lavfi -i "$MUSIC" -ac 2 -sample_fmt s16 music_genuine.flac
via_mp3 music_genuine.flac -b:a 192k music_fake_192.flac
via_mp3 music_genuine.flac -b:a 320k music_fake_320.flac

# Quiet genuine + a 128k transcode of it (caught only by the flat-shelf rule).
ff -f lavfi -i "$QUIET" -ac 2 -sample_fmt s16 quiet_genuine.flac
via_mp3 quiet_genuine.flac -b:a 128k quiet_fake_128.flac

# 16-bit master padded into a 24-bit container: still full CD quality.
ff -i genuine_44k.flac -sample_fmt s32 padded_24bit.flac

# Upsampled: 44.1k content resampled up to 96k with a steep filter (hard wall
# at the old 22.05 kHz Nyquist) — the fake-hires signature.
ff -i genuine_44k.flac \
    -af "aresample=96000:resampler=soxr:precision=28" -sample_fmt s32 upsampled_96k.flac

# Worse: a 128k MP3 upsampled to 96k and sold as hi-res.
ff -i fake_from_mp3_128.flac \
    -af "aresample=96000:resampler=soxr:precision=28" -sample_fmt s32 upsampled_from_mp3_96k.flac

# Openly lossy file.
ff -i music_genuine.flac -c:a libmp3lame -b:a 192k lossy_192.mp3

# Clipped: a loud 1 kHz tone driven well past full scale.
ff -f lavfi -i "sine=frequency=1000:duration=15:sample_rate=44100" \
    -af "volume=30dB" -ac 2 -sample_fmt s16 clipped.flac

echo "Done:"
ls -1 *.flac *.mp3
