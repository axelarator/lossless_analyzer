"""ffprobe metadata + ffmpeg PCM decode, wrapped for numpy.

We deliberately avoid soundfile/scipy so the tool runs on a bleeding-edge
Python with no pip wheels. ffmpeg decodes to interleaved signed 32-bit PCM;
ffmpeg left-justifies smaller bit depths into the 32-bit word, which is what
lets us recover the *effective* bit depth later.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

import numpy as np

from .model import StreamInfo

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# Audio extensions worth scanning. We can analyze lossy too (useful to confirm
# a file really is what its extension claims).
AUDIO_EXTS = {
    ".flac", ".wav", ".aiff", ".aif", ".alac", ".m4a", ".ape", ".wv",
    ".mp3", ".ogg", ".opus", ".aac", ".wma", ".dsf", ".dff",
}


class DecodeError(RuntimeError):
    pass


def have_tools() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def probe(path: str) -> StreamInfo:
    """Read container/codec metadata for the first audio stream."""
    cmd = [
        FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a:0", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise DecodeError(f"ffprobe failed: {e.stderr.strip()[:200]}") from e
    data = json.loads(out or "{}")

    streams = data.get("streams", [])
    if not streams:
        raise DecodeError("no audio stream found")
    s = streams[0]
    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in {**fmt.get("tags", {}), **s.get("tags", {})}.items()}

    # bits_per_raw_sample is the meaningful one for FLAC; fall back to
    # bits_per_sample (set for PCM/WAV).
    claimed_bits = _to_int(s.get("bits_per_raw_sample")) or _to_int(s.get("bits_per_sample"))

    return StreamInfo(
        path=path,
        codec=s.get("codec_name", ""),
        container=(fmt.get("format_name", "") or "").split(",")[0],
        sample_rate=_to_int(s.get("sample_rate")),
        channels=_to_int(s.get("channels")),
        claimed_bits=claimed_bits,
        duration=float(fmt.get("duration") or s.get("duration") or 0.0),
        bit_rate=_to_int(fmt.get("bit_rate") or s.get("bit_rate")),
        title=tags.get("title", ""),
        artist=tags.get("artist", ""),
        album=tags.get("album", ""),
    )


def decode_pcm(
    info: StreamInfo, max_seconds: Optional[float] = None
) -> np.ndarray:
    """Decode to a float64 array shaped (frames, channels), range ~[-1, 1].

    Returns the raw signed-32-bit data scaled by 2**31 so that bit-depth
    inspection is still possible by the caller via ``raw_int32``.
    """
    raw = decode_int32(info, max_seconds)
    return raw.astype(np.float64) / 2147483648.0


def decode_int32(
    info: StreamInfo, max_seconds: Optional[float] = None
) -> np.ndarray:
    """Decode to an int32 array shaped (frames, channels), left-justified PCM."""
    if info.channels <= 0:
        raise DecodeError("unknown channel count")

    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += [
        "-i", info.path,
        "-map", "0:a:0",
        "-c:a", "pcm_s32le",
        "-f", "s32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:200]}")
    if not proc.stdout:
        raise DecodeError("ffmpeg produced no audio data")

    data = np.frombuffer(proc.stdout, dtype="<i4")
    ch = info.channels
    usable = (data.size // ch) * ch
    if usable == 0:
        raise DecodeError("decoded audio was empty")
    return data[:usable].reshape(-1, ch)
