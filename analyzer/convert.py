"""FLAC -> MP3 conversion for the iPod.

Uses ffmpeg's libmp3lame. Defaults to 320 kbps CBR, copies tags and embedded
cover art, and writes ID3v2.3 (the most player-compatible flavor, including
Rockbox on the iPod Classic). Hi-res sources are resampled to 44.1 kHz (MP3
tops out at 48 kHz, and Rockbox's mixer runs at 44.1 kHz anyway); multichannel
sources are downmixed to stereo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from .decode import probe

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# LAME's own recommended algorithm quality ("-q 2", aka "-h"). -q 0 is slower
# and not reliably better.
_LAME_Q = ["-compression_level", "2"]

# Named quality presets -> ffmpeg libmp3lame args.
PRESETS: dict[str, list[str]] = {
    "320": ["-b:a", "320k", *_LAME_Q],  # CBR (default)
    "V0": ["-q:a", "0", *_LAME_Q],      # ~245 kbps VBR
    "V2": ["-q:a", "2", *_LAME_Q],      # ~190 kbps VBR
    "256": ["-b:a", "256k", *_LAME_Q],
    "192": ["-b:a", "192k", *_LAME_Q],
}
DEFAULT_QUALITY = "320"


@dataclass
class ConvertResult:
    ok: bool
    output: str
    message: str = ""


def output_path(src: str, out_dir: str | None = None, rel_to: str | None = None) -> str:
    """Where the MP3 for ``src`` goes.

    With ``rel_to`` (the scanned library root), the source's directory layout
    under that root is mirrored inside ``out_dir`` so identically named tracks
    from different albums ("01 Intro.flac") don't collide.
    """
    base = os.path.splitext(os.path.basename(src))[0] + ".mp3"
    src_dir = os.path.dirname(os.path.abspath(src))
    if not out_dir:
        return os.path.join(src_dir, base)
    if rel_to:
        rel = os.path.relpath(src_dir, os.path.abspath(rel_to))
        if rel != os.curdir and not rel.startswith(os.pardir):
            return os.path.join(out_dir, rel, base)
    return os.path.join(out_dir, base)


def _format_args(src: str) -> list[str]:
    """Output sample-rate / channel args based on the source stream."""
    try:
        info = probe(src)
    except Exception:  # noqa: BLE001 - let ffmpeg pick if probing fails
        return []
    args: list[str] = []
    if info.sample_rate > 48000:
        args += ["-ar", "44100"]
    if info.channels > 2:
        args += ["-ac", "2"]
    return args


def convert_to_mp3(
    src: str,
    out_dir: str | None = None,
    quality: str = DEFAULT_QUALITY,
    overwrite: bool = False,
    rel_to: str | None = None,
) -> ConvertResult:
    """Encode ``src`` to MP3. Returns the output path inside a ConvertResult."""
    if quality not in PRESETS:
        return ConvertResult(False, "", f"unknown quality preset: {quality}")
    if not os.path.isfile(src):
        return ConvertResult(False, "", f"source not found: {src}")

    out_path = output_path(src, out_dir, rel_to)
    if os.path.exists(out_path) and not overwrite:
        return ConvertResult(True, out_path, "already exists (skipped)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Encode to a temp name and rename on success, so an interrupted or failed
    # encode never leaves a truncated .mp3 that later runs would skip.
    tmp_path = out_path + ".part"
    fmt = _format_args(src)
    audio = ["-c:a", "libmp3lame", *PRESETS[quality], *fmt,
             "-id3v2_version", "3", "-map_metadata", "0"]
    head = [FFMPEG, "-v", "error", "-nostdin", "-y", "-i", src]

    cmd = [
        *head,
        "-map", "0:a:0",          # audio
        "-map", "0:v?",           # cover art if present
        *audio,
        "-c:v", "copy",           # keep embedded art as-is
        "-disposition:v", "attached_pic",
        "-f", "mp3", tmp_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Retry without the video/art stream (some sources confuse the muxer).
        cmd_noart = [*head, "-map", "0:a:0", *audio, "-f", "mp3", tmp_path]
        proc = subprocess.run(cmd_noart, capture_output=True, text=True)
        if proc.returncode != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return ConvertResult(False, out_path, proc.stderr.strip()[:300])
    os.replace(tmp_path, out_path)
    return ConvertResult(True, out_path, f"encoded ({quality})")
