"""FLAC -> MP3 conversion for quick A/B listening on the iPod.

Uses ffmpeg's libmp3lame. Defaults to LAME V0 VBR (~245 kbps), copies tags and
embedded cover art, and writes ID3v2.3 (the most player-compatible flavor,
including Rockbox on the iPod Classic).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# Named quality presets -> ffmpeg libmp3lame args.
PRESETS: dict[str, list[str]] = {
    "V0": ["-q:a", "0"],     # ~245 kbps VBR (default)
    "V2": ["-q:a", "2"],     # ~190 kbps VBR
    "320": ["-b:a", "320k"],  # CBR
    "256": ["-b:a", "256k"],
    "192": ["-b:a", "192k"],
}


@dataclass
class ConvertResult:
    ok: bool
    output: str
    message: str = ""


def convert_to_mp3(
    src: str,
    out_dir: str | None = None,
    quality: str = "V0",
    overwrite: bool = False,
) -> ConvertResult:
    """Encode ``src`` to MP3. Returns the output path inside a ConvertResult."""
    if quality not in PRESETS:
        return ConvertResult(False, "", f"unknown quality preset: {quality}")
    if not os.path.isfile(src):
        return ConvertResult(False, "", f"source not found: {src}")

    base = os.path.splitext(os.path.basename(src))[0]
    out_dir = out_dir or os.path.dirname(src)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}.mp3")

    if os.path.exists(out_path) and not overwrite:
        return ConvertResult(True, out_path, "already exists (skipped)")

    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-y" if overwrite else "-n",
        "-i", src,
        "-map", "0:a:0",          # audio
        "-map", "0:v?",           # cover art if present
        "-c:a", "libmp3lame",
        *PRESETS[quality],
        "-c:v", "copy",           # keep embedded art as-is
        "-id3v2_version", "3",
        "-map_metadata", "0",
        "-disposition:v", "attached_pic",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Retry without the video/art stream (some sources confuse the muxer).
        cmd_noart = [
            FFMPEG, "-v", "error", "-nostdin", "-y" if overwrite else "-n",
            "-i", src, "-map", "0:a:0", "-c:a", "libmp3lame",
            *PRESETS[quality], "-id3v2_version", "3", "-map_metadata", "0",
            out_path,
        ]
        proc = subprocess.run(cmd_noart, capture_output=True, text=True)
        if proc.returncode != 0:
            return ConvertResult(False, out_path, proc.stderr.strip()[:300])
    return ConvertResult(True, out_path, f"encoded ({quality})")
