"""Lossless audio analyzer.

A spectrum / quality analyzer for a local FLAC library. Decodes audio via
ffmpeg, computes spectral and loudness metrics with numpy, and flags files
that are likely lossy transcodes ("fake FLAC") or upsampled.

Everything here depends only on numpy plus the system ffmpeg/ffprobe binaries,
so it runs without any pip installs on a stock CachyOS Python.
"""

from .analyze import analyze_file, scan_one
from .model import AnalysisResult, Verdict

__all__ = ["analyze_file", "scan_one", "AnalysisResult", "Verdict"]
__version__ = "0.1.0"
