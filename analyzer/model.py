"""Data structures shared across the analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


class Verdict(str, Enum):
    """Top-level authenticity call for a file."""

    GENUINE = "genuine"          # full-band, content consistent with claimed format
    SUSPECT = "suspect"          # something looks off but it's not conclusive
    LIKELY_FAKE = "likely_fake"  # strong lossy-transcode signature
    UPSAMPLED = "upsampled"      # real content well below claimed Nyquist
    LOSSY = "lossy"              # openly lossy file (MP3/AAC/Opus...), not a lossless source
    ERROR = "error"              # could not analyze

    @property
    def label(self) -> str:
        return {
            Verdict.GENUINE: "Likely genuine",
            Verdict.SUSPECT: "Suspect",
            Verdict.LIKELY_FAKE: "Likely fake (lossy transcode)",
            Verdict.UPSAMPLED: "Upsampled",
            Verdict.LOSSY: "Lossy file",
            Verdict.ERROR: "Error",
        }[self]

    @property
    def short(self) -> str:
        """Compact label for narrow table columns."""
        return {
            Verdict.GENUINE: "OK",
            Verdict.SUSPECT: "SUSPECT",
            Verdict.LIKELY_FAKE: "FAKE",
            Verdict.UPSAMPLED: "UPSAMPLED",
            Verdict.LOSSY: "LOSSY",
            Verdict.ERROR: "ERR",
        }[self]

    @property
    def color(self) -> str:
        """Hex color for GUI rows."""
        return {
            Verdict.GENUINE: "#2e7d32",
            Verdict.SUSPECT: "#f9a825",
            Verdict.LIKELY_FAKE: "#c62828",
            Verdict.UPSAMPLED: "#ad1457",
            Verdict.LOSSY: "#1565c0",
            Verdict.ERROR: "#616161",
        }[self]


@dataclass
class StreamInfo:
    """Container/codec metadata as reported by ffprobe (the *claimed* format)."""

    path: str
    codec: str = ""
    container: str = ""
    sample_rate: int = 0
    channels: int = 0
    claimed_bits: int = 0          # bits_per_raw_sample / bits_per_sample
    duration: float = 0.0
    bit_rate: int = 0              # overall, bits/s (0 if unknown)
    title: str = ""
    artist: str = ""
    album: str = ""

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0 if self.sample_rate else 0.0


@dataclass
class Metrics:
    """Numeric measurements computed from the decoded PCM."""

    effective_bits: int = 0        # highest bit actually carrying data
    peak_dbfs: float = 0.0
    rms_dbfs: float = 0.0
    crest_db: float = 0.0
    dr: float = 0.0                # TT/Pleasurize DR value (DR14-style)
    clipped_samples: int = 0
    clip_runs: int = 0             # runs of >=3 consecutive full-scale samples
    clip_pct: float = 0.0
    cutoff_hz: float = 0.0         # estimated top of real spectral content
    wall_hz: float = 0.0           # frequency of the steepest spectral cliff
    wall_drop_db: float = 0.0      # dB drop across that cliff (~1 kHz windows)
    shelf_db: float = -120.0       # spectrum level above the cliff (dB rel. spectrum peak)
    noise_floor_db: float = -120.0   # spectrum floor near Nyquist (dB rel. spectrum peak)
    bit_padded: bool = False       # claims >=24-bit but only <=16 bits carry data


@dataclass
class Spectrum:
    """Average power spectrum and a downsampled spectrogram for display."""

    freqs: np.ndarray = field(default_factory=lambda: np.array([]))      # Hz
    power_db: np.ndarray = field(default_factory=lambda: np.array([]))   # dB, peak=0
    spec_times: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    spec_db: np.ndarray = field(default_factory=lambda: np.array([[]]))  # [freq, time]


@dataclass
class AnalysisResult:
    info: StreamInfo
    metrics: Metrics = field(default_factory=Metrics)
    spectrum: Optional[Spectrum] = None
    verdict: Verdict = Verdict.ERROR
    confidence: float = 0.0        # 0..1, how sure we are about the verdict
    reasons: list[str] = field(default_factory=list)
    suspected_source: str = ""     # e.g. "MP3 ~192 kbps"
    error: str = ""

    def summary_row(self) -> dict:
        """Flat dict for CLI tables / CSV export (no heavy arrays)."""
        i, m = self.info, self.metrics
        return {
            "file": i.path,
            "codec": i.codec,
            "sample_rate": i.sample_rate,
            "claimed_bits": i.claimed_bits,
            "effective_bits": m.effective_bits,
            "cutoff_hz": round(m.cutoff_hz),
            "dr": round(m.dr, 1),
            "peak_dbfs": round(m.peak_dbfs, 2),
            "clip_runs": m.clip_runs,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 2),
            "suspected_source": self.suspected_source,
            "convertible": self.convertible,
        }

    @property
    def convertible(self) -> bool:
        """Is this a good source for a lossy (e.g. 320k MP3) encode?

        True for genuine lossless files and for files whose only problem is
        being *more* than CD quality on paper (upsampled from CD rate, or
        16-bit padded to 24-bit) — their real content is still full CD
        quality. False for anything that looks lossy-sourced or unanalyzed.
        """
        v = self.verdict
        if v == Verdict.GENUINE:
            return True
        if v == Verdict.UPSAMPLED:
            return not self.suspected_source  # set when the pre-upsample source looks lossy
        if v == Verdict.SUSPECT:
            return self.metrics.bit_padded and not self.suspected_source
        return False

    def as_dict(self) -> dict:
        """JSON-friendly dict (drops the spectrum arrays)."""
        d = {
            "info": asdict(self.info),
            "metrics": asdict(self.metrics),
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "suspected_source": self.suspected_source,
            "convertible": self.convertible,
            "error": self.error,
        }
        return d
