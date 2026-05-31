"""Orchestrator: a path in, a fully populated AnalysisResult out."""

from __future__ import annotations

from typing import Optional

from . import decode, dsp, verdict
from .model import AnalysisResult, Metrics, Spectrum, Verdict


def analyze_file(
    path: str,
    max_seconds: Optional[float] = 1800.0,
    want_spectrogram: bool = False,
) -> AnalysisResult:
    """Analyze one audio file.

    ``max_seconds`` caps how much audio is decoded (default 30 min) to bound
    memory on pathologically long files. ``want_spectrogram`` adds the heavier
    2-D STFT for the GUI detail view; the CLI leaves it off.
    """
    try:
        info = decode.probe(path)
    except Exception as e:  # noqa: BLE001 - report, don't crash the scan
        res = AnalysisResult(info=_empty_info(path))
        res.error = f"probe failed: {e}"
        return res

    res = AnalysisResult(info=info)
    try:
        raw = decode.decode_int32(info, max_seconds=max_seconds)
        pcm = raw.astype("float64") / 2147483648.0
        mono = dsp.to_mono(pcm)
        sr = info.sample_rate

        m = Metrics()
        m.effective_bits = dsp.effective_bits(raw)
        m.peak_dbfs, m.rms_dbfs, m.crest_db = dsp.peak_rms_crest(mono)
        m.dr = dsp.dynamic_range(pcm, sr)
        m.clipped_samples, m.clip_runs, m.clip_pct = dsp.clipping(pcm, m.effective_bits)

        freqs, power_db = dsp.welch_spectrum(mono, sr)
        ci = dsp.estimate_cutoff(freqs, power_db, sr)
        m.cutoff_hz = ci.content_top_hz
        m.wall_hz = ci.wall_hz
        m.wall_drop_db = ci.wall_drop_db
        m.shelf_db = ci.shelf_db
        m.noise_floor_db = ci.floor_db
        res.metrics = m

        spec = Spectrum(freqs=freqs, power_db=power_db)
        if want_spectrogram:
            spec.spec_times, spec.spec_freqs, spec.spec_db = dsp.spectrogram(mono, sr)
        res.spectrum = spec

        v, conf, reasons, source = verdict.classify(info, m)
        res.verdict, res.confidence = v, conf
        res.reasons, res.suspected_source = reasons, source
    except Exception as e:  # noqa: BLE001
        res.verdict = Verdict.ERROR
        res.error = f"analysis failed: {e}"
    return res


def scan_one(path: str) -> AnalysisResult:
    """Top-level, picklable entry point for ProcessPoolExecutor workers.

    Spectrogram is skipped here; the GUI computes it lazily on selection.
    """
    return analyze_file(path, want_spectrogram=False)


def _empty_info(path: str):
    from .model import StreamInfo

    return StreamInfo(path=path)
