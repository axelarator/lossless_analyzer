"""Signal-processing primitives: spectrum, spectrogram, loudness, clipping.

Pure numpy. All functions take PCM as float (channels-last) and the sample
rate, and return plain numbers / arrays.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

EPS = 1e-12


def to_mono(pcm: np.ndarray) -> np.ndarray:
    if pcm.ndim == 1:
        return pcm
    return pcm.mean(axis=1)


# --------------------------------------------------------------------------- #
# Bit depth
# --------------------------------------------------------------------------- #
def effective_bits(raw_int32: np.ndarray) -> int:
    """Highest bit position that actually carries data.

    ffmpeg left-justifies smaller depths into the 32-bit word, so a true
    16-bit source padded to 24/32 bit has 16 trailing zero bits. We OR every
    sample together and count trailing zeros.
    """
    flat = raw_int32.reshape(-1)
    if flat.size == 0:
        return 0
    # Work in uint32 space; trailing zeros are unaffected by the sign bit.
    orv = int(np.bitwise_or.reduce(flat.astype(np.uint32)))
    if orv == 0:
        return 0  # pure silence
    trailing = (orv & -orv).bit_length() - 1
    return 32 - trailing


# --------------------------------------------------------------------------- #
# Loudness / dynamics
# --------------------------------------------------------------------------- #
def peak_rms_crest(pcm: np.ndarray) -> tuple[float, float, float]:
    """Sample peak and RMS in dBFS, taken from the loudest channel.

    Measured per channel rather than on a mono downmix, which would
    understate the peak of wide or out-of-phase material.
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.size:
        peak = float(np.max(np.abs(pcm)))
        rms = float(np.max(np.sqrt(np.mean(pcm**2, axis=0))))
    else:
        peak = rms = 0.0
    peak_db = 20.0 * np.log10(max(peak, EPS))
    rms_db = 20.0 * np.log10(max(rms, EPS))
    return peak_db, rms_db, peak_db - rms_db


def dynamic_range(pcm: np.ndarray, sr: int) -> float:
    """TT/Pleasurize DR value (the DR14 meter), averaged across channels.

    Per channel: 3 s blocks -> block RMS (with the 2x convention) and block
    peak -> quadratic mean of the loudest 20% of block RMS values ->
    20*log10(2nd-highest *block* peak / that RMS). Using the 2nd-highest block
    peak (not sample) is what makes the meter ignore a single stray transient.
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    block = max(int(round(3.0 * sr)), 1)
    n_blocks = pcm.shape[0] // block
    if n_blocks < 1:
        # Too short for the windowed method; fall back to crest factor.
        _, _, crest = peak_rms_crest(pcm)
        return max(crest, 0.0)

    drs = []
    for c in range(pcm.shape[1]):
        x = pcm[: n_blocks * block, c].reshape(n_blocks, block)
        # 2x convention from the spec.
        block_rms = np.sqrt(2.0 * np.mean(x**2, axis=1))
        n_top = max(int(round(0.2 * n_blocks)), 1)
        top = np.sort(block_rms)[::-1][:n_top]
        rms_avg = np.sqrt(np.mean(top**2))
        block_peak = np.sort(np.abs(x).max(axis=1))
        peak2 = block_peak[-2] if block_peak.size >= 2 else block_peak[-1]
        dr = 20.0 * np.log10(max(peak2, EPS) / max(rms_avg, EPS))
        drs.append(dr)
    return float(np.clip(np.mean(drs), 0.0, 30.0))


def clipping(pcm: np.ndarray, eff_bits: int, run_len: int = 3) -> tuple[int, int, float]:
    """Detect digital clipping: runs of >= ``run_len`` full-scale samples.

    Returns (clipped_sample_count, run_count, percent_of_samples).
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    # Full scale relative to the *effective* depth (a 16-bit max in a 32-bit
    # word is ~0.99997, not 1.0).
    eff_bits = max(eff_bits, 1)
    fs = (2 ** (eff_bits - 1) - 1) / (2 ** (eff_bits - 1))
    thresh = fs * 0.99995

    total_clipped = 0
    total_runs = 0
    for c in range(pcm.shape[1]):
        hot = np.abs(pcm[:, c]) >= thresh
        if not hot.any():
            continue
        # Find runs of consecutive True values.
        idx = np.flatnonzero(np.diff(np.concatenate(([0], hot.view(np.int8), [0]))))
        starts, ends = idx[0::2], idx[1::2]
        lengths = ends - starts
        long_runs = lengths[lengths >= run_len]
        total_runs += int(long_runs.size)
        total_clipped += int(long_runs.sum())
    pct = 100.0 * total_clipped / max(pcm.size, 1)
    return total_clipped, total_runs, pct


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #
def welch_spectrum(mono: np.ndarray, sr: int, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectrum (Welch, Hann window, 50% overlap), dB peak=0."""
    n = mono.size
    if n < nfft:
        nfft = 1 << max(int(np.log2(max(n, 2))), 8)  # at least 256
        nfft = min(nfft, max(n, 256))
    win = np.hanning(nfft)
    hop = nfft // 2
    starts = range(0, max(n - nfft, 0) + 1, hop)
    acc = np.zeros(nfft // 2 + 1)
    count = 0
    for s in starts:
        seg = mono[s : s + nfft]
        if seg.size < nfft:
            break
        spec = np.fft.rfft(seg * win)
        acc += np.abs(spec) ** 2
        count += 1
    if count == 0:  # very short file
        seg = np.zeros(nfft)
        seg[: mono.size] = mono
        acc = np.abs(np.fft.rfft(seg * win)) ** 2
        count = 1
    psd = acc / count
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    power_db = 10.0 * np.log10(psd + EPS)
    power_db -= power_db.max()  # normalize peak to 0 dB
    return freqs, power_db


def spectrogram(mono: np.ndarray, sr: int, nfft: int = 4096,
                max_cols: int = 1200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT magnitude in dB, returned as (times, freqs, db[freq, time])."""
    n = mono.size
    if n < nfft:
        pad = np.zeros(nfft)
        pad[:n] = mono
        mono = pad
        n = nfft
    # Choose hop so we land near max_cols columns for a manageable image.
    n_cols_full = (n - nfft) // (nfft // 4) + 1
    hop = nfft // 4
    if n_cols_full > max_cols:
        hop = max((n - nfft) // (max_cols - 1), 1)
    win = np.hanning(nfft)
    starts = np.arange(0, n - nfft + 1, hop)
    cols = []
    for s in starts:
        seg = mono[s : s + nfft] * win
        cols.append(np.abs(np.fft.rfft(seg)))
    mag = np.array(cols).T if cols else np.abs(np.fft.rfft(mono[:nfft] * win))[:, None]
    db = 20.0 * np.log10(mag + EPS)
    db -= db.max()
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    times = starts / sr
    return times, freqs, db


class CutoffInfo(NamedTuple):
    content_top_hz: float   # highest freq with meaningful energy (display)
    wall_hz: float          # frequency of the steepest cliff found
    wall_drop_db: float     # dB the spectrum drops across the cliff (~1.5 kHz)
    shelf_db: float         # spectrum level just above the cliff (dB, peak=0)
    floor_db: float         # spectrum floor near Nyquist (dB, peak=0)


def estimate_cutoff(freqs: np.ndarray, power_db: np.ndarray, sr: int) -> CutoffInfo:
    """Locate a lossy lowpass *cliff* and characterize the band edge.

    A lossy encoder leaves a near-vertical wall: the spectrum drops tens of dB
    within ~1 kHz and then sits on a flat, low noise-floor shelf. Genuine
    content (even content that simply lacks high treble) rolls off *gradually*
    and keeps declining. So instead of judging by cutoff frequency alone, we
    find the steepest downward step in the upper band and measure how steep it
    is and what sits above it. The caller decides what the numbers mean.
    """
    nyq = sr / 2.0
    if freqs.size < 16:
        return CutoffInfo(nyq, nyq, 0.0, float(power_db.min()), float(power_db.min()))

    bin_hz = freqs[1] - freqs[0]

    # Smooth over ~150 Hz with edge padding (avoids a zero-pad lifting the top
    # bins and hiding a real brick-wall floor).
    win = max(int(round(150.0 / bin_hz)) | 1, 3)
    kernel = np.ones(win) / win
    pad = win // 2
    sm = np.convolve(np.pad(power_db, pad, mode="edge"), kernel, mode="valid")[: power_db.size]

    # Floor near Nyquist (top 4% of the band).
    hi_start = int(0.96 * sm.size)
    floor = float(np.median(sm[hi_start:])) if hi_start < sm.size else float(sm.min())

    # Content top: highest freq clearing the floor by a margin (display only).
    above = sm > floor + 6.0
    content_top = float(freqs[int(np.flatnonzero(above)[-1])]) if above.any() else nyq

    # --- Find the steepest cliff in the upper band ------------------------- #
    # Compare the median level in a ~1 kHz window just below each candidate
    # frequency against the median just above it; the largest such step is the
    # wall. Restrict the search to 11 kHz .. 0.995*Nyquist. The window above
    # each candidate needs room, so in practice the search stops ~1 kHz short
    # of Nyquist (~21.05 kHz at 44.1k, ~0.95*Nyquist): an anti-alias filter at
    # 21.5-22 kHz registers *at* that ceiling, which verdict.py treats as
    # "wall at Nyquist" rather than a lossy lowpass.
    w = max(int(round(1000.0 / bin_hz)), 2)
    lo_i = int(np.searchsorted(freqs, 11000.0))
    hi_i = int(np.searchsorted(freqs, 0.995 * nyq))
    lo_i = max(lo_i, w)
    hi_i = min(hi_i, sm.size - w - 1)

    wall_hz = content_top
    wall_drop = 0.0
    shelf = floor
    if hi_i > lo_i:
        below = np.array([np.median(sm[i - w:i]) for i in range(lo_i, hi_i)])
        above_lvl = np.array([np.median(sm[i:i + w]) for i in range(lo_i, hi_i)])
        step = below - above_lvl
        k = int(np.argmax(step))
        wall_idx = lo_i + k
        wall_hz = float(freqs[wall_idx])
        wall_drop = float(step[k])
        # Shelf: median level from the wall up to Nyquist (what sits above it).
        shelf = float(np.median(sm[min(wall_idx + w, sm.size - 1):]))

    return CutoffInfo(content_top, wall_hz, wall_drop, shelf, floor)
