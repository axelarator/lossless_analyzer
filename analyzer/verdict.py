"""Turn raw metrics into an authenticity verdict + confidence.

The decisive signal is the *shape of the band edge*, not the cutoff frequency.
A lossy encoder leaves a near-vertical wall (tens of dB dropped within ~1 kHz)
followed by a flat noise-floor shelf. Genuine audio — even a track that simply
lacks high treble — rolls off gradually. So we judge by how steep the steepest
cliff is and where it sits, and report a confidence rather than a hard verdict.

A sharp wall *right at* Nyquist is NOT evidence of a lossy source: it's the
normal anti-alias filter of a legitimate downsample (e.g. a real 24/96 master
resampled to 24/44). Only a sharp wall well *below* Nyquist indicates lossy.
"""

from __future__ import annotations

from .model import StreamInfo, Metrics, Verdict

LOSSLESS_CODECS = {
    "flac", "alac", "ape", "wavpack", "tta", "tak", "mlp", "truehd",
    "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "pcm_s24be",
}

# Wall steepness (dB dropped across ~1 kHz). Calibrated from real data:
# genuine rolloffs measured 4-16 dB; real MP3/AAC walls measured 80+ dB.
WALL_FAKE = 35.0      # a wall this steep is an artificial lowpass
WALL_SUSPECT = 30.0   # ambiguously steep — surface it but stay unsure
                      # (genuine rolloffs top out ~16 dB; real walls are 50+,
                      #  so the 30-35 band is a deliberately narrow hedge)

# Map a wall frequency to a likely lossy source (Hz ceiling, label).
_SOURCE_TABLE = [
    (16500, "lossy ~128 kbps (MP3/AAC)"),
    (17500, "MP3 ~160 kbps"),
    (18800, "MP3 ~192 kbps"),
    (19700, "MP3 ~256 kbps"),
    (20600, "MP3 ~320 kbps / V0"),
]


def _suspected_source(wall_hz: float) -> str:
    for ceiling, label in _SOURCE_TABLE:
        if wall_hz < ceiling:
            return label
    return "MP3 ~320 kbps / V0"


def classify(info: StreamInfo, m: Metrics) -> tuple[Verdict, float, list[str], str]:
    """Return (verdict, confidence 0..1, reasons, suspected_source)."""
    reasons: list[str] = []
    sr = info.sample_rate
    nyq = info.nyquist or 1.0
    wall = m.wall_hz
    drop = m.wall_drop_db
    rel = wall / nyq if nyq else 1.0
    is_lossless = info.codec in LOSSLESS_CODECS

    edge_note = (
        f"Steepest band edge: ~{drop:.0f} dB drop at {wall/1000:.1f} kHz "
        f"(Nyquist {nyq/1000:g} kHz)."
    )

    # --- Padded bit depth (informational) ---------------------------------- #
    bit_padded = info.claimed_bits >= 24 and 0 < m.effective_bits <= 16
    if bit_padded:
        reasons.append(
            f"Container claims {info.claimed_bits}-bit but only "
            f"{m.effective_bits} bits carry data (padded)."
        )

    # --- Upsampling: hi-res claim with a hard wall near the old Nyquist ----- #
    if sr >= 88200:
        if drop >= WALL_FAKE and wall < 26000:
            conf = min(0.75 + (drop - WALL_FAKE) / 200.0, 0.97)
            reasons.insert(
                0,
                f"Claimed {sr/1000:g} kHz but a hard {drop:.0f} dB wall sits at "
                f"~{wall/1000:.1f} kHz — content was upsampled from "
                f"{'44.1' if wall < 23000 else '48'} kHz.",
            )
            if bit_padded:
                conf = min(conf + 0.05, 0.98)
            return Verdict.UPSAMPLED, conf, reasons, ""
        # No wall: a genuine hi-res file (even one light on ultrasonics).
        reasons.append(edge_note)
        return Verdict.GENUINE, 0.9, reasons, ""

    # --- Lossy transcode hidden in a lossless container -------------------- #
    if is_lossless and sr <= 48000:
        if drop >= WALL_FAKE and rel < 0.97:
            source = _suspected_source(wall)
            reasons.insert(0, edge_note)
            reasons.insert(
                1,
                f"That's an artificial brick-wall lowpass; the band above it is "
                f"a flat shelf near {m.shelf_db:.0f} dBFS — the signature of a "
                f"lossy source.",
            )
            if rel < 0.90:
                # Wall clearly below Nyquist: confidently lossy.
                depth = min((0.90 - rel) / 0.20, 1.0)
                conf = min(0.72 + 0.2 * depth + (drop - WALL_FAKE) / 300.0, 0.97)
                return Verdict.LIKELY_FAKE, conf, reasons, source
            elif rel < 0.95:
                # ~20-21 kHz: 320k transcode vs. a tight downsample filter —
                # genuinely hard to call. Flag, but only as suspect.
                conf = min(0.5 + (drop - WALL_FAKE) / 250.0, 0.7)
                reasons.append(
                    "Edge is near Nyquist, so a legitimate downsample filter "
                    "can't be ruled out — check the spectrogram."
                )
                return Verdict.SUSPECT, conf, reasons, source
            # rel >= 0.95: wall is right at Nyquist => normal anti-alias filter.
            reasons.append(
                "Wall sits at Nyquist — consistent with a clean downsample, "
                "not a lossy transcode."
            )
            return Verdict.GENUINE, 0.85, reasons, ""

        if drop >= WALL_SUSPECT and rel < 0.92:
            reasons.insert(0, edge_note)
            reasons.append(
                "Moderately steep edge below Nyquist — probably a naturally "
                "dull master, but worth a look."
            )
            return Verdict.SUSPECT, 0.4, reasons, _suspected_source(wall)

    # --- Nothing alarming -------------------------------------------------- #
    if bit_padded:
        return Verdict.SUSPECT, 0.45, reasons, ""

    reasons.insert(0, edge_note)
    reasons.append(
        f"Gradual rolloff, no brick-wall cliff — content tapers naturally to "
        f"~{m.cutoff_hz/1000:.1f} kHz. Consistent with genuine lossless."
    )
    return Verdict.GENUINE, 0.9, reasons, ""
