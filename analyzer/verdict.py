"""Turn raw metrics into an authenticity verdict + confidence.

The decisive signal is the *shape of the band edge*, not the cutoff frequency.
A lossy encoder leaves a near-vertical wall followed by a dead-flat shelf
sitting exactly on the noise floor. Genuine audio — even a track that simply
lacks high treble — rolls off gradually. So we judge by how steep the steepest
cliff is, where it sits, and how flat the band above it is, and report a
confidence rather than a hard verdict.

A sharp wall *right at* Nyquist is NOT evidence of a lossy source: it's the
normal anti-alias filter of the ADC or of a legitimate downsample. Only a sharp
wall well *below* Nyquist indicates lossy.

Blind spot: an encoder that applies no lowpass leaves no wall. Notably LAME
3.100 -V0 (as driven by ffmpeg) is full-band, so V0-sourced fakes are not
detectable by this method.
"""

from __future__ import annotations

from .model import StreamInfo, Metrics, Verdict

LOSSLESS_CODECS = {
    "flac", "alac", "ape", "wavpack", "tta", "tak", "mlp", "truehd",
    "dsd_lsbf", "dsd_msbf", "dsd_lsbf_planar", "dsd_msbf_planar",
}


def _is_lossless(codec: str) -> bool:
    return codec in LOSSLESS_CODECS or codec.startswith("pcm_")


# Wall steepness (dB dropped between ~1 kHz windows either side of the edge).
# Calibrated on real 44.1k masters and on the same tracks pushed through
# LAME 3.100 at 112-320 kbps and re-wrapped as FLAC:
#   * genuine walls near Nyquist (anti-alias filters): up to ~50 dB — which is
#     why position matters as much as depth;
#   * genuine edges below 0.92*Nyquist: <= ~17 dB, with a shelf that is not
#     flat (it keeps falling or sits several dB above the floor);
#   * lossy walls: 17-60 dB depending on how much treble the music has, always
#     with a shelf within ~0.3 dB of the floor.
WALL_FAKE = 35.0      # this steep below Nyquist = artificial lowpass
WALL_SUSPECT = 30.0   # ambiguously steep — surface it but stay unsure
WALL_SOFT = 15.0      # moderate edge; only suspicious with a flat shelf
FLAT_SHELF_DB = 1.0   # |shelf - floor| below this = dead-flat band above the edge

# Measured wall positions (steepest-step frequency, slightly below LAME's
# nominal lowpass) -> likely source. AAC and other encoders use different
# lowpasses, so treat the label as a rough guide.
_SOURCE_TABLE = [
    (15800, "lossy <=112 kbps"),
    (17050, "MP3 ~128 kbps"),
    (17850, "MP3 ~160 kbps"),
    (19050, "MP3 ~192 kbps / V2"),
    (19600, "MP3 ~224-256 kbps"),
]


def _suspected_source(wall_hz: float) -> str:
    for ceiling, label in _SOURCE_TABLE:
        if wall_hz < ceiling:
            return label
    return "MP3 ~320 kbps"


def classify(info: StreamInfo, m: Metrics) -> tuple[Verdict, float, list[str], str]:
    """Return (verdict, confidence 0..1, reasons, suspected_source).

    Also sets ``m.bit_padded``.
    """
    reasons: list[str] = []
    sr = info.sample_rate
    nyq = info.nyquist or 1.0
    wall = m.wall_hz
    drop = m.wall_drop_db
    rel = wall / nyq
    flat_shelf = abs(m.shelf_db - m.noise_floor_db) <= FLAT_SHELF_DB

    edge_note = (
        f"Steepest band edge: ~{drop:.0f} dB drop at {wall/1000:.1f} kHz "
        f"(Nyquist {nyq/1000:g} kHz)."
    )

    # --- Openly lossy file: nothing to authenticate ------------------------ #
    if not _is_lossless(info.codec):
        reasons.append(
            f"Codec is {info.codec or 'unknown'} — a lossy format, not a lossless "
            f"source. Re-encoding it to MP3 would only lose more."
        )
        reasons.append(edge_note)
        return Verdict.LOSSY, 1.0, reasons, ""

    # --- Padded bit depth (informational) ---------------------------------- #
    m.bit_padded = info.claimed_bits >= 24 and 0 < m.effective_bits <= 16
    if m.bit_padded:
        reasons.append(
            f"Container claims {info.claimed_bits}-bit but only "
            f"{m.effective_bits} bits carry data (padded from a 16-bit master). "
            f"Still full CD quality."
        )

    # --- Upsampling: hi-res claim with a hard wall near the old Nyquist ----- #
    if sr >= 88200:
        if drop >= WALL_FAKE and wall < 26000:
            conf = min(0.75 + (drop - WALL_FAKE) / 200.0, 0.97)
            if m.bit_padded:
                conf = min(conf + 0.05, 0.98)
            reasons.insert(
                0,
                f"Claimed {sr/1000:g} kHz but a hard {drop:.0f} dB wall sits at "
                f"~{wall/1000:.1f} kHz — content was upsampled from "
                f"{'44.1' if wall < 23000 else '48'} kHz.",
            )
            source = ""
            if wall < 20500:
                # A CD-rate source would reach ~20.5-22 kHz; lower than that
                # means the pre-upsample source was itself lossy.
                source = _suspected_source(wall)
                reasons.append(
                    f"The wall is too low for a CD-rate source — the material "
                    f"was likely lossy ({source}) before being upsampled."
                )
            else:
                reasons.append("Real content is CD quality — fine to encode from.")
            return Verdict.UPSAMPLED, conf, reasons, source
        # No wall: a genuine hi-res file (even one light on ultrasonics).
        reasons.append(edge_note)
        return Verdict.GENUINE, 0.9, reasons, ""

    # --- Lossy transcode hidden in a lossless container -------------------- #
    if sr <= 48000:
        if drop >= WALL_FAKE and rel < 0.97:
            source = _suspected_source(wall)
            reasons.insert(0, edge_note)
            reasons.insert(
                1,
                f"That's an artificial brick-wall lowpass; the band above it is "
                f"a flat shelf near {m.shelf_db:.0f} dB (rel. peak) — the "
                f"signature of a lossy source.",
            )
            if rel < 0.90:
                # Wall clearly below Nyquist: confidently lossy.
                depth = min((0.90 - rel) / 0.20, 1.0)
                conf = min(0.72 + 0.2 * depth + (drop - WALL_FAKE) / 300.0, 0.97)
                return Verdict.LIKELY_FAKE, conf, reasons, source
            elif rel < 0.95:
                # ~20-21 kHz: 320k transcode vs. a tight anti-alias filter —
                # genuinely hard to call. Flag, but only as suspect.
                conf = min(0.5 + (drop - WALL_FAKE) / 250.0, 0.7)
                reasons.append(
                    "Edge is near Nyquist, so a legitimate anti-alias filter "
                    "can't be ruled out — check the spectrogram."
                )
                return Verdict.SUSPECT, conf, reasons, source
            # rel >= 0.95: wall is at Nyquist => normal anti-alias filter. The
            # cliff search stops 1 kHz short of Nyquist (dsp.estimate_cutoff),
            # so filters at 21.5-22 kHz register at that ~0.95 ceiling.
            reasons.append(
                "Wall sits at Nyquist — consistent with a normal anti-alias "
                "filter, not a lossy transcode."
            )
            return Verdict.GENUINE, 0.85, reasons, ""

        if drop >= WALL_SUSPECT and rel < 0.92:
            reasons.insert(0, edge_note)
            reasons.append(
                "Moderately steep edge below Nyquist — probably a naturally "
                "dull master, but worth a look."
            )
            return Verdict.SUSPECT, 0.5 if flat_shelf else 0.4, reasons, _suspected_source(wall)

        if drop >= WALL_SOFT and rel < 0.92 and flat_shelf:
            # Lossy walls on quiet or treble-light material are shallow because
            # there's little HF energy above the floor to cut — but the band
            # above them is still perfectly flat.
            conf = min(0.3 + (drop - WALL_SOFT) / 50.0, 0.5)
            reasons.insert(0, edge_note)
            reasons.append(
                "The band above that edge is dead flat on the noise floor — how a "
                "lossy transcode looks on quiet or treble-light material. Could "
                "also be a dull genuine master; check the spectrogram."
            )
            return Verdict.SUSPECT, conf, reasons, _suspected_source(wall)

    # --- Nothing alarming -------------------------------------------------- #
    if m.bit_padded:
        return Verdict.SUSPECT, 0.45, reasons, ""

    reasons.insert(0, edge_note)
    reasons.append(
        f"No lossy brick wall — content tapers naturally to "
        f"~{m.cutoff_hz/1000:.1f} kHz. Consistent with genuine lossless."
    )
    return Verdict.GENUINE, 0.9, reasons, ""
