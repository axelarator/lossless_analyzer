# Lossless Analyzer

A desktop spectrum/quality analyzer for a local FLAC library. Point it at a
folder and it tells you, per file, whether the audio is **truly lossless** or a
**lossy transcode re-wrapped as FLAC** ("fake FLAC") or an **upsampled** file —
plus dynamic range, clipping, and effective bit depth. Genuine files can be
converted to MP3 for quick A/B listening on a portable player.

Built for Linux (developed on CachyOS). Native GTK/Qt desktop app, **not** a web app.

## What it measures

| Check | How |
|-------|-----|
| **Fake lossless** (lossy transcode) | Detects a lowpass *brick wall* in the spectrum. MP3/AAC encoders cut everything above a bitrate-dependent ceiling (~16 kHz at 128k, ~20 kHz at 320k). A sharp wall well below the 22.05 kHz Nyquist in a FLAC = transcoded from lossy. Reports a suspected source bitrate. |
| **Upsampling** | A file claiming 96/192 kHz but whose real content stops near 22 kHz was upsampled from CD rate. Compares actual spectral content against the claimed Nyquist. |
| **Effective bit depth** | A "24-bit" file padded from a 16-bit master shows 16 bits of real data. Found by inspecting the low bits of the decoded PCM. |
| **Dynamic range (DR)** | TT/Pleasurize DR14-style metric — useful for spotting brickwall-limited "loudness war" masters. |
| **Clipping** | Counts runs of consecutive full-scale samples (digital clipping). |
| **Spectrum + spectrogram** | Average power spectrum and an STFT spectrogram so you can eyeball edge cases yourself. |

Each verdict comes with a **confidence score**, not just yes/no — 320 kbps MP3
transcodes are genuinely hard to distinguish and get flagged as *suspect* rather
than a false-confident *fake*.

## Requirements

Everything is already on a typical CachyOS install — **no `pip install` needed**:

- `python` (3.11+)
- `python-numpy`, `python-matplotlib`, `python-pyqt6`
- `ffmpeg` (with `libmp3lame`) and `ffprobe`

```sh
sudo pacman -S --needed python-numpy python-matplotlib python-pyqt6 ffmpeg
```

Audio decoding and metadata go through `ffmpeg`/`ffprobe` (so anything ffmpeg
can read works: FLAC, ALAC, Wav, APE, WavPack, MP3, Opus…). No `scipy`,
`soundfile`, or `mutagen` — deliberately, so it runs on bleeding-edge Python
where those may lack wheels.

## Usage

### GUI

```sh
./gui.py
```

1. **Choose Directory** → pick your music folder (Recursive scans subfolders).
2. **Scan** — analysis runs in parallel, off the UI thread.
3. Browse the color-coded table. **Suspects only** hides the genuine files.
4. Click a row to see its spectrogram, average spectrum, metrics, and the
   reasons behind the verdict.
5. **Convert to MP3** (quality dropdown, default V0) for A/B listening.

### CLI (batch)

```sh
# Scan a library, show only flagged files
./cli.py ~/Music -r --suspect-only

# Full report to CSV + JSON, 4 workers
./cli.py ~/Music -r --csv report.csv --json report.json -j4

# Convert every genuine file to V0 MP3 into ~/ipod
./cli.py ~/Music -r --convert-genuine --quality V0 --out ~/ipod
```

MP3 presets: `V0` (~245k VBR, default), `V2` (~190k), `320`, `256`, `192`.
Conversions copy tags and embedded cover art and write ID3v2.3 (Rockbox-friendly).

## How reliable is the fake detection?

- **Low-bitrate transcodes (≤256 kbps)**: very reliable — the lowpass wall is
  obvious and well below 20 kHz.
- **320 kbps / V0 transcodes**: hard. Their cutoff (~20 kHz) is close to where
  some genuine masters naturally roll off, so these surface as *suspect* with
  modest confidence. Use the spectrogram to judge: a transcode shows a dead-flat
  noise floor with a razor-sharp edge; genuine content tapers.
- **Naturally dull / old masters** can look suspicious — confidence stays low
  and the spectrogram tells the real story. This is why the tool reports
  evidence, not just a verdict.

## FLAC vs MP3 on the iPod Classic 7G (Rockbox + Meze Alba)

Short answer: **keep FLAC as your archive; for the iPod, MP3 V0 is the smarter
choice — but for battery life, not sound quality.**

- **Sound quality**: Through the Meze Alba (a single-dynamic-driver earbud) you
  will not hear a difference between FLAC and a well-encoded 320/V0 MP3. LAME V0
  is transparent for essentially all listeners and material. The Alba's
  resolution and the iPod's output stage are nowhere near the limiting factor.
- **Battery life**: This is the real trade-off. The iPod Classic 6/7G uses a
  low-power ARM (Samsung S5L8702). Decoding FLAC costs noticeably more CPU than
  MP3, which shortens battery runtime on Rockbox. MP3 lets the CPU idle more.
- **Storage**: You said it's a non-issue (flash mod), so that doesn't push the
  decision either way.

So: the spectrum analyzer's job is to make sure your *archive* is genuinely
lossless. Once a file passes, transcode a V0 copy for the iPod and enjoy the
longer battery life with no audible penalty on the Alba.

## Project layout

```
analyzer/
  decode.py    ffprobe metadata + ffmpeg PCM decode -> numpy
  dsp.py       spectrum, spectrogram, DR, clipping, bit depth, cutoff estimate
  verdict.py   metrics -> verdict + confidence + suspected source
  analyze.py   orchestrator: path -> AnalysisResult
  convert.py   FLAC/etc -> MP3 (libmp3lame)
  model.py     dataclasses (StreamInfo, Metrics, Spectrum, AnalysisResult, Verdict)
cli.py         batch terminal scanner
gui.py         PyQt6 desktop app
testdata/      make_fixtures.sh -> synthetic genuine/fake/upsampled fixtures
```

## Test fixtures

The `testdata/` FLACs aren't committed (they're generated binaries). Recreate
them with ffmpeg:

```sh
./testdata/make_fixtures.sh
./cli.py testdata --no-color     # sanity-check the detector
```

This produces genuine, fake-transcode, upsampled, and clipped examples so you
can confirm the verdicts behave as expected.
