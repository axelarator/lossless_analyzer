# Lossless Analyzer

A spectrum/quality analyzer for a local FLAC library, and a converter for
turning the files that pass into **320 kbps MP3** for a portable player.

Point it at a folder and it tells you, per file, whether the audio is **truly
lossless**, a **lossy transcode re-wrapped as FLAC** ("fake FLAC"), or
**upsampled** — plus dynamic range, clipping, and effective bit depth. Files
that are good sources are then encoded to MP3 with tags and cover art intact,
mirroring your library's folder layout.

Desktop Qt app plus a batch CLI. Runs on Linux (developed on CachyOS) and macOS.

## What it measures

| Check | How |
|-------|-----|
| **Fake lossless** (lossy transcode) | Looks for a lowpass *brick wall* below Nyquist with a dead-flat noise floor above it. LAME's lowpass depends on bitrate (~17 kHz at 128k, ~18.6 kHz at 192k, ~20.5 kHz at 320k; older/FhG encoders cut ~16 kHz at 128k). Reports a suspected source bitrate. |
| **Upsampling** | A file claiming 88.2–192 kHz whose content stops at a hard wall near 22–24 kHz was upsampled from CD/48k rate. If that wall is even lower, the pre-upsample source was lossy too. |
| **Effective bit depth** | A "24-bit" file padded from a 16-bit master shows 16 bits of real data. Found by inspecting the low bits of the decoded PCM. |
| **Dynamic range (DR)** | TT/Pleasurize DR meter (3 s blocks, 2nd-highest block peak vs. RMS of the loudest 20%) — spots brickwall-limited "loudness war" masters. |
| **Clipping** | Counts runs of ≥3 consecutive full-scale samples. |
| **Spectrum + spectrogram** | Average power spectrum and an STFT spectrogram so you can eyeball edge cases yourself. |

Each verdict comes with a **confidence score**, not just yes/no.

### Verdicts and what gets converted

| Verdict | Meaning | `--convert` |
|---------|---------|-------------|
| Likely genuine | No lossy wall | **converted** |
| Upsampled (from CD rate) | Hi-res label, CD-quality content | **converted** — still a full-quality source |
| Suspect, bit-padded only | 24-bit label, 16-bit content | **converted** — still full CD quality |
| Suspect (band edge) | Looks like a transcode but not conclusive | listed for manual review |
| Upsampled (from lossy) | Wall too low for a CD source | listed for manual review |
| Likely fake | Clear lossy brick wall | skipped |
| Lossy file | It's an MP3/AAC/Opus… already | skipped |

## Requirements

- Python 3.11+ with `numpy`, `matplotlib`, `PyQt6` (the last two only for the GUI)
- `ffmpeg` (with `libmp3lame`) and `ffprobe`

No `scipy`, `soundfile`, or `mutagen` — all decoding and metadata go through
ffmpeg, so anything it reads works (FLAC, ALAC, WAV, AIFF, APE, WavPack, DSD…).

**Linux (Arch/CachyOS)** — everything is packaged, no pip needed:

```sh
sudo pacman -S --needed python-numpy python-matplotlib python-pyqt6 ffmpeg
```

**macOS** — ffmpeg from Homebrew, Python packages in a venv:

```sh
brew install ffmpeg
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python gui.py        # or: source .venv/bin/activate
```

## Usage

### GUI

```sh
./gui.py
```

1. **Choose Directory** → pick your music folder (Recursive scans subfolders).
2. **Scan** — analysis runs in parallel, off the UI thread. The button becomes **Stop**.
3. Browse the color-coded table (click headers to sort). **Suspects only** hides genuine files.
4. Click a row to see its spectrogram, average spectrum, metrics, and the
   reasons behind the verdict.
5. **Convert to MP3** (quality dropdown, default 320). The output keeps the file's
   folder path relative to the scanned directory. You get a warning first if the
   file isn't a clean source.

### CLI (batch)

```sh
# Scan a library, show only flagged files
./cli.py ~/Music -r --suspect-only

# Full report to CSV + JSON, 4 workers
./cli.py ~/Music -r --csv report.csv --json report.json -j4

# Convert every good source to 320 kbps MP3 into ~/ipod
./cli.py ~/Music -r --convert --out ~/ipod
```

`--convert` encodes the convertible files (table above), mirrors the folder
layout under `--out` (`~/Music/Artist/Album/01.flac` → `~/ipod/Artist/Album/01.mp3`,
so same-named tracks from different albums don't collide), then prints the files
to review by hand and the files it skipped. Existing MP3s are left alone, so
re-running only encodes new files.

MP3 presets (`--quality`): `320` (CBR, default), `V0` (~245k VBR), `V2` (~190k),
`256`, `192`. All use LAME's recommended `-q 2` algorithm quality. Output is
ID3v2.3 with tags and embedded cover art (Rockbox-friendly). Hi-res sources are
resampled to 44.1 kHz (MP3 tops out at 48 kHz) and multichannel is downmixed to
stereo.

**320 CBR vs V0:** both are transparent to practically everyone. CBR spends
320 kbps on every frame; V0 varies the bitrate per frame and averages ~220–260
kbps, so files are ~25% smaller at the same perceived quality. With storage
not an issue, 320 is the "maximum, no thinking" choice.

## How reliable is the fake detection?

Measured on real 44.1k masters passed through LAME 3.100 and re-wrapped as FLAC:

- **Bright/loud material, ≤256 kbps:** reliably flagged as *likely fake*, with
  the right bitrate family.
- **320 kbps:** its wall (~20 kHz) sits where tight anti-alias filters of
  genuine masters sit, so it's usually *suspect* rather than *fake*.
- **Quiet or treble-light material:** there's little treble above the noise
  floor for the encoder to cut, so the wall is shallow (15–25 dB). These are
  caught by the flat-shelf check and surface as *suspect* with modest confidence.
  A few very dull tracks can slip through entirely.
- **V0 transcodes made with current LAME are undetectable here:** LAME 3.100
  `-V0` applies no lowpass, so there is no wall to find. (Older LAME and many
  other encoders do cut at ~19.5 kHz and are detected.)
- **Genuine anti-alias filters** at 21–22 kHz can be very steep (up to 50 dB);
  they're recognized by position and don't count against the file.

The tool reports evidence, not just a verdict — when in doubt, look at the
spectrogram: a transcode shows a razor-sharp edge that stays at the same
frequency for the whole track, with nothing above it.

## FLAC vs MP3 on the iPod Classic (Rockbox)

- **Keep FLAC as your archive.** This analyzer's job is to make sure that
  archive is genuinely lossless; the MP3s are disposable copies you can
  regenerate at any time.
- **Sound quality:** a well-encoded 320k or V0 MP3 is transparent for
  essentially all listeners and material — earbuds and the iPod's output stage
  are not the limiting factor.
- **Battery life:** a minor factor either way. FLAC is actually one of the
  cheapest codecs for Rockbox to decode; the extra cost is reading ~3–4× more
  data from storage, which matters little on a flash mod.
- **Why MP3 then:** smaller files and universal compatibility.

## Project layout

```
analyzer/
  decode.py    ffprobe metadata + ffmpeg PCM decode -> numpy
  dsp.py       spectrum, spectrogram, DR, clipping, bit depth, cutoff estimate
  verdict.py   metrics -> verdict + confidence + suspected source
  analyze.py   orchestrator: path -> AnalysisResult
  convert.py   FLAC/etc -> MP3 (libmp3lame)
  model.py     dataclasses (StreamInfo, Metrics, Spectrum, AnalysisResult, Verdict)
cli.py         batch terminal scanner / converter
gui.py         PyQt6 desktop app
testdata/      make_fixtures.sh -> synthetic genuine/fake/upsampled fixtures
tests/         unittest suite
```

## Tests

The `testdata/` fixtures aren't committed (they're generated binaries).
Recreate them with ffmpeg, then run the suite:

```sh
bash testdata/make_fixtures.sh
python -m unittest discover tests -v
./cli.py testdata --no-color     # eyeball the verdicts
```

The fixtures cover genuine (white-noise, music-like, quiet, hi-res), MP3
transcodes at several bitrates, a padded 24-bit file, clean and lossy-sourced
upsamples, an openly lossy MP3, and a clipped tone.
