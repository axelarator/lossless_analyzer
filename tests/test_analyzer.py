"""Tests for the analyzer. Run from the repo root:

    python -m unittest discover tests -v

Fixture-based tests need ``bash testdata/make_fixtures.sh`` first and are
skipped otherwise; conversion tests need ffmpeg with libmp3lame.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyzer import analyze_file, dsp  # noqa: E402
from analyzer.convert import convert_to_mp3, output_path  # noqa: E402
from analyzer.decode import have_tools, probe  # noqa: E402
from analyzer.model import Metrics, StreamInfo, Verdict  # noqa: E402
from analyzer.verdict import _suspected_source, classify  # noqa: E402

TESTDATA = os.path.join(ROOT, "testdata")


def fixture(name: str) -> str:
    path = os.path.join(TESTDATA, name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{name} missing — run: bash testdata/make_fixtures.sh")
    return path


# --------------------------------------------------------------------------- #
# End-to-end verdicts on the generated fixtures
# --------------------------------------------------------------------------- #
# name -> (allowed verdicts, convertible, substring expected in suspected_source)
EXPECTED = {
    "genuine_44k.flac": ({Verdict.GENUINE}, True, None),
    "genuine_96k.flac": ({Verdict.GENUINE}, True, None),
    "music_genuine.flac": ({Verdict.GENUINE}, True, None),
    "quiet_genuine.flac": ({Verdict.GENUINE}, True, None),
    "clipped.flac": ({Verdict.GENUINE}, True, None),
    "padded_24bit.flac": ({Verdict.SUSPECT}, True, None),
    "upsampled_96k.flac": ({Verdict.UPSAMPLED}, True, None),
    "fake_from_mp3_128.flac": ({Verdict.LIKELY_FAKE}, False, "128"),
    "music_fake_192.flac": ({Verdict.LIKELY_FAKE}, False, "192"),
    # 320k walls sit right where tight anti-alias filters do: flagged either way.
    "fake_from_mp3_320.flac": ({Verdict.SUSPECT, Verdict.LIKELY_FAKE}, False, "320"),
    "music_fake_320.flac": ({Verdict.SUSPECT, Verdict.LIKELY_FAKE}, False, "320"),
    "quiet_fake_128.flac": ({Verdict.SUSPECT, Verdict.LIKELY_FAKE}, False, "128"),
    "upsampled_from_mp3_96k.flac": ({Verdict.UPSAMPLED}, False, "128"),
    "lossy_192.mp3": ({Verdict.LOSSY}, False, None),
}


class TestFixtureVerdicts(unittest.TestCase):
    def test_fixtures(self):
        for name, (verdicts, convertible, source) in EXPECTED.items():
            with self.subTest(name):
                res = analyze_file(fixture(name))
                self.assertFalse(res.error, res.error)
                self.assertIn(res.verdict, verdicts, res.reasons)
                self.assertEqual(res.convertible, convertible)
                if source:
                    self.assertIn(source, res.suspected_source)

    def test_padded_bits_detected(self):
        res = analyze_file(fixture("padded_24bit.flac"))
        self.assertEqual(res.info.claimed_bits, 24)
        self.assertEqual(res.metrics.effective_bits, 16)
        self.assertTrue(res.metrics.bit_padded)

    def test_clipping_detected(self):
        res = analyze_file(fixture("clipped.flac"))
        self.assertGreater(res.metrics.clip_runs, 100)


# --------------------------------------------------------------------------- #
# Classifier rules on hand-built metrics
# --------------------------------------------------------------------------- #
def _info(codec="flac", sr=44100, bits=16) -> StreamInfo:
    return StreamInfo(path="x", codec=codec, sample_rate=sr, channels=2, claimed_bits=bits)


class TestClassify(unittest.TestCase):
    def test_lossy_codec_is_not_genuine(self):
        v, *_ = classify(_info(codec="mp3", bits=0), Metrics(wall_hz=21000))
        self.assertEqual(v, Verdict.LOSSY)

    def test_pcm_counts_as_lossless(self):
        m = Metrics(wall_hz=16500, wall_drop_db=60, shelf_db=-100, noise_floor_db=-100)
        v, *_ = classify(_info(codec="pcm_s24be"), m)
        self.assertEqual(v, Verdict.LIKELY_FAKE)

    def test_anti_alias_wall_at_nyquist_is_genuine(self):
        m = Metrics(wall_hz=21040, wall_drop_db=50, shelf_db=-120, noise_floor_db=-110)
        v, *_ = classify(_info(), m)
        self.assertEqual(v, Verdict.GENUINE)

    def test_shallow_wall_with_flat_shelf_is_suspect(self):
        m = Metrics(wall_hz=16300, wall_drop_db=22, shelf_db=-85.0, noise_floor_db=-85.2)
        v, _, _, source = classify(_info(), m)
        self.assertEqual(v, Verdict.SUSPECT)
        self.assertIn("128", source)

    def test_shallow_edge_with_sloping_shelf_is_genuine(self):
        # A dull genuine master: moderate edge, but the band keeps falling.
        m = Metrics(wall_hz=17340, wall_drop_db=16.6, shelf_db=-84.5, noise_floor_db=-87.9)
        v, *_ = classify(_info(), m)
        self.assertEqual(v, Verdict.GENUINE)

    def test_source_table_matches_measured_lame_walls(self):
        # Wall positions measured on real music through LAME 3.100.
        cases = {15300: "112", 16400: "128", 16950: "128", 17300: "160",
                 18500: "192", 19300: "256", 20000: "320"}
        for wall, label in cases.items():
            with self.subTest(wall=wall):
                self.assertIn(label, _suspected_source(wall))


# --------------------------------------------------------------------------- #
# DSP measurements
# --------------------------------------------------------------------------- #
class TestDsp(unittest.TestCase):
    def test_dr_uses_block_peaks_not_sample_peaks(self):
        sr = 1000
        rng = np.random.default_rng(0)
        x = 0.1 * rng.standard_normal((30 * sr, 1))
        x[5000, 0] = 0.99
        one_spike = dsp.dynamic_range(x, sr)
        x[5001, 0] = 0.99  # second spike in the same 3 s block
        # The 2nd-highest *block* peak is unchanged, so DR barely moves. (Taking
        # the 2nd-highest *sample* would jump ~6 dB to the spike level.)
        self.assertAlmostEqual(dsp.dynamic_range(x, sr), one_spike, delta=0.2)

    def test_peak_uses_loudest_channel_not_downmix(self):
        t = np.linspace(0, 1, 44100, endpoint=False)
        s = 0.5 * np.sin(2 * np.pi * 1000 * t)
        stereo = np.stack([s, -s], axis=1)  # mono downmix would be silence
        peak_db, rms_db, _ = dsp.peak_rms_crest(stereo)
        self.assertAlmostEqual(peak_db, 20 * np.log10(0.5), places=2)
        self.assertAlmostEqual(rms_db, 20 * np.log10(0.5 / np.sqrt(2)), places=2)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
class TestOutputPath(unittest.TestCase):
    def test_mirrors_library_layout(self):
        src = "/lib/Artist/Album/01 Intro.flac"
        self.assertEqual(output_path(src, "/out", rel_to="/lib"),
                         "/out/Artist/Album/01 Intro.mp3")

    def test_flat_without_root(self):
        self.assertEqual(output_path("/lib/a/01.flac", "/out"), "/out/01.mp3")

    def test_next_to_source_without_out_dir(self):
        self.assertEqual(output_path("/lib/a/01.flac"), "/lib/a/01.mp3")


@unittest.skipUnless(have_tools(), "ffmpeg/ffprobe not installed")
class TestConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_same_name_different_albums_do_not_collide(self):
        src = fixture("music_genuine.flac")
        lib = os.path.join(self.tmp, "lib")
        for album in ("A", "B"):
            os.makedirs(os.path.join(lib, album))
            shutil.copy(src, os.path.join(lib, album, "01.flac"))
        out = os.path.join(self.tmp, "out")
        for album in ("A", "B"):
            cr = convert_to_mp3(os.path.join(lib, album, "01.flac"), out, rel_to=lib)
            self.assertTrue(cr.ok, cr.message)
            self.assertEqual(cr.message, "encoded (320)")
        self.assertTrue(os.path.exists(os.path.join(out, "A", "01.mp3")))
        self.assertTrue(os.path.exists(os.path.join(out, "B", "01.mp3")))

    def test_default_is_320_cbr_and_hires_is_resampled(self):
        cr = convert_to_mp3(fixture("genuine_96k.flac"), self.tmp)
        self.assertTrue(cr.ok, cr.message)
        info = probe(cr.output)
        self.assertEqual(info.codec, "mp3")
        self.assertEqual(info.sample_rate, 44100)
        self.assertEqual(info.channels, 2)
        self.assertAlmostEqual(info.bit_rate / 1000, 320, delta=3)

    def test_failed_encode_leaves_no_file(self):
        bad = os.path.join(self.tmp, "bad.flac")
        with open(bad, "wb") as fh:
            fh.write(b"not audio at all")
        cr = convert_to_mp3(bad, self.tmp)
        self.assertFalse(cr.ok)
        self.assertEqual([n for n in os.listdir(self.tmp) if n != "bad.flac"], [])

    def test_tags_and_cover_art_are_copied(self):
        src = os.path.join(self.tmp, "tagged.flac")
        cover = os.path.join(self.tmp, "cover.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=red:s=64x64", "-frames:v", "1", cover], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", fixture("music_genuine.flac"),
                        "-i", cover, "-map", "0:a", "-map", "1:v", "-c:a", "copy",
                        "-c:v", "png", "-disposition:v", "attached_pic",
                        "-metadata", "title=Song", "-metadata", "artist=Band", src],
                       check=True)
        cr = convert_to_mp3(src, os.path.join(self.tmp, "out"))
        self.assertTrue(cr.ok, cr.message)
        info = probe(cr.output)
        self.assertEqual((info.title, info.artist), ("Song", "Band"))
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", cr.output], capture_output=True, text=True).stdout.split()
        self.assertIn("video", streams)  # attached cover art


if __name__ == "__main__":
    unittest.main()
