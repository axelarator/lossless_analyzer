#!/usr/bin/env python3
"""Command-line batch scanner for a lossless music library.

Examples
--------
Scan a directory, show only files that look fake/suspect/upsampled:

    ./cli.py ~/Music --suspect-only

Scan, write a CSV report, and use 4 workers:

    ./cli.py ~/Music -r --csv report.csv -j4

Convert every file that's a good source (genuine, or upsampled/padded CD
quality) to 320 kbps MP3 into ~/ipod, mirroring the folder layout:

    ./cli.py ~/Music -r --convert --out ~/ipod
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from analyzer import analyze_file
from analyzer.decode import AUDIO_EXTS, have_tools
from analyzer.model import Verdict
from analyzer.convert import DEFAULT_QUALITY, PRESETS, convert_to_mp3

# ANSI colors for the terminal table.
_C = {
    Verdict.GENUINE: "\033[32m",
    Verdict.SUSPECT: "\033[33m",
    Verdict.LIKELY_FAKE: "\033[31m",
    Verdict.UPSAMPLED: "\033[35m",
    Verdict.LOSSY: "\033[34m",
    Verdict.ERROR: "\033[90m",
}
_RESET = "\033[0m"


def gather_files(paths: list[str], recursive: bool) -> dict[str, str | None]:
    """Map each audio file found to the directory it was found under.

    That root is used to mirror the library layout when converting. Files
    given directly on the command line have no root (None).
    """
    files: dict[str, str | None] = {}
    for p in paths:
        if os.path.isfile(p):
            files.setdefault(p, None)
        elif os.path.isdir(p):
            if recursive:
                for root, _, names in os.walk(p):
                    for n in names:
                        if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                            files.setdefault(os.path.join(root, n), p)
            else:
                for n in os.listdir(p):
                    fp = os.path.join(p, n)
                    if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                        files.setdefault(fp, p)
    return dict(sorted(files.items()))


def _worker(path: str):
    # Spectrogram is GUI-only; skip it for speed in batch mode.
    return analyze_file(path, want_spectrogram=False)


def print_row(res, use_color: bool):
    i, m = res.info, res.metrics
    v = res.verdict
    name = os.path.basename(i.path)
    if len(name) > 42:
        name = name[:39] + "..."
    label = v.label
    if res.suspected_source:
        label += f" [{res.suspected_source}]"
    line = (
        f"{name:<42} "
        f"{(i.codec or '?'):>5} "
        f"{(str(i.sample_rate//1000)+'k' if i.sample_rate else '?'):>5} "
        f"{(f'{m.effective_bits}/{i.claimed_bits}b' if i.claimed_bits else '-'):>7} "
        f"{('cut '+str(round(m.cutoff_hz/1000,1))+'k'):>9} "
        f"{('DR'+str(round(m.dr))):>5} "
        f"{('clip'+str(m.clip_runs) if m.clip_runs else ''):>7}  "
        f"{label} ({res.confidence:.0%})"
    )
    if use_color:
        line = f"{_C.get(v,'')}{line}{_RESET}"
    print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lossless audio authenticity scanner")
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into directories")
    ap.add_argument("-j", "--jobs", type=int, default=max((os.cpu_count() or 2) // 2, 1),
                    help="parallel workers (default: half your cores)")
    ap.add_argument("--suspect-only", action="store_true",
                    help="only print files that aren't plain genuine lossless")
    ap.add_argument("--csv", metavar="FILE", help="write a CSV report")
    ap.add_argument("--json", metavar="FILE", help="write a JSON report")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--convert", "--convert-genuine", dest="convert", action="store_true",
                    help="convert good sources (genuine, or upsampled/padded CD quality) "
                         "to MP3; suspects are listed for review, fakes are skipped")
    ap.add_argument("--quality", default=DEFAULT_QUALITY, choices=list(PRESETS),
                    help=f"MP3 preset for conversion (default {DEFAULT_QUALITY})")
    ap.add_argument("--out", help="output directory for conversions "
                                  "(folder layout under each scanned directory is mirrored)")
    args = ap.parse_args(argv)

    if not have_tools():
        sys.exit("error: ffmpeg/ffprobe not found on PATH")

    files = gather_files(args.paths, args.recursive)
    if not files:
        sys.exit("no audio files found")

    use_color = sys.stdout.isatty() and not args.no_color
    print(f"Scanning {len(files)} file(s) with {args.jobs} worker(s)...\n", file=sys.stderr)

    results = []
    counts: dict[Verdict, int] = {}
    show_only = {Verdict.SUSPECT, Verdict.LIKELY_FAKE, Verdict.UPSAMPLED,
                 Verdict.LOSSY, Verdict.ERROR}

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, f): f for f in files}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            counts[res.verdict] = counts.get(res.verdict, 0) + 1
            if args.suspect_only and res.verdict not in show_only:
                continue
            print_row(res, use_color)

    # Summary.
    print("\n" + "-" * 60, file=sys.stderr)
    order = [Verdict.GENUINE, Verdict.SUSPECT, Verdict.LIKELY_FAKE,
             Verdict.UPSAMPLED, Verdict.LOSSY, Verdict.ERROR]
    summary = "  ".join(f"{v.label}: {counts.get(v,0)}" for v in order if counts.get(v))
    print(summary, file=sys.stderr)

    results.sort(key=lambda r: r.info.path)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].summary_row().keys()))
            w.writeheader()
            for r in results:
                w.writerow(r.summary_row())
        print(f"wrote {args.csv}", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([r.as_dict() for r in results], fh, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)

    if args.convert:
        good = [r for r in results if r.convertible]
        review = [r for r in results if not r.convertible
                  and r.verdict in (Verdict.SUSPECT, Verdict.UPSAMPLED)]
        skipped = [r for r in results if r.verdict in (Verdict.LIKELY_FAKE, Verdict.LOSSY)]
        print(f"\nConverting {len(good)} file(s) to MP3 {args.quality}...", file=sys.stderr)
        failed = 0
        for r in good:
            cr = convert_to_mp3(r.info.path, out_dir=args.out, quality=args.quality,
                                rel_to=files.get(r.info.path))
            failed += not cr.ok
            status = "ok" if cr.ok else "FAIL"
            print(f"  [{status}] {cr.output or r.info.path} — {cr.message}", file=sys.stderr)
        if review:
            print(f"\nNot converted — review the spectrogram in the GUI ({len(review)}):",
                  file=sys.stderr)
            for r in review:
                print(f"  {r.info.path}  [{r.verdict.label}, {r.suspected_source or 'no source guess'}]",
                      file=sys.stderr)
        if skipped:
            print(f"\nSkipped — lossy or likely lossy-sourced ({len(skipped)}):", file=sys.stderr)
            for r in skipped:
                print(f"  {r.info.path}  [{r.verdict.label}]", file=sys.stderr)
        if failed:
            sys.exit(f"\n{failed} conversion(s) failed")


if __name__ == "__main__":
    main()
