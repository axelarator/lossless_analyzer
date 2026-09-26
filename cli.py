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

Bulk-convert a whole library to MP3 without analyzing anything first (much
faster for thousands of files; originals are left untouched):

    ./cli.py ~/Music -r --bulk-convert --out ~/ipod

Same, but preview what would be written before committing to the run:

    ./cli.py ~/Music -r --bulk-convert --out ~/ipod --dry-run

Convert one artist, keeping it nested the way it sits in the library
(~/Music/Artist/Album/01.flac -> ~/ipod/Artist/Album/01.mp3):

    ./cli.py ~/Music/Artist -r --bulk-convert --out ~/ipod --library-root ~/Music
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from analyzer import analyze_file
from analyzer.bulk import (
    SKIP_MARKER, bulk_convert, check_out_dir, filter_lossless, gather_files,
)
from analyzer.decode import have_tools
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


def run_bulk_convert(args, files, use_color: bool) -> int:
    """Encode everything in ``files`` to MP3. Returns a process exit code."""
    if not args.out:
        # Without an output directory the MP3s would land next to every source,
        # scattering thousands of files through the library.
        print("error: --bulk-convert needs --out DIR", file=sys.stderr)
        return 1
    if not args.include_lossy:
        files = filter_lossless(files)
    if not files:
        print("no lossless source files found (use --include-lossy to convert "
              "lossy sources too)", file=sys.stderr)
        return 1
    err = check_out_dir(args.out, args.paths)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    total = len(files)
    # Bulk encoding gets every core: one libmp3lame encode is single-threaded,
    # unlike the analysis pass which is memory-hungry and defaults to half.
    jobs = args.jobs or (os.cpu_count() or 2)
    verb = "Would convert" if args.dry_run else "Converting"
    anchor = (f" (paths relative to {os.path.abspath(args.library_root)})"
              if args.library_root else "")
    print(f"{verb} {total} file(s) to MP3 {args.quality} -> {args.out}{anchor}"
          f"{'' if args.dry_run else f' with {jobs} worker(s)'}...\n",
          file=sys.stderr)

    # On a TTY, rewrite a single counter line; otherwise emit a line every so
    # often so piped output and logs stay readable.
    tty = sys.stderr.isatty() and use_color
    state = {"ok": 0, "skip": 0, "fail": 0}

    def on_progress(done, total, src, res):
        if not res.ok:
            state["fail"] += 1
        elif res.message.startswith(SKIP_MARKER):
            state["skip"] += 1
        else:
            state["ok"] += 1
        if args.dry_run:
            print(f"  {res.output}  ({res.message})")
            return
        if not res.ok:
            if tty:
                print("\r\033[K", end="", file=sys.stderr)
            print(f"  [FAIL] {src} — {res.message}", file=sys.stderr)
        line = (f"[{done}/{total}] ok={state['ok']} skip={state['skip']} "
                f"fail={state['fail']}  {os.path.basename(src)}")
        if tty:
            print(f"\r\033[K{line[:110]}", end="", file=sys.stderr, flush=True)
        elif done % 50 == 0 or done == total:
            print(line, file=sys.stderr)

    summary = bulk_convert(
        files, out_dir=args.out, quality=args.quality, jobs=jobs,
        overwrite=args.overwrite, dry_run=args.dry_run,
        progress_cb=on_progress,
    )
    if tty and not args.dry_run:
        print("\r\033[K", end="", file=sys.stderr)

    print("\n" + "-" * 60, file=sys.stderr)
    noun = "would convert" if args.dry_run else "converted"
    print(f"{noun}: {summary.converted}  skipped: {summary.skipped}  "
          f"failed: {summary.failed}", file=sys.stderr)

    if args.error_log and summary.failures:
        with open(args.error_log, "w") as fh:
            for src, msg in summary.failures:
                fh.write(f"{src}\t{msg}\n")
        print(f"wrote {args.error_log}", file=sys.stderr)
    if summary.failures and not args.error_log:
        print("\nFailures:", file=sys.stderr)
        for src, msg in summary.failures[:20]:
            print(f"  {src} — {msg}", file=sys.stderr)
        if len(summary.failures) > 20:
            print(f"  ... and {len(summary.failures) - 20} more "
                  f"(use --error-log to capture them all)", file=sys.stderr)
    return 1 if summary.failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lossless audio authenticity scanner")
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into directories")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="parallel workers (default: half your cores when scanning, "
                         "all of them for --bulk-convert)")
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
                                  "(the scanned folder and its layout are mirrored inside)")
    ap.add_argument("--library-root", metavar="DIR",
                    help="mirror output paths relative to this library root instead of "
                         "the parent of each scanned folder, so scanning ~/Music and "
                         "scanning ~/Music/Artist both write <out>/Artist/Album/...")
    ap.add_argument("--bulk-convert", action="store_true",
                    help="convert every source file to MP3 without analyzing it first; "
                         "much faster for a large library. Originals are never modified")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-encode files whose MP3 already exists (default: skip them, "
                         "so an interrupted run resumes where it left off)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --bulk-convert, list what would be written and exit")
    ap.add_argument("--include-lossy", action="store_true",
                    help="with --bulk-convert, also convert lossy sources (mp3, ogg, ...) "
                         "instead of only lossless ones")
    ap.add_argument("--error-log", metavar="FILE",
                    help="with --bulk-convert, write failed sources and their errors here")
    args = ap.parse_args(argv)

    if not have_tools():
        sys.exit("error: ffmpeg/ffprobe not found on PATH")

    if args.library_root and not os.path.isdir(args.library_root):
        sys.exit(f"error: library root is not a directory: {args.library_root}")

    files = gather_files(args.paths, args.recursive, args.library_root)
    if not files:
        sys.exit("no audio files found")

    use_color = sys.stdout.isatty() and not args.no_color

    if args.bulk_convert:
        sys.exit(run_bulk_convert(args, files, use_color))

    jobs = args.jobs or max((os.cpu_count() or 2) // 2, 1)
    print(f"Scanning {len(files)} file(s) with {jobs} worker(s)...\n", file=sys.stderr)

    results = []
    counts: dict[Verdict, int] = {}
    show_only = {Verdict.SUSPECT, Verdict.LIKELY_FAKE, Verdict.UPSAMPLED,
                 Verdict.LOSSY, Verdict.ERROR}

    with ProcessPoolExecutor(max_workers=jobs) as ex:
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
                                overwrite=args.overwrite,
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
