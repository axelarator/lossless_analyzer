"""Bulk FLAC -> MP3 conversion for filling a portable player.

Walks files and folders, then encodes every source it finds into an output
directory, mirroring the library's folder layout. Sources are only ever read,
so the originals stay exactly where they are.

Encoding runs in a thread pool rather than a process pool: ``convert_to_mp3``
is a blocking ``subprocess.run`` around ffmpeg, which releases the GIL while it
waits, so the real parallelism lives in the ffmpeg processes. Threads avoid the
pickling and spawn overhead a process pool would add for nothing.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .convert import DEFAULT_QUALITY, ConvertResult, convert_to_mp3, output_path
from .decode import AUDIO_EXTS, is_lossless_codec, probe

# The subset of AUDIO_EXTS worth feeding to the encoder. Transcoding an MP3 to
# MP3 only loses quality, and skipping lossy sources also stops a second run
# from picking up its own output if the output directory ever lands inside the
# music folder.
LOSSLESS_EXTS = {
    ".flac", ".wav", ".aiff", ".aif", ".aifc", ".alac", ".ape", ".wv",
    ".tta", ".tak", ".dsf", ".dff",
}
# Containers that hold either kind (.m4a is AAC or ALAC, .wma is WMA or WMA
# Lossless): the codec decides, so these are probed.
MIXED_EXTS = {".m4a", ".wma"}

# convert_to_mp3 reports a pre-existing output as a success; this is how we
# tell "already there" apart from "just encoded it".
SKIP_MARKER = "already exists"


@dataclass
class BulkSummary:
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    stopped: bool = False
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.converted + self.skipped + self.failed


def mirror_root(path: str, library_root: str | None = None) -> str | None:
    """The directory that output paths are mirrored relative to.

    With an explicit ``library_root`` every file is placed by its path under
    that root, so scanning ``~/Music`` and scanning ``~/Music/Artist`` both put
    the track at ``<out>/Artist/Album/``. Without one, the default root is the
    *parent* of the folder being scanned, so the folder you picked is itself
    reproduced in the output rather than being flattened away.
    """
    if library_root:
        return os.path.abspath(library_root)
    return os.path.dirname(os.path.abspath(path)) or None


def gather_files(paths: Iterable[str], recursive: bool,
                 library_root: str | None = None) -> dict[str, str | None]:
    """Map each audio file found to the root its output path is mirrored under.

    See ``mirror_root`` for how that root is chosen. A file named directly on
    the command line, with no library root to place it under, has none (None)
    and lands flat in the output directory.
    """
    files: dict[str, str | None] = {}
    for p in paths:
        root = mirror_root(p, library_root)
        if os.path.isfile(p):
            files.setdefault(p, root if library_root else None)
        elif os.path.isdir(p):
            if recursive:
                for r, _, names in os.walk(p):
                    for n in names:
                        if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                            files.setdefault(os.path.join(r, n), root)
            else:
                for n in os.listdir(p):
                    fp = os.path.join(p, n)
                    if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                        files.setdefault(fp, root)
    return dict(sorted(files.items()))


def _probes_lossless(path: str) -> bool:
    try:
        return is_lossless_codec(probe(path).codec)
    except Exception:  # noqa: BLE001 - unreadable: leave it out
        return False


def filter_lossless(files: dict[str, str | None]) -> dict[str, str | None]:
    """Drop lossy sources (already-MP3s and friends) from a gathered mapping."""
    def ext(p):
        return os.path.splitext(p)[1].lower()

    mixed = [p for p in files if ext(p) in MIXED_EXTS]
    with ThreadPoolExecutor(max_workers=min(len(mixed), 16) or 1) as ex:
        lossless_mixed = {p for p, ok in zip(mixed, ex.map(_probes_lossless, mixed)) if ok}
    return {p: root for p, root in files.items()
            if ext(p) in LOSSLESS_EXTS or p in lossless_mixed}


def check_out_dir(out_dir: str, paths: Iterable[str]) -> str | None:
    """Return an error message if ``out_dir`` sits inside one of ``paths``.

    Writing MP3s underneath a folder we scan means a second run would treat its
    own output as source material.
    """
    out = os.path.abspath(out_dir)
    for p in paths:
        if not os.path.isdir(p):
            continue
        p_abs = os.path.abspath(p)
        if out == p_abs or out.startswith(p_abs + os.sep):
            return (f"output directory {out} is inside the scanned folder "
                    f"{p_abs}; choose a location outside your library")
    return None


def _convert_one(src: str, root: str | None, out_dir: str | None,
                 quality: str, overwrite: bool) -> tuple[str, ConvertResult]:
    """Encode one file, tagging the result with its source path."""
    return src, convert_to_mp3(src, out_dir=out_dir, quality=quality,
                               overwrite=overwrite, rel_to=root)


def bulk_convert(
    files: dict[str, str | None],
    out_dir: str | None = None,
    quality: str = DEFAULT_QUALITY,
    jobs: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress_cb: Callable[[int, int, str, ConvertResult], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> BulkSummary:
    """Convert every file in ``files`` to MP3 under ``out_dir``.

    ``files`` maps each source to the library root it was found under (see
    ``gather_files``), which is what mirrors the folder layout in the output.
    ``progress_cb`` is called once per finished file with
    ``(done, total, src, result)``; ``should_stop`` is polled between
    completions so a GUI can cancel a long run.
    """
    summary = BulkSummary()
    total = len(files)
    if not total:
        return summary

    if dry_run:
        for done, (src, root) in enumerate(files.items(), 1):
            out = output_path(src, out_dir, root)
            exists = os.path.exists(out)
            if exists and not overwrite:
                summary.skipped += 1
                res = ConvertResult(True, out, f"{SKIP_MARKER} (would skip)")
            else:
                summary.converted += 1
                res = ConvertResult(True, out, "would encode")
            if progress_cb:
                progress_cb(done, total, src, res)
        return summary

    done = 0
    workers = max(jobs or os.cpu_count() or 2, 1)
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(_convert_one, src, root, out_dir, quality, overwrite)
                for src, root in files.items()]
        for fut in as_completed(futs):
            if should_stop and should_stop():
                summary.stopped = True
                break
            src, res = fut.result()
            done += 1
            if not res.ok:
                summary.failed += 1
                summary.failures.append((src, res.message))
            elif res.message.startswith(SKIP_MARKER):
                summary.skipped += 1
            else:
                summary.converted += 1
            if progress_cb:
                progress_cb(done, total, src, res)
    finally:
        # Drop queued work instead of waiting for the whole library to finish.
        ex.shutdown(wait=False, cancel_futures=True)
    return summary
