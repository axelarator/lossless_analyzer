#!/usr/bin/env python3
"""PyQt6 desktop GUI for the lossless analyzer.

Pick a directory, scan it (in parallel, off the UI thread), browse results in a
color-coded table, inspect a file's spectrogram / average spectrum / metrics in
the detail pane, and convert genuine files to MP3 for A/B listening. A whole
folder can also be bulk-converted to MP3 without scanning it first.

Run:  ./gui.py   (or  python3 gui.py)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from analyzer import analyze_file, scan_one  # noqa: E402
from analyzer.analyze import result_or_error  # noqa: E402
from analyzer.bulk import (  # noqa: E402
    bulk_convert, check_out_dir, filter_lossless, gather_files, mirror_root,
)
from analyzer.decode import have_tools  # noqa: E402
from analyzer.model import AnalysisResult, Verdict  # noqa: E402
from analyzer.convert import convert_to_mp3, DEFAULT_QUALITY, PRESETS  # noqa: E402


# --------------------------------------------------------------------------- #
# Background workers
# --------------------------------------------------------------------------- #
class ScanWorker(QtCore.QThread):
    """Analyze a list of files in a process pool, emitting results as they land."""

    result = QtCore.pyqtSignal(object)        # AnalysisResult
    progress = QtCore.pyqtSignal(int, int)    # done, total
    done = QtCore.pyqtSignal()

    def __init__(self, files: list[str], jobs: int):
        super().__init__()
        self.files = files
        self.jobs = max(jobs, 1)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        total = len(self.files)
        done = 0
        ex = ProcessPoolExecutor(max_workers=self.jobs)
        try:
            futs = {ex.submit(scan_one, f): f for f in self.files}
            for fut in as_completed(futs):
                if self._stop:
                    break
                self.result.emit(result_or_error(fut, futs[fut]))
                done += 1
                self.progress.emit(done, total)
        finally:
            # Drop queued files instead of waiting for the whole library
            # (a `with` block would block here until every file finished).
            ex.shutdown(wait=False, cancel_futures=True)
            self.done.emit()


class DetailWorker(QtCore.QThread):
    """Re-analyze one file with the spectrogram for the detail pane."""

    ready = QtCore.pyqtSignal(object)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        self.ready.emit(analyze_file(self.path, want_spectrogram=True))


class ConvertWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)  # ConvertResult

    def __init__(self, src, out_dir, quality, rel_to=None):
        super().__init__()
        self.src, self.out_dir, self.quality, self.rel_to = src, out_dir, quality, rel_to

    def run(self):
        self.done.emit(convert_to_mp3(self.src, self.out_dir, self.quality,
                                      overwrite=True, rel_to=self.rel_to))


class BulkConvertWorker(QtCore.QThread):
    """Encode a whole library to MP3 off the UI thread."""

    progress = QtCore.pyqtSignal(int, int, str)   # done, total, source path
    finished_summary = QtCore.pyqtSignal(object)  # BulkSummary

    def __init__(self, files: dict[str, str | None], out_dir: str, quality: str,
                 jobs: int):
        super().__init__()
        self.files, self.out_dir, self.quality = files, out_dir, quality
        self.jobs = max(jobs, 1)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        summary = bulk_convert(
            self.files, out_dir=self.out_dir, quality=self.quality,
            jobs=self.jobs,
            progress_cb=lambda done, total, src, res:
                self.progress.emit(done, total, src),
            should_stop=lambda: self._stop,
        )
        self.finished_summary.emit(summary)


# --------------------------------------------------------------------------- #
# Spectrum / spectrogram plot
# --------------------------------------------------------------------------- #
class SpectrumView(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), tight_layout=True, facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.ax_spec = self.fig.add_subplot(2, 1, 1)
        self.ax_avg = self.fig.add_subplot(2, 1, 2)
        self.clear_plot("Select a file to view its spectrum")

    def _style(self, ax):
        ax.set_facecolor("#111111")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.tick_params(colors="#ccc", labelsize=8)
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")

    def clear_plot(self, msg=""):
        for ax in (self.ax_spec, self.ax_avg):
            ax.clear()
            self._style(ax)
        if msg:
            self.ax_spec.text(0.5, 0.5, msg, ha="center", va="center",
                              color="#888", transform=self.ax_spec.transAxes)
        self.draw()

    def show_result(self, res: AnalysisResult):
        sp = res.spectrum
        if sp is None or sp.freqs.size == 0:
            self.clear_plot("No spectrum available")
            return
        nyq = res.info.nyquist
        cutoff_khz = res.metrics.cutoff_hz / 1000.0

        # Spectrogram.
        ax = self.ax_spec
        ax.clear(); self._style(ax)
        if sp.spec_db.size > 1 and sp.spec_times.size:
            extent = [0, float(sp.spec_times[-1]), 0, float(sp.spec_freqs[-1]) / 1000.0]
            ax.imshow(sp.spec_db, origin="lower", aspect="auto", extent=extent,
                      cmap="magma", vmin=-120, vmax=0)
            ax.set_ylabel("kHz")
            ax.set_xlabel("seconds")
            ax.set_title(f"Spectrogram — {os.path.basename(res.info.path)}", fontsize=9)
            if res.metrics.cutoff_hz < nyq * 0.98:
                ax.axhline(cutoff_khz, color="#3df", lw=1.0, ls="--", alpha=0.8)

        # Average spectrum.
        ax = self.ax_avg
        ax.clear(); self._style(ax)
        ax.plot(sp.freqs / 1000.0, sp.power_db, color="#5cf", lw=0.8)
        ax.set_xlim(0, nyq / 1000.0)
        ax.set_ylim(-140, 3)
        ax.set_xlabel("kHz")
        ax.set_ylabel("dB")
        ax.set_title("Average spectrum", fontsize=9)
        if res.metrics.cutoff_hz < nyq * 0.98:
            ax.axvline(cutoff_khz, color="#3df", lw=1.0, ls="--",
                       label=f"cutoff {cutoff_khz:.1f} kHz")
        ax.axvline(nyq / 1000.0, color="#888", lw=0.8, ls=":",
                   label=f"Nyquist {nyq/1000:.1f} kHz")
        ax.legend(fontsize=7, facecolor="#222", edgecolor="#555", labelcolor="#ccc")
        self.draw()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
COLUMNS = ["File", "Codec", "Rate", "Bits", "Cutoff", "DR", "Clip", "Verdict", "Conf"]
SORT_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class SortItem(QtWidgets.QTableWidgetItem):
    """Table cell that sorts by a numeric key (SORT_ROLE) when one is set."""

    def __lt__(self, other):
        a, b = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lossless Analyzer")
        self.resize(1280, 760)
        self.results: dict[str, AnalysisResult] = {}
        self.scan_worker: ScanWorker | None = None
        # Every DetailWorker stays referenced until its thread finishes; letting
        # a running QThread be garbage-collected aborts the whole app.
        self.detail_workers: set[DetailWorker] = set()
        self.convert_worker: ConvertWorker | None = None
        self.bulk_worker: BulkConvertWorker | None = None
        self.scan_dir = os.path.expanduser("~")
        self.out_dir = ""
        # None = anchor output paths at the parent of the scanned folder, so the
        # folder you picked is reproduced in the output. Set it to your library
        # root to keep paths identical no matter which subfolder you convert.
        self.library_root: str | None = None
        # Selection is debounced: arrowing through the table would otherwise
        # start a full decode for every row passed over.
        self.detail_timer = QtCore.QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(150)
        self.detail_timer.timeout.connect(self._load_detail)

        self._build_ui()
        if not have_tools():
            QtWidgets.QMessageBox.critical(
                self, "Missing tools",
                "ffmpeg / ffprobe were not found on PATH. Install them and restart.",
            )

    # -- UI construction ---------------------------------------------------- #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Toolbar row.
        bar = QtWidgets.QHBoxLayout()
        self.dir_label = QtWidgets.QLineEdit(self.scan_dir)
        self.dir_label.setReadOnly(True)
        btn_choose = QtWidgets.QPushButton("Choose Directory…")
        btn_choose.clicked.connect(self.choose_dir)
        self.recursive_cb = QtWidgets.QCheckBox("Recursive")
        self.recursive_cb.setChecked(True)
        self.suspect_cb = QtWidgets.QCheckBox("Suspects only")
        self.suspect_cb.stateChanged.connect(self.apply_filter)
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_scan.clicked.connect(self.start_scan)
        bar.addWidget(QtWidgets.QLabel("Folder:"))
        bar.addWidget(self.dir_label, 1)
        bar.addWidget(btn_choose)
        bar.addWidget(self.recursive_cb)
        bar.addWidget(self.suspect_cb)
        bar.addWidget(self.btn_scan)
        root.addLayout(bar)

        # Splitter: table | detail.
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        for c in range(1, len(COLUMNS)):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.on_select)
        split.addWidget(self.table)

        # Detail pane.
        detail = QtWidgets.QWidget()
        dl = QtWidgets.QVBoxLayout(detail)
        self.plot = SpectrumView()
        dl.addWidget(self.plot, 1)

        self.verdict_label = QtWidgets.QLabel("")
        self.verdict_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.verdict_label.setWordWrap(True)
        dl.addWidget(self.verdict_label)

        self.metrics_text = QtWidgets.QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(170)
        self.metrics_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        dl.addWidget(self.metrics_text)

        # Convert row.
        conv = QtWidgets.QHBoxLayout()
        conv.addWidget(QtWidgets.QLabel("MP3 quality:"))
        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(list(PRESETS.keys()))
        self.quality_combo.setCurrentText(DEFAULT_QUALITY)
        conv.addWidget(self.quality_combo)
        self.btn_convert = QtWidgets.QPushButton("Convert to MP3")
        self.btn_convert.clicked.connect(self.convert_selected)
        self.btn_convert.setEnabled(False)
        conv.addWidget(self.btn_convert)
        self.btn_folder = QtWidgets.QPushButton("Open folder")
        self.btn_folder.clicked.connect(self.open_folder)
        self.btn_folder.setEnabled(False)
        conv.addWidget(self.btn_folder)
        conv.addStretch(1)
        dl.addLayout(conv)

        # Bulk convert row — works straight off the chosen folder, no scan needed.
        bulk = QtWidgets.QHBoxLayout()
        self.btn_bulk = QtWidgets.QPushButton("Convert folder to MP3…")
        self.btn_bulk.setToolTip(
            "Convert every lossless file in the chosen folder to MP3.\n"
            "Originals are never modified. Files already converted are skipped.")
        self.btn_bulk.clicked.connect(self.bulk_convert_folder)
        bulk.addWidget(self.btn_bulk)
        self.lossy_cb = QtWidgets.QCheckBox("Include lossy sources")
        self.lossy_cb.setToolTip(
            "Also convert mp3/ogg/aac sources. Off by default: re-encoding "
            "lossy audio only loses more quality.")
        bulk.addWidget(self.lossy_cb)
        bulk.addStretch(1)
        dl.addLayout(bulk)

        # Library root row — what output paths are mirrored relative to.
        rootrow = QtWidgets.QHBoxLayout()
        rootrow.addWidget(QtWidgets.QLabel("Library root:"))
        self.root_label = QtWidgets.QLineEdit()
        self.root_label.setReadOnly(True)
        self.root_label.setToolTip(
            "Output paths are mirrored relative to this folder.\n"
            "Set it to your music library so converting one artist lands at\n"
            "<output>/Artist/Album/ just as it sits in the library.")
        rootrow.addWidget(self.root_label, 1)
        btn_root = QtWidgets.QPushButton("Set…")
        btn_root.clicked.connect(self.choose_library_root)
        rootrow.addWidget(btn_root)
        self.btn_root_auto = QtWidgets.QPushButton("Auto")
        self.btn_root_auto.setToolTip("Go back to anchoring at the scanned folder's parent.")
        self.btn_root_auto.clicked.connect(self.clear_library_root)
        rootrow.addWidget(self.btn_root_auto)
        dl.addLayout(rootrow)
        self._refresh_root_label()

        split.addWidget(detail)
        split.setSizes([620, 660])

        # Status bar.
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(240)
        self.statusBar().addPermanentWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status)

    # -- Scanning ----------------------------------------------------------- #
    def choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose music folder", self.scan_dir)
        if d:
            self.scan_dir = d
            self.dir_label.setText(d)
            self._refresh_root_label()

    def _effective_root(self) -> str | None:
        return mirror_root(self.scan_dir, self.library_root)

    def _refresh_root_label(self):
        root = self._effective_root() or ""
        self.root_label.setText(
            root if self.library_root else f"{root}   (auto: parent of scanned folder)")
        self.btn_root_auto.setEnabled(self.library_root is not None)

    def choose_library_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Library root (output paths are mirrored relative to this)",
            self.library_root or self.scan_dir)
        if d:
            self.library_root = d
            self._refresh_root_label()

    def clear_library_root(self):
        self.library_root = None
        self._refresh_root_label()

    def _gather(self) -> list[str]:
        return list(gather_files([self.scan_dir], self.recursive_cb.isChecked(),
                                 self.library_root))

    def start_scan(self):
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.stop()
            return
        if self.bulk_worker and self.bulk_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Conversion running",
                "Stop the bulk conversion before scanning.")
            return
        files = self._gather()
        if not files:
            self.status.setText("No audio files found in folder.")
            return
        self.table.setRowCount(0)
        self.results.clear()
        self.table.setSortingEnabled(False)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.status.setText(f"Scanning {len(files)} file(s)…")
        self.btn_scan.setText("Stop")

        jobs = max((os.cpu_count() or 2) // 2, 1)
        self.scan_worker = ScanWorker(files, jobs)
        self.scan_worker.result.connect(self.add_result)
        self.scan_worker.progress.connect(
            lambda d, t: (self.progress.setValue(d),
                          self.status.setText(f"Analyzed {d}/{t}")))
        self.scan_worker.done.connect(self.scan_finished)
        self.scan_worker.start()

    def scan_finished(self):
        self.btn_scan.setText("Scan")
        self.table.setSortingEnabled(True)
        n = len(self.results)
        errors = sum(1 for r in self.results.values() if r.verdict == Verdict.ERROR)
        flagged = sum(1 for r in self.results.values()
                      if r.verdict not in (Verdict.GENUINE, Verdict.ERROR))
        good = sum(1 for r in self.results.values() if r.convertible)
        self.status.setText(
            f"Done — {n} file(s), {flagged} flagged, {good} OK to convert"
            + (f", {errors} could not be read." if errors else "."))

    def add_result(self, res: AnalysisResult):
        self.results[res.info.path] = res
        self._insert_row(res)

    def _insert_row(self, res: AnalysisResult):
        i, m = res.info, res.metrics
        row = self.table.rowCount()
        self.table.insertRow(row)
        bits = f"{m.effective_bits}/{i.claimed_bits}" if i.claimed_bits else str(m.effective_bits)
        cells = [
            os.path.basename(i.path),
            i.codec or "?",
            f"{i.sample_rate/1000:g}k" if i.sample_rate else "?",
            bits,
            f"{m.cutoff_hz/1000:.1f}k" if m.cutoff_hz else "-",
            f"{m.dr:.0f}",
            str(m.clip_runs) if m.clip_runs else "",
            res.verdict.short,
            f"{res.confidence:.0%}",
        ]
        color = QtGui.QColor(res.verdict.color)
        full_verdict = res.verdict.label + (
            f" [{res.suspected_source}]" if res.suspected_source else "")
        # Numeric sort keys for the numeric-ish columns.
        keys = {2: i.sample_rate, 3: m.effective_bits, 4: m.cutoff_hz, 5: m.dr,
                6: m.clip_runs, 8: res.confidence}
        for col, txt in enumerate(cells):
            item = SortItem(txt)
            if col in keys:
                item.setData(SORT_ROLE, float(keys[col]))
            if col == 0:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, i.path)
                item.setToolTip(i.path)
            if col == 7:
                item.setForeground(color)
                item.setToolTip(full_verdict)
                f = item.font(); f.setBold(True); item.setFont(f)
            self.table.setItem(row, col, item)
        self.apply_filter_row(row)

    # -- Filtering ---------------------------------------------------------- #
    def apply_filter(self):
        for row in range(self.table.rowCount()):
            self.apply_filter_row(row)

    def apply_filter_row(self, row: int):
        if not self.suspect_cb.isChecked():
            self.table.setRowHidden(row, False)
            return
        item = self.table.item(row, 0)
        path = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        res = self.results.get(path)
        hide = res is not None and res.verdict == Verdict.GENUINE
        self.table.setRowHidden(row, hide)

    # -- Selection / detail ------------------------------------------------- #
    def _selected_path(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def on_select(self):
        path = self._selected_path()
        if not path:
            return
        res = self.results.get(path)
        if not res:
            return
        self.btn_convert.setEnabled(res.verdict != Verdict.ERROR)
        self.btn_folder.setEnabled(True)
        self._show_text(res)
        if res.spectrum is not None and res.spectrum.spec_db.size > 1:
            self.detail_timer.stop()
            self.plot.show_result(res)  # spectrogram already computed
            return
        if res.verdict == Verdict.ERROR:
            self.detail_timer.stop()
            self.plot.clear_plot("No spectrum available")
            return
        self.plot.clear_plot("Computing spectrogram…")
        self.detail_timer.start()

    def _load_detail(self):
        path = self._selected_path()
        if not path:
            return
        # Compute the spectrogram off-thread.
        worker = DetailWorker(path)
        worker.ready.connect(self._on_detail)
        worker.finished.connect(lambda w=worker: self.detail_workers.discard(w))
        self.detail_workers.add(worker)
        worker.start()

    def _on_detail(self, res: AnalysisResult):
        # Only render if it's still the selected file.
        if res.info.path == self._selected_path():
            self.results[res.info.path] = res  # cache spectrogram
            self.plot.show_result(res)
            self._show_text(res)

    def _show_text(self, res: AnalysisResult):
        i, m = res.info, res.metrics
        self.verdict_label.setText(
            f'<b style="color:{res.verdict.color}; font-size:14px;">'
            f'{res.verdict.label}</b> &nbsp; confidence {res.confidence:.0%}'
            + (f' &nbsp;|&nbsp; suspected source: <b>{res.suspected_source}</b>'
               if res.suspected_source else '')
        )
        if res.error:
            self.metrics_text.setPlainText(f"ERROR: {res.error}")
            return
        lines = [
            f"{i.artist} — {i.title}" if (i.artist or i.title) else os.path.basename(i.path),
            f"codec={i.codec}  container={i.container}  {i.sample_rate} Hz  "
            f"{i.channels}ch  claimed {i.claimed_bits}-bit  ({i.duration:.0f}s)",
            "",
            f"effective bit depth : {m.effective_bits} bit",
            f"content top         : {m.cutoff_hz/1000:.2f} kHz  "
            f"(Nyquist {i.nyquist/1000:g} kHz, floor {m.noise_floor_db:.0f} dB rel. peak)",
            f"steepest band edge  : {m.wall_drop_db:.0f} dB drop at "
            f"{m.wall_hz/1000:.2f} kHz, shelf {m.shelf_db:.0f} dB rel. peak",
            f"                      (lossy: shelf = floor; genuine edges below "
            f"Nyquist rarely exceed ~17 dB)",
            f"dynamic range (DR)  : {m.dr:.1f}",
            f"peak / RMS          : {m.peak_dbfs:.2f} / {m.rms_dbfs:.2f} dBFS "
            f"(crest {m.crest_db:.1f} dB)",
            f"clipping            : {m.clip_runs} runs, {m.clipped_samples} samples "
            f"({m.clip_pct:.2f}%)",
            "",
            "reasons:",
        ]
        lines += [f"  • {r}" for r in res.reasons]
        self.metrics_text.setPlainText("\n".join(lines))

    # -- Actions ------------------------------------------------------------ #
    def open_folder(self):
        path = self._selected_path()
        if path:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(os.path.dirname(path)))

    def convert_selected(self):
        path = self._selected_path()
        if not path:
            return
        res = self.results.get(path)
        if res and not res.convertible:
            answer = QtWidgets.QMessageBox.question(
                self, "Not a clean source",
                f"This file is flagged “{res.verdict.label}”"
                + (f" ({res.suspected_source})" if res.suspected_source else "")
                + ". Encoding it to MP3 will not give you a true "
                  f"{self.quality_combo.currentText()} copy.\n\nConvert anyway?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output folder for MP3 (library layout is mirrored)",
            os.path.dirname(path))
        if not out:
            return
        self.btn_convert.setEnabled(False)
        self.status.setText("Converting…")
        self.convert_worker = ConvertWorker(path, out, self.quality_combo.currentText(),
                                            rel_to=self._effective_root())
        self.convert_worker.done.connect(self._convert_done)
        self.convert_worker.start()

    def _convert_done(self, cr):
        self.btn_convert.setEnabled(True)
        if cr.ok:
            self.status.setText(f"Wrote {os.path.basename(cr.output)} — {cr.message}")
        else:
            self.status.setText(f"Convert failed: {cr.message}")
            QtWidgets.QMessageBox.warning(self, "Conversion failed", cr.message)

    # -- Bulk conversion ---------------------------------------------------- #
    def bulk_convert_folder(self):
        if self.bulk_worker and self.bulk_worker.isRunning():
            self.bulk_worker.stop()
            self.status.setText("Stopping…")
            return
        if self.scan_worker and self.scan_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Scan running",
                "Stop the scan before starting a bulk conversion.")
            return

        files = gather_files([self.scan_dir], self.recursive_cb.isChecked(),
                             self.library_root)
        if not self.lossy_cb.isChecked():
            files = filter_lossless(files)
        if not files:
            self.status.setText("No source files found in folder.")
            return

        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output folder for MP3s (library layout is mirrored)",
            self.out_dir or os.path.expanduser("~"))
        if not out:
            return
        err = check_out_dir(out, [self.scan_dir])
        if err:
            QtWidgets.QMessageBox.warning(self, "Bad output folder", err)
            return
        self.out_dir = out

        quality = self.quality_combo.currentText()
        answer = QtWidgets.QMessageBox.question(
            self, "Bulk convert",
            f"Convert {len(files)} file(s) to MP3 {quality} into\n{out}?\n\n"
            "Your originals are not modified, and anything already converted "
            "there is skipped.")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.status.setText(f"Converting {len(files)} file(s)…")
        self.btn_bulk.setText("Stop")
        self.btn_scan.setEnabled(False)

        jobs = os.cpu_count() or 2
        self.bulk_worker = BulkConvertWorker(files, out, quality, jobs)
        self.bulk_worker.progress.connect(self._bulk_progress)
        self.bulk_worker.finished_summary.connect(self._bulk_done)
        self.bulk_worker.start()

    def _bulk_progress(self, done, total, src):
        self.progress.setValue(done)
        self.status.setText(f"Converting {done}/{total} — {os.path.basename(src)}")

    def _bulk_done(self, summary):
        self.btn_bulk.setText("Convert folder to MP3…")
        self.btn_scan.setEnabled(True)
        head = "Stopped" if summary.stopped else "Bulk convert done"
        self.status.setText(
            f"{head} — {summary.converted} converted, {summary.skipped} skipped, "
            f"{summary.failed} failed.")
        if not summary.failures:
            return

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Some conversions failed")
        box.setText(f"{summary.failed} file(s) could not be converted.")
        box.setDetailedText("\n\n".join(f"{src}\n  {msg}"
                                        for src, msg in summary.failures))
        save = box.addButton("Save list…", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is save:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save failure list", os.path.join(self.out_dir, "failed.txt"))
            if path:
                with open(path, "w") as fh:
                    for src, msg in summary.failures:
                        fh.write(f"{src}\t{msg}\n")


DARK_QSS = """
QWidget { background:#1e1e1e; color:#ddd; }
QLineEdit, QPlainTextEdit, QComboBox { background:#2a2a2a; border:1px solid #444; }
QPushButton { background:#333; border:1px solid #555; padding:4px 10px; border-radius:3px; }
QPushButton:hover { background:#3d3d3d; }
QPushButton:disabled { color:#666; border-color:#333; }
QHeaderView::section { background:#2a2a2a; color:#ccc; border:0; padding:4px; }
QTableWidget { gridline-color:#333; selection-background-color:#3a4a5a; }
QProgressBar { border:1px solid #444; text-align:center; }
QProgressBar::chunk { background:#3a6ea5; }
"""


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
