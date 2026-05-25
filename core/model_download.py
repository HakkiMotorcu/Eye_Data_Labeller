"""Lazy downloader for the fine-tuned SAM-HeLa checkpoint.

We don't bundle the 400 MB+ ``sam_hela/best.pt`` with the app — it's
fetched on first use from a URL the deployer configures (Hugging Face
repo, GitHub release asset, S3 bucket, or any plain HTTPS URL).

Configuration
-------------

The URL is resolved in this order:

  1. QSettings key ``model/sam_hela_url``  — set via the I/O settings
     dialog (future) or programmatically.
  2. Environment variable ``EYE_LABELLER_SAM_HELA_URL``.
  3. The constant ``DEFAULT_SAM_HELA_URL`` below, which deployers can
     set when packaging the app.

If none of these resolve, ``ensure_sam_hela_checkpoint`` raises a
``MissingModelURL`` and the SAM Service surfaces a friendly error.

Implementation notes
--------------------

* Downloads stream into a ``.part`` file and atomically rename on
  success (so a Ctrl+C never leaves a half-written checkpoint).
* If a Qt parent is provided, shows a modal ``QProgressDialog`` with a
  Cancel button; otherwise downloads silently with print progress.
* Optional SHA-256 verification — pass ``expected_sha256`` to refuse
  corrupted downloads.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from typing import Optional

# Empty by default — overwrite in your fork / build config to ship a
# default URL with your binaries. Hugging Face repos are the simplest
# host: a public dataset / model repo gives you a stable HTTPS URL.
DEFAULT_SAM_HELA_URL: str = ""

ENV_VAR = "EYE_LABELLER_SAM_HELA_URL"
SETTINGS_KEY = "model/sam_hela_url"

# A user-pointed local checkpoint file takes precedence over downloads.
# Useful while we don't have a public hosting URL yet — collaborators
# drop the file somewhere on their disk and we just read it in place.
LOCAL_PATH_ENV_VAR = "EYE_LABELLER_SAM_HELA_LOCAL_PATH"
LOCAL_PATH_SETTINGS_KEY = "model/sam_hela_local_path"


class MissingModelURL(RuntimeError):
    """Raised when no SAM-HeLa download URL is configured."""


def resolve_sam_hela_local_path() -> str:
    """Return a user-configured filesystem path to an existing
    sam_hela/best.pt, or ``""`` when nothing's set."""
    try:
        from PyQt6.QtCore import QSettings
        v = QSettings().value(LOCAL_PATH_SETTINGS_KEY, "")
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(LOCAL_PATH_ENV_VAR, "")


def resolve_sam_hela_url() -> str:
    """Return the configured SAM-HeLa URL, or ``""`` when nothing's set."""
    # 1) QSettings (only if Qt is available — keep this import optional
    #    so headless callers don't hit a PyQt dependency just to download).
    try:
        from PyQt6.QtCore import QSettings
        v = QSettings().value(SETTINGS_KEY, "")
        if v:
            return str(v)
    except Exception:
        pass
    # 2) Environment variable.
    v = os.environ.get(ENV_VAR, "")
    if v:
        return v
    # 3) Compile-time default.
    return DEFAULT_SAM_HELA_URL


def _sha256_of(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, target_path: str, *,
                   expected_sha256: Optional[str] = None,
                   qt_parent=None,
                   label: Optional[str] = None) -> str:
    """Stream *url* into *target_path* atomically.

    Returns the final path. Raises ``IOError`` (network), ``ValueError``
    (sha256 mismatch), or ``InterruptedError`` (user cancelled).
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)) or ".",
                exist_ok=True)
    tmp = target_path + ".part"
    label = label or os.path.basename(target_path)

    dialog = None
    if qt_parent is not None:
        try:
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt as _Qt
            dialog = QProgressDialog(
                f"Downloading {label}…\n{url}", "Cancel", 0, 100, qt_parent)
            dialog.setWindowTitle("Downloading SAM model")
            dialog.setWindowModality(_Qt.WindowModality.ApplicationModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            dialog.show()
        except Exception:
            dialog = None

    try:
        with urllib.request.urlopen(url) as resp, open(tmp, 'wb') as out:
            total = int(resp.headers.get('Content-Length') or 0)
            done = 0
            chunk = 1024 * 256  # 256 KB
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if dialog is not None:
                    if dialog.wasCanceled():
                        raise InterruptedError("User cancelled download.")
                    if total:
                        pct = int(done * 100 / total)
                        dialog.setValue(pct)
                        dialog.setLabelText(
                            f"Downloading {label}\n"
                            f"{done // 1024 // 1024} / "
                            f"{total // 1024 // 1024} MB")
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents()
                else:
                    if total and done % (chunk * 32) == 0:
                        print(f"  {done // 1024 // 1024} / "
                              f"{total // 1024 // 1024} MB")
        if expected_sha256:
            got = _sha256_of(tmp)
            if got.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SHA-256 mismatch for {target_path}: "
                    f"expected {expected_sha256[:12]}…, got {got[:12]}…")
        os.replace(tmp, target_path)
        return target_path
    except BaseException:
        # Clean up partial download on any failure / cancel.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    finally:
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass


def ensure_sam_hela_checkpoint(target_path: str, *,
                                qt_parent=None,
                                expected_sha256: Optional[str] = None) -> str:
    """Return ``target_path`` once a SAM-HeLa checkpoint exists there.

    If the file is already present, returns immediately. Otherwise
    resolves the configured URL (see module docstring) and downloads.
    Raises ``MissingModelURL`` when no URL is configured.
    """
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path
    url = resolve_sam_hela_url()
    if not url:
        raise MissingModelURL(
            "No SAM-HeLa checkpoint URL configured. Set the QSettings "
            f"key '{SETTINGS_KEY}' (via the I/O settings dialog), the "
            f"environment variable '{ENV_VAR}', or DEFAULT_SAM_HELA_URL "
            "in core/model_download.py.")
    return download_file(
        url, target_path,
        expected_sha256=expected_sha256,
        qt_parent=qt_parent,
        label="sam_hela/best.pt")


def prompt_for_local_path(parent=None) -> str:
    """Pop up a dialog asking the user where the SAM-HeLa best.pt lives.

    Accepts EITHER:
      - a .pt / .pth file directly (used as-is), OR
      - a folder (scanned for ``best.pt`` → ``best*.pt`` → ``*.pt``;
        if multiple matches, a follow-up picker lets the user choose).

    On success, persists the resolved file path to ``QSettings`` under
    ``LOCAL_PATH_SETTINGS_KEY`` so future launches find it without
    re-prompting. Returns the path. Returns ``""`` on cancel.

    Requires an active ``QApplication`` (i.e. there must be a running
    Qt event loop, since the dialog is modal). Returns ``""`` if Qt
    isn't available so headless callers can fall back to other paths.
    """
    try:
        from PyQt6.QtWidgets import (
            QApplication, QDialog, QDialogButtonBox, QFileDialog,
            QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
        )
        from PyQt6.QtCore import QSettings
    except Exception:
        return ""

    if QApplication.instance() is None:
        return ""

    import glob as _glob

    dlg = QDialog(parent)
    dlg.setWindowTitle("SAM-HeLa checkpoint needed")
    dlg.setModal(True)
    dlg.resize(520, 180)

    msg = QLabel(
        "<b>SAM needs the fine-tuned checkpoint (<code>best.pt</code>) "
        "to segment cells.</b><br><br>"
        "Pick the <b>file</b> directly, OR pick a <b>folder</b> "
        "that contains it. Your choice is remembered across launches "
        "— you only do this once."
    )
    msg.setWordWrap(True)

    btn_file = QPushButton("Pick file (best.pt)…")
    btn_folder = QPushButton("Pick folder…")
    btn_cancel = QPushButton("Cancel")
    btn_cancel.setAutoDefault(False)

    chosen: list[str] = [""]

    def _accept_path(path: str):
        chosen[0] = path
        dlg.accept()

    def pick_file():
        path, _ = QFileDialog.getOpenFileName(
            dlg, "Pick SAM-HeLa best.pt", "",
            "PyTorch checkpoint (*.pt *.pth);;All files (*)")
        if not path:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(dlg, "Not a file", f"Not a regular file:\n{path}")
            return
        _accept_path(path)

    def pick_folder():
        folder = QFileDialog.getExistingDirectory(
            dlg, "Pick a folder containing best.pt", "")
        if not folder:
            return
        # Search priority: exact `best.pt` → `best*.pt` → any `*.pt` / `*.pth`
        matches: list[str] = []
        for pattern in ("best.pt", "best*.pt", "*.pt", "*.pth"):
            matches = sorted(_glob.glob(os.path.join(folder, pattern)))
            if matches:
                break
        if not matches:
            QMessageBox.warning(
                dlg, "No checkpoint found",
                f"Couldn't find a .pt or .pth file in:\n{folder}\n\n"
                "Pick a different folder, or use 'Pick file' instead.")
            return
        if len(matches) == 1:
            _accept_path(matches[0])
            return
        # Multiple .pt files — let the user pick one explicitly.
        path, _ = QFileDialog.getOpenFileName(
            dlg, "Multiple checkpoints found — pick one", folder,
            "PyTorch checkpoint (*.pt *.pth)")
        if path:
            _accept_path(path)

    btn_file.clicked.connect(pick_file)
    btn_folder.clicked.connect(pick_folder)
    btn_cancel.clicked.connect(dlg.reject)

    layout = QVBoxLayout(dlg)
    layout.addWidget(msg)
    layout.addStretch(1)
    row = QHBoxLayout()
    row.addWidget(btn_file)
    row.addWidget(btn_folder)
    row.addStretch(1)
    row.addWidget(btn_cancel)
    layout.addLayout(row)

    if dlg.exec() == QDialog.DialogCode.Accepted and chosen[0]:
        # Persist for next launch + so the I/O Settings dialog shows it.
        QSettings().setValue(LOCAL_PATH_SETTINGS_KEY, chosen[0])
        QSettings().sync()
        return chosen[0]
    return ""
