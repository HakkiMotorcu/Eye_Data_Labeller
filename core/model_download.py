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
