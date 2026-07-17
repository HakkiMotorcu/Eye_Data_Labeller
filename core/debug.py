"""Logging facility for the Eye Data Labeller.

Two sinks, one switch:

* **Log file** — always on. Errors are written to the session log file
  no matter what; verbose action logs are added when debug is enabled.
  Files live under ``user_data_root()/logs/`` (one per app session,
  oldest pruned) so a collaborator can send the file after a crash
  without having run anything special.
* **stderr** — mirrors the same lines when a console exists. Guarded:
  under ``pythonw.exe`` on Windows ``sys.stderr`` is ``None`` and
  writing would crash — we skip it there.

Toggling debug (any of these):
  1. In-app: I/O Settings → "Detailed logging". Applies immediately and
     persists via QSettings (``debug/verbose_logging``).
  2. CLI flag: ``python main.py --debug``.
  3. Environment variable: ``EYE_LABELLER_DEBUG=1`` before launch.

Usage:
    from core.debug import log, log_error, log_action

    log('sam', 'loading checkpoint', path=ckpt_path)

    @log_action('sam')
    def auto_segment(self, frame):
        ...

    try:
        ...
    except Exception as e:
        log_error('sam', 'auto_segment failed', exc=e)
        raise

Every line carries the thread name — threading bugs (like a dialog
opened from a worker) are visible right in the log.
"""

import os
import sys
import threading
import time
import traceback
from functools import wraps

# QSettings key for the in-app toggle (read at startup by main.py,
# written by the I/O settings dialog).
SETTING_DEBUG_KEY = 'debug/verbose_logging'

_KEEP_LOG_FILES = 20  # newest session logs kept; older ones pruned

# Runtime debug flag. Starts from the env var; set_debug() flips it
# live (the in-app toggle calls that).
_debug_enabled = os.environ.get(
    'EYE_LABELLER_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')

_log_file = None          # opened lazily on first write
_log_file_path = None
_log_lock = threading.Lock()


def is_debug():
    return _debug_enabled


def set_debug(on):
    """Flip verbose logging at runtime (the in-app toggle calls this)."""
    global _debug_enabled
    on = bool(on)
    if on == _debug_enabled:
        return
    _debug_enabled = on
    # Announce the flip in the file so a log reader can tell which
    # stretches were recorded verbose.
    _write_line(f'[{_ts()} LOG debug] verbose logging '
                f'{"ENABLED" if on else "DISABLED"}\n', force_file=True)


def log_dir():
    """Folder holding the session log files (created on demand)."""
    from core.app_paths import user_data_root
    d = os.path.join(user_data_root(), 'logs')
    os.makedirs(d, exist_ok=True)
    return d


def log_file_path():
    """Path of this session's log file (None until the first write)."""
    return _log_file_path


def _ts():
    return time.strftime('%H:%M:%S')


def _open_log_file():
    """Open this session's log file and prune old ones. Never raises."""
    global _log_file, _log_file_path
    try:
        d = log_dir()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(d, f'session_{stamp}_{os.getpid()}.log')
        _log_file = open(path, 'a', encoding='utf-8')
        _log_file_path = path
        # Prune oldest sessions beyond the keep limit.
        try:
            logs = sorted(
                f for f in os.listdir(d)
                if f.startswith('session_') and f.endswith('.log'))
            for old in logs[:-_KEEP_LOG_FILES]:
                try:
                    os.remove(os.path.join(d, old))
                except OSError:
                    pass
        except OSError:
            pass
    except Exception:
        # Logging must never take the app down — no file, no problem.
        _log_file = None
        _log_file_path = None


def _write_line(line, *, force_file=False):
    """Write one formatted line to stderr (if present) and the log file.

    Debug-only lines reach the file only while debug is on; error lines
    pass force_file=True so they are always captured.
    """
    if sys.stderr is not None:  # None under pythonw.exe — skip, don't crash
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
    if not (force_file or _debug_enabled):
        return
    with _log_lock:
        if _log_file is None:
            _open_log_file()
        if _log_file is not None:
            try:
                _log_file.write(line)
                _log_file.flush()
            except Exception:
                pass


def _fmt_kwargs(kwargs):
    if not kwargs:
        return ''
    return ' ' + ' '.join(f'{k}={v!r}' for k, v in kwargs.items())


def _thread_tag():
    name = threading.current_thread().name
    return '' if name == 'MainThread' else f' @{name}'


def log(component, msg, **kwargs):
    """Verbose progress log — emitted only when debug is on."""
    if not _debug_enabled:
        return
    _write_line(f'[{_ts()} DBG {component}{_thread_tag()}] '
                f'{msg}{_fmt_kwargs(kwargs)}\n')


def log_error(component, msg, exc=None, **kwargs):
    """Always-on error log. Message + kwargs + (if given) the exception,
    with a full traceback in the log file (stderr gets the traceback
    only in debug mode, as before)."""
    header = (f'[{_ts()} ERR {component}{_thread_tag()}] '
              f'{msg}{_fmt_kwargs(kwargs)}\n')
    lines = [header]
    if exc is not None:
        lines.append(f'    {type(exc).__name__}: {exc}\n')
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        for line in tb:
            for ln in line.rstrip().splitlines():
                lines.append(f'      {ln}\n')
    # stderr: keep the old behavior (traceback only when debugging);
    # file: always record everything.
    stderr_text = ''.join(lines if (_debug_enabled or exc is None)
                          else lines[:2])
    if sys.stderr is not None:
        try:
            sys.stderr.write(stderr_text)
            sys.stderr.flush()
        except Exception:
            pass
    with _log_lock:
        if _log_file is None:
            _open_log_file()
        if _log_file is not None:
            try:
                _log_file.write(''.join(lines))
                _log_file.flush()
            except Exception:
                pass


def log_action(component):
    """Decorator that brackets a method with debug-mode entry/exit logs.

    No-op when debug is off (zero-overhead path is one bool check).
    """
    def wrap(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            if is_debug():
                log(component, f'>>> {fn.__name__}')
                t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if is_debug():
                    dt_ms = (time.perf_counter() - t0) * 1000
                    log(component, f'<<< {fn.__name__} ({dt_ms:.1f} ms)')
        return inner
    return wrap


def log_startup_banner(extra=None):
    """Write a one-time environment header into the log file.

    Called by main.py after Qt identity is set. Always file-logged —
    a collaborator's very first log line answers 'which OS, which
    Python, which versions?' without a follow-up email.
    """
    import platform
    info = {
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'frozen': getattr(sys, 'frozen', False),
        'cwd': os.getcwd(),
        'argv': sys.argv,
    }
    for mod in ('PyQt6.QtCore', 'numpy', 'cv2', 'torch', 'tifffile'):
        try:
            m = __import__(mod, fromlist=['__name__'])
            ver = (getattr(m, 'QT_VERSION_STR', None)
                   or getattr(m, '__version__', '?'))
            info[mod.split('.')[0]] = ver
        except Exception as e:
            info[mod.split('.')[0]] = f'unavailable ({type(e).__name__})'
    if extra:
        info.update(extra)
    _write_line(f'[{_ts()} LOG startup] Eye Data Labeller session start '
                f'{_fmt_kwargs(info)}\n', force_file=True)


def install_qt_message_handler():
    """Route Qt's own warnings (qWarning etc.) into this log."""
    try:
        from PyQt6.QtCore import qInstallMessageHandler
    except Exception:
        return

    def _handler(mode, context, message):
        _write_line(f'[{_ts()} QT ] {message}\n', force_file=True)

    try:
        qInstallMessageHandler(_handler)
    except Exception:
        pass
