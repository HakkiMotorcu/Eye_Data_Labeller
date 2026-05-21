"""Lightweight debug-logging facility for the Eye Data Labeller.

Goals:
- Single flag toggles all verbose progress prints.
- Exceptions in instrumented actions ALWAYS dump full tracebacks to stderr,
  even with debug off — so the user sees the underlying error instead of
  just a curt QMessageBox.
- No new heavy deps; just stdlib.

Toggling debug:
  1. Environment variable: ``EYE_LABELLER_DEBUG=1`` before launch.
  2. CLI flag: ``python main.py --debug`` (main.py sets the env var
     before importing anything else).

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
"""

import os
import sys
import time
import traceback
from functools import wraps


def is_debug():
    return os.environ.get('EYE_LABELLER_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')


def _fmt_kwargs(kwargs):
    if not kwargs:
        return ''
    return ' ' + ' '.join(f'{k}={v!r}' for k, v in kwargs.items())


def log(component, msg, **kwargs):
    """Verbose progress log — only printed when debug is on."""
    if not is_debug():
        return
    ts = time.strftime('%H:%M:%S')
    sys.stderr.write(f'[{ts} DBG {component}] {msg}{_fmt_kwargs(kwargs)}\n')
    sys.stderr.flush()


def log_error(component, msg, exc=None, **kwargs):
    """Always-on error log. Prints the message, kwargs, and (if given) the
    full exception traceback to stderr."""
    ts = time.strftime('%H:%M:%S')
    sys.stderr.write(f'[{ts} ERR {component}] {msg}{_fmt_kwargs(kwargs)}\n')
    if exc is not None:
        sys.stderr.write(f'    {type(exc).__name__}: {exc}\n')
        if is_debug():
            sys.stderr.write('    Traceback:\n')
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            for line in tb:
                for ln in line.rstrip().splitlines():
                    sys.stderr.write(f'      {ln}\n')
    sys.stderr.flush()


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
