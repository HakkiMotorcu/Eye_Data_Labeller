"""Model registry — named SAM checkpoints the user can add / edit / remove.

Stored as JSON in QSettings:
  - ``model/registry``    : a list of entries (see below)
  - ``model/active_tag``  : the tag of the model the app currently loads

Each registry entry::

    {"tag": str, "path": str, "base": str, "engine": "sam"}

- ``tag``    unique display name (what you see in the combo / registry).
- ``path``   unique local checkpoint file; "" means a registry-download
             variant (micro_sam fetches the base weights on first use).
- ``base``   the SAM architecture the checkpoint targets — micro_sam
             needs this to load a .pt (vit_b / vit_l / vit_h / vit_t).
- ``engine`` model family. Only "sam" is loadable today; the field is
             reserved so a future non-SAM engine can slot in without a
             schema change.

Uniqueness rules (enforced on add / edit): tags are unique, and
non-empty paths are unique — no two models may share a tag or a path.

Built-in micro_sam variants (no checkpoint) are always offered
alongside the user's registered models, so the combo is never empty.
"""

import json
import os

REGISTRY_KEY = 'model/registry'
ACTIVE_KEY = 'model/active_tag'
# Legacy single-checkpoint key (pre-registry). Migrated on first load.
LEGACY_LOCAL_KEY = 'model/sam_hela_local_path'

ENGINE_SAM = 'sam'

# SAM architectures a fine-tuned checkpoint can target.
SAM_BASES = ('vit_b', 'vit_l', 'vit_h', 'vit_t')

# Always-available registry-download variants (no local checkpoint).
BUILTINS = (
    {'tag': 'vit_b_lm (light microscopy)', 'path': '', 'base': 'vit_b_lm',
     'engine': ENGINE_SAM, 'builtin': True},
    {'tag': 'vit_t (mobile, fastest)', 'path': '', 'base': 'vit_t',
     'engine': ENGINE_SAM, 'builtin': True},
    {'tag': 'vit_b (SAM base)', 'path': '', 'base': 'vit_b',
     'engine': ENGINE_SAM, 'builtin': True},
    {'tag': 'vit_l (SAM large)', 'path': '', 'base': 'vit_l',
     'engine': ENGINE_SAM, 'builtin': True},
)


def _qsettings():
    from PyQt6.QtCore import QSettings
    return QSettings()


def load_registry():
    """User-registered models (list of dicts). Never raises."""
    raw = _qsettings().value(REGISTRY_KEY, '')
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out = []
        for e in data:
            if not isinstance(e, dict) or not e.get('tag'):
                continue
            out.append({
                'tag': str(e['tag']),
                'path': str(e.get('path', '')),
                'base': str(e.get('base', 'vit_b')),
                'engine': str(e.get('engine', ENGINE_SAM)),
                'builtin': False,
            })
        return out
    except (ValueError, TypeError):
        return []


def save_registry(entries):
    s = _qsettings()
    slim = [{'tag': e['tag'], 'path': e.get('path', ''),
             'base': e.get('base', 'vit_b'),
             'engine': e.get('engine', ENGINE_SAM)}
            for e in entries if not e.get('builtin')]
    s.setValue(REGISTRY_KEY, json.dumps(slim))
    s.sync()


def all_models():
    """Registered models first (what the user cares about), then the
    built-in download variants."""
    return load_registry() + [dict(b) for b in BUILTINS]


def _find(models, tag):
    return next((m for m in models if m['tag'] == tag), None)


def validate(tag, path, *, exclude_tag=None, require_existing_path=True):
    """Return an error string, or None when (tag, path) is acceptable.

    ``exclude_tag`` skips one existing entry (for edits). Tags must be
    unique; non-empty paths must be unique and (optionally) exist.
    """
    tag = (tag or '').strip()
    path = (path or '').strip()
    if not tag:
        return "Give the model a tag (a short name)."
    others = [m for m in load_registry() if m['tag'] != exclude_tag]
    builtins = list(BUILTINS)
    if any(m['tag'] == tag for m in others + builtins):
        return f"A model called '{tag}' already exists — tags must be unique."
    if path:
        if any(m.get('path') == path for m in others):
            return "That checkpoint path is already registered under " \
                   "another tag — paths must be unique."
        if require_existing_path and not os.path.isfile(path):
            return f"No file at:\n{path}"
    return None


def add_model(tag, path, base, engine=ENGINE_SAM, require_existing_path=True):
    """Add and return the new entry. Raises ValueError on a rule break."""
    err = validate(tag, path, require_existing_path=require_existing_path)
    if err:
        raise ValueError(err)
    reg = load_registry()
    entry = {'tag': tag.strip(), 'path': (path or '').strip(),
             'base': base, 'engine': engine}
    reg.append(entry)
    save_registry(reg)
    return entry


def update_model(old_tag, tag, path, base, engine=ENGINE_SAM,
                 require_existing_path=True):
    err = validate(tag, path, exclude_tag=old_tag,
                   require_existing_path=require_existing_path)
    if err:
        raise ValueError(err)
    reg = load_registry()
    for e in reg:
        if e['tag'] == old_tag:
            e.update({'tag': tag.strip(), 'path': (path or '').strip(),
                      'base': base, 'engine': engine})
            break
    else:
        raise ValueError(f"No registered model named '{old_tag}'.")
    save_registry(reg)
    # keep the active pointer valid if we renamed the active model
    if get_active_tag() == old_tag:
        set_active(tag.strip())


def remove_model(tag):
    reg = [e for e in load_registry() if e['tag'] != tag]
    save_registry(reg)
    if get_active_tag() == tag:
        # fall back to the first remaining model (registered or built-in)
        models = all_models()
        set_active(models[0]['tag'] if models else '')


def get_active_tag():
    return str(_qsettings().value(ACTIVE_KEY, '') or '')


def set_active(tag):
    s = _qsettings()
    s.setValue(ACTIVE_KEY, tag)
    s.sync()


def get_active():
    """The active model entry, resolved against registry + built-ins.
    Falls back to the first available model when the pointer is stale."""
    models = all_models()
    entry = _find(models, get_active_tag())
    if entry is not None:
        return entry
    return models[0] if models else dict(BUILTINS[2])  # vit_b base


def _legacy_checkpoint_candidates():
    """Every place the pre-registry app looked for sam_hela/best.pt —
    the documented locations installers, the env var, and the download
    step still target. Order matters (explicit config wins)."""
    from core import model_download
    from core import app_paths
    return [
        # QSettings key (written by deploy/configure_model.py and the
        # old in-app picker) OR the env var — resolve handles both.
        model_download.resolve_sam_hela_local_path(),
        # The in-project "magic folder" documented in
        # models/checkpoints/sam_hela/README.md and INSTALL.md.
        os.path.join(app_paths.bundled_root(),
                     'models', 'checkpoints', 'sam_hela', 'best.pt'),
        # Where the URL downloader writes for bundled installs.
        app_paths.default_sam_hela_checkpoint_path(),
    ]


def ensure_migrated():
    """One-time adoption of pre-registry checkpoint configuration.

    With no registry yet, look everywhere the old app looked — the
    QSettings key / env var, the in-project magic folder, the per-user
    download location — and register the first hit as 'sam_hela'
    (active). A configured-but-missing path still migrates, so the app
    can show "checkpoint missing" instead of silently ignoring it.
    """
    if load_registry():
        if not get_active_tag():
            set_active(all_models()[0]['tag'])
        return
    candidates = _legacy_checkpoint_candidates()
    legacy = next((p for p in candidates if p and os.path.exists(p)), '')
    if not legacy:
        # Nothing on disk — but an explicitly configured path (settings
        # or env var) is still worth registering so the UI can say the
        # checkpoint is missing rather than pretending nothing was set.
        legacy = candidates[0]
    if legacy:
        try:
            add_model('sam_hela', legacy, 'vit_b',
                      require_existing_path=False)
            set_active('sam_hela')
            return
        except ValueError:
            pass
    # No custom model — default the active pointer to a built-in so the
    # combo has a valid selection.
    set_active(BUILTINS[2]['tag'])  # vit_b (SAM base)
