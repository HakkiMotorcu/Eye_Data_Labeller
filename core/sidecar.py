"""Sidecar JSON files persisted next to ``{video}_Masks.tif``.

Two roles, one schema:

  ``{video}_meta.json``     — per-instance metadata
  ``{video}_drafts.json``   — pending bbox prompts not yet committed to masks
                              (reserved for the SAM-prompt UX in Phase 4)

Schema (meta):

    {
      "version": 1,
      "instances": {
        "<instance_id>": {
          "name": str,
          "class_type": "cell" | "vessel" | "capillary",
          "locked": bool,
          "notes": str
        },
        ...
      }
    }

Loading is lenient — unknown keys are ignored; missing fields fall back
to defaults. Saving writes a deterministic, sorted JSON for diff-ability.
"""

import json
import os

SCHEMA_VERSION = 1


def meta_path_for(image_path):
    base, _ = os.path.splitext(image_path)
    return f"{base}_Meta.json"


def drafts_path_for(image_path):
    base, _ = os.path.splitext(image_path)
    return f"{base}_Drafts.json"


def collect_meta_from_annotations(annotations):
    """Build the meta dict keyed by instance_id from a list of Annotation2D.

    When the same instance_id appears on multiple frames (one identity,
    many frames), the metadata from the LOCKED instance wins; otherwise
    the first occurrence wins. Annotations with no instance_id are
    skipped (they're drafts, not committed instances).
    """
    by_id = {}
    locked_winners = set()
    for anno in annotations:
        iid = getattr(anno, 'instance_id', None)
        if iid is None:
            continue
        key = str(int(iid))
        record = {
            "name": anno.name,
            "class_type": anno.class_type,
            "locked": bool(anno.is_locked),
            "notes": getattr(anno, 'notes', '') or '',
        }
        if key in by_id and not anno.is_locked:
            continue
        if key in by_id and key in locked_winners and not anno.is_locked:
            continue
        by_id[key] = record
        if anno.is_locked:
            locked_winners.add(key)
    return {"version": SCHEMA_VERSION, "instances": by_id}


def save_meta(meta, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_meta(path):
    """Return the meta dict, or None when the sidecar isn't there.

    Always normalizes the legacy ``class_type='vein'`` value to
    ``'vessel'`` so loaded annotations adopt the current taxonomy.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or 'instances' not in data:
        return None
    for rec in data['instances'].values():
        if rec.get('class_type') == 'vein':
            rec['class_type'] = 'vessel'
    return data
