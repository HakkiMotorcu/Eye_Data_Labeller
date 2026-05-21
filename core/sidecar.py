"""Sidecar JSON files persisted next to ``{video}_Masks.tif``.

Two roles, one schema:

  ``{video}_meta.json``     — per-instance metadata
  ``{video}_drafts.json``   — pending bbox prompts not yet committed to masks
                              (reserved for the SAM-prompt UX in Phase 4)

Schema (meta) v2:

    {
      "version": 2,
      "instances": {
        "<class_type>:<instance_id>": {
          "name": str,
          "class_type": "cell" | "vessel" | "capillary",
          "locked": bool,
          "notes": str
        },
        ...
      }
    }

v1 used ``"<instance_id>"`` alone as the key — fine while everything
shared a single instance namespace, but after the multi-class refactor
vessel iid=4 and capillary iid=4 collide on the same key and one wins.
v2 keys by ``"<class_type>:<instance_id>"``. The loader accepts both
formats: v1 entries fall back to the inline ``class_type`` field (or
``"cell"`` when missing) so older Meta.json files migrate cleanly.

Loading is lenient — unknown keys are ignored; missing fields fall back
to defaults. Saving writes a deterministic, sorted JSON for diff-ability.
"""

import json
import os

SCHEMA_VERSION = 2

# Composite key helpers — keep call-site code readable.
def _meta_key(class_type, iid):
    return f"{class_type}:{int(iid)}"


def meta_path_for(image_path):
    base, _ = os.path.splitext(image_path)
    return f"{base}_Meta.json"


def drafts_path_for(image_path):
    base, _ = os.path.splitext(image_path)
    return f"{base}_Drafts.json"


def collect_meta_from_annotations(annotations):
    """Build the meta dict keyed by (class_type, instance_id) from a list
    of Annotation2D.

    When the same (class_type, instance_id) appears on multiple frames
    (one identity, many frames), the metadata from the LOCKED instance
    wins; otherwise the first occurrence wins. Annotations with no
    instance_id are skipped (they're drafts).
    """
    by_id = {}
    locked_winners = set()
    for anno in annotations:
        iid = getattr(anno, 'instance_id', None)
        if iid is None:
            continue
        ct = getattr(anno, 'class_type', 'cell')
        key = _meta_key(ct, iid)
        record = {
            "name": anno.name,
            "class_type": ct,
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


def meta_lookup(meta, class_type, instance_id):
    """Look up a meta record by (class_type, instance_id), tolerating
    both v2 (`"cell:4"`) and v1 (`"4"`) key shapes.

    v1 records had no per-class scoping. We treat them as belonging to
    whichever class they list inline (defaulting to 'cell'), so an old
    record that was already tagged class_type='vessel' applies to the
    vessel layer's iid=4, not the cell layer's.
    """
    if meta is None:
        return None
    instances = meta.get('instances', {}) if isinstance(meta, dict) else {}
    key = _meta_key(class_type, instance_id)
    rec = instances.get(key)
    if rec is not None:
        return rec
    # v1 fallback: try the bare iid key, but only return it if the
    # record's inline class_type matches what the caller asked for.
    legacy = instances.get(str(int(instance_id)))
    if legacy is None:
        return None
    legacy_ct = legacy.get('class_type', 'cell')
    if legacy_ct == 'vein':
        legacy_ct = 'vessel'
    return legacy if legacy_ct == class_type else None


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
