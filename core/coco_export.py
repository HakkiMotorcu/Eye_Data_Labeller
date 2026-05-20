"""COCO instance-segmentation export.

Bundles:
  - one ``images[]`` entry per frame in the input image stack
  - one ``annotations[]`` entry per (frame, instance) with bbox + RLE mask
  - one ``categories[]`` entry per class_type present in the metadata

Uses pycocotools' MaskApi for RLE encoding — the canonical COCO format.

This is an export-only path; round-trip import is intentionally not part
of the deliverable. Anyone training a detector/segmenter on the labeled
data can read this JSON with the standard COCO tooling.
"""

import json
import os
import numpy as np


_CATEGORY_ORDER = ['cell', 'vessel', 'capillary']


def _rle_encode(binary_mask):
    """Encode a 2D binary mask as a COCO RLE dict ('counts' as ASCII str)."""
    from pycocotools import mask as maskapi
    rle = maskapi.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def _bbox_from_mask(binary_mask):
    """Return COCO-style ``[x, y, w, h]`` for a binary mask."""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def export_coco(image_path, seg_data, meta, output_path):
    """Write a COCO JSON describing every instance in ``seg_data.masks``.

    Parameters
    ----------
    image_path : str
        Source image path (used for image filename + frame naming).
    seg_data : SegmentationData
        Has ``.masks`` of shape ``(T, H, W)`` with uint instance labels.
    meta : dict | None
        Output of ``sidecar.collect_meta_from_annotations``. Used to look
        up class_type per instance_id (defaults to 'cell' when missing).
    output_path : str
        Where to write the JSON.
    """
    masks = seg_data.masks
    T, H, W = masks.shape

    meta_instances = (meta or {}).get('instances', {}) if meta else {}

    images = []
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    for t in range(T):
        images.append({
            "id": t,
            "file_name": f"{image_stem}_frame{t:04d}.png",
            "width": int(W),
            "height": int(H),
        })

    annotations = []
    used_classes = set()
    anno_id = 1
    for t in range(T):
        frame_mask = masks[t]
        ids = np.unique(frame_mask)
        for iid in ids:
            if iid == 0:
                continue
            binary = (frame_mask == iid)
            if not binary.any():
                continue
            category = meta_instances.get(str(int(iid)), {}).get('class_type', 'cell')
            used_classes.add(category)
            rle = _rle_encode(binary)
            annotations.append({
                "id": anno_id,
                "image_id": int(t),
                "category_id": _CATEGORY_ORDER.index(category) + 1,
                "bbox": _bbox_from_mask(binary),
                "area": int(binary.sum()),
                "iscrowd": 0,
                "segmentation": rle,
                "instance_id": int(iid),
            })
            anno_id += 1

    # Emit categories in stable order — only those actually used (plus 'cell'
    # always, so empty masks still produce a valid COCO file).
    used_classes.add('cell')
    categories = [
        {"id": _CATEGORY_ORDER.index(c) + 1, "name": c, "supercategory": "retinal"}
        for c in _CATEGORY_ORDER if c in used_classes
    ]

    coco = {
        "info": {
            "description": f"Eye Data Labeller export: {image_stem}",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=2)
    return len(annotations)
