"""Shareable export bundle: mask TIFs + overlay video + per-instance CSV.

One folder your collaborators can open without the app:

    export/
      Cells.tif / Vessels.tif / Capillaries.tif   uint16 instance masks
      Overlay.mp4 (or .avi fallback)              annotations burned in
      Summary.csv                                 one row per instance
                                                  per frame (area, bbox)

The overlay video is the quality-control artifact — anyone can scrub
it and spot a bad mask without installing anything.

Qt-free by design: the UI passes a progress callback; returning False
from it aborts cleanly.
"""

import csv
import os

import numpy as np

CLASS_FILES = (('cell', 'Cells.tif'),
               ('vessel', 'Vessels.tif'),
               ('capillary', 'Capillaries.tif'))
# Overlay compositing order — bottom to top, cells on top.
CLASS_ORDER = ('vessel', 'capillary', 'cell')

_FALLBACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 120, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 255, 128), (255, 64, 64), (64, 255, 64), (64, 64, 255),
]


def _color_for(iid, colors):
    if iid in colors:
        return colors[iid]
    return _FALLBACK_COLORS[(iid - 1) % len(_FALLBACK_COLORS)]


def write_bundle(seg, names, out_dir, get_frame, num_frames,
                 fps=8.0, alpha=0.45, progress=None):
    """Write the full bundle into *out_dir*. Returns written paths.

    seg:        SegmentationData (layers + per-class color dicts)
    names:      {(frame_idx, class_type, instance_id): display_name}
    get_frame:  callable(i) -> (H, W) uint8 video frame
    progress:   callable(done, total) -> bool; False aborts (partial
                files are removed).
    """
    import cv2
    import tifffile

    os.makedirs(out_dir, exist_ok=True)
    written = []

    # ---- 1. Mask TIFs (snapshot copies — not the canonical save) ----
    for ct, fname in CLASS_FILES:
        layer = seg.get_layer(ct)
        if layer is None or not layer.any():
            continue
        p = os.path.join(out_dir, fname)
        tifffile.imwrite(p, layer.astype(np.uint16), compression='zlib')
        written.append(p)

    # ---- 2. Overlay video ------------------------------------------
    first = get_frame(0)
    vh, vw = first.shape[:2]
    video_path = os.path.join(out_dir, 'Overlay.mp4')
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (vw, vh))
    if not writer.isOpened():
        # mp4v unavailable in this OpenCV build — fall back to MJPG/AVI.
        video_path = os.path.join(out_dir, 'Overlay.avi')
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'MJPG'), float(fps), (vw, vh))
        if not writer.isOpened():
            raise IOError(
                "No usable video encoder (tried mp4v and MJPG) — "
                "overlay video cannot be written on this OpenCV build.")
    aborted = False
    try:
        for i in range(num_frames):
            if progress is not None and not progress(i, num_frames):
                aborted = True
                break
            frame = get_frame(i)
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frame[..., :3].copy()
            overlay = bgr.copy()
            for ct in CLASS_ORDER:
                layer = seg.get_layer(ct)
                if layer is None or i >= layer.shape[0]:
                    continue
                mask = layer[i]
                if mask.shape != (vh, vw):
                    mask = cv2.resize(mask, (vw, vh),
                                      interpolation=cv2.INTER_NEAREST)
                ids = np.unique(mask)
                ids = ids[ids != 0]
                if ids.size == 0:
                    continue
                colors = seg.get_colors(ct)
                lut = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
                for iid in ids:
                    r, g, b = _color_for(int(iid), colors)
                    lut[int(iid)] = (b, g, r)  # BGR for OpenCV
                colored = lut[mask]
                sel = mask != 0
                overlay[sel] = colored[sel]
            blended = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)
            writer.write(blended)
    finally:
        writer.release()
    if aborted:
        try:
            os.remove(video_path)
        except OSError:
            pass
        return written
    written.append(video_path)

    # ---- 3. Per-instance CSV ---------------------------------------
    csv_path = os.path.join(out_dir, 'Summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        wtr = csv.writer(f)
        wtr.writerow(['frame', 'class', 'instance_id', 'name',
                      'track_length', 'n_pixels', 'x0', 'y0', 'x1', 'y1'])
        for ct, _fname in CLASS_FILES:
            layer = seg.get_layer(ct)
            if layer is None or not layer.any():
                continue
            # track_length = how many frames each id appears on in this
            # class (post-tracking, a coalesced cell shares one id, so
            # this is its track length; untracked ids read as 1).
            frames_per_id = {}
            for fi in range(layer.shape[0]):
                for iid in np.unique(layer[fi]):
                    if iid:
                        frames_per_id[int(iid)] = \
                            frames_per_id.get(int(iid), 0) + 1
            for fi in range(layer.shape[0]):
                m = layer[fi]
                ids = np.unique(m)
                ids = ids[ids != 0]
                for iid in ids:
                    iid = int(iid)
                    ys, xs = np.nonzero(m == iid)
                    name = (names.get((fi, ct, iid))
                            or names.get((0, ct, iid), ''))
                    wtr.writerow([
                        fi, ct, iid, name, frames_per_id.get(iid, 1),
                        int(ys.size),
                        int(xs.min()), int(ys.min()),
                        int(xs.max()) + 1, int(ys.max()) + 1])
    written.append(csv_path)
    return written
