import os
import numpy as np
import cv2


class VideoData:
    """Load a single-channel video (AVI) as a (T, H, W) array."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.frames = None      # (T, H, W) numpy array
        self.num_frames = 0
        self.height = 0
        self.width = 0
        self.load_data()

    def load_data(self):
        print(f"Loading video: {self.filepath}...")
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.filepath}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames found in: {self.filepath}")

        self.frames = np.stack(frames, axis=0)
        self.num_frames, self.height, self.width = self.frames.shape
        print(f"Video loaded — {self.num_frames} frames, "
              f"{self.width}x{self.height}, dtype: {self.frames.dtype}")

    def get_frame(self, idx):
        """Return frame at index idx as (H, W) array."""
        idx = max(0, min(idx, self.num_frames - 1))
        return self.frames[idx]


class SegmentationData:
    """Load instance segmentation maps from an AVI video.

    Converts each frame to grayscale, finds the background intensity
    (most frequent value), then extracts foreground blobs via thresholding
    + connected components.  Each connected component is identified by
    the *median intensity* of its pixels (quantised), so the same cell
    keeps the same ID and colour across frames.
    """

    MEDIAN_K = 5          # median-blur kernel size
    BG_THRESHOLD = 5      # |pixel − background| must exceed this
    MIN_INSTANCE_PX = 30  # minimum pixels for a valid instance
    MORPH_CLOSE_ITER = 2  # morphological close iterations
    STAIR_QUANT = 4       # quantise median intensity to multiples of this
    MIN_FILL_RATIO = 0.10 # reject blobs whose pixels / bbox area < this
    BORDER_MARGIN = 3     # pixels from image edge to count as "touching"

    # Three independent semantic layers. A pixel can belong to one
    # instance per layer — so a cell painted on a vessel keeps both.
    CLASS_TYPES = ('cell', 'vessel', 'capillary')

    def __init__(self, filepath):
        self.filepath = filepath
        # Per-class (T, H, W) instance masks and per-class color tables.
        # See get_layer() / get_colors(); legacy callers can still use
        # the .masks / .instance_colors aliases below, which point at the
        # cell layer (the historical default).
        self._layers = {ct: None for ct in self.CLASS_TYPES}
        self._colors = {ct: {} for ct in self.CLASS_TYPES}
        self.num_frames = 0
        self.height = 0
        self.width = 0
        self.load_data()

    # ----- Backward-compat aliases ------------------------------------
    # Legacy code addresses cells via .masks / .instance_colors directly.
    # Keep these as live properties so existing snapshots/restores work
    # unchanged. The vessel/capillary layers are addressed via the new
    # get_layer / get_colors API.
    @property
    def masks(self):
        return self._layers['cell']

    @masks.setter
    def masks(self, value):
        self._layers['cell'] = value

    @property
    def instance_colors(self):
        return self._colors['cell']

    @instance_colors.setter
    def instance_colors(self, value):
        self._colors['cell'] = value

    def get_layer(self, class_type='cell'):
        """Return the (T, H, W) mask array for the requested class."""
        return self._layers[class_type]

    def set_layer(self, class_type, arr):
        self._layers[class_type] = arr

    def get_colors(self, class_type='cell'):
        """Return the {instance_id: (r,g,b)} color dict for a class."""
        return self._colors[class_type]

    def _ensure_layer(self, class_type):
        """Allocate an empty layer on demand (used by paint/erase when the
        user starts annotating a class with no existing pixels)."""
        if self._layers[class_type] is None:
            self._layers[class_type] = np.zeros(
                (self.num_frames, self.height, self.width), dtype=np.int32)
        return self._layers[class_type]

    def load_data(self):
        print(f"Loading segmentation maps: {self.filepath}...")
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open segmentation file: {self.filepath}")

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            raise ValueError(f"No frames in segmentation file: {self.filepath}")

        T = len(raw_frames)
        H, W = raw_frames[0].shape
        self.num_frames = T
        self.height = H
        self.width = W

        # First pass — extract instances using intensity stairs.
        # An AVI-loaded seg file always populates the cell layer;
        # vessels/capillaries are user-painted later (empty here).
        self._layers['cell']      = np.zeros((T, H, W), dtype=np.int32)
        self._layers['vessel']    = np.zeros((T, H, W), dtype=np.int32)
        self._layers['capillary'] = np.zeros((T, H, W), dtype=np.int32)
        all_stairs = set()
        for t in range(T):
            mask = self._extract_instances(raw_frames[t])
            self._layers['cell'][t] = mask
            ids = np.unique(mask)
            all_stairs.update(int(x) for x in ids if x != 0)

        # Assign stable colours keyed by stair value
        self._assign_colors(sorted(all_stairs))

        print(f"Segmentation loaded — {self.num_frames} frames, "
              f"{len(all_stairs)} unique stairs, dtype: {self.masks.dtype}")

    def _extract_instances(self, gray):
        """Extract instances from a single grayscale frame.

        Returns (H, W) int32 mask where each pixel's value is the
        quantised median intensity of its connected component
        (0 = background).

        Filtering:
        - Remove components smaller than MIN_INSTANCE_PX
        - Remove components with fill ratio (pixels / bbox area) below
          MIN_FILL_RATIO — these are scattered noise, not compact blobs
        - Remove components that touch 3+ image edges (border artifacts)
        - Skip components whose stair value is too close to background
        """
        H, W = gray.shape
        filt = cv2.medianBlur(gray, self.MEDIAN_K)

        # Find background as the most frequent value
        vals, counts = np.unique(filt, return_counts=True)
        bg_val = int(vals[np.argmax(counts)])

        # Foreground = pixels far from background
        diff = np.abs(filt.astype(np.int16) - bg_val)
        fg = (diff > self.BG_THRESHOLD).astype(np.uint8) * 255

        # Morphological close to fill small gaps, then open to remove
        # tiny noise bridges between components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel,
                              iterations=self.MORPH_CLOSE_ITER)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        # Connected components
        n_labels, labels = cv2.connectedComponents(fg)

        # Use median intensity of each component as stair ID
        q = self.STAIR_QUANT
        m = self.BORDER_MARGIN
        mask = np.zeros((H, W), dtype=np.int32)

        for c in range(1, n_labels):
            px = (labels == c)
            n_px = np.count_nonzero(px)

            # --- size filter ---
            if n_px < self.MIN_INSTANCE_PX:
                continue

            # --- fill-ratio filter ---
            ys, xs = np.where(px)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
            if bbox_area > 0 and (n_px / bbox_area) < self.MIN_FILL_RATIO:
                continue

            # --- border-touching filter ---
            n_edges = ((y0 <= m) + (y1 >= H - 1 - m) +
                       (x0 <= m) + (x1 >= W - 1 - m))
            if n_edges >= 3:
                continue

            # --- stair assignment ---
            med = int(np.median(filt[px]))
            stair = round(med / q) * q
            if abs(stair - bg_val) <= self.BG_THRESHOLD:
                continue

            mask[px] = stair

        return mask

    def _assign_colors(self, stairs):
        """Assign a distinct colour to each stair value."""
        palette = [
            (255, 80, 80),   (80, 255, 80),   (80, 160, 255),  (255, 255, 80),
            (255, 80, 255),  (80, 255, 255),  (255, 160, 80),  (160, 80, 255),
            (80, 255, 160),  (255, 128, 128), (128, 255, 128), (128, 128, 255),
            (220, 220, 80),  (220, 80, 220),  (80, 220, 220),  (255, 200, 120),
            (120, 200, 255), (200, 255, 120), (240, 128, 180), (128, 240, 180),
        ]
        self.instance_colors = {}
        for i, stair in enumerate(stairs):
            self.instance_colors[stair] = palette[i % len(palette)]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def get_mask(self, idx, class_type='cell'):
        """Return integer instance-ID mask at frame *idx* — shape (H, W)."""
        idx = max(0, min(idx, self.num_frames - 1))
        return self._ensure_layer(class_type)[idx]

    def set_mask(self, idx, mask, class_type='cell'):
        """Replace segmentation mask at frame idx for the given class."""
        idx = max(0, min(idx, self.num_frames - 1))
        self._ensure_layer(class_type)[idx] = mask

    def get_instance_ids(self, frame_idx, class_type='cell'):
        """Get unique instance IDs in a specific frame (excluding background=0)."""
        layer = self._ensure_layer(class_type)
        mask = layer[frame_idx]
        ids = np.unique(mask)
        return ids[ids != 0]

    def get_all_instance_ids(self, class_type='cell'):
        """Get all unique instance IDs across all frames (excluding background=0)."""
        layer = self._ensure_layer(class_type)
        ids = np.unique(layer)
        return ids[ids != 0]

    def get_instance_bbox(self, frame_idx, instance_id, class_type='cell'):
        """Get bounding box for an instance in a specific frame.
        Returns (x0, y0, w, h) or None if instance not present."""
        mask = self._ensure_layer(class_type)[frame_idx]
        ys, xs = np.where(mask == instance_id)
        if len(ys) == 0:
            return None
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    def get_all_bboxes(self, frame_idx):
        """Get bounding boxes for all blobs in a frame.

        Because a single stair may contain multiple disconnected blobs,
        this runs connected components *per stair* and returns one entry
        per blob.  Keys are ``(stair_id, blob_index)`` tuples.

        Returns dict {(stair_id, blob_idx): (x0, y0, w, h)}.
        """
        mask = self.masks[frame_idx]
        ids = np.unique(mask)
        ids = ids[ids != 0]
        bboxes = {}
        for iid in ids:
            stair_mask = (mask == iid).astype(np.uint8)
            n_labels, labels = cv2.connectedComponents(stair_mask)
            blob_idx = 0
            for c in range(1, n_labels):
                cys, cxs = np.where(labels == c)
                if len(cys) < self.MIN_INSTANCE_PX:
                    continue
                x0, y0 = int(cxs.min()), int(cys.min())
                x1, y1 = int(cxs.max()), int(cys.max())
                bboxes[(int(iid), blob_idx)] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
                blob_idx += 1
        return bboxes

    def delete_instance(self, instance_id, frame_idx=None, class_type='cell'):
        """Remove an instance (set to 0). If frame_idx is None, remove from all frames."""
        layer = self._ensure_layer(class_type)
        if frame_idx is not None:
            layer[frame_idx][layer[frame_idx] == instance_id] = 0
        else:
            layer[layer == instance_id] = 0

    def erase_bbox(self, frame_idx, instance_id, x0, y0, w, h, class_type='cell'):
        """Set mask pixels of *instance_id* inside a bbox to 0.
        Only erases pixels that match the given instance_id."""
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape
        r0 = max(0, int(y0))
        r1 = min(H, int(y0 + h))
        c0 = max(0, int(x0))
        c1 = min(W, int(x0 + w))
        region = mask[r0:r1, c0:c1]
        region[region == instance_id] = 0

    def move_instance_pixels(self, frame_idx, instance_id, old_bbox, new_bbox,
                              class_type='cell'):
        """Translate mask pixels of *instance_id* inside *old_bbox* by the
        displacement between old_bbox and new_bbox.

        old_bbox / new_bbox: (x, y, w, h) in mask coordinates.
        """
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape

        ox, oy, ow, oh = [int(round(v)) for v in old_bbox]
        nx, ny, nw, nh = [int(round(v)) for v in new_bbox]

        # Clamp source region
        sr0, sr1 = max(0, oy), min(H, oy + oh)
        sc0, sc1 = max(0, ox), min(W, ox + ow)

        # Extract the pixels belonging to this instance inside the old bbox
        region = mask[sr0:sr1, sc0:sc1].copy()
        inst_mask = (region == instance_id)
        if not np.any(inst_mask):
            return

        # Erase old pixels
        mask[sr0:sr1, sc0:sc1][inst_mask] = 0

        # Compute displacement
        dx = nx - ox
        dy = ny - oy

        # Get source coordinates (relative to region) and shift
        ry, rx = np.where(inst_mask)
        ty = sr0 + ry + dy
        tx = sc0 + rx + dx

        # Keep only those that land inside the image
        valid = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        mask[ty[valid], tx[valid]] = instance_id

    def resize_instance_pixels(self, frame_idx, instance_id, old_bbox, new_bbox,
                                class_type='cell'):
        """Rescale mask pixels of *instance_id* from old_bbox into new_bbox."""
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape

        ox, oy, ow, oh = [int(round(v)) for v in old_bbox]
        nx, ny, nw, nh = [int(round(v)) for v in new_bbox]

        sr0, sr1 = max(0, oy), min(H, oy + oh)
        sc0, sc1 = max(0, ox), min(W, ox + ow)

        region = mask[sr0:sr1, sc0:sc1].copy()
        inst_mask = (region == instance_id).astype(np.uint8)
        if not np.any(inst_mask):
            return

        # Erase old
        mask[sr0:sr1, sc0:sc1][region == instance_id] = 0

        # Resize the binary mask to new dimensions
        if nw < 1 or nh < 1:
            return
        resized = cv2.resize(inst_mask, (max(1, nw), max(1, nh)),
                             interpolation=cv2.INTER_NEAREST)

        # Place at new position
        dr0, dr1 = max(0, ny), min(H, ny + nh)
        dc0, dc1 = max(0, nx), min(W, nx + nw)
        # Offsets into resized array if new bbox extends above/left of image
        ro = max(0, -ny)
        co = max(0, -nx)
        rh = dr1 - dr0
        rw = dc1 - dc0
        patch = resized[ro:ro + rh, co:co + rw]
        mask[dr0:dr1, dc0:dc1][patch > 0] = instance_id

    # ------------------------------------------------------------------
    #  Brush / editing operations
    # ------------------------------------------------------------------
    def paint_circle(self, frame_idx, instance_id, cx, cy, radius, force=False,
                      class_type='cell'):
        """Paint a filled circle of *instance_id* at (cx, cy) in mask coords.

        Writes to the per-class layer addressed by *class_type*. The
        force/safe semantics only consider other instances in the *same*
        layer — cells never erase vessels and vice versa.
        """
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape
        r = int(round(radius))
        y0 = max(0, int(cy) - r)
        y1 = min(H, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(W, int(cx) + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        circle = dist2 <= radius ** 2
        region = mask[y0:y1, x0:x1]
        if force:
            region[circle] = instance_id
        else:
            # Safe mode: only paint on background or own pixels
            paintable = circle & ((region == 0) | (region == instance_id))
            region[paintable] = instance_id

    def erase_circle(self, frame_idx, instance_id, cx, cy, radius,
                      class_type='cell'):
        """Erase pixels of *instance_id* inside a circle at (cx, cy)."""
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape
        r = int(round(radius))
        y0 = max(0, int(cy) - r)
        y1 = min(H, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(W, int(cx) + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        circle = dist2 <= radius ** 2
        region = mask[y0:y1, x0:x1]
        region[(region == instance_id) & circle] = 0

    def fill_bbox(self, frame_idx, instance_id, x0, y0, w, h, force=False,
                   class_type='cell'):
        """Fill an entire bbox region with *instance_id*.

        Operates on the per-class layer. Same force/safe semantics as
        paint_circle — only the active class's layer is affected.
        """
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape
        r0 = max(0, int(round(y0)))
        r1 = min(H, int(round(y0 + h)))
        c0 = max(0, int(round(x0)))
        c1 = min(W, int(round(x0 + w)))
        region = mask[r0:r1, c0:c1]
        if force:
            region[:] = instance_id
        else:
            # Safe mode: only fill background or own pixels
            region[(region == 0) | (region == instance_id)] = instance_id

    @classmethod
    def empty(cls, width, height, num_frames):
        """Construct an empty SegmentationData of the given shape.

        Used when the user starts labeling on a raw video/image with no
        pre-existing segmentation file — paint/erase need a (T, H, W) int
        mask layer to write into.
        """
        seg = cls.__new__(cls)
        seg.filepath = None
        seg.num_frames = int(num_frames)
        seg.height = int(height)
        seg.width = int(width)
        shape = (seg.num_frames, seg.height, seg.width)
        seg._layers = {
            'cell':      np.zeros(shape, dtype=np.int32),
            'vessel':    np.zeros(shape, dtype=np.int32),
            'capillary': np.zeros(shape, dtype=np.int32),
        }
        seg._colors = {'cell': {}, 'vessel': {}, 'capillary': {}}
        return seg

    def next_instance_id(self, class_type='cell'):
        """Return an unused instance ID for the given class layer."""
        layer = self._ensure_layer(class_type)
        colors = self._colors[class_type]
        max_mask = int(layer.max()) if layer is not None and layer.size > 0 else 0
        max_reg = max(colors.keys()) if colors else 0
        max_id = max(max_mask, max_reg)
        q = self.STAIR_QUANT
        return (max_id // q + 1) * q

    _PALETTE = [
        (255, 80, 80),   (80, 255, 80),   (80, 160, 255),  (255, 255, 80),
        (255, 80, 255),  (80, 255, 255),  (255, 160, 80),  (160, 80, 255),
        (80, 255, 160),  (255, 128, 128), (128, 255, 128), (128, 128, 255),
        (220, 220, 80),  (220, 80, 220),  (80, 220, 220),  (255, 200, 120),
        (120, 200, 255), (200, 255, 120), (240, 128, 180), (128, 240, 180),
    ]
    # Vessels render in shades of purple; capillaries in shades of pink/
    # magenta — matches the spawn-button color scheme.
    _PALETTE_VESSEL = [
        (147, 112, 219), (138,  43, 226), (153,  50, 204), (123, 104, 238),
        (186,  85, 211), (160,  82, 222), (148, 100, 200), (130,  70, 210),
    ]
    _PALETTE_CAPILLARY = [
        (235, 130, 200), (240, 100, 180), (220, 110, 170), (255, 145, 200),
        (230, 120, 175), (250, 130, 195), (215,  95, 165), (245, 120, 190),
    ]

    def register_instance_color(self, instance_id, color=None, class_type='cell'):
        """Register a color for a new instance ID in the given class layer."""
        colors = self._colors[class_type]
        if instance_id in colors:
            return colors[instance_id]
        if color is None:
            palette = {
                'cell': self._PALETTE,
                'vessel': self._PALETTE_VESSEL,
                'capillary': self._PALETTE_CAPILLARY,
            }[class_type]
            idx = len(colors) % len(palette)
            color = palette[idx]
        colors[instance_id] = color
        return color

    def crop_instance_to_bbox(self, frame_idx, instance_id, x, y, w, h,
                                class_type='cell'):
        """Erase pixels of *instance_id* that fall outside the given bbox.

        Used when a bbox is resized smaller — the painted pixels that no
        longer fit inside the new box are cropped away.
        Coords are in mask (seg) pixel space.
        """
        mask = self._ensure_layer(class_type)[frame_idx]
        H, W = mask.shape
        r0 = max(0, int(round(y)))
        r1 = min(H, int(round(y + h)))
        c0 = max(0, int(round(x)))
        c1 = min(W, int(round(x + w)))
        # Build a boolean map of pixels OUTSIDE the new bbox
        outside = np.ones((H, W), dtype=bool)
        if r1 > r0 and c1 > c0:
            outside[r0:r1, c0:c1] = False
        mask[(mask == instance_id) & outside] = 0

    def propagate_instance_mask(self, instance_id, source_frame, target_frames,
                                overwrite=False, class_type='cell'):
        """Copy *instance_id* pixels from *source_frame* into each *target_frame*.

        Parameters
        ----------
        overwrite : bool
            False (default) — skip target frames that already have any pixels
            for *instance_id* (preserves per-frame manual edits).
            True — clear the existing pixels for this instance in each target
            frame before copying (full overwrite; used when the reference
            frame was repainted).

        Returns the number of frames actually updated.
        """
        layer = self._ensure_layer(class_type)
        source_pixels = (layer[source_frame] == instance_id)
        updated = 0
        for fi in target_frames:
            if fi == source_frame:
                continue
            target = layer[fi]
            if not overwrite and np.any(target == instance_id):
                continue  # preserve manual edits on this frame
            # Clear old pixels of this instance so we don't leave ghosts
            target[target == instance_id] = 0
            # Copy from source — safe mode: only land on background pixels
            target[(source_pixels) & (target == 0)] = instance_id
            updated += 1
        return updated

    def save_masks(self, filepath, class_type='cell'):
        """Save the mask array for a single class as a grayscale AVI.

        The new canonical save path is the 3-TIF export in mask_io;
        this AVI helper is preserved for legacy callers only.
        """
        masks = self._ensure_layer(class_type)
        T, H, W = masks.shape
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # lossless
        writer = cv2.VideoWriter(filepath, fourcc, 30, (W, H), isColor=False)
        if not writer.isOpened():
            raise IOError(f"Cannot create video file: {filepath}")
        for t in range(T):
            frame = np.clip(masks[t], 0, 255).astype(np.uint8)
            writer.write(frame)
        writer.release()
        print(f"Segmentation masks saved to: {filepath}")