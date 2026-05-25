# src/postprocess.py
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _binary_mask(binary_logits: np.ndarray, lane_probability_threshold: float = 0.8) -> np.ndarray:
    """
    Convert model binary output to a boolean mask using lane probability.

    binary_logits: (2, H, W)  – raw logits (class-0 = background, class-1 = lane)
    lane_probability_threshold:
        Minimum softmax probability required to mark a pixel as lane.
    Returns:       (H, W)     – bool array, True = lane pixel
    """
    logits = binary_logits - binary_logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
    return probs[1] >= lane_probability_threshold


def _find_horizon_row(mask: np.ndarray, min_pixels: int = 5) -> int:
    """
    Return the topmost row index that contains at least `min_pixels` lane pixels.
    This is the 'horizon' – the highest point where real lanes appear.
    If no such row exists, returns 0.
    """
    for row in range(mask.shape[0]):
        if mask[row].sum() >= min_pixels:
            return row
    return 0


def _filter_small_blobs(mask: np.ndarray, min_blob_pixels: int) -> np.ndarray:
    """Remove connected components smaller than min_blob_pixels."""
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    cleaned = np.zeros_like(mask_uint8)
    for label in range(1, num_labels):          # skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] >= min_blob_pixels:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Main postprocess entry point
# ─────────────────────────────────────────────────────────────────────────────

def my_postprocess(
    binary_logits: np.ndarray,
    embedding: np.ndarray,
    # ── binary mask settings ──────────────────────────────────────────────
    min_blob_pixels: int   = 30,
    lane_probability_threshold: float = 0.8,
    # ── horizon settings ──────────────────────────────────────────────────
    horizon_min_pixels: int = 5,
    horizon_padding: int    = 20,
    # ── DBSCAN settings ───────────────────────────────────────────────────
    eps: float              = 0.35,
    min_samples: int        = 20,
    # ── spatial weight in feature vector ─────────────────────────────────
    spatial_weight: float   = 0.5,
    # ── per-lane quality gate ─────────────────────────────────────────────
    min_pixels: int         = 100,
    # ── polynomial fitting ────────────────────────────────────────────────
    poly_degree: int        = 2,
) -> list:
    """
    Post-process raw model outputs into a list of lane polynomials.

    Parameters
    ----------
    binary_logits : np.ndarray  shape (2, H, W)
        Raw logits from the binary segmentation head.
    embedding : np.ndarray  shape (D, H, W)
        Instance embedding map from the embedding head.
    min_blob_pixels : int
        Minimum connected-component area to keep (noise filter).
    lane_probability_threshold : float
        Minimum lane-class softmax probability needed to keep a lane pixel.
    horizon_min_pixels : int
        A row needs ≥ this many lane pixels to count as the horizon.
    horizon_padding : int
        Extra rows to skip below the horizon (lane density is high there,
        which confuses DBSCAN).
    eps : float
        DBSCAN neighbourhood radius (in the normalised feature space).
    min_samples : int
        DBSCAN minimum cluster size.
    spatial_weight : float
        Weight applied to normalised (x, y) coordinates before they are
        concatenated to the embedding features.  Increase to make spatial
        position matter more during clustering.
    min_pixels : int
        Minimum number of pixels a cluster must contain to be kept.
    poly_degree : int
        Degree of the polynomial fitted to each lane cluster.

    Returns
    -------
    lanes : list of dict
        Each dict has keys:
          'poly'    – np.ndarray  polynomial coefficients (np.polyfit x=f(y))
          'y_min'   – int         lowest  y in this cluster (highest in image)
          'y_max'   – int         highest y in this cluster (lowest  in image)
          'pixels'  – (ys, xs)   raw pixel coords belonging to this lane
    """
    H, W = binary_logits.shape[1], binary_logits.shape[2]

    # ── 1. Binary mask ────────────────────────────────────────────────────
    mask = _binary_mask(binary_logits, lane_probability_threshold)          # (H, W) bool

    # ── 2. Remove small blobs (salt-and-pepper noise) ─────────────────────
    mask = _filter_small_blobs(mask, min_blob_pixels)

    # ── 3. Find horizon and apply padding ────────────────────────────────
    horizon_row = _find_horizon_row(mask, horizon_min_pixels)
    cutoff_row  = horizon_row + horizon_padding  # ignore everything above this

    # zero-out everything above (and including) the horizon + padding zone
    mask[:cutoff_row, :] = False

    # collect surviving lane pixels
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    # ── 4. Build feature matrix: embedding + spatial coords ───────────────
    #   embedding features: shape (D,) per pixel
    emb_features = embedding[:, ys, xs].T          # (N, D)

    #   normalised spatial coordinates, scaled by spatial_weight
    x_norm = (xs / (W - 1)).reshape(-1, 1) * spatial_weight
    y_norm = (ys / (H - 1)).reshape(-1, 1) * spatial_weight

    features = np.concatenate([emb_features, x_norm, y_norm], axis=1)  # (N, D+2)

    # ── 5. DBSCAN clustering ───────────────────────────────────────────────
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(features)

    # ── 6. Fit a polynomial to each valid cluster ─────────────────────────
    lanes = []
    unique_labels = set(labels) - {-1}            # -1 = noise

    for label in unique_labels:
        mask_cluster = labels == label
        cy = ys[mask_cluster]
        cx = xs[mask_cluster]

        if len(cy) < min_pixels:
            continue                               # too small — discard

        # fit x = poly(y)  (better than y=poly(x) for near-vertical lanes)
        coeffs = np.polyfit(cy, cx, poly_degree)

        lanes.append({
            'poly':   coeffs,
            'y_min':  int(cy.min()),
            'y_max':  int(cy.max()),
            'pixels': (cy, cx),
        })

    # ── 7. Sort lanes left-to-right by x at the bottom of the image ───────
    def _x_at_bottom(lane):
        return float(np.polyval(lane['poly'], lane['y_max']))

    lanes.sort(key=_x_at_bottom)

    return lanes





# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

# Distinct colours for up to 8 lanes (BGR for OpenCV, RGB for matplotlib/st)
_LANE_COLOURS_RGB = [
    (255,  80,  80),   # red
    ( 80, 200,  80),   # green
    ( 80, 120, 255),   # blue
    (255, 200,  50),   # yellow
    (200,  80, 255),   # purple
    ( 50, 220, 220),   # cyan
    (255, 140,  30),   # orange
    (180, 255,  80),   # lime
]


def draw_lanes(
    image_rgb: np.ndarray,
    lanes: list,
    thickness: int = 3,
    point_step: int = 4,
) -> np.ndarray:
    """
    Draw detected lane polynomials onto `image_rgb` (H×W×3, uint8, RGB).

    Parameters
    ----------
    image_rgb  : np.ndarray  RGB image (will not be modified in-place).
    lanes      : list        Output of my_postprocess().
    thickness  : int         Line thickness in pixels.
    point_step : int         Sample every N rows when drawing the curve.

    Returns
    -------
    Annotated RGB image (same dtype / shape as input).
    """
    canvas = image_rgb.copy()
    H, W   = canvas.shape[:2]

    for i, lane in enumerate(lanes):
        colour = _LANE_COLOURS_RGB[i % len(_LANE_COLOURS_RGB)]
        y_min  = lane['y_min']
        y_max  = lane['y_max']

        # clamp to image bounds
        y_min = max(0, y_min)
        y_max = min(H - 1, y_max)

        ys = np.arange(y_min, y_max + 1, point_step)
        if len(ys) < 2:
            continue

        xs = np.polyval(lane['poly'], ys).astype(int)

        # draw line segments between consecutive sampled points
        for j in range(len(ys) - 1):
            x0, y0 = int(np.clip(xs[j],     0, W - 1)), int(ys[j])
            x1, y1 = int(np.clip(xs[j + 1], 0, W - 1)), int(ys[j + 1])
            cv2.line(canvas, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)

    return canvas
