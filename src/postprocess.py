import numpy as np
from sklearn.cluster import DBSCAN


def get_binary_mask(binary_logits, threshold=None):
    """
    Convert model's binary logits to a (H, W) bool mask.

    binary_logits: numpy or torch tensor of shape (2, H, W) or (1, 2, H, W).
                   If torch, must already be on CPU and converted with .numpy().
    threshold:     if None, use argmax. Otherwise threshold the lane probability.
    """
    if binary_logits.ndim == 4:
        binary_logits = binary_logits[0]  # drop batch dim

    if threshold is None:
        return binary_logits.argmax(axis=0).astype(bool)
    else:
        # softmax channel 1 (lane class)
        e = np.exp(binary_logits - binary_logits.max(axis=0, keepdims=True))
        prob_lane = e[1] / e.sum(axis=0)
        return prob_lane > threshold


def cluster_embeddings(embedding, binary_mask, eps=1.5, min_samples=50):
    """
    Cluster lane pixels by their embedding vectors.

    embedding:   numpy array of shape (D, H, W).
    binary_mask: numpy bool array of shape (H, W).
    eps, min_samples: DBSCAN hyperparameters.

    Returns:
        labels: (N,) int array — cluster id per lane pixel, -1 = noise.
        ys, xs: (N,) int arrays — image coordinates of the lane pixels.
    """
    if embedding.ndim == 4:
        embedding = embedding[0]

    ys, xs = np.where(binary_mask)
    if len(ys) == 0:
        return np.array([], dtype=int), ys, xs

    features = embedding[:, ys, xs].T  # (N, D)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)
    return labels, ys, xs


def fit_lanes(labels, ys, xs, poly_degree=2, min_pixels=100):
    """
    Fit a polynomial x = f(y) per cluster.

    labels:      (N,) int — output of cluster_embeddings
    ys, xs:      (N,) int — pixel coords
    poly_degree: 2 or 3
    min_pixels:  reject clusters with fewer than this many pixels (noise filter)

    Returns: list of dicts, one per detected lane:
        {'cluster_id': int,
         'poly':       np.ndarray of polynomial coefficients,
         'y_range':    (y_min, y_max),
         'pixels':     (n, 2) array of (y, x) coords}
    """
    lanes = []
    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        if mask.sum() < min_pixels:
            continue
        cys, cxs = ys[mask], xs[mask]
        coeffs = np.polyfit(cys, cxs, deg=poly_degree)
        lanes.append({
            'cluster_id': int(cid),
            'poly':       coeffs,
            'y_range':    (int(cys.min()), int(cys.max())),
            'pixels':     np.stack([cys, cxs], axis=1),
        })

    # Sort lanes left-to-right by their average x position (useful for ego-lane logic)
    lanes.sort(key=lambda l: l['pixels'][:, 1].mean())
    return lanes


def postprocess(binary_logits, embedding,
                eps=1.5, min_samples=50,
                poly_degree=2, min_pixels=100,
                threshold=None):
    """
    Full pipeline: model outputs -> list of lane dicts.

    binary_logits, embedding: numpy arrays. Single-image (batch dim optional).
    """
    binary_mask = get_binary_mask(binary_logits, threshold=threshold)
    labels, ys, xs = cluster_embeddings(embedding, binary_mask,
                                        eps=eps, min_samples=min_samples)
    return fit_lanes(labels, ys, xs,
                     poly_degree=poly_degree, min_pixels=min_pixels)


def draw_lanes(image_rgb, lanes, color_palette=None, thickness=3):
    """
    Draw fitted polynomial curves on an image. Returns annotated copy.

    image_rgb: (H, W, 3) uint8.
    lanes:     output of postprocess().
    """
    import cv2
    if color_palette is None:
        color_palette = [
            (255,  56,  56),   # red
            ( 56, 255,  56),   # green
            ( 56,  56, 255),   # blue
            (255, 255,  56),   # yellow
            (255,  56, 255),   # magenta
            ( 56, 255, 255),   # cyan
        ]
    out = image_rgb.copy()
    for i, lane in enumerate(lanes):
        y_min, y_max = lane['y_range']
        ys = np.arange(y_min, y_max + 1)
        xs = np.polyval(lane['poly'], ys).astype(int)
        # Clip to image bounds
        h, w = out.shape[:2]
        valid = (xs >= 0) & (xs < w)
        ys, xs = ys[valid], xs[valid]
        pts = np.stack([xs, ys], axis=1)  # cv2 expects (x, y) pairs
        color = color_palette[i % len(color_palette)]
        for j in range(len(pts) - 1):
            cv2.line(out, tuple(pts[j]), tuple(pts[j + 1]), color, thickness)
    return out