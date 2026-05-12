import numpy as np
from sklearn.cluster import DBSCAN, MeanShift
import cv2

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def _l2_normalize(features, eps=1e-8):
    """
    Normalize each feature vector to unit length.

    features: (..., D)
    """
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, eps)


def _normalized_xy_features(ys, xs, height, width):
    """
    Return coordinate features in [0, 1] range as (N, 2), ordered as x, y.
    Normalizing keeps coordinate scale comparable across input resolutions.
    """
    x_norm = xs.astype(np.float32) / max(width - 1, 1)
    y_norm = ys.astype(np.float32) / max(height - 1, 1)
    return np.stack([x_norm, y_norm], axis=1)


def _make_cluster_features(embedding_features, coord_features=None,
                           normalize_embeddings=True,
                           embedding_weight=1.0,
                           coord_weight=0.25):
    """
    Build clustering features from model embeddings plus optional image coordinates.

    embedding_features: (N, D)
    coord_features:     (N, 2), normalized x/y coordinates
    """
    features = embedding_features.astype(np.float32)

    if normalize_embeddings:
        features = _l2_normalize(features)

    features = features * embedding_weight

    if coord_features is not None and coord_weight > 0:
        coord_features = coord_features.astype(np.float32) * coord_weight
        features = np.concatenate([features, coord_features], axis=1)

    return features


def my_postprocess(binary_logits, embedding, threshold=None, eps=1.5, 
                   min_samples=50, poly_degree=2, min_pixels=100, bandwidth=1.0, 
                   clustering_algorithm='dbscan', horizon_min_pixels=5, horizon_padding=10,
                   min_blob_pixels=30, newVersion=True,
                   normalize_embeddings=True, use_coordinates=True,
                   embedding_weight=1.0, coord_weight=0.25):
    """
    Binary logits: It has shape (1,2,H,W). Example shape binary_logits -> (1, 2, 256, 512) torch.float32 min=-6.530 max=8.266 mean=0.069
    Embedding: It has shape (1,4,H,W). Example shape embedding -> (1, 4, 256, 512) torch.float32 min=-11.913 max=10.448 mean=-0.115
    
    Example values at a single pixel (y=150, x=250):
    Pixel (150,250):
        binary_logits = [-3.21  4.78]  # [bg, lane]
        embedding     = [ 1.42 -0.31  0.85 -1.07]      # 4-D vector
    We can say that this pixel is likely a lane pixel (since lane logit > bg logit) and its embedding vector is [1.42, -0.31, 0.85, -1.07].
    Here we will try to look at these embeddings across the whole image and cluster them to find which pixels belong to 
    the same lane. The intuition is that pixels belonging to the same lane should have similar embedding vectors, 
    so they will be close in the embedding space and can be clustered together using a method like DBSCAN. After 
    clustering, we can fit a polynomial curve to the pixels in each cluster to get a smooth lane line.

    threshold: We will use it to convert the binary logits into a binary mask. If binary logit value for lane class is 
    greater than the threshold, we will consider that pixel as a lane pixel (True), otherwise it will be considered as 
    background (False). If threshold is None, we will simply take the argmax of the binary 
    logits to determine the binary mask.

    eps: This is the maximum distance between two samples for them to be considered as in the same neighborhood in DBSCAN.
    min_samples: This is the minimum number of samples in a neighborhood to form a lane cluster in DBSCAN.

    poly_degree: This is the degree of the polynomial we will fit to each lane cluster.
    min_pixels: This is the minimum number of pixels required in a cluster to consider it as a valid lane.

    bandwidth: This is the bandwidth parameter for MeanShift clustering, which defines the radius of the area 
    used to compute the mean shift.

    clustering_algorithm: This is the clustering algorithm to use for grouping lane pixels based on their embeddings. 
    It can be either 'dbscan' or 'meanshift'. 
    
    
    horizon_min_pixels: Dynamic horizon cutoff. We scan rows top-to-bottom and find the first row that has at least
    this many lane pixels. Rows above that are treated as horizon/sky noise and zeroed out before clustering.
    Set to None to disable. Default 5.

    horizon_padding: Extra rows to cut below the detected horizon, in pixels. Positive values are more aggressive
    (cut more of the image), negative values are more lenient. Default 10. Useful because the first row to pass
    the threshold often still has a few noisy pixels — a small padding gives a cleaner cutoff.

    min_blob_pixels: Spatial blob filter. Connected components in the binary mask with fewer than this many pixels
    are removed BEFORE the embedding clustering step. This prevents small noise blobs from being merged into real
    lane clusters by DBSCAN, which would distort the polynomial fit. Different from min_pixels (which filters
    AFTER clustering, in embedding-cluster space). Set to None to disable. Default 30.

    normalize_embeddings: L2-normalize embedding vectors before clustering. This makes DBSCAN/MeanShift compare
    embedding direction instead of raw magnitude, which usually matches LaneNet-style embeddings better.

    use_coordinates: Append normalized (x, y) pixel coordinates to the clustering features. This discourages
    grouping pixels that have similar embeddings but are far apart in the image.

    embedding_weight: Scale applied to embedding features before clustering.

    coord_weight: Scale applied to coordinate features before clustering. Increase if far-apart lanes are being
    merged; decrease if dashed segments of the same lane stop merging.
    """




    #Step 1: Converting binary logits to a binary mask

    #In binary logits, ndim = 4. We need to drop the batch dimension (the first dimension) to get a shape of (2, H, W)
    if binary_logits.ndim == 4:
        binary_logits = binary_logits[0]
    
    binary_mask = None

    if threshold is None:
        binary_mask = binary_logits.argmax(axis=0).astype(bool)
    else:
        #We will aplly the softmax function to the binary logits to get probabilities. The softmax function is defined as:
        #softmax(x_i) = exp(x_i) / sum_j exp(x_j)
        #binary_logits - binary_logits.max(axis=0, keepdims=True) is a common numerical stability trick to prevent 
        #overflow when computing the exponential. By subtracting the maximum logit value from all logits, 
        #we ensure that the largest value passed to the exponential function is 0, which prevents very large numbers 
        #that could cause overflow. 
        e = np.exp(binary_logits - binary_logits.max(axis=0, keepdims=True))
        #e[1] gives us the exponentiated lane logits, and e.sum(axis=0) gives us the sum of exponentiated logits 
        #across the two classes (background and lane) for each pixel. By dividing e[1] by e.sum(axis=0), we get 
        #the probability of the lane class for each pixel. Finally, we compare this probability to the threshold to 
        #get a binary mask where pixels with a lane probability greater than the threshold are marked as True (lane) 
        #and others as False (background).  
        prob_lane = e[1] / e.sum(axis=0)
        binary_mask = prob_lane > threshold
    

    #Dynamically find the horizon by looking at row-wise lane pixel density. The topmost row
    #with at least horizon_min_pixels lane pixels marks where "real" lane content starts.
    #horizon_padding lets us cut additional rows beyond that — useful because the first row
    #to pass the threshold is often still partly horizon noise. Clamped to [0, H] so a large
    #padding can't break the masking.
    if horizon_min_pixels is not None and horizon_min_pixels > 0:
        H = binary_mask.shape[0]
        row_counts = binary_mask.sum(axis=1)
        significant = np.where(row_counts >= horizon_min_pixels)[0]
        if len(significant) > 0:
            y_cutoff = int(significant[0]) + horizon_padding
            y_cutoff = max(0, min(y_cutoff, H))
            binary_mask[:y_cutoff, :] = False

    
    #Spatial blob filter. Find connected components in the binary mask and discard any blob
    #with fewer than min_blob_pixels pixels. This runs in image space, BEFORE the embedding
    #clustering — so small noise blobs (lane-mark-like pixels on signs, distant cars, debris)
    #never get a chance to merge into a real lane cluster and pull the polynomial fit off-track.
    #
    #Different from min_pixels (which filters AFTER DBSCAN, in embedding-cluster space).
    if min_blob_pixels is not None and min_blob_pixels > 0:
        mask_uint8 = binary_mask.astype(np.uint8)
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8)
        #stats[:, cv2.CC_STAT_AREA] is pixel count per component. Component 0 is the background.
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_blob_pixels:
                binary_mask[label_map == i] = False




    if newVersion:
        #Step 2: Hybrid clustering — connected components + per-blob embedding clustering.
        #
        #Why hybrid: pure connected components can't merge dashed lane segments or lanes split
        #by occlusion. Pure DBSCAN on pixel-level embeddings is slow and can wrongly merge
        #spatially-separated lanes when their embeddings happen to be similar.
        #
        #Hybrid steps:
        #  1. Find spatial blobs via connected components.
        #  2. Compute one mean embedding per blob.
        #  3. Cluster blobs by their mean embeddings (tiny DBSCAN, ~free).
        #  4. Each blob-cluster = one lane. Merge pixels from constituent blobs.
        if embedding.ndim == 4:
            embedding = embedding[0]
        
        mask_uint8 = binary_mask.astype(np.uint8)
        n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        H, W = binary_mask.shape

        #Per-blob: collect (y, x) pixel coords + mean embedding vector + mean image position.
        blob_pixels = []     # list of (ys, xs) per blob
        blob_means  = []     # list of mean embedding per blob
        blob_coords = []     # list of normalized mean (x, y) per blob
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_blob_pixels:
                continue
            ys_b, xs_b = np.where(label_map == i)
            emb_b = embedding[:, ys_b, xs_b]                # (D, n_blob)
            blob_pixels.append((ys_b, xs_b))
            blob_means.append(emb_b.mean(axis=1))           # (D,)
            blob_coords.append(_normalized_xy_features(
                ys_b, xs_b, H, W).mean(axis=0))             # (2,)
        
        if len(blob_means) == 0:
            return []
        
        #Cluster blobs by their mean embedding. Each blob is a single 4-D point now, so DBSCAN
        #here runs on N=10-30 points instead of N=thousands — orders of magnitude faster.
        #min_samples=1 because every blob should belong to *some* lane; we let eps decide grouping.
        blob_means_np = np.array(blob_means)
        blob_coords_np = np.array(blob_coords) if use_coordinates else None
        cluster_features = _make_cluster_features(
            blob_means_np,
            coord_features=blob_coords_np,
            normalize_embeddings=normalize_embeddings,
            embedding_weight=embedding_weight,
            coord_weight=coord_weight,
        )

        if clustering_algorithm == 'dbscan':
            db = DBSCAN(eps=eps, min_samples=1)
            blob_labels = db.fit_predict(cluster_features)
        else:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1)
            blob_labels = ms.fit_predict(cluster_features)
        
        #Step 3: Merge blobs within the same lane and fit polynomials.
        lanes = []
        for cid in np.unique(blob_labels):
            if cid == -1:
                continue
            
            #Concatenate pixels from all blobs assigned to this lane.
            member_indices = np.where(blob_labels == cid)[0]
            cys = np.concatenate([blob_pixels[i][0] for i in member_indices])
            cxs = np.concatenate([blob_pixels[i][1] for i in member_indices])
            
            if len(cys) < min_pixels:
                continue
            
            coeffs = np.polyfit(cys, cxs, deg=poly_degree)
            lanes.append({
                'cluster_id': int(cid),
                'poly':       coeffs,
                'y_range':    (int(cys.min()), int(cys.max())),
                'pixels':     np.stack([cys, cxs], axis=1),
            })
        
        lanes.sort(key=lambda l: l['pixels'][:, 1].mean())
        return lanes
    
    else:


        #Step 2: Clustering lane pixels by their embedding vectors

        #We have to drop the batch dimension from the embedding as well.
        if embedding.ndim == 4:
            embedding = embedding[0]

        #Here we just get the (y, x) coordinates of the pixels where the binary mask is True.
        ys, xs = np.where(binary_mask)
        #If there are no lane pixels detected then ys and xs will be empty arrays.
        if len(ys) == 0:
            return []

        # : means take all embedding dimensions, and ys, xs are the coordinates of the lane pixels. Here we get the 
        #embeeding values for the given lane pixel coordinates.
        features = embedding[:, ys, xs] # shape (D, N)
        #We need to transpose the features to have shape (N, D) because DBSCAN expects samples as rows and features as columns.
        features = features.T  #shape (N, D) for DBSCAN
        H, W = binary_mask.shape
        coord_features = _normalized_xy_features(ys, xs, H, W) if use_coordinates else None
        features = _make_cluster_features(
            features,
            coord_features=coord_features,
            normalize_embeddings=normalize_embeddings,
            embedding_weight=embedding_weight,
            coord_weight=coord_weight,
        )
        #Example matrix would be like this:
        #           Dim 0    Dim 1    Dim 2    Dim 3
        #           (Ch 0)   (Ch 1)   (Ch 2)   (Ch 3)
        #          ___________________________________
        # Pixel 1 |  8.26,  -2.14,   5.50,  -11.91  |  
        # Pixel 2 |  8.10,  -2.20,   5.45,  -11.85  |  
        # Pixel 3 | -6.53,   9.44,  -1.20,    4.10  |  
        # Pixel 4 | -6.40,   9.30,  -1.15,    4.05  |  
        # ...     |  ...     ...     ...      ...  
        # Pixel N |  0.06,  -0.11,   1.22,   -3.44  |  
        #          -----------------------------------

        labels = None
        if clustering_algorithm == 'dbscan':
            #Here we apply DBSCAN clustering to the features. DBSCAN will group together pixels that have similar 
            #embedding vectors and mark outliers as noise (label -1). 
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(features)
        else:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
            labels = ms.fit_predict(features)
        
        #lables is a 1D array of length N (number of lane pixels) where each element is the cluster ID assigned by DBSCAN.
        #For example, labels = np.array([0, 0, 0, -1, 0, 1, 1, 1, 1, -1])







        #Step 3: Fitting polynomial curves to each cluster of lane pixels

        lanes = []
        for cid in np.unique(labels):

            #Ignore the noise points labeled as -1 by DBSCAN
            if cid == -1:
                continue

            #Get the mask for the current cluster ID. This will give us a boolean array where True 
            #corresponds to pixels belonging to the current cluster.
            mask = labels == cid

            #Here we check if the number of pixels in the cluster is less than the minimum required pixels. 
            if mask.sum() < min_pixels:
                continue

            #Get the (y, x) coordinates of the pixels in the current cluster using the mask.
            cys, cxs = ys[mask], xs[mask]

            #We fit a polynomial of the specified degree to the (y, x) coordinates of the pixels in the cluster.
            coeffs = np.polyfit(cys, cxs, deg=poly_degree)

            #We create a dictionary for the current lane cluster containing the cluster ID, polynomial coefficients, 
            #the range of y values, and the pixel coordinates.
            lanes.append({
                'cluster_id': int(cid),
                'poly':       coeffs,
                'y_range':    (int(cys.min()), int(cys.max())),
                'pixels':     np.stack([cys, cxs], axis=1),
            })

        #Sort lanes left-to-right by their average x position (useful for ego-lane logic)
        #I can change this later to something like this lanes.sort(key=lambda l: l['poly'](l['y_range'][1])) but for now 
        #I will just sort by the mean x coordinate of the pixels in the lane cluster.
        lanes.sort(key=lambda l: l['pixels'][:, 1].mean())

        return lanes
    


def draw_lanes(image_rgb, lanes, color_palette=None, thickness=3):
    """
    Draw fitted polynomial curves on an image. Returns annotated copy.

    image_rgb: (H, W, 3) uint8.
    lanes:     output of postprocess().
    """
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










# import numpy as np
# from sklearn.cluster import DBSCAN, MeanShift
# import cv2

# from sklearn.linear_model import RANSACRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline

# def my_postprocess(binary_logits, embedding, threshold=None, eps=1.5, 
#                    min_samples=50, poly_degree=2, min_pixels=100, bandwidth=1.0, 
#                    clustering_algorithm='dbscan', horizon_min_pixels=5, horizon_padding=10,
#                    min_blob_pixels=30, newVersion = True):
#     """
#     Binary logits: It has shape (1,2,H,W). Example shape binary_logits -> (1, 2, 256, 512) torch.float32 min=-6.530 max=8.266 mean=0.069
#     Embedding: It has shape (1,4,H,W). Example shape embedding -> (1, 4, 256, 512) torch.float32 min=-11.913 max=10.448 mean=-0.115
    
#     Example values at a single pixel (y=150, x=250):
#     Pixel (150,250):
#         binary_logits = [-3.21  4.78]  # [bg, lane]
#         embedding     = [ 1.42 -0.31  0.85 -1.07]      # 4-D vector
#     We can say that this pixel is likely a lane pixel (since lane logit > bg logit) and its embedding vector is [1.42, -0.31, 0.85, -1.07].
#     Here we will try to look at these embeddings across the whole image and cluster them to find which pixels belong to 
#     the same lane. The intuition is that pixels belonging to the same lane should have similar embedding vectors, 
#     so they will be close in the embedding space and can be clustered together using a method like DBSCAN. After 
#     clustering, we can fit a polynomial curve to the pixels in each cluster to get a smooth lane line.

#     threshold: We will use it to convert the binary logits into a binary mask. If binary logit value for lane class is 
#     greater than the threshold, we will consider that pixel as a lane pixel (True), otherwise it will be considered as 
#     background (False). If threshold is None, we will simply take the argmax of the binary 
#     logits to determine the binary mask.

#     eps: This is the maximum distance between two samples for them to be considered as in the same neighborhood in DBSCAN.
#     min_samples: This is the minimum number of samples in a neighborhood to form a lane cluster in DBSCAN.

#     poly_degree: This is the degree of the polynomial we will fit to each lane cluster.
#     min_pixels: This is the minimum number of pixels required in a cluster to consider it as a valid lane.

#     bandwidth: This is the bandwidth parameter for MeanShift clustering, which defines the radius of the area 
#     used to compute the mean shift.

#     clustering_algorithm: This is the clustering algorithm to use for grouping lane pixels based on their embeddings. 
#     It can be either 'dbscan' or 'meanshift'. 
    
    
#     horizon_min_pixels: Dynamic horizon cutoff. We scan rows top-to-bottom and find the first row that has at least
#     this many lane pixels. Rows above that are treated as horizon/sky noise and zeroed out before clustering.
#     Set to None to disable. Default 5.

#     horizon_padding: Extra rows to cut below the detected horizon, in pixels. Positive values are more aggressive
#     (cut more of the image), negative values are more lenient. Default 10. Useful because the first row to pass
#     the threshold often still has a few noisy pixels — a small padding gives a cleaner cutoff.

#     min_blob_pixels: Spatial blob filter. Connected components in the binary mask with fewer than this many pixels
#     are removed BEFORE the embedding clustering step. This prevents small noise blobs from being merged into real
#     lane clusters by DBSCAN, which would distort the polynomial fit. Different from min_pixels (which filters
#     AFTER clustering, in embedding-cluster space). Set to None to disable. Default 30.
#     """




#     #Step 1: Converting binary logits to a binary mask

#     #In binary logits, ndim = 4. We need to drop the batch dimension (the first dimension) to get a shape of (2, H, W)
#     if binary_logits.ndim == 4:
#         binary_logits = binary_logits[0]
    
#     binary_mask = None

#     if threshold is None:
#         binary_mask = binary_logits.argmax(axis=0).astype(bool)
#     else:
#         #We will aplly the softmax function to the binary logits to get probabilities. The softmax function is defined as:
#         #softmax(x_i) = exp(x_i) / sum_j exp(x_j)
#         #binary_logits - binary_logits.max(axis=0, keepdims=True) is a common numerical stability trick to prevent 
#         #overflow when computing the exponential. By subtracting the maximum logit value from all logits, 
#         #we ensure that the largest value passed to the exponential function is 0, which prevents very large numbers 
#         #that could cause overflow. 
#         e = np.exp(binary_logits - binary_logits.max(axis=0, keepdims=True))
#         #e[1] gives us the exponentiated lane logits, and e.sum(axis=0) gives us the sum of exponentiated logits 
#         #across the two classes (background and lane) for each pixel. By dividing e[1] by e.sum(axis=0), we get 
#         #the probability of the lane class for each pixel. Finally, we compare this probability to the threshold to 
#         #get a binary mask where pixels with a lane probability greater than the threshold are marked as True (lane) 
#         #and others as False (background).  
#         prob_lane = e[1] / e.sum(axis=0)
#         binary_mask = prob_lane > threshold
    

#     #Dynamically find the horizon by looking at row-wise lane pixel density. The topmost row
#     #with at least horizon_min_pixels lane pixels marks where "real" lane content starts.
#     #horizon_padding lets us cut additional rows beyond that — useful because the first row
#     #to pass the threshold is often still partly horizon noise. Clamped to [0, H] so a large
#     #padding can't break the masking.
#     if horizon_min_pixels is not None and horizon_min_pixels > 0:
#         H = binary_mask.shape[0]
#         row_counts = binary_mask.sum(axis=1)
#         significant = np.where(row_counts >= horizon_min_pixels)[0]
#         if len(significant) > 0:
#             y_cutoff = int(significant[0]) + horizon_padding
#             y_cutoff = max(0, min(y_cutoff, H))
#             binary_mask[:y_cutoff, :] = False

    
#     #Spatial blob filter. Find connected components in the binary mask and discard any blob
#     #with fewer than min_blob_pixels pixels. This runs in image space, BEFORE the embedding
#     #clustering — so small noise blobs (lane-mark-like pixels on signs, distant cars, debris)
#     #never get a chance to merge into a real lane cluster and pull the polynomial fit off-track.
#     #
#     #Different from min_pixels (which filters AFTER DBSCAN, in embedding-cluster space).
#     if min_blob_pixels is not None and min_blob_pixels > 0:
#         mask_uint8 = binary_mask.astype(np.uint8)
#         num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
#             mask_uint8, connectivity=8)
#         #stats[:, cv2.CC_STAT_AREA] is pixel count per component. Component 0 is the background.
#         for i in range(1, num_labels):
#             if stats[i, cv2.CC_STAT_AREA] < min_blob_pixels:
#                 binary_mask[label_map == i] = False




#     if newVersion:
#         #Step 2: Hybrid clustering — connected components + per-blob embedding clustering.
#         #
#         #Why hybrid: pure connected components can't merge dashed lane segments or lanes split
#         #by occlusion. Pure DBSCAN on pixel-level embeddings is slow and can wrongly merge
#         #spatially-separated lanes when their embeddings happen to be similar.
#         #
#         #Hybrid steps:
#         #  1. Find spatial blobs via connected components.
#         #  2. Compute one mean embedding per blob.
#         #  3. Cluster blobs by their mean embeddings (tiny DBSCAN, ~free).
#         #  4. Each blob-cluster = one lane. Merge pixels from constituent blobs.
#         if embedding.ndim == 4:
#             embedding = embedding[0]
        
#         mask_uint8 = binary_mask.astype(np.uint8)
#         n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
#         #Per-blob: collect (y, x) pixel coords + mean embedding vector.
#         blob_pixels = []     # list of (ys, xs) per blob
#         blob_means  = []     # list of mean embedding per blob
#         for i in range(1, n_labels):
#             if stats[i, cv2.CC_STAT_AREA] < min_blob_pixels:
#                 continue
#             ys_b, xs_b = np.where(label_map == i)
#             emb_b = embedding[:, ys_b, xs_b]                # (D, n_blob)
#             blob_pixels.append((ys_b, xs_b))
#             blob_means.append(emb_b.mean(axis=1))           # (D,)
        
#         if len(blob_means) == 0:
#             return []
        
#         #Cluster blobs by their mean embedding. Each blob is a single 4-D point now, so DBSCAN
#         #here runs on N=10-30 points instead of N=thousands — orders of magnitude faster.
#         #min_samples=1 because every blob should belong to *some* lane; we let eps decide grouping.
#         blob_means_np = np.array(blob_means)
#         if clustering_algorithm == 'dbscan':
#             db = DBSCAN(eps=eps, min_samples=1)
#             blob_labels = db.fit_predict(blob_means_np)
#         else:
#             ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1)
#             blob_labels = ms.fit_predict(blob_means_np)
        
#         #Step 3: Merge blobs within the same lane and fit polynomials.
#         lanes = []
#         for cid in np.unique(blob_labels):
#             if cid == -1:
#                 continue
            
#             #Concatenate pixels from all blobs assigned to this lane.
#             member_indices = np.where(blob_labels == cid)[0]
#             cys = np.concatenate([blob_pixels[i][0] for i in member_indices])
#             cxs = np.concatenate([blob_pixels[i][1] for i in member_indices])
            
#             if len(cys) < min_pixels:
#                 continue
            
#             coeffs = np.polyfit(cys, cxs, deg=poly_degree)
#             lanes.append({
#                 'cluster_id': int(cid),
#                 'poly':       coeffs,
#                 'y_range':    (int(cys.min()), int(cys.max())),
#                 'pixels':     np.stack([cys, cxs], axis=1),
#             })
        
#         lanes.sort(key=lambda l: l['pixels'][:, 1].mean())
#         return lanes
    
#     else:


#         #Step 2: Clustering lane pixels by their embedding vectors

#         #We have to drop the batch dimension from the embedding as well.
#         if embedding.ndim == 4:
#             embedding = embedding[0]

#         #Here we just get the (y, x) coordinates of the pixels where the binary mask is True.
#         ys, xs = np.where(binary_mask)
#         #If there are no lane pixels detected then ys and xs will be empty arrays.
#         if len(ys) == 0:
#             return []

#         # : means take all embedding dimensions, and ys, xs are the coordinates of the lane pixels. Here we get the 
#         #embeeding values for the given lane pixel coordinates.
#         features = embedding[:, ys, xs] # shape (D, N)
#         #We need to transpose the features to have shape (N, D) because DBSCAN expects samples as rows and features as columns.
#         features = features.T  #shape (N, D) for DBSCAN
#         #Example matrix would be like this:
#         #           Dim 0    Dim 1    Dim 2    Dim 3
#         #           (Ch 0)   (Ch 1)   (Ch 2)   (Ch 3)
#         #          ___________________________________
#         # Pixel 1 |  8.26,  -2.14,   5.50,  -11.91  |  
#         # Pixel 2 |  8.10,  -2.20,   5.45,  -11.85  |  
#         # Pixel 3 | -6.53,   9.44,  -1.20,    4.10  |  
#         # Pixel 4 | -6.40,   9.30,  -1.15,    4.05  |  
#         # ...     |  ...     ...     ...      ...  
#         # Pixel N |  0.06,  -0.11,   1.22,   -3.44  |  
#         #          -----------------------------------

#         labels = None
#         if clustering_algorithm == 'dbscan':
#             #Here we apply DBSCAN clustering to the features. DBSCAN will group together pixels that have similar 
#             #embedding vectors and mark outliers as noise (label -1). 
#             db = DBSCAN(eps=eps, min_samples=min_samples)
#             labels = db.fit_predict(features)
#         else:
#             ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
#             labels = ms.fit_predict(features)
        
#         #lables is a 1D array of length N (number of lane pixels) where each element is the cluster ID assigned by DBSCAN.
#         #For example, labels = np.array([0, 0, 0, -1, 0, 1, 1, 1, 1, -1])







#         #Step 3: Fitting polynomial curves to each cluster of lane pixels

#         lanes = []
#         for cid in np.unique(labels):

#             #Ignore the noise points labeled as -1 by DBSCAN
#             if cid == -1:
#                 continue

#             #Get the mask for the current cluster ID. This will give us a boolean array where True 
#             #corresponds to pixels belonging to the current cluster.
#             mask = labels == cid

#             #Here we check if the number of pixels in the cluster is less than the minimum required pixels. 
#             if mask.sum() < min_pixels:
#                 continue

#             #Get the (y, x) coordinates of the pixels in the current cluster using the mask.
#             cys, cxs = ys[mask], xs[mask]

#             #We fit a polynomial of the specified degree to the (y, x) coordinates of the pixels in the cluster.
#             coeffs = np.polyfit(cys, cxs, deg=poly_degree)

#             #We create a dictionary for the current lane cluster containing the cluster ID, polynomial coefficients, 
#             #the range of y values, and the pixel coordinates.
#             lanes.append({
#                 'cluster_id': int(cid),
#                 'poly':       coeffs,
#                 'y_range':    (int(cys.min()), int(cys.max())),
#                 'pixels':     np.stack([cys, cxs], axis=1),
#             })

#         #Sort lanes left-to-right by their average x position (useful for ego-lane logic)
#         #I can change this later to something like this lanes.sort(key=lambda l: l['poly'](l['y_range'][1])) but for now 
#         #I will just sort by the mean x coordinate of the pixels in the lane cluster.
#         lanes.sort(key=lambda l: l['pixels'][:, 1].mean())

#         return lanes
    


# def draw_lanes(image_rgb, lanes, color_palette=None, thickness=3):
#     """
#     Draw fitted polynomial curves on an image. Returns annotated copy.

#     image_rgb: (H, W, 3) uint8.
#     lanes:     output of postprocess().
#     """
#     if color_palette is None:
#         color_palette = [
#             (255,  56,  56),   # red
#             ( 56, 255,  56),   # green
#             ( 56,  56, 255),   # blue
#             (255, 255,  56),   # yellow
#             (255,  56, 255),   # magenta
#             ( 56, 255, 255),   # cyan
#         ]
#     out = image_rgb.copy()
#     for i, lane in enumerate(lanes):
#         y_min, y_max = lane['y_range']
#         ys = np.arange(y_min, y_max + 1)
#         xs = np.polyval(lane['poly'], ys).astype(int)
#         # Clip to image bounds
#         h, w = out.shape[:2]
#         valid = (xs >= 0) & (xs < w)
#         ys, xs = ys[valid], xs[valid]
#         pts = np.stack([xs, ys], axis=1)  # cv2 expects (x, y) pairs
#         color = color_palette[i % len(color_palette)]
#         for j in range(len(pts) - 1):
#             cv2.line(out, tuple(pts[j]), tuple(pts[j + 1]), color, thickness)
#     return out