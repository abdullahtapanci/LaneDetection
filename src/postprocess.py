import numpy as np
from sklearn.cluster import DBSCAN, MeanShift
import cv2





#Here we define the perspective transformation matrices to convert between the original image space and the 
#bird's-eye view (BEV) space.
#src_pts are the coordinates of the four points in the original image that form a rectangle around the lane markings.
src_pts = np.array([
    [0, 256],   # 4 image-space points forming a road rectangle
    [512, 256],
    [100, 120],
    [412, 120],
], dtype=np.float32)
#dst_pts are the coordinates of the four points in the bird's-eye view space that correspond to the src_pts. In the BEV 
#space, we want these points to form a straight rectangle where the lanes are parallel and vertical.
dst_pts = np.array([
    [100, 600],
    [300, 600],
    [100,   0],
    [300,   0],
], dtype=np.float32)

H_to_bev   = cv2.getPerspectiveTransform(src_pts, dst_pts)
H_from_bev = cv2.getPerspectiveTransform(dst_pts, src_pts)





def my_postprocess(binary_logits, embedding, threshold=None, eps=1.5, 
                   min_samples=50, poly_degree=2, min_pixels=100, bandwidth=1.0, 
                   clustering_algorithm='dbscan', using_bev=False):
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

        if using_bev:
            #If we are using bird's-eye view (BEV) for polynomial fitting, we will call the fit_lanes_using_bev function instead of fitting in the original image space.
            lanes.append(fit_lanes_using_bev(H_to_bev, cys, cxs, poly_degree, cid))
        else:
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
    

#Here we try to fit polynomial curves in the bird's-eye view space instead of the original image space. 
#This is because in the BEV space, the lanes are more likely to be straight and parallel, which makes polynomial 
#fitting more accurate and stable.
def fit_lanes_using_bev(H_to_bev, cys, cxs, poly_degree=2, cid=0):
        
    #Here we convert the (y, x) coordinates of the lane pixels from the original image space to the bird's-eye 
    #view (BEV) space using the perspective transformation matrix H_to_bev.
    pts_img = np.column_stack([cxs, cys, np.ones(len(cxs))]).T   # (3, N)
    pts_bev = H_to_bev @ pts_img
    pts_bev = pts_bev[:2] / pts_bev[2:]                           # divide by w
    bev_xs, bev_ys = pts_bev[0], pts_bev[1]

    #here we fit in BEV space (lanes are nearly straight here)
    coeffs = np.polyfit(bev_ys, bev_xs, deg=poly_degree)

    return {
        'cluster_id':  int(cid),
        'poly':    coeffs,
        'y_range': (float(bev_ys.min()), float(bev_ys.max())),
        'pixels':      np.stack([cys, cxs], axis=1),
    }

def draw_lanes_using_bev(image_rgb, lanes, color_palette=None, thickness=4):
    if color_palette is None:
        color_palette = [(255,56,56), (56,255,56), (56,56,255), (255,255,56)]
    out = image_rgb.copy()
    H_img, W_img = out.shape[:2]

    for i, lane in enumerate(lanes):
        bev_y_min, bev_y_max = lane['y_range']

        # ─── Sample polynomial in BEV ──────────────────
        bev_ys = np.linspace(bev_y_min, bev_y_max, 100)
        bev_xs = np.polyval(lane['poly'], bev_ys)

        # ─── Warp back to image space ──────────────────
        pts_bev = np.column_stack([bev_xs, bev_ys, np.ones(len(bev_ys))]).T
        pts_img = H_from_bev @ pts_bev
        pts_img = pts_img[:2] / pts_img[2:]
        img_xs = pts_img[0].astype(int)
        img_ys = pts_img[1].astype(int)

        # ─── Draw, clipped to image bounds ─────────────
        valid = (img_xs >= 0) & (img_xs < W_img) & \
                (img_ys >= 0) & (img_ys < H_img)
        pts = np.stack([img_xs[valid], img_ys[valid]], axis=1)
        color = color_palette[i % len(color_palette)]
        for j in range(len(pts) - 1):
            cv2.line(out, tuple(pts[j]), tuple(pts[j+1]), color, thickness)
    return out

def draw_lanes_without_bev(image_rgb, lanes, color_palette=None, thickness=3):
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



def draw_lanes(image_rgb, lanes, using_bev=False, color_palette=None, thickness=3):
    if using_bev:
        return draw_lanes_using_bev(image_rgb, lanes, color_palette, thickness)
    else:
        return draw_lanes_without_bev(image_rgb, lanes, color_palette, thickness)