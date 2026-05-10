import torch
import torch.nn as nn
import torch.nn.functional as F
import src.config as cfg

#Normally in lane detaction, about 95% of pixels are background and only 5% are lane markings.
#This class imbalance can cause the model to be biased towards predicting the majority class (background)
#and perform poorly on the minority class (lane markings). Following loss function is used to address this issue.
def compute_binary_loss(binary_logits, binary_mask, class_weights=None):
    """
    binary_logits: (B, 2, H, W) raw logits from binary decoder. B is batch size and The 2 in the shape represents
    two "channels" of scores: one for Background and one for Lane.

    binary_mask:   (B, 1, H, W) ground truth, values in {0., 1.}
    """

    if class_weights is not None:
        class_weights = class_weights.to(binary_logits.device)

    #In order to compute the cross-entropy loss, We need to convert the binary_mask from a shape of (B, 1, H, W) to
    #(B, H, W) and from float values (0. and 1.) to integer class IDs (0 and 1).
    target = binary_mask.squeeze(1).long()

    #Cross-entropy loss uses Softmax to convert the raw logits into probabilities for each class and then computes
    #the loss based on the target class IDs using the class weights to give more importance to the minority class
    #(lane markings).
    loss = F.cross_entropy(binary_logits, target, weight=class_weights)

    return loss


#Dice loss directly optimizes the overlap between predicted lane pixels and ground truth lane pixels.
#Unlike cross-entropy, it does not punish confident-wrong predictions exponentially — once a pixel is on
#the wrong side of the decision boundary, being more confident does not make Dice worse. This is exactly
#what we want for thin structures like lane markings where CE tends to drive the model into overconfident
#errors near lane edges.
#
#Formula:
#    dice = (2 * |pred ∩ gt| + smooth) / (|pred| + |gt| + smooth)
#    loss = 1 - dice
#
#We use the softmax probability of the lane class (channel 1) as the soft prediction, which keeps the loss
#differentiable end-to-end.
def compute_dice_loss(binary_logits, binary_mask, smooth=1.0):
    """
    binary_logits: (B, 2, H, W) raw logits from binary decoder.
    binary_mask:   (B, 1, H, W) ground truth, values in {0., 1.}.
    smooth:        small constant added to numerator and denominator. Stabilizes the gradient and avoids
                   division-by-zero when an image has no lane pixels.
    """
    #Softmax across the class dimension and grab the lane channel. probs is (B, H, W).
    probs = F.softmax(binary_logits, dim=1)[:, 1, :, :]
    target = binary_mask.squeeze(1).float()                # (B, H, W)

    #Per-sample dice so each image contributes equally regardless of how many lane pixels it has.
    #Without per-sample reduction, a single image with many lane pixels would dominate the batch.
    intersection = (probs * target).sum(dim=(1, 2))         # (B,)
    pred_sum     = probs.sum(dim=(1, 2))
    target_sum   = target.sum(dim=(1, 2))

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1.0 - dice.mean()




def discriminative_loss_single(embedding, instance_mask,
                               delta_var=0.3, delta_dist=1.5,
                               alpha=1.0, beta=1.0, gamma=0.001):
    """
    embedding:     (D, H, W) — raw embeddings for one image
    instance_mask: (1, H, W) or (H, W) — integer instance IDs, 0 = background
    """
    D, H, W = embedding.shape
    embedding_flat = embedding.reshape(D, H * W)              # (D, N)
    inst_flat = instance_mask.reshape(-1)                      # (N,)

    #Get unique lane IDs, drop background (0). In instance_mask, we have different lanes specified with 
    #different gray scale values and the background is represented with 0. So we get the unique 
    #lane IDs. by looking at these vakues and ignore the background.
    unique_ids = torch.unique(inst_flat)
    unique_ids = unique_ids[unique_ids != 0]
    K = len(unique_ids)

    #Edge case: no lanes in this image. We use embedding.sum() * 0.0 to create a zero loss that still has a 
    #gradient graph, so it won't break backpropagation.
    if K == 0:
        zero = embedding.sum() * 0.0
        return zero, zero, zero, zero

    # ---- Compute the K cluster means ----
    means = []
    for cid in unique_ids:
        #Here we get the pixels that belong to the current lane cluster (cid). N is the total number of pixels (H * W).
        pixel_mask = (inst_flat == cid)                        #(N,) bool
        #Then we select the corresponding embeddings for those pixels, resulting in a tensor of shape (D, n_c), 
        #where n_c is the number of pixels in that lane cluster. 
        cluster_pixels = embedding_flat[:, pixel_mask]         # (D, n_c)
        #Here we compute the mean embedding for the cuurrent lane cluster over the n_c pixels, resulting in a mean 
        #vector of shape (D,). dim=1 means we average over the n_c pixels, keeping the D dimensions of the embedding.
        means.append(cluster_pixels.mean(dim=1))               # (D,)
    #Stack does not concatenate along an existing dimension, but rather creates a new dimension at the specified 
    #position (dim=0 in this case).
    means = torch.stack(means, dim=0)                          # (K, D)

    # ---- Term 1: variance (pull) ----
    var_loss = 0.0
    for i, cid in enumerate(unique_ids):
        pixel_mask = (inst_flat == cid)
        cluster_pixels = embedding_flat[:, pixel_mask]         # (D, n_c)
        #Distance from each pixel to its cluster mean. Before that we have to unsqueeze the mean to (D, 1) 
        #so that it can be broadcasted against the (D, n_c) cluster_pixels.
        diff = cluster_pixels - means[i].unsqueeze(1)          # (D, n_c)
        dist = torch.norm(diff, p=2, dim=0)                    # (n_c,)
        #Hinge: we only penalize pixels that are farther than delta_var from their cluster mean. The values 
        #below zero are set to 0. By squaring the hinge, we penalize larger distances more heavily, encouraging 
        #tighter clusters.
        hinged = torch.clamp(dist - delta_var, min=0.0) ** 2
        var_loss = var_loss + hinged.mean()
    var_loss = var_loss / K

    # ---- Term 2: distance (push) — only if K > 1 ----
    if K > 1:
        # Pairwise distances between means: (K, K)
        pairwise = torch.cdist(means, means, p=2)
        # Hinge: penalize pairs closer than 2 * delta_dist
        hinged = torch.clamp(2 * delta_dist - pairwise, min=0.0) ** 2
        # Zero out the diagonal (mean to itself = 0, would be hinged to (2*δ_d)²)
        hinged = hinged - torch.diag(torch.diag(hinged))
        # Average over K * (K - 1) off-diagonal pairs
        dist_loss = hinged.sum() / (K * (K - 1))
    else:
        dist_loss = embedding.sum() * 0.0

    # ---- Term 3: regularization (keep means small) ----
    reg_loss = torch.norm(means, p=2, dim=1).mean()

    total = alpha * var_loss + beta * dist_loss + gamma * reg_loss
    return total, var_loss, dist_loss, reg_loss



def discriminative_loss(embedding_batch, instance_mask_batch, **kwargs):
    """
    embedding_batch:     (B, D, H, W)
    instance_mask_batch: (B, 1, H, W)
    """
    B = embedding_batch.shape[0]
    total = 0.0
    var_total = 0.0
    dist_total = 0.0
    reg_total = 0.0

    for b in range(B):
        t, v, d, r = discriminative_loss_single(
            embedding_batch[b], instance_mask_batch[b], **kwargs)
        total += t
        var_total += v
        dist_total += d
        reg_total += r

    return total / B, var_total / B, dist_total / B, reg_total / B




def compute_loss(binary_logits, embedding,
                 binary_mask, instance_mask,
                 binary_weight=1.0, disc_weight=1.0,
                 ce_weight=0.5, dice_weight=0.5,
                 class_weights=None):
    """
    Returns (total_loss, dict_of_components_for_logging).

    binary head loss is a weighted sum of cross-entropy (handles class imbalance via class_weights)
    and Dice (handles overconfidence and directly optimizes overlap). Equal 0.5/0.5 weighting is a
    sensible default — CE alone tends to drive confident-wrong predictions at lane edges, Dice alone
    can be slow to converge on background. The mix gives you the best of both.
    """
    #Two halves of the binary loss, then combine.
    binary_ce   = compute_binary_loss(binary_logits, binary_mask, class_weights)
    binary_dice = compute_dice_loss(binary_logits, binary_mask)
    binary = ce_weight * binary_ce + dice_weight * binary_dice

    #Discriminative loss for the embedding head is unchanged.
    disc, var, dist, reg = discriminative_loss(embedding, instance_mask)

    total = binary_weight * binary + disc_weight * disc

    components = {
        'total':       total.item(),
        'binary':      binary.item(),
        'binary_ce':   binary_ce.item(),
        'binary_dice': binary_dice.item(),
        'disc':        disc.item(),
        'variance':    var.item(),
        'distance':    dist.item(),
        'reg':         reg.item(),
    }
    return total, components


