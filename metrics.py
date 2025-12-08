import numpy as np

from scipy.spatial import cKDTree

# ----------------------- Pixel Accuracy (PA) -----------------------
def pixel_accuracy(pred, true, threshold=0.5):
    
    """
    Compute pixel accuracy.
    
    Args:
        pred (np.ndarray): predicted map (H, W), continuous values in [0,1]
        true (np.ndarray): ground truth map (H, W), values {0,1}
        threshold (float): threshold to binarize predictions
    
    Returns:
        float: accuracy score in [0,1]
    """

    # Binarize predictions and ground truth
    pred_bin = pred > threshold
    true_bin = true == 1

    # Number of correct pixels (True Positives + True Negatives)
    correct = (pred_bin == true_bin).sum()

    # Total number of pixels
    total = pred_bin.size

    return correct / total

# ----------------------- Jaccard Similarity -----------------------
def jaccard_similarity(pred, true, threshold=0.5):

    """
    Compute Jaccard Similarity Coefficient (Intersection over Union) between prediction and ground truth.

    Returns:
        float: accuracy score in [0,1]
    """

    pred_bin = pred > threshold
    true_bin = true == 1

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / (union + 1e-10)


# ----------------------- Hausdorff Distance (HD) -----------------------
def hausdorff_distance(pred, true, threshold=0.5):
    """
    Compute symmetric 95th percentile Hausdorff Distance (HD-95).

    Returns:
        float: HD-95 in pixels
    """

    pred_bin = pred > threshold
    true_bin = true == 1

    pred_points = np.argwhere(pred_bin)
    true_points = np.argwhere(true_bin)

    if len(pred_points) == 0 or len(true_points) == 0:
        return np.nan
    
    # Build KD-trees for efficient nearest-neighbor queries
    tree_pred = cKDTree(pred_points)
    tree_true = cKDTree(true_points)

    # Distances from each ground-truth point to nearest predicted point
    dists_true_to_pred, _ = tree_pred.query(true_points)

    # Distances from each predicted point to nearest ground-truth point
    dists_pred_to_true, _ = tree_true.query(pred_points)

    # 95th percentile in each direction
    hd_true = np.percentile(dists_true_to_pred, 95)
    hd_pred = np.percentile(dists_pred_to_true, 95)

    # Symmetric HD-95
    hd95 = max(hd_true, hd_pred)

    return hd95