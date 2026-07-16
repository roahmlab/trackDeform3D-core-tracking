"""Evaluation metrics for the fabric driver."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def filter_ee_outliers(ee_poses_3d, velocity_threshold=100.0, window_size=5):
    """Filter outlier EE positions using velocity-based detection."""
    from scipy.ndimage import median_filter

    filtered = ee_poses_3d.copy()
    outlier_frames = []
    n_frames = len(ee_poses_3d)

    for arm_idx in range(2):  # 0=left, 1=right
        positions = ee_poses_3d[:, arm_idx, :]

        velocities = np.zeros(n_frames)
        for i in range(1, n_frames):
            velocities[i] = np.linalg.norm(positions[i] - positions[i-1])

        outlier_mask = velocities > velocity_threshold

        for i in range(1, n_frames - 1):
            if velocities[i] > velocity_threshold and velocities[i+1] > velocity_threshold * 0.5:
                outlier_mask[i] = True

        outlier_indices = np.where(outlier_mask)[0]

        if len(outlier_indices) > 0:
            arm_name = "left" if arm_idx == 0 else "right"
            print(f"  WARNING: Detected {len(outlier_indices)} outlier frames for {arm_name} EE: {outlier_indices.tolist()}")

            for idx in outlier_indices:
                outlier_frames.append((idx, arm_idx))

            for idx in outlier_indices:
                prev_valid = idx - 1
                while prev_valid >= 0 and outlier_mask[prev_valid]:
                    prev_valid -= 1

                next_valid = idx + 1
                while next_valid < n_frames and outlier_mask[next_valid]:
                    next_valid += 1

                if prev_valid >= 0 and next_valid < n_frames:
                    t = (idx - prev_valid) / (next_valid - prev_valid)
                    filtered[idx, arm_idx] = (1 - t) * positions[prev_valid] + t * positions[next_valid]
                elif prev_valid >= 0:
                    filtered[idx, arm_idx] = positions[prev_valid]
                elif next_valid < n_frames:
                    filtered[idx, arm_idx] = positions[next_valid]

        if window_size > 1:
            for dim in range(3):
                filtered[:, arm_idx, dim] = median_filter(filtered[:, arm_idx, dim], size=window_size)

    return filtered, outlier_frames

def compute_edge_metrics(keypoints, edges, reference_lengths):
    """Compute edge length metrics."""
    if keypoints is None or len(keypoints) == 0 or edges is None or len(edges) == 0:
        return {
            'pct_errors': np.array([]), 'abs_errors': np.array([]),
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0, 'rmse_mm': 0.0,
            'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    pct_errors = []
    abs_errors = []
    for edge_idx, (i, j) in enumerate(edges):
        if i >= len(keypoints) or j >= len(keypoints):
            continue

        if isinstance(reference_lengths, dict):
            ref_length = reference_lengths.get((i, j), reference_lengths.get(edge_idx, 0))
        else:
            ref_length = reference_lengths[edge_idx] if edge_idx < len(reference_lengths) else 0

        if ref_length > 1e-6:
            current_length = np.linalg.norm(keypoints[i] - keypoints[j])
            abs_err = abs(current_length - ref_length)
            pct_err = abs_err / ref_length
            pct_errors.append(pct_err)
            abs_errors.append(abs_err)

    pct_errors = np.array(pct_errors)
    abs_errors = np.array(abs_errors)

    if len(pct_errors) == 0:
        return {
            'pct_errors': np.array([]), 'abs_errors': np.array([]),
            'pct_mean': 0.0, 'pct_std': 0.0, 'pct_max': 0.0, 'rmse_mm': 0.0,
            'under_2pct': 0.0, 'under_5pct': 0.0, 'under_10pct': 0.0,
        }

    return {
        'pct_errors': pct_errors,
        'abs_errors': abs_errors,
        'pct_mean': np.mean(pct_errors) * 100,
        'pct_std': np.std(pct_errors) * 100,
        'pct_max': np.max(pct_errors) * 100,
        'rmse_mm': np.sqrt(np.mean(abs_errors ** 2)),
        'under_2pct': np.mean(pct_errors < 0.02) * 100,
        'under_5pct': np.mean(pct_errors < 0.05) * 100,
        'under_10pct': np.mean(pct_errors < 0.10) * 100,
    }

def compute_position_metrics(keypoints, point_cloud):
    """Compute position metrics (distance to nearest point in surface)."""
    if keypoints is None or len(keypoints) == 0 or point_cloud is None or len(point_cloud) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }

    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud)
    distances, _ = nn.kneighbors(keypoints)
    distances = distances.flatten()

    return {
        'distances': distances,
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,
        'under_5mm': np.mean(distances < 5.0) * 100,
        'under_10mm': np.mean(distances < 10.0) * 100,
    }

def sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=10):
    """Sample points uniformly on quad faces for Chamfer distance."""
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if len(keypoints) != grid_rows * grid_cols:
        return keypoints.copy()

    sampled_points = []

    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            idx_tl = r * grid_cols + c
            idx_tr = r * grid_cols + c + 1
            idx_bl = (r + 1) * grid_cols + c
            idx_br = (r + 1) * grid_cols + c + 1

            p_tl = keypoints[idx_tl]
            p_tr = keypoints[idx_tr]
            p_bl = keypoints[idx_bl]
            p_br = keypoints[idx_br]

            for _ in range(n_samples_per_face):
                u = np.random.random()
                v = np.random.random()

                p_top = (1 - u) * p_tl + u * p_tr
                p_bot = (1 - u) * p_bl + u * p_br
                p = (1 - v) * p_top + v * p_bot

                sampled_points.append(p)

    if len(sampled_points) == 0:
        return keypoints.copy()

    return np.array(sampled_points, dtype=np.float32)

def compute_chamfer_metrics(pred_cloud, ref_cloud):
    """Compute Chamfer Distance metrics."""
    empty_result = {
        'pred2ref_avg': 0.0, 'ref2pred_avg': 0.0, 'cd': 0.0,
        'precision_2mm': 0.0, 'precision_5mm': 0.0, 'precision_10mm': 0.0,
        'recall_2mm': 0.0, 'recall_5mm': 0.0, 'recall_10mm': 0.0,
        'f_2mm': 0.0, 'f_5mm': 0.0, 'f_10mm': 0.0,
    }

    if pred_cloud is None or len(pred_cloud) == 0 or ref_cloud is None or len(ref_cloud) == 0:
        return empty_result

    nn_ref = NearestNeighbors(n_neighbors=1).fit(ref_cloud)
    pred2ref_dists, _ = nn_ref.kneighbors(pred_cloud)
    pred2ref_dists = pred2ref_dists.flatten()

    nn_pred = NearestNeighbors(n_neighbors=1).fit(pred_cloud)
    ref2pred_dists, _ = nn_pred.kneighbors(ref_cloud)
    ref2pred_dists = ref2pred_dists.flatten()

    pred2ref_avg = np.mean(pred2ref_dists)
    ref2pred_avg = np.mean(ref2pred_dists)
    cd = (pred2ref_avg + ref2pred_avg) / 2

    def compute_pr_f(p2r, r2p, thresh):
        prec = np.mean(p2r < thresh) * 100
        rec = np.mean(r2p < thresh) * 100
        f = 2 * prec * rec / (prec + rec + 1e-8)
        return prec, rec, f

    p2, r2, f2 = compute_pr_f(pred2ref_dists, ref2pred_dists, 2.0)
    p5, r5, f5 = compute_pr_f(pred2ref_dists, ref2pred_dists, 5.0)
    p10, r10, f10 = compute_pr_f(pred2ref_dists, ref2pred_dists, 10.0)

    return {
        'pred2ref_avg': pred2ref_avg,
        'ref2pred_avg': ref2pred_avg,
        'cd': cd,
        'precision_2mm': p2, 'precision_5mm': p5, 'precision_10mm': p10,
        'recall_2mm': r2, 'recall_5mm': r5, 'recall_10mm': r10,
        'f_2mm': f2, 'f_5mm': f5, 'f_10mm': f10,
    }
