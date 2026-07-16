"""Trajectory smoothing shared by all four trackers."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_trajectories(keypoints_3d_seq: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing to keypoint trajectories along the time axis.

    NaN frames are filled by linear interpolation before smoothing, then
    smoothing uses 'nearest' boundary handling. Mirrors the post-processing
    in deformable_seg/smooth_bdlo_keypoints.py.

    Args:
        keypoints_3d_seq: T x K x 3 array of keypoints over time (NaN allowed)
        sigma: Gaussian filter sigma (default: 2.0)

    Returns:
        T x K x 3 smoothed keypoints
    """
    if keypoints_3d_seq is None or len(keypoints_3d_seq) < 2:
        return keypoints_3d_seq.copy() if keypoints_3d_seq is not None else keypoints_3d_seq
    T, K, D = keypoints_3d_seq.shape
    smoothed = np.zeros_like(keypoints_3d_seq, dtype=np.float64)
    indices = np.arange(T)
    for k in range(K):
        for d in range(D):
            traj = keypoints_3d_seq[:, k, d]
            valid = ~np.isnan(traj)
            if np.sum(valid) > 2:
                traj_interp = np.interp(indices, indices[valid], traj[valid])
                smoothed[:, k, d] = gaussian_filter1d(traj_interp, sigma=sigma, mode='nearest')
            else:
                smoothed[:, k, d] = traj
    return smoothed
