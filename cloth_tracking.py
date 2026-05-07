"""
Clean Cloth tracking experiment on cloth datasets.

Runs ONLY the full cloth tracker (no ablation, no CDCPD baseline).
Processes chunks with multiple clips, reinitializing the tracker per clip.
Cloth uses rectangle-aligned grid initialization with configurable N×N grid topology.

Key differences from cloth_batch_experiment.py:
- No ablation methods (NoSnap, NoGeometry removed)
- No CDCPD baseline comparison
- Single "Full" method only

Usage:
    python cloth_tracking.py --chunk 0 --clip_seconds 10 \
        --segment_interior_nodes "1,1,5,3,5,1,1,7"

Loads from: input_data/cloth/chunk_<N>/  (calibration in input_data/cloth/calibration/)

Author: Auto-generated
Date: 2026-05-06
"""

import argparse
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from cloth_tracker import ClothTracker


# ============================================================================
# POST-PROCESS: TRAJECTORY SMOOTHING
# ============================================================================

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


# ============================================================================
# CONSTANTS
# ============================================================================

# Paths (resolved relative to this script: input_data/cloth/chunk_<N>/)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE = SCRIPT_DIR / "input_data" / "cloth"
CALIB_DIR = DATA_BASE / "calibration"
OUTPUT_BASE = SCRIPT_DIR / "output" / "cloth"

# Frame rate
FPS = 30


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunk_data(chunk_dir: Path, max_frames: int = 600) -> dict:
    """Load all data from a chunk directory.

    Args:
        chunk_dir: Path to chunk directory
        max_frames: Maximum frames to load (from end of recording)

    Returns:
        Dictionary with color, depth, fg_mask, poses, and frame count
    """
    print(f"Loading data from {chunk_dir}...")

    # Load RGBD
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    n_total = rgbd['color'].shape[0]
    start_idx = max(0, n_total - max_frames)

    color = rgbd['color'][start_idx:]
    depth = rgbd['depth'][start_idx:]

    # Load foreground mask (from obtain_foreground_mask.py)
    fg_mask_path = chunk_dir / 'fg_masks' / 'masks.npz'
    if fg_mask_path.exists():
        fg_mask = np.load(fg_mask_path)['masks'][start_idx:]
    else:
        raise FileNotFoundError(f"fg_masks/masks.npz not found in {chunk_dir}. "
                                f"Run obtain_foreground_mask.py first.")

    # Load EE poses
    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')

    n_poses = len(left_poses_npz.files)
    pose_start_idx = max(0, n_poses - max_frames)
    n_frames = min(max_frames, n_poses, n_total)

    left_poses = np.array([left_poses_npz[f'arr_{i}']
                           for i in range(pose_start_idx, n_poses)])
    right_poses = np.array([right_poses_npz[f'arr_{i}']
                            for i in range(pose_start_idx, n_poses)])

    print(f"  Loaded {n_frames} frames")
    print(f"  Color shape: {color.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  FG mask shape: {fg_mask.shape}")

    return {
        'color': color,
        'depth': depth,
        'fg_mask': fg_mask,
        'left_poses': left_poses,
        'right_poses': right_poses,
        'n_frames': n_frames,
    }


def load_transforms(calib_dir: Path) -> dict:
    """Load camera-robot transforms and intrinsics."""
    tf = np.load(calib_dir / 'transform_ee_cam_world.npz')
    return {
        'T_left_base2cam': tf['T_left_base2cam'],
        'T_right_base2cam': tf['T_right_base2cam'],
        'K': tf['K'],
    }


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qw,qx,qy,qz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    quat = pose[3:]
    # scipy expects [qx, qy, qz, qw], but pose is [qw, qx, qy, qz]
    T[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def get_ee_positions_cam(left_pose, right_pose, T_left_base2cam, T_right_base2cam):
    """Convert EE poses to camera frame (mm)."""
    T_left_ee = pose7_to_matrix(left_pose)
    left_pos_cam = (T_left_base2cam @ T_left_ee)[:3, 3]
    T_right_ee = pose7_to_matrix(right_pose)
    right_pos_cam = (T_right_base2cam @ T_right_ee)[:3, 3]
    return np.array([left_pos_cam * 1000, right_pos_cam * 1000])  # m → mm


def filter_ee_outliers(ee_poses_3d, velocity_threshold=100.0, window_size=5):
    """Filter outlier EE positions using velocity-based detection."""
    from scipy.ndimage import median_filter

    filtered = ee_poses_3d.copy()
    outlier_frames = []
    n_frames = len(ee_poses_3d)

    for arm_idx in range(2):  # 0=left, 1=right
        positions = ee_poses_3d[:, arm_idx, :]  # (N, 3)

        # Compute frame-to-frame velocity (displacement)
        velocities = np.zeros(n_frames)
        for i in range(1, n_frames):
            velocities[i] = np.linalg.norm(positions[i] - positions[i-1])

        # Detect outliers: frames with velocity > threshold
        outlier_mask = velocities > velocity_threshold

        # Also check for sudden returns (outlier followed by return to normal)
        for i in range(1, n_frames - 1):
            if velocities[i] > velocity_threshold and velocities[i+1] > velocity_threshold * 0.5:
                outlier_mask[i] = True

        outlier_indices = np.where(outlier_mask)[0]

        if len(outlier_indices) > 0:
            arm_name = "left" if arm_idx == 0 else "right"
            print(f"  WARNING: Detected {len(outlier_indices)} outlier frames for {arm_name} EE: {outlier_indices.tolist()}")

            for idx in outlier_indices:
                outlier_frames.append((idx, arm_idx))

            # Interpolate outliers using neighboring valid frames
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

        # Optional: apply light median filtering to smooth remaining noise
        if window_size > 1:
            for dim in range(3):
                filtered[:, arm_idx, dim] = median_filter(filtered[:, arm_idx, dim], size=window_size)

    return filtered, outlier_frames


# ============================================================================
# METRICS
# ============================================================================

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

        # Skip edges with NaN nodes (from T-cropping)
        if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
            continue

        # Get reference length
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

    # Filter out NaN keypoints (from T-cropping)
    valid_mask = ~np.any(np.isnan(keypoints), axis=1)
    valid_keypoints = keypoints[valid_mask]

    if len(valid_keypoints) == 0:
        return {
            'distances': np.array([]),
            'rmse_mm': 0.0,
            'under_2mm': 0.0, 'under_5mm': 0.0, 'under_10mm': 0.0,
        }

    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud)
    distances, _ = nn.kneighbors(valid_keypoints)
    distances = distances.flatten()

    return {
        'distances': distances,
        'rmse_mm': np.sqrt(np.mean(distances ** 2)),
        'under_2mm': np.mean(distances < 2.0) * 100,
        'under_5mm': np.mean(distances < 5.0) * 100,
        'under_10mm': np.mean(distances < 10.0) * 100,
    }


def extract_surface_point_cloud(fg_mask, depth, intrinsics, max_points=5000):
    """Extract 3D point cloud from foreground mask."""
    if fg_mask is None or depth is None:
        return np.zeros((0, 3), dtype=np.float32)

    rows, cols = np.where(fg_mask > 0)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z_vals = depth[rows, cols].astype(np.float32)
    valid = z_vals > 0
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]

    if len(z_vals) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_vals = (cols - cx) * z_vals / fx
    y_vals = (rows - cy) * z_vals / fy

    pc = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)

    if len(pc) > max_points:
        indices = np.random.choice(len(pc), max_points, replace=False)
        pc = pc[indices]

    return pc


def save_border_init_visualization(
    keypoints: np.ndarray,
    border_grid_indices: list,
    detected_corners_3d: np.ndarray,
    contour_3d: np.ndarray,
    save_path: Path,
    segment_interior_nodes: list = None,
):
    """
    Simple visualization to check border initialization.

    Shows:
    - Green contour line
    - Yellow diamonds: detected corners (C0-C7)
    - Colored circles: border grid nodes (each segment different color)
    - Blue text: grid index at each border node
    """
    if keypoints is None or border_grid_indices is None:
        print("  [Border Viz] Missing data")
        return

    # Segment colors (8 distinct colors for 8 segments)
    segment_colors = [
        'red', 'blue', 'magenta', 'cyan',
        'orange', 'purple', 'brown', 'pink'
    ]

    traces = []

    # Contour (green line)
    if contour_3d is not None and len(contour_3d) > 0:
        traces.append(go.Scatter3d(
            x=contour_3d[:, 0], y=contour_3d[:, 1], z=contour_3d[:, 2],
            mode='lines',
            line=dict(color='green', width=3),
            name='Contour',
        ))

    # Detected corners (yellow diamonds with C0-C7 labels)
    if detected_corners_3d is not None and len(detected_corners_3d) > 0:
        traces.append(go.Scatter3d(
            x=detected_corners_3d[:, 0], y=detected_corners_3d[:, 1], z=detected_corners_3d[:, 2],
            mode='markers+text',
            marker=dict(size=14, color='yellow', symbol='diamond',
                       line=dict(color='black', width=2)),
            text=[f'C{i}' for i in range(len(detected_corners_3d))],
            textposition='top center',
            textfont=dict(size=14, color='black'),
            name=f'Corners ({len(detected_corners_3d)})',
        ))

        # Straight lines between each consecutive corner pair (dashed, per-segment color)
        n_c = len(detected_corners_3d)
        for seg_idx in range(n_c):
            c_start = detected_corners_3d[seg_idx]
            c_end = detected_corners_3d[(seg_idx + 1) % n_c]
            if np.any(np.isnan(c_start)) or np.any(np.isnan(c_end)):
                continue
            color = segment_colors[seg_idx % len(segment_colors)]
            traces.append(go.Scatter3d(
                x=[c_start[0], c_end[0]], y=[c_start[1], c_end[1]], z=[c_start[2], c_end[2]],
                mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                name=f'Line C{seg_idx}→C{(seg_idx+1)%n_c}',
                showlegend=False,
            ))

    # Border grid nodes - color by segment
    border_pts = []
    border_labels = []
    valid_indices = []
    n_nan_border = 0
    for i, grid_idx in enumerate(border_grid_indices):
        if grid_idx < len(keypoints) and not np.any(np.isnan(keypoints[grid_idx])):
            border_pts.append(keypoints[grid_idx])
            border_labels.append(f'{grid_idx}')
            valid_indices.append(i)
        else:
            n_nan_border += 1
    print(f"  [Border Viz] border_grid_indices has {len(border_grid_indices)} entries, "
          f"{len(border_pts)} valid, {n_nan_border} NaN")

    if len(border_pts) > 0:
        border_pts = np.array(border_pts)

        # Determine segment for each border node
        if segment_interior_nodes is not None:
            segment_starts = [0]
            for n_int in segment_interior_nodes:
                segment_starts.append(segment_starts[-1] + 1 + n_int)

            node_colors = []
            for orig_idx in valid_indices:
                seg_idx = 0
                for s in range(len(segment_starts) - 1):
                    if segment_starts[s] <= orig_idx < segment_starts[s + 1]:
                        seg_idx = s
                        break
                node_colors.append(segment_colors[seg_idx % len(segment_colors)])
        else:
            node_colors = ['orange'] * len(border_pts)

        traces.append(go.Scatter3d(
            x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
            mode='markers+text',
            marker=dict(size=12, color=node_colors, symbol='circle',
                       line=dict(color='black', width=2)),
            text=border_labels,
            textposition='bottom center',
            textfont=dict(size=10, color='blue'),
            name=f'Border Nodes ({len(border_pts)})',
            hovertext=[f'Border[{valid_indices[i]}] → Grid {border_grid_indices[valid_indices[i]]}' for i in range(len(border_pts))],
            hoverinfo='text',
        ))

        # Draw edges between consecutive border nodes, colored by segment
        if segment_interior_nodes is not None:
            segment_starts = [0]
            for n_int in segment_interior_nodes:
                segment_starts.append(segment_starts[-1] + 1 + n_int)

            for seg_idx in range(len(segment_interior_nodes)):
                seg_start = segment_starts[seg_idx]
                seg_end = segment_starts[seg_idx + 1]
                color = segment_colors[seg_idx % len(segment_colors)]

                edge_x, edge_y, edge_z = [], [], []
                for orig_i in range(seg_start, seg_end):
                    orig_j = orig_i + 1
                    if orig_j >= len(border_grid_indices):
                        orig_j = 0  # Wrap around

                    if orig_i in valid_indices and orig_j in valid_indices:
                        vi = valid_indices.index(orig_i)
                        vj = valid_indices.index(orig_j)
                        edge_x.extend([border_pts[vi, 0], border_pts[vj, 0], None])
                        edge_y.extend([border_pts[vi, 1], border_pts[vj, 1], None])
                        edge_z.extend([border_pts[vi, 2], border_pts[vj, 2], None])

                if edge_x:
                    traces.append(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color=color, width=4),
                        name=f'Seg {seg_idx}',
                        showlegend=True,
                    ))
        else:
            edge_x, edge_y, edge_z = [], [], []
            for i in range(len(border_pts)):
                j = (i + 1) % len(border_pts)
                edge_x.extend([border_pts[i, 0], border_pts[j, 0], None])
                edge_y.extend([border_pts[i, 1], border_pts[j, 1], None])
                edge_z.extend([border_pts[i, 2], border_pts[j, 2], None])

            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='orange', width=4, dash='dash'),
                name='Border Chain',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Border Init: {len(border_grid_indices)} samples → {len(border_pts)} grid nodes',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [Border Viz] Saved: {save_path}")


def save_init_visualization_3d(
    keypoints: np.ndarray,
    edges: list,
    point_cloud: np.ndarray,
    save_path: Path,
    corner_indices: list = None,
    border_indices: list = None,
    downsample_pc: int = 2000,
    contour_3d: np.ndarray = None,
    contour_3d_raw: np.ndarray = None,
    ee_poses: np.ndarray = None,
    segment_lengths: dict = None,
    rect_corners_3d: np.ndarray = None,
    detected_corners_3d: np.ndarray = None,
    all_grid_edges: list = None,
    border_grid_indices: list = None,
    valid_faces: list = None,
):
    """Save interactive 3D visualization of initialization using Plotly."""
    if keypoints is None or len(keypoints) == 0:
        print("  [Init Vis] No keypoints to visualize")
        return

    traces = []

    # Downsample point cloud if needed
    if point_cloud is not None and len(point_cloud) > 0:
        pc = point_cloud.copy()
        if len(pc) > downsample_pc:
            indices = np.random.choice(len(pc), downsample_pc, replace=False)
            pc = pc[indices]

        traces.append(go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=pc[:, 2],
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='Depth (mm)', x=1.02, len=0.5),
            ),
            name='Point Cloud',
            hoverinfo='skip',
        ))

    # Raw/noisy contour trace (red dashed line)
    if contour_3d_raw is not None and len(contour_3d_raw) > 0:
        contour_raw_vis = contour_3d_raw[::5] if len(contour_3d_raw) > 200 else contour_3d_raw
        contour_raw_vis = np.vstack([contour_raw_vis, contour_raw_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_raw_vis[:, 0], y=contour_raw_vis[:, 1], z=contour_raw_vis[:, 2],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Raw Contour',
            hoverinfo='skip',
        ))

    # Denoised contour trace (green solid line)
    if contour_3d is not None and len(contour_3d) > 0:
        contour_vis = contour_3d[::5] if len(contour_3d) > 200 else contour_3d
        contour_vis = np.vstack([contour_vis, contour_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_vis[:, 0], y=contour_vis[:, 1], z=contour_vis[:, 2],
            mode='lines',
            line=dict(color='lime', width=5),
            name='Contour',
            hoverinfo='skip',
        ))

    # Detected corners on contour
    if detected_corners_3d is not None and len(detected_corners_3d) > 0:
        valid_corners = ~np.any(np.isnan(detected_corners_3d), axis=1)
        corners_valid = detected_corners_3d[valid_corners]
        corner_labels = [f'C{i}' for i in range(len(detected_corners_3d))]
        valid_labels = [corner_labels[i] for i in range(len(detected_corners_3d)) if valid_corners[i]]
        if len(corners_valid) > 0:
            traces.append(go.Scatter3d(
                x=corners_valid[:, 0], y=corners_valid[:, 1], z=corners_valid[:, 2],
                mode='markers',
                marker=dict(size=14, color='yellow', symbol='diamond',
                           line=dict(color='black', width=2)),
                name='Detected Corners',
                text=valid_labels,
                hoverinfo='text',
            ))

    # Straight lines between consecutive corner pairs
    if detected_corners_3d is not None and len(detected_corners_3d) > 1:
        seg_colors = ['red', 'blue', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink']
        n_c = len(detected_corners_3d)
        for seg_idx in range(n_c):
            c_s = detected_corners_3d[seg_idx]
            c_e = detected_corners_3d[(seg_idx + 1) % n_c]
            if np.any(np.isnan(c_s)) or np.any(np.isnan(c_e)):
                continue
            color = seg_colors[seg_idx % len(seg_colors)]
            traces.append(go.Scatter3d(
                x=[c_s[0], c_e[0]], y=[c_s[1], c_e[1]], z=[c_s[2], c_e[2]],
                mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                name=f'Line C{seg_idx}→C{(seg_idx+1)%n_c}',
                showlegend=False,
            ))

    # EE poses (purple)
    if ee_poses is not None and len(ee_poses) > 0:
        valid_ee = ~np.any(np.isnan(ee_poses), axis=1)
        ee_valid = ee_poses[valid_ee]
        ee_idx = np.where(valid_ee)[0]
        if len(ee_valid) > 0:
            traces.append(go.Scatter3d(
                x=ee_valid[:, 0], y=ee_valid[:, 1], z=ee_valid[:, 2],
                mode='markers',
                marker=dict(size=12, color='purple', symbol='diamond'),
                name='EE Poses',
                text=[f'EE{i}' for i in ee_idx],
                hoverinfo='text',
            ))

    # Bounding rectangle (cyan dashed line)
    if rect_corners_3d is not None and len(rect_corners_3d) == 4:
        rect_closed = np.vstack([rect_corners_3d, rect_corners_3d[0:1]])
        traces.append(go.Scatter3d(
            x=rect_closed[:, 0], y=rect_closed[:, 1], z=rect_closed[:, 2],
            mode='lines+markers',
            line=dict(color='cyan', width=6, dash='dash'),
            marker=dict(size=10, color='cyan', symbol='square'),
            name='Bounding Rect',
            text=['TL', 'TR', 'BR', 'BL', 'TL'],
            hoverinfo='text',
        ))

    # Full rectangular grid edges (gray)
    grid_positions = None
    if all_grid_edges is not None and len(all_grid_edges) > 0:
        full_grid_x, full_grid_y, full_grid_z = [], [], []
        n_grid = int(np.sqrt(len(keypoints)))
        if rect_corners_3d is not None and len(rect_corners_3d) == 4:
            TL, TR, BR, BL = rect_corners_3d
            grid_positions = np.zeros((len(keypoints), 3))
            for idx in range(len(keypoints)):
                row, col = idx // n_grid, idx % n_grid
                u = col / (n_grid - 1)
                v = row / (n_grid - 1)
                top = (1 - u) * TL + u * TR
                bottom = (1 - u) * BL + u * BR
                grid_positions[idx] = (1 - v) * top + v * bottom

            for i, j in all_grid_edges:
                if i < len(grid_positions) and j < len(grid_positions):
                    full_grid_x.extend([grid_positions[i, 0], grid_positions[j, 0], None])
                    full_grid_y.extend([grid_positions[i, 1], grid_positions[j, 1], None])
                    full_grid_z.extend([grid_positions[i, 2], grid_positions[j, 2], None])

            traces.append(go.Scatter3d(
                x=full_grid_x, y=full_grid_y, z=full_grid_z,
                mode='lines',
                line=dict(color='lightgray', width=1),
                name='Bilinear Grid (reference)',
                hoverinfo='skip',
                opacity=0.4,
            ))

    # Quad faces (semi-transparent mesh)
    if valid_faces is not None and len(valid_faces) > 0:
        vert_set = set()
        for tl, tr, br, bl in valid_faces:
            vert_set.update([tl, tr, br, bl])
        vert_list = sorted(vert_set)
        vert_remap = {v: i for i, v in enumerate(vert_list)}
        mesh_verts = keypoints[vert_list]

        tri_i, tri_j, tri_k = [], [], []
        for tl, tr, br, bl in valid_faces:
            if any(np.any(np.isnan(keypoints[idx])) for idx in [tl, tr, br, bl]):
                continue
            a, b, c, d = vert_remap[tl], vert_remap[tr], vert_remap[br], vert_remap[bl]
            tri_i.extend([a, a])
            tri_j.extend([b, c])
            tri_k.extend([c, d])

        if tri_i:
            traces.append(go.Mesh3d(
                x=mesh_verts[:, 0], y=mesh_verts[:, 1], z=mesh_verts[:, 2],
                i=tri_i, j=tri_j, k=tri_k,
                color='lightskyblue',
                opacity=0.3,
                name=f'Faces ({len(valid_faces)} quads)',
                hoverinfo='skip',
            ))

    # T-Line edges (blue)
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                continue
            edge_x.extend([keypoints[i, 0], keypoints[j, 0], None])
            edge_y.extend([keypoints[i, 1], keypoints[j, 1], None])
            edge_z.extend([keypoints[i, 2], keypoints[j, 2], None])

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='T-Line Edges',
        hoverinfo='skip',
    ))

    # ACTUAL keypoint positions with indices — split by corner / border / interior
    valid_idx = [i for i in range(len(keypoints)) if not np.any(np.isnan(keypoints[i]))]
    n_grid = int(np.sqrt(len(keypoints)))
    corner_set = set(corner_indices) if corner_indices else set()
    border_set = set(border_indices) if border_indices else set()

    corner_idx = [i for i in valid_idx if i in corner_set]
    border_idx = [i for i in valid_idx if i in border_set and i not in corner_set]
    interior_idx = [i for i in valid_idx if i not in corner_set and i not in border_set]

    print(f"  [Init Vis] Corners: {corner_idx}")
    print(f"  [Init Vis] Border (non-corner): {border_idx}")
    print(f"  [Init Vis] Interior: {interior_idx}")

    if corner_idx:
        pts = keypoints[corner_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='purple', symbol='diamond', opacity=1.0),
            text=[str(i) for i in corner_idx],
            textposition='top center',
            textfont=dict(size=11, color='purple'),
            name=f'Corners ({len(corner_idx)})',
            hovertext=[f'Corner idx {i} = [{i//n_grid},{i%n_grid}]' for i in corner_idx],
            hoverinfo='text',
        ))
    if border_idx:
        pts = keypoints[border_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=12, color='gold', opacity=0.9),
            text=[str(i) for i in border_idx],
            textposition='top center',
            textfont=dict(size=9, color='goldenrod'),
            name=f'Border ({len(border_idx)})',
            hovertext=[f'Border idx {i} = [{i//n_grid},{i%n_grid}]' for i in border_idx],
            hoverinfo='text',
        ))
    if interior_idx:
        pts = keypoints[interior_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='green', opacity=0.9),
            text=[str(i) for i in interior_idx],
            textposition='top center',
            textfont=dict(size=8, color='green'),
            name=f'Interior ({len(interior_idx)})',
            hovertext=[f'Interior idx {i} = [{i//n_grid},{i%n_grid}]' for i in interior_idx],
            hoverinfo='text',
        ))

    # Border nodes (orange circles)
    if border_grid_indices is not None and len(border_grid_indices) > 0:
        border_valid_idx = []
        for idx, grid_idx in enumerate(border_grid_indices):
            if grid_idx < len(keypoints) and not np.any(np.isnan(keypoints[grid_idx])):
                border_valid_idx.append(idx)

        if len(border_valid_idx) > 0:
            border_grid_nodes = [border_grid_indices[i] for i in border_valid_idx]
            border_pts = keypoints[border_grid_nodes]
            traces.append(go.Scatter3d(
                x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
                mode='markers',
                marker=dict(size=12, color='orange', symbol='circle',
                           line=dict(color='black', width=2)),
                name=f'Border ({len(border_grid_nodes)} nodes)',
                hovertext=[f'Border[{i}] = Grid {border_grid_indices[i]}' for i in border_valid_idx],
                hoverinfo='text',
            ))

            border_edge_x, border_edge_y, border_edge_z = [], [], []
            for i in range(len(border_grid_nodes)):
                j = (i + 1) % len(border_grid_nodes)
                p1 = keypoints[border_grid_nodes[i]]
                p2 = keypoints[border_grid_nodes[j]]
                border_edge_x.extend([p1[0], p2[0], None])
                border_edge_y.extend([p1[1], p2[1], None])
                border_edge_z.extend([p1[2], p2[2], None])

            traces.append(go.Scatter3d(
                x=border_edge_x, y=border_edge_y, z=border_edge_z,
                mode='lines',
                line=dict(color='orange', width=3, dash='dash'),
                name='Border Chain',
                hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)

    # Compute edge length stats for title (filter NaN)
    edge_lengths = []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            if not np.any(np.isnan(keypoints[i])) and not np.any(np.isnan(keypoints[j])):
                edge_lengths.append(np.linalg.norm(keypoints[i] - keypoints[j]))

    n_valid = np.sum(~np.any(np.isnan(keypoints), axis=1))
    n_valid_edges = len(edge_lengths)

    if edge_lengths:
        avg_len = np.mean(edge_lengths)
        std_len = np.std(edge_lengths)
        n_faces = len(valid_faces) if valid_faces else 0
        title = f'Init: {n_valid} nodes, {n_valid_edges} edges, {n_faces} faces | Avg edge: {avg_len:.1f}mm, Std: {std_len:.1f}mm ({std_len/avg_len*100:.1f}%)'
    else:
        n_faces = len(valid_faces) if valid_faces else 0
        title = f'Init: {n_valid} nodes, {n_valid_edges} edges, {n_faces} faces'

    if segment_lengths:
        seg_str = ' | Seg: ' + ', '.join([f'{k}:{v:.0f}mm' for k, v in segment_lengths.items()])
        title += seg_str

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [Init Vis] Saved 3D visualization to {save_path}")


def sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=10, valid_faces=None):
    """Sample points uniformly on quad faces for Chamfer distance."""
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if valid_faces is None:
        valid_faces = []
        if len(keypoints) == grid_rows * grid_cols:
            for r in range(grid_rows - 1):
                for c in range(grid_cols - 1):
                    tl = r * grid_cols + c
                    tr = r * grid_cols + c + 1
                    bl = (r + 1) * grid_cols + c
                    br = (r + 1) * grid_cols + c + 1
                    if not any(np.any(np.isnan(keypoints[idx])) for idx in [tl, tr, bl, br]):
                        valid_faces.append((tl, tr, br, bl))

    sampled_points = []
    for tl, tr, br, bl in valid_faces:
        p_tl, p_tr, p_br, p_bl = keypoints[tl], keypoints[tr], keypoints[br], keypoints[bl]
        if any(np.any(np.isnan(p)) for p in [p_tl, p_tr, p_br, p_bl]):
            continue

        for _ in range(n_samples_per_face):
            u = np.random.random()
            v = np.random.random()
            p_top = (1 - u) * p_tl + u * p_tr
            p_bot = (1 - u) * p_bl + u * p_br
            sampled_points.append((1 - v) * p_top + v * p_bot)

    if len(sampled_points) == 0:
        return keypoints[~np.any(np.isnan(keypoints), axis=1)].copy()

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

    valid_pred = pred_cloud[~np.any(np.isnan(pred_cloud), axis=1)]
    valid_ref = ref_cloud[~np.any(np.isnan(ref_cloud), axis=1)]

    if len(valid_pred) == 0 or len(valid_ref) == 0:
        return empty_result

    nn_ref = NearestNeighbors(n_neighbors=1).fit(valid_ref)
    pred2ref_dists, _ = nn_ref.kneighbors(valid_pred)
    pred2ref_dists = pred2ref_dists.flatten()

    nn_pred = NearestNeighbors(n_neighbors=1).fit(valid_pred)
    ref2pred_dists, _ = nn_pred.kneighbors(valid_ref)
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


# ============================================================================
# CLOTH TRACKER (FULL METHOD)
# ============================================================================

class ClothTrackerFull(ClothTracker):
    """ClothTracker with the FULL pipeline (snap + geometry + EE constraint, no CPD).

    Corner treatment:
        - Corners are NEVER replaced with EE FK positions (may have calibration errors)
        - EE association is established to know which 2 corners are grasped (EE corners)
        - EE corners: FIXED during optimization
        - Non-EE corners: Constrained to move along contour (like border nodes)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Full configuration: snap on, geometry on, EE on, CPD off
        self.enable_snap = True
        self.enable_geometry_constraint = True
        self.enable_ee_constraint = True
        self.enable_cpd = False

    @property
    def ee_corner_indices(self) -> list:
        """Get the corner indices that are EE-mapped (should be fixed)."""
        if self.ee_to_corner_mapping is None:
            return []
        return list(self.ee_to_corner_mapping.values())

    @property
    def non_ee_corner_indices(self) -> list:
        """Get corner indices NOT EE-mapped (should be free to move)."""
        ee_corners = set(self.ee_corner_indices)
        return [idx for idx in self.CORNER_INDICES if idx not in ee_corners]

    def _joint_constraint_optimization_with_contour_full(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray,
        fixed_set: set = None,
    ) -> np.ndarray:
        """
        Geometry optimization with contour constraints.

        - fixed_set nodes (C0, C1): FIXED — not moved at all
        - Other corner + border nodes: constrained to contour (can slide along it)
        - Interior nodes: soft project to point cloud
        """
        K = len(keypoints)
        if fixed_set is None:
            fixed_set = set()

        # Build nearest neighbor structures
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)

        # Full contour NN
        contour_nn = None
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)

        # Contour-constrained nodes: corners + borders, minus fixed
        corner_set = set(self.contour_corner_grid_indices) if self.contour_corner_grid_indices else set()
        border_set = set(self.border_grid_indices) if self.border_grid_indices else set()
        contour_set = (corner_set | border_set) - fixed_set

        for outer_iter in range(self.n_outer_iterations):
            # Edge length correction (skip NaN nodes)
            for edge_iter in range(self.n_edge_iterations):
                for (i, j), target_length in self.reference_lengths.items():
                    if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                        continue

                    current_vec = keypoints[j] - keypoints[i]
                    current_length = np.linalg.norm(current_vec)

                    if current_length < 1e-6:
                        continue

                    error = (current_length - target_length) / target_length

                    if abs(error) > self.edge_tolerance:
                        direction = current_vec / current_length
                        correction = (current_length - target_length) * self.edge_weight / 2

                        i_fixed = i in fixed_set
                        j_fixed = j in fixed_set

                        if not i_fixed and not j_fixed:
                            keypoints[i] += correction * direction
                            keypoints[j] -= correction * direction
                        elif not i_fixed:
                            keypoints[i] += 2 * correction * direction
                        elif not j_fixed:
                            keypoints[j] -= 2 * correction * direction

            # Projection step
            for i in range(K):
                if np.any(np.isnan(keypoints[i])):
                    continue

                if i in fixed_set:
                    continue  # C0, C1: fixed from detection

                if i in contour_set:
                    # Other corners + border nodes: snap to contour
                    if contour_nn is not None:
                        _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = contour_3d[idx[0, 0]]
                else:
                    # Interior nodes: soft projection to point cloud
                    _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
                    nearest = point_cloud[idx[0, 0]]
                    alpha = 0.3
                    keypoints[i] = (1 - alpha) * keypoints[i] + alpha * nearest

        return keypoints

    def track(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        frame_idx: int,
    ) -> dict:
        """Track with the full pipeline.

        - Run corner detection each frame
        - C0 (top-left-most) and C1 (top-right-most) are hard-replaced and FIXED
        - Other corners + borders: project from previous positions onto contour
        - Optimization: C0/C1 fixed, other contour nodes slide on contour
        """
        t_start = time.time()

        # Extract point cloud
        point_cloud = self._extract_point_cloud(mask, depth)

        if len(point_cloud) < self.min_foreground_pixels:
            self.consecutive_skips += 1
            return {
                'success': False,
                'reason': 'insufficient_points',
                'mode': 'skip',
            }

        # Extract 3D contour
        contour_3d = self._extract_contour_3d(mask, depth)

        # Run corner detection
        _, detected_corners_3d = self._detect_contour_corners(mask, depth)

        # Start from previous keypoints (CPD disabled)
        t_cpd_start = time.time()
        keypoints = self.prev_keypoints.copy()
        cpd_time = time.time() - t_cpd_start

        # --- Detect 4 anchor corners from detected corners ---
        ordered_corners = getattr(self, 'ordered_corner_grid_indices', None)
        fixed_set = set()
        if (detected_corners_3d is not None
                and len(detected_corners_3d) >= 4 and ordered_corners and len(ordered_corners) >= 5):
            x_vals = detected_corners_3d[:, 0]
            y_vals = detected_corners_3d[:, 1]
            c0_det = detected_corners_3d[np.argmin(x_vals + y_vals)]  # top-left-most
            c1_det = detected_corners_3d[np.argmax(x_vals - y_vals)]  # top-right-most
            c2_det = detected_corners_3d[np.argmin(x_vals - y_vals)]  # bottom-left-most
            c3_det = detected_corners_3d[np.argmax(x_vals + y_vals)]  # bottom-right-most

            # Hard-replace grid nodes with detected positions
            keypoints[ordered_corners[0]] = c0_det.copy()
            keypoints[ordered_corners[-1]] = c1_det.copy()
            keypoints[ordered_corners[3]] = c2_det.copy()
            keypoints[ordered_corners[4]] = c3_det.copy()

            fixed_set = {ordered_corners[0], ordered_corners[-1],
                         ordered_corners[3], ordered_corners[4]}

            # Store for visualization (4 points: C0, C7, C3, C4)
            self.detected_corners_3d = np.array([c0_det, c1_det, c2_det, c3_det])

        # Project other corners + borders onto current contour (skip C0/C1)
        if contour_3d is not None and len(contour_3d) > 0:
            for idx in (self.contour_corner_grid_indices or []):
                if idx in fixed_set:
                    continue
                if not np.any(np.isnan(keypoints[idx])):
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

            for idx in (self.border_grid_indices or []):
                if not np.any(np.isnan(keypoints[idx])):
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Geometry constraint optimization
        t_geom_start = time.time()
        keypoints = self._joint_constraint_optimization_with_contour_full(
            keypoints, point_cloud, contour_3d, fixed_set=fixed_set
        )
        geom_time = time.time() - t_geom_start

        # Final snap: other corners + borders back to contour (skip C0/C1)
        if contour_3d is not None and len(contour_3d) > 0:
            for idx in (self.contour_corner_grid_indices or []):
                if idx in fixed_set:
                    continue
                if not np.any(np.isnan(keypoints[idx])):
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)
            for idx in (self.border_grid_indices or []):
                if not np.any(np.isnan(keypoints[idx])):
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Update state
        self.prev_keypoints = keypoints.copy()
        self.frame_count += 1
        self.consecutive_skips = 0

        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)

        # Compute edge errors
        edge_errors = self._compute_edge_errors(keypoints)

        track_time = time.time() - t_start

        return {
            'success': True,
            'mode': 'track',
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.valid_edges,
            'edge_errors': edge_errors,
            'timing': {
                'cpd': cpd_time,
                'geom': geom_time,
                'total': track_time,
            },
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_full_tracking_visualization(rgb, mask, keypoints_2d, edges, frame_idx, mode,
                                       traj_history=None, tail_length=60,
                                       corner_indices=None, border_indices=None,
                                       detected_corners_2d=None):
    """Create 4-panel visualization (matches fabric_batch_experiment layout)."""
    H, W = rgb.shape[:2]

    MASK_COLOR = (255, 0, 0)  # Blue (BGR) for contour
    EDGE_COLOR = (255, 165, 0)
    CORNER_COLOR = (255, 0, 0)
    BORDER_COLOR = (255, 255, 0)
    INTERIOR_COLOR = (0, 255, 255)
    TAIL_COLOR = (144, 238, 144)  # Light green (BGR)

    corner_indices = corner_indices or []
    border_indices = border_indices or []

    # Panel 1: Binary mask
    panel1 = np.zeros((H, W, 3), dtype=np.uint8)
    if mask is not None:
        panel1[mask > 0] = MASK_COLOR

    # Panel 2: Mask overlay
    panel2 = rgb.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, contours, -1, MASK_COLOR, 2)
    cv2.putText(panel2, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel2, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw 4 anchor corners on panel 2: C0, C7, C3, C4
    if detected_corners_2d is not None and len(detected_corners_2d) >= 4:
        anchor_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        anchor_labels = ["C0", "C7", "C3", "C4"]
        for k in range(4):
            dc = detected_corners_2d[k]
            if np.any(np.isnan(dc)):
                continue
            pt = tuple(dc.astype(int))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                sz = 10
                diamond = np.array([
                    [pt[0], pt[1] - sz],
                    [pt[0] + sz, pt[1]],
                    [pt[0], pt[1] + sz],
                    [pt[0] - sz, pt[1]],
                ], dtype=np.int32)
                cv2.fillPoly(panel2, [diamond], anchor_colors[k])
                cv2.polylines(panel2, [diamond], True, (0, 0, 0), 1)
                cv2.putText(panel2, anchor_labels[k], (pt[0] + 12, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 3: Keypoints only
    panel3 = np.zeros((H, W, 3), dtype=np.uint8)
    if mask is not None:
        panel3[mask > 0] = [30, 30, 30]

    def draw_keypoints_and_edges(canvas):
        if traj_history is not None and len(traj_history) > 1:
            history = np.array(traj_history)
            n_hist = len(history)
            start = max(0, n_hist - tail_length)
            for t in range(start, n_hist - 1):
                alpha = (t - start) / (n_hist - start)
                color = tuple(int(c * alpha) for c in TAIL_COLOR)
                for k in range(history.shape[1]):
                    if np.any(np.isnan(history[t, k])) or np.any(np.isnan(history[t + 1, k])):
                        continue
                    pt1 = tuple(history[t, k].astype(int))
                    pt2 = tuple(history[t + 1, k].astype(int))
                    if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
                        cv2.line(canvas, pt1, pt2, color, 2)

        if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
            for (i, j) in edges:
                if i < len(keypoints_2d) and j < len(keypoints_2d):
                    if np.any(np.isnan(keypoints_2d[i])) or np.any(np.isnan(keypoints_2d[j])):
                        continue
                    pt1 = tuple(keypoints_2d[i].astype(int))
                    pt2 = tuple(keypoints_2d[j].astype(int))
                    cv2.line(canvas, pt1, pt2, EDGE_COLOR, 2)

            for idx in range(len(keypoints_2d)):
                if np.any(np.isnan(keypoints_2d[idx])):
                    continue
                pt = tuple(keypoints_2d[idx].astype(int))
                if idx in corner_indices:
                    color = CORNER_COLOR
                elif idx in border_indices:
                    color = BORDER_COLOR
                else:
                    color = INTERIOR_COLOR
                cv2.circle(canvas, pt, 5, color, -1)

    draw_keypoints_and_edges(panel3)

    # Panel 4: Full overlay
    panel4 = rgb.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel4, contours, -1, MASK_COLOR, 2)
    draw_keypoints_and_edges(panel4)
    cv2.putText(panel4, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel4, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    row1 = np.concatenate([panel1, panel2], axis=1)
    row2 = np.concatenate([panel3, panel4], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    return grid


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame,
                 output_dir, grid_rows, grid_cols, tail_length=60, fps=30,
                 segment_interior_nodes=None, sigma=2.0):
    """Process a single clip with the full method only.

    Pipeline:
      1. Track every frame, compute metrics on raw keypoints, save raw 3D.
      2. Apply Gaussian smoothing (sigma) along the time axis to the 3D keypoints.
      3. Re-project smoothed 3D to 2D and render the tracking video from those.
      4. Save smoothed 3D in a separate `smoothed_3d_keypoints.npz`.
    Evaluation metrics are always computed on the RAW keypoints, never smoothed.
    """
    clip_dir = output_dir / f"clip_{clip_idx:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    n_frames = end_frame - start_frame
    K = transforms['K']

    color = data['color'][start_frame:end_frame]
    depth = data['depth'][start_frame:end_frame]
    fg_mask = data['fg_mask'][start_frame:end_frame]

    clip_ee_poses = ee_poses_3d[start_frame:end_frame] if ee_poses_3d is not None else None

    # Tracker parameters (same as cloth_batch_experiment.py)
    tracker_params = {
        'intrinsics': K,
        'max_depth': 2000.0,
        'cpd_beta': 10.0,
        'cpd_lambda': 2.0,
        'cpd_w': 0.1,
        'cpd_max_iter': 100,
        'cpd_tol': 1e-3,
        'cpd_downsample': 2000,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.5,
        'edge_tolerance': 0.01,
        'repulsion_iterations': 500,
        'repulsion_lr': 5.0,
        'ee_poses_3d': clip_ee_poses,
        'segment_interior_nodes': segment_interior_nodes,
    }

    method_name = 'Full'

    # Initialize tracker
    tracker = ClothTrackerFull(**tracker_params)

    # Storage
    results = {
        'keypoints': [],
        'keypoints_2d': [],
        'traj_history': [],
        'modes': [],
        'edge_metrics': [],
        'pos_metrics': [],
        'cd_metrics': [],
    }

    reference_lengths = None
    init_vis_saved = False

    # Note: tracking video writer is created post-loop, after smoothing
    H, W = color.shape[1:3]

    # Per-frame visualization context captured during tracking, used by post-loop renderer
    render_state = []   # list of {mode, detected_corners_2d}

    print(f"\n  Processing clip {clip_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")

    for frame_idx in tqdm(range(n_frames), desc=f"  Clip {clip_idx}"):
        rgb = cv2.cvtColor(color[frame_idx], cv2.COLOR_BGR2RGB)
        d = depth[frame_idx]
        mask = fg_mask[frame_idx]

        surface_pc = extract_surface_point_cloud(mask, d, K)

        result = tracker.process_frame(d, mask, frame_idx)

        mode = result.get('mode', 'unknown')
        keypoints = result.get('keypoints')
        keypoints_2d = result.get('keypoints_2d')
        edges = result.get('edges', tracker.valid_edges if hasattr(tracker, 'valid_edges') else [])

        results['keypoints'].append(keypoints)
        results['keypoints_2d'].append(keypoints_2d)
        results['modes'].append(mode)

        if keypoints_2d is not None:
            results['traj_history'].append(keypoints_2d.copy())

        if reference_lengths is None and result.get('success'):
            if hasattr(tracker, 'reference_lengths') and tracker.reference_lengths is not None:
                reference_lengths = tracker.reference_lengths

        # Save init visualization (once)
        if mode == 'init' and not init_vis_saved and keypoints is not None:
            # Extract full fg point cloud with stride 8 for visualization
            rows, cols = np.where(mask > 0)
            if len(rows) > 0:
                z_vals = d[rows, cols].astype(np.float32)
                valid = z_vals > 0
                rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
                rows, cols, z_vals = rows[::8], cols[::8], z_vals[::8]
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                x_vals = (cols - cx) * z_vals / fx
                y_vals = (rows - cy) * z_vals / fy
                fg_pc_full = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)
            else:
                fg_pc_full = surface_pc

            # Extract raw and denoised contours for visualization
            contour_3d_raw = None
            contour_3d_vis = None
            segment_lengths = None

            corners_3d_vis = None
            if hasattr(tracker, 'detected_corners_3d') and tracker.detected_corners_3d is not None:
                corners_3d_vis = tracker.detected_corners_3d
            elif hasattr(tracker, '_find_mask_corners'):
                corners_2d = tracker._find_mask_corners(mask, d)
                corners_3d_vis = tracker._pixel_to_3d(corners_2d, d) if corners_2d is not None else None

            if hasattr(tracker, '_extract_contour_3d'):
                if hasattr(tracker, '_extract_contour_3d_raw'):
                    contour_3d_raw = tracker._extract_contour_3d_raw(mask, d)
                contour_3d_vis = tracker._extract_contour_3d(mask, d, corners_3d=corners_3d_vis)
                if hasattr(tracker, '_compute_contour_segment_lengths') and contour_3d_vis is not None and corners_3d_vis is not None:
                    segment_lengths = tracker._compute_contour_segment_lengths(contour_3d_vis, corners_3d_vis)

            ee_poses_frame = clip_ee_poses[frame_idx] if clip_ee_poses is not None else None

            rect_corners = None
            if hasattr(tracker, 'rect_corners_3d') and tracker.rect_corners_3d is not None:
                rect_corners = tracker.rect_corners_3d

            detected_corners = None
            if hasattr(tracker, 'detected_corners_3d') and tracker.detected_corners_3d is not None:
                detected_corners = tracker.detected_corners_3d

            border_grid = None
            if hasattr(tracker, 'border_grid_indices') and tracker.border_grid_indices is not None:
                border_grid = tracker.border_grid_indices

            save_init_visualization_3d(
                keypoints=keypoints,
                edges=edges,
                point_cloud=fg_pc_full,
                save_path=clip_dir / 'init_3d.html',
                corner_indices=getattr(tracker, 'contour_corner_grid_indices', None) or getattr(tracker, 'CORNER_INDICES', []),
                border_indices=getattr(tracker, 'border_grid_indices', None) or getattr(tracker, 'BORDER_INDICES', []),
                downsample_pc=50000,
                contour_3d=contour_3d_vis,
                contour_3d_raw=contour_3d_raw,
                ee_poses=ee_poses_frame,
                segment_lengths=segment_lengths,
                rect_corners_3d=rect_corners,
                detected_corners_3d=detected_corners,
                all_grid_edges=tracker.grid_edges if hasattr(tracker, 'grid_edges') else None,
                border_grid_indices=border_grid,
                valid_faces=tracker.valid_faces if hasattr(tracker, 'valid_faces') else None,
            )

            seg_int_nodes = tracker.segment_interior_nodes if hasattr(tracker, 'segment_interior_nodes') else None
            save_border_init_visualization(
                keypoints=keypoints,
                border_grid_indices=border_grid,
                detected_corners_3d=detected_corners,
                contour_3d=contour_3d_vis,
                save_path=clip_dir / 'border_init.html',
                segment_interior_nodes=seg_int_nodes,
            )

            init_vis_saved = True

        if reference_lengths is not None:
            edge_metrics = compute_edge_metrics(keypoints, edges, reference_lengths)
        else:
            edge_metrics = compute_edge_metrics(None, None, None)

        pos_metrics = compute_position_metrics(keypoints, surface_pc)

        # CD metrics: sample on FACES
        if keypoints is not None and len(keypoints) > 0:
            tracker_faces = getattr(tracker, 'valid_faces', None)
            n_faces = len(tracker_faces) if tracker_faces else (grid_rows - 1) * (grid_cols - 1)
            n_ref_points = len(surface_pc) if surface_pc is not None else 5000
            n_samples_per_face = max(10, n_ref_points // max(n_faces, 1))
            pred_cloud = sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=n_samples_per_face, valid_faces=tracker_faces)
            cd_metrics = compute_chamfer_metrics(pred_cloud, surface_pc)
        else:
            cd_metrics = compute_chamfer_metrics(None, None)

        results['edge_metrics'].append(edge_metrics)
        results['pos_metrics'].append(pos_metrics)
        results['cd_metrics'].append(cd_metrics)

        # Capture per-frame visualization state for the post-loop renderer.
        # detected_corners_3d is a per-frame field on the tracker (mutated each track step),
        # so we project + snapshot it now.
        detected_corners_2d = None
        if hasattr(tracker, 'detected_corners_3d') and tracker.detected_corners_3d is not None:
            detected_corners_2d = tracker._project_3d_to_2d(tracker.detected_corners_3d)
            if detected_corners_2d is not None:
                detected_corners_2d = np.asarray(detected_corners_2d).copy()
        render_state.append({
            'mode': mode,
            'detected_corners_2d': detected_corners_2d,
        })

    # ==================================================================
    # POST-PROCESS: SMOOTH 3D KEYPOINTS + RENDER VIDEO FROM SMOOTHED
    # (metrics above were already computed on the RAW keypoints)
    # ==================================================================
    print(f"  Smoothing 3D trajectories (sigma={sigma}) and rendering video...")

    n_grid = grid_rows * grid_cols
    raw_3d = np.array([kp if kp is not None else np.full((n_grid, 3), np.nan)
                        for kp in results['keypoints']])
    smoothed_3d = smooth_trajectories(raw_3d, sigma=sigma)

    # Re-project smoothed 3D → 2D using the tracker's projection (cloth returns col, row).
    smoothed_2d_list = []
    for kp_3d in smoothed_3d:
        if kp_3d is None or len(kp_3d) == 0 or np.all(np.isnan(kp_3d)):
            smoothed_2d_list.append(np.full((n_grid, 2), np.nan, dtype=np.float32))
        else:
            kp_2d = tracker._project_3d_to_2d(kp_3d.astype(np.float64))
            if kp_2d is None or len(kp_2d) == 0:
                kp_2d = np.full((n_grid, 2), np.nan, dtype=np.float32)
            smoothed_2d_list.append(np.asarray(kp_2d, dtype=np.float32))

    # Render the tracking video using smoothed 2D keypoints.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    full_writer = cv2.VideoWriter(str(clip_dir / 'tracking_full.mp4'), fourcc, fps, (W * 2, H * 2))
    smoothed_traj_hist = []
    final_corner_indices = (getattr(tracker, 'contour_corner_grid_indices', None)
                            or tracker.CORNER_INDICES)
    final_border_indices = (getattr(tracker, 'border_grid_indices', None)
                            or tracker.BORDER_INDICES)
    final_edges = tracker.valid_edges
    for frame_idx in range(n_frames):
        rgb = cv2.cvtColor(color[frame_idx], cv2.COLOR_BGR2RGB)
        mask = fg_mask[frame_idx]
        ctx = render_state[frame_idx]
        smoothed_kp_2d = smoothed_2d_list[frame_idx]
        smoothed_traj_hist.append(smoothed_kp_2d.copy())

        full_vis = create_full_tracking_visualization(
            rgb, mask,
            smoothed_kp_2d,
            final_edges,
            frame_idx, ctx['mode'],
            np.array(smoothed_traj_hist), tail_length,
            final_corner_indices,
            final_border_indices,
            ctx['detected_corners_2d'],
        )
        full_writer.write(cv2.cvtColor(full_vis, cv2.COLOR_RGB2BGR))

    full_writer.release()

    # ==================================================================
    # OUTPUT (BDLO-STYLE, single method)
    # ==================================================================

    # 1. Save per_frame.csv
    per_frame_csv = clip_dir / 'per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write('LocalFrame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,'
                'PosRMSE,Pos<2mm,Pos<5mm,Pos<10mm,'
                'CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,'
                'Rec@2mm,Rec@5mm,Rec@10mm,F@2mm,F@5mm,F@10mm\n')
        for local_idx in range(n_frames):
            global_idx = start_frame + local_idx
            em = results['edge_metrics'][local_idx]
            pm = results['pos_metrics'][local_idx]
            cm = results['cd_metrics'][local_idx]
            f.write(f"{local_idx},{global_idx},{method_name},"
                    f"{em['pct_mean']:.4f},{em['pct_std']:.4f},{em['pct_max']:.4f},{em['rmse_mm']:.4f},"
                    f"{pm['rmse_mm']:.4f},{pm['under_2mm']:.4f},{pm['under_5mm']:.4f},{pm['under_10mm']:.4f},"
                    f"{cm['cd']:.4f},{cm['pred2ref_avg']:.4f},{cm['ref2pred_avg']:.4f},"
                    f"{cm['precision_2mm']:.4f},{cm['precision_5mm']:.4f},{cm['precision_10mm']:.4f},"
                    f"{cm['recall_2mm']:.4f},{cm['recall_5mm']:.4f},{cm['recall_10mm']:.4f},"
                    f"{cm['f_2mm']:.4f},{cm['f_5mm']:.4f},{cm['f_10mm']:.4f}\n")

    # 2. Compute clip summary
    edge_metrics_list = results['edge_metrics']
    pos_metrics_list = results['pos_metrics']
    cd_metrics_list = results['cd_metrics']

    valid_edge = [m for m in edge_metrics_list if m['pct_mean'] > 0]
    valid_pos = [m for m in pos_metrics_list if m['rmse_mm'] > 0]
    valid_cd = [m for m in cd_metrics_list if m['cd'] > 0]

    summary_row = {
        'method': method_name,
        # Edge metrics
        'edge_pct_mean_avg': np.mean([m['pct_mean'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_pct_mean_std': np.std([m['pct_mean'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_rmse_avg': np.mean([m['rmse_mm'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_rmse_std': np.std([m['rmse_mm'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_under_2pct': np.mean([m['under_2pct'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_under_5pct': np.mean([m['under_5pct'] for m in valid_edge]) if valid_edge else 0.0,
        'edge_under_10pct': np.mean([m['under_10pct'] for m in valid_edge]) if valid_edge else 0.0,
        # Position metrics
        'pos_rmse_avg': np.mean([m['rmse_mm'] for m in valid_pos]) if valid_pos else 0.0,
        'pos_rmse_std': np.std([m['rmse_mm'] for m in valid_pos]) if valid_pos else 0.0,
        'pos_under_2mm': np.mean([m['under_2mm'] for m in valid_pos]) if valid_pos else 0.0,
        'pos_under_5mm': np.mean([m['under_5mm'] for m in valid_pos]) if valid_pos else 0.0,
        'pos_under_10mm': np.mean([m['under_10mm'] for m in valid_pos]) if valid_pos else 0.0,
        # CD metrics
        'cd_avg': np.mean([m['cd'] for m in valid_cd]) if valid_cd else 0.0,
        'cd_std': np.std([m['cd'] for m in valid_cd]) if valid_cd else 0.0,
        'cd_pred2ref_avg': np.mean([m['pred2ref_avg'] for m in valid_cd]) if valid_cd else 0.0,
        'cd_ref2pred_avg': np.mean([m['ref2pred_avg'] for m in valid_cd]) if valid_cd else 0.0,
        'precision_2mm': np.mean([m['precision_2mm'] for m in valid_cd]) if valid_cd else 0.0,
        'precision_5mm': np.mean([m['precision_5mm'] for m in valid_cd]) if valid_cd else 0.0,
        'precision_10mm': np.mean([m['precision_10mm'] for m in valid_cd]) if valid_cd else 0.0,
        'recall_2mm': np.mean([m['recall_2mm'] for m in valid_cd]) if valid_cd else 0.0,
        'recall_5mm': np.mean([m['recall_5mm'] for m in valid_cd]) if valid_cd else 0.0,
        'recall_10mm': np.mean([m['recall_10mm'] for m in valid_cd]) if valid_cd else 0.0,
        'f_2mm': np.mean([m['f_2mm'] for m in valid_cd]) if valid_cd else 0.0,
        'f_5mm': np.mean([m['f_5mm'] for m in valid_cd]) if valid_cd else 0.0,
        'f_10mm': np.mean([m['f_10mm'] for m in valid_cd]) if valid_cd else 0.0,
    }
    summary_rows = [summary_row]

    # 3. Save summary.txt (3 tables)
    summary_txt = clip_dir / 'summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Clip {clip_idx} Summary (frames {start_frame}-{end_frame}, {n_frames} frames)\n")
        f.write("=" * 100 + "\n\n")

        # Table 1: Edge Length Metrics
        f.write("Edge Length Metrics\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
        f.write("-" * 100 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>5.2f}% | "
                    f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
                    f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%\n")

        f.write("\n")

        # Table 2: Position RMSE Metrics
        f.write("Position RMSE Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
        f.write("-" * 80 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>5.2f} mm   | "
                    f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%\n")

        f.write("\n")

        # Table 3: Chamfer Distance Metrics
        f.write("Chamfer Distance Metrics\n")
        f.write("-" * 130 + "\n")
        f.write(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<8} | {'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}\n")
        f.write("-" * 130 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | "
                    f"{s['cd_pred2ref_avg']:>7.2f} mm | {s['cd_ref2pred_avg']:>7.2f} mm | "
                    f"{s['precision_2mm']:>5.1f}% | {s['precision_5mm']:>5.1f}% | {s['precision_10mm']:>5.1f}% | "
                    f"{s['recall_2mm']:>5.1f}% | {s['recall_5mm']:>5.1f}% | {s['recall_10mm']:>5.1f}%\n")

        f.write("\n")
        f.write("F-Scores\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}\n")
        f.write("-" * 60 + "\n")
        for s in summary_rows:
            f.write(f"{s['method']:<12} | {s['f_2mm']:>8.2f}% | {s['f_5mm']:>8.2f}% | {s['f_10mm']:>8.2f}%\n")

    # 4. Save rmse_over_time.png
    color_full = 'blue'

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    frames = list(range(n_frames))
    edge_rmses = [m['rmse_mm'] for m in results['edge_metrics']]
    pos_rmses = [m['rmse_mm'] for m in results['pos_metrics']]

    axes[0].plot(frames, edge_rmses, label=method_name, color=color_full, alpha=0.8)
    axes[1].plot(frames, pos_rmses, label=method_name, color=color_full, alpha=0.8)

    axes[0].set_ylabel('Edge RMSE (mm)')
    axes[0].set_title(f'Clip {clip_idx}: RMSE Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Position RMSE (mm)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(clip_dir / 'rmse_over_time.png', dpi=150)
    plt.close(fig)

    # 5. Save cd_over_time.png
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    cd_values = [m['cd'] for m in results['cd_metrics']]
    pred2ref_values = [m['pred2ref_avg'] for m in results['cd_metrics']]
    ref2pred_values = [m['ref2pred_avg'] for m in results['cd_metrics']]

    axes[0].plot(frames, cd_values, label=method_name, color=color_full, alpha=0.8)
    axes[1].plot(frames, pred2ref_values, label=method_name, color=color_full, alpha=0.8)
    axes[2].plot(frames, ref2pred_values, label=method_name, color=color_full, alpha=0.8)

    axes[0].set_ylabel('CD (mm)')
    axes[0].set_title(f'Clip {clip_idx}: Chamfer Distance Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Pred→Ref (mm)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Ref→Pred (mm)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(clip_dir / 'cd_over_time.png', dpi=150)
    plt.close(fig)

    # 6. Save 3d_keypoints.npz (raw — used for evaluation) and smoothed_3d_keypoints.npz (viz only)
    stored_edges = tracker.valid_edges if hasattr(tracker, 'valid_edges') else []
    stored_ref_lens = reference_lengths

    np.savez(
        clip_dir / '3d_keypoints.npz',
        full=raw_3d,
        edge_connections=np.array(stored_edges) if stored_edges else np.array([]),
        reference_lengths=np.array(list(stored_ref_lens.values())) if stored_ref_lens else np.array([]),
    )
    np.savez(
        clip_dir / 'smoothed_3d_keypoints.npz',
        full=smoothed_3d,
        sigma=np.array(sigma),
        edge_connections=np.array(stored_edges) if stored_edges else np.array([]),
        reference_lengths=np.array(list(stored_ref_lens.values())) if stored_ref_lens else np.array([]),
    )

    # ==================================================================
    # BACKWARD-COMPATIBLE clip_metrics DICT
    # ==================================================================
    clip_metrics = {
        method_name: {
            'n_frames': n_frames,
            'edge_pct_mean': summary_row['edge_pct_mean_avg'],
            'edge_pct_std': summary_row['edge_pct_mean_std'],
            'edge_rmse': summary_row['edge_rmse_avg'],
            'edge_under_5pct': summary_row['edge_under_5pct'],
            'pos_rmse': summary_row['pos_rmse_avg'],
            'pos_under_5mm': summary_row['pos_under_5mm'],
            'cd': summary_row['cd_avg'],
            'f_5mm': summary_row['f_5mm'],
            'f_10mm': summary_row['f_10mm'],
        }
    }

    print(f"    Saved outputs to: {clip_dir}")

    # Print summary table to console
    print(f"\n    Clip {clip_idx} Summary:")
    print(f"    {'-' * 90}")
    print(f"    {'Method':<12} | {'Edge%':<12} | {'EdgeRMSE':<12} | {'PosRMSE':<12} | {'CD':<12} | {'F@10mm':<10}")
    print(f"    {'-' * 90}")
    for s in summary_rows:
        print(f"    {s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} | "
              f"{s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>4.2f} | "
              f"{s['cd_avg']:>5.2f} ±{s['cd_std']:>3.2f} | "
              f"{s['f_10mm']:>6.1f}%")
    print(f"    {'-' * 90}")

    # Build per-frame metrics list for chunk aggregation
    all_metrics = {method_name: []}
    for local_idx in range(n_frames):
        global_frame = start_frame + local_idx
        em = results['edge_metrics'][local_idx]
        pm = results['pos_metrics'][local_idx]
        cm = results['cd_metrics'][local_idx]
        all_metrics[method_name].append({
            'frame': local_idx,
            'global_frame': global_frame,
            'success': em['pct_mean'] > 0,
            # Edge metrics
            'edge_pct_mean': em['pct_mean'],
            'edge_pct_std': em['pct_std'],
            'edge_pct_max': em['pct_max'],
            'edge_rmse_mm': em['rmse_mm'],
            'edge_under_2pct': em['under_2pct'],
            'edge_under_5pct': em['under_5pct'],
            'edge_under_10pct': em['under_10pct'],
            # Position metrics
            'pos_rmse_mm': pm['rmse_mm'],
            'pos_under_2mm': pm['under_2mm'],
            'pos_under_5mm': pm['under_5mm'],
            'pos_under_10mm': pm['under_10mm'],
            # CD metrics
            'cd': cm['cd'],
            'cd_pred2ref': cm['pred2ref_avg'],
            'cd_ref2pred': cm['ref2pred_avg'],
            'precision_2mm': cm['precision_2mm'],
            'precision_5mm': cm['precision_5mm'],
            'precision_10mm': cm['precision_10mm'],
            'recall_2mm': cm['recall_2mm'],
            'recall_5mm': cm['recall_5mm'],
            'recall_10mm': cm['recall_10mm'],
            'f_2mm': cm['f_2mm'],
            'f_5mm': cm['f_5mm'],
            'f_10mm': cm['f_10mm'],
        })

    return {
        'clip_idx': clip_idx,
        'n_frames': n_frames,
        'clip_metrics': clip_metrics,
        'summary_rows': summary_rows,
        'all_metrics': all_metrics,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean cloth tracking experiment (Full method only)")
    parser.add_argument('--chunk', type=int, required=True,
                        help='Chunk index to process (looks under input_data/cloth/chunk_<N>/)')
    parser.add_argument('--clip_seconds', type=int, default=10,
                        help='Clip length in seconds (default: 10)')
    parser.add_argument('--max_frames', type=int, default=10000,
                        help='Maximum frames to load from chunk (default: 10000)')
    parser.add_argument('--tail_length', type=int, default=60,
                        help='Trajectory tail length in frames (default: 60)')
    parser.add_argument('--grid_rows', type=int, default=9,
                        help='Number of grid rows (default: 9)')
    parser.add_argument('--grid_cols', type=int, default=9,
                        help='Number of grid columns (default: 9)')
    parser.add_argument('--segment_interior_nodes', type=str, default=None,
                        help='Comma-separated interior node counts per segment, e.g., "7,1,1,5,3,5,1,1"')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma applied to 3D trajectories for visualization '
                             '(default: 3.0; metrics are still on raw)')
    args = parser.parse_args()

    # Parse segment_interior_nodes
    segment_interior_nodes = None
    if args.segment_interior_nodes:
        segment_interior_nodes = [int(x) for x in args.segment_interior_nodes.split(',')]
        print(f"Using manual segment interior nodes: {segment_interior_nodes}")

    # Set ClothTracker class variables for custom grid size
    ClothTracker.GRID_ROWS = args.grid_rows
    ClothTracker.GRID_COLS = args.grid_cols

    chunk_dir = DATA_BASE / f"chunk_{args.chunk}"
    output_dir = OUTPUT_BASE / f"chunk_{args.chunk}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLOTH TRACKING (FULL METHOD ONLY)")
    print("=" * 70)
    print(f"Data base: {DATA_BASE}")
    print(f"Chunk: {args.chunk}")
    print(f"Grid size: {args.grid_rows} × {args.grid_cols}")
    print(f"Clip length: {args.clip_seconds}s")
    print(f"Output: {output_dir}")

    data = load_chunk_data(chunk_dir, max_frames=args.max_frames)
    transforms = load_transforms(CALIB_DIR)

    print(f"\nCalibration loaded from: {CALIB_DIR}")
    print(f"  K: {transforms['K'][0,0]:.1f}, {transforms['K'][1,1]:.1f}")

    # Precompute EE positions in camera frame
    n_frames = data['n_frames']
    ee_poses_3d_raw = np.zeros((n_frames, 2, 3), dtype=np.float32)

    for i in range(n_frames):
        ee_poses_3d_raw[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam']
        )

    # Filter EE outliers
    print("\nChecking for EE position outliers...")
    ee_poses_3d, outlier_frames = filter_ee_outliers(
        ee_poses_3d_raw,
        velocity_threshold=80.0,
        window_size=3
    )

    if len(outlier_frames) > 0:
        print(f"  Filtered {len(outlier_frames)} outlier EE positions")
    else:
        print("  No outliers detected")

    print(f"\nEE positions in camera frame: {ee_poses_3d.shape}")
    print(f"  Left EE depth range: [{ee_poses_3d[:, 0, 2].min():.0f}, {ee_poses_3d[:, 0, 2].max():.0f}] mm")
    print(f"  Right EE depth range: [{ee_poses_3d[:, 1, 2].min():.0f}, {ee_poses_3d[:, 1, 2].max():.0f}] mm")

    frames_per_clip = args.clip_seconds * FPS
    n_clips = max(1, n_frames // frames_per_clip)

    print(f"\nSplitting into {n_clips} clips of {frames_per_clip} frames each")

    all_clip_results = []

    for clip_idx in range(n_clips):
        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, n_frames)

        clip_result = process_clip(
            data, transforms, ee_poses_3d,
            clip_idx, start_frame, end_frame,
            output_dir, args.grid_rows, args.grid_cols, args.tail_length, FPS,
            segment_interior_nodes=segment_interior_nodes,
            sigma=args.sigma,
        )
        all_clip_results.append(clip_result)

    # ==================================================================
    # CHUNK SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("CHUNK SUMMARY")
    print("=" * 70)

    method_names = ['Full']

    summary_dir = output_dir / 'chunk_summary'
    summary_dir.mkdir(exist_ok=True)

    # Aggregate all clips' metrics
    all_clips_metrics = {m: [] for m in method_names}
    for clip_result in all_clip_results:
        for method in method_names:
            all_clips_metrics[method].extend(clip_result['all_metrics'][method])

    # Save stacked per-frame CSV
    stacked_csv = summary_dir / 'all_clips_metrics.csv'
    with open(stacked_csv, 'w') as f:
        f.write('Clip,Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,'
                'CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,Rec@2mm,Rec@5mm,Rec@10mm,F@2mm,F@5mm,F@10mm\n')
        for clip_result in all_clip_results:
            clip_idx = clip_result['clip_idx']
            for method in method_names:
                for m in clip_result['all_metrics'][method]:
                    f.write(f"{clip_idx},{m['frame']},{m['global_frame']},{method},"
                            f"{m['edge_pct_mean']:.6f},{m['edge_pct_std']:.6f},{m['edge_pct_max']:.6f},"
                            f"{m['edge_rmse_mm']:.6f},{m['pos_rmse_mm']:.6f},"
                            f"{m['cd']:.4f},{m['cd_pred2ref']:.4f},{m['cd_ref2pred']:.4f},"
                            f"{m['precision_2mm']:.4f},{m['precision_5mm']:.4f},{m['precision_10mm']:.4f},"
                            f"{m['recall_2mm']:.4f},{m['recall_5mm']:.4f},{m['recall_10mm']:.4f},"
                            f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")

    # Compute chunk aggregate summary (Frame-weighted: pool all frames)
    chunk_summary_frame_weighted = []
    for method in method_names:
        metrics_list = all_clips_metrics[method]
        if len(metrics_list) == 0:
            continue

        edge_pct_means = [m['edge_pct_mean'] for m in metrics_list if m['edge_pct_mean'] > 0]
        edge_rmses = [m['edge_rmse_mm'] for m in metrics_list if m['edge_rmse_mm'] > 0]
        pos_rmses = [m['pos_rmse_mm'] for m in metrics_list if m['pos_rmse_mm'] > 0]
        edge_under_2 = [m['edge_under_2pct'] for m in metrics_list if m['success']]
        edge_under_5 = [m['edge_under_5pct'] for m in metrics_list if m['success']]
        edge_under_10 = [m['edge_under_10pct'] for m in metrics_list if m['success']]
        pos_under_2 = [m['pos_under_2mm'] for m in metrics_list if m['success']]
        pos_under_5 = [m['pos_under_5mm'] for m in metrics_list if m['success']]
        pos_under_10 = [m['pos_under_10mm'] for m in metrics_list if m['success']]

        # CD metrics
        cd_vals = [m['cd'] for m in metrics_list if m['success']]
        precision_2 = [m['precision_2mm'] for m in metrics_list if m['success']]
        precision_5 = [m['precision_5mm'] for m in metrics_list if m['success']]
        precision_10 = [m['precision_10mm'] for m in metrics_list if m['success']]
        recall_2 = [m['recall_2mm'] for m in metrics_list if m['success']]
        recall_5 = [m['recall_5mm'] for m in metrics_list if m['success']]
        recall_10 = [m['recall_10mm'] for m in metrics_list if m['success']]
        f_2 = [m['f_2mm'] for m in metrics_list if m['success']]
        f_5 = [m['f_5mm'] for m in metrics_list if m['success']]
        f_10 = [m['f_10mm'] for m in metrics_list if m['success']]

        chunk_summary_frame_weighted.append({
            'method': method,
            'edge_pct_mean_avg': np.mean(edge_pct_means) if edge_pct_means else 0.0,
            'edge_pct_mean_std': np.std(edge_pct_means) if edge_pct_means else 0.0,
            'edge_rmse_avg': np.mean(edge_rmses) if edge_rmses else 0.0,
            'edge_rmse_std': np.std(edge_rmses) if edge_rmses else 0.0,
            'edge_under_2pct': np.mean(edge_under_2) if edge_under_2 else 0.0,
            'edge_under_5pct': np.mean(edge_under_5) if edge_under_5 else 0.0,
            'edge_under_10pct': np.mean(edge_under_10) if edge_under_10 else 0.0,
            'pos_rmse_avg': np.mean(pos_rmses) if pos_rmses else 0.0,
            'pos_rmse_std': np.std(pos_rmses) if pos_rmses else 0.0,
            'pos_under_2mm': np.mean(pos_under_2) if pos_under_2 else 0.0,
            'pos_under_5mm': np.mean(pos_under_5) if pos_under_5 else 0.0,
            'pos_under_10mm': np.mean(pos_under_10) if pos_under_10 else 0.0,
            # CD metrics
            'cd_avg': np.mean(cd_vals) if cd_vals else 0.0,
            'cd_std': np.std(cd_vals) if cd_vals else 0.0,
            'precision_2mm': np.mean(precision_2) if precision_2 else 0.0,
            'precision_5mm': np.mean(precision_5) if precision_5 else 0.0,
            'precision_10mm': np.mean(precision_10) if precision_10 else 0.0,
            'recall_2mm': np.mean(recall_2) if recall_2 else 0.0,
            'recall_5mm': np.mean(recall_5) if recall_5 else 0.0,
            'recall_10mm': np.mean(recall_10) if recall_10 else 0.0,
            'f_2mm': np.mean(f_2) if f_2 else 0.0,
            'f_5mm': np.mean(f_5) if f_5 else 0.0,
            'f_10mm': np.mean(f_10) if f_10 else 0.0,
        })

    # Save chunk summary txt
    summary_txt = summary_dir / 'chunk_summary.txt'
    with open(summary_txt, 'w') as f:
        f.write(f"Chunk {args.chunk} Summary ({n_clips} clips, {n_frames} total frames)\n")
        f.write("=" * 120 + "\n\n")

        f.write("FRAME-WEIGHTED SUMMARY (pooling all frames across clips)\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Method':<12} | {'Edge%':<15} | {'EdgeRMSE (mm)':<15} | {'<5%':<8} | {'PosRMSE (mm)':<15} | {'<5mm':<8} | {'CD (mm)':<15} | {'F@10mm':<8}\n")
        f.write("-" * 120 + "\n")
        for s in chunk_summary_frame_weighted:
            f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f} ±{s['edge_pct_mean_std']:>5.2f}% | "
                    f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | {s['edge_under_5pct']:>5.1f}% | "
                    f"{s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>4.2f} mm | {s['pos_under_5mm']:>5.1f}% | "
                    f"{s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | {s['f_10mm']:>5.1f}%\n")

        f.write("\n")
        f.write("Precision/Recall/F-Score:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Prec@2mm':<10} | {'Prec@5mm':<10} | {'Prec@10mm':<10} | {'Rec@2mm':<10} | {'Rec@5mm':<10} | {'Rec@10mm':<10} | {'F@2mm':<8} | {'F@5mm':<8} | {'F@10mm':<8}\n")
        f.write("-" * 100 + "\n")
        for s in chunk_summary_frame_weighted:
            f.write(f"{s['method']:<12} | {s['precision_2mm']:>7.1f}% | {s['precision_5mm']:>7.1f}% | {s['precision_10mm']:>7.1f}% | "
                    f"{s['recall_2mm']:>7.1f}% | {s['recall_5mm']:>7.1f}% | {s['recall_10mm']:>7.1f}% | "
                    f"{s['f_2mm']:>6.1f}% | {s['f_5mm']:>6.1f}% | {s['f_10mm']:>6.1f}%\n")

    print(f"\nResults saved to: {summary_dir}")

    # Console output
    print(f"\n{'Method':<12} | {'Edge%':<15} | {'EdgeRMSE':<12} | {'PosRMSE':<12} | {'CD':<12} | {'F@10mm':<8}")
    print("-" * 85)

    for s in chunk_summary_frame_weighted:
        print(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>4.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>3.2f}mm | "
              f"{s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>3.2f}mm | "
              f"{s['cd_avg']:>5.2f} ±{s['cd_std']:>3.2f}mm | {s['f_10mm']:>5.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
