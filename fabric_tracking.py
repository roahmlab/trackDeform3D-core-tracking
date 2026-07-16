"""
Clean Fabric tracking experiment on cloth datasets.

Runs ONLY the full fabric tracker (no ablation, no CDCPD baseline).
Processes chunks with multiple clips, reinitializing the tracker per clip.
Fabric uses a configurable N×N grid topology with corners held by robot EEs.

Key differences from fabric_batch_experiment.py:
- No ablation methods (NoSnap, NoGeometry removed)
- No CDCPD baseline comparison
- Single "Full" method only

Usage:
    python fabric_tracking.py --chunk 14 --clip_seconds 10

Loads from: input_data/fabric/chunk_<N>/  (calibration in input_data/fabric/calibration/)

Author: Auto-generated
Date: 2026-05-06
"""

import argparse
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tracker.fabric_tracker import FabricTracker, FabricTrackerFull
from utils.smoothing import smooth_trajectories
from utils.transforms import load_transforms, pose7_to_matrix, get_ee_positions_cam
from utils.metrics_fabric import (filter_ee_outliers, compute_edge_metrics,
                                  compute_position_metrics, sample_points_on_faces,
                                  compute_chamfer_metrics)
from utils.pointcloud import extract_surface_point_cloud


# ============================================================================
# POST-PROCESS: TRAJECTORY SMOOTHING
# ============================================================================


# ============================================================================
# CONSTANTS
# ============================================================================

# Paths (resolved relative to this script: input_data/fabric/chunk_<N>/)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE = SCRIPT_DIR / "input_data" / "fabric"
CALIB_DIR = DATA_BASE / "calibration"
OUTPUT_BASE = SCRIPT_DIR / "output" / "fabric"

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
    fg_mask_path = chunk_dir / 'fg_mask.npz'
    if fg_mask_path.exists():
        fg_mask = np.load(fg_mask_path)['fg_mask'][start_idx:]
    else:
        raise FileNotFoundError(f"fg_mask.npz not found in {chunk_dir}. "
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


# ============================================================================
# METRICS
# ============================================================================


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
            marker=dict(size=1.5, color='lightgrey', opacity=0.5),
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
            line=dict(color='red', width=2, dash='dash'),
            name='Raw Contour',
            hoverinfo='skip',
        ))

    # Denoised contour trace (blue solid line)
    if contour_3d is not None and len(contour_3d) > 0:
        contour_vis = contour_3d[::5] if len(contour_3d) > 200 else contour_3d
        contour_vis = np.vstack([contour_vis, contour_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_vis[:, 0], y=contour_vis[:, 1], z=contour_vis[:, 2],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Denoised Contour',
            hoverinfo='skip',
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

    # Edge traces (blue lines)
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            edge_x.extend([keypoints[i, 0], keypoints[j, 0], None])
            edge_y.extend([keypoints[i, 1], keypoints[j, 1], None])
            edge_z.extend([keypoints[i, 2], keypoints[j, 2], None])

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Edges',
        hoverinfo='skip',
    ))

    # Keypoint traces - color by type
    corner_indices = corner_indices or []
    border_indices = border_indices or []

    # Interior nodes (green)
    interior_mask = [i not in corner_indices and i not in border_indices for i in range(len(keypoints))]
    interior_pts = keypoints[interior_mask]
    interior_idx = [i for i in range(len(keypoints)) if interior_mask[i]]
    if len(interior_pts) > 0:
        traces.append(go.Scatter3d(
            x=interior_pts[:, 0], y=interior_pts[:, 1], z=interior_pts[:, 2],
            mode='markers',
            marker=dict(size=6, color='green'),
            name='Interior',
            text=[f'Node {i}' for i in interior_idx],
            hoverinfo='text',
        ))

    # Border nodes (orange)
    border_pts = keypoints[[i for i in border_indices if i < len(keypoints)]]
    border_idx = [i for i in border_indices if i < len(keypoints)]
    if len(border_pts) > 0:
        traces.append(go.Scatter3d(
            x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
            mode='markers',
            marker=dict(size=8, color='orange'),
            name='Border',
            text=[f'Node {i}' for i in border_idx],
            hoverinfo='text',
        ))

    # Corner nodes (red, larger)
    corner_pts = keypoints[[i for i in corner_indices if i < len(keypoints)]]
    corner_idx = [i for i in corner_indices if i < len(keypoints)]
    if len(corner_pts) > 0:
        traces.append(go.Scatter3d(
            x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Corners',
            text=[f'Corner {i}' for i in corner_idx],
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)

    # Compute edge length stats for title
    edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges if i < len(keypoints) and j < len(keypoints)]
    if edge_lengths:
        avg_len = np.mean(edge_lengths)
        std_len = np.std(edge_lengths)
        title = f'Init: {len(keypoints)} nodes, {len(edges)} edges | Avg edge: {avg_len:.1f}mm, Std: {std_len:.1f}mm ({std_len/avg_len*100:.1f}%)'
    else:
        title = f'Init: {len(keypoints)} nodes, {len(edges)} edges'

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


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_full_tracking_visualization(rgb, fg_mask, keypoints_2d, edges,
                                       frame_idx, mode, traj_history_2d=None,
                                       tail_length=60, corner_indices=None, border_indices=None):
    """Create 4-panel visualization."""
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
    if fg_mask is not None:
        panel1[fg_mask > 0] = MASK_COLOR

    # Panel 2: Mask overlay
    panel2 = rgb.copy()
    if fg_mask is not None:
        contours, _ = cv2.findContours(fg_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, contours, -1, MASK_COLOR, 2)
    cv2.putText(panel2, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel2, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: Keypoints only
    panel3 = np.zeros((H, W, 3), dtype=np.uint8)
    if fg_mask is not None:
        panel3[fg_mask > 0] = [30, 30, 30]

    def draw_keypoints_and_edges(canvas):
        if traj_history_2d is not None and len(traj_history_2d) > 1:
            n_hist = len(traj_history_2d)
            start = max(0, n_hist - tail_length)
            for t in range(start, n_hist - 1):
                alpha = (t - start) / (n_hist - start)
                color = tuple(int(c * alpha) for c in TAIL_COLOR)
                for k in range(len(traj_history_2d[t])):
                    pt1 = tuple(traj_history_2d[t, k, ::-1].astype(int))
                    pt2 = tuple(traj_history_2d[t + 1, k, ::-1].astype(int))
                    if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
                        cv2.line(canvas, pt1, pt2, color, 2)

        if keypoints_2d is not None and len(keypoints_2d) > 0 and edges is not None:
            for (i, j) in edges:
                if i < len(keypoints_2d) and j < len(keypoints_2d):
                    pt1 = tuple(keypoints_2d[i, ::-1].astype(int))
                    pt2 = tuple(keypoints_2d[j, ::-1].astype(int))
                    cv2.line(canvas, pt1, pt2, EDGE_COLOR, 2)

            for idx in range(len(keypoints_2d)):
                pt = tuple(keypoints_2d[idx, ::-1].astype(int))
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
    if fg_mask is not None:
        contours, _ = cv2.findContours(fg_mask.astype(np.uint8),
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
                 output_dir, grid_rows, grid_cols, tail_length=60, fps=30, sigma=2.0):
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

    # Tracker parameters (same as fabric_batch_experiment.py)
    tracker_params = {
        'intrinsics': K,
        'max_depth': 2000.0,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 500,
        'repulsion_lr': 5.0,
        'ee_poses_3d': clip_ee_poses,
    }

    method_name = 'Full'

    # Initialize tracker
    tracker = FabricTrackerFull(**tracker_params)

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
        edges = result.get('edges', tracker.grid_edges if hasattr(tracker, 'grid_edges') else [])

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
            if hasattr(tracker, '_extract_contour_3d') and hasattr(tracker, '_find_mask_corners'):
                corners_2d = tracker._find_mask_corners(mask, d)
                corners_3d_vis = tracker._pixel_to_3d(corners_2d, d) if corners_2d is not None else None
                if hasattr(tracker, '_extract_contour_3d_raw'):
                    contour_3d_raw = tracker._extract_contour_3d_raw(mask, d)
                contour_3d_vis = tracker._extract_contour_3d(mask, d, corners_3d=corners_3d_vis)
                if hasattr(tracker, '_compute_contour_segment_lengths') and contour_3d_vis is not None and corners_3d_vis is not None:
                    segment_lengths = tracker._compute_contour_segment_lengths(contour_3d_vis, corners_3d_vis)

            ee_poses_frame = clip_ee_poses[frame_idx] if clip_ee_poses is not None else None

            save_init_visualization_3d(
                keypoints=keypoints,
                edges=edges,
                point_cloud=fg_pc_full,
                save_path=clip_dir / 'init_3d.html',
                corner_indices=tracker.CORNER_INDICES if hasattr(tracker, 'CORNER_INDICES') else [],
                border_indices=tracker.BORDER_INDICES if hasattr(tracker, 'BORDER_INDICES') else [],
                downsample_pc=50000,
                contour_3d=contour_3d_vis,
                contour_3d_raw=contour_3d_raw,
                ee_poses=ee_poses_frame,
                segment_lengths=segment_lengths,
            )
            init_vis_saved = True

        if reference_lengths is not None:
            edge_metrics = compute_edge_metrics(keypoints, edges, reference_lengths)
        else:
            edge_metrics = compute_edge_metrics(None, None, None)

        pos_metrics = compute_position_metrics(keypoints, surface_pc)

        # CD metrics: sample on FACES
        if keypoints is not None and len(keypoints) > 0:
            n_faces = (grid_rows - 1) * (grid_cols - 1)
            n_ref_points = len(surface_pc) if surface_pc is not None else 5000
            n_samples_per_face = max(10, n_ref_points // n_faces)
            pred_cloud = sample_points_on_faces(keypoints, grid_rows, grid_cols, n_samples_per_face=n_samples_per_face)
            cd_metrics = compute_chamfer_metrics(pred_cloud, surface_pc)
        else:
            cd_metrics = compute_chamfer_metrics(None, None)

        results['edge_metrics'].append(edge_metrics)
        results['pos_metrics'].append(pos_metrics)
        results['cd_metrics'].append(cd_metrics)

    # ==================================================================
    # POST-PROCESS: SMOOTH 3D KEYPOINTS + RENDER VIDEO FROM SMOOTHED
    # (metrics above were already computed on the RAW keypoints)
    # ==================================================================
    print(f"  Smoothing 3D trajectories (sigma={sigma}) and rendering video...")

    n_grid = grid_rows * grid_cols
    raw_3d = np.array([kp if kp is not None else np.full((n_grid, 3), np.nan)
                        for kp in results['keypoints']])
    smoothed_3d = smooth_trajectories(raw_3d, sigma=sigma)

    # Re-project smoothed 3D → 2D using the tracker's own projection (row, col).
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
    for frame_idx in range(n_frames):
        rgb = cv2.cvtColor(color[frame_idx], cv2.COLOR_BGR2RGB)
        mask = fg_mask[frame_idx]
        mode = results['modes'][frame_idx]
        smoothed_kp_2d = smoothed_2d_list[frame_idx]
        smoothed_traj_hist.append(smoothed_kp_2d.copy())

        full_vis = create_full_tracking_visualization(
            rgb, mask,
            smoothed_kp_2d,
            tracker.grid_edges,
            frame_idx, mode,
            np.array(smoothed_traj_hist), tail_length,
            tracker.CORNER_INDICES, tracker.BORDER_INDICES
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
    stored_edges = tracker.grid_edges if hasattr(tracker, 'grid_edges') else []
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
    parser = argparse.ArgumentParser(description="Clean fabric tracking experiment (Full method only)")
    parser.add_argument('--chunk', type=int, required=True,
                        help='Chunk index to process (looks under input_data/fabric/chunk_<N>/)')
    parser.add_argument('--clip_seconds', type=int, default=10,
                        help='Clip length in seconds (default: 10)')
    parser.add_argument('--max_frames', type=int, default=10000,
                        help='Maximum frames to load from chunk (default: 10000)')
    parser.add_argument('--tail_length', type=int, default=60,
                        help='Trajectory tail length in frames (default: 60)')
    parser.add_argument('--grid_rows', type=int, default=6,
                        help='Number of grid rows (default: 6)')
    parser.add_argument('--grid_cols', type=int, default=6,
                        help='Number of grid columns (default: 6)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma applied to 3D trajectories for visualization '
                             '(default: 3.0; metrics are still on raw)')
    args = parser.parse_args()

    # NOTE: FabricTracker class currently has GRID_ROWS/GRID_COLS as class constants.
    FabricTracker.GRID_ROWS = args.grid_rows
    FabricTracker.GRID_COLS = args.grid_cols

    chunk_dir = DATA_BASE / f"chunk_{args.chunk}"
    output_dir = OUTPUT_BASE / f"chunk_{args.chunk}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FABRIC TRACKING (FULL METHOD ONLY)")
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
