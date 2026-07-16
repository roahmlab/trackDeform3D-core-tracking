"""
Clean BDLO (branched DLO) tracking experiment on bdlo_no_contact_2sec dataset.

Runs ONLY the full WireTracker pipeline (no ablation, no CDCPD baseline).
Processes chunks with multiple clips, reinitializing the tracker per clip.
BDLO has 2 branch nodes and 4 leaf nodes (Y-shape topology).

Key differences from bdlo1_faster_batch_experiment.py:
- No ablation methods (NoSnap, NoGeometry removed)
- No CDCPD baseline comparison (CDCPDBDLOTracker class removed)
- Single "Full" method only

Usage:
    python bdlo_tracking.py --chunk 7 --clip_seconds 10 --n_keypoints 25 \
        --keypoints_per_segment 4 4 3 3 5

Author: Auto-generated
Date: 2026-05-06
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tracker.wire_tracker import WireTracker
from utils.smoothing import smooth_trajectories
from utils.transforms import load_transforms, pose7_to_matrix, get_ee_positions_cam
from utils.metrics_wire import (compute_edge_metrics, compute_position_metrics,
                                sample_points_on_edges, compute_chamfer_metrics)
from utils.summary import write_summary_tables, print_summary_tables


# ============================================================================
# POST-PROCESS: TRAJECTORY SMOOTHING
# ============================================================================


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunk_data(chunk_dir: Path) -> dict:
    """Load all data from a chunk directory (last 600 frames only)."""
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    color = rgbd['color'][-600:]
    depth = rgbd['depth'][-600:]

    masks_path = chunk_dir / 'masks' / 'masks.npz'
    if masks_path.exists():
        dlo_masks = np.load(masks_path)['masks'][-600:]
    else:
        dlo_masks = None

    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')

    n_frames_total = len(left_poses_npz.files)
    start_idx = max(0, n_frames_total - 600)
    n_frames = min(600, n_frames_total)
    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(start_idx, n_frames_total)])
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(start_idx, n_frames_total)])

    return {
        'color': color,
        'depth': depth,
        'dlo_masks': dlo_masks,
        'left_poses': left_poses,
        'right_poses': right_poses,
        'n_frames': n_frames,
    }


# ============================================================================
# METRICS
# ============================================================================


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_full_tracking_visualization(rgb, depth_augmented_mask, skeleton_mask, keypoints_2d, edges,
                                        frame_idx, mode, traj_history_2d=None, tail_length=60):
    """Create 2x2 visualization grid for the Full method.

    Panels:
        1 (top-left): RGB + frame info
        2 (top-right): Depth-augmented SAM2 mask (green overlay)
        3 (bottom-left): Skeleton + keypoints only
        4 (bottom-right): RGB + edges + labeled keypoints
    """
    H, W = rgb.shape[:2]

    SKELETON_COLOR = (0, 191, 255)  # Deep sky blue (RGB)
    EDGE_COLOR = (50, 205, 50)      # Green (RGB)
    LEAF_COLOR = (255, 255, 0)      # Yellow (RGB)
    INTER_COLOR = (255, 165, 0)     # Orange (RGB)
    TAIL_COLOR = (100, 255, 100)    # Light green (RGB)

    def draw_trajectory_tail(canvas, traj_history, tail_len):
        if traj_history is None or len(traj_history) < 2:
            return
        T_hist, K, _ = traj_history.shape
        actual_tail = min(tail_len, T_hist)
        for idx in range(K):
            traj = traj_history[-actual_tail:, idx, :]
            for t in range(len(traj) - 1):
                pt1, pt2 = traj[t], traj[t + 1]
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                row1, col1 = int(pt1[0]), int(pt1[1])
                row2, col2 = int(pt2[0]), int(pt2[1])
                if not (0 <= row1 < H and 0 <= col1 < W and 0 <= row2 < H and 0 <= col2 < W):
                    continue
                alpha = (t + 1) / len(traj)
                color = tuple(int(c * alpha) for c in TAIL_COLOR)
                cv2.line(canvas, (col1, row1), (col2, row2), color, 2)

    # Panel 1: RGB + frame info
    panel1 = rgb.copy()
    cv2.putText(panel1, f"Frame {frame_idx} - {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2: Depth-augmented SAM2 mask (SAM2 mask filtered by valid depth)
    panel2 = rgb.copy()
    mask_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    mask_overlay[depth_augmented_mask > 0] = (0, 255, 0)
    panel2 = cv2.addWeighted(panel2, 0.7, mask_overlay, 0.3, 0)
    cv2.putText(panel2, "SAM2 + Depth Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: Skeleton + keypoints only (black background)
    panel3 = np.zeros((H, W, 3), dtype=np.uint8)
    skeleton_thick = cv2.dilate(skeleton_mask, np.ones((3, 3), np.uint8), iterations=1)
    panel3[skeleton_thick > 0] = SKELETON_COLOR

    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        n_kp = len(kp_int)

        # Draw edges on panel3
        if edges is not None:
            for (i, j) in edges:
                if i < n_kp and j < n_kp:
                    row_i, col_i = kp_int[i]
                    row_j, col_j = kp_int[j]
                    if (0 <= row_i < H and 0 <= col_i < W and 0 <= row_j < H and 0 <= col_j < W):
                        cv2.line(panel3, (col_i, row_i), (col_j, row_j), EDGE_COLOR, 2)

        # BDLO coloring for panel3
        BRANCH_COLOR = (128, 0, 128)  # Purple
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx < 2:  # Branch
                    color = BRANCH_COLOR
                elif idx < 6:  # Leaf
                    color = LEAF_COLOR
                else:  # Intermediate
                    color = INTER_COLOR
                cv2.circle(panel3, (col, row), 5, color, -1)

    cv2.putText(panel3, "Skeleton + Keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    panel4 = rgb.copy()
    draw_trajectory_tail(panel3, traj_history_2d, tail_length)
    draw_trajectory_tail(panel4, traj_history_2d, tail_length)

    if keypoints_2d is not None and len(keypoints_2d) > 0:
        kp_int = keypoints_2d.astype(int)
        n_kp = len(kp_int)

        if edges is not None:
            for edge_idx, (i, j) in enumerate(edges):
                if i < n_kp and j < n_kp:
                    row_i, col_i = kp_int[i]
                    row_j, col_j = kp_int[j]
                    if (0 <= row_i < H and 0 <= col_i < W and 0 <= row_j < H and 0 <= col_j < W):
                        cv2.line(panel4, (col_i, row_i), (col_j, row_j), EDGE_COLOR, 3)

        # BDLO ordering: [branch_0, branch_1, leaf_0..leaf_3, intermediate...]
        BRANCH_COLOR = (128, 0, 128)  # Purple for branch nodes
        for idx, (row, col) in enumerate(kp_int):
            if 0 <= row < H and 0 <= col < W:
                if idx < 2:  # Branch nodes (0, 1)
                    color, label = BRANCH_COLOR, f"B{idx}"
                elif idx < 6:  # Leaf nodes (2, 3, 4, 5)
                    color, label = LEAF_COLOR, f"L{idx-2}"
                else:  # Intermediate nodes
                    color, label = INTER_COLOR, str(idx)
                cv2.circle(panel4, (col, row), 5, color, -1)
                cv2.putText(panel4, label, (col + 8, row + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.putText(panel4, f"BDLO: 2 branch, 4 leaf", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel4, f"Edges: {len(edges) if edges else 0}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    row1 = np.concatenate([panel1, panel2], axis=1)
    row2 = np.concatenate([panel3, panel4], axis=1)
    return np.concatenate([row1, row2], axis=0)


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame,
                 output_dir, n_keypoints=21, tail_length=60, fps=30,
                 keypoints_per_segment=None, sigma=2.0):
    """Process a single clip with the full method only.

    Pipeline:
      1. Track every frame, compute metrics on raw keypoints, save raw 3D.
      2. Apply Gaussian smoothing (sigma) along the time axis to the 3D keypoints.
      3. Re-project smoothed 3D to 2D and render the tracking video from those.
      4. Save smoothed 3D in a separate `smoothed_3d_keypoints.npz`.
    Evaluation metrics are always computed on the RAW keypoints, never smoothed.
    """

    clip_output_dir = output_dir / f'clip_{clip_idx}'
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = end_frame - start_frame
    print(f"\n  Clip {clip_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")

    K = transforms['K']
    intrinsics = np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])

    # Extract clip-specific EE poses
    clip_ee_poses = ee_poses_3d[start_frame:end_frame]

    # Tracker params (Full config baked in)
    tracker_params = {
        'intrinsics': intrinsics,
        'n_keypoints': n_keypoints,
        'target_branch_nodes': 2,
        'target_leaf_nodes': 4,
        'max_depth': 2000.0,
        'top_k_components': 1,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        'keypoints_per_segment': keypoints_per_segment,
        # Full config
        'enable_node_matching': True,
        'enable_geometry_constraint': True,
        'enable_ee_injection': True,
        'ee_poses_3d': clip_ee_poses,
    }

    method_name = 'Full'

    # Initialize fresh tracker for this clip
    tracker = WireTracker(**tracker_params)

    # Video writer (created post-loop, after smoothing)
    tracking_video_path = clip_output_dir / 'tracking_full.mp4'

    all_metrics_list = []
    traj_history = []           # raw 2D, kept for reference / debug only
    keypoints_3d_history = []
    # Per-frame state captured during tracking, used by the post-loop renderer
    render_state = []           # list of {global_idx, mode, skeleton_mask, depth_augmented_mask}
    stored_edges = None
    stored_reference_lengths = None

    for local_idx, global_idx in enumerate(tqdm(range(start_frame, end_frame), desc=f"    Clip {clip_idx}")):
        rgb = cv2.cvtColor(data['color'][global_idx], cv2.COLOR_BGR2RGB)  # Source is BGR
        depth = data['depth'][global_idx].astype(np.float32)
        dlo_mask = data['dlo_masks'][global_idx]
        exclude_mask = (1 - dlo_mask).astype(np.uint8)

        result = tracker.process_frame(
            depth=depth, arm_depth=None, rgb=rgb,
            precomputed_arm_mask=exclude_mask,
        )

        if result['success']:
            keypoints = result['keypoints']
            keypoints_2d = result['keypoints_2d']
            edges = result['edges']
            traj_history.append(keypoints_2d.copy())
            keypoints_3d_history.append(keypoints.copy())
            if stored_edges is None and edges is not None:
                stored_edges = list(edges)
            if stored_reference_lengths is None and tracker.reference_lengths is not None:
                stored_reference_lengths = tracker.reference_lengths.copy()

            # Print segment and edge lengths on initialization (frame 0)
            if local_idx == 0:
                print(f"\n  === {method_name} Initialization Summary ===")
                print(f"  Total keypoints: {len(keypoints)}")
                print(f"  Total edges: {len(edges)}")

                # Edge lengths
                edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
                print(f"  Edge lengths (mm): {[f'{l:.1f}' for l in edge_lengths]}")
                print(f"  Edge length mean: {np.mean(edge_lengths):.1f} mm, std: {np.std(edge_lengths):.1f} mm")

                # Segment info
                if hasattr(tracker, 'segment_edges') and tracker.segment_edges is not None:
                    seg_names = ['ee0→b0', 'ee1→b1', 'b0→free0', 'b1→free1', 'trunk']
                    print(f"  Segment edges:")
                    for seg_idx, seg_edges in enumerate(tracker.segment_edges):
                        seg_name = seg_names[seg_idx] if seg_idx < len(seg_names) else f'seg{seg_idx}'
                        seg_len = sum(np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg_edges)
                        n_edges = len(seg_edges)
                        avg_edge = seg_len / n_edges if n_edges > 0 else 0
                        print(f"    {seg_name}: {n_edges} edges, length={seg_len:.1f}mm, avg_edge={avg_edge:.1f}mm")

                # Reference lengths
                if tracker.reference_lengths is not None:
                    print(f"  Reference lengths (mm): {[f'{l:.1f}' for l in tracker.reference_lengths]}")
                print(f"  ==============================\n")
        else:
            keypoints = np.empty((0, 3))
            keypoints_2d = np.empty((0, 2))
            edges = []
            if len(traj_history) > 0:
                traj_history.append(np.full_like(traj_history[-1], np.nan))
            else:
                traj_history.append(np.full((n_keypoints, 2), np.nan))
            keypoints_3d_history.append(np.full((n_keypoints, 3), np.nan))

        if result['success'] and tracker.reference_lengths is not None:
            skeleton_pc = result.get('skeleton_pc')

            # Augment with EE positions for evaluation
            ee_pos = clip_ee_poses[local_idx]
            if skeleton_pc is not None and len(skeleton_pc) > 0:
                if ee_pos is not None and len(ee_pos) > 10000:
                    ref_pc = np.vstack([skeleton_pc, np.array(ee_pos, dtype=np.float32).reshape(-1, 3)])
                else:
                    ref_pc = skeleton_pc
            else:
                ref_pc = np.array(ee_pos, dtype=np.float32).reshape(-1, 3) if ee_pos is not None else np.empty((0, 3))

            ref_lengths = stored_reference_lengths if stored_reference_lengths is not None else tracker.reference_lengths
            edge_m = compute_edge_metrics(keypoints, edges, ref_lengths)
            pos_m = compute_position_metrics(keypoints, ref_pc, extra_gt_points=None)

            # Compute Chamfer Distance metrics
            n_ref_points = len(ref_pc) if ref_pc is not None and len(ref_pc) > 0 else 100
            pred_cloud = sample_points_on_edges(keypoints, edges, n_ref_points)
            cd_m = compute_chamfer_metrics(pred_cloud, ref_pc)

            metrics = {
                'frame': local_idx,
                'global_frame': global_idx,
                'success': True,
                'edge_pct_mean': edge_m['pct_mean'],
                'edge_pct_std': edge_m['pct_std'],
                'edge_pct_max': edge_m['pct_max'],
                'edge_rmse_mm': edge_m['rmse_mm'],
                'edge_under_2pct': edge_m['under_2pct'],
                'edge_under_5pct': edge_m['under_5pct'],
                'edge_under_10pct': edge_m['under_10pct'],
                'pos_rmse_mm': pos_m['rmse_mm'],
                'pos_under_2mm': pos_m['under_2mm'],
                'pos_under_5mm': pos_m['under_5mm'],
                'pos_under_10mm': pos_m['under_10mm'],
                # Chamfer Distance metrics
                'cd': cd_m['cd'],
                'cd_pred2ref': cd_m['pred2ref_avg'],
                'cd_ref2pred': cd_m['ref2pred_avg'],
                'precision_2mm': cd_m['precision_2mm'],
                'precision_5mm': cd_m['precision_5mm'],
                'precision_10mm': cd_m['precision_10mm'],
                'recall_2mm': cd_m['recall_2mm'],
                'recall_5mm': cd_m['recall_5mm'],
                'recall_10mm': cd_m['recall_10mm'],
                'f_2mm': cd_m['f_2mm'],
                'f_5mm': cd_m['f_5mm'],
                'f_10mm': cd_m['f_10mm'],
            }
        else:
            metrics = {
                'frame': local_idx,
                'global_frame': global_idx,
                'success': False,
                'edge_pct_mean': 0.0, 'edge_pct_std': 0.0, 'edge_pct_max': 0.0,
                'edge_rmse_mm': 0.0, 'edge_under_2pct': 0.0, 'edge_under_5pct': 0.0,
                'edge_under_10pct': 0.0, 'pos_rmse_mm': 0.0, 'pos_under_2mm': 0.0,
                'pos_under_5mm': 0.0, 'pos_under_10mm': 0.0,
                'cd': 0.0, 'cd_pred2ref': 0.0, 'cd_ref2pred': 0.0,
                'precision_2mm': 0.0, 'precision_5mm': 0.0, 'precision_10mm': 0.0,
                'recall_2mm': 0.0, 'recall_5mm': 0.0, 'recall_10mm': 0.0,
                'f_2mm': 0.0, 'f_5mm': 0.0, 'f_10mm': 0.0,
            }

        all_metrics_list.append(metrics)

        # Capture per-frame visualization state for the post-loop renderer
        render_state.append({
            'global_idx': global_idx,
            'mode': result.get('mode', 'failed'),
            'skeleton_mask': result.get('skeleton_mask', np.zeros_like(depth, dtype=np.uint8)),
            'depth_augmented_mask': result.get('foreground_mask', dlo_mask),
        })

    # ==================================================================
    # POST-PROCESS: SMOOTH 3D KEYPOINTS + RENDER VIDEO FROM SMOOTHED
    # (metrics above were already computed on the RAW keypoints)
    # ==================================================================
    print(f"    Smoothing 3D trajectories (sigma={sigma}) and rendering video...")

    raw_3d = np.array(keypoints_3d_history)  # T x K x 3 (NaN where tracking failed)
    smoothed_3d = smooth_trajectories(raw_3d, sigma=sigma)

    # Re-project smoothed 3D → 2D using the tracker's own projection (row, col).
    smoothed_2d_list = []
    for kp_3d in smoothed_3d:
        if kp_3d is None or len(kp_3d) == 0 or np.all(np.isnan(kp_3d)):
            smoothed_2d_list.append(np.full((n_keypoints, 2), np.nan, dtype=np.float32))
        else:
            kp_2d = tracker._project_3d_to_2d(kp_3d.astype(np.float64))
            if kp_2d is None or len(kp_2d) == 0:
                kp_2d = np.full((n_keypoints, 2), np.nan, dtype=np.float32)
            smoothed_2d_list.append(np.asarray(kp_2d, dtype=np.float32))

    # Render the tracking video using smoothed 2D keypoints.
    smoothed_traj_hist = []
    tracking_video_writer = None
    for local_idx, ctx in enumerate(render_state):
        global_idx = ctx['global_idx']
        rgb = cv2.cvtColor(data['color'][global_idx], cv2.COLOR_BGR2RGB)
        smoothed_kp_2d = smoothed_2d_list[local_idx]
        smoothed_traj_hist.append(smoothed_kp_2d.copy())

        full_vis = create_full_tracking_visualization(
            rgb=rgb,
            depth_augmented_mask=ctx['depth_augmented_mask'],
            skeleton_mask=ctx['skeleton_mask'],
            keypoints_2d=smoothed_kp_2d,
            edges=stored_edges if stored_edges else [],
            frame_idx=local_idx,
            mode=ctx['mode'],
            traj_history_2d=np.array(smoothed_traj_hist),
            tail_length=tail_length,
        )

        if tracking_video_writer is None:
            Ht, Wt = full_vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            tracking_video_writer = cv2.VideoWriter(str(tracking_video_path), fourcc, fps, (Wt, Ht))

        tracking_video_writer.write(cv2.cvtColor(full_vis, cv2.COLOR_RGB2BGR))

    if tracking_video_writer is not None:
        tracking_video_writer.release()

    # Save per-frame CSV
    per_frame_csv = clip_output_dir / 'per_frame.csv'
    with open(per_frame_csv, 'w') as f:
        f.write('Frame,GlobalFrame,Method,EdgePctMean,EdgePctStd,EdgePctMax,EdgeRMSE,PosRMSE,'
                'Edge<2%,Edge<5%,Edge<10%,Pos<2mm,Pos<5mm,Pos<10mm,'
                'CD,Pred2Ref,Ref2Pred,Prec@2mm,Prec@5mm,Prec@10mm,Rec@2mm,Rec@5mm,Rec@10mm,F@2mm,F@5mm,F@10mm\n')
        for local_idx in range(n_frames):
            m = all_metrics_list[local_idx]
            f.write(f"{local_idx},{m['global_frame']},{method_name},{m['edge_pct_mean']:.6f},{m['edge_pct_std']:.6f},"
                    f"{m['edge_pct_max']:.6f},{m['edge_rmse_mm']:.6f},{m['pos_rmse_mm']:.6f},"
                    f"{m['edge_under_2pct']:.4f},{m['edge_under_5pct']:.4f},{m['edge_under_10pct']:.4f},"
                    f"{m['pos_under_2mm']:.4f},{m['pos_under_5mm']:.4f},{m['pos_under_10mm']:.4f},"
                    f"{m['cd']:.4f},{m['cd_pred2ref']:.4f},{m['cd_ref2pred']:.4f},"
                    f"{m['precision_2mm']:.4f},{m['precision_5mm']:.4f},{m['precision_10mm']:.4f},"
                    f"{m['recall_2mm']:.4f},{m['recall_5mm']:.4f},{m['recall_10mm']:.4f},"
                    f"{m['f_2mm']:.4f},{m['f_5mm']:.4f},{m['f_10mm']:.4f}\n")

    # Compute clip summary (skipping the init frame)
    metrics_list = all_metrics_list[1:] if len(all_metrics_list) > 1 else all_metrics_list

    if len(metrics_list) > 0:
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
        cd_pred2ref_vals = [m['cd_pred2ref'] for m in metrics_list if m['success']]
        cd_ref2pred_vals = [m['cd_ref2pred'] for m in metrics_list if m['success']]
        precision_2 = [m['precision_2mm'] for m in metrics_list if m['success']]
        precision_5 = [m['precision_5mm'] for m in metrics_list if m['success']]
        precision_10 = [m['precision_10mm'] for m in metrics_list if m['success']]
        recall_2 = [m['recall_2mm'] for m in metrics_list if m['success']]
        recall_5 = [m['recall_5mm'] for m in metrics_list if m['success']]
        recall_10 = [m['recall_10mm'] for m in metrics_list if m['success']]
        f_2 = [m['f_2mm'] for m in metrics_list if m['success']]
        f_5 = [m['f_5mm'] for m in metrics_list if m['success']]
        f_10 = [m['f_10mm'] for m in metrics_list if m['success']]

        summary_row = {
            'method': method_name,
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
            'cd_pred2ref_avg': np.mean(cd_pred2ref_vals) if cd_pred2ref_vals else 0.0,
            'cd_ref2pred_avg': np.mean(cd_ref2pred_vals) if cd_ref2pred_vals else 0.0,
            'precision_2mm': np.mean(precision_2) if precision_2 else 0.0,
            'precision_5mm': np.mean(precision_5) if precision_5 else 0.0,
            'precision_10mm': np.mean(precision_10) if precision_10 else 0.0,
            'recall_2mm': np.mean(recall_2) if recall_2 else 0.0,
            'recall_5mm': np.mean(recall_5) if recall_5 else 0.0,
            'recall_10mm': np.mean(recall_10) if recall_10 else 0.0,
            'f_2mm': np.mean(f_2) if f_2 else 0.0,
            'f_5mm': np.mean(f_5) if f_5 else 0.0,
            'f_10mm': np.mean(f_10) if f_10 else 0.0,
        }
    else:
        summary_row = None

    summary_rows = [summary_row] if summary_row is not None else []

    # Save summary with three tables
    summary_txt = clip_output_dir / 'summary.txt'
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
        f.write("Chamfer Distance Metrics \n")
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

    # Save RMSE over time plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    color_full = 'blue'

    frames = [m['frame'] for m in all_metrics_list]
    edge_rmses = [m['edge_rmse_mm'] for m in all_metrics_list]
    pos_rmses = [m['pos_rmse_mm'] for m in all_metrics_list]

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
    plt.savefig(clip_output_dir / 'rmse_over_time.png', dpi=150)
    plt.close(fig)

    # Save CD over time plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    cd_values = [m['cd'] for m in all_metrics_list]
    pred2ref_values = [m['cd_pred2ref'] for m in all_metrics_list]
    ref2pred_values = [m['cd_ref2pred'] for m in all_metrics_list]

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
    plt.savefig(clip_output_dir / 'cd_over_time.png', dpi=150)
    plt.close(fig)

    # Save 3D keypoints (raw — used for evaluation)
    keypoints_3d_path = clip_output_dir / '3d_keypoints.npz'
    np.savez(
        keypoints_3d_path,
        full=raw_3d,
        edge_connection=np.array(stored_edges) if stored_edges else np.array([]),
        reference_lengths=np.array(stored_reference_lengths) if stored_reference_lengths is not None else np.array([]),
    )

    # Save smoothed 3D keypoints (used for visualization, NOT for evaluation)
    smoothed_3d_path = clip_output_dir / 'smoothed_3d_keypoints.npz'
    np.savez(
        smoothed_3d_path,
        full=smoothed_3d,
        sigma=np.array(sigma),
        edge_connection=np.array(stored_edges) if stored_edges else np.array([]),
        reference_lengths=np.array(stored_reference_lengths) if stored_reference_lengths is not None else np.array([]),
    )

    print(f"    Saved: {clip_output_dir}")

    return {
        'clip_idx': clip_idx,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'all_metrics': {method_name: all_metrics_list},
        'summary_rows': summary_rows,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Clean BDLO tracking experiment (Full method only)')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index (0-48)')
    parser.add_argument('--clip_seconds', type=int, default=20, help='Clip duration in seconds (default: 20)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    parser.add_argument('--n_keypoints', type=int, default=21, help='Number of keypoints (default: 21 for BDLO)')
    parser.add_argument('--keypoints_per_segment', type=int, nargs=5, default=None,
                        help='Intermediate keypoints per segment: [ee0, ee1, free0, free1, trunk] (default: auto)')
    parser.add_argument('--skip_clips', type=int, nargs='*', default=[],
                        help='Clip indices to skip (e.g., --skip_clips 0 1)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma applied to 3D trajectories for visualization '
                             '(default: 3.0; metrics are still on raw)')
    args = parser.parse_args()

    skip_clips_set = set(args.skip_clips)

    # Compute n_keypoints from keypoints_per_segment if provided
    n_keypoints = args.n_keypoints
    if args.keypoints_per_segment is not None:
        # BDLO: 2 branch + 4 leaf + sum(intermediate)
        computed_n = 2 + 4 + sum(args.keypoints_per_segment)
        if n_keypoints != computed_n:
            print(f"Auto-computed n_keypoints={computed_n} from keypoints_per_segment={args.keypoints_per_segment}")
            n_keypoints = computed_n

    # Paths (relative to this script: input_data/bdlo/chunk_<N>/, calibration in input_data/bdlo/calibration/)
    script_dir = Path(__file__).resolve().parent
    data_base = script_dir / 'input_data' / 'bdlo'
    calib_dir = data_base / 'calibration'
    output_base = script_dir / 'output'/ 'bdlo'

    chunk_dir = data_base / f'chunk_{args.chunk}'
    output_dir = output_base / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"BDLO TRACKING (FULL METHOD ONLY) - Chunk {args.chunk}")
    print("=" * 80)

    # Load data
    print(f"\nLoading chunk_{args.chunk} data...")
    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(calib_dir)

    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  BDLO masks: {data['dlo_masks'].shape if data['dlo_masks'] is not None else 'None'}")
    print(f"  Total frames: {data['n_frames']}")

    if data['dlo_masks'] is None:
        print("ERROR: No BDLO masks found!")
        return

    # Precompute EE positions
    print("\nConverting EE poses to camera frame...")
    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )

    # Calculate clips (include last partial clip)
    frames_per_clip = args.clip_seconds * args.fps
    n_clips = (data['n_frames'] + frames_per_clip - 1) // frames_per_clip  # ceiling division

    print(f"\nClip configuration:")
    print(f"  Clip duration: {args.clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")
    last_clip_frames = data['n_frames'] - (n_clips - 1) * frames_per_clip
    if last_clip_frames < frames_per_clip:
        print(f"  Last clip: {last_clip_frames} frames ({last_clip_frames / args.fps:.1f}s)")

    if skip_clips_set:
        print(f"  Skipping clips: {sorted(skip_clips_set)}")

    # Process each clip
    all_clip_results = []
    for clip_idx in range(n_clips):
        if clip_idx in skip_clips_set:
            print(f"\n  Skipping clip {clip_idx} (in skip list)")
            continue

        start_frame = clip_idx * frames_per_clip
        end_frame = min(start_frame + frames_per_clip, data['n_frames'])

        clip_result = process_clip(
            data=data,
            transforms=transforms,
            ee_poses_3d=ee_poses_3d,
            clip_idx=clip_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            output_dir=output_dir,
            n_keypoints=n_keypoints,
            fps=args.fps,
            keypoints_per_segment=args.keypoints_per_segment,
            sigma=args.sigma,
        )
        all_clip_results.append(clip_result)

    # Create chunk summary
    chunk_summary_dir = output_dir / 'chunk_summary'
    chunk_summary_dir.mkdir(parents=True, exist_ok=True)

    method_names = ['Full']

    # Aggregate all clips' metrics
    all_clips_metrics = {m: [] for m in method_names}
    for clip_result in all_clip_results:
        for method in method_names:
            all_clips_metrics[method].extend(clip_result['all_metrics'][method])

    # Save stacked per-frame CSV
    stacked_csv = chunk_summary_dir / 'all_clips_metrics.csv'
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
        cd_pred2ref_vals = [m['cd_pred2ref'] for m in metrics_list if m['success']]
        cd_ref2pred_vals = [m['cd_ref2pred'] for m in metrics_list if m['success']]
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
            'cd_pred2ref_avg': np.mean(cd_pred2ref_vals) if cd_pred2ref_vals else 0.0,
            'cd_ref2pred_avg': np.mean(cd_ref2pred_vals) if cd_ref2pred_vals else 0.0,
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

    # Compute chunk aggregate summary (Clip-weighted: average each clip's summary)
    chunk_summary_clip_weighted = []
    for method in method_names:
        clip_summaries = []
        for clip_result in all_clip_results:
            for s in clip_result['summary_rows']:
                if s['method'] == method:
                    clip_summaries.append(s)
                    break

        if len(clip_summaries) == 0:
            continue

        chunk_summary_clip_weighted.append({
            'method': method,
            'edge_pct_mean_avg': np.mean([s['edge_pct_mean_avg'] for s in clip_summaries]),
            'edge_pct_mean_std': np.std([s['edge_pct_mean_avg'] for s in clip_summaries]),
            'edge_rmse_avg': np.mean([s['edge_rmse_avg'] for s in clip_summaries]),
            'edge_rmse_std': np.std([s['edge_rmse_avg'] for s in clip_summaries]),
            'edge_under_2pct': np.mean([s['edge_under_2pct'] for s in clip_summaries]),
            'edge_under_5pct': np.mean([s['edge_under_5pct'] for s in clip_summaries]),
            'edge_under_10pct': np.mean([s['edge_under_10pct'] for s in clip_summaries]),
            'pos_rmse_avg': np.mean([s['pos_rmse_avg'] for s in clip_summaries]),
            'pos_rmse_std': np.std([s['pos_rmse_avg'] for s in clip_summaries]),
            'pos_under_2mm': np.mean([s['pos_under_2mm'] for s in clip_summaries]),
            'pos_under_5mm': np.mean([s['pos_under_5mm'] for s in clip_summaries]),
            'pos_under_10mm': np.mean([s['pos_under_10mm'] for s in clip_summaries]),
            # CD metrics
            'cd_avg': np.mean([s['cd_avg'] for s in clip_summaries]),
            'cd_std': np.std([s['cd_avg'] for s in clip_summaries]),
            'cd_pred2ref_avg': np.mean([s['cd_pred2ref_avg'] for s in clip_summaries]),
            'cd_ref2pred_avg': np.mean([s['cd_ref2pred_avg'] for s in clip_summaries]),
            'precision_2mm': np.mean([s['precision_2mm'] for s in clip_summaries]),
            'precision_5mm': np.mean([s['precision_5mm'] for s in clip_summaries]),
            'precision_10mm': np.mean([s['precision_10mm'] for s in clip_summaries]),
            'recall_2mm': np.mean([s['recall_2mm'] for s in clip_summaries]),
            'recall_5mm': np.mean([s['recall_5mm'] for s in clip_summaries]),
            'recall_10mm': np.mean([s['recall_10mm'] for s in clip_summaries]),
            'f_2mm': np.mean([s['f_2mm'] for s in clip_summaries]),
            'f_5mm': np.mean([s['f_5mm'] for s in clip_summaries]),
            'f_10mm': np.mean([s['f_10mm'] for s in clip_summaries]),
        })


    # Save chunk aggregate summary with both aggregation methods
    chunk_summary_txt = chunk_summary_dir / 'chunk_aggregate_summary.txt'
    with open(chunk_summary_txt, 'w') as f:
        f.write(f"Chunk {args.chunk} Aggregate Summary ({n_clips} clips)\n")
        f.write("=" * 100 + "\n\n")

        f.write(">>> FRAME-WEIGHTED (all frames pooled, each frame weighted equally)\n\n")
        write_summary_tables(f, chunk_summary_frame_weighted)

        f.write("\n" + "=" * 100 + "\n\n")

        f.write(">>> CLIP-WEIGHTED (each clip's summary averaged, each clip weighted equally)\n\n")
        write_summary_tables(f, chunk_summary_clip_weighted)

    # Combine 3D keypoints from all clips (raw and smoothed)
    combined_3d_keypoints = {m: [] for m in method_names}
    combined_3d_keypoints_smoothed = {m: [] for m in method_names}
    combined_edges = None
    combined_reference_lengths = []
    for clip_result in all_clip_results:
        clip_dir = output_dir / f"clip_{clip_result['clip_idx']}"
        clip_kp_path = clip_dir / '3d_keypoints.npz'
        if clip_kp_path.exists():
            clip_kp = np.load(clip_kp_path)
            combined_3d_keypoints['Full'].append(clip_kp['full'])
            if combined_edges is None and len(clip_kp['edge_connection']) > 0:
                combined_edges = clip_kp['edge_connection']
            if len(clip_kp['reference_lengths']) > 0:
                combined_reference_lengths.append(clip_kp['reference_lengths'])
        smoothed_kp_path = clip_dir / 'smoothed_3d_keypoints.npz'
        if smoothed_kp_path.exists():
            smoothed_kp = np.load(smoothed_kp_path)
            combined_3d_keypoints_smoothed['Full'].append(smoothed_kp['full'])

    # Stack and save combined keypoints (raw)
    combined_kp_path = chunk_summary_dir / 'all_clips_3d_keypoints.npz'
    np.savez(
        combined_kp_path,
        full=np.concatenate(combined_3d_keypoints['Full'], axis=0) if combined_3d_keypoints['Full'] else np.array([]),
        edge_connection=combined_edges if combined_edges is not None else np.array([]),
        reference_lengths_per_clip=np.array(combined_reference_lengths) if combined_reference_lengths else np.array([]),
    )

    # Stack and save combined SMOOTHED keypoints (visualization-only — not used in metrics)
    combined_smoothed_path = chunk_summary_dir / 'all_clips_smoothed_3d_keypoints.npz'
    np.savez(
        combined_smoothed_path,
        full=np.concatenate(combined_3d_keypoints_smoothed['Full'], axis=0) if combined_3d_keypoints_smoothed['Full'] else np.array([]),
        sigma=np.array(args.sigma),
        edge_connection=combined_edges if combined_edges is not None else np.array([]),
        reference_lengths_per_clip=np.array(combined_reference_lengths) if combined_reference_lengths else np.array([]),
    )


    # Print final summary
    print("\n" + "=" * 100)
    print("CHUNK AGGREGATE SUMMARY")
    print("=" * 100)

    print("\n>>> FRAME-WEIGHTED (all frames pooled)")
    print_summary_tables(chunk_summary_frame_weighted)

    print("\n" + "=" * 100)
    print("\n>>> CLIP-WEIGHTED (each clip's summary averaged)")
    print_summary_tables(chunk_summary_clip_weighted)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Per-clip: {output_dir}/clip_*/")
    print(f"  Chunk summary: {chunk_summary_dir}/")


if __name__ == "__main__":
    main()
