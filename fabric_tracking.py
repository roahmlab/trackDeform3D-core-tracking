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
from pathlib import Path
from tqdm import tqdm

from tracker.fabric_tracker import FabricTracker, FabricTrackerFull
from utils.smoothing import smooth_trajectories
from utils.transforms import load_transforms, pose7_to_matrix, get_ee_positions_cam
from utils.metrics_fabric import (filter_ee_outliers, compute_edge_metrics,
                                  compute_position_metrics, sample_points_on_faces,
                                  compute_chamfer_metrics)
from utils.pointcloud import extract_surface_point_cloud
from utils.data_loading import load_chunk_data
from utils.init_visualization import save_init_visualization_3d_fabric
from utils.evaluation import (evaluate_frames, summarize, write_clip_summary,
                              write_chunk_aggregate, print_summary_tables)
from utils.visualization import grid_gradient_t, render_tracking_video


# Paths (resolved relative to this script: input_data/fabric/chunk_<N>/)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE = SCRIPT_DIR / "input_data" / "fabric"
CALIB_DIR = DATA_BASE / "calibration"
OUTPUT_BASE = SCRIPT_DIR / "output" / "fabric"

# Frame rate
FPS = 30


# ============================================================================
# CLIP PROCESSING
# ============================================================================

def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame,
                 output_dir, grid_rows, grid_cols, tail_length=60, fps=30, sigma=2.0):
    """Track one clip, save keypoints, render the video, evaluate.

    Stages (in this order):
      1. TRACK      -> 3d_keypoints.npz + smoothed_3d_keypoints.npz
      2. VISUALIZE  -> tracking_full.mp4 from the SMOOTHED keypoints
      3. EVALUATE   -> summary.txt (metrics on the RAW keypoints)
    """
    clip_dir = output_dir / f"clip_{clip_idx:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    n_frames = end_frame - start_frame
    K = transforms['K']

    color = data['color'][start_frame:end_frame]
    depth = data['depth'][start_frame:end_frame]
    fg_mask = data['masks'][start_frame:end_frame]

    clip_ee_poses = ee_poses_3d[start_frame:end_frame] if ee_poses_3d is not None else None

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
    tracker = FabricTrackerFull(**tracker_params)

    # ------------------------------------------------------------------
    # 1. TRACK — raw keypoints + per-frame reference clouds for stage 3
    # ------------------------------------------------------------------
    keypoints_hist = []
    success_flags = []
    ref_clouds = []
    stored_edges = None
    stored_ref_lens = None
    init_vis_saved = False

    print(f"\n  Processing clip {clip_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")

    for frame_idx in tqdm(range(n_frames), desc=f"  Clip {clip_idx}"):
        d = depth[frame_idx]
        mask = fg_mask[frame_idx]

        surface_pc = extract_surface_point_cloud(mask, d, K)
        ref_clouds.append(surface_pc)

        result = tracker.process_frame(d, mask, frame_idx)

        mode = result.get('mode', 'unknown')
        keypoints = result.get('keypoints')
        edges = result.get('edges', tracker.grid_edges if hasattr(tracker, 'grid_edges') else [])

        keypoints_hist.append(keypoints)
        success_flags.append(keypoints is not None and len(keypoints) > 0)

        if stored_edges is None and edges is not None and len(edges) > 0:
            stored_edges = list(edges)
        if stored_ref_lens is None and result.get('success'):
            if hasattr(tracker, 'reference_lengths') and tracker.reference_lengths is not None:
                stored_ref_lens = tracker.reference_lengths

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

            save_init_visualization_3d_fabric(
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

    # Save raw + smoothed keypoints
    n_grid = grid_rows * grid_cols
    raw_3d = np.array([kp if kp is not None else np.full((n_grid, 3), np.nan)
                       for kp in keypoints_hist])
    smoothed_3d = smooth_trajectories(raw_3d, sigma=sigma)
    edges_arr = np.array(stored_edges) if stored_edges else np.array([])
    ref_lens_arr = (np.array(list(stored_ref_lens.values()))
                    if stored_ref_lens else np.array([]))
    np.savez(clip_dir / '3d_keypoints.npz',
             full=raw_3d, edge_connections=edges_arr, reference_lengths=ref_lens_arr)
    np.savez(clip_dir / 'smoothed_3d_keypoints.npz',
             full=smoothed_3d, sigma=np.array(sigma),
             edge_connections=edges_arr, reference_lengths=ref_lens_arr)

    # ------------------------------------------------------------------
    # 2. VISUALIZE — smoothed keypoints, single-panel style
    # ------------------------------------------------------------------
    if stored_edges and any(success_flags):
        print(f"  Rendering tracking video (sigma={sigma})...")
        node_t = grid_gradient_t(grid_rows, grid_cols)
        anchors = set(tracker.CORNER_INDICES) if hasattr(tracker, 'CORNER_INDICES') else set()
        render_tracking_video(clip_dir / 'tracking_fabric.mp4',
                              color, smoothed_3d, K, edges_arr, node_t,
                              anchors=anchors, masks=fg_mask,
                              fps=fps, trail=tail_length)

    # ------------------------------------------------------------------
    # 3. EVALUATE — raw keypoints -> summary.txt
    # ------------------------------------------------------------------
    def cd_fn(kp, ref_pc):
        n_faces = (grid_rows - 1) * (grid_cols - 1)
        n_ref_points = len(ref_pc) if ref_pc is not None else 5000
        n_samples_per_face = max(10, n_ref_points // n_faces)
        pred_cloud = sample_points_on_faces(kp, grid_rows, grid_cols,
                                            n_samples_per_face=n_samples_per_face)
        return compute_chamfer_metrics(pred_cloud, ref_pc)

    all_metrics_list = evaluate_frames(
        kp_seq=keypoints_hist,
        success_seq=success_flags,
        global_indices=list(range(start_frame, end_frame)),
        edges=stored_edges,
        reference_lengths=stored_ref_lens,
        ref_clouds=ref_clouds,
        edge_fn=compute_edge_metrics,
        pos_fn=compute_position_metrics,
        cd_fn=cd_fn,
    )

    summary_row = summarize(all_metrics_list, method=method_name, skip_first=True)
    write_clip_summary(
        clip_dir / 'summary.txt',
        f"Clip {clip_idx} Summary (frames {start_frame}-{end_frame}, {n_frames} frames)",
        summary_row,
    )

    print(f"    Saved: {clip_dir}")
    if summary_row is not None:
        print_summary_tables([summary_row])

    return {
        'clip_idx': clip_idx,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'all_metrics': {method_name: all_metrics_list},
        'summary_rows': [summary_row] if summary_row is not None else [],
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
    parser.add_argument('--tail_length', type=int, default=30,
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

    data = load_chunk_data(chunk_dir, mask_file='fg_mask.npz', mask_key='fg_mask',
                           max_frames=args.max_frames, required=True)
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

    summary_dir = output_dir / 'chunk_summary'
    summary_dir.mkdir(exist_ok=True)

    pooled_rows = []
    per_clip_summary_rows = []
    for clip_result in all_clip_results:
        pooled_rows.extend(clip_result['all_metrics']['Full'])
        per_clip_summary_rows.extend(clip_result['summary_rows'])

    write_chunk_aggregate(
        summary_dir / 'chunk_aggregate_summary.txt',
        f"Fabric Chunk {args.chunk} Aggregate Summary ({n_clips} clips)",
        pooled_rows, per_clip_summary_rows,
    )
    print_summary_tables(per_clip_summary_rows)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Per-clip: {output_dir}/clip_*/")
    print(f"  Chunk summary: {summary_dir}/")
    print("\nDone!")

if __name__ == "__main__":
    main()
