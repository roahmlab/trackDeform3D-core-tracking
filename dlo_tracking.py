"""DLO (single open-chain) tracking driver.

Per clip the pipeline runs in three separate stages (in this order):
  1. TRACK      — WireTracker over every frame; save the raw keypoints as
                  3d_keypoints.npz and the Gaussian-smoothed ones (sigma) as
                  smoothed_3d_keypoints.npz.
  2. VISUALIZE  — render tracking_full.mp4 from the SMOOTHED keypoints
                  (utils.visualization, single-panel style).
  3. EVALUATE   — metrics on the RAW keypoints (utils.evaluation) written to
                  summary.txt; the chunk level writes chunk_aggregate_summary.txt.

Usage:
    python dlo_tracking.py --chunk 1 --clip_seconds 10 --n_keypoints 15
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tracker.wire_tracker import WireTracker
from utils.data_loading import load_chunk_data
from utils.smoothing import smooth_trajectories
from utils.transforms import load_transforms, pose7_to_matrix, get_ee_positions_cam
from utils.metrics_wire import (compute_edge_metrics, compute_position_metrics,
                                sample_points_on_edges, compute_chamfer_metrics)
from utils.evaluation import (evaluate_frames, summarize, write_clip_summary,
                              write_chunk_aggregate, print_summary_tables)
from utils.visualization import path_gradient_t, render_tracking_video


# ============================================================================
# CLIP PROCESSING:  1. track (+ save)  2. visualize  3. evaluate
# ============================================================================

def process_clip(data, transforms, ee_poses_3d, clip_idx, start_frame, end_frame,
                 output_dir, n_keypoints=15, tail_length=60, fps=30, sigma=2.0):
    """Track one clip, save keypoints, render the video, evaluate.

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

    clip_ee_poses = ee_poses_3d[start_frame:end_frame]

    tracker_params = {
        'intrinsics': intrinsics,
        'n_keypoints': n_keypoints,
        'target_branch_nodes': 0,
        'target_leaf_nodes': 2,
        'max_depth': 2000.0,
        'top_k_components': 1,
        'n_outer_iterations': 20,
        'n_edge_iterations': 15,
        'edge_weight': 0.5,
        'edge_tolerance': 0.02,
        'repulsion_iterations': 200,
        'repulsion_lr': 10.0,
        'repulsion_k_neighbors': 3,
        'enable_node_matching': True,
        'enable_geometry_constraint': True,
        'enable_ee_injection': True,
        'ee_poses_3d': clip_ee_poses,
    }

    method_name = 'Full'
    tracker = WireTracker(**tracker_params)

    # ------------------------------------------------------------------
    # 1. TRACK — raw keypoints + per-frame reference clouds for stage 3
    # ------------------------------------------------------------------
    keypoints_3d_history = []
    success_flags = []
    ref_clouds = []
    stored_edges = None
    stored_reference_lengths = None

    for local_idx, global_idx in enumerate(tqdm(range(start_frame, end_frame), desc=f"    Clip {clip_idx}")):
        rgb = data['color'][global_idx]
        depth = data['depth'][global_idx].astype(np.float32)
        dlo_mask = data['masks'][global_idx]
        exclude_mask = (1 - dlo_mask).astype(np.uint8)

        result = tracker.process_frame(
            depth=depth, arm_depth=None, rgb=rgb,
            precomputed_arm_mask=exclude_mask,
        )

        if result['success']:
            keypoints = result['keypoints']
            keypoints_3d_history.append(keypoints.copy())
            success_flags.append(True)
            if stored_edges is None and result['edges'] is not None:
                stored_edges = list(result['edges'])
            if stored_reference_lengths is None and tracker.reference_lengths is not None:
                stored_reference_lengths = tracker.reference_lengths.copy()

            if local_idx == 0:
                edges = result['edges']
                print(f"\n  === {method_name} Initialization Summary ===")
                print(f"  Total keypoints: {len(keypoints)}")
                print(f"  Total edges: {len(edges)}")
                edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
                print(f"  Edge lengths (mm): {[f'{l:.1f}' for l in edge_lengths]}")
                print(f"  Edge length mean: {np.mean(edge_lengths):.1f} mm, std: {np.std(edge_lengths):.1f} mm")
                if tracker.reference_lengths is not None:
                    print(f"  Reference lengths (mm): {[f'{l:.1f}' for l in tracker.reference_lengths]}")
                print(f"  ==============================\n")
        else:
            keypoints_3d_history.append(np.full((n_keypoints, 3), np.nan))
            success_flags.append(False)

        # Reference cloud for the post-hoc evaluation.  The DLO driver ALWAYS
        # augments the skeleton cloud with the two EE positions (unlike BDLO).
        skeleton_pc = result.get('skeleton_pc') if result['success'] else None
        ee_pos = clip_ee_poses[local_idx]
        if skeleton_pc is not None and len(skeleton_pc) > 0:
            if ee_pos is not None and len(ee_pos) > 0:
                ref_clouds.append(np.vstack([skeleton_pc,
                                             np.array(ee_pos, dtype=np.float32).reshape(-1, 3)]))
            else:
                ref_clouds.append(skeleton_pc)
        else:
            ref_clouds.append(np.array(ee_pos, dtype=np.float32).reshape(-1, 3)
                              if ee_pos is not None else np.empty((0, 3)))

    # Save raw + smoothed keypoints
    raw_3d = np.array(keypoints_3d_history)
    smoothed_3d = smooth_trajectories(raw_3d, sigma=sigma)
    edges_arr = np.array(stored_edges) if stored_edges else np.array([])
    ref_lengths_arr = (np.array(stored_reference_lengths)
                       if stored_reference_lengths is not None else np.array([]))
    np.savez(clip_output_dir / '3d_keypoints.npz',
             full=raw_3d, edge_connection=edges_arr, reference_lengths=ref_lengths_arr)
    np.savez(clip_output_dir / 'smoothed_3d_keypoints.npz',
             full=smoothed_3d, sigma=np.array(sigma),
             edge_connection=edges_arr, reference_lengths=ref_lengths_arr)

    # ------------------------------------------------------------------
    # 2. VISUALIZE — smoothed keypoints, single-panel style
    # ------------------------------------------------------------------
    if stored_edges and any(success_flags):
        print(f"    Rendering tracking video (sigma={sigma})...")
        ee_leaves = sorted(int(v) for v in (tracker.ee_to_leaf_mapping or {}).values())
        if len(ee_leaves) >= 2:
            grad_start, grad_end = ee_leaves[0], ee_leaves[-1]
        else:
            grad_start, grad_end = 0, len(raw_3d[0]) - 1
        kp0 = smoothed_3d[int(np.argmax(success_flags))]
        node_t = path_gradient_t(kp0, edges_arr, grad_start, grad_end)
        anchors = set(ee_leaves)  # single DLO: the two chain ends
        masks_clip = (data['masks'][start_frame:end_frame]
                      if data['masks'] is not None else None)
        render_tracking_video(clip_output_dir / 'tracking_dlo.mp4',
                              data['color'][start_frame:end_frame],
                              smoothed_3d, intrinsics, edges_arr, node_t,
                              anchors=anchors, masks=masks_clip,
                              fps=fps, trail=tail_length)

    # ------------------------------------------------------------------
    # 3. EVALUATE — raw keypoints -> summary.txt
    # ------------------------------------------------------------------
    def cd_fn(kp, ref_pc):
        n_ref_points = len(ref_pc) if ref_pc is not None and len(ref_pc) > 0 else 100
        pred_cloud = sample_points_on_edges(kp, stored_edges, n_ref_points)
        return compute_chamfer_metrics(pred_cloud, ref_pc)

    all_metrics_list = evaluate_frames(
        kp_seq=keypoints_3d_history,
        success_seq=success_flags,
        global_indices=list(range(start_frame, end_frame)),
        edges=stored_edges,
        reference_lengths=stored_reference_lengths,
        ref_clouds=ref_clouds,
        edge_fn=compute_edge_metrics,
        pos_fn=lambda kp, ref: compute_position_metrics(kp, ref, extra_gt_points=None),
        cd_fn=cd_fn,
    )

    summary_row = summarize(all_metrics_list, method=method_name, skip_first=True)
    write_clip_summary(
        clip_output_dir / 'summary.txt',
        f"Clip {clip_idx} Summary (frames {start_frame}-{end_frame}, {n_frames} frames)",
        summary_row,
    )

    print(f"    Saved: {clip_output_dir}")

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
    parser = argparse.ArgumentParser(description='Clean DLO tracking experiment (Full method only)')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index (0-19)')
    parser.add_argument('--clip_seconds', type=int, default=10, help='Clip duration in seconds (default: 10)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    parser.add_argument('--n_keypoints', type=int, default=15, help='Number of keypoints (default: 15)')
    parser.add_argument('--tail_length', type=int, default=30,
                        help='Trajectory tail length in the rendered video (frames)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian smoothing sigma applied to 3D trajectories for visualization '
                             '(default: 3.0; metrics are still on raw)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_base = script_dir / 'input_data' / 'dlo'
    calib_dir = data_base / 'calibration'
    output_base = script_dir / 'output' / 'dlo'

    chunk_dir = data_base / f'chunk_{args.chunk}'
    output_dir = output_base / f'chunk_{args.chunk}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DLO TRACKING (FULL METHOD ONLY) - Chunk {args.chunk}")
    print("=" * 80)

    print(f"\nLoading chunk_{args.chunk} data...")
    data = load_chunk_data(chunk_dir)
    transforms = load_transforms(calib_dir)

    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  DLO masks: {data['masks'].shape if data['masks'] is not None else 'None'}")
    print(f"  Total frames: {data['n_frames']}")

    if data['masks'] is None:
        print("ERROR: No DLO masks found!")
        return

    print("\nConverting EE poses to camera frame...")
    ee_poses_3d = np.zeros((data['n_frames'], 2, 3))
    for i in range(data['n_frames']):
        ee_poses_3d[i] = get_ee_positions_cam(
            data['left_poses'][i], data['right_poses'][i],
            transforms['T_left_base2cam'], transforms['T_right_base2cam'],
        )

    frames_per_clip = args.clip_seconds * args.fps
    n_clips = (data['n_frames'] + frames_per_clip - 1) // frames_per_clip  # ceiling division

    print(f"\nClip configuration:")
    print(f"  Clip duration: {args.clip_seconds}s ({frames_per_clip} frames)")
    print(f"  Number of clips: {n_clips}")
    last_clip_frames = data['n_frames'] - (n_clips - 1) * frames_per_clip
    if last_clip_frames < frames_per_clip:
        print(f"  Last clip: {last_clip_frames} frames ({last_clip_frames / args.fps:.1f}s)")

    all_clip_results = []
    for clip_idx in range(n_clips):
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
            n_keypoints=args.n_keypoints,
            tail_length=args.tail_length,
            fps=args.fps,
            sigma=args.sigma,
        )
        all_clip_results.append(clip_result)

    # ==================================================================
    # Chunk aggregate: summary txt + combined keypoint npz
    # ==================================================================
    chunk_summary_dir = output_dir / 'chunk_summary'
    chunk_summary_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    per_clip_summary_rows = []
    for clip_result in all_clip_results:
        pooled_rows.extend(clip_result['all_metrics']['Full'])
        per_clip_summary_rows.extend(clip_result['summary_rows'])

    write_chunk_aggregate(
        chunk_summary_dir / 'chunk_aggregate_summary.txt',
        f"Chunk {args.chunk} Aggregate Summary ({n_clips} clips)",
        pooled_rows, per_clip_summary_rows,
    )

    # Combine 3D keypoints from all clips (raw and smoothed)
    combined_raw, combined_smoothed, combined_reference_lengths = [], [], []
    combined_edges = None
    for clip_result in all_clip_results:
        clip_dir = output_dir / f"clip_{clip_result['clip_idx']}"
        clip_kp_path = clip_dir / '3d_keypoints.npz'
        if clip_kp_path.exists():
            clip_kp = np.load(clip_kp_path)
            combined_raw.append(clip_kp['full'])
            if combined_edges is None and len(clip_kp['edge_connection']) > 0:
                combined_edges = clip_kp['edge_connection']
            if len(clip_kp['reference_lengths']) > 0:
                combined_reference_lengths.append(clip_kp['reference_lengths'])
        smoothed_kp_path = clip_dir / 'smoothed_3d_keypoints.npz'
        if smoothed_kp_path.exists():
            combined_smoothed.append(np.load(smoothed_kp_path)['full'])

    np.savez(
        chunk_summary_dir / 'all_clips_3d_keypoints.npz',
        full=np.concatenate(combined_raw, axis=0) if combined_raw else np.array([]),
        edge_connection=combined_edges if combined_edges is not None else np.array([]),
        reference_lengths_per_clip=np.array(combined_reference_lengths) if combined_reference_lengths else np.array([]),
    )
    np.savez(
        chunk_summary_dir / 'all_clips_smoothed_3d_keypoints.npz',
        full=np.concatenate(combined_smoothed, axis=0) if combined_smoothed else np.array([]),
        sigma=np.array(args.sigma),
        edge_connection=combined_edges if combined_edges is not None else np.array([]),
        reference_lengths_per_clip=np.array(combined_reference_lengths) if combined_reference_lengths else np.array([]),
    )

    print("\n" + "=" * 100)
    print("CHUNK AGGREGATE SUMMARY (see chunk_aggregate_summary.txt)")
    print("=" * 100)
    print_summary_tables(per_clip_summary_rows)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Per-clip: {output_dir}/clip_*/")
    print(f"  Chunk summary: {chunk_summary_dir}/")


if __name__ == "__main__":
    main()
