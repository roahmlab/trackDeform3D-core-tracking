"""Unified chunk-data loader for all four tracking drivers.

Per-driver differences are parameters:
    dlo    : load_chunk_data(d)                                    # all frames, masks/masks.npz
    bdlo   : load_chunk_data(d, max_frames=600)                    # last 600
    cloth  : load_chunk_data(d, 'fg_masks/masks.npz', 'masks', max_frames, required=True)
    fabric : load_chunk_data(d, 'fg_mask.npz', 'fg_mask', max_frames, required=True)
Returns dict keys: color, depth, masks, left_poses, right_poses, n_frames.
"""
from pathlib import Path

import numpy as np


def load_chunk_data(chunk_dir: Path, mask_file: str = 'masks/masks.npz',
                    mask_key: str = 'masks', max_frames: int = None,
                    required: bool = False) -> dict:
    print(f"Loading data from {chunk_dir}...")
    rgbd = np.load(chunk_dir / 'rgbd.npz')
    n_total = rgbd['color'].shape[0]
    start_idx = max(0, n_total - max_frames) if max_frames else 0

    color = rgbd['color'][start_idx:]
    depth = rgbd['depth'][start_idx:]

    mask_path = chunk_dir / mask_file
    if mask_path.exists():
        masks = np.load(mask_path)[mask_key][start_idx:]
    elif required:
        raise FileNotFoundError(f"{mask_file} not found in {chunk_dir}")
    else:
        masks = None

    left_poses_npz = np.load(chunk_dir / 'left_arm_poses.npz')
    right_poses_npz = np.load(chunk_dir / 'right_arm_poses.npz')
    n_poses = len(left_poses_npz.files)
    pose_start = max(0, n_poses - max_frames) if max_frames else 0
    n_frames = min(max_frames, n_poses, n_total) if max_frames else n_poses

    left_poses = np.array([left_poses_npz[f'arr_{i}'] for i in range(pose_start, n_poses)])
    right_poses = np.array([right_poses_npz[f'arr_{i}'] for i in range(pose_start, n_poses)])

    print(f"  Loaded {n_frames} frames  color {color.shape}  depth {depth.shape}"
          + (f"  masks {masks.shape}" if masks is not None else "  masks None"))

    return {
        'color': color,
        'depth': depth,
        'masks': masks,
        'left_poses': left_poses,
        'right_poses': right_poses,
        'n_frames': n_frames,
    }
