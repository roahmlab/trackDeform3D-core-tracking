"""Point-cloud extraction shared by the cloth/fabric drivers."""
import numpy as np


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
