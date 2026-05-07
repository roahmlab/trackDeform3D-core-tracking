"""
FabricTracker: Fabric tracking with 10x10 grid topology.

Pipeline:
    Phase 1: Segmentation (every frame)
        - Load precomputed mask + depth threshold
    
    Phase 2: Initialization (Frame 0 only)
        - Use EE positions + mask contour to define grid
        - FPS with EE anchors + grid topology
        - Repulsion with grid constraints
    
    Phase 3: CPD Tracking (Frame N > 0)
        - CPD registration + geometry constraints

Grid Layout (10x10 = 100 keypoints):
    Indices:
         0  1  2  3  4  5  6  7  8  9
        10 11 12 13 14 15 16 17 18 19
        20 21 22 23 24 25 26 27 28 29
        30 31 32 33 34 35 36 37 38 39
        40 41 42 43 44 45 46 47 48 49
        50 51 52 53 54 55 56 57 58 59
        60 61 62 63 64 65 66 67 68 69
        70 71 72 73 74 75 76 77 78 79
        80 81 82 83 84 85 86 87 88 89
        90 91 92 93 94 95 96 97 98 99
    
    Corner nodes (degree 2): 0, 9, 90, 99
    Border nodes (degree 3): top (1-8), bottom (91-98), left (10,20,...,80), right (19,29,...,89)
    Interior nodes (degree 4): 8x8 = 64 nodes (11-18, 21-28, ..., 81-88)

Author: Auto-generated
Date: 2026-02-23
"""

import numpy as np
import cv2
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict, Set, Optional


def _build_edge_definitions(grid_rows: int, grid_cols: int) -> list:
    """
    Build edge definitions for contour segmentation.
    
    Args:
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
    
    Returns:
        List of (corner_start_idx, corner_end_idx, [grid_indices for border nodes])
        corner indices refer to corners_3d array: 0=TL, 1=TR, 2=BR, 3=BL
    """
    # Top edge: TL(0) -> TR(1), indices 1 to grid_cols-2
    top_edge = list(range(1, grid_cols - 1))
    
    # Right edge: TR(1) -> BR(2), indices (grid_cols-1) + r*grid_cols for r in 1..grid_rows-2
    right_edge = [r * grid_cols + grid_cols - 1 for r in range(1, grid_rows - 1)]
    
    # Bottom edge: BR(2) -> BL(3), indices (grid_rows-1)*grid_cols + (grid_cols-2) down to 1
    bottom_edge = list(range((grid_rows - 1) * grid_cols + grid_cols - 2, 
                              (grid_rows - 1) * grid_cols, -1))
    
    # Left edge: BL(3) -> TL(0), indices (grid_rows-2)*grid_cols down to grid_cols
    left_edge = [r * grid_cols for r in range(grid_rows - 2, 0, -1)]
    
    return [
        (0, 1, top_edge),     # Top edge: TL -> TR
        (1, 2, right_edge),   # Right edge: TR -> BR
        (2, 3, bottom_edge),  # Bottom edge: BR -> BL
        (3, 0, left_edge),    # Left edge: BL -> TL
    ]


class FabricTracker:
    """
    Fabric tracking with configurable grid topology.
    
    Usage:
        tracker = FabricTracker(intrinsics)
        for frame_idx, (depth, mask) in enumerate(frames):
            result = tracker.process_frame(depth, mask)
            if result['success']:
                keypoints = result['keypoints']
                edges = result['edges']
    """
    
    # Grid constants - CHANGE THESE TO ADJUST GRID SIZE
    GRID_ROWS = 6
    GRID_COLS = 6
    
    def __init__(
        self,
        intrinsics: np.ndarray,
        # Segmentation parameters
        max_depth: float = 1250.0,
        # CPD parameters
        cpd_beta: float = 10.0,
        cpd_lambda: float = 2.0,
        cpd_w: float = 0.1,
        cpd_max_iter: int = 100,
        cpd_tol: float = 1e-3,
        cpd_downsample: int = 1000,
        # Geometry constraint parameters
        n_outer_iterations: int = 5,
        n_edge_iterations: int = 20,
        edge_weight: float = 0.5,
        edge_tolerance: float = 0.15,
        # Repulsion parameters
        repulsion_iterations: int = 500,
        repulsion_lr: float = 5.0,
        # Warm restart
        max_skips_before_restart: int = 3,
        min_foreground_pixels: int = 500,
        # End-effector pose injection
        ee_poses_3d: np.ndarray = None,
    ):
        """
        Initialize FabricTracker.
        
        Args:
            intrinsics: 3×3 camera intrinsic matrix
            max_depth: Maximum valid depth (mm)
            cpd_*: CPD registration parameters
            n_outer_iterations: Edge + projection cycles
            n_edge_iterations: Edge constraint iterations per cycle
            edge_weight: Edge constraint strength [0, 1]
            edge_tolerance: Allowed edge length deviation fraction
            repulsion_iterations: Repulsion relaxation iterations
            repulsion_lr: Repulsion learning rate
            max_skips_before_restart: Consecutive skips before warm restart
            min_foreground_pixels: Minimum foreground pixels to process
            ee_poses_3d: (n_frames, 2, 3) array of EE positions
        """
        # Camera intrinsics
        self.intrinsics = np.array(intrinsics, dtype=np.float64)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        
        # Compute grid-derived constants
        self.N_KEYPOINTS = self.GRID_ROWS * self.GRID_COLS
        
        # Corner indices: TL, TR, BL, BR
        self.CORNER_INDICES = [
            0,                                              # Top-left
            self.GRID_COLS - 1,                             # Top-right
            (self.GRID_ROWS - 1) * self.GRID_COLS,          # Bottom-left
            self.GRID_ROWS * self.GRID_COLS - 1,            # Bottom-right
        ]
        
        # Border indices: all edge nodes except corners
        self.BORDER_INDICES = (
            list(range(1, self.GRID_COLS - 1)) +  # Top edge
            list(range((self.GRID_ROWS - 1) * self.GRID_COLS + 1, 
                       self.GRID_ROWS * self.GRID_COLS - 1)) +  # Bottom edge
            [r * self.GRID_COLS for r in range(1, self.GRID_ROWS - 1)] +  # Left edge
            [r * self.GRID_COLS + self.GRID_COLS - 1 for r in range(1, self.GRID_ROWS - 1)]  # Right edge
        )
        
        # Interior indices: all non-edge nodes
        self.INTERIOR_INDICES = [
            r * self.GRID_COLS + c 
            for r in range(1, self.GRID_ROWS - 1) 
            for c in range(1, self.GRID_COLS - 1)
        ]
        
        # Edge definitions for contour segmentation
        self.EDGE_DEFINITIONS = _build_edge_definitions(self.GRID_ROWS, self.GRID_COLS)
        
        # Segmentation parameters
        self.max_depth = max_depth
        
        # CPD parameters
        self.cpd_beta = cpd_beta
        self.cpd_lambda = cpd_lambda
        self.cpd_w = cpd_w
        self.cpd_max_iter = cpd_max_iter
        self.cpd_tol = cpd_tol
        self.cpd_downsample = cpd_downsample
        
        # Geometry constraint parameters
        self.n_outer_iterations = n_outer_iterations
        self.n_edge_iterations = n_edge_iterations
        self.edge_weight = edge_weight
        self.edge_tolerance = edge_tolerance
        
        # Repulsion parameters
        self.repulsion_iterations = repulsion_iterations
        self.repulsion_lr = repulsion_lr
        
        # Warm restart parameters
        self.max_skips_before_restart = max_skips_before_restart
        self.min_foreground_pixels = min_foreground_pixels
        
        # End-effector pose injection
        self.ee_poses_3d = ee_poses_3d  # (n_frames, 2, 3) or None
        self.ee_to_corner_mapping = None  # {0: corner_idx, 1: corner_idx}
        
        # Build grid topology (static)
        self.grid_edges = self._build_grid_edges()
        self.node_neighbors = self._build_node_neighbors()
        
        # State (set during initialization)
        self.reference_keypoints = None    # N_KEYPOINTS × 3
        self.reference_lengths = None      # Dict of edge -> length
        self.prev_keypoints = None         # N_KEYPOINTS × 3
        self.consecutive_skips = 0
        self.is_initialized = False
        self.frame_count = 0
    
    # ================================================================
    # GRID TOPOLOGY
    # ================================================================
    
    def _build_grid_edges(self) -> List[Tuple[int, int]]:
        """
        Build edges for grid.
        
        Each node connects to its 4-connected neighbors (up, down, left, right).
        
        Returns:
            edges: List of (i, j) tuples where i < j
        """
        edges = []
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                idx = row * self.GRID_COLS + col
                
                # Right neighbor
                if col < self.GRID_COLS - 1:
                    right_idx = idx + 1
                    edges.append((idx, right_idx))
                
                # Down neighbor
                if row < self.GRID_ROWS - 1:
                    down_idx = idx + self.GRID_COLS
                    edges.append((idx, down_idx))
        
        return edges
    
    def _build_node_neighbors(self) -> Dict[int, List[int]]:
        """
        Build neighbor list for each node based on grid topology.
        
        Returns:
            neighbors: Dict mapping node index to list of neighbor indices
        """
        neighbors = {i: [] for i in range(self.N_KEYPOINTS)}
        
        for i, j in self.grid_edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
        
        return neighbors
    
    def _get_node_degree(self, idx: int) -> int:
        """Get expected degree for a node based on its position."""
        if idx in self.CORNER_INDICES:
            return 2
        elif idx in self.BORDER_INDICES:
            return 3
        else:
            return 4
    
    def _idx_to_grid_pos(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) grid position."""
        row = idx // self.GRID_COLS
        col = idx % self.GRID_COLS
        return row, col
    
    def _grid_pos_to_idx(self, row: int, col: int) -> int:
        """Convert (row, col) grid position to linear index."""
        return row * self.GRID_COLS + col
    
    # ================================================================
    # GEOMETRY UTILITIES
    # ================================================================
    
    def _pixel_to_3d(self, pixel_coords: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Back-project 2D pixel coordinates to 3D.
        
        Args:
            pixel_coords: N × 2 array of (row, col)
            depth: H × W depth image
        
        Returns:
            coords_3d: N × 3 array of (x, y, z)
        """
        if len(pixel_coords) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        coords_3d = []
        H, W = depth.shape
        
        for row, col in pixel_coords:
            row, col = int(row), int(col)
            if 0 <= row < H and 0 <= col < W:
                z = depth[row, col]
                if z > 0 and z < self.max_depth:
                    x = (col - self.cx) * z / self.fx
                    y = (row - self.cy) * z / self.fy
                    coords_3d.append([x, y, z])
                else:
                    coords_3d.append([np.nan, np.nan, np.nan])
            else:
                coords_3d.append([np.nan, np.nan, np.nan])
        
        return np.array(coords_3d, dtype=np.float64)
    
    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D pixel coordinates.
        
        Args:
            points_3d: N × 3 array of (x, y, z)
        
        Returns:
            pixel_coords: N × 2 array of (row, col)
        """
        if len(points_3d) == 0:
            return np.empty((0, 2), dtype=np.float64)
        
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        z_safe = np.maximum(z, 1e-6)
        
        u = (x * self.fx) / z_safe + self.cx  # col
        v = (y * self.fy) / z_safe + self.cy  # row
        
        return np.stack([v, u], axis=1)
    
    def _extract_point_cloud(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Extract 3D points from masked region.
        
        Args:
            mask: H × W binary mask
            depth: H × W depth image
        
        Returns:
            points: N × 3 point cloud
        """
        valid = (mask > 0) & (depth > 0) & (depth < self.max_depth)
        rows, cols = np.where(valid)
        
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        z = depth[rows, cols].astype(np.float64)
        x = (cols - self.cx) * z / self.fx
        y = (rows - self.cy) * z / self.fy
        
        return np.stack([x, y, z], axis=1)
    
    def _extract_contour_3d(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray, 
        smooth_window: int = 15,
        corners_3d: np.ndarray = None,
    ) -> np.ndarray:
        """
        Extract 3D contour points from mask boundary.
        
        Args:
            mask: H × W binary mask
            depth: H × W depth image
            smooth_window: Window size for moving average smoothing (odd number)
            corners_3d: (4, 3) or (2, 3) corner positions - if provided, denoises segments
        
        Returns:
            contour_3d: N × 3 contour points in 3D (smoothed and denoised)
        """
        # Create valid mask
        erode_kernel = np.ones((9, 9), np.uint8)  # 5x5 kernel for erosion
        mask_eroded = cv2.erode((mask > 0).astype(np.uint8), erode_kernel, iterations=1)
        valid_mask = (mask_eroded > 0) & (depth > 0) & (depth < self.max_depth)
        
        # Find contours on valid mask
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_2d = largest_contour.reshape(-1, 2)  # (col, row) format
        
        # Convert to (row, col) and lift to 3D
        contour_3d = []
        for col, row in contour_2d:
            if 0 < depth[row, col] < self.max_depth:
                z = float(depth[row, col])
                x = (col - self.cx) * z / self.fx
                y = (row - self.cy) * z / self.fy
                contour_3d.append([x, y, z])
        
        if len(contour_3d) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        contour_3d = np.array(contour_3d, dtype=np.float64)
        
        # Denoise all contour segments if corners are provided
        if corners_3d is not None and len(corners_3d) >= 2:
            contour_3d = self._denoise_all_segments(contour_3d, corners_3d)
        
        # Smooth the 3D contour with circular moving average
        if smooth_window > 1 and len(contour_3d) > smooth_window:
            half_w = smooth_window // 2
            smoothed = np.zeros_like(contour_3d)
            n = len(contour_3d)
            for i in range(n):
                # Circular indices for closed contour
                indices = [(i + j) % n for j in range(-half_w, half_w + 1)]
                smoothed[i] = np.mean(contour_3d[indices], axis=0)
            contour_3d = smoothed
        
        return contour_3d
    
    def _fix_ee_segment(
        self, 
        contour_3d: np.ndarray, 
        ee_positions: np.ndarray,
        z_threshold: float = 20.0,
    ) -> np.ndarray:
        """
        Fix the noisy EE-to-EE contour segment using 25-50-75 percentile reference.
        
        Deprecated: Use _denoise_all_segments instead.
        """
        # Delegate to the general method with just 2 corners
        return self._denoise_all_segments(contour_3d, ee_positions, z_threshold)
    
    def _denoise_all_segments(
        self, 
        contour_3d: np.ndarray, 
        corners_3d: np.ndarray,
        z_threshold: float = 25.0,
    ) -> np.ndarray:
        """
        Denoise all contour segments between corners using 25-50-75 percentile reference.
        
        For each segment between adjacent corners, creates a pseudo-reference using 
        25%, 50%, 75% points and removes noisy points that deviate too far in z.
        
        Args:
            contour_3d: N × 3 contour points
            corners_3d: K × 3 corner positions (2 for EE only, or 4 for all corners)
            z_threshold: Z deviation threshold to detect noisy points (mm)
        
        Returns:
            contour_3d: N × 3 fixed contour (noisy points removed)
        """
        if len(contour_3d) < 10 or len(corners_3d) < 2:
            return contour_3d
        
        # Find contour indices nearest to each corner
        corner_contour_indices = []
        for corner in corners_3d:
            if np.any(np.isnan(corner)):
                continue
            dists = np.linalg.norm(contour_3d - corner, axis=1)
            corner_contour_indices.append(np.argmin(dists))
        
        if len(corner_contour_indices) < 2:
            return contour_3d
        
        # Sort corner indices by their position on contour
        corner_contour_indices = sorted(corner_contour_indices)
        
        # Build segments between consecutive corners (including wrap-around)
        n = len(contour_3d)
        n_corners = len(corner_contour_indices)
        
        # Collect all noisy point indices
        noisy_indices = set()
        total_removed = 0
        
        for seg_idx in range(n_corners):
            idx_start = corner_contour_indices[seg_idx]
            idx_end = corner_contour_indices[(seg_idx + 1) % n_corners]
            
            # Build segment indices
            if idx_start < idx_end:
                segment_indices = list(range(idx_start, idx_end + 1))
            else:
                # Wrap around
                segment_indices = list(range(idx_start, n)) + list(range(0, idx_end + 1))
            
            if len(segment_indices) < 5:
                continue
            
            # Get segment points
            segment_pts = contour_3d[segment_indices].copy()
            n_seg = len(segment_pts)
            
            # Find 25%, 50%, 75% reference points
            idx_25 = n_seg // 4
            idx_50 = n_seg // 2
            idx_75 = (3 * n_seg) // 4
            
            ref_pts = np.array([
                segment_pts[0],           # 0% (start)
                segment_pts[idx_25],      # 25%
                segment_pts[idx_50],      # 50% (midpoint)
                segment_pts[idx_75],      # 75%
                segment_pts[-1],          # 100% (end)
            ])
            ref_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            
            # Interpolate pseudo-reference for each point in segment
            t_vals = np.linspace(0, 1, n_seg)
            pseudo_ref = np.zeros_like(segment_pts)
            for dim in range(3):
                pseudo_ref[:, dim] = np.interp(t_vals, ref_t, ref_pts[:, dim])
            
            # Find noisy points: z-deviation from pseudo-reference
            z_deviation = np.abs(segment_pts[:, 2] - pseudo_ref[:, 2])
            noisy_mask = z_deviation > z_threshold
            
            n_noisy = np.sum(noisy_mask)
            if n_noisy > 0:
                # Mark noisy segment points for removal
                for i, idx in enumerate(segment_indices):
                    if noisy_mask[i]:
                        noisy_indices.add(idx)
                total_removed += n_noisy
        
        if total_removed > 0:
            print(f"  [Contour] Denoised all segments: removed {total_removed} noisy pts (>{z_threshold}mm z-dev)")
            
            # Build mask for keeping points
            keep_mask = np.ones(len(contour_3d), dtype=bool)
            for idx in noisy_indices:
                keep_mask[idx] = False
            
            contour_3d = contour_3d[keep_mask]
            print(f"  [Contour] {len(contour_3d)} pts remaining")
        
        return contour_3d
    
    def _extract_contour_3d_raw(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Extract raw 3D contour points (without denoising or smoothing).
        
        Args:
            mask: H × W binary mask
            depth: H × W depth image
        
        Returns:
            contour_3d: N × 3 raw contour points
        """
        valid_mask = (mask > 0) & (depth > 0) & (depth < self.max_depth)
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_2d = largest_contour.reshape(-1, 2)
        
        contour_3d = []
        for col, row in contour_2d:
            if 0 < depth[row, col] < self.max_depth:
                z = float(depth[row, col])
                x = (col - self.cx) * z / self.fx
                y = (row - self.cy) * z / self.fy
                contour_3d.append([x, y, z])
        
        return np.array(contour_3d, dtype=np.float64) if contour_3d else np.empty((0, 3), dtype=np.float64)
    
    def _compute_contour_segment_lengths(
        self, 
        contour_3d: np.ndarray, 
        corners_3d: np.ndarray
    ) -> dict:
        """
        Compute the arc-length of each contour segment between corners.
        
        Args:
            contour_3d: N × 3 contour points
            corners_3d: K × 3 corner positions
        
        Returns:
            segment_lengths: Dict mapping segment name to arc-length (mm)
        """
        if len(contour_3d) < 10 or len(corners_3d) < 2:
            return {}
        
        # Find contour indices nearest to each corner
        corner_contour_indices = []
        corner_labels = ['TL', 'TR', 'BR', 'BL']  # corners_3d order from _find_mask_corners
        valid_labels = []
        for i, corner in enumerate(corners_3d):
            if np.any(np.isnan(corner)):
                continue
            dists = np.linalg.norm(contour_3d - corner, axis=1)
            corner_contour_indices.append((np.argmin(dists), corner_labels[i] if i < len(corner_labels) else f'C{i}'))
            valid_labels.append(corner_labels[i] if i < len(corner_labels) else f'C{i}')
        
        if len(corner_contour_indices) < 2:
            return {}
        
        # Sort by contour index
        corner_contour_indices = sorted(corner_contour_indices, key=lambda x: x[0])
        
        n = len(contour_3d)
        n_corners = len(corner_contour_indices)
        segment_lengths = {}
        
        for seg_idx in range(n_corners):
            idx_start, label_start = corner_contour_indices[seg_idx]
            idx_end, label_end = corner_contour_indices[(seg_idx + 1) % n_corners]
            
            # Build segment indices
            if idx_start < idx_end:
                segment_indices = list(range(idx_start, idx_end + 1))
            else:
                segment_indices = list(range(idx_start, n)) + list(range(0, idx_end + 1))
            
            if len(segment_indices) < 2:
                continue
            
            # Compute arc-length
            segment_pts = contour_3d[segment_indices]
            arc_length = np.sum(np.linalg.norm(np.diff(segment_pts, axis=0), axis=1))
            
            seg_name = f'{label_start}-{label_end}'
            segment_lengths[seg_name] = arc_length
        
        # Print segment lengths
        total_len = sum(segment_lengths.values())
        print(f"  [Contour] Segment lengths: {', '.join([f'{k}:{v:.0f}mm' for k, v in segment_lengths.items()])} | Total: {total_len:.0f}mm")
        
        return segment_lengths
    
    def _farthest_point_sampling(
        self, 
        points: np.ndarray, 
        n_samples: int, 
        seed_points: np.ndarray = None
    ) -> np.ndarray:
        """
        Farthest Point Sampling with optional seed points as anchors.
        
        Args:
            points: N × 3 points to sample from
            n_samples: Number of samples to return
            seed_points: Optional K × 3 anchor points (included in distance but not returned)
        
        Returns:
            sampled: n_samples × 3 sampled points
        """
        N = len(points)
        if N == 0:
            return np.array([]).reshape(0, 3)
        
        if n_samples >= N:
            return points.copy()
        
        if seed_points is not None and len(seed_points) > 0:
            distances = np.full(N, np.inf)
            for seed in seed_points:
                dist_to_seed = np.linalg.norm(points - seed, axis=1)
                distances = np.minimum(distances, dist_to_seed)
        else:
            distances = np.full(N, np.inf)
            first_idx = np.random.randint(N)
            distances = np.linalg.norm(points - points[first_idx], axis=1)
        
        sampled_indices = []
        for _ in range(n_samples):
            farthest_idx = np.argmax(distances)
            sampled_indices.append(farthest_idx)
            dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
            distances = np.minimum(distances, dist_to_new)
        
        return points[sampled_indices]
    
    def _arc_length_sample(self, segment: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample points at uniform arc-length intervals along a contour segment.
        
        This ensures evenly spaced border nodes (unlike FPS which maximizes min-distance).
        
        Args:
            segment: M × 3 contour segment points (ordered)
            n_samples: Number of interior samples to return (excludes endpoints)
        
        Returns:
            sampled: n_samples × 3 uniformly spaced points
        """
        if len(segment) < 2:
            return segment.copy()
        
        # Compute cumulative arc length
        diffs = np.diff(segment, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length < 1e-6:
            # Degenerate case: all points same
            return np.tile(segment[0], (n_samples, 1))
        
        # Target arc-lengths for uniformly spaced samples (exclude endpoints)
        # n_samples interior points divide the segment into (n_samples + 1) intervals
        target_lengths = np.linspace(0, total_length, n_samples + 2)[1:-1]
        
        # Interpolate to find 3D positions at target arc-lengths
        sampled_points = []
        for target in target_lengths:
            # Find which segment the target falls in
            idx = np.searchsorted(cumulative_length, target, side='right') - 1
            idx = np.clip(idx, 0, len(segment) - 2)
            
            # Interpolate within that segment
            local_dist = target - cumulative_length[idx]
            seg_length = edge_lengths[idx] if idx < len(edge_lengths) else 1e-6
            t = local_dist / max(seg_length, 1e-6)
            t = np.clip(t, 0, 1)
            
            point = (1 - t) * segment[idx] + t * segment[idx + 1]
            sampled_points.append(point)
        
        return np.array(sampled_points, dtype=np.float64)
    
    def _snap_to_contour_3d(self, point_3d: np.ndarray, contour_3d: np.ndarray) -> np.ndarray:
        """
        Snap a 3D point to the nearest point on the 3D contour.
        
        Args:
            point_3d: 3D point (x, y, z)
            contour_3d: N × 3 contour points
        
        Returns:
            snapped: 3D point snapped to contour
        """
        if len(contour_3d) == 0:
            return point_3d
        
        distances = np.linalg.norm(contour_3d - point_3d, axis=1)
        nearest_idx = np.argmin(distances)
        return contour_3d[nearest_idx].copy()

    # ================================================================
    # INITIALIZATION
    # ================================================================
    
    def _find_mask_corners(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Find the 4 corners of the valid mask (mask AND valid depth).
        
        Args:
            mask: H × W binary mask
            depth: H × W depth image
        
        Returns:
            corners: 4 × 2 array of (row, col) for corners
                     Order: top-left, top-right, bottom-right, bottom-left
        """
        # Create valid mask: mask pixels with valid depth
        valid_mask = (mask > 0) & (depth > 0) & (depth < self.max_depth)
        
        # Get all valid pixel coordinates
        rows, cols = np.where(valid_mask)
        
        if len(rows) == 0:
            return None
        
        # Find corners using extreme points on valid mask
        # Top-left: minimize (row + col)
        top_left_scores = rows + cols
        top_left_idx = np.argmin(top_left_scores)
        top_left = np.array([rows[top_left_idx], cols[top_left_idx]])
        
        # Top-right: minimize (row - col) = min row, max col
        top_right_scores = rows - cols
        top_right_idx = np.argmin(top_right_scores)
        top_right = np.array([rows[top_right_idx], cols[top_right_idx]])
        
        # Bottom-right: maximize (row + col)
        bottom_right_idx = np.argmax(top_left_scores)
        bottom_right = np.array([rows[bottom_right_idx], cols[bottom_right_idx]])
        
        # Bottom-left: maximize (row - col) = max row, min col
        bottom_left_idx = np.argmax(top_right_scores)
        bottom_left = np.array([rows[bottom_left_idx], cols[bottom_left_idx]])
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
        
        return corners
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Sort corners in order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: 4 × 2 array of (row, col)
        
        Returns:
            sorted_corners: 4 × 2 array in correct order
        """
        # Center point
        center = corners.mean(axis=0)
        
        # Compute angles from center
        angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
        
        # Sort by angle (top-left has smallest angle when measured correctly)
        # Actually, let's use sum and difference method
        
        # Sum of row + col: smallest = top-left, largest = bottom-right
        # Diff of col - row: smallest = bottom-left, largest = top-right
        
        sums = corners[:, 0] + corners[:, 1]
        diffs = corners[:, 1] - corners[:, 0]
        
        top_left_idx = np.argmin(sums)
        bottom_right_idx = np.argmax(sums)
        top_right_idx = np.argmax(diffs)
        bottom_left_idx = np.argmin(diffs)
        
        return np.array([
            corners[top_left_idx],
            corners[top_right_idx],
            corners[bottom_right_idx],
            corners[bottom_left_idx],
        ])
    
    def _initialize_grid_from_corners(
        self, 
        corners_3d: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray = None,
    ) -> np.ndarray:
        """
        Initialize grid keypoints using Approach 4:
        - Corners: Use detected corners (on contour)
        - Border nodes: FPS on contour segments (guaranteed ON contour)
        - Interior nodes: Bilinear interpolation snapped to point cloud
        
        Args:
            corners_3d: 4 × 3 corner positions (top-left, top-right, bottom-right, bottom-left)
            point_cloud: N × 3 foreground point cloud
            contour_3d: M × 3 contour points (required for Approach 4)
        
        Returns:
            keypoints: N_KEYPOINTS × 3 grid keypoints
        """
        keypoints = np.zeros((self.N_KEYPOINTS, 3), dtype=np.float64)
        
        # Step 1: Place corners using computed indices
        # CORNER_INDICES = [TL, TR, BL, BR]
        # corners_3d order from _find_mask_corners: [TL, TR, BR, BL]
        keypoints[self.CORNER_INDICES[0]] = corners_3d[0]   # TL
        keypoints[self.CORNER_INDICES[1]] = corners_3d[1]   # TR
        keypoints[self.CORNER_INDICES[3]] = corners_3d[2]   # BR
        keypoints[self.CORNER_INDICES[2]] = corners_3d[3]   # BL
        
        print("  [Init] Step 1: Corners placed")
        
        # If no contour, fall back to bilinear interpolation
        if contour_3d is None or len(contour_3d) < 12:
            print("  [Init] No contour, using bilinear interpolation")
            return self._initialize_grid_bilinear(corners_3d, point_cloud)
        
        # Step 2: Find corner positions on contour
        nn_contour = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn_contour.fit(contour_3d)
        _, corner_contour_indices = nn_contour.kneighbors(corners_3d)
        corner_contour_indices = corner_contour_indices.flatten()
        
        print(f"  [Init] Step 2: Corner indices on contour: {corner_contour_indices}")
        
        # Step 3: FPS on each contour segment for border nodes
        n_contour = len(contour_3d)
        n_border_per_edge = self.GRID_COLS - 2
        
        print("  [Init] Step 3: FPS on contour segments for border nodes")
        for edge_id, (c_start, c_end, grid_indices) in enumerate(self.EDGE_DEFINITIONS):
            idx_start = corner_contour_indices[c_start]
            idx_end = corner_contour_indices[c_end]
            
            # Choose shorter path around contour
            if idx_start <= idx_end:
                forward_len = idx_end - idx_start + 1
                backward_len = n_contour - idx_end + idx_start + 1
            else:
                forward_len = n_contour - idx_start + idx_end + 1
                backward_len = idx_start - idx_end + 1
            
            if forward_len <= backward_len:
                if idx_start <= idx_end:
                    segment = contour_3d[idx_start:idx_end+1]
                else:
                    segment = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end+1]])
            else:
                if idx_start >= idx_end:
                    segment = contour_3d[idx_end:idx_start+1][::-1]
                else:
                    segment = np.vstack([contour_3d[idx_end:], contour_3d[:idx_start+1]])[::-1]
            
            print(f"    Edge {edge_id}: {len(segment)} pts, need {n_border_per_edge} border nodes")
            
            # Get corner positions for this edge
            corner_start = corners_3d[c_start]
            corner_end = corners_3d[c_end]
            
            if len(segment) >= n_border_per_edge + 2:
                # FPS with corners as seed points (anchors)
                fps_points = self._farthest_point_sampling(
                    segment, n_border_per_edge, 
                    seed_points=np.array([corner_start, corner_end])
                )
                
                # Order FPS results by distance from start corner
                dists_from_start = np.linalg.norm(fps_points - corner_start, axis=1)
                fps_points = fps_points[np.argsort(dists_from_start)]
                
                for i, idx in enumerate(grid_indices):
                    keypoints[idx] = fps_points[i]
            else:
                # Linear interpolation fallback
                n_segments = len(grid_indices) + 1
                for i, idx in enumerate(grid_indices):
                    t = (i + 1) / n_segments
                    keypoints[idx] = (1 - t) * corner_start + t * corner_end
        
        # Validate: check for uninitialized nodes (zeros)
        uninitialized = []
        for idx in range(self.N_KEYPOINTS):
            if np.allclose(keypoints[idx], 0.0):
                uninitialized.append(idx)
        if uninitialized:
            print(f"  [Init] WARNING: {len(uninitialized)} uninitialized nodes: {uninitialized}")
            # Fallback: use bilinear interpolation for uninitialized nodes
            for idx in uninitialized:
                row, col = self._idx_to_grid_pos(idx)
                u = col / (self.GRID_COLS - 1)
                v = row / (self.GRID_ROWS - 1)
                tl = self.CORNER_INDICES[0]
                tr = self.CORNER_INDICES[1]
                bl = self.CORNER_INDICES[2]
                br = self.CORNER_INDICES[3]
                top = (1 - u) * keypoints[tl] + u * keypoints[tr]
                bottom = (1 - u) * keypoints[bl] + u * keypoints[br]
                keypoints[idx] = (1 - v) * top + v * bottom
        
        # Step 4: Interior nodes - bilinear + snap to point cloud
        print("  [Init] Step 4: Bilinear interior + snap to point cloud")
        if len(point_cloud) > 0:
            nn_cloud = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nn_cloud.fit(point_cloud)
            
            for idx in self.INTERIOR_INDICES:
                row, col = self._idx_to_grid_pos(idx)
                u = col / (self.GRID_COLS - 1)
                v = row / (self.GRID_ROWS - 1)
                
                # Use computed corner indices: TL, TR, BL, BR
                tl = self.CORNER_INDICES[0]
                tr = self.CORNER_INDICES[1]
                bl = self.CORNER_INDICES[2]
                br = self.CORNER_INDICES[3]
                top = (1 - u) * keypoints[tl] + u * keypoints[tr]
                bottom = (1 - u) * keypoints[bl] + u * keypoints[br]
                expected = (1 - v) * top + v * bottom
                
                _, nearest_idx = nn_cloud.kneighbors(expected.reshape(1, -1))
                keypoints[idx] = point_cloud[nearest_idx[0, 0]]
        
        return keypoints
    
    def _initialize_grid_bilinear(
        self, 
        corners_3d: np.ndarray,
        point_cloud: np.ndarray,
    ) -> np.ndarray:
        """
        Fallback: Initialize grid using bilinear interpolation.
        
        Args:
            corners_3d: 4 × 3 corner positions (top-left, top-right, bottom-right, bottom-left)
            point_cloud: N × 3 foreground point cloud
        
        Returns:
            keypoints: N_KEYPOINTS × 3 grid keypoints
        """
        top_left, top_right, bottom_right, bottom_left = corners_3d
        
        keypoints = np.zeros((self.N_KEYPOINTS, 3), dtype=np.float64)
        
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                u = col / (self.GRID_COLS - 1)
                v = row / (self.GRID_ROWS - 1)
                
                top = (1 - u) * top_left + u * top_right
                bottom = (1 - u) * bottom_left + u * bottom_right
                point = (1 - v) * top + v * bottom
                
                idx = self._grid_pos_to_idx(row, col)
                keypoints[idx] = point
        
        # Snap keypoints to point cloud
        if len(point_cloud) > 0:
            keypoints = self._snap_to_point_cloud(keypoints, point_cloud)
        
        return keypoints
    
    def _snap_to_point_cloud(
        self, 
        keypoints: np.ndarray, 
        point_cloud: np.ndarray,
        max_distance: float = 50.0,
    ) -> np.ndarray:
        """
        Snap keypoints to nearest points in point cloud.
        
        Args:
            keypoints: K × 3 keypoints
            point_cloud: N × 3 point cloud
            max_distance: Maximum snap distance (mm)
        
        Returns:
            snapped: K × 3 snapped keypoints
        """
        if len(point_cloud) == 0:
            return keypoints.copy()
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(point_cloud)
        
        distances, indices = nn.kneighbors(keypoints)
        distances = distances.flatten()
        indices = indices.flatten()
        
        snapped = keypoints.copy()
        for i in range(len(keypoints)):
            if distances[i] < max_distance:
                snapped[i] = point_cloud[indices[i]]
        
        return snapped
    
    def _get_border_direction(self, idx: int, keypoints: np.ndarray) -> np.ndarray:
        """
        Get the direction along which a border node can move.
        
        Border nodes can only move along the edge of the grid.
        
        Args:
            idx: Node index
            keypoints: Current keypoints
        
        Returns:
            direction: Unit vector along the border, or None if not a border node
        """
        row, col = self._idx_to_grid_pos(idx)
        
        # Top border (row=0, col=1,2,3): direction is along row 0
        if row == 0 and col > 0 and col < self.GRID_COLS - 1:
            left_idx = self._grid_pos_to_idx(0, col - 1)
            right_idx = self._grid_pos_to_idx(0, col + 1)
            direction = keypoints[right_idx] - keypoints[left_idx]
        # Bottom border (row=4, col=1,2,3): direction is along row 4
        elif row == self.GRID_ROWS - 1 and col > 0 and col < self.GRID_COLS - 1:
            left_idx = self._grid_pos_to_idx(self.GRID_ROWS - 1, col - 1)
            right_idx = self._grid_pos_to_idx(self.GRID_ROWS - 1, col + 1)
            direction = keypoints[right_idx] - keypoints[left_idx]
        # Left border (col=0, row=1,2,3): direction is along col 0
        elif col == 0 and row > 0 and row < self.GRID_ROWS - 1:
            up_idx = self._grid_pos_to_idx(row - 1, 0)
            down_idx = self._grid_pos_to_idx(row + 1, 0)
            direction = keypoints[down_idx] - keypoints[up_idx]
        # Right border (col=4, row=1,2,3): direction is along col 4
        elif col == self.GRID_COLS - 1 and row > 0 and row < self.GRID_ROWS - 1:
            up_idx = self._grid_pos_to_idx(row - 1, self.GRID_COLS - 1)
            down_idx = self._grid_pos_to_idx(row + 1, self.GRID_COLS - 1)
            direction = keypoints[down_idx] - keypoints[up_idx]
        else:
            return None
        
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        return direction / norm
    
    def _print_edge_stats(self, keypoints: np.ndarray, label: str = "") -> None:
        """Print edge length statistics."""
        if keypoints is None or len(keypoints) == 0:
            return
        
        edge_lengths = []
        for i, j in self.grid_edges:
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            edge_lengths.append(length)
        
        edge_lengths = np.array(edge_lengths)
        avg = np.mean(edge_lengths)
        
        # Compute % deviation from average
        pct_errors = np.abs(edge_lengths - avg) / avg * 100
        
        print(f"  [Init] Edge stats ({label}):")
        print(f"    Avg length: {avg:.2f} mm")
        print(f"    Min/Max: {edge_lengths.min():.2f} / {edge_lengths.max():.2f} mm")
        print(f"    Std: {np.std(edge_lengths):.2f} mm ({np.std(edge_lengths)/avg*100:.1f}%)")
        print(f"    Error from avg: mean={np.mean(pct_errors):.1f}%, max={np.max(pct_errors):.1f}%")
    
    def _repulsion_relaxation_grid(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray = None,
    ) -> np.ndarray:
        """
        Spring-based relaxation with grid topology and proper constraints.
        
        Constraints:
        - Corner nodes (0, 4, 20, 24): Fixed, no movement
        - Border nodes: Can only move along the 3D contour (if provided)
        - Interior nodes: Free to move in 3D
        
        Args:
            keypoints: 25 × 3 keypoints
            point_cloud: N × 3 points to project onto
            contour_3d: M × 3 contour points (optional, for border constraints)
        
        Returns:
            relaxed: 25 × 3 relaxed keypoints
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8
        
        if K <= 1 or len(point_cloud) == 0:
            return keypoints
        
        # Build NN index for projection
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)
        
        # Build NN index for contour if provided
        contour_nn = None
        contour_length = 0.0
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)
            # Compute total contour arc-length (including closing edge for closed contour)
            contour_length = np.sum(np.linalg.norm(np.diff(contour_3d, axis=0), axis=1))
            # Add closing edge if contour is a closed loop
            closing_dist = np.linalg.norm(contour_3d[-1] - contour_3d[0])
            if closing_dist < 100:  # Only add if reasonable (< 100mm)
                contour_length += closing_dist
        
        # Compute target edge length from contour
        # Border edges = 4 sides × (grid_size - 1) edges per side
        n_border_edges = 4 * (self.GRID_COLS - 1)  # e.g., 4 * 5 = 20 for 6x6 grid
        
        if contour_length > epsilon:
            target_length = contour_length / n_border_edges
            print(f"  [Repulsion] Target edge length from contour: {target_length:.1f}mm (contour={contour_length:.0f}mm / {n_border_edges} edges)", flush=True)
        else:
            # Fallback to mean of initial edges
            edge_lengths = []
            for i, j in self.grid_edges:
                length = np.linalg.norm(keypoints[i] - keypoints[j])
                if length > epsilon:
                    edge_lengths.append(length)
            
            if len(edge_lengths) == 0:
                return keypoints
            
            target_length = np.mean(edge_lengths)
            print(f"  [Repulsion] Target edge length from mean: {target_length:.1f}mm (no contour)", flush=True)
        
        # Learning rate from parameter
        lr = self.repulsion_lr / 25.0  # Scale: repulsion_lr=5.0 -> lr=0.2
        
        print(f"  [Repulsion] Running {self.repulsion_iterations} iterations, lr={lr:.3f}", flush=True)
        
        # Relaxation iterations
        prev_std = float('inf')
        for iteration in range(self.repulsion_iterations):
            # Compute spring forces
            forces = np.zeros_like(keypoints)
            
            for i, j in self.grid_edges:
                vec = keypoints[j] - keypoints[i]
                current_length = np.linalg.norm(vec)
                
                if current_length < epsilon:
                    continue
                
                # Spring force: pull if stretched, push if compressed
                direction = vec / current_length
                force_magnitude = (current_length - target_length)
                force = force_magnitude * direction
                
                forces[i] += force
                forces[j] -= force
            
            # Apply forces with constraints
            for i in range(K):
                # Corner nodes: completely fixed
                if i in self.CORNER_INDICES:
                    continue
                
                # Border nodes: move along contour (if available) or grid border direction
                elif i in self.BORDER_INDICES:
                    if contour_nn is not None:
                        # Apply force then snap to contour
                        keypoints[i] += lr * forces[i]
                        # Snap to nearest contour point
                        _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = contour_3d[idx[0, 0]].copy()
                    else:
                        # Fall back to border direction constraint
                        border_dir = self._get_border_direction(i, keypoints)
                        if border_dir is not None:
                            force_along_border = np.dot(forces[i], border_dir) * border_dir
                            keypoints[i] += lr * force_along_border
                
                # Interior nodes: free movement
                else:
                    keypoints[i] += lr * forces[i]
            
            # Check convergence every 50 iterations
            if iteration % 50 == 49:
                edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in self.grid_edges]
                curr_std = np.std(edge_lengths)
                curr_mean = np.mean(edge_lengths)
                print(f"  [Repulsion] iter {iteration+1}: mean={curr_mean:.1f}mm (target={target_length:.1f}mm), std={curr_std:.1f}mm", flush=True)
                if abs(prev_std - curr_std) < 0.1:  # Converged
                    print(f"  [Repulsion] Converged at iteration {iteration+1}", flush=True)
                    break
                prev_std = curr_std
        
        # After repulsion, project interior nodes to point cloud (soft)
        for i in range(K):
            if i in self.CORNER_INDICES or i in self.BORDER_INDICES:
                continue
            _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
            nearest = point_cloud[idx[0, 0]]
            alpha = 0.3  # Soft projection at the end
            keypoints[i] = (1 - alpha) * keypoints[i] + alpha * nearest
        
        # Final summary
        final_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in self.grid_edges]
        print(f"  [Repulsion] DONE: mean={np.mean(final_lengths):.1f}mm, std={np.std(final_lengths):.1f}mm, min={np.min(final_lengths):.1f}mm, max={np.max(final_lengths):.1f}mm", flush=True)
        
        return keypoints
    
    def _establish_ee_to_corner_mapping(
        self, 
        keypoints: np.ndarray, 
        frame_idx: int
    ) -> None:
        """
        Establish mapping from EE indices to corner keypoint indices.
        
        EE0 (top-left of mask) should map to corner index 0 (top-left of grid)
        EE1 (top-right of mask) should map to corner index 4 (top-right of grid)
        
        Args:
            keypoints: 25 × 3 keypoints
            frame_idx: Current frame index
        """
        if self.ee_poses_3d is None:
            return
        
        if frame_idx >= len(self.ee_poses_3d):
            return
        
        ee_positions = self.ee_poses_3d[frame_idx]  # (2, 3)
        
        # Get corner keypoints (indices 0, 4, 20, 24)
        corner_positions = keypoints[self.CORNER_INDICES]  # (4, 3)
        
        # Match EE to corners using Hungarian algorithm
        cost_matrix = cdist(ee_positions, corner_positions)
        ee_indices, corner_local_indices = linear_sum_assignment(cost_matrix)
        
        self.ee_to_corner_mapping = {}
        for ee_idx, corner_local_idx in zip(ee_indices, corner_local_indices):
            corner_global_idx = self.CORNER_INDICES[corner_local_idx]
            self.ee_to_corner_mapping[ee_idx] = corner_global_idx
        
        print(f"  EE to corner mapping: {self.ee_to_corner_mapping}")
    
    def _replace_with_ee_poses(
        self, 
        keypoints: np.ndarray, 
        frame_idx: int
    ) -> np.ndarray:
        """
        Replace corner keypoints with EE poses.
        
        Args:
            keypoints: 25 × 3 keypoints
            frame_idx: Current frame index
        
        Returns:
            keypoints: 25 × 3 with corners replaced by EE poses
        """
        if self.ee_poses_3d is None or self.ee_to_corner_mapping is None:
            return keypoints
        
        if frame_idx >= len(self.ee_poses_3d):
            return keypoints
        
        ee_positions = self.ee_poses_3d[frame_idx]  # (2, 3)
        keypoints = keypoints.copy()
        
        for ee_idx, corner_idx in self.ee_to_corner_mapping.items():
            if not np.any(np.isnan(ee_positions[ee_idx])):
                keypoints[corner_idx] = ee_positions[ee_idx]
        
        return keypoints
    
    def initialize(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray,
        frame_idx: int = 0,
    ) -> dict:
        """
        Phase 2: Initialize keypoints and topology from first frame.
        
        Args:
            mask: H × W binary mask (already depth-thresholded)
            depth: H × W depth image
            frame_idx: Frame index (for EE poses)
        
        Returns:
            dict with initialization results
        """
        t_start = time.time()
        
        # Extract point cloud
        point_cloud = self._extract_point_cloud(mask, depth)
        
        if len(point_cloud) < self.min_foreground_pixels:
            return {
                'success': False,
                'reason': 'insufficient_points',
                'mode': 'init',
            }
        
        # Find mask corners (on valid depth-filtered mask)
        corners_2d = self._find_mask_corners(mask, depth)
        if corners_2d is None:
            return {
                'success': False,
                'reason': 'no_corners_found',
                'mode': 'init',
            }
        
        # Back-project corners to 3D (guaranteed valid depth since we found them on valid mask)
        corners_3d = self._pixel_to_3d(corners_2d, depth)
        
        # Check for valid corners
        if np.any(np.isnan(corners_3d)):
            # Try to snap invalid corners to point cloud
            for i in range(4):
                if np.any(np.isnan(corners_3d[i])):
                    # Find nearest point in point cloud
                    row, col = int(corners_2d[i, 0]), int(corners_2d[i, 1])
                    H, W = mask.shape
                    
                    # Search in neighborhood
                    search_radius = 20
                    best_dist = float('inf')
                    best_point = None
                    
                    for dr in range(-search_radius, search_radius + 1):
                        for dc in range(-search_radius, search_radius + 1):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if mask[nr, nc] > 0 and 0 < depth[nr, nc] < self.max_depth:
                                    dist = np.sqrt(dr**2 + dc**2)
                                    if dist < best_dist:
                                        best_dist = dist
                                        z = depth[nr, nc]
                                        x = (nc - self.cx) * z / self.fx
                                        y = (nr - self.cy) * z / self.fy
                                        best_point = np.array([x, y, z])
                    
                    if best_point is not None:
                        corners_3d[i] = best_point
        
        if np.any(np.isnan(corners_3d)):
            return {
                'success': False,
                'reason': 'invalid_corner_depth',
                'mode': 'init',
            }
        
        # If we have EE poses, snap them to nearest detected contour corner
        # This corrects for calibration errors in EE FK
        if self.ee_poses_3d is not None and frame_idx < len(self.ee_poses_3d):
            ee_positions = self.ee_poses_3d[frame_idx]
            # corners_3d order: [TL, TR, BR, BL]
            # Find nearest contour corner for each EE
            for ee_idx in range(2):
                if not np.any(np.isnan(ee_positions[ee_idx])):
                    # Find nearest among detected corners
                    dists = np.linalg.norm(corners_3d - ee_positions[ee_idx], axis=1)
                    nearest_corner_idx = np.argmin(dists)
                    print(f"  [Init] EE{ee_idx} snapped to corner {nearest_corner_idx} (dist={dists[nearest_corner_idx]:.1f}mm)")
                    # Use detected corner position, not EE FK position
                    # But remember which corner this EE maps to
                    if ee_idx == 0:
                        # EE0 should map to TL (0) or BL (3)
                        if nearest_corner_idx in [0, 3]:
                            corners_3d[nearest_corner_idx] = corners_3d[nearest_corner_idx]  # Keep detected
                        else:
                            print(f"    WARNING: EE0 matched to unexpected corner {nearest_corner_idx}")
                    else:
                        # EE1 should map to TR (1) or BR (2)
                        if nearest_corner_idx in [1, 2]:
                            corners_3d[nearest_corner_idx] = corners_3d[nearest_corner_idx]  # Keep detected
                        else:
                            print(f"    WARNING: EE1 matched to unexpected corner {nearest_corner_idx}")
        
        # Extract 3D contour for border constraints (needed for Approach 4 initialization)
        # Pass all 4 corners to denoise all segments
        contour_3d = self._extract_contour_3d(mask, depth, corners_3d=corners_3d)
        print(f"  [Init] Extracted 3D contour with {len(contour_3d)} points")
        
        # Initialize grid from corners using Approach 4:
        # - Border nodes: FPS on contour segments
        # - Interior nodes: Bilinear + snap to point cloud
        keypoints = self._initialize_grid_from_corners(corners_3d, point_cloud, contour_3d)
        
        # Print edge stats BEFORE repulsion
        self._print_edge_stats(keypoints, "Before repulsion")
        
        # Repulsion relaxation with grid topology
        # - Corner nodes: fixed
        # - Border nodes: constrained to move along 3D contour
        # - Interior nodes: free to move
        keypoints = self._repulsion_relaxation_grid(keypoints, point_cloud, contour_3d)
        
        # Print edge stats AFTER repulsion
        self._print_edge_stats(keypoints, "After repulsion")
        
        # Establish EE to corner mapping (for tracking, not for overriding init corners)
        self._establish_ee_to_corner_mapping(keypoints, frame_idx)
        
        # NOTE: Don't replace with EE poses in init - use detected contour corners instead
        # The EE FK may be off due to calibration errors
        # keypoints = self._replace_with_ee_poses(keypoints, frame_idx)
        
        # Compute reference edge lengths
        self.reference_lengths = {}
        for i, j in self.grid_edges:
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            self.reference_lengths[(i, j)] = length
        
        # Store state
        self.reference_keypoints = keypoints.copy()
        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True
        self.frame_count = 1
        self.consecutive_skips = 0
        
        init_time = time.time() - t_start
        
        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        return {
            'success': True,
            'mode': 'init',
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.grid_edges,
            'corners_3d': corners_3d,
            'timing': {'init': init_time},
        }
    
    # ================================================================
    # CPD REGISTRATION
    # ================================================================
    
    def _cpd_register(self, Y: np.ndarray, X: np.ndarray) -> tuple:
        """
        Coherent Point Drift registration.
        
        Args:
            Y: M × 3 source points (previous keypoints)
            X: N × 3 target points (current point cloud)
        
        Returns:
            T: M × 3 transformed source points
            P: M × N correspondence matrix
        """
        M, D = Y.shape
        N = X.shape[0]
        
        if N == 0:
            return Y.copy(), np.zeros((M, 1))
        
        # Downsample target if needed
        if N > self.cpd_downsample:
            indices = np.random.choice(N, self.cpd_downsample, replace=False)
            X = X[indices]
            N = self.cpd_downsample
        
        # Initialize
        sigma2 = np.sum((X[None, :, :] - Y[:, None, :]) ** 2) / (M * N * D)
        T = Y.copy()
        
        # Construct G matrix for motion coherence
        G = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * self.cpd_beta ** 2))
        
        for iteration in range(self.cpd_max_iter):
            # E-step: compute P
            dist2 = cdist(T, X, 'sqeuclidean')
            c = (2 * np.pi * sigma2) ** (D / 2) * self.cpd_w / (1 - self.cpd_w) * M / N
            
            P = np.exp(-dist2 / (2 * sigma2))
            den = P.sum(axis=0, keepdims=True) + c + 1e-10
            P = P / den
            
            # M-step: update T
            P1 = P.sum(axis=1)
            PX = P @ X
            
            # Solve for W: (G + lambda * sigma2 * diag(1/P1)) W = (PX - diag(P1) Y) / sigma2
            diag_P1_inv = np.diag(1.0 / (P1 + 1e-10))
            A = G + self.cpd_lambda * sigma2 * diag_P1_inv
            B = diag_P1_inv @ PX - Y
            
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            
            T_new = Y + G @ W
            
            # Update sigma2
            diff = X[None, :, :] - T_new[:, None, :]
            sigma2_new = np.sum(P * np.sum(diff ** 2, axis=2)) / (np.sum(P) * D + 1e-10)
            sigma2_new = max(sigma2_new, 1e-6)
            
            # Check convergence
            change = np.max(np.abs(T_new - T))
            T = T_new
            sigma2 = sigma2_new
            
            if change < self.cpd_tol:
                break
        
        return T, P
    
    # ================================================================
    # GEOMETRY CONSTRAINTS
    # ================================================================
    
    def _joint_constraint_optimization(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
    ) -> np.ndarray:
        """
        Joint edge length + surface projection optimization.
        
        Args:
            keypoints: 25 × 3 keypoints
            point_cloud: N × 3 target points
        
        Returns:
            optimized: 25 × 3 optimized keypoints
        """
        return self._joint_constraint_optimization_with_contour(keypoints, point_cloud, None)
    
    def _joint_constraint_optimization_with_contour(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Joint edge length + surface projection optimization with contour constraint.
        
        Constraints:
        - Corner nodes: Fixed
        - Border nodes: Can only move along the 3D contour
        - Interior nodes: Free to move, projected to point cloud
        
        Args:
            keypoints: 25 × 3 keypoints
            point_cloud: N × 3 target points
            contour_3d: N × 3 contour points (or None)
        
        Returns:
            optimized: 25 × 3 optimized keypoints
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8
        
        if len(point_cloud) == 0:
            return keypoints
        
        # Build NN for projection
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)
        
        # Build NN for contour if available
        contour_nn = None
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)
        
        for outer_iter in range(self.n_outer_iterations):
            # Edge correction
            for edge_iter in range(self.n_edge_iterations):
                for i, j in self.grid_edges:
                    # Skip if both are corners (fixed)
                    if i in self.CORNER_INDICES and j in self.CORNER_INDICES:
                        continue
                    
                    target_length = self.reference_lengths.get((i, j), 0)
                    if target_length < epsilon:
                        continue
                    
                    current_vec = keypoints[j] - keypoints[i]
                    current_length = np.linalg.norm(current_vec)
                    
                    if current_length < epsilon:
                        continue
                    
                    error = (current_length - target_length) / target_length
                    
                    if abs(error) > self.edge_tolerance:
                        direction = current_vec / current_length
                        correction = (current_length - target_length) * self.edge_weight / 2
                        
                        # Apply correction based on node type
                        i_is_corner = i in self.CORNER_INDICES
                        j_is_corner = j in self.CORNER_INDICES
                        
                        if not i_is_corner and not j_is_corner:
                            keypoints[i] += correction * direction
                            keypoints[j] -= correction * direction
                        elif not i_is_corner:
                            keypoints[i] += 2 * correction * direction
                        elif not j_is_corner:
                            keypoints[j] -= 2 * correction * direction
            
            # Snap border nodes to contour, interior nodes to point cloud
            for i in range(K):
                if i in self.CORNER_INDICES:
                    continue  # Corners are fixed
                
                if i in self.BORDER_INDICES:
                    # Border nodes: snap to contour
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
    
    # ================================================================
    # TRACKING
    # ================================================================
    
    def track(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        frame_idx: int,
    ) -> dict:
        """
        Phase 3: Track keypoints using CPD + geometry constraints.
        
        After CPD:
        1. Snap corner nodes to detected corners from current mask
        2. Snap border nodes to the 3D contour
        3. Apply geometry constraints
        
        Args:
            mask: H × W binary mask (already depth-thresholded)
            depth: H × W depth image
            frame_idx: Current frame index
        
        Returns:
            dict with tracking results
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
        
        # Detect current frame's corners first (for contour denoising)
        corners_2d = self._find_mask_corners(mask, depth)
        corners_3d = self._pixel_to_3d(corners_2d, depth) if corners_2d is not None else None
        
        # Extract 3D contour for border constraint
        # Pass all 4 corners to denoise all segments
        contour_3d = self._extract_contour_3d(mask, depth, corners_3d=corners_3d)
        
        # CPD registration
        t_cpd_start = time.time()
        cpd_keypoints, _ = self._cpd_register(self.prev_keypoints, point_cloud)
        cpd_time = time.time() - t_cpd_start
        
        keypoints = cpd_keypoints.copy()
        
        # Snap corner nodes to detected corners (if valid)
        if corners_3d is not None and not np.any(np.isnan(corners_3d)):
            # corners_3d order: top-left, top-right, bottom-right, bottom-left
            # CORNER_INDICES order: [TL, TR, BL, BR]
            corner_mapping = {
                self.CORNER_INDICES[0]: 0,   # grid TL -> corners TL
                self.CORNER_INDICES[1]: 1,   # grid TR -> corners TR
                self.CORNER_INDICES[3]: 2,   # grid BR -> corners BR
                self.CORNER_INDICES[2]: 3,   # grid BL -> corners BL
            }
            for grid_idx, corner_idx in corner_mapping.items():
                keypoints[grid_idx] = corners_3d[corner_idx]
        
        # Snap border nodes to 3D contour
        if len(contour_3d) > 0:
            for idx in self.BORDER_INDICES:
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)
        
        # Geometry constraint optimization
        t_geom_start = time.time()
        keypoints = self._joint_constraint_optimization_with_contour(
            keypoints, point_cloud, contour_3d
        )
        geom_time = time.time() - t_geom_start
        
        # NOTE: Don't replace with EE poses - use detected contour corners instead
        # The EE FK may be off due to calibration errors
        # keypoints = self._replace_with_ee_poses(keypoints, frame_idx)
        
        # Final snap: ensure border nodes are still on contour after geometry optimization
        if len(contour_3d) > 0:
            for idx in self.BORDER_INDICES:
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
            'edges': self.grid_edges,
            'edge_errors': edge_errors,
            'timing': {
                'cpd': cpd_time,
                'geom': geom_time,
                'total': track_time,
            },
        }
    
    def _compute_edge_errors(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute relative edge length errors."""
        errors = []
        for i, j in self.grid_edges:
            current = np.linalg.norm(keypoints[i] - keypoints[j])
            target = self.reference_lengths.get((i, j), current)
            if target > 1e-8:
                errors.append(abs(current - target) / target)
        return np.array(errors)
    
    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    
    def process_frame(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        frame_idx: int = None,
    ) -> dict:
        """
        Process a single frame.
        
        Args:
            depth: H × W depth image
            mask: H × W binary mask (raw, will be depth-thresholded)
            frame_idx: Frame index (for EE poses)
        
        Returns:
            dict with processing results
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        
        # # Shrink mask by 2 pixels to avoid noisy depth at edges
        # kernel = np.ones((15, 15), np.uint8)  # 5x5 kernel for ~2 pixel erosion
        # mask_eroded = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
        mask_eroded = mask.copy()  # No erosion for now, since we rely on contour detection for border constraints
        
        # Apply depth threshold to eroded mask
        # IMPORTANT: Only keep pixels with valid depth (0 < depth < max_depth)
        valid_depth = (depth > 0) & (depth < self.max_depth)
        mask_filtered = (mask_eroded > 0) & valid_depth
        
        # Check foreground pixels
        n_foreground = np.sum(mask_filtered > 0)
        if n_foreground < self.min_foreground_pixels:
            self.consecutive_skips += 1
            return {
                'success': False,
                'reason': 'insufficient_foreground',
                'mode': 'skip',
                'foreground_mask': mask_filtered,
            }
        
        # Initialize or track
        if not self.is_initialized:
            result = self.initialize(mask_filtered, depth, frame_idx)
        else:
            # Check for warm restart
            if self.consecutive_skips >= self.max_skips_before_restart:
                print(f"  Warm restart after {self.consecutive_skips} skips")
                result = self.initialize(mask_filtered, depth, frame_idx)
                result['mode'] = 'restart'
            else:
                result = self.track(mask_filtered, depth, frame_idx)
        
        result['foreground_mask'] = mask_filtered
        return result
    
    def get_state(self) -> dict:
        """Get current tracker state."""
        return {
            'is_initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'consecutive_skips': self.consecutive_skips,
            'reference_keypoints': self.reference_keypoints,
            'prev_keypoints': self.prev_keypoints,
            'grid_edges': self.grid_edges,
            'ee_to_corner_mapping': self.ee_to_corner_mapping,
        }
