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
from initialization.fabric_init import FabricInitMixin


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


class FabricTracker(FabricInitMixin):
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
    
    
    # ================================================================
    # CPD REGISTRATION
    # ================================================================
    
    
    # ================================================================
    # GEOMETRY CONSTRAINTS
    # ================================================================
    
    
    # ================================================================
    # TRACKING
    # ================================================================
    
    
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


class FabricTrackerFull(FabricTracker):
    """FabricTracker with the FULL pipeline (snap + geometry + EE constraint, no CPD).

    Corner treatment:
        - Corners are NEVER replaced with EE FK positions (may have calibration errors)
        - EE association is established to know which 2 corners are grasped (EE corners)
        - EE corners: FIXED during optimization
        - Non-EE corners: Constrained to move along contour (like border nodes)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    ) -> np.ndarray:
        """
        Geometry optimization with corner/border constraints.

        - EE-mapped corners are FIXED (2 corners)
        - Non-EE corners are FREE to move and projected to point cloud / contour
        - Border nodes: snapped to contour
        - Interior nodes: soft-projected to point cloud
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8

        if len(point_cloud) == 0:
            return keypoints

        fixed_corners = set(self.ee_corner_indices)

        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)

        contour_nn = None
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)

        for outer_iter in range(self.n_outer_iterations):
            # Edge correction
            for edge_iter in range(self.n_edge_iterations):
                for i, j in self.grid_edges:
                    if i in fixed_corners and j in fixed_corners:
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

                        i_is_fixed = i in fixed_corners
                        j_is_fixed = j in fixed_corners

                        if not i_is_fixed and not j_is_fixed:
                            keypoints[i] += correction * direction
                            keypoints[j] -= correction * direction
                        elif not i_is_fixed:
                            keypoints[i] += 2 * correction * direction
                        elif not j_is_fixed:
                            keypoints[j] -= 2 * correction * direction

            # Project nodes to surfaces
            for i in range(K):
                if i in fixed_corners:
                    continue  # EE corners are fixed

                if i in self.BORDER_INDICES or i in self.CORNER_INDICES:
                    # Border nodes AND non-EE corners: snap to contour
                    if contour_nn is not None:
                        _, idx = contour_nn.kneighbors(keypoints[i:i+1])
                        keypoints[i] = contour_3d[idx[0, 0]]
                    else:
                        _, idx = cloud_nn.kneighbors(keypoints[i:i+1])
                        nearest = point_cloud[idx[0, 0]]
                        alpha = 0.3
                        keypoints[i] = (1 - alpha) * keypoints[i] + alpha * nearest
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

        - EE-associated corners (2 grasped): FIXED during optimization
        - Non-EE corners (2 free): Constrained to move along contour (like borders)
        - Border nodes: Snapped to 3D contour
        - Interior nodes: Soft-projected to point cloud (0.3 blend)
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

        # Extract 3D contour for border constraint
        contour_3d = self._extract_contour_3d(mask, depth)

        # Detect current frame's corners
        corners_2d = self._find_mask_corners(mask, depth)
        corners_3d = self._pixel_to_3d(corners_2d, depth) if corners_2d is not None else None

        # Start from previous keypoints
        keypoints = self.prev_keypoints.copy()

        # Corner snapping (snap corner nodes to detected corners if valid)
        if corners_3d is not None and not np.any(np.isnan(corners_3d)):
            corner_mapping = {
                0: 0,                                        # grid TL -> corners TL
                self.GRID_COLS - 1: 1,                       # grid TR -> corners TR
                self.GRID_ROWS * self.GRID_COLS - 1: 2,      # grid BR -> corners BR
                (self.GRID_ROWS - 1) * self.GRID_COLS: 3,    # grid BL -> corners BL
            }
            for grid_idx, corner_idx in corner_mapping.items():
                if grid_idx < len(keypoints) and corner_idx < len(corners_3d):
                    keypoints[grid_idx] = corners_3d[corner_idx]

        # Snap border nodes to 3D contour
        if len(contour_3d) > 0:
            for idx in self.BORDER_INDICES:
                if idx < len(keypoints):
                    keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Geometry constraint optimization
        t_geom_start = time.time()
        keypoints = self._joint_constraint_optimization_with_contour_full(
            keypoints, point_cloud, contour_3d
        )
        geom_time = time.time() - t_geom_start

        # Final snap: ensure border nodes are still on contour after geometry optimization
        if len(contour_3d) > 0:
            for idx in self.BORDER_INDICES:
                if idx < len(keypoints):
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
                'geom': geom_time,
                'total': track_time,
            },
        }
