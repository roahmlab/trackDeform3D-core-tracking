"""
Cloth Tracker using Rectangle-Aligned Grid Initialization.

Based on FabricTracker but uses max-inscribed rectangle orientation for grid alignment.
This produces better grid placement for arbitrarily-oriented cloth.

Key differences from FabricTracker:
- Uses find_max_inscribed_rectangle_rotated to get optimal rectangle orientation
- Grid is aligned to the bounding rectangle with that orientation (not axis-aligned mask corners)
- Default grid size is 8×8 (vs 6×6 for fabric)

Author: Auto-generated
Date: 2025-02-28
"""

import numpy as np
import cv2
import time
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from initialization.cloth_init import ClothInitMixin


class ClothTracker(ClothInitMixin):
    """
    Grid-based cloth tracker with rectangle-aligned initialization.
    
    Tracks a rectangular piece of cloth using a configurable N×N grid topology.
    The grid is initialized by fitting a bounding rectangle aligned with the
    max-inscribed rectangle orientation (not axis-aligned).
    
    Grid indices (8×8 example, row-major):
        0   1   2   3   4   5   6   7      (row 0: top)
        8   9  10  11  12  13  14  15
       16  17  18  19  20  21  22  23
       24  25  26  27  28  29  30  31
       32  33  34  35  36  37  38  39
       40  41  42  43  44  45  46  47
       48  49  50  51  52  53  54  55
       56  57  58  59  60  61  62  63      (row 7: bottom)
    
    Corner indices: [0, 7, 56, 63] = [TL, TR, BL, BR]
    """
    
    # Default grid configuration (can be overridden)
    GRID_ROWS = 8
    GRID_COLS = 8
    
    def __init__(
        self,
        intrinsics: np.ndarray,
        max_depth: float = 2000.0,
        min_foreground_pixels: int = 500,
        # Geometry constraint parameters
        n_outer_iterations: int = 3,
        n_edge_iterations: int = 10,
        edge_weight: float = 0.5,
        edge_tolerance: float = 0.01,
        # Repulsion relaxation parameters
        repulsion_iterations: int = 500,
        repulsion_lr: float = 5.0,
        # Restart threshold
        max_skips_before_restart: int = 5,
        # EE poses (optional)
        ee_poses_3d: Optional[np.ndarray] = None,
        # Grid configuration
        grid_rows: int = None,
        # Contour corner detection (for T-shirt, use 8)
        n_contour_corners: int = 8,
        grid_cols: int = None,
        # Manual segment interior node counts (e.g., [7,1,1,5,3,5,1,1] for T-shirt)
        segment_interior_nodes: list = None,
        # Segment directions for T-shape border walk (e.g., [(0,1),(1,0),(0,-1),(1,0),(0,-1),(-1,0),(0,-1),(-1,0)])
        segment_directions: list = None,
    ):
        """
        Initialize ClothTracker.
        
        Args:
            intrinsics: 3×3 camera intrinsic matrix
            max_depth: Maximum valid depth (mm)
            min_foreground_pixels: Minimum foreground pixels required
            n_outer_iterations: Number of geometry constraint outer loops
            n_edge_iterations: Number of edge correction iterations per outer loop
            edge_weight: Edge correction weight (0-1)
            edge_tolerance: Edge error tolerance for correction
            repulsion_iterations: Number of repulsion relaxation iterations
            repulsion_lr: Repulsion learning rate
            max_skips_before_restart: Consecutive skips before warm restart
            ee_poses_3d: End-effector poses (N, 2, 3) for corner anchoring
            grid_rows: Number of grid rows (default: 8)
            grid_cols: Number of grid columns (default: 8)
        """
        self.intrinsics = np.array(intrinsics, dtype=np.float64)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        
        self.max_depth = max_depth
        self.min_foreground_pixels = min_foreground_pixels
        
        # Geometry constraint parameters
        self.n_outer_iterations = n_outer_iterations
        self.n_edge_iterations = n_edge_iterations
        self.edge_weight = edge_weight
        self.edge_tolerance = edge_tolerance
        
        # Repulsion relaxation parameters
        self.repulsion_iterations = repulsion_iterations
        self.repulsion_lr = repulsion_lr
        
        # Restart threshold
        self.max_skips_before_restart = max_skips_before_restart
        
        # EE poses
        self.ee_poses_3d = ee_poses_3d
        
        # Grid configuration
        if grid_rows is not None:
            self.GRID_ROWS = grid_rows
        if grid_cols is not None:
            self.GRID_COLS = grid_cols
        
        # Computed grid constants
        self.N_KEYPOINTS = self.GRID_ROWS * self.GRID_COLS
        
        # Corner indices: [TL, TR, BL, BR]
        self.CORNER_INDICES = [
            0,                                          # Top-left
            self.GRID_COLS - 1,                         # Top-right
            (self.GRID_ROWS - 1) * self.GRID_COLS,      # Bottom-left
            self.GRID_ROWS * self.GRID_COLS - 1,        # Bottom-right
        ]
        
        # Border indices (excluding corners)
        self.BORDER_INDICES = (
            list(range(1, self.GRID_COLS - 1)) +  # Top edge
            list(range((self.GRID_ROWS - 1) * self.GRID_COLS + 1, self.GRID_ROWS * self.GRID_COLS - 1)) +  # Bottom
            [r * self.GRID_COLS for r in range(1, self.GRID_ROWS - 1)] +  # Left edge
            [r * self.GRID_COLS + self.GRID_COLS - 1 for r in range(1, self.GRID_ROWS - 1)]  # Right
        )
        
        # Interior indices (all nodes not on border or corners)
        border_and_corner = set(self.CORNER_INDICES + self.BORDER_INDICES)
        self.INTERIOR_INDICES = [i for i in range(self.N_KEYPOINTS) if i not in border_and_corner]
        
        # Build grid edges
        self.grid_edges = self._build_grid_edges()
        
        # Build node neighbors dictionary
        self.node_neighbors = self._build_node_neighbors()
        
        # Valid edges/neighbors (same as grid until T-cropping removes some)
        self.valid_edges = self.grid_edges.copy()
        self.valid_neighbors = {k: v.copy() for k, v in self.node_neighbors.items()}
        self.valid_faces = []  # List of (tl, tr, br, bl) quad tuples
        
        # State
        self.is_initialized = False
        self.reference_keypoints = None
        self.prev_keypoints = None
        self.reference_lengths = {}
        self.ee_to_corner_mapping = None
        self.rect_corners_3d = None  # Bounding rectangle corners [TL, TR, BR, BL]
        self.detected_corners_3d = None  # Real contour corners (8 for T-shirt)
        self.border_grid_indices = None  # Ordered border grid indices from sequential chain
        self.n_contour_corners = n_contour_corners  # Number of corners to detect on real contour
        self.segment_interior_nodes = segment_interior_nodes  # Manual interior node counts per segment
        self.segment_directions = segment_directions  # Direction (dr, dc) per segment for shape border walk
        self.contour_corner_grid_indices = None  # Grid indices of all N contour corners
        self.fixed_indices = set()  # Indices fixed during relaxation (contour corners)
        self.frame_count = 0
        self.consecutive_skips = 0
    
    def _build_grid_edges(self) -> list:
        """Build grid edges (horizontal and vertical connections)."""
        edges = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                idx = r * self.GRID_COLS + c
                # Horizontal edge (right)
                if c < self.GRID_COLS - 1:
                    edges.append((idx, idx + 1))
                # Vertical edge (down)
                if r < self.GRID_ROWS - 1:
                    edges.append((idx, idx + self.GRID_COLS))
        return edges
    
    def _rebuild_valid_edges(self, keypoints: np.ndarray) -> None:
        """
        Rebuild edge list and neighbors to exclude edges connecting to NaN nodes.
        
        After T-cropping, some nodes become NaN. Edges to/from these nodes are invalid.
        This method filters the edge list and rebuilds the neighbor dictionary.
        
        Args:
            keypoints: N_KEYPOINTS × 3 array with NaN for invalid nodes
        """
        # Find valid (non-NaN) nodes
        valid_mask = ~np.isnan(keypoints[:, 0])
        valid_indices = set(np.where(valid_mask)[0])
        
        # Filter edges: keep only if BOTH endpoints are valid
        self.valid_edges = [
            (i, j) for (i, j) in self.grid_edges
            if i in valid_indices and j in valid_indices
        ]
        
        # Rebuild neighbor dictionary based on valid edges only
        self.valid_neighbors = {i: [] for i in range(self.N_KEYPOINTS)}
        for i, j in self.valid_edges:
            self.valid_neighbors[i].append(j)
            self.valid_neighbors[j].append(i)
        
        n_original = len(self.grid_edges)
        n_valid = len(self.valid_edges)
        print(f"  [Init] Edge filtering: {n_valid}/{n_original} edges valid (T-topology)")
    
    def _rebuild_valid_faces(self, keypoints: np.ndarray, n_target_faces: int = None) -> None:
        """
        Build quad faces from grid cells, keep only the largest n_target_faces.

        Each grid cell (r, c) defines a quad: TL, TR, BR, BL.
        Only quads where all 4 vertices are valid (non-NaN) are considered.
        Faces are sorted by area descending and truncated to n_target_faces.

        Args:
            keypoints: N_KEYPOINTS × 3 array with NaN for invalid nodes
            n_target_faces: Keep this many largest faces (None = keep all valid)
        """
        valid_mask = ~np.isnan(keypoints[:, 0])
        faces_with_area = []

        for r in range(self.GRID_ROWS - 1):
            for c in range(self.GRID_COLS - 1):
                tl = r * self.GRID_COLS + c
                tr = r * self.GRID_COLS + c + 1
                bl = (r + 1) * self.GRID_COLS + c
                br = (r + 1) * self.GRID_COLS + c + 1

                if not (valid_mask[tl] and valid_mask[tr] and valid_mask[bl] and valid_mask[br]):
                    continue

                # Quad area via cross product of diagonals
                diag1 = keypoints[br] - keypoints[tl]
                diag2 = keypoints[tr] - keypoints[bl]
                area = 0.5 * np.linalg.norm(np.cross(diag1, diag2))
                faces_with_area.append(((tl, tr, br, bl), area))

        # Sort by area descending, keep largest n_target_faces
        faces_with_area.sort(key=lambda x: x[1], reverse=True)
        if n_target_faces is not None and len(faces_with_area) > n_target_faces:
            faces_with_area = faces_with_area[:n_target_faces]

        self.valid_faces = [f for f, _ in faces_with_area]
        print(f"  [Init] Faces: {len(self.valid_faces)} valid quad faces"
              f"{f' (kept top {n_target_faces})' if n_target_faces else ''}")

    def _classify_nodes_by_topology(
        self,
        keypoints: np.ndarray,
        contour_3d: np.ndarray,
        detected_corners_3d: np.ndarray,
    ) -> None:
        """
        Classify valid nodes by neighbor count after face/edge rebuild.

        - 2 neighbors → corner
        - 3 neighbors → border
        - 4 neighbors → interior (unless matched to a detected contour corner → "special corner")

        Sets:
            self.contour_corner_grid_indices  (list of grid indices)
            self.border_grid_indices          (list of grid indices)
            self.fixed_indices                (set: corners only)
        """
        valid_mask = ~np.isnan(keypoints[:, 0])
        valid_indices = np.where(valid_mask)[0]

        corners_2n = []   # 2-neighbor corners
        borders_3n = []   # 3-neighbor borders
        interior_4n = []  # 4-neighbor interior

        for idx in valid_indices:
            n_neighbors = len(self.valid_neighbors[idx])
            if n_neighbors <= 2:
                corners_2n.append(int(idx))
            elif n_neighbors == 3:
                borders_3n.append(int(idx))
            else:
                interior_4n.append(int(idx))

        # Match detected contour corners to 57 vertices via NN
        # Detected corners that match a 4-neighbor vertex → "special corner"
        special_corners = []
        if detected_corners_3d is not None and len(detected_corners_3d) > 0:
            valid_positions = keypoints[valid_indices]  # 57 × 3
            for corner_3d in detected_corners_3d:
                dists = np.linalg.norm(valid_positions - corner_3d, axis=1)
                nearest_valid_local = np.argmin(dists)
                nearest_grid_idx = int(valid_indices[nearest_valid_local])
                if nearest_grid_idx in interior_4n:
                    special_corners.append(nearest_grid_idx)
                    interior_4n.remove(nearest_grid_idx)

        all_corners = corners_2n + special_corners
        self.contour_corner_grid_indices = all_corners
        self.border_grid_indices = borders_3n
        self.fixed_indices = set(all_corners)

        print(f"  [Classify] Corners (2-neighbor): {corners_2n}")
        print(f"  [Classify] Special corners (4-neighbor, matched): {special_corners}")
        print(f"  [Classify] Border (3-neighbor): {len(borders_3n)} nodes — {borders_3n}")
        print(f"  [Classify] Interior (4-neighbor): {len(interior_4n)} nodes")


    def _build_node_neighbors(self) -> dict:
        """Build dictionary mapping node index to list of neighbor indices."""
        neighbors = {i: [] for i in range(self.N_KEYPOINTS)}
        for i, j in self.grid_edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
        return neighbors
    
    def _grid_pos_to_idx(self, row: int, col: int) -> int:
        """Convert grid position to linear index."""
        return row * self.GRID_COLS + col
    
    def _idx_to_grid_pos(self, idx: int) -> tuple:
        """Convert linear index to grid position (row, col)."""
        return (idx // self.GRID_COLS, idx % self.GRID_COLS)
    
    # ================================================================
    # GEOMETRY UTILITIES
    # ================================================================
    
    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D pixel coordinates (col, row) = (x, y)."""
        if points_3d is None or len(points_3d) == 0:
            return np.array([])

        points_2d = []
        for x, y, z in points_3d:
            if z > 0:
                col = self.fx * x / z + self.cx
                row = self.fy * y / z + self.cy
                points_2d.append([col, row])
            else:
                points_2d.append([np.nan, np.nan])

        return np.array(points_2d, dtype=np.float64)
    
    def _extract_point_cloud(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Extract 3D point cloud from masked depth."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        
        z_vals = depth[rows, cols].astype(np.float64)
        valid = (z_vals > 0) & (z_vals < self.max_depth)
        rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
        
        if len(z_vals) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        
        x_vals = (cols - self.cx) * z_vals / self.fx
        y_vals = (rows - self.cy) * z_vals / self.fy
        
        return np.column_stack([x_vals, y_vals, z_vals])
    
    def _pixel_to_3d(self, pixels_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Convert 2D pixel coordinates to 3D points using depth.
        
        Args:
            pixels_2d: (N, 2) array of [row, col] pixel coordinates
            depth: (H, W) depth image
            
        Returns:
            (N, 3) array of 3D points [x, y, z]
        """
        if pixels_2d is None or len(pixels_2d) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        points_3d = []
        for row, col in pixels_2d:
            row_i, col_i = int(round(row)), int(round(col))
            if 0 <= row_i < depth.shape[0] and 0 <= col_i < depth.shape[1]:
                z = float(depth[row_i, col_i])
                if 0 < z < self.max_depth:
                    x = (col_i - self.cx) * z / self.fx
                    y = (row_i - self.cy) * z / self.fy
                    points_3d.append([x, y, z])
                else:
                    points_3d.append([np.nan, np.nan, np.nan])
            else:
                points_3d.append([np.nan, np.nan, np.nan])
        
        return np.array(points_3d, dtype=np.float64)
    
    def _find_mask_corners(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Find corners of the mask using min-area bounding rectangle.
        
        Args:
            mask: Binary mask
            depth: Depth image (for validation)
            
        Returns:
            (4, 2) array of corner pixel coordinates [row, col], ordered [TL, TR, BR, BL]
        """
        valid_mask = (mask > 0) & (depth > 0) & (depth < self.max_depth)
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)  # (4, 2) in [col, row] format
        
        # Sort corners: find top-left, top-right, bottom-right, bottom-left
        # Convert to [row, col] format  
        box_rc = box[:, ::-1]  # Convert [col, row] to [row, col]
        
        # Order by angle from centroid
        center = box_rc.mean(axis=0)
        angles = np.arctan2(box_rc[:, 0] - center[0], box_rc[:, 1] - center[1])
        sorted_indices = np.argsort(angles)
        box_rc = box_rc[sorted_indices]
        
        # Reorder to [TL, TR, BR, BL] based on row/col positions
        # Top-left has smallest row+col sum
        sums = box_rc[:, 0] + box_rc[:, 1]
        tl_idx = np.argmin(sums)
        ordered = np.roll(box_rc, -tl_idx, axis=0)
        
        return ordered.astype(np.float64)
    
    def _extract_contour_3d_raw(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray,
    ) -> np.ndarray:
        """Extract raw 3D contour without denoising (for visualization).
        
        Args:
            mask: Binary mask
            depth: Depth image
            
        Returns:
            (N, 3) array of 3D contour points
        """
        # Smooth mask slightly
        kernel = np.ones((3, 3), np.uint8)
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        valid_mask = (mask_smooth > 0) & (depth > 0) & (depth < self.max_depth)
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
        
        if len(contour_3d) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        return np.array(contour_3d, dtype=np.float64)
    
    def _extract_contour_3d(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray,
        corners_3d: np.ndarray = None,
        z_threshold: float = 25.0,
    ) -> np.ndarray:
        """
        Extract 3D contour with denoising.
        
        Using a smoothed contour via morphological operations.
        """
        # Smooth mask (use same 5x5 kernel as corner detection for consistency)
        kernel = np.ones((5, 5), np.uint8)
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        valid_mask = (mask_smooth > 0) & (depth > 0) & (depth < self.max_depth)
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
        
        if len(contour_3d) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        contour_3d = np.array(contour_3d, dtype=np.float64)
        
        # Denoise: remove points with large z deviation (same as FabricTracker)
        if corners_3d is not None and len(corners_3d) >= 2:
            contour_3d = self._denoise_contour(contour_3d, corners_3d, z_threshold)
        
        return contour_3d
    
    def _denoise_contour(
        self, 
        contour_3d: np.ndarray, 
        corners_3d: np.ndarray,
        z_threshold: float = 25.0,
    ) -> np.ndarray:
        """
        Denoise contour by removing points with large z deviation.
        
        Same algorithm as FabricTracker._denoise_all_segments.
        """
        if len(contour_3d) < 20 or len(corners_3d) < 2:
            return contour_3d
        
        # Find contour indices nearest to each corner
        corner_contour_indices = []
        for corner in corners_3d:
            if not np.any(np.isnan(corner)):
                dists = np.linalg.norm(contour_3d - corner, axis=1)
                corner_contour_indices.append(np.argmin(dists))
        
        if len(corner_contour_indices) < 2:
            return contour_3d
        
        # Sort corner indices by their position on contour (MUST sort!)
        corner_contour_indices = sorted(corner_contour_indices)
        
        # Build segments between consecutive corners (including wrap-around)
        n = len(contour_3d)
        n_corners = len(corner_contour_indices)
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
            
            # Interpolate pseudo-reference for each point
            t_vals = np.linspace(0, 1, n_seg)
            pseudo_ref = np.zeros_like(segment_pts)
            for dim in range(3):
                pseudo_ref[:, dim] = np.interp(t_vals, ref_t, ref_pts[:, dim])
            
            # Find noisy points: z-deviation from pseudo-reference
            z_deviation = np.abs(segment_pts[:, 2] - pseudo_ref[:, 2])
            noisy_mask = z_deviation > z_threshold
            
            n_noisy = np.sum(noisy_mask)
            if n_noisy > 0:
                for i, idx in enumerate(segment_indices):
                    if noisy_mask[i]:
                        noisy_indices.add(idx)
                total_removed += n_noisy
        
        if total_removed > 0:
            print(f"  [Contour] Denoised: removed {total_removed} noisy pts (>{z_threshold}mm z-dev)")
            keep_mask = np.ones(len(contour_3d), dtype=bool)
            for idx in noisy_indices:
                keep_mask[idx] = False
            contour_3d = contour_3d[keep_mask]
            print(f"  [Contour] {len(contour_3d)} pts remaining")
        
        return contour_3d
    
    def _snap_to_contour_3d(self, point_3d: np.ndarray, contour_3d: np.ndarray) -> np.ndarray:
        """Snap a 3D point to the nearest point on the 3D contour."""
        if len(contour_3d) == 0:
            return point_3d

        distances = np.linalg.norm(contour_3d - point_3d, axis=1)
        nearest_idx = np.argmin(distances)
        return contour_3d[nearest_idx].copy()


    def _is_point_inside_mask(self, point_3d: np.ndarray, mask: np.ndarray) -> bool:
        """
        Check if a 3D point projects inside the mask.
        
        Args:
            point_3d: [x, y, z] 3D point
            mask: H × W binary mask
            
        Returns:
            True if point projects inside mask, False otherwise
        """
        if np.any(np.isnan(point_3d)):
            return False
        
        x, y, z = point_3d
        if z <= 0:
            return False
        
        # Project to pixel
        col = int(round(self.fx * x / z + self.cx))
        row = int(round(self.fy * y / z + self.cy))
        
        H, W = mask.shape
        if 0 <= row < H and 0 <= col < W:
            return mask[row, col] > 0
        
        return False
    
    
    def _detect_contour_corners(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray,
        n_corners: int = None,
    ) -> tuple:
        """
        Detect corners on the real smoothed contour using approxPolyDP.
        
        For T-shirt: typically 8 corners (4 outer + 4 inner at waist).
        
        Args:
            mask: Binary mask
            depth: Depth image
            n_corners: Target number of corners (default: self.n_contour_corners)
            
        Returns:
            corners_2d: (N, 2) array of corner pixel coordinates [col, row]
            corners_3d: (N, 3) array of corner 3D coordinates
        """
        if n_corners is None:
            n_corners = self.n_contour_corners
        
        # Smooth mask to get cleaner contour
        kernel = np.ones((5, 5), np.uint8)
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        valid_mask = (mask_smooth > 0) & (depth > 0) & (depth < self.max_depth)
        
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        
        # Binary search for epsilon to get desired number of corners
        eps_low, eps_high = 0.0001, 0.15
        best_corners = None
        best_diff = float('inf')

        for _ in range(40):
            eps_mid = (eps_low + eps_high) / 2
            approx = cv2.approxPolyDP(largest_contour, eps_mid * peri, True)
            n_found = len(approx)

            diff = abs(n_found - n_corners)
            if diff < best_diff or (diff == best_diff and n_found >= n_corners):
                best_diff = diff
                best_corners = approx.squeeze()

            if n_found == n_corners:
                break
            elif n_found > n_corners:
                eps_low = eps_mid
            else:
                eps_high = eps_mid

        if best_corners is None or len(best_corners) < 3:
            return None, None

        corners_2d = best_corners  # Keep all — Hungarian matching will select the correct ones
        
        # Convert to 3D
        H, W = depth.shape
        avg_depth = np.mean(depth[(depth > 0) & (depth < self.max_depth)])
        
        corners_3d = []
        corners_2d_valid = []
        for col, row in corners_2d:
            row_i, col_i = int(np.clip(row, 0, H-1)), int(np.clip(col, 0, W-1))
            z = depth[row_i, col_i]
            if z <= 0 or z >= self.max_depth:
                z = avg_depth
            x = (col - self.cx) * z / self.fx
            y = (row - self.cy) * z / self.fy
            corners_3d.append([x, y, z])
            corners_2d_valid.append([col, row])
        
        return np.array(corners_2d_valid), np.array(corners_3d)

    # ================================================================
    # INITIALIZATION
    # ================================================================
    

    def _get_grid_border_clockwise(self) -> list:
        """
        Get grid border indices in clockwise order starting from TL (index 0).
        
        For 9×9 grid:
        TL(0) → 1 → 2 → ... → 8(TR) → 17 → 26 → ... → 80(BR) → 79 → ... → 72(BL) → 63 → ... → 9 → 0
        
        Returns:
            List of grid indices for border nodes in clockwise order
        """
        border = []
        
        # Top edge: left to right (row 0, col 0 to COLS-1)
        for col in range(self.GRID_COLS):
            border.append(self._grid_pos_to_idx(0, col))
        
        # Right edge: top to bottom (row 1 to ROWS-1, col COLS-1)
        for row in range(1, self.GRID_ROWS):
            border.append(self._grid_pos_to_idx(row, self.GRID_COLS - 1))
        
        # Bottom edge: right to left (row ROWS-1, col COLS-2 to 0)
        for col in range(self.GRID_COLS - 2, -1, -1):
            border.append(self._grid_pos_to_idx(self.GRID_ROWS - 1, col))
        
        # Left edge: bottom to top (row ROWS-2 to 1, col 0)
        for row in range(self.GRID_ROWS - 2, 0, -1):
            border.append(self._grid_pos_to_idx(row, 0))
        
        return border

    def _compute_shape_border(self, seg_int_nodes: list, seg_directions: list) -> list:
        """
        Compute shape border indices by walking the grid using segment directions.

        For a T-shirt with segment_directions=[(0,1),(1,0),(0,-1),(1,0),(0,-1),(-1,0),(0,-1),(-1,0)]:
        Walks: RIGHT(top), DOWN(right shoulder), LEFT(armpit inward), DOWN(body right),
               LEFT(bottom), UP(body left), LEFT(armpit inward), UP(left shoulder)

        Args:
            seg_int_nodes: List of interior node counts per segment
            seg_directions: List of (dr, dc) direction tuples per segment

        Returns:
            List of grid indices forming the shape border in CW order (no duplicates)
        """
        border = []
        r, c = 0, 0  # Start at top-left corner C0
        n_segments = len(seg_directions)

        for seg_idx in range(n_segments):
            dr, dc = seg_directions[seg_idx]
            n_steps = seg_int_nodes[seg_idx] + 1  # interior + 1 to reach next corner

            # Add start corner + interior nodes (NOT the end corner - it's the next seg's start)
            for step in range(n_steps):
                border.append(self._grid_pos_to_idx(r + dr * step, c + dc * step))

            # Advance to next corner position
            r += dr * n_steps
            c += dc * n_steps

        return border

    def _compute_valid_nodes_from_border(self, border: list) -> set:
        """
        Determine valid grid nodes from the shape border polygon.

        For each row, finds min/max column among border nodes and marks all
        columns in between as valid. This fills the interior of the shape.

        Args:
            border: List of grid indices forming the shape border

        Returns:
            Set of valid grid indices (border + interior)
        """
        valid = set(border)

        # Find row extents from border nodes
        row_extents = {}
        for grid_idx in border:
            r, c = self._idx_to_grid_pos(grid_idx)
            if r not in row_extents:
                row_extents[r] = (c, c)
            else:
                min_c, max_c = row_extents[r]
                row_extents[r] = (min(min_c, c), max(max_c, c))

        # Fill interior: all columns between min and max for each row
        for r, (min_c, max_c) in row_extents.items():
            for c in range(min_c, max_c + 1):
                valid.add(self._grid_pos_to_idx(r, c))

        return valid

    
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
        """Compute relative edge length errors (only for valid edges)."""
        errors = []
        for i, j in self.valid_edges:
            if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                continue
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
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        
        # Apply depth threshold
        valid_depth = (depth > 0) & (depth < self.max_depth)
        mask_filtered = (mask > 0) & valid_depth
        
        n_foreground = np.sum(mask_filtered > 0)
        if n_foreground < self.min_foreground_pixels:
            self.consecutive_skips += 1
            return {
                'success': False,
                'reason': 'insufficient_foreground',
                'mode': 'skip',
                'foreground_mask': mask_filtered,
            }
        
        if not self.is_initialized:
            result = self.initialize(mask_filtered, depth, frame_idx)
        else:
            if self.consecutive_skips >= self.max_skips_before_restart:
                print(f"  Warm restart after {self.consecutive_skips} skips")
                result = self.initialize(mask_filtered, depth, frame_idx)
                result['mode'] = 'restart'
            else:
                result = self.track(mask_filtered, depth, frame_idx)
        
        result['foreground_mask'] = mask_filtered
        return result


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

        # Start from previous keypoints
        keypoints = self.prev_keypoints.copy()

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
                'geom': geom_time,
                'total': track_time,
            },
        }
