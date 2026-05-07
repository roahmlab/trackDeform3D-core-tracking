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

from cloth_rect_contour_init import (
    find_max_inscribed_rectangle_rotated,
    get_bounding_rect_same_orientation,
)


class ClothTracker:
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
        # CPD parameters
        cpd_beta: float = 10.0,
        cpd_lambda: float = 0.1,
        cpd_w: float = 0.1,
        cpd_max_iter: int = 200,
        cpd_tol: float = 1e-4,
        cpd_downsample: int = 3000,
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
            cpd_*: CPD registration parameters
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

    def _project_boundary_to_contour(
        self,
        keypoints: np.ndarray,
        contour_3d: np.ndarray,
    ) -> None:
        """
        Project corner and border nodes to nearest point on contour.
        Modifies keypoints in-place.
        """
        if contour_3d is None or len(contour_3d) == 0:
            return

        contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        contour_nn.fit(contour_3d)

        boundary_indices = list(self.contour_corner_grid_indices) + list(self.border_grid_indices)
        n_projected = 0
        for idx in boundary_indices:
            if np.any(np.isnan(keypoints[idx])):
                continue
            _, nn_idx = contour_nn.kneighbors(keypoints[idx:idx + 1])
            keypoints[idx] = contour_3d[nn_idx[0, 0]].copy()
            n_projected += 1

        print(f"  [Project] Projected {n_projected} boundary nodes to contour")

    def _align_corners_to_grid(self, detected_corners: np.ndarray, rect_corners: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Use _order_corners_along_contour instead.
        Kept for compatibility but just returns corners as-is.
        """
        return detected_corners
    
    def _order_corners_along_contour(
        self,
        detected_corners: np.ndarray,
        contour_3d: np.ndarray
    ) -> np.ndarray:
        """
        Order detected corners by walking the contour from C0 to C1.

        C0 = top-left most corner (smallest X, tiebreak smallest Y)
        C1 = top-right most corner (largest X, tiebreak smallest Y)
        Walk direction: from C0 toward C1 such that C1 is the FIRST
        corner encountered (C0 directly connected to C1 on contour).

        Coordinate system:
            left = small X, right = big X
            top = small Y (including negative), down = big Y

        Args:
            detected_corners: N × 3 detected contour corners (8 for T-shirt)
            contour_3d: M × 3 contour points (ordered around contour)

        Returns:
            ordered_corners: N × 3 corners ordered as C0, C1, C2, ..., C(N-1)
            walk_forward: bool, True if walking forward (increasing contour index)
        """
        n_corners = len(detected_corners)
        if n_corners < 2 or len(contour_3d) < 10:
            return detected_corners, True

        # ============================================================
        # Step 1: C0 = top-left most (smallest X), C1 = top-right most (largest X)
        # ============================================================
        x_vals = detected_corners[:, 0]
        c0_idx = int(np.argmin(x_vals))
        c1_idx = int(np.argmax(x_vals))

        c0 = detected_corners[c0_idx]
        c1 = detected_corners[c1_idx]

        print(f"  [Init] C0 (topleft): corner {c0_idx}, pos=({c0[0]:.0f}, {c0[1]:.0f}, {c0[2]:.0f})")
        print(f"  [Init] C1 (topright): corner {c1_idx}, pos=({c1[0]:.0f}, {c1[1]:.0f}, {c1[2]:.0f})")

        # ============================================================
        # Step 2: Project all corners to contour indices
        # ============================================================
        corner_contour_indices = []
        for corner in detected_corners:
            dists = np.linalg.norm(contour_3d - corner, axis=1)
            corner_contour_indices.append(int(np.argmin(dists)))

        c0_ci = corner_contour_indices[c0_idx]
        n_contour = len(contour_3d)

        # ============================================================
        # Step 3: Walk direction — pick the direction where C1 is the
        #         FIRST corner encountered from C0 (directly connected)
        # ============================================================
        # Compute forward distance (increasing contour index) from C0 to every corner
        forward_dists = {}
        for i in range(n_corners):
            if i == c0_idx:
                continue
            ci = corner_contour_indices[i]
            if ci >= c0_ci:
                forward_dists[i] = ci - c0_ci
            else:
                forward_dists[i] = (n_contour - c0_ci) + ci

        # Backward distance = complement
        backward_dists = {i: n_contour - forward_dists[i] for i in forward_dists}

        # First corner encountered in each direction
        first_forward = min(forward_dists, key=forward_dists.get)
        first_backward = min(backward_dists, key=backward_dists.get)

        if first_forward == c1_idx:
            walk_forward = True
        elif first_backward == c1_idx:
            walk_forward = False
        else:
            # C1 not directly adjacent in either direction; pick shorter path to C1
            walk_forward = forward_dists[c1_idx] <= backward_dists[c1_idx]
            print(f"  [Init] WARNING: C1 not directly adjacent to C0, using shorter path")

        print(f"  [Init] Contour walk: C0 at idx {c0_ci}, forward={walk_forward}")

        # ============================================================
        # Step 4: Order all corners by contour distance from C0
        # ============================================================
        if walk_forward:
            sorted_others = sorted(
                [i for i in range(n_corners) if i != c0_idx],
                key=lambda i: forward_dists[i]
            )
        else:
            sorted_others = sorted(
                [i for i in range(n_corners) if i != c0_idx],
                key=lambda i: backward_dists[i]
            )

        ordered_indices = [c0_idx] + sorted_others
        ordered_corners = detected_corners[ordered_indices]

        if sorted_others[0] != c1_idx:
            print(f"  [Init] WARNING: C1 is not second in order! Got corner {sorted_others[0]} instead")

        print(f"  [Init] Corner order: {ordered_indices}")
        return ordered_corners, walk_forward
    
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

    def _split_contour_at_corners(
        self,
        contour_3d: np.ndarray,
        keypoints: np.ndarray,
    ) -> list:
        """
        Split 3D contour into segments between adjacent ordered corners.

        Uses self.ordered_corner_grid_indices (set during init) to find
        corner positions on the current contour, then extracts the contour
        arc between each pair of adjacent corners.

        Returns:
            List of N segments (np.ndarray), where segment[i] is the
            contour arc from ordered corner i to ordered corner i+1.
            Returns None if data is insufficient.
        """
        if not hasattr(self, 'ordered_corner_grid_indices') or self.ordered_corner_grid_indices is None:
            return None
        n_corners = len(self.ordered_corner_grid_indices)
        if n_corners < 2 or contour_3d is None or len(contour_3d) < 10:
            return None

        # Find each corner's index on the current contour
        corner_contour_idx = []
        for grid_idx in self.ordered_corner_grid_indices:
            if np.any(np.isnan(keypoints[grid_idx])):
                return None
            dists = np.linalg.norm(contour_3d - keypoints[grid_idx], axis=1)
            corner_contour_idx.append(int(np.argmin(dists)))

        n_contour = len(contour_3d)
        segments = []
        for i in range(n_corners):
            idx_start = corner_contour_idx[i]
            idx_end = corner_contour_idx[(i + 1) % n_corners]

            if idx_start <= idx_end:
                segment = contour_3d[idx_start:idx_end + 1]
            else:
                # Wrap around the contour
                segment = np.concatenate([contour_3d[idx_start:], contour_3d[:idx_end + 1]])

            if len(segment) < 2:
                # Degenerate segment — fallback to full contour
                return None
            segments.append(segment)

        return segments

    def _compute_target_edge_length(self, corners_3d: np.ndarray) -> float:
        """
        Compute target edge length from rectangle corners.
        
        target_edge = avg(rect_width, rect_height) / (grid_size - 1)
        
        Args:
            corners_3d: 4 × 3 rectangle corners [TL, TR, BR, BL]
            
        Returns:
            Target edge length in mm
        """
        TL, TR, BR, BL = corners_3d
        width = (np.linalg.norm(TR - TL) + np.linalg.norm(BR - BL)) / 2
        height = (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR)) / 2
        
        # Average dimension divided by number of grid cells
        avg_dim = (width + height) / 2
        target_edge = avg_dim / (max(self.GRID_ROWS, self.GRID_COLS) - 1)
        
        return target_edge
    
    def _compute_segment_arc_length(self, segment: np.ndarray) -> float:
        """
        Compute arc length of a 3D segment (sum of edge lengths).
        
        Args:
            segment: M × 3 ordered 3D points
            
        Returns:
            Total arc length in mm
        """
        if len(segment) < 2:
            return 0.0
        
        diffs = np.diff(segment, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(edge_lengths))
    
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
    
    def _arc_length_sample(self, segment: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample points at uniform arc-length intervals along a contour segment.
        
        This ensures evenly spaced border nodes (unlike nearest-neighbor snap).
        
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
            return np.tile(segment[0], (n_samples, 1))
        
        # Target arc-lengths for uniformly spaced samples (exclude endpoints)
        target_lengths = np.linspace(0, total_length, n_samples + 2)[1:-1]
        
        # Interpolate to find 3D positions at target arc-lengths
        sampled_points = []
        for target in target_lengths:
            idx = np.searchsorted(cumulative_length, target, side='right') - 1
            idx = np.clip(idx, 0, len(segment) - 2)
            
            local_dist = target - cumulative_length[idx]
            seg_length = edge_lengths[idx] if idx < len(edge_lengths) else 1e-6
            t = local_dist / max(seg_length, 1e-6)
            t = np.clip(t, 0, 1)
            
            point = (1 - t) * segment[idx] + t * segment[idx + 1]
            sampled_points.append(point)
        
        return np.array(sampled_points, dtype=np.float64)
    
    def _get_contour_segment(
        self,
        contour_3d: np.ndarray,
        idx_start: int,
        idx_end: int,
        walk_forward: bool,
        all_corner_contour_indices: list = None,
    ) -> np.ndarray:
        """
        Extract the contour segment between two consecutive corners.

        Tries both directions on the contour and picks the one that does NOT
        pass through any other detected corner.  Falls back to ``walk_forward``
        if both (or neither) direction is clean.

        Args:
            contour_3d: N × 3 full contour points (ordered around contour)
            idx_start: contour index of the start corner
            idx_end: contour index of the end corner
            walk_forward: hint direction (used as tiebreaker only)
            all_corner_contour_indices: contour indices for ALL detected
                corners — required to choose the clean direction

        Returns:
            segment: M × 3 contour points from start to end (no other
            corners in between)
        """
        n = len(contour_3d)
        if n < 4:
            return contour_3d

        if idx_start == idx_end:
            return contour_3d[idx_start:idx_start + 1].copy()

        other_corners = (
            set(all_corner_contour_indices) - {idx_start, idx_end}
            if all_corner_contour_indices is not None
            else set()
        )

        def _extract(forward: bool):
            """Return (segment_array, interior_contour_indices) for one direction."""
            if forward:
                if idx_end > idx_start:
                    seg = contour_3d[idx_start:idx_end + 1]
                    interior = set(range(idx_start + 1, idx_end))
                else:
                    seg = np.vstack([contour_3d[idx_start:], contour_3d[:idx_end + 1]])
                    interior = set(range(idx_start + 1, n)) | set(range(0, idx_end))
            else:
                if idx_end < idx_start:
                    seg = contour_3d[idx_end:idx_start + 1][::-1]
                    interior = set(range(idx_end + 1, idx_start))
                else:
                    seg = np.vstack([
                        contour_3d[:idx_start + 1][::-1],
                        contour_3d[idx_end:][::-1],
                    ])
                    interior = set(range(0, idx_start)) | set(range(idx_end + 1, n))
            return seg, interior

        seg_fwd, interior_fwd = _extract(True)
        seg_bwd, interior_bwd = _extract(False)

        fwd_clean = len(other_corners & interior_fwd) == 0
        bwd_clean = len(other_corners & interior_bwd) == 0

        if fwd_clean and not bwd_clean:
            return seg_fwd
        elif bwd_clean and not fwd_clean:
            return seg_bwd
        elif fwd_clean and bwd_clean:
            # Both clean — pick the shorter one (or use hint)
            if len(seg_fwd) <= len(seg_bwd):
                return seg_fwd
            else:
                return seg_bwd
        else:
            # Neither clean — fall back to hint direction
            print(f"  [Contour] WARNING: no clean path [{idx_start}→{idx_end}], "
                  f"fwd crosses {sorted(other_corners & interior_fwd)}, "
                  f"bwd crosses {sorted(other_corners & interior_bwd)}")
            return seg_fwd if walk_forward else seg_bwd

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
    
    def _find_rect_corners_from_contour_3d(
        self, 
        contour_3d: np.ndarray,
    ) -> tuple:
        """
        Find 4 corners of bounding rectangle from denoised 3D contour.
        Projects contour to 2D, finds max inscribed rect orientation, then bounding rect.
        
        Args:
            contour_3d: Denoised 3D contour points (N, 3)
            
        Returns:
            corners_3d: 4 × 3 array of 3D corners [TL, TR, BR, BL]
            angle: Orientation angle in degrees
        """
        if len(contour_3d) < 10:
            return None, None
        
        # Project 3D contour to 2D pixel coordinates
        contour_2d = []
        for x, y, z in contour_3d:
            if z > 0:
                col = self.fx * x / z + self.cx
                row = self.fy * y / z + self.cy
                contour_2d.append([col, row])
        
        contour_2d = np.array(contour_2d)
        if len(contour_2d) < 10:
            return None, None
        
        # Create mask from contour for inscribed rectangle
        contour_int = contour_2d.astype(np.int32).reshape(-1, 1, 2)
        H, W = 1080, 1920  # Assume standard resolution
        temp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [contour_int], 255)
        
        # Find max inscribed rectangle to get orientation
        result = find_max_inscribed_rectangle_rotated(temp_mask, n_angles=90)
        if result is None:
            return None, None
        
        _, _, _, _, angle = result
        
        # Get bounding rectangle with same orientation from the 2D contour
        bounding_corners_2d = get_bounding_rect_same_orientation(contour_2d, angle)
        
        # Order corners: TL, TR, BR, BL
        min_sum_idx = np.argmin(bounding_corners_2d[:, 0] + bounding_corners_2d[:, 1])
        bounding_corners_2d = np.roll(bounding_corners_2d, -min_sum_idx, axis=0)
        
        # Check orientation: TR should be to the right of TL
        v1 = bounding_corners_2d[1] - bounding_corners_2d[0]
        v2 = bounding_corners_2d[3] - bounding_corners_2d[0]
        if v1[0] < v2[0]:
            bounding_corners_2d = bounding_corners_2d[[0, 3, 2, 1]]
        
        # Convert corners to 3D using average depth of nearby contour points
        corners_3d = []
        for col, row in bounding_corners_2d:
            # Find nearest contour point to get depth
            dists = np.linalg.norm(contour_2d - np.array([col, row]), axis=1)
            nearest_idx = np.argmin(dists)
            z = contour_3d[nearest_idx, 2]
            
            x = (col - self.cx) * z / self.fx
            y = (row - self.cy) * z / self.fy
            corners_3d.append([x, y, z])
        
        return np.array(corners_3d), angle

    def _find_rect_corners(self, mask: np.ndarray, depth: np.ndarray) -> tuple:
        """
        Find 4 corners of the bounding rectangle aligned with max inscribed rect.
        Uses smoothed contour for better fitting on T-shirt shapes.
        
        Returns:
            corners_2d: 4 × 2 array of (col, row) for corners [TL, TR, BR, BL]
            angle: Orientation angle in degrees
        """
        # Smooth mask to get cleaner contour (important for T-shirt)
        kernel = np.ones((5, 5), np.uint8)
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        valid_mask = (mask_smooth > 0) & (depth > 0) & (depth < self.max_depth)
        
        # Find max inscribed rectangle to get orientation
        result = find_max_inscribed_rectangle_rotated(valid_mask, n_angles=90)
        if result is None:
            return None, None
        
        _, _, _, _, angle = result
        
        # Get contour points from smoothed mask
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            return None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_2d = largest_contour.squeeze()  # (N, 2) in (col, row)
        
        # Get bounding rectangle with same orientation
        bounding_corners_2d = get_bounding_rect_same_orientation(contour_2d, angle)
        
        # Order corners: TL, TR, BR, BL
        # Find corner with min (x + y) as TL
        min_sum_idx = np.argmin(bounding_corners_2d[:, 0] + bounding_corners_2d[:, 1])
        bounding_corners_2d = np.roll(bounding_corners_2d, -min_sum_idx, axis=0)
        
        # Check orientation: TR should be to the right of TL
        v1 = bounding_corners_2d[1] - bounding_corners_2d[0]
        v2 = bounding_corners_2d[3] - bounding_corners_2d[0]
        if v1[0] < v2[0]:
            bounding_corners_2d = bounding_corners_2d[[0, 3, 2, 1]]
        
        return bounding_corners_2d, angle
    
    def _corners_2d_to_3d(self, corners_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Convert 2D corner coordinates (col, row) to 3D."""
        H, W = depth.shape
        avg_depth = np.mean(depth[(depth > 0) & (depth < self.max_depth)])
        
        corners_3d = []
        for col, row in corners_2d:
            row_int, col_int = int(np.clip(row, 0, H-1)), int(np.clip(col, 0, W-1))
            z = depth[row_int, col_int]
            if z <= 0 or z >= self.max_depth:
                z = avg_depth
            x = (col - self.cx) * z / self.fx
            y = (row - self.cy) * z / self.fy
            corners_3d.append([x, y, z])
        
        return np.array(corners_3d, dtype=np.float64)
    
    def _initialize_grid_from_rect(
        self,
        corners_3d: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray = None,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Initialize grid keypoints with shape-border-based placement + row-wise interior.

        Algorithm:
        1. Compute shape border (T-shape via segment_directions, or rectangle fallback)
        2. Map contour corners to shape border positions via segment_interior_nodes
        3. Per segment: straight line between corners → even spacing → project to contour
        4. Determine valid nodes from shape polygon
        5. Interior nodes via piecewise row-wise interpolation from border + snap to point cloud
        6. Rebuild valid edges

        Args:
            corners_3d: 4 × 3 rectangle corner positions [TL, TR, BR, BL]
            point_cloud: N × 3 foreground point cloud
            contour_3d: M × 3 contour points (ordered around contour)
            mask: H × W binary mask (for fallback T-cropping)

        Returns:
            keypoints: N_KEYPOINTS × 3 grid keypoints (NaN for invalid/cropped nodes)
        """
        keypoints = np.full((self.N_KEYPOINTS, 3), np.nan, dtype=np.float64)

        TL, TR, BR, BL = corners_3d
        border_assigned = set()
        border_grid_order = []

        if contour_3d is not None and len(contour_3d) > 20 and self.detected_corners_3d is not None:
            detected_corners = self.detected_corners_3d
            n_corners = len(detected_corners)

            # Order corners along contour
            detected_corners, walk_forward = self._order_corners_along_contour(detected_corners, contour_3d)
            self.detected_corners_3d = detected_corners

            # Find contour indices for each detected corner
            corner_contour_indices = []
            for corner in detected_corners:
                dists = np.linalg.norm(contour_3d - corner, axis=1)
                corner_contour_indices.append(np.argmin(dists))

            # ============================================================
            # Step A: Compute/validate segment_interior_nodes
            # ============================================================
            if self.segment_interior_nodes is not None:
                seg_int_nodes = self.segment_interior_nodes
            else:
                # Default for rectangle (4 corners)
                rect_border = self._get_grid_border_clockwise()
                n_border = len(rect_border)
                n_interior_total = n_border - n_corners
                n_per_seg = n_interior_total // n_corners
                remainder = n_interior_total % n_corners
                seg_int_nodes = [n_per_seg + (1 if i < remainder else 0) for i in range(n_corners)]

            # Compute default directions for rectangle if not provided
            if self.segment_directions is not None:
                seg_directions = self.segment_directions
            elif n_corners == 4:
                seg_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # RIGHT, DOWN, LEFT, UP
            else:
                # Auto-infer segment directions from 3D corner displacements
                # Use rectangle corners to define reference frame:
                #   TL→TR = "right" (grid col+), TL→BL = "down" (grid row+)
                right_dir = TR - TL
                right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-12)
                down_dir = BL - TL
                down_dir = down_dir / (np.linalg.norm(down_dir) + 1e-12)

                seg_directions = []
                for i in range(n_corners):
                    c_start = detected_corners[i]
                    c_end = detected_corners[(i + 1) % n_corners]
                    disp = c_end - c_start

                    right_proj = np.dot(disp, right_dir)
                    down_proj = np.dot(disp, down_dir)

                    if abs(right_proj) > abs(down_proj):
                        seg_directions.append((0, 1) if right_proj > 0 else (0, -1))
                    else:
                        seg_directions.append((1, 0) if down_proj > 0 else (-1, 0))

                print(f"  [Init] Auto-inferred segment_directions = {seg_directions}")

            print(f"  [Init] Step A: segment_interior_nodes = {seg_int_nodes}")

            # ============================================================
            # Step B: Compute shape border and corner mapping
            # ============================================================
            if seg_directions is not None:
                shape_border = self._compute_shape_border(seg_int_nodes, seg_directions)
                valid_nodes = self._compute_valid_nodes_from_border(shape_border)
            else:
                shape_border = self._get_grid_border_clockwise()
                valid_nodes = set(range(self.N_KEYPOINTS))  # all valid, T-crop later

            n_shape_border = len(shape_border)
            expected_border = n_corners + sum(seg_int_nodes)
            print(f"  [Init] Step B: Shape border = {n_shape_border} nodes "
                  f"(expected {expected_border}), valid nodes = {len(valid_nodes)}")

            # Corner positions in shape_border: cumulative offsets
            cumsum = [0]
            for i in range(n_corners):
                cumsum.append(cumsum[-1] + 1 + seg_int_nodes[i])

            contour_corner_grid_indices = [shape_border[cumsum[i]] for i in range(n_corners)]
            self.contour_corner_grid_indices = contour_corner_grid_indices
            self.fixed_indices = set(contour_corner_grid_indices)

            print(f"  [Init]   Corner grid indices = {contour_corner_grid_indices}")
            for i in range(n_corners):
                r, c = self._idx_to_grid_pos(contour_corner_grid_indices[i])
                print(f"    C{i} → grid ({r},{c}) = idx {contour_corner_grid_indices[i]}")

            # ============================================================
            # Step C: Place corner nodes
            # ============================================================
            for i in range(n_corners):
                grid_idx = contour_corner_grid_indices[i]
                keypoints[grid_idx] = detected_corners[i].copy()
                border_assigned.add(grid_idx)
                border_grid_order.append(grid_idx)

            # ============================================================
            # Step D: Place non-corner border nodes per segment
            #   Straight line between corners → even spacing → project to full contour
            # ============================================================
            print(f"  [Init] Step D: Border nodes via straight-line + project to contour")
            for seg_idx in range(n_corners):
                n_int = seg_int_nodes[seg_idx]
                if n_int == 0:
                    continue

                c_start = detected_corners[seg_idx]
                c_end = detected_corners[(seg_idx + 1) % n_corners]

                # Even spacing on straight line, then project to full contour
                for k in range(1, n_int + 1):
                    t = k / (n_int + 1)
                    line_pt = (1.0 - t) * c_start + t * c_end

                    # Project to nearest point on the full contour
                    dists = np.linalg.norm(contour_3d - line_pt, axis=1)
                    projected_pt = contour_3d[np.argmin(dists)].copy()

                    border_pos = cumsum[seg_idx] + k
                    grid_idx = shape_border[border_pos]
                    keypoints[grid_idx] = projected_pt
                    border_assigned.add(grid_idx)

                print(f"    Seg C{seg_idx}→C{(seg_idx+1)%n_corners}: n_interior={n_int}")

            # Store per-segment assignment for border nodes
            # border_segment_assignment: grid_idx → segment_id (0..n_corners-1)
            self.border_segment_assignment = {}
            for seg_idx in range(n_corners):
                n_int = seg_int_nodes[seg_idx]
                for k in range(1, n_int + 1):
                    border_pos = cumsum[seg_idx] + k
                    grid_idx = shape_border[border_pos]
                    self.border_segment_assignment[grid_idx] = seg_idx

            # Store contour-ordered corner grid indices (for splitting contour into segments)
            self.ordered_corner_grid_indices = list(contour_corner_grid_indices)

            # Use shape_border as the authoritative border order (correct CW walk)
            border_grid_order = list(shape_border)
            print(f"  [Init]   Assigned {len(border_assigned)} border nodes")
            print(f"  [Init]   Segment assignment: {len(self.border_segment_assignment)} border nodes across {n_corners} segments")

            # ============================================================
            # Step E: Interior nodes - piecewise row-wise interpolation
            #   For each row, find border nodes sorted by column.
            #   For each gap between consecutive border nodes, interpolate interior.
            # ============================================================
            print(f"  [Init] Step E: Interior nodes - piecewise row-wise interpolation")
            cloud_nn = None
            if len(point_cloud) > 0:
                cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                cloud_nn.fit(point_cloud)

            # Build per-row border columns lookup
            border_cols_by_row = {}
            for grid_idx in border_assigned:
                r, c = self._idx_to_grid_pos(grid_idx)
                border_cols_by_row.setdefault(r, []).append(c)
            for r in border_cols_by_row:
                border_cols_by_row[r].sort()

            n_interior_placed = 0
            for r in range(self.GRID_ROWS):
                if r not in border_cols_by_row:
                    continue
                bcols = border_cols_by_row[r]
                if len(bcols) < 2:
                    continue

                # For each pair of consecutive border columns, interpolate interior
                for bi in range(len(bcols) - 1):
                    left_c = bcols[bi]
                    right_c = bcols[bi + 1]

                    if right_c - left_c <= 1:
                        continue  # Adjacent border nodes, no interior gap

                    left_idx = self._grid_pos_to_idx(r, left_c)
                    right_idx = self._grid_pos_to_idx(r, right_c)
                    left_pos = keypoints[left_idx]
                    right_pos = keypoints[right_idx]

                    if np.any(np.isnan(left_pos)) or np.any(np.isnan(right_pos)):
                        continue

                    for c in range(left_c + 1, right_c):
                        grid_idx = self._grid_pos_to_idx(r, c)
                        if grid_idx in border_assigned:
                            continue
                        if grid_idx not in valid_nodes:
                            continue

                        t = (c - left_c) / (right_c - left_c)
                        keypoints[grid_idx] = (1.0 - t) * left_pos + t * right_pos

                        # Snap to point cloud
                        if cloud_nn is not None:
                            _, nearest_idx = cloud_nn.kneighbors(keypoints[grid_idx:grid_idx+1])
                            keypoints[grid_idx] = point_cloud[nearest_idx[0, 0]]
                        n_interior_placed += 1

            print(f"  [Init]   Placed {n_interior_placed} interior nodes")

            # ============================================================
            # Step F: Invalidate nodes outside the shape
            # ============================================================
            n_invalidated = 0
            for idx in range(self.N_KEYPOINTS):
                if idx not in valid_nodes and not np.isnan(keypoints[idx, 0]):
                    keypoints[idx] = np.nan
                    n_invalidated += 1
            print(f"  [Init] Step F: Shape crop invalidated {n_invalidated} nodes outside T-shape")

        else:
            # Fallback: use rectangular border with bilinear positions
            print(f"  [Init] Border nodes - bilinear fallback (no contour/corners)")
            grid_border = self._get_grid_border_clockwise()
            for grid_idx in grid_border:
                row, col = self._idx_to_grid_pos(grid_idx)
                u = col / (self.GRID_COLS - 1)
                v = row / (self.GRID_ROWS - 1)
                top = (1 - u) * TL + u * TR
                bottom = (1 - u) * BL + u * BR
                keypoints[grid_idx] = (1 - v) * top + v * bottom
                border_assigned.add(grid_idx)
                border_grid_order.append(grid_idx)
            self.contour_corner_grid_indices = list(self.CORNER_INDICES)
            self.fixed_indices = set(self.CORNER_INDICES)

            # Interior: bilinear + snap
            cloud_nn = None
            if len(point_cloud) > 0:
                cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                cloud_nn.fit(point_cloud)
            for idx in range(self.N_KEYPOINTS):
                if idx in border_assigned:
                    continue
                row, col = self._idx_to_grid_pos(idx)
                u = col / (self.GRID_COLS - 1)
                v = row / (self.GRID_ROWS - 1)
                top = (1 - u) * TL + u * TR
                bottom = (1 - u) * BL + u * BR
                keypoints[idx] = (1 - v) * top + v * bottom
                if cloud_nn is not None:
                    _, nearest_idx = cloud_nn.kneighbors(keypoints[idx:idx+1])
                    keypoints[idx] = point_cloud[nearest_idx[0, 0]]

            # T-crop via mask
            if mask is not None:
                for idx in range(self.N_KEYPOINTS):
                    if idx in border_assigned:
                        continue
                    if not np.any(np.isnan(keypoints[idx])):
                        if not self._is_point_inside_mask(keypoints[idx], mask):
                            keypoints[idx] = np.nan

        # Store border info for visualization
        self.border_grid_indices = border_grid_order

        # ================================================================
        # Rebuild valid faces → keep top 40 → invalidate orphan vertices → rebuild edges
        # ================================================================
        self._rebuild_valid_faces(keypoints, n_target_faces=40)

        # Invalidate keypoints not part of any face
        face_verts = set()
        for tl, tr, br, bl in self.valid_faces:
            face_verts.update([tl, tr, br, bl])
        n_orphaned = 0
        for idx in range(self.N_KEYPOINTS):
            if not np.isnan(keypoints[idx, 0]) and idx not in face_verts:
                keypoints[idx] = np.nan
                n_orphaned += 1
        if n_orphaned > 0:
            print(f"  [Init] Removed {n_orphaned} orphan vertices not in any face")

        # Rebuild edges after orphan removal
        self._rebuild_valid_edges(keypoints)

        n_valid = np.sum(~np.isnan(keypoints[:, 0]))
        print(f"  [Init] Final: {n_valid} vertices, "
              f"{len(self.valid_edges)} edges, {len(self.valid_faces)} faces")

        # Classify nodes by topology and project boundary to contour
        self._classify_nodes_by_topology(keypoints, contour_3d, self.detected_corners_3d)
        self._project_boundary_to_contour(keypoints, contour_3d)

        return keypoints
    
    def _arc_length_sample_with_endpoints(self, segment: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample segment at uniform arc-length intervals INCLUDING endpoints.
        
        Args:
            segment: M × 3 contour segment points (ordered)
            n_samples: Total number of samples including both endpoints
        
        Returns:
            sampled: n_samples × 3 uniformly spaced points
        """
        if len(segment) < 2 or n_samples < 2:
            return segment.copy()
        
        # Compute cumulative arc length
        diffs = np.diff(segment, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length < 1e-6:
            return np.tile(segment[0], (n_samples, 1))
        
        # Target arc-lengths (including endpoints)
        target_lengths = np.linspace(0, total_length, n_samples)
        
        # Interpolate to find 3D positions
        sampled_points = []
        for target in target_lengths:
            idx = np.searchsorted(cumulative_length, target, side='right') - 1
            idx = np.clip(idx, 0, len(segment) - 2)
            
            local_dist = target - cumulative_length[idx]
            seg_length = edge_lengths[idx] if idx < len(edge_lengths) else 1e-6
            t = local_dist / max(seg_length, 1e-6)
            t = np.clip(t, 0, 1)
            
            point = (1 - t) * segment[idx] + t * segment[idx + 1]
            sampled_points.append(point)
        
        return np.array(sampled_points, dtype=np.float64)
    
    def _sample_contour_uniform(self, contour_3d: np.ndarray, target_edge: float) -> np.ndarray:
        """
        Sample contour at uniform arc-length intervals.
        
        Args:
            contour_3d: M × 3 ordered contour points
            target_edge: Target spacing between samples (mm)
            
        Returns:
            samples: K × 3 uniformly spaced samples around contour
        """
        if len(contour_3d) < 2:
            return contour_3d.copy()
        
        # Compute cumulative arc length (closed contour)
        diffs = np.diff(contour_3d, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        # Add closing edge
        closing_edge = np.linalg.norm(contour_3d[0] - contour_3d[-1])
        edge_lengths = np.append(edge_lengths, closing_edge)
        
        cumulative_length = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cumulative_length[-1]
        
        if total_length < target_edge:
            return contour_3d.copy()
        
        # Number of samples
        n_samples = max(4, int(round(total_length / target_edge)))
        
        # Target arc-lengths
        target_lengths = np.linspace(0, total_length, n_samples, endpoint=False)
        
        # Interpolate positions (handle wrap-around)
        samples = []
        extended_contour = np.vstack([contour_3d, contour_3d[0:1]])  # Close the loop
        
        for target in target_lengths:
            idx = np.searchsorted(cumulative_length, target, side='right') - 1
            idx = np.clip(idx, 0, len(contour_3d) - 1)
            
            local_dist = target - cumulative_length[idx]
            seg_length = edge_lengths[idx] if idx < len(edge_lengths) else 1e-6
            t = local_dist / max(seg_length, 1e-6)
            t = np.clip(t, 0, 1)
            
            next_idx = (idx + 1) % len(extended_contour)
            point = (1 - t) * contour_3d[idx] + t * extended_contour[next_idx]
            samples.append(point)
        
        return np.array(samples, dtype=np.float64)
    
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

    def _connection_aware_mapping(
        self,
        contour_samples: np.ndarray,
        start_idx: int,
        grid_border_order: list,
        target_edge: float,
    ) -> dict:
        """
        Map contour samples to grid border nodes with connection-awareness.
        
        Start at contour_samples[start_idx] → grid_border_order[0].
        Walk contour: each next sample maps to a NEIGHBOR of the current grid node.
        
        Args:
            contour_samples: K × 3 uniformly spaced contour samples
            start_idx: Index into contour_samples to start from
            grid_border_order: Grid border indices in clockwise order
            target_edge: Target edge length for distance validation
            
        Returns:
            mapping: dict {contour_idx: grid_idx}
        """
        n_contour = len(contour_samples)
        n_grid = len(grid_border_order)
        
        mapping = {}
        used_grid = set()
        
        # Start: contour_samples[start_idx] → grid_border_order[0]
        mapping[start_idx] = grid_border_order[0]
        used_grid.add(grid_border_order[0])
        
        current_grid_pos = 0  # Position in grid_border_order
        
        # Walk contour in order
        for step in range(1, n_contour):
            contour_idx = (start_idx + step) % n_contour
            contour_pt = contour_samples[contour_idx]
            
            # Candidate: next grid border node in order
            next_grid_pos = (current_grid_pos + 1) % n_grid
            
            # Check if we should advance grid position
            # Compare distance to current vs next grid position
            if next_grid_pos < len(grid_border_order) and grid_border_order[next_grid_pos] not in used_grid:
                # Advance to next grid border node
                current_grid_pos = next_grid_pos
                grid_idx = grid_border_order[current_grid_pos]
                mapping[contour_idx] = grid_idx
                used_grid.add(grid_idx)
        
        return mapping
    
    def _repulsion_relaxation(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray = None,
    ) -> np.ndarray:
        """
        Spring-based relaxation with grid topology (fabric_tracker style).

        Constraints:
        - Corner nodes (self.fixed_indices): completely fixed
        - Border nodes (self.border_grid_indices): apply force → snap to contour
        - Interior nodes: free movement
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8

        if K <= 1 or len(point_cloud) == 0:
            return keypoints

        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)

        border_set = set(self.border_grid_indices) if self.border_grid_indices else set()

        # Build contour NN for border snapping
        contour_nn = None
        contour_length = 0.0
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)
            contour_length = np.sum(np.linalg.norm(np.diff(contour_3d, axis=0), axis=1))
            closing_dist = np.linalg.norm(contour_3d[-1] - contour_3d[0])
            if closing_dist < 100:
                contour_length += closing_dist

        # Compute target edge length from mean of current valid edges
        edge_lengths = []
        for i, j in self.valid_edges:
            if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                continue
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            if length > epsilon:
                edge_lengths.append(length)
        if len(edge_lengths) == 0:
            return keypoints
        target_length = np.mean(edge_lengths)
        print(f"  [Repulsion] Target edge from mean: {target_length:.1f}mm "
              f"(std={np.std(edge_lengths):.1f}mm, min={np.min(edge_lengths):.1f}mm, "
              f"max={np.max(edge_lengths):.1f}mm)", flush=True)

        lr = self.repulsion_lr / 25.0
        print(f"  [Repulsion] Running {self.repulsion_iterations} iterations, lr={lr:.3f}", flush=True)

        prev_std = float('inf')
        for iteration in range(self.repulsion_iterations):
            forces = np.zeros_like(keypoints)

            for i, j in self.valid_edges:
                if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                    continue
                vec = keypoints[j] - keypoints[i]
                current_length = np.linalg.norm(vec)
                if current_length < epsilon:
                    continue

                direction = vec / current_length
                force_magnitude = (current_length - target_length)
                force = force_magnitude * direction

                forces[i] += force
                forces[j] -= force

            # Apply forces with constraints
            for i in range(K):
                if np.any(np.isnan(keypoints[i])):
                    continue
                if i in self.fixed_indices:
                    continue  # corners: completely fixed
                elif i in border_set:
                    # Border: apply force, snap every 10 iters for first 50, then every 5
                    keypoints[i] += lr * forces[i]
                    snap_freq = 10 if iteration < 50 else 5
                    if contour_nn is not None and iteration % snap_freq == snap_freq - 1:
                        _, idx = contour_nn.kneighbors(keypoints[i:i + 1])
                        keypoints[i] = contour_3d[idx[0, 0]].copy()
                else:
                    # Interior: free movement
                    keypoints[i] += lr * forces[i]

            # Convergence check every 50 iterations
            if iteration % 50 == 49:
                edge_lens = [
                    np.linalg.norm(keypoints[i] - keypoints[j])
                    for i, j in self.valid_edges
                    if not (np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])))
                ]
                if edge_lens:
                    curr_std = np.std(edge_lens)
                    curr_mean = np.mean(edge_lens)
                    print(f"  [Repulsion] iter {iteration+1}: mean={curr_mean:.1f}mm "
                          f"(target={target_length:.1f}mm), std={curr_std:.1f}mm", flush=True)
                    if abs(prev_std - curr_std) < 0.1:
                        print(f"  [Repulsion] Converged at iteration {iteration+1}", flush=True)
                        break
                    prev_std = curr_std

        # Final: soft project interior to point cloud (α=0.3)
        for i in range(K):
            if np.any(np.isnan(keypoints[i])):
                continue
            if i in self.fixed_indices or i in border_set:
                continue
            _, idx = cloud_nn.kneighbors(keypoints[i:i + 1])
            nearest = point_cloud[idx[0, 0]]
            keypoints[i] = 0.7 * keypoints[i] + 0.3 * nearest

        # Final summary
        final_lens = [
            np.linalg.norm(keypoints[i] - keypoints[j])
            for i, j in self.valid_edges
            if not (np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])))
        ]
        if final_lens:
            print(f"  [Repulsion] DONE: mean={np.mean(final_lens):.1f}mm, "
                  f"std={np.std(final_lens):.1f}mm, min={np.min(final_lens):.1f}mm, "
                  f"max={np.max(final_lens):.1f}mm", flush=True)

        return keypoints
    
    def _establish_ee_to_corner_mapping(self, keypoints: np.ndarray, frame_idx: int) -> None:
        """Establish mapping from EE indices to corner keypoint indices."""
        if self.ee_poses_3d is None or frame_idx >= len(self.ee_poses_3d):
            return
        
        ee_positions = self.ee_poses_3d[frame_idx]
        corner_positions = keypoints[self.CORNER_INDICES]
        
        # Filter out NaN corners
        valid_mask = ~np.any(np.isnan(corner_positions), axis=1)
        if not np.any(valid_mask):
            print("  [Warning] All corners are NaN, skipping EE mapping")
            return
        
        valid_corner_indices = [self.CORNER_INDICES[i] for i in range(len(self.CORNER_INDICES)) if valid_mask[i]]
        valid_corner_positions = corner_positions[valid_mask]
        
        cost_matrix = cdist(ee_positions, valid_corner_positions)
        ee_indices, corner_local_indices = linear_sum_assignment(cost_matrix)
        
        self.ee_to_corner_mapping = {}
        for ee_idx, corner_local_idx in zip(ee_indices, corner_local_indices):
            corner_global_idx = valid_corner_indices[corner_local_idx]
            self.ee_to_corner_mapping[ee_idx] = corner_global_idx
        
        print(f"  EE to corner mapping: {self.ee_to_corner_mapping}")
    
    def initialize(
        self, 
        mask: np.ndarray, 
        depth: np.ndarray,
        frame_idx: int = 0,
    ) -> dict:
        """
        Initialize keypoints from rectangle-aligned grid.
        
        Args:
            mask: H × W binary mask
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
        
        # Detect real contour corners first (8 for T-shirt)
        detected_corners_2d, detected_corners_3d = self._detect_contour_corners(mask, depth)
        if detected_corners_3d is not None:
            self.detected_corners_3d = detected_corners_3d.copy()
            print(f"  [Init] Detected {len(detected_corners_3d)} corners on real contour")
        else:
            self.detected_corners_3d = None
        
        # Extract 3D contour with z-denoising using detected corners
        contour_3d = self._extract_contour_3d(mask, depth, corners_3d=detected_corners_3d)
        print(f"  [Init] Extracted denoised 3D contour with {len(contour_3d)} points")
        
        if len(contour_3d) < 20:
            return {
                'success': False,
                'reason': 'insufficient_contour_points',
                'mode': 'init',
            }
        
        # Fit rectangle to the DENOISED contour (not raw mask)
        corners_3d, angle = self._find_rect_corners_from_contour_3d(contour_3d)
        if corners_3d is None:
            return {
                'success': False,
                'reason': 'no_rect_found',
                'mode': 'init',
            }
        
        print(f"  [Init] Rectangle orientation: {angle:.1f}° (fit to denoised contour)")
        
        # Store bounding rectangle corners for visualization
        self.rect_corners_3d = corners_3d.copy()
        
        # Initialize grid from rectangle corners (with T-cropping using mask)
        keypoints = self._initialize_grid_from_rect(corners_3d, point_cloud, contour_3d, mask)
        
        # Repulsion relaxation: equalize edge lengths, border snaps to contour
        keypoints = self._repulsion_relaxation(keypoints, point_cloud, contour_3d)
        
        # Establish EE to corner mapping
        self._establish_ee_to_corner_mapping(keypoints, frame_idx)
        
        # Compute reference edge lengths (only for valid edges after T-cropping)
        self.reference_lengths = {}
        for i, j in self.valid_edges:
            if not np.any(np.isnan(keypoints[i])) and not np.any(np.isnan(keypoints[j])):
                length = np.linalg.norm(keypoints[i] - keypoints[j])
                self.reference_lengths[(i, j)] = length
        
        # Store state
        self.reference_keypoints = keypoints.copy()
        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True
        self.frame_count = 1
        self.consecutive_skips = 0
        
        init_time = time.time() - t_start
        
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        return {
            'success': True,
            'mode': 'init',
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.valid_edges,
            'corners_3d': corners_3d,
            'detected_corners_3d': self.detected_corners_3d,
            'timing': {'init': init_time},
        }
    
    # ================================================================
    # CPD REGISTRATION
    # ================================================================
    
    def _cpd_register(self, Y: np.ndarray, X: np.ndarray) -> tuple:
        """Coherent Point Drift registration."""
        M, D = Y.shape
        N = X.shape[0]
        
        if N == 0:
            return Y.copy(), np.zeros((M, 1))
        
        if N > self.cpd_downsample:
            indices = np.random.choice(N, self.cpd_downsample, replace=False)
            X = X[indices]
            N = self.cpd_downsample
        
        sigma2 = np.sum((X[None, :, :] - Y[:, None, :]) ** 2) / (M * N * D)
        T = Y.copy()
        
        G = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * self.cpd_beta ** 2))
        
        for iteration in range(self.cpd_max_iter):
            dist2 = cdist(T, X, 'sqeuclidean')
            c = (2 * np.pi * sigma2) ** (D / 2) * self.cpd_w / (1 - self.cpd_w) * M / N
            
            P = np.exp(-dist2 / (2 * sigma2))
            den = P.sum(axis=0, keepdims=True) + c + 1e-10
            P = P / den
            
            P1 = P.sum(axis=1)
            PX = P @ X
            
            diag_P1_inv = np.diag(1.0 / (P1 + 1e-10))
            A = G + self.cpd_lambda * sigma2 * diag_P1_inv
            B = diag_P1_inv @ PX - Y
            
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            
            T_new = Y + G @ W
            
            diff = X[None, :, :] - T_new[:, None, :]
            sigma2_new = np.sum(P * np.sum(diff ** 2, axis=2)) / (np.sum(P) * D + 1e-10)
            sigma2_new = max(sigma2_new, 1e-6)
            
            change = np.max(np.abs(T_new - T))
            T = T_new
            sigma2 = sigma2_new
            
            if change < self.cpd_tol:
                break
        
        return T, P
    
    # ================================================================
    # GEOMETRY CONSTRAINTS
    # ================================================================
    
    def _joint_constraint_optimization_with_contour(
        self,
        keypoints: np.ndarray,
        point_cloud: np.ndarray,
        contour_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Joint edge length + surface projection optimization with contour constraint.

        Constraints (matches fabric_tracker):
        - Corner nodes (self.fixed_indices): Fixed
        - Border nodes (self.border_grid_indices): Snap to contour
        - Interior nodes: Soft projection to point cloud
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8

        if len(point_cloud) == 0:
            return keypoints

        # Build NN for point cloud projection
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        cloud_nn.fit(point_cloud)

        # Build NN for contour if available
        contour_nn = None
        if contour_3d is not None and len(contour_3d) > 0:
            contour_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
            contour_nn.fit(contour_3d)

        border_set = set(self.border_grid_indices) if self.border_grid_indices else set()

        for outer_iter in range(self.n_outer_iterations):
            # Edge correction (only valid edges, skip NaN nodes)
            for edge_iter in range(self.n_edge_iterations):
                for i, j in self.valid_edges:
                    if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                        continue

                    if i in self.fixed_indices and j in self.fixed_indices:
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

                        i_is_fixed = i in self.fixed_indices
                        j_is_fixed = j in self.fixed_indices

                        if not i_is_fixed and not j_is_fixed:
                            keypoints[i] += correction * direction
                            keypoints[j] -= correction * direction
                        elif not i_is_fixed:
                            keypoints[i] += 2 * correction * direction
                        elif not j_is_fixed:
                            keypoints[j] -= 2 * correction * direction

            # Snap border nodes to contour, interior nodes to point cloud
            for i in range(K):
                if np.any(np.isnan(keypoints[i])):
                    continue
                if i in self.fixed_indices:
                    continue  # Corners are fixed

                if i in border_set:
                    # Border nodes: snap to contour
                    if contour_nn is not None:
                        _, idx = contour_nn.kneighbors(keypoints[i:i + 1])
                        keypoints[i] = contour_3d[idx[0, 0]]
                else:
                    # Interior nodes: soft projection to point cloud
                    _, idx = cloud_nn.kneighbors(keypoints[i:i + 1])
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
        Track keypoints using CPD + geometry constraints.
        """
        t_start = time.time()
        
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

        # CPD registration — only on valid (non-NaN) nodes
        valid_mask = ~np.isnan(self.prev_keypoints[:, 0])
        valid_indices = np.where(valid_mask)[0]
        valid_prev = self.prev_keypoints[valid_indices]

        t_cpd_start = time.time()
        cpd_valid, _ = self._cpd_register(valid_prev, point_cloud)
        cpd_time = time.time() - t_cpd_start

        keypoints = self.prev_keypoints.copy()
        keypoints[valid_indices] = cpd_valid

        # Snap corner nodes to contour
        if contour_3d is not None and len(contour_3d) > 0 and self.contour_corner_grid_indices:
            for idx in self.contour_corner_grid_indices:
                if np.any(np.isnan(keypoints[idx])):
                    continue
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Snap border nodes to contour
        if contour_3d is not None and len(contour_3d) > 0 and self.border_grid_indices:
            for idx in self.border_grid_indices:
                if np.any(np.isnan(keypoints[idx])):
                    continue
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Geometry constraint optimization
        t_geom_start = time.time()
        keypoints = self._joint_constraint_optimization_with_contour(
            keypoints, point_cloud, contour_3d
        )
        geom_time = time.time() - t_geom_start

        # Final snap: corner + border nodes back to contour
        if contour_3d is not None and len(contour_3d) > 0:
            for idx in (self.contour_corner_grid_indices or []):
                if np.any(np.isnan(keypoints[idx])):
                    continue
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)
            for idx in (self.border_grid_indices or []):
                if np.any(np.isnan(keypoints[idx])):
                    continue
                keypoints[idx] = self._snap_to_contour_3d(keypoints[idx], contour_3d)

        # Update state
        self.prev_keypoints = keypoints.copy()
        self.frame_count += 1
        self.consecutive_skips = 0
        
        keypoints_2d = self._project_3d_to_2d(keypoints)
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
