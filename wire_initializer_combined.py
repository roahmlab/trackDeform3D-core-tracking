"""
WireInitializer: Frame 0 initialization for wire tracking.

Extracts the initialization logic from WireTracker for standalone use.
This class performs:
    1. Segmentation (background subtraction + depth threshold + top-K CC + skeletonization)
    2. Node identification (MST degree analysis)
    3. Topology pruning (prune to target branch/leaf count)
    4. FPS keypoint placement (with branch/leaf as anchors)
    5. EE pose injection (optional)
    6. Repulsion relaxation (spring-based uniform spacing)
    7. Topology construction (MST with skeleton path validation)
    8. Segment extraction (ordered edges for geometry correction)

Author: Auto-generated
Date: 2026-02-21
"""

import math
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors, KDTree
from typing import List, Tuple, Set, Dict, Optional


class WireInitializer:
    """
    Wire initialization for Frame 0.
    
    Performs segmentation + node identification + keypoint placement + topology extraction.
    
    Keypoint ordering: [branch_0, ..., branch_{N_b-1}, leaf_0, ..., leaf_{N_l-1}, intermediate...]
    
    Usage:
        initializer = WireInitializer(intrinsics)
        result = initializer.initialize(depth, arm_depth)
        if result['success']:
            keypoints = result['keypoints']
            edges = result['edges']
    """
    
    def __init__(
        self,
        intrinsics: np.ndarray,
        n_keypoints: int = 21,
        target_branch_nodes: int = 2,
        target_leaf_nodes: int = 4,
        # Segmentation parameters
        bg_threshold: float = 80.0,
        max_depth: float = 1000.0,
        top_k_components: int = 5,
        arm_dilation_pixels: int = 5,
        # Repulsion parameters
        repulsion_iterations: int = 500,
        repulsion_lr: float = 5.0,
        # Keypoint placement method
        placement_method: str = 'fps',
        # Minimum skeleton pixels
        min_skeleton_pixels: int = 100,
        # End-effector pose injection
        ee_poses_3d: np.ndarray = None,
    ):
        """
        Initialize WireInitializer.
        
        Args:
            intrinsics: 3×3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
            n_keypoints: Total number of keypoints to extract
            target_branch_nodes: Target number of branch nodes (pruning)
            target_leaf_nodes: Target number of leaf nodes (pruning)
            bg_threshold: Background subtraction distance threshold (mm)
            max_depth: Maximum valid depth (mm)
            top_k_components: Number of largest connected components to keep
            arm_dilation_pixels: Pixels to dilate arm mask
            repulsion_iterations: Repulsion relaxation iterations
            repulsion_lr: Repulsion learning rate
            min_skeleton_pixels: Minimum skeleton pixels to process
            ee_poses_3d: (n_frames, 2, 3) array of end-effector 3D positions
        """
        # Camera intrinsics
        self.intrinsics = np.array(intrinsics, dtype=np.float64)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        
        # Keypoint configuration
        self.n_keypoints = n_keypoints
        self.target_branch_nodes = target_branch_nodes
        self.target_leaf_nodes = target_leaf_nodes
        
        # Segmentation parameters
        self.bg_threshold = bg_threshold
        self.max_depth = max_depth
        self.top_k_components = top_k_components
        self.arm_dilation_pixels = arm_dilation_pixels
        
        # Repulsion parameters
        self.repulsion_iterations = repulsion_iterations
        self.repulsion_lr = repulsion_lr
        
        # Keypoint placement method
        self.placement_method = placement_method
        
        # Minimum skeleton pixels
        self.min_skeleton_pixels = min_skeleton_pixels
        
        # End-effector pose injection
        self.ee_poses_3d = ee_poses_3d  # (n_frames, 2, 3) or None
        self.ee_to_leaf_mapping = None  # {0: kp_idx, 1: kp_idx} set during initialization
        
        # Output state (set during initialization)
        self.reference_keypoints = None    # K × 3
        self.reference_edges = None        # List of (i, j)
        self.reference_lengths = None      # Array of edge lengths
        self.reference_n_branch = 0
        self.reference_n_leaf = 0
        self.segment_edges = None          # List of segment edge lists
        self.anchor_set = None             # Set of anchor keypoint indices
        self.free_leaf_indices = None      # List of free leaf indices
    
    # ================================================================
    # PHASE 1: SEGMENTATION
    # ================================================================
    
    def _depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to 3D point cloud."""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        z = depth.astype(np.float64)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.stack([x, y, z], axis=-1)
    
    def _expand_arm_depth(self, arm_depth: np.ndarray) -> np.ndarray:
        """Expand arm depth by dilating valid region and filling with nearest values."""
        if self.arm_dilation_pixels <= 0:
            return arm_depth.copy()
        
        arm_valid_mask = (arm_depth > 0).astype(np.uint8)
        kernel_size = 2 * self.arm_dilation_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        arm_valid_mask_dilated = cv2.dilate(arm_valid_mask, kernel, iterations=1)
        
        arm_depth_expanded = arm_depth.copy()
        new_pixels = (arm_valid_mask_dilated > 0) & (arm_valid_mask == 0)
        
        if np.any(new_pixels):
            dist, indices = ndimage.distance_transform_edt(
                arm_valid_mask == 0, return_indices=True
            )
            arm_depth_expanded[new_pixels] = arm_depth[
                indices[0][new_pixels], indices[1][new_pixels]
            ]
        
        return arm_depth_expanded
    
    def _background_subtraction(
        self, full_pc: np.ndarray, arm_pc: np.ndarray
    ) -> np.ndarray:
        """Remove robot arm via 3D point cloud distance."""
        diff = np.linalg.norm(full_pc - arm_pc, axis=-1)
        foreground_mask = (diff > self.bg_threshold).astype(np.uint8)
        return foreground_mask
    
    def _apply_depth_threshold(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Filter mask by valid depth range."""
        filtered = mask.copy()
        filtered[depth > self.max_depth] = 0
        filtered[depth <= 0] = 0
        filtered[np.isnan(depth)] = 0
        filtered[np.isinf(depth)] = 0
        return filtered
    
    def _get_top_k_components(self, mask: np.ndarray, k: int = None) -> np.ndarray:
        """Keep only the k largest connected components."""
        if k is None:
            k = self.top_k_components
        
        labeled, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return np.zeros_like(mask)
        
        component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        k = min(k, num_features)
        largest_labels = np.argsort(component_sizes)[::-1][:k] + 1
        
        return np.isin(labeled, largest_labels).astype(np.uint8)
    
    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Skeletonize binary mask."""
        return skeletonize(mask > 0).astype(np.uint8)
    
    def segment(self, depth: np.ndarray, arm_depth: np.ndarray = None,
                precomputed_arm_mask: np.ndarray = None) -> dict:
        """
        Phase 1: Create foreground and skeleton masks.
        
        For Frame 0 initialization, uses n_components=1 (single largest component).
        
        Args:
            depth: H × W current frame depth
            arm_depth: H × W arm-only depth (not needed if precomputed_arm_mask provided)
            precomputed_arm_mask: H × W binary mask where 1=arm (to be removed), 0=keep
        
        Returns:
            dict with 'foreground_mask', 'skeleton_mask', 'skeleton_pc', 'seg_time'
        """
        t_start = time.time()
        
        if precomputed_arm_mask is not None:
            # Use precomputed mask directly
            foreground_mask = ((precomputed_arm_mask == 0) & (depth > 0)).astype(np.uint8)
        else:
            # Compute from arm_depth
            arm_depth_expanded = self._expand_arm_depth(arm_depth)
            full_pc = self._depth_to_point_cloud(depth)
            arm_pc = self._depth_to_point_cloud(arm_depth_expanded)
            foreground_mask = self._background_subtraction(full_pc, arm_pc)
        
        # Depth thresholding
        foreground_mask = self._apply_depth_threshold(foreground_mask, depth)
        
        # Frame 0: use single largest connected component
        foreground_mask = self._get_top_k_components(foreground_mask, k=1)
        
        # Skeletonization
        skeleton_mask = self._skeletonize(foreground_mask)
        
        # Extract skeleton point cloud
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        
        seg_time = time.time() - t_start
        
        return {
            'foreground_mask': foreground_mask,
            'skeleton_mask': skeleton_mask,
            'skeleton_pc': skeleton_pc,
            'seg_time': seg_time,
        }
    
    # ================================================================
    # GEOMETRY UTILITIES
    # ================================================================
    
    def _pixel_to_3d(self, pixel_coords: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Back-project 2D pixel coordinates to 3D.
        
        IMPORTANT: Returns an array with the same length as pixel_coords.
        For pixels with invalid depth (out of bounds, zero, or too far),
        returns [0, 0, 0] as a placeholder to preserve index correspondence.
        """
        if len(pixel_coords) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        H, W = depth.shape
        n_points = len(pixel_coords)
        coords_3d = np.zeros((n_points, 3), dtype=np.float64)
        
        for i, (row, col) in enumerate(pixel_coords):
            row, col = int(row), int(col)
            if 0 <= row < H and 0 <= col < W:
                z = depth[row, col]
                if z > 0 and z < self.max_depth:
                    x = (col - self.cx) * z / self.fx
                    y = (row - self.cy) * z / self.fy
                    coords_3d[i] = [x, y, z]
                # else: coords_3d[i] stays [0, 0, 0]
            # else: coords_3d[i] stays [0, 0, 0]
        
        return coords_3d
    
    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D pixel coordinates."""
        if len(points_3d) == 0:
            return np.empty((0, 2), dtype=np.float64)
        
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        z_safe = np.maximum(z, 1e-6)
        
        u = (x * self.fx) / z_safe + self.cx  # col
        v = (y * self.fy) / z_safe + self.cy  # row
        
        return np.stack([v, u], axis=1)
    
    def _extract_point_cloud(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Extract 3D points from masked region."""
        valid = (mask > 0) & (depth > 0) & (depth < self.max_depth)
        rows, cols = np.where(valid)
        
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        z = depth[rows, cols].astype(np.float64)
        x = (cols - self.cx) * z / self.fx
        y = (rows - self.cy) * z / self.fy
        
        return np.stack([x, y, z], axis=1)
    
    # ================================================================
    # NODE IDENTIFICATION
    # ================================================================
    
    def _node_identification(self, skeleton_mask: np.ndarray) -> tuple:
        """
        Identify branch and leaf nodes via MST degree analysis.
        
        Returns:
            branch_2d: B × 2 branch node coords (row, col)
            leaf_2d: L × 2 leaf node coords (row, col)
            mst_adjacency: N × N MST adjacency matrix
            coords: N × 2 all skeleton coords
        """
        binary = skeleton_mask > 0
        coords = np.column_stack(np.nonzero(binary)).astype(np.int64)
        
        if coords.shape[0] == 0:
            empty = np.empty((0, 2), dtype=np.int64)
            return empty, empty, None, None
        
        if coords.shape[0] == 1:
            empty = np.empty((0, 2), dtype=np.int64)
            return empty, coords.copy(), None, coords
        
        # Build distance matrix
        dists = cdist(coords, coords, metric='euclidean')
        
        # 8-connected adjacency
        adjacency = np.where(dists <= np.sqrt(2) + 1e-6, dists, 0)
        np.fill_diagonal(adjacency, 0)
        
        # Build MST
        sparse_adj = csr_matrix(adjacency)
        mst = minimum_spanning_tree(sparse_adj)
        mst_dense = mst.toarray()
        mst_symmetric = mst_dense + mst_dense.T
        
        # Compute degrees
        degrees = (mst_symmetric > 0).sum(axis=1)
        
        # Classify
        branch_indices = np.where(degrees >= 3)[0]
        branch_2d = coords[branch_indices]
        
        leaf_indices = np.where(degrees == 1)[0]
        leaf_2d = coords[leaf_indices]
        
        return branch_2d, leaf_2d, mst_symmetric, coords
    
    def _prune_to_target_topology(
        self, adjacency: np.ndarray, coords: np.ndarray
    ) -> dict:
        """
        Prune MST to reach target branch/leaf count.
        
        Iteratively removes shortest leaf segments until target_leaf_nodes is reached.
        """
        if adjacency is None or coords is None or len(coords) == 0:
            return {
                "adjacency": None,
                "coords": np.empty((0, 2), dtype=np.int64),
                "branch_coords": np.empty((0, 2), dtype=np.int64),
                "leaf_coords": np.empty((0, 2), dtype=np.int64),
            }
        
        adjacency = np.array(adjacency, dtype=np.float64)
        coords = np.array(coords, dtype=np.int64)
        n = adjacency.shape[0]
        
        # Make symmetric
        adjacency = np.maximum(adjacency, adjacency.T)
        
        # Track active nodes
        active = np.ones(n, dtype=bool)
        max_iterations = n
        
        # Prune leaf segments
        for _ in range(max_iterations):
            degrees = np.zeros(n, dtype=np.int64)
            for i in range(n):
                if active[i]:
                    degrees[i] = np.sum((adjacency[i, :] > 0) & active)
            
            leaf_mask = (degrees == 1) & active
            num_leaves = np.sum(leaf_mask)
            
            if num_leaves <= self.target_leaf_nodes:
                break
            
            # Find shortest leaf segment
            leaf_indices = np.where(leaf_mask)[0]
            min_length = np.inf
            prune_idx = -1
            
            for leaf_idx in leaf_indices:
                current = leaf_idx
                path = [current]
                visited = {current}
                
                while True:
                    neighbors = np.where((adjacency[current, :] > 0) & active)[0]
                    neighbors = [nb for nb in neighbors if nb not in visited]
                    
                    if len(neighbors) == 0:
                        break
                    
                    next_node = neighbors[0]
                    path.append(next_node)
                    visited.add(next_node)
                    
                    deg = np.sum((adjacency[next_node, :] > 0) & active)
                    if deg >= 3 or deg == 1:
                        break
                    
                    current = next_node
                
                path_length = sum(
                    adjacency[path[j], path[j + 1]]
                    for j in range(len(path) - 1)
                )
                
                if path_length < min_length:
                    min_length = path_length
                    prune_idx = leaf_idx
            
            if prune_idx >= 0:
                active[prune_idx] = False
                adjacency[prune_idx, :] = 0
                adjacency[:, prune_idx] = 0
        
        # Extract results
        active_indices = np.where(active)[0]
        new_coords = coords[active_indices]
        new_adjacency = adjacency[np.ix_(active_indices, active_indices)]
        
        new_degrees = (new_adjacency > 0).sum(axis=1)
        branch_mask = new_degrees >= 3
        leaf_mask = new_degrees == 1
        
        return {
            "adjacency": new_adjacency,
            "coords": new_coords,
            "branch_coords": new_coords[branch_mask],
            "leaf_coords": new_coords[leaf_mask],
        }
    
    # ================================================================
    # SKELETON GRAPH UTILITIES
    # ================================================================
    
    def _get_skeleton_path(
        self,
        adjacency: List[List[Tuple[int, float]]],
        start: int,
        goal: int,
    ) -> List[int]:
        """
        Find the shortest path in skeleton graph from start to goal using BFS.
        
        Returns:
            List of skeleton pixel indices along the path (empty if no path).
        """
        if start == goal:
            return [start]
        if start < 0 or goal < 0:
            return []
        
        # BFS with parent tracking
        visited: Set[int] = set()
        parent: Dict[int, int] = {}
        queue = [start]
        visited.add(start)
        found = False
        
        while queue and not found:
            node = queue.pop(0)
            
            for nbr, _ in adjacency[node]:
                if nbr in visited:
                    continue
                parent[nbr] = node
                if nbr == goal:
                    found = True
                    break
                visited.add(nbr)
                queue.append(nbr)
        
        if not found:
            return []
        
        # Reconstruct path
        path = [goal]
        current = goal
        while current != start:
            current = parent[current]
            path.append(current)
        path.reverse()
        
        return path
    
    def compute_mst_skeleton_pixels(
        self,
        keypoints: np.ndarray,
        edges: List[Tuple[int, int]],
        skeleton_mask: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Count skeleton pixels covered by MST edges.
        
        For each edge in the MST, finds the skeleton path between the two
        keypoint locations and collects all unique skeleton pixels.
        
        Args:
            keypoints: K × 3 keypoints
            edges: List of (i, j) edge tuples (MST edges)
            skeleton_mask: H × W skeleton mask
        
        Returns:
            mst_pixel_count: Number of unique skeleton pixels in MST
            mst_skeleton_mask: H × W mask showing only MST skeleton pixels
        """
        # Build skeleton graph
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if skeleton_coords.shape[0] == 0:
            return 0, np.zeros_like(skeleton_mask)
        
        skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
        
        # Collect all unique skeleton pixel indices covered by MST edges
        mst_pixel_indices: Set[int] = set()
        
        for (i, j) in edges:
            # Project keypoint i to 2D and snap to skeleton
            x_i, y_i, z_i = keypoints[i]
            if z_i > 1e-6:
                col_i = int(round(x_i * self.fx / z_i + self.cx))
                row_i = int(round(y_i * self.fy / z_i + self.cy))
                skel_i = self._snap_to_skeleton((row_i, col_i), skeleton_coords, skeleton_tree)
            else:
                skel_i = -1
            
            # Project keypoint j to 2D and snap to skeleton
            x_j, y_j, z_j = keypoints[j]
            if z_j > 1e-6:
                col_j = int(round(x_j * self.fx / z_j + self.cx))
                row_j = int(round(y_j * self.fy / z_j + self.cy))
                skel_j = self._snap_to_skeleton((row_j, col_j), skeleton_coords, skeleton_tree)
            else:
                skel_j = -1
            
            # Find path on skeleton graph
            if skel_i >= 0 and skel_j >= 0:
                path = self._get_skeleton_path(adjacency, skel_i, skel_j)
                mst_pixel_indices.update(path)
        
        # Create MST skeleton mask
        H, W = skeleton_mask.shape
        mst_skeleton_mask = np.zeros((H, W), dtype=np.uint8)
        for idx in mst_pixel_indices:
            row, col = skeleton_coords[idx]
            mst_skeleton_mask[row, col] = 255
        
        return len(mst_pixel_indices), mst_skeleton_mask
    
    def _build_skeleton_graph(
        self, skeleton_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[List[Tuple[int, float]]]]:
        """Build a graph representation of the skeleton mask for path queries."""
        skeleton_coords = np.argwhere(skeleton_mask > 0)
        n_nodes = skeleton_coords.shape[0]
        
        if n_nodes == 0:
            return np.empty((0, 2), dtype=np.int64), {}, []
        
        coord_to_idx = {tuple(coord.tolist()): idx for idx, coord in enumerate(skeleton_coords)}
        
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        neighbor_offsets = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
            (0, -1, 1.0),                          (0, 1, 1.0),
            (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2)),
        ]
        
        for idx, (row, col) in enumerate(skeleton_coords):
            for dr, dc, dist in neighbor_offsets:
                neighbor = (row + dr, col + dc)
                neighbor_idx = coord_to_idx.get(neighbor)
                if neighbor_idx is not None:
                    adjacency[idx].append((neighbor_idx, dist))
        
        return skeleton_coords, coord_to_idx, adjacency
    
    def _prune_skeleton_to_node_paths(
        self,
        skeleton_mask: np.ndarray,
        branch_2d: np.ndarray,
        leaf_2d: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Prune skeleton mask to only keep pixels on the 5 MST segment paths.
        
        After EE injection, we have 6 final nodes: 2 branch + 4 leaf (2 EE + 2 free).
        The 5 segments are:
            - Each leaf → its nearest branch (4 paths)
            - Branch → branch trunk (1 path)
        
        Only skeleton pixels on these 5 paths are kept. Extra spurs are removed.
        
        Note: leaf_2d should already have EE leaves updated to their snapped-to-skeleton
        2D positions before calling this method.
        
        Args:
            skeleton_mask: H × W raw skeleton mask
            branch_2d: B × 2 branch node coords (row, col)
            leaf_2d: L × 2 leaf node coords (row, col), with EE leaves updated
        
        Returns:
            pruned_mask: H × W pruned skeleton mask
            n_before: pixel count before pruning
            n_after: pixel count after pruning
        """
        n_before = int(np.sum(skeleton_mask > 0))
        
        # Build skeleton graph from raw skeleton
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if skeleton_coords.shape[0] == 0:
            return skeleton_mask, n_before, n_before
        
        skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
        
        # Snap branch nodes to skeleton indices
        branch_skel_indices = []
        for (row, col) in branch_2d:
            idx = self._snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            branch_skel_indices.append(idx)
        
        # Snap leaf nodes to skeleton indices
        leaf_skel_indices = []
        for li, (row, col) in enumerate(leaf_2d):
            idx = self._snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            leaf_skel_indices.append(idx)
            snap_coord = skeleton_coords[idx]
            snap_dist = np.linalg.norm(np.array([row, col]) - snap_coord)
            print(f"    Leaf {li}: ({int(row)}, {int(col)}) -> skel_idx {idx} at ({snap_coord[0]}, {snap_coord[1]}), snap_dist={snap_dist:.1f}px")
        
        path_pixel_indices: Set[int] = set()
        
        # Find 4 leaf→nearest_branch paths
        for leaf_idx, leaf_skel in enumerate(leaf_skel_indices):
            best_path = None
            best_len = float('inf')
            
            for branch_idx, branch_skel in enumerate(branch_skel_indices):
                path = self._get_skeleton_path(adjacency, leaf_skel, branch_skel)
                if path:
                    # Use path pixel count as proxy for length
                    if len(path) < best_len:
                        best_len = len(path)
                        best_path = path
            
            if best_path is not None:
                path_pixel_indices.update(best_path)
                print(f"    Leaf {leaf_idx} -> branch: {len(best_path)} px path")
            else:
                print(f"    WARNING: Leaf {leaf_idx} has no path to any branch")
        
        # Find trunk path (branch → branch)
        if len(branch_skel_indices) == 2:
            trunk_path = self._get_skeleton_path(
                adjacency, branch_skel_indices[0], branch_skel_indices[1]
            )
            if trunk_path:
                path_pixel_indices.update(trunk_path)
                print(f"    Trunk: {len(trunk_path)} px path")
            else:
                print(f"    WARNING: No trunk path between branches")
        
        # Build pruned skeleton mask
        H, W = skeleton_mask.shape
        pruned_mask = np.zeros((H, W), dtype=np.uint8)
        for idx in path_pixel_indices:
            r, c = skeleton_coords[idx]
            if 0 <= r < H and 0 <= c < W:
                pruned_mask[r, c] = 1
        
        n_after = int(np.sum(pruned_mask > 0))
        
        return pruned_mask, n_before, n_after

    def _snap_to_skeleton(
        self, 
        pixel: Tuple[int, int], 
        skeleton_coords: np.ndarray,
        skeleton_tree: Optional[KDTree] = None
    ) -> int:
        """Find the nearest skeleton pixel index to a given pixel."""
        if skeleton_coords.shape[0] == 0:
            return -1
        
        row, col = pixel
        if skeleton_tree is not None:
            _, idx = skeleton_tree.query([[row, col]], k=1)
            return int(np.asarray(idx).reshape(-1)[0])
        else:
            diffs = skeleton_coords - np.array([row, col])
            dists_sq = np.sum(diffs * diffs, axis=1)
            return int(np.argmin(dists_sq))
    
    def _skeleton_path_exists(
        self,
        adjacency: List[List[Tuple[int, float]]],
        start: int,
        goal: int,
        blocked: Set[int],
    ) -> bool:
        """Check if a path exists in skeleton graph from start to goal."""
        if start == goal:
            return True
        if start < 0 or goal < 0:
            return False
        
        # BFS
        visited: Set[int] = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            node = queue.pop(0)
            
            for nbr, _ in adjacency[node]:
                if nbr == goal:
                    return True
                if nbr in visited:
                    continue
                if nbr in blocked:
                    continue
                visited.add(nbr)
                queue.append(nbr)
        
        return False
    
    def _get_keypoint_blocked_regions(
        self,
        keypoints: np.ndarray,
        skeleton_coords: np.ndarray,
        skeleton_tree: Optional[KDTree],
        block_radius: int = 3,
    ) -> Tuple[np.ndarray, List[Set[int]]]:
        """For each keypoint, find its skeleton index and blocked region."""
        K = keypoints.shape[0]
        keypoint_skel_indices = np.zeros(K, dtype=int)
        blocked_regions: List[Set[int]] = []
        
        skeleton_coords_float = skeleton_coords.astype(np.float64)
        
        for i in range(K):
            # Project 3D keypoint to 2D
            x, y, z = keypoints[i]
            if z <= 1e-6:
                keypoint_skel_indices[i] = -1
                blocked_regions.append(set())
                continue
            
            col = int(round(x * self.fx / z + self.cx))
            row = int(round(y * self.fy / z + self.cy))
            
            # Snap to nearest skeleton pixel
            skel_idx = self._snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            keypoint_skel_indices[i] = skel_idx
            
            if skel_idx < 0:
                blocked_regions.append(set())
                continue
            
            # Find all skeleton pixels within block_radius
            center = skeleton_coords_float[skel_idx]
            diffs = skeleton_coords_float - center
            dists_sq = np.sum(diffs * diffs, axis=1)
            within_radius = np.where(dists_sq <= block_radius * block_radius)[0]
            blocked_regions.append(set(within_radius.tolist()))
        
        return keypoint_skel_indices, blocked_regions
    
    # ================================================================
    # FPS + REPULSION
    # ================================================================
    
    def _compute_skeleton_segment_structure(
        self,
        branch_2d: np.ndarray,
        leaf_2d: np.ndarray,
        skeleton_mask: np.ndarray,
        depth: np.ndarray,
    ) -> dict:
        """
        Compute segment structure from skeleton paths between branch/leaf nodes.
        
        Returns:
            dict with:
                - segment_paths: List of skeleton pixel index lists for each segment
                - segment_3d_lengths: List of 3D lengths for each segment
                - segment_endpoints: List of (start_type, end_type) for each segment
                - skeleton_coords: N × 2 skeleton pixel coords
                - skeleton_3d: N × 3 skeleton 3D points
                - adjacency: skeleton graph adjacency
        """
        # Build skeleton graph
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if skeleton_coords.shape[0] == 0:
            return None
        
        # Convert skeleton to 3D
        skeleton_3d = self._pixel_to_3d(skeleton_coords, depth)
        
        skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
        
        # Snap branch/leaf nodes to skeleton indices
        branch_skel_indices = []
        for bi, (row, col) in enumerate(branch_2d):
            idx = self._snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            branch_skel_indices.append(idx)
            snap_dist = np.linalg.norm(skeleton_coords[idx] - np.array([row, col]))
            print(f"    Branch {bi}: ({row}, {col}) -> skel_idx {idx} (snap_dist={snap_dist:.1f}px)")
        
        leaf_skel_indices = []
        for li, (row, col) in enumerate(leaf_2d):
            idx = self._snap_to_skeleton((row, col), skeleton_coords, skeleton_tree)
            leaf_skel_indices.append(idx)
            snap_dist = np.linalg.norm(skeleton_coords[idx] - np.array([row, col]))
            print(f"    Leaf {li}: ({row}, {col}) -> skel_idx {idx} (snap_dist={snap_dist:.1f}px)")
        
        # Find paths for each segment (leaf → branch, branch → branch)
        # Store all paths with their info for later reordering
        all_leaf_segments = []  # List of (leaf_idx, branch_idx, path, length)
        
        # For each leaf, find path to nearest branch
        leaf_to_branch = {}
        for leaf_idx, leaf_skel in enumerate(leaf_skel_indices):
            best_path = None
            best_branch = -1
            best_len = float('inf')
            
            for branch_idx, branch_skel in enumerate(branch_skel_indices):
                path = self._get_skeleton_path(adjacency, leaf_skel, branch_skel)
                if path:
                    path_len = self._compute_path_3d_length(path, skeleton_3d)
                    if path_len < best_len:
                        best_len = path_len
                        best_path = path
                        best_branch = branch_idx
                else:
                    print(f"    WARNING: No path from leaf {leaf_idx} (skel={leaf_skel}) to branch {branch_idx} (skel={branch_skel})")
            
            if best_path is not None:
                all_leaf_segments.append({
                    'leaf_idx': leaf_idx,
                    'branch_idx': best_branch,
                    'path': best_path,
                    'length': best_len,
                })
                leaf_to_branch[leaf_idx] = best_branch
                print(f"    Leaf {leaf_idx} -> Branch {best_branch}: path_len={best_len:.1f}mm, path_size={len(best_path)}")
            else:
                print(f"    ERROR: Leaf {leaf_idx} has NO path to any branch!")
        
        # Find trunk path (between branches)
        trunk_segment = None
        if len(branch_skel_indices) == 2:
            trunk_path = self._get_skeleton_path(
                adjacency, branch_skel_indices[0], branch_skel_indices[1]
            )
            if trunk_path:
                trunk_length = self._compute_path_3d_length(trunk_path, skeleton_3d)
                trunk_segment = {
                    'path': trunk_path,
                    'length': trunk_length,
                    'branch_0': 0,
                    'branch_1': 1,
                }
                print(f"    Trunk (Branch 0 -> Branch 1): path_len={trunk_length:.1f}mm, path_size={len(trunk_path)}")
            else:
                print(f"    WARNING: No trunk path between branches!")
        
        return {
            'all_leaf_segments': all_leaf_segments,
            'trunk_segment': trunk_segment,
            'skeleton_coords': skeleton_coords,
            'skeleton_3d': skeleton_3d,
            'adjacency': adjacency,
            'branch_skel_indices': branch_skel_indices,
            'leaf_skel_indices': leaf_skel_indices,
            'leaf_to_branch': leaf_to_branch,
        }
    
    def _compute_path_3d_length(
        self,
        path: List[int],
        skeleton_3d: np.ndarray,
    ) -> float:
        """Compute total 3D length along a skeleton path."""
        if len(path) < 2:
            return 0.0
        
        total_len = 0.0
        for i in range(len(path) - 1):
            p1 = skeleton_3d[path[i]]
            p2 = skeleton_3d[path[i + 1]]
            total_len += np.linalg.norm(p2 - p1)
        
        return total_len
    
    # =========================================================================
    # NEW STREAMLINED METHODS for unified Build Topology phase
    # =========================================================================
    
    def _estimate_segment_length(
        self,
        path: List[int],
        skeleton_3d: np.ndarray,
        downsample_factor: int = 10,
    ) -> float:
        """
        Estimate segment length using downsampled path.
        
        Always includes first and last points. Subsamples every `downsample_factor`
        points in between for efficiency.
        
        Args:
            path: List of skeleton indices forming the path
            skeleton_3d: N×3 array of skeleton 3D coordinates
            downsample_factor: Sample every Nth point (default 10)
            
        Returns:
            Estimated length in mm (summing euclidean distances along downsampled path)
        """
        if len(path) < 2:
            return 0.0
        
        # Downsample but always keep first and last
        indices = [0]
        for i in range(downsample_factor, len(path) - 1, downsample_factor):
            indices.append(i)
        indices.append(len(path) - 1)
        
        # Remove duplicates (if path is very short)
        indices = sorted(set(indices))
        
        # Compute length
        total_len = 0.0
        for i in range(len(indices) - 1):
            p1 = skeleton_3d[path[indices[i]]]
            p2 = skeleton_3d[path[indices[i + 1]]]
            total_len += np.linalg.norm(p2 - p1)
        
        return total_len
    
    def _fps_on_segment(
        self,
        segment_3d: np.ndarray,
        n_points: int,
        anchor_seeds: List[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Farthest Point Sampling on a single segment.
        
        Args:
            segment_3d: M×3 array of 3D points along the segment
            n_points: Number of points to sample
            anchor_seeds: List of 3D points to use as initial seeds (e.g., start/end anchors)
            
        Returns:
            n_points×3 array of sampled points
        """
        if len(segment_3d) == 0:
            return np.zeros((n_points, 3))
        
        if n_points <= 0:
            return np.zeros((0, 3))
        
        # If we have fewer points than requested, return what we have
        if len(segment_3d) <= n_points:
            return segment_3d[:n_points].copy()
        
        # Initialize selected points
        selected = []
        
        # If anchors provided, add them first
        if anchor_seeds is not None:
            for anchor in anchor_seeds:
                selected.append(anchor)
        
        # If we already have enough from anchors
        if len(selected) >= n_points:
            return np.array(selected[:n_points])
        
        # Initialize distances to infinity
        min_dist = np.full(len(segment_3d), np.inf)
        
        # Update distances from anchor seeds
        for anchor in selected:
            dists = np.linalg.norm(segment_3d - anchor, axis=1)
            min_dist = np.minimum(min_dist, dists)
        
        # If no anchors, start with first point
        if len(selected) == 0:
            selected.append(segment_3d[0])
            min_dist = np.linalg.norm(segment_3d - segment_3d[0], axis=1)
        
        # FPS iterations
        while len(selected) < n_points:
            # Find point with maximum minimum distance
            farthest_idx = np.argmax(min_dist)
            farthest_pt = segment_3d[farthest_idx]
            selected.append(farthest_pt)
            
            # Update distances
            dists = np.linalg.norm(segment_3d - farthest_pt, axis=1)
            min_dist = np.minimum(min_dist, dists)
        
        return np.array(selected)
    
    def _fps_on_segment_with_indices(
        self,
        segment_3d: np.ndarray,
        n_points: int,
        anchor_seeds: List[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Farthest Point Sampling on a segment, returning both points and path indices.
        
        Args:
            segment_3d: M×3 array of 3D points along the segment (in path order)
            n_points: Number of points to sample
            anchor_seeds: List of 3D points to use as initial seeds
            
        Returns:
            Tuple of:
                - n_points×3 array of sampled points
                - List of path indices for each sampled point
        """
        if len(segment_3d) == 0:
            return np.zeros((n_points, 3)), list(range(n_points))
        
        if n_points <= 0:
            return np.zeros((0, 3)), []
        
        # If we have fewer points than requested, return what we have
        if len(segment_3d) <= n_points:
            return segment_3d[:n_points].copy(), list(range(min(n_points, len(segment_3d))))
        
        # Initialize selected points and their path indices
        selected = []
        selected_indices = []
        
        # If anchors provided, find their nearest path indices
        if anchor_seeds is not None:
            for anchor in anchor_seeds:
                dists = np.linalg.norm(segment_3d - anchor, axis=1)
                nearest_idx = np.argmin(dists)
                selected.append(segment_3d[nearest_idx])
                selected_indices.append(nearest_idx)
        
        # If we already have enough from anchors
        if len(selected) >= n_points:
            return np.array(selected[:n_points]), selected_indices[:n_points]
        
        # Initialize distances to infinity
        min_dist = np.full(len(segment_3d), np.inf)
        
        # Update distances from anchor seeds
        for anchor in selected:
            dists = np.linalg.norm(segment_3d - anchor, axis=1)
            min_dist = np.minimum(min_dist, dists)
        
        # If no anchors, start with first point
        if len(selected) == 0:
            selected.append(segment_3d[0])
            selected_indices.append(0)
            min_dist = np.linalg.norm(segment_3d - segment_3d[0], axis=1)
        
        # FPS iterations
        selected_set = set(selected_indices)
        while len(selected) < n_points:
            # Mask already selected points
            masked_dist = min_dist.copy()
            for idx in selected_set:
                masked_dist[idx] = -1
            
            # Find point with maximum minimum distance
            farthest_idx = np.argmax(masked_dist)
            farthest_pt = segment_3d[farthest_idx]
            selected.append(farthest_pt)
            selected_indices.append(farthest_idx)
            selected_set.add(farthest_idx)
            
            # Update distances
            dists = np.linalg.norm(segment_3d - farthest_pt, axis=1)
            min_dist = np.minimum(min_dist, dists)
        
        return np.array(selected), selected_indices
    
    def _connect_keypoints_sequential(
        self,
        start_idx: int,
        end_idx: int,
        intermediate_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Connect keypoints in a chain sequentially (assumes already sorted by path order).
        
        Args:
            start_idx: Keypoint index to start the chain
            end_idx: Keypoint index to end the chain
            intermediate_indices: List of intermediate keypoint indices (already in path order)
            
        Returns:
            List of (i, j) edges forming the chain
        """
        edges = []
        
        if not intermediate_indices:
            # Direct connection
            edges.append((start_idx, end_idx))
            return edges
        
        # Connect start -> intermediate[0]
        edges.append((start_idx, intermediate_indices[0]))
        
        # Connect intermediate[i] -> intermediate[i+1]
        for i in range(len(intermediate_indices) - 1):
            edges.append((intermediate_indices[i], intermediate_indices[i + 1]))
        
        # Connect intermediate[-1] -> end
        edges.append((intermediate_indices[-1], end_idx))
        
        return edges

    def _connect_keypoints_greedy(
        self,
        keypoints: np.ndarray,
        start_idx: int,
        end_idx: int,
        intermediate_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Connect keypoints in a chain using greedy nearest-neighbor.
        
        Starts from start_idx, greedily picks nearest unvisited intermediate,
        repeat until all intermediates visited, then connect to end_idx.
        
        Args:
            keypoints: N×3 array of all keypoints
            start_idx: Keypoint index to start the chain
            end_idx: Keypoint index to end the chain
            intermediate_indices: List of intermediate keypoint indices
            
        Returns:
            List of (i, j) edges forming the chain
        """
        edges = []
        
        if not intermediate_indices:
            # Direct connection
            edges.append((start_idx, end_idx))
            return edges
        
        current = start_idx
        remaining = set(intermediate_indices)
        
        while remaining:
            # Find nearest unvisited intermediate
            best_dist = np.inf
            best_idx = None
            for idx in remaining:
                dist = np.linalg.norm(keypoints[current] - keypoints[idx])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            edges.append((current, best_idx))
            remaining.remove(best_idx)
            current = best_idx
        
        # Connect to end
        edges.append((current, end_idx))
        
        return edges
    
    def _build_topology(
        self,
        branch_3d: np.ndarray,
        leaf_3d: np.ndarray,
        branch_2d: np.ndarray,
        leaf_2d: np.ndarray,
        skeleton_mask: np.ndarray,
        depth: np.ndarray,
        ee_to_leaf_kp: Dict[int, int],
        n_keypoints_per_segment: List[int] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[Tuple[int, int]]], List[dict]]:
        """
        Unified Build Topology phase (replaces phases 5b, 6, 7, 8).
        
        Steps:
            6.1: Find MST paths between nodes using BFS, order by EE mapping
            6.2: Estimate segment lengths using downsampled path (factor=10)
            6.3: Allocate keypoints proportionally or use explicit n_keypoints_per_segment
            6.4: FPS per segment with anchor seeds
            6.5: Connect keypoints using greedy nearest-neighbor chain
        
        Args:
            branch_3d: B×3 branch node positions
            leaf_3d: L×3 leaf node positions
            branch_2d: B×2 branch node pixel coords
            leaf_2d: L×2 leaf node pixel coords
            skeleton_mask: H×W binary skeleton
            depth: H×W depth image
            ee_to_leaf_kp: {ee_idx: keypoint_idx} mapping
            n_keypoints_per_segment: Optional explicit keypoints per segment
            
        Returns:
            keypoints: N×3 keypoint positions
            edges: List of (i, j) edges
            segment_edges: List of edge lists per segment
            ordered_segments: Segment metadata
        """
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        
        print("  Step 6.1: Finding MST paths and ordering segments...")
        
        # Compute segment structure (paths between nodes on skeleton)
        segment_structure = self._compute_skeleton_segment_structure(
            branch_2d, leaf_2d, skeleton_mask, depth
        )
        
        if segment_structure is None:
            raise ValueError("Failed to compute segment structure")
        
        # Order segments by EE mapping
        ordered_segments, b_ee0, b_ee1 = self._order_segments_by_ee(
            segment_structure, ee_to_leaf_kp, n_branch
        )
        
        print(f"    Found {len(ordered_segments)} segments")
        
        # Step 6.2: Estimate segment lengths using downsampled path
        print("  Step 6.2: Estimating segment lengths (downsampled)...")
        skeleton_3d = segment_structure['skeleton_3d']
        segment_lengths = []
        for seg in ordered_segments:
            length = self._estimate_segment_length(
                seg['path'], skeleton_3d, downsample_factor=10
            )
            segment_lengths.append(length)
            seg['estimated_length'] = length
        
        total_length = sum(segment_lengths)
        print(f"    Segment lengths: {[f'{l:.1f}' for l in segment_lengths]} mm")
        print(f"    Total estimated length: {total_length:.1f} mm")
        
        # Step 6.3: Allocate keypoints per segment
        print("  Step 6.3: Allocating keypoints per segment...")
        n_anchors = n_branch + n_leaf
        n_intermediate = self.n_keypoints - n_anchors
        n_segments = len(ordered_segments)
        
        if n_keypoints_per_segment is not None:
            # Use explicit allocation
            kp_per_seg = list(n_keypoints_per_segment)
            assert len(kp_per_seg) == n_segments, f"Expected {n_segments} segments, got {len(kp_per_seg)}"
        else:
            # Proportional allocation based on segment lengths
            if total_length > 0:
                # Allocate proportionally, minimum 0 per segment
                kp_per_seg = []
                for length in segment_lengths:
                    proportion = length / total_length
                    kp_per_seg.append(max(0, int(round(n_intermediate * proportion))))
                
                # Adjust to ensure total matches
                diff = n_intermediate - sum(kp_per_seg)
                if diff != 0:
                    # Add/subtract from longest segments
                    sorted_indices = sorted(range(n_segments), key=lambda i: segment_lengths[i], reverse=True)
                    for i in range(abs(diff)):
                        idx = sorted_indices[i % n_segments]
                        kp_per_seg[idx] += 1 if diff > 0 else -1
                        kp_per_seg[idx] = max(0, kp_per_seg[idx])
            else:
                # Equal distribution if lengths are zero
                base = n_intermediate // n_segments
                kp_per_seg = [base] * n_segments
                for i in range(n_intermediate % n_segments):
                    kp_per_seg[i] += 1
        
        print(f"    Intermediate keypoints per segment: {kp_per_seg}")
        print(f"    Total intermediate: {sum(kp_per_seg)} (target: {n_intermediate})")
        
        # Step 6.4: FPS per segment with anchor seeds
        print("  Step 6.4: Running FPS per segment...")
        
        # CRITICAL: Snap EE leaf positions to nearest skeleton point
        # The EE gripper position may be off the skeleton surface
        mst_indices_set = set()
        for seg in ordered_segments:
            mst_indices_set.update(seg['path'])
        mst_skeleton_3d = skeleton_3d[sorted(mst_indices_set)]
        
        leaf_3d_snapped = leaf_3d.copy()
        if self.ee_poses_3d is not None and self.ee_to_leaf_mapping:
            for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
                leaf_local_idx = kp_idx - n_branch
                ee_pos = leaf_3d[leaf_local_idx]
                
                # Find nearest MST skeleton point
                dists = np.linalg.norm(mst_skeleton_3d - ee_pos, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_skel_pos = mst_skeleton_3d[nearest_idx]
                snap_dist = dists[nearest_idx]
                
                print(f"    EE {ee_idx} (leaf {leaf_local_idx}): snap to skeleton, dist={snap_dist:.2f}mm")
                leaf_3d_snapped[leaf_local_idx] = nearest_skel_pos
        
        # Initialize keypoints array: [branches, leaves, intermediates...]
        keypoints = np.zeros((self.n_keypoints, 3))
        keypoints[:n_branch] = branch_3d
        keypoints[n_branch:n_anchors] = leaf_3d_snapped  # Use snapped positions!
        
        # Track intermediate keypoint indices
        next_intermediate_idx = n_anchors
        segment_intermediate_indices = []  # For each segment, list of intermediate kp indices
        
        for seg_idx, seg in enumerate(ordered_segments):
            n_intermediate_seg = kp_per_seg[seg_idx]
            
            # Get 3D points along segment path
            path = seg['path']
            segment_3d_points = skeleton_3d[path]
            
            # Get anchor positions (start and end of segment)
            start_kp = seg['start_kp']
            end_kp = seg['end_kp']
            start_pos = keypoints[start_kp]
            end_pos = keypoints[end_kp]
            
            # FPS with anchors as seeds, track path indices for ordering
            if n_intermediate_seg > 0:
                fps_points, fps_path_indices = self._fps_on_segment_with_indices(
                    segment_3d_points,
                    n_points=n_intermediate_seg + 2,  # Include anchors
                    anchor_seeds=[start_pos, end_pos]
                )
                
                # Extract intermediate points with their path indices (exclude anchor duplicates)
                intermediate_pts_with_path = []
                for pt, path_idx in zip(fps_points, fps_path_indices):
                    is_anchor = (np.linalg.norm(pt - start_pos) < 1e-3 or 
                                 np.linalg.norm(pt - end_pos) < 1e-3)
                    if not is_anchor:
                        intermediate_pts_with_path.append((pt, path_idx))
                
                # Take up to n_intermediate_seg points
                intermediate_pts_with_path = intermediate_pts_with_path[:n_intermediate_seg]
                
                # Sort by path index to ensure correct ordering along segment
                intermediate_pts_with_path.sort(key=lambda x: x[1])
                
                # Check if path order matches segment direction (start_kp -> end_kp)
                # If first point is closer to end than start, reverse the order
                if len(intermediate_pts_with_path) > 0:
                    first_pt = intermediate_pts_with_path[0][0]
                    dist_to_start = np.linalg.norm(first_pt - start_pos)
                    dist_to_end = np.linalg.norm(first_pt - end_pos)
                    if dist_to_end < dist_to_start:
                        # Path order is opposite to segment edge direction, reverse
                        intermediate_pts_with_path.reverse()
                
                # Assign to keypoints array (now in correct edge order)
                seg_intermediate_indices = []
                for pt, _ in intermediate_pts_with_path:
                    keypoints[next_intermediate_idx] = pt
                    seg_intermediate_indices.append(next_intermediate_idx)
                    next_intermediate_idx += 1
                
                segment_intermediate_indices.append(seg_intermediate_indices)
            else:
                segment_intermediate_indices.append([])
        
        # Step 6.5: Connect keypoints sequentially (already sorted by path order)
        print("  Step 6.5: Connecting keypoints (path-ordered)...")
        
        all_edges = []
        segment_edges = []
        
        for seg_idx, seg in enumerate(ordered_segments):
            start_kp = seg['start_kp']
            end_kp = seg['end_kp']
            intermediate_indices = segment_intermediate_indices[seg_idx]
            
            # Connect sequentially: start -> intermediate[0] -> ... -> intermediate[n-1] -> end
            seg_edges = self._connect_keypoints_sequential(
                start_kp, end_kp, intermediate_indices
            )
            
            all_edges.extend(seg_edges)
            segment_edges.append(seg_edges)
            
            seg_type = seg['type']
            print(f"    Segment {seg_idx} ({seg_type}): {len(seg_edges)} edges, {len(intermediate_indices)} intermediate")
        
        print(f"    Total: {len(all_edges)} edges across {len(segment_edges)} segments")
        
        return keypoints, all_edges, segment_edges, ordered_segments
    
    def _establish_ee_mapping_from_leaves(
        self,
        leaf_3d: np.ndarray,
        n_branch: int,
    ) -> Dict[int, int]:
        """
        Establish EE to leaf mapping based on EE poses and leaf positions.
        
        Returns:
            ee_to_leaf_kp: {ee_idx: leaf_keypoint_idx}
        """
        if self.ee_poses_3d is None:
            return {}
        
        ee_positions = self.ee_poses_3d[0]  # Frame 0, shape (2, 3)
        
        # Compute distances from each EE to each leaf
        cost_matrix = cdist(ee_positions, leaf_3d)
        
        # Hungarian algorithm for optimal assignment
        ee_idx_arr, leaf_local_idx_arr = linear_sum_assignment(cost_matrix)
        
        ee_to_leaf_kp = {}
        for ee_idx, leaf_local_idx in zip(ee_idx_arr, leaf_local_idx_arr):
            kp_idx = n_branch + leaf_local_idx  # leaf keypoint index
            ee_to_leaf_kp[ee_idx] = kp_idx
        
        return ee_to_leaf_kp
    
    def _order_segments_by_ee(
        self,
        segment_structure: dict,
        ee_to_leaf_kp: Dict[int, int],
        n_branch: int,
    ) -> List[dict]:
        """
        Order segments based on EE mapping.
        
        Final order:
            Segment 0: ee_leaf with EE 0 → its branch (b_ee0)
            Segment 1: ee_leaf with EE 1 → its branch (b_ee1)
            Segment 2: free_leaf connected to b_ee0
            Segment 3: free_leaf connected to b_ee1
            Segment 4: trunk (b_ee0 → b_ee1)
        
        Returns:
            List of segment dicts in correct order, each with:
                - path: skeleton path
                - length: 3D length
                - type: 'ee_leaf', 'free_leaf', or 'trunk'
                - start_kp: start keypoint index
                - end_kp: end keypoint index
        """
        all_leaf_segments = segment_structure['all_leaf_segments']
        trunk_segment = segment_structure['trunk_segment']
        leaf_to_branch = segment_structure['leaf_to_branch']
        
        # Build reverse mapping: leaf_kp → ee_idx
        leaf_kp_to_ee = {v: k for k, v in ee_to_leaf_kp.items()}
        
        # Find which branch each EE's leaf connects to
        ee_to_branch = {}
        for ee_idx, leaf_kp in ee_to_leaf_kp.items():
            leaf_local_idx = leaf_kp - n_branch
            if leaf_local_idx in leaf_to_branch:
                ee_to_branch[ee_idx] = leaf_to_branch[leaf_local_idx]
        
        b_ee0 = ee_to_branch.get(0, 0)
        b_ee1 = ee_to_branch.get(1, 1 if len(segment_structure['branch_skel_indices']) > 1 else 0)
        
        # Categorize leaf segments
        ee_leaf_segments = {}  # ee_idx → segment_info
        free_leaf_segments_by_branch = {}  # branch_idx → list of segment_info

        for seg_info in all_leaf_segments:
            leaf_idx = seg_info['leaf_idx']
            branch_idx = seg_info['branch_idx']
            leaf_kp = n_branch + leaf_idx

            if leaf_kp in leaf_kp_to_ee:
                ee_idx = leaf_kp_to_ee[leaf_kp]
                ee_leaf_segments[ee_idx] = {
                    'path': seg_info['path'],
                    'length': seg_info['length'],
                    'type': 'ee_leaf',
                    'start_kp': leaf_kp,
                    'end_kp': branch_idx,  # branch keypoint index
                    'ee_idx': ee_idx,
                }
            else:
                if branch_idx not in free_leaf_segments_by_branch:
                    free_leaf_segments_by_branch[branch_idx] = []
                free_leaf_segments_by_branch[branch_idx].append({
                    'path': seg_info['path'],
                    'length': seg_info['length'],
                    'type': 'free_leaf',
                    'start_kp': leaf_kp,
                    'end_kp': branch_idx,
                    'leaf_local_idx': leaf_idx,
                })

        # Sort free leaves per branch by length (longest first) for consistent ordering
        for branch_idx in free_leaf_segments_by_branch:
            free_leaf_segments_by_branch[branch_idx].sort(
                key=lambda s: s['length'], reverse=True
            )

        # Build ordered segment list
        ordered_segments = []

        # EE leaf segments first
        if 0 in ee_leaf_segments:
            ordered_segments.append(ee_leaf_segments[0])
        if 1 in ee_leaf_segments:
            ordered_segments.append(ee_leaf_segments[1])

        # All free leaves connected to b_ee0
        if b_ee0 in free_leaf_segments_by_branch:
            ordered_segments.extend(free_leaf_segments_by_branch[b_ee0])

        # All free leaves connected to b_ee1
        if b_ee1 in free_leaf_segments_by_branch and b_ee1 != b_ee0:
            ordered_segments.extend(free_leaf_segments_by_branch[b_ee1])

        # Trunk last
        if trunk_segment is not None:
            ordered_segments.append({
                'path': trunk_segment['path'],
                'length': trunk_segment['length'],
                'type': 'trunk',
                'start_kp': b_ee0,
                'end_kp': b_ee1,
            })
        
        return ordered_segments, b_ee0, b_ee1
    
    def _allocate_keypoints_per_segment(
        self,
        segment_3d_lengths: List[float],
        n_total_keypoints: int,
        n_anchor_keypoints: int,
    ) -> List[int]:
        """
        Allocate interior keypoints to each segment based on edge-count proportion.
        
        The key insight: edge length directly corresponds to segment length.
        Total edges in a tree = n_total_keypoints - 1. We distribute edges
        proportionally by segment 3D length, then derive interior keypoints
        as edges_per_segment - 1 (since each segment already has 2 endpoints).
        
        Args:
            segment_3d_lengths: 3D length of each segment
            n_total_keypoints: Total keypoints to place
            n_anchor_keypoints: Number of anchor keypoints (branch + leaf) [unused, kept for API]
        
        Returns:
            List of interior keypoint counts per segment
        """
        n_segments = len(segment_3d_lengths)
        total_edges = n_total_keypoints - 1  # Tree has K-1 edges for K keypoints
        
        if total_edges <= 0:
            return [0] * n_segments
        
        total_length = sum(segment_3d_lengths)
        if total_length == 0:
            # Equal distribution of edges
            base = total_edges // n_segments
            remainder = total_edges % n_segments
            edges_per_segment = [base + (1 if i < remainder else 0) for i in range(n_segments)]
        else:
            # Proportional allocation of edges by segment length
            edges_per_segment = []
            for seg_len in segment_3d_lengths:
                proportion = seg_len / total_length
                n_edges = max(1, round(proportion * total_edges))  # At least 1 edge per segment
                edges_per_segment.append(n_edges)
            
            # Adjust to ensure sum equals total_edges
            diff = total_edges - sum(edges_per_segment)
            if diff != 0:
                # Sort segments by proportional error, prefer longer segments for adjustment
                errors = []
                for i, (seg_len, n_edges) in enumerate(zip(segment_3d_lengths, edges_per_segment)):
                    ideal = (seg_len / total_length) * total_edges
                    errors.append((abs(ideal - n_edges), -seg_len, i))
                errors.sort(reverse=True)
                
                for _, _, idx in errors:
                    if diff > 0:
                        edges_per_segment[idx] += 1
                        diff -= 1
                    elif diff < 0:
                        if edges_per_segment[idx] > 1:  # Keep at least 1 edge
                            edges_per_segment[idx] -= 1
                            diff += 1
                    if diff == 0:
                        break
        
        # Interior keypoints = edges - 1 (each segment has 2 endpoints from anchors)
        interior_per_segment = [max(0, e - 1) for e in edges_per_segment]
        
        return interior_per_segment
    
    def _place_keypoints_along_segments(
        self,
        segment_structure: dict,
        interior_per_segment: List[int],
        branch_3d: np.ndarray,
        leaf_3d: np.ndarray,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Place keypoints uniformly along each segment.
        
        Keypoint ordering: [branch_0, ..., branch_{N_b-1}, leaf_0, ..., leaf_{N_l-1}, interior...]
        
        Returns:
            keypoints: K × 3 keypoints
            segment_edges: List of edge lists per segment
        """
        skeleton_3d = segment_structure['skeleton_3d']
        segment_paths = segment_structure['segment_paths']
        segment_endpoints = segment_structure['segment_endpoints']
        
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        
        # Start with anchor keypoints
        keypoints_list = []
        keypoints_list.extend(branch_3d.tolist())
        keypoints_list.extend(leaf_3d.tolist())
        
        # Track keypoint indices for each segment endpoint
        # branch_kp_idx[i] = keypoint index for branch i
        branch_kp_idx = list(range(n_branch))
        # leaf_kp_idx[i] = keypoint index for leaf i
        leaf_kp_idx = list(range(n_branch, n_branch + n_leaf))
        
        segment_edges = []
        
        for seg_idx, (path, endpoints, n_interior) in enumerate(
            zip(segment_paths, segment_endpoints, interior_per_segment)
        ):
            start_type, start_idx, end_type, end_idx = endpoints
            
            # Get keypoint indices for endpoints
            if start_type == 'leaf':
                start_kp = leaf_kp_idx[start_idx]
            else:
                start_kp = branch_kp_idx[start_idx]
            
            if end_type == 'leaf':
                end_kp = leaf_kp_idx[end_idx]
            else:
                end_kp = branch_kp_idx[end_idx]
            
            # Place interior keypoints uniformly along path
            if n_interior > 0 and len(path) > 2:
                # Compute cumulative 3D distance along path
                cum_dist = [0.0]
                for i in range(len(path) - 1):
                    d = np.linalg.norm(skeleton_3d[path[i + 1]] - skeleton_3d[path[i]])
                    cum_dist.append(cum_dist[-1] + d)
                total_dist = cum_dist[-1]
                
                # Place interior points at uniform distances
                interior_kp_indices = []
                for k in range(1, n_interior + 1):
                    target_dist = k * total_dist / (n_interior + 1)
                    
                    # Find position on path
                    for i in range(len(cum_dist) - 1):
                        if cum_dist[i] <= target_dist <= cum_dist[i + 1]:
                            # Interpolate between path[i] and path[i+1]
                            t = (target_dist - cum_dist[i]) / (cum_dist[i + 1] - cum_dist[i] + 1e-8)
                            pt = (1 - t) * skeleton_3d[path[i]] + t * skeleton_3d[path[i + 1]]
                            keypoints_list.append(pt.tolist())
                            interior_kp_indices.append(len(keypoints_list) - 1)
                            break
                
                # Build edges for this segment: start → interior_0 → ... → interior_n → end
                edges = []
                prev_kp = start_kp
                for int_kp in interior_kp_indices:
                    edges.append((prev_kp, int_kp))
                    prev_kp = int_kp
                edges.append((prev_kp, end_kp))
                segment_edges.append(edges)
            else:
                # No interior points, single edge
                segment_edges.append([(start_kp, end_kp)])
        
        keypoints = np.array(keypoints_list, dtype=np.float64)
        
        return keypoints, segment_edges
    
    def _place_keypoints_along_ordered_segments(
        self,
        ordered_segments: List[dict],
        branch_3d: np.ndarray,
        leaf_3d: np.ndarray,
        segment_structure: dict,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Place keypoints using GLOBAL FPS on MST with anchor seeds, then assign to segments.
        
        Pipeline:
            1. Build MST point cloud (union of all segment paths)
            2. Run global FPS with anchors (branch + leaf) as seeds
            3. Assign each interior FPS point to its segment
            4. Sort points within each segment by arc-length
            5. Build edges as chains within each segment
        
        Keypoint ordering: [branch_0, ..., branch_{N_b-1}, leaf_0, ..., leaf_{N_l-1}, interior...]
        
        Args:
            ordered_segments: List of segment dicts from _order_segments_by_ee()
            branch_3d: Branch keypoint 3D positions
            leaf_3d: Leaf keypoint 3D positions
            segment_structure: Original segment structure (for skeleton_3d)
        
        Returns:
            keypoints: K × 3 keypoints
            segment_edges: List of edge lists per segment (in same order as ordered_segments)
        """
        skeleton_3d = segment_structure['skeleton_3d']
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        n_anchor = n_branch + n_leaf
        n_interior_total = self.n_keypoints - n_anchor
        
        # ============================================================
        # Step 1: Build MST point cloud (union of all segment paths)
        # ============================================================
        mst_indices_set = set()
        for seg in ordered_segments:
            mst_indices_set.update(seg['path'])
        mst_indices = sorted(mst_indices_set)
        
        # Map from skeleton index to MST local index
        skel_to_mst = {skel_idx: mst_idx for mst_idx, skel_idx in enumerate(mst_indices)}
        
        mst_skeleton_3d = skeleton_3d[mst_indices]  # MST 3D points
        print(f"    MST point cloud: {len(mst_indices)} points")
        
        # ============================================================
        # Step 2: Run global FPS with anchors as seeds
        # ============================================================
        anchors_3d = np.vstack([branch_3d, leaf_3d])  # (n_anchor, 3)
        
        fps_interior_points, fps_mst_indices = self._global_fps_with_anchors(
            mst_skeleton_3d, anchors_3d, n_interior_total
        )
        print(f"    Global FPS selected {len(fps_interior_points)} interior points")
        
        # Convert MST indices back to skeleton indices
        fps_skel_indices = [mst_indices[mst_idx] for mst_idx in fps_mst_indices]
        
        # ============================================================
        # Step 3: Assign each interior FPS point to nearest segment
        # ============================================================
        point_to_segment = self._assign_points_to_segments(
            fps_skel_indices, fps_interior_points, ordered_segments, skeleton_3d
        )
        
        # ============================================================
        # Step 4 & 5: Sort by arc-length and build edges per segment
        # ============================================================
        # Start with anchor keypoints
        keypoints_list = []
        keypoints_list.extend(branch_3d.tolist())
        keypoints_list.extend(leaf_3d.tolist())
        
        segment_edges = []
        
        for seg_idx, seg_info in enumerate(ordered_segments):
            path = seg_info['path']
            start_kp = seg_info['start_kp']
            end_kp = seg_info['end_kp']
            seg_type = seg_info['type']
            
            # Get FPS points assigned to this segment
            seg_fps_indices = [i for i, s in enumerate(point_to_segment) if s == seg_idx]
            
            if len(seg_fps_indices) > 0:
                # Get the FPS points
                seg_fps_points = fps_interior_points[seg_fps_indices]
                
                # Add interior keypoints to list (order will be determined below)
                interior_kp_indices = []
                for pt in seg_fps_points:
                    keypoints_list.append(pt.tolist())
                    interior_kp_indices.append(len(keypoints_list) - 1)
                
                # Find chain ordering that minimizes total Euclidean distance
                # We need to connect: start_kp → [permutation of interior] → end_kp
                start_pos = np.array(keypoints_list[start_kp])
                end_pos = np.array(keypoints_list[end_kp])
                n_int = len(interior_kp_indices)
                
                if n_int <= 8:
                    # Exact: try all permutations (n! ≤ 40320)
                    from itertools import permutations
                    best_order = list(range(n_int))
                    best_cost = float('inf')
                    
                    for perm in permutations(range(n_int)):
                        cost = np.linalg.norm(start_pos - seg_fps_points[perm[0]])
                        for k in range(len(perm) - 1):
                            cost += np.linalg.norm(seg_fps_points[perm[k]] - seg_fps_points[perm[k+1]])
                        cost += np.linalg.norm(seg_fps_points[perm[-1]] - end_pos)
                        if cost < best_cost:
                            best_cost = cost
                            best_order = list(perm)
                else:
                    # Greedy nearest-neighbor chain for larger segments
                    remaining = set(range(n_int))
                    best_order = []
                    current_pos = start_pos
                    
                    while remaining:
                        nearest = min(remaining, 
                                      key=lambda idx: np.linalg.norm(current_pos - seg_fps_points[idx]))
                        best_order.append(nearest)
                        current_pos = seg_fps_points[nearest]
                        remaining.remove(nearest)
                
                # Build edges using optimal ordering
                ordered_interior = [interior_kp_indices[i] for i in best_order]
                edges = []
                prev_kp = start_kp
                for int_kp in ordered_interior:
                    edges.append((prev_kp, int_kp))
                    prev_kp = int_kp
                edges.append((prev_kp, end_kp))
                segment_edges.append(edges)
                
                print(f"    Segment {seg_idx} ({seg_type:10s}): {len(ordered_interior)} interior, "
                      f"{len(edges)} edges, kp {start_kp} -> {end_kp}")
            else:
                # No interior points assigned to this segment
                segment_edges.append([(start_kp, end_kp)])
                print(f"    Segment {seg_idx} ({seg_type:10s}): 0 interior, 1 edge, kp {start_kp} -> {end_kp}")
        
        keypoints = np.array(keypoints_list, dtype=np.float64)
        
        return keypoints, segment_edges
    
    def _global_fps_with_anchors(
        self,
        mst_points: np.ndarray,
        anchors: np.ndarray,
        n_interior: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Run global FPS on MST point cloud with anchors as initial seeds.
        
        Seeds FPS distances from the nearest MST points to each anchor
        (not from anchor positions directly), so the distance field is
        consistent with the point cloud even when anchors are off-skeleton.
        
        Args:
            mst_points: M × 3 MST skeleton points
            anchors: A × 3 anchor points (branch + leaf nodes)
            n_interior: Number of interior points to select
        
        Returns:
            fps_points: n_interior × 3 selected interior points
            fps_mst_indices: MST indices of selected points
        """
        M = mst_points.shape[0]
        n_anchors = anchors.shape[0]
        
        if M == 0 or n_interior <= 0:
            return np.empty((0, 3)), []
        
        # Find nearest MST points to each anchor
        anchor_mst_indices = []
        for anchor in anchors:
            dists = np.linalg.norm(mst_points - anchor, axis=1)
            nearest_idx = np.argmin(dists)
            anchor_mst_indices.append(nearest_idx)
        
        # Initialize FPS distances from the NEAREST MST POINTS to each anchor
        # (not from anchor positions), so distances are consistent with the
        # MST point cloud even when anchors (e.g. EE positions) are off-skeleton.
        min_distances = np.full(M, np.inf)
        for mst_idx in anchor_mst_indices:
            dists = np.linalg.norm(mst_points - mst_points[mst_idx], axis=1)
            min_distances = np.minimum(min_distances, dists)
        
        # Mark anchor-adjacent points as already "covered"
        anchor_set = set(anchor_mst_indices)
        
        # Select n_interior points using FPS
        chosen_indices = []
        for _ in range(n_interior):
            # Find point with maximum minimum distance (excluding already chosen)
            candidates = min_distances.copy()
            for idx in chosen_indices:
                candidates[idx] = -1  # Exclude already chosen
            for idx in anchor_set:
                candidates[idx] = -1  # Exclude anchor-adjacent points
            
            if np.max(candidates) <= 0:
                break  # No more valid points
            
            farthest_idx = np.argmax(candidates)
            chosen_indices.append(farthest_idx)
            
            # Update minimum distances
            new_dists = np.linalg.norm(mst_points - mst_points[farthest_idx], axis=1)
            min_distances = np.minimum(min_distances, new_dists)
        
        if len(chosen_indices) == 0:
            return np.empty((0, 3)), []
        
        fps_points = mst_points[chosen_indices]
        return fps_points, chosen_indices
    
    def _assign_points_to_segments(
        self,
        fps_skel_indices: List[int],
        fps_points: np.ndarray,
        ordered_segments: List[dict],
        skeleton_3d: np.ndarray,
    ) -> List[int]:
        """
        Assign each FPS interior point to its nearest segment.
        
        For each point:
            1. Check if it lies exactly on a segment's path
            2. If not, assign to segment whose path is nearest
        
        Args:
            fps_skel_indices: Skeleton indices of FPS points
            fps_points: N × 3 FPS interior points
            ordered_segments: List of segment dicts
            skeleton_3d: Full skeleton 3D array
        
        Returns:
            List of segment indices (one per FPS point)
        """
        n_points = len(fps_skel_indices)
        assignments = []
        
        # Build path sets for each segment
        segment_paths = [set(seg['path']) for seg in ordered_segments]
        
        for i, (skel_idx, pt) in enumerate(zip(fps_skel_indices, fps_points)):
            # Check if point is exactly on a segment's path
            assigned = False
            for seg_idx, path_set in enumerate(segment_paths):
                if skel_idx in path_set:
                    assignments.append(seg_idx)
                    assigned = True
                    break
            
            if not assigned:
                # Point not on any path, assign to nearest segment
                min_dist = float('inf')
                best_seg = 0
                
                for seg_idx, seg in enumerate(ordered_segments):
                    path = seg['path']
                    path_points = skeleton_3d[path]
                    dists = np.linalg.norm(path_points - pt, axis=1)
                    min_path_dist = np.min(dists)
                    
                    if min_path_dist < min_dist:
                        min_dist = min_path_dist
                        best_seg = seg_idx
                
                assignments.append(best_seg)
        
        return assignments
    
    def _find_arc_length_for_point(
        self,
        point: np.ndarray,
        path: List[int],
        skeleton_3d: np.ndarray,
        cum_dist: List[float],
    ) -> float:
        """
        Find arc-length position for a point not exactly on the path.
        
        Projects the point onto the path and returns interpolated arc-length.
        """
        # Find nearest path segment
        min_dist = float('inf')
        best_arc_len = 0.0
        
        for i in range(len(path) - 1):
            p1 = skeleton_3d[path[i]]
            p2 = skeleton_3d[path[i + 1]]
            
            # Project point onto line segment p1-p2
            v = p2 - p1
            w = point - p1
            
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)
            
            if c2 < 1e-10:
                t = 0.0
            else:
                t = np.clip(c1 / c2, 0.0, 1.0)
            
            proj = p1 + t * v
            dist = np.linalg.norm(point - proj)
            
            if dist < min_dist:
                min_dist = dist
                # Interpolate arc-length
                best_arc_len = cum_dist[i] + t * (cum_dist[i + 1] - cum_dist[i])
        
        return best_arc_len
    
    def _local_fps_on_segment(
        self,
        segment_points: np.ndarray,
        path_indices: List[int],
        n_samples: int,
        skeleton_3d: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Run Farthest Point Sampling on a segment's skeleton points.
        
        Args:
            segment_points: M × 3 skeleton points for this segment (interior only)
            path_indices: Original skeleton indices for these points
            n_samples: Number of points to sample
            skeleton_3d: Full skeleton 3D array (for reference)
        
        Returns:
            fps_points: n_samples × 3 sampled points
            fps_path_indices: Original skeleton indices of sampled points
        """
        M = segment_points.shape[0]
        
        if M == 0:
            return np.empty((0, 3)), []
        
        if n_samples >= M:
            # Return all points if we need more than available
            return segment_points.copy(), list(path_indices)
        
        # Standard FPS algorithm
        chosen_local = [0]  # Start with first point
        min_distances = np.full(M, np.inf)
        
        # Initialize distances from first chosen point
        min_distances = np.linalg.norm(segment_points - segment_points[0], axis=1)
        
        for _ in range(1, n_samples):
            # Pick the point with maximum minimum distance
            farthest_idx = np.argmax(min_distances)
            chosen_local.append(farthest_idx)
            
            # Update minimum distances
            new_distances = np.linalg.norm(segment_points - segment_points[farthest_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
        
        fps_points = segment_points[chosen_local]
        fps_path_indices = [path_indices[i] for i in chosen_local]
        
        return fps_points, fps_path_indices
    
    def _compute_cumulative_arc_length(
        self,
        path: List[int],
        skeleton_3d: np.ndarray,
    ) -> List[float]:
        """
        Compute cumulative arc-length along a skeleton path.
        
        Args:
            path: List of skeleton indices defining the path
            skeleton_3d: N × 3 skeleton 3D points
        
        Returns:
            cum_dist: List of cumulative distances, same length as path
        """
        cum_dist = [0.0]
        for i in range(len(path) - 1):
            d = np.linalg.norm(skeleton_3d[path[i + 1]] - skeleton_3d[path[i]])
            cum_dist.append(cum_dist[-1] + d)
        return cum_dist

    def _fps_with_anchors(
        self, points: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """
        Farthest Point Sampling with anchor points as initial seeds.
        
        Args:
            points: N × 3 point cloud
            anchors: A × 3 anchor points (branch + leaf nodes)
        
        Returns:
            keypoints: K × 3 selected points
        """
        N = points.shape[0]
        n_anchors = anchors.shape[0]
        
        if N == 0:
            return anchors.copy() if n_anchors > 0 else np.empty((0, 3))
        
        if n_anchors > 0:
            # Find nearest points to anchors
            nn = NearestNeighbors(n_neighbors=1).fit(points)
            _, anchor_indices = nn.kneighbors(anchors)
            anchor_indices = anchor_indices.flatten().tolist()
            
            chosen = anchor_indices.copy()
            chosen_set = set(chosen)
            
            # Initialize distances
            distances = np.full(N, np.inf)
            for idx in chosen:
                new_dist = np.linalg.norm(points - points[idx], axis=1)
                distances = np.minimum(distances, new_dist)
        else:
            # Start from centroid-farthest point
            centroid = np.mean(points, axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            start_idx = int(np.argmax(dists))
            
            chosen = [start_idx]
            chosen_set = {start_idx}
            distances = np.linalg.norm(points - points[start_idx], axis=1)
        
        # FPS iterations
        n_additional = self.n_keypoints - len(chosen)
        for _ in range(n_additional):
            masked = distances.copy()
            for idx in chosen_set:
                masked[idx] = -np.inf
            
            next_idx = int(np.argmax(masked))
            chosen.append(next_idx)
            chosen_set.add(next_idx)
            
            new_dist = np.linalg.norm(points - points[next_idx], axis=1)
            distances = np.minimum(distances, new_dist)
        
        chosen = np.array(chosen[:self.n_keypoints], dtype=np.int64)
        return points[chosen]
    
    def _repulsion_relaxation_with_topology(
        self, 
        keypoints: np.ndarray, 
        target_points: np.ndarray, 
        fixed_mask: np.ndarray,
        edges: List[Tuple[int, int]],
        segment_edges: List[List[Tuple[int, int]]] = None,
        segment_lengths: List[float] = None,
    ) -> np.ndarray:
        """
        Gauss-Seidel sequential spring relaxation with per-segment target edge lengths.
        
        Algorithm:
            OUTER LOOP (repulsion_iterations):
                1. Project all FREE keypoints to cloud (once per iteration)
                2. INNER LOOP: for each segment, for each edge (i, j):
                       - Compute spring force for this edge
                       - Apply correction to FREE nodes IMMEDIATELY (in-place)
            
            NO final projection - edge constraints have final say
        
        Key differences from Jacobi (previous):
            - Updates are applied immediately after each edge (not batched)
            - Corrections propagate along the chain within a single iteration
            - Faster convergence for chain/tree structures
        
        Args:
            keypoints: N×3 keypoint positions
            target_points: M×3 point cloud for surface projection
            fixed_mask: N boolean mask (True = anchor, don't move)
            edges: List of (i, j) edges
            segment_edges: List of edge lists per segment
            segment_lengths: List of target segment lengths (mm)
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8
        
        # Save anchor positions for verification
        anchor_positions_before = keypoints[fixed_mask].copy()
        n_anchors = np.sum(fixed_mask)
        print(f"  Anchors: {n_anchors} keypoints (indices: {np.where(fixed_mask)[0].tolist()})")
        
        if K <= 1 or len(target_points) == 0:
            return keypoints
        
        # Build NN index for projection
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        cloud_nn.fit(target_points)
        
        # Compute per-edge target lengths based on segment info
        edge_to_target: Dict[Tuple[int, int], float] = {}
        
        if segment_edges is not None and segment_lengths is not None:
            # Per-segment target: segment_length / n_edges
            for seg_idx, seg_edge_list in enumerate(segment_edges):
                if seg_idx < len(segment_lengths) and len(seg_edge_list) > 0:
                    seg_len = segment_lengths[seg_idx]
                    n_edges = len(seg_edge_list)
                    target_edge_len = seg_len / n_edges
                    
                    for edge in seg_edge_list:
                        i, j = edge
                        edge_to_target[(i, j)] = target_edge_len
                        edge_to_target[(j, i)] = target_edge_len
            
            print(f"  Per-segment target edge lengths:")
            for seg_idx, seg_edge_list in enumerate(segment_edges):
                if seg_idx < len(segment_lengths) and len(seg_edge_list) > 0:
                    seg_len = segment_lengths[seg_idx]
                    n_edges = len(seg_edge_list)
                    print(f"    Segment {seg_idx}: {seg_len:.1f}mm / {n_edges} edges = {seg_len/n_edges:.1f}mm per edge")
        
        # Fallback: use global mean edge length if no segment info
        use_global_target = len(edge_to_target) == 0
        if use_global_target:
            all_lens = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
            global_tgt = np.mean(all_lens) if all_lens else 50.0
            print(f"  Using global target edge length: {global_tgt:.1f}mm")
        
        # Determine edge processing order
        if segment_edges is not None and len(segment_edges) > 0:
            # Use segment-ordered edges for sequential propagation
            edge_order = [(i, j) for seg in segment_edges for (i, j) in seg]
        else:
            # Fallback to provided edges
            edge_order = edges
        
        print(f"  Gauss-Seidel relaxation: {self.repulsion_iterations} iterations, {len(edge_order)} edges, lr={self.repulsion_lr}")
        
        # Helper to compute edge length error
        def compute_edge_error():
            errors = []
            for (i, j) in edge_order:
                d = np.linalg.norm(keypoints[j] - keypoints[i])
                if use_global_target:
                    tgt = global_tgt
                else:
                    tgt = edge_to_target.get((i, j), edge_to_target.get((j, i), 50.0))
                errors.append(abs(d - tgt))
            return np.mean(errors), np.max(errors)
        
        # Project every N iterations (not every iteration - it fights edge corrections)
        project_every = 10
        
        # OUTER LOOP
        for iteration in range(self.repulsion_iterations):
            
            # Step 1: Edge corrections FIRST (Gauss-Seidel sequential)
            # Process edges in segment order for proper chain propagation
            for (i, j) in edge_order:
                
                # Determine if endpoints are free
                i_free = not fixed_mask[i]
                j_free = not fixed_mask[j]
                
                # Skip if both are anchored
                if not i_free and not j_free:
                    continue
                
                # Compute spring force for this edge
                v = keypoints[j] - keypoints[i]  # Vector from i to j
                d = np.linalg.norm(v)
                
                if d < epsilon:
                    continue  # Skip degenerate edges
                
                unit_v = v / d
                
                # Get target length for this edge
                if use_global_target:
                    tgt = global_tgt
                else:
                    tgt = edge_to_target.get((i, j), edge_to_target.get((j, i), 50.0))
                
                # Spring force magnitude: positive = expand, negative = contract
                force_mag = (tgt - d) / tgt
                
                # Determine weights based on anchor status
                if i_free and j_free:
                    weight_i, weight_j = 0.5, 0.5
                elif i_free:
                    weight_i, weight_j = 1.0, 0.0
                else:  # j_free
                    weight_i, weight_j = 0.0, 1.0
                
                # Apply correction IMMEDIATELY (in-place, Gauss-Seidel style)
                # Correction direction: i moves opposite to v, j moves along v
                correction = force_mag * self.repulsion_lr * unit_v
                if i_free:
                    keypoints[i] -= correction * weight_i
                if j_free:
                    keypoints[j] += correction * weight_j
            
            # Project to surface every N iterations with strong strength
            # This keeps keypoints near the surface while allowing edge optimization to adapt
            if (iteration + 1) % 10 == 0:
                _, target_indices = cloud_nn.kneighbors(keypoints)
                proj_strength = 0.5  # Strong blend toward surface
                for k in range(K):
                    if not fixed_mask[k]:
                        cloud_pt = target_points[target_indices[k, 0]]
                        keypoints[k] = (1 - proj_strength) * keypoints[k] + proj_strength * cloud_pt
            
            # Log convergence at key iterations
            if iteration == 0 or iteration == self.repulsion_iterations - 1 or (iteration + 1) % 100 == 0:
                mean_err, max_err = compute_edge_error()
                _, tmp_indices = cloud_nn.kneighbors(keypoints)
                tmp_dists = np.linalg.norm(keypoints - target_points[tmp_indices[:, 0]], axis=1)
                surf_rmse = np.sqrt(np.mean(tmp_dists[~fixed_mask] ** 2)) if np.sum(~fixed_mask) > 0 else 0.0
                print(f"    Iter {iteration+1:4d}: edge_err={mean_err:.2f}mm (max={max_err:.2f}), surf_rmse={surf_rmse:.2f}mm")
        
        # Final hard projection: snap all free keypoints to nearest cloud point
        _, final_proj_indices = cloud_nn.kneighbors(keypoints)
        for k in range(K):
            if not fixed_mask[k]:
                keypoints[k] = target_points[final_proj_indices[k, 0]]
        print(f"    Final projection: snapped all free keypoints to surface")
        
        # Log final edge error after projection
        mean_err_final, max_err_final = compute_edge_error()
        _, final_indices = cloud_nn.kneighbors(keypoints)
        final_dists = np.linalg.norm(keypoints - target_points[final_indices[:, 0]], axis=1)
        final_surf_rmse = np.sqrt(np.mean(final_dists[~fixed_mask] ** 2)) if np.sum(~fixed_mask) > 0 else 0.0
        print(f"    Final: edge_err={mean_err_final:.2f}mm, max={max_err_final:.2f}mm, surf_rmse={final_surf_rmse:.2f}mm")
        
        # Verify anchors didn't move
        anchor_positions_after = keypoints[fixed_mask]
        anchor_drift = np.linalg.norm(anchor_positions_after - anchor_positions_before, axis=1)
        max_anchor_drift = np.max(anchor_drift) if len(anchor_drift) > 0 else 0.0
        if max_anchor_drift > 1e-6:
            print(f"  WARNING: Anchors moved! Max drift = {max_anchor_drift:.6f} mm")
            print(f"    Anchor drifts: {anchor_drift}")
        else:
            print(f"  Anchors verified: no drift (max={max_anchor_drift:.2e})")
        
        # Compute surface distance stats for reporting
        dists, _ = cloud_nn.kneighbors(keypoints)
        free_dists = dists[~fixed_mask, 0]
        surface_rmse = np.sqrt(np.mean(free_dists ** 2)) if len(free_dists) > 0 else 0.0
        print(f"  Surface RMSE (free keypoints to cloud): {surface_rmse:.3f} mm")
        
        return keypoints

    def _gmm_keypoint_placement(
        self,
        keypoints: np.ndarray,
        target_points: np.ndarray,
        fixed_mask: np.ndarray,
        edges: List[Tuple[int, int]],
        segment_edges: List[List[Tuple[int, int]]],
        ordered_segments: List[dict],
        segment_structure: dict,
    ) -> np.ndarray:
        """
        Global GMM keypoint refinement with chain-uniform penalty.
        
        Takes FPS-initialized keypoints (already well-positioned with correct edges)
        and refines positions using GMM EM while maintaining topology.
        
        Key design choices:
        1. Small initial covariance (~edge_length^2) so each component covers local region
        2. NO projection during EM — let optimization find continuous optima
        3. Chain-uniform penalty to keep equal spacing
        4. Final projection ONLY after convergence
        
        All K keypoints are GMM components fitted to the entire skeleton point cloud.
        - Anchor keypoints (fixed_mask=True): mean and covariance FIXED
        - Free keypoints (fixed_mask=False): mean and covariance UPDATED
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = len(keypoints)
        N = len(target_points)
        
        if K <= 1 or N == 0:
            return keypoints
        
        # Build NN for final projection only
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        cloud_nn.fit(target_points)
        
        # Hyperparameters
        max_iter = 100
        tol = 1e-4
        lambda_uniform = 1.0    # uniform penalty weight (applied after M-step)
        eta = 0.3               # gradient step size for penalty
        sigma_fix = 1.0         # anchor covariance (mm) - small to attract nearby points
        covariance_reg = 1e-6
        eps = 1e-8
        
        # Compute initial edge length to set appropriate covariance scale
        init_edge_lengths = [np.linalg.norm(keypoints[j] - keypoints[i]) for i, j in edges]
        avg_edge_len = np.mean(init_edge_lengths) if init_edge_lengths else 30.0
        
        # Initialize means directly from FPS keypoints
        means = keypoints.copy()
        
        # Initialize covariances: diagonal (K, 3)
        # Use edge-length-based variance so each component covers ~1 edge region
        # sigma ~ avg_edge_len / 2, so variance ~ (avg_edge_len/2)^2
        init_var = (avg_edge_len / 2.0) ** 2
        
        covariances = np.zeros((K, 3))
        for k in range(K):
            if fixed_mask[k]:
                covariances[k] = sigma_fix ** 2
            else:
                covariances[k] = init_var
        
        # Equal mixing weights (fixed)
        log_pi = -np.log(K)
        
        prev_ll = -np.inf
        
        print(f"  GMM init: avg_edge={avg_edge_len:.1f}mm, init_sigma={np.sqrt(init_var):.1f}mm")
        
        for em_iter in range(max_iter):
            # Precompute precision and log-det
            precisions = 1.0 / covariances  # (K, 3)
            log_dets = np.sum(np.log(covariances), axis=1)  # (K,)
            
            # === E-step ===
            log_resp = np.empty((N, K))
            for k in range(K):
                diff = target_points - means[k]  # (N, 3)
                mahal = np.sum(diff ** 2 * precisions[k], axis=1)  # (N,)
                log_resp[:, k] = log_pi - 0.5 * (3 * np.log(2 * np.pi) + log_dets[k] + mahal)
            
            # Log-sum-exp normalization
            log_resp_max = np.max(log_resp, axis=1, keepdims=True)
            stable = log_resp - log_resp_max
            exp_stable = np.exp(stable)
            sum_exp = np.sum(exp_stable, axis=1, keepdims=True)
            resp = exp_stable / sum_exp  # (N, K)
            
            # Log-likelihood
            ll = float(np.sum(log_resp_max.ravel() + np.log(sum_exp.ravel())))
            
            if em_iter > 0 and abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            
            # === M-step: update free means and covariances ===
            for k in range(K):
                if fixed_mask[k]:
                    continue
                
                Nk = np.sum(resp[:, k])
                if Nk < 1e-10:
                    continue
                
                # Update mean: weighted centroid
                means[k] = np.sum(resp[:, k:k+1] * target_points, axis=0) / Nk
                
                # Update covariance: diagonal (but clamp to reasonable range)
                diff = target_points - means[k]
                new_var = np.sum(resp[:, k:k+1] * (diff ** 2), axis=0) / Nk + covariance_reg
                # Clamp variance to prevent explosion or collapse
                min_var = (avg_edge_len / 4.0) ** 2
                max_var = (avg_edge_len * 2.0) ** 2
                covariances[k] = np.clip(new_var, min_var, max_var)
            
            # === Chain-uniform penalty: gradient descent on free means ===
            # Compute current edge lengths
            edge_lengths = {}
            for i, j in edges:
                edge_lengths[(i, j)] = np.linalg.norm(means[j] - means[i])
            
            if len(edge_lengths) > 0:
                d_bar = np.mean(list(edge_lengths.values()))
                
                # Accumulate gradient for each keypoint
                grad = np.zeros((K, 3))
                for (i, j), d_e in edge_lengths.items():
                    if d_e < eps:
                        continue
                    direction = (means[j] - means[i]) / d_e
                    # Penalty: (d_e - d_bar)^2
                    # grad w.r.t. mu_i = -2*(d_e - d_bar) * direction
                    # grad w.r.t. mu_j = +2*(d_e - d_bar) * direction
                    grad[i] -= 2.0 * (d_e - d_bar) * direction
                    grad[j] += 2.0 * (d_e - d_bar) * direction
                
                # Apply gradient only to free keypoints (NO projection here!)
                for k in range(K):
                    if not fixed_mask[k]:
                        means[k] -= eta * lambda_uniform * grad[k]
        
        # === Final projection: snap all free keypoints to nearest cloud point ===
        for k in range(K):
            if not fixed_mask[k]:
                _, idx = cloud_nn.kneighbors(means[k:k+1])
                means[k] = target_points[idx[0, 0]]
        
        # Compute final edge length stats
        final_lengths = [np.linalg.norm(means[j] - means[i]) for i, j in edges]
        if len(final_lengths) > 0:
            edge_std = np.std(final_lengths)
            edge_mean = np.mean(final_lengths)
            print(f"  GMM converged in {em_iter+1} iters, edge_mean={edge_mean:.1f}mm, edge_std={edge_std:.1f}mm")
        
        # Surface RMSE
        dists, _ = cloud_nn.kneighbors(means)
        free_dists = dists[~fixed_mask, 0]
        surface_rmse = np.sqrt(np.mean(free_dists ** 2)) if len(free_dists) > 0 else 0.0
        print(f"  Surface RMSE (free keypoints to cloud): {surface_rmse:.3f} mm")
        
        return means
    
    def _build_keypoint_topology(
        self, 
        keypoints: np.ndarray,
        skeleton_mask: np.ndarray = None,
    ) -> tuple:
        """
        Build MST edges on keypoints, with skeleton path validation and edge repair.
        
        Returns:
            edges: List of (i, j) tuples
            lengths: Array of edge lengths
        """
        K = keypoints.shape[0]
        if K <= 1:
            return [], np.array([])
        
        # Build distance matrix
        dists = cdist(keypoints, keypoints)
        
        # MST
        sparse = csr_matrix(dists)
        mst = minimum_spanning_tree(sparse)
        mst_dense = mst.toarray()
        
        # Extract MST edges
        mst_edges = []
        for i in range(K):
            for j in range(i + 1, K):
                if mst_dense[i, j] > 0 or mst_dense[j, i] > 0:
                    mst_edges.append((i, j))
        
        # If no skeleton_mask, return MST edges directly
        if skeleton_mask is None:
            edges = mst_edges
            lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
            return edges, np.array(lengths)
        
        # Build skeleton graph for validation
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if skeleton_coords.shape[0] == 0:
            edges = mst_edges
            lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
            return edges, np.array(lengths)
        
        skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
        
        # Get skeleton indices and blocked regions for each keypoint
        keypoint_skel_indices, blocked_regions = self._get_keypoint_blocked_regions(
            keypoints, skeleton_coords, skeleton_tree, block_radius=3
        )
        
        # Union of all blocked regions
        all_blocked: Set[int] = set()
        for region in blocked_regions:
            all_blocked.update(region)
        
        def is_valid_edge(i: int, j: int) -> bool:
            """Check if edge (i, j) has a valid skeleton path."""
            skel_i = keypoint_skel_indices[i]
            skel_j = keypoint_skel_indices[j]
            
            if skel_i < 0 or skel_j < 0:
                return True  # Can't validate, assume valid
            
            blocked = all_blocked - blocked_regions[i] - blocked_regions[j]
            return self._skeleton_path_exists(adjacency, skel_i, skel_j, blocked)
        
        # Validate MST edges and collect rejected ones
        validated_edges: List[Tuple[int, int]] = []
        rejected_edges: List[Tuple[int, int]] = []
        
        for i, j in mst_edges:
            if is_valid_edge(i, j):
                validated_edges.append((i, j))
            else:
                rejected_edges.append((i, j))
        
        # For each rejected edge, find a replacement to maintain connectivity
        def get_connected_components(edges: List[Tuple[int, int]], n_nodes: int) -> List[Set[int]]:
            """Find connected components given edges."""
            adj: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
            for i, j in edges:
                adj[i].add(j)
                adj[j].add(i)
            
            visited = set()
            components = []
            
            for start in range(n_nodes):
                if start in visited:
                    continue
                component = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    component.add(node)
                    for nbr in adj[node]:
                        if nbr not in visited:
                            queue.append(nbr)
                components.append(component)
            
            return components
        
        # Process rejected edges - try to find replacements
        for rej_i, rej_j in rejected_edges:
            components = get_connected_components(validated_edges, K)
            
            comp_i = None
            comp_j = None
            for idx, comp in enumerate(components):
                if rej_i in comp:
                    comp_i = idx
                if rej_j in comp:
                    comp_j = idx
            
            if comp_i == comp_j:
                continue
            
            candidates = []
            for node_i in components[comp_i]:
                for node_j in components[comp_j]:
                    if node_i != node_j:
                        candidates.append((node_i, node_j, dists[node_i, node_j]))
            
            candidates.sort(key=lambda x: x[2])
            
            replacement_found = False
            for cand_i, cand_j, _ in candidates:
                edge = (min(cand_i, cand_j), max(cand_i, cand_j))
                if edge not in validated_edges and is_valid_edge(cand_i, cand_j):
                    validated_edges.append(edge)
                    replacement_found = True
                    break
            
            if not replacement_found:
                validated_edges.append((rej_i, rej_j))
        
        edges = validated_edges
        lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
        
        return edges, np.array(lengths)
    
    def _extract_segment_ordered_edges(self) -> None:
        """
        Extract edges ordered by segment for sequential geometry correction.
        
        Guarantees segment ordering based on EE mapping:
            Segment 0: ee_leaf with EE 0 → its connected branch (b_ee0)
            Segment 1: ee_leaf with EE 1 → its connected branch (b_ee1)
            Segment 2: free_leaf connected to b_ee0
            Segment 3: free_leaf connected to b_ee1
            Segment 4: trunk (b_ee0 → b_ee1)
        
        Sets:
            self.segment_edges: List of segment edge lists
            self.anchor_set: Set of anchor keypoint indices
            self.free_leaf_indices: List of free leaf indices
        """
        if self.reference_edges is None:
            print("Warning: reference_edges not set, cannot extract segments")
            return
        
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        K = self.n_keypoints
        
        # Identify node types
        branch_indices = set(range(n_branch))
        leaf_indices = set(range(n_branch, n_branch + n_leaf))
        
        # EE-mapped leaves are anchors, others are free
        if self.ee_to_leaf_mapping is not None:
            ee_leaf_indices = set(self.ee_to_leaf_mapping.values())
        else:
            ee_leaf_indices = set()
        
        free_leaf_indices = leaf_indices - ee_leaf_indices
        
        self.anchor_set = branch_indices | ee_leaf_indices
        self.free_leaf_indices = list(free_leaf_indices)
        
        print(f"  Anchor nodes: {sorted(self.anchor_set)}")
        print(f"  Free leaf nodes: {sorted(self.free_leaf_indices)}")
        
        # Build adjacency list from reference edges
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(K)}
        edge_set = set()
        for i, j in self.reference_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
            edge_set.add((min(i, j), max(i, j)))
        
        # Helper: trace path from leaf to branch, return (path, branch_idx)
        def trace_leaf_to_branch(leaf: int) -> Tuple[List[int], int]:
            path = [leaf]
            current = leaf
            visited = {leaf}
            
            while current not in branch_indices:
                next_node = None
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        if neighbor in leaf_indices and neighbor != leaf:
                            continue
                        next_node = neighbor
                        break
                
                if next_node is None:
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            connected_branch = current if current in branch_indices else -1
            return path, connected_branch
        
        # Build leaf info: {leaf_kp_idx: (path, connected_branch)}
        leaf_info: Dict[int, Tuple[List[int], int]] = {}
        for leaf in leaf_indices:
            path, connected_branch = trace_leaf_to_branch(leaf)
            if connected_branch >= 0:
                leaf_info[leaf] = (path, connected_branch)
            else:
                print(f"  Warning: Leaf {leaf} does not connect to any branch")
        
        segment_edges = []
        used_edges = set()
        
        # Track which branch each EE connects to (via its mapped leaf)
        # ee_to_branch: {ee_idx: branch_idx}
        ee_to_branch: Dict[int, int] = {}
        if self.ee_to_leaf_mapping is not None:
            for ee_idx, leaf_kp_idx in self.ee_to_leaf_mapping.items():
                if leaf_kp_idx in leaf_info:
                    _, connected_branch = leaf_info[leaf_kp_idx]
                    ee_to_branch[ee_idx] = connected_branch
        
        # 1. Process ee_leaves in EE index order (EE 0 first, then EE 1)
        if self.ee_to_leaf_mapping is not None:
            for ee_idx in sorted(self.ee_to_leaf_mapping.keys()):  # 0, 1
                leaf_kp_idx = self.ee_to_leaf_mapping[ee_idx]
                if leaf_kp_idx in leaf_info:
                    path, connected_branch = leaf_info[leaf_kp_idx]
                    # ee_leaf: path goes leaf → branch (for correction from EE)
                    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    for e in path_edges:
                        used_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                    segment_edges.append(path_edges)
                    print(f"  Segment {len(segment_edges)-1} (ee_leaf  ): {path}, {len(path_edges)} edges, EE{ee_idx}->branch{connected_branch}")
        
        # 2. Process free_leaves in EE branch order (branch of EE 0 first, then branch of EE 1)
        # Group free leaves by their connected branch
        branch_to_free_leaves: Dict[int, List[Tuple[int, List[int]]]] = {b: [] for b in branch_indices}
        for leaf in free_leaf_indices:
            if leaf in leaf_info:
                path, connected_branch = leaf_info[leaf]
                branch_to_free_leaves[connected_branch].append((leaf, path))
        
        # Process in EE branch order
        for ee_idx in sorted(ee_to_branch.keys()):  # 0, 1
            branch_idx = ee_to_branch[ee_idx]
            for leaf, path in branch_to_free_leaves[branch_idx]:
                # free_leaf: path goes branch → leaf (for correction from branch)
                path = path[::-1]  # Reverse: branch → leaf
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                for e in path_edges:
                    used_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                segment_edges.append(path_edges)
                print(f"  Segment {len(segment_edges)-1} (free_leaf): {path}, {len(path_edges)} edges, branch{branch_idx}(=b_ee{ee_idx})")
                # Mark as processed
                branch_to_free_leaves[branch_idx] = [(l, p) for l, p in branch_to_free_leaves[branch_idx] if l != leaf]
        
        # 3. Trunk: trace from b_ee0 to b_ee1 (always last)
        if len(ee_to_branch) >= 2:
            b_ee0 = ee_to_branch[0]
            b_ee1 = ee_to_branch[1]
            
            path = [b_ee0]
            current = b_ee0
            visited = {b_ee0}
            
            while current != b_ee1:
                next_node = None
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        edge_key = (min(current, neighbor), max(current, neighbor))
                        if edge_key in used_edges:
                            continue
                        if neighbor in leaf_indices:
                            continue
                        next_node = neighbor
                        break
                
                if next_node is None:
                    print(f"  Warning: Dead end at node {current} tracing trunk")
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if current == b_ee1:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                for e in path_edges:
                    used_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                segment_edges.append(path_edges)
                print(f"  Segment {len(segment_edges)-1} (trunk    ): {path}, {len(path_edges)} edges, b_ee0({b_ee0})->b_ee1({b_ee1})")
        
        # Check coverage
        uncovered = edge_set - used_edges
        if uncovered:
            print(f"  Warning: {len(uncovered)} edges not covered: {uncovered}")
            segment_edges.append(list(uncovered))
        
        self.segment_edges = segment_edges
        print(f"  Total segments: {len(self.segment_edges)}, total edges: {len(used_edges)}")
    
    def _print_segment_summary(self, keypoints: np.ndarray, edges: list) -> None:
        """
        Print organized segment summary with edge lengths.
        
        Args:
            keypoints: K × 3 keypoint positions
            edges: List of (i, j) edge tuples
        """
        if self.segment_edges is None:
            return
        
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        
        branch_indices = set(range(n_branch))
        leaf_indices = set(range(n_branch, n_branch + n_leaf))
        
        # Build edge to length mapping
        edge_to_length = {}
        for i, j in edges:
            length = np.linalg.norm(keypoints[i] - keypoints[j])
            edge_to_length[(min(i, j), max(i, j))] = length
        
        print(f"\n  {'='*60}")
        print(f"  SEGMENT SUMMARY")
        print(f"  {'='*60}")
        
        total_wire_length = 0
        
        for seg_idx, seg in enumerate(self.segment_edges):
            if len(seg) == 0:
                continue
            
            # Get nodes in segment
            nodes = [seg[0][0]] + [e[1] for e in seg]
            start_node = nodes[0]
            end_node = nodes[-1]
            
            # Determine segment type
            if start_node in leaf_indices or end_node in leaf_indices:
                leaf_node = end_node if end_node in leaf_indices else start_node
                if leaf_node in self.free_leaf_indices:
                    seg_type = "free_leaf"
                else:
                    seg_type = "ee_leaf"
            elif start_node in branch_indices and end_node in branch_indices:
                seg_type = "trunk"
            else:
                seg_type = "other"
            
            # Calculate edge lengths for this segment
            seg_lengths = []
            edge_strs = []
            for e in seg:
                edge_key = (min(e[0], e[1]), max(e[0], e[1]))
                length = edge_to_length.get(edge_key, 0)
                seg_lengths.append(length)
                edge_strs.append(f"({e[0]:2d}->{e[1]:2d}): {length:6.1f}")
            
            seg_total = sum(seg_lengths)
            total_wire_length += seg_total
            
            print(f"\n  Segment {seg_idx} ({seg_type:10s}): {nodes[0]} -> {nodes[-1]}")
            print(f"    Nodes: {nodes}")
            print(f"    Edges ({len(seg)}):")
            for edge_str in edge_strs:
                print(f"      {edge_str} mm")
            print(f"    Segment length: {seg_total:.1f} mm")
        
        print(f"\n  {'-'*60}")
        print(f"  Total wire length: {total_wire_length:.1f} mm")
        print(f"  Total edges: {len(edges)}")
        print(f"  Avg edge length: {np.mean(self.reference_lengths):.1f} mm")
        print(f"  Edge length range: [{np.min(self.reference_lengths):.1f}, {np.max(self.reference_lengths):.1f}] mm")
        print(f"  {'='*60}")
    
    # ================================================================
    # END-EFFECTOR POSE INJECTION
    # ================================================================
    
    def _establish_ee_to_leaf_mapping(self, keypoints: np.ndarray) -> None:
        """
        Establish mapping from EE indices to leaf keypoint indices.
        
        Uses Hungarian assignment to match 2 EE positions to 2 closest leaf nodes.
        """
        if self.ee_poses_3d is None:
            return
        
        ee_positions = self.ee_poses_3d[0]  # Frame 0: (2, 3)
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        
        leaf_indices = list(range(n_branch, n_branch + n_leaf))
        leaf_keypoints = keypoints[leaf_indices]
        
        cost_matrix = cdist(ee_positions, leaf_keypoints)
        ee_idx_arr, leaf_local_idx_arr = linear_sum_assignment(cost_matrix)
        
        self.ee_to_leaf_mapping = {}
        for ee_idx, leaf_local_idx in zip(ee_idx_arr, leaf_local_idx_arr):
            kp_idx = leaf_indices[leaf_local_idx]
            self.ee_to_leaf_mapping[ee_idx] = kp_idx
            dist = cost_matrix[ee_idx, leaf_local_idx]
            print(f"  EE {ee_idx} -> Leaf keypoint {kp_idx} (distance: {dist:.2f} mm)")
    
    def _replace_with_ee_poses(self, keypoints: np.ndarray) -> np.ndarray:
        """Replace mapped leaf keypoints with EE positions."""
        if self.ee_poses_3d is None or self.ee_to_leaf_mapping is None:
            return keypoints
        
        ee_positions = self.ee_poses_3d[0]  # Frame 0
        
        for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
            keypoints[kp_idx] = ee_positions[ee_idx]
        
        return keypoints
    
    # ================================================================
    # MAIN INITIALIZATION
    # ================================================================
    
    def initialize(self, depth: np.ndarray, arm_depth: np.ndarray = None,
                   precomputed_arm_mask: np.ndarray = None) -> dict:
        """
        Main initialization pipeline.
        
        Pipeline:
            1. Segmentation (background subtraction + skeletonization)
            2. Node identification (branch/leaf from MST degree)
            3. Topology pruning (prune to target branch/leaf count)
            4. Segment structure computation (paths between branch/leaf on skeleton)
            5. EE pose injection into anchor positions (before FPS)
            6. Order segments by EE mapping
            7. Global FPS on MST point cloud with anchors as seeds
               (algorithm naturally distributes points, no pre-allocation)
            8. Assign FPS points to segments, build chain edges
            9. Compute segment lengths from edge sums (post-FPS)
            10. Repulsion relaxation
        
        Args:
            depth: H × W current frame depth
            arm_depth: H × W arm-only depth (not needed if precomputed_arm_mask provided)
            precomputed_arm_mask: H × W binary mask where 1=arm, 0=keep
        
        Returns:
            dict with success, keypoints, edges, segment_edges, etc.
        """
        timing = {}
        total_start = time.time()
        
        # Phase 1: Segmentation
        print("Phase 1: Segmentation...")
        t0 = time.time()
        seg_result = self.segment(depth, arm_depth, precomputed_arm_mask)
        timing['segmentation'] = time.time() - t0
        
        foreground_mask = seg_result['foreground_mask']
        skeleton_mask = seg_result['skeleton_mask']
        skeleton_pc = seg_result['skeleton_pc']
        
        n_skeleton_pixels = np.sum(skeleton_mask > 0)
        print(f"  Skeleton pixels: {n_skeleton_pixels}")
        
        if n_skeleton_pixels < self.min_skeleton_pixels:
            return {
                'success': False,
                'reason': f'insufficient_skeleton_pixels ({n_skeleton_pixels} < {self.min_skeleton_pixels})',
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
            }
        
        # Phase 2: Node identification
        print("Phase 2: Node identification...")
        t0 = time.time()
        branch_2d, leaf_2d, adjacency, coords = self._node_identification(skeleton_mask)
        timing['node_detection'] = time.time() - t0
        print(f"  Detected: {len(branch_2d)} branch, {len(leaf_2d)} leaf nodes")
        
        # Phase 3: Topology pruning
        print("Phase 3: Topology pruning...")
        t0 = time.time()
        if adjacency is not None:
            pruned = self._prune_to_target_topology(adjacency, coords)
            branch_2d = pruned["branch_coords"]
            leaf_2d = pruned["leaf_coords"]
        timing['pruning'] = time.time() - t0
        print(f"  After pruning: {len(branch_2d)} branch, {len(leaf_2d)} leaf nodes")
        
        # Phase 4: 2D → 3D for branch/leaf
        t0 = time.time()
        branch_3d = self._pixel_to_3d(branch_2d, depth)
        leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        timing['2d_to_3d'] = time.time() - t0
        
        # Store reference counts
        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf
        
        # Phase 5: Establish EE-to-leaf mapping and inject EE positions
        # This must happen BEFORE segment structure computation so that
        # the MST paths are built with the correct anchor positions.
        print("Phase 5: EE mapping and injection...")
        t0 = time.time()
        ee_to_leaf_kp = self._establish_ee_mapping_from_leaves(leaf_3d, n_branch)
        
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            self.ee_to_leaf_mapping = ee_to_leaf_kp
            ee_positions = self.ee_poses_3d[0]  # Frame 0: (2, 3)
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch  # Convert to leaf-local index
                old_pos = leaf_3d[leaf_local_idx].copy()
                leaf_3d[leaf_local_idx] = ee_positions[ee_idx]
                dist = np.linalg.norm(old_pos - ee_positions[ee_idx])
                print(f"  EE {ee_idx} -> Leaf keypoint {kp_idx} (shift={dist:.1f} mm)")
        else:
            self.ee_to_leaf_mapping = ee_to_leaf_kp if ee_to_leaf_kp else None
            if ee_to_leaf_kp:
                for ee_idx, leaf_kp in ee_to_leaf_kp.items():
                    print(f"  EE {ee_idx} -> Leaf keypoint {leaf_kp} (no pose injection)")
            else:
                print("  No EE poses provided, skipping")
        timing['ee_injection'] = time.time() - t0
        
        # Phase 5b: Prune skeleton to only keep the 5 MST segment paths.
        # After EE injection we have 6 final nodes (2 branch + 2 EE leaf + 2 free leaf).
        # We find 5 paths: each leaf→nearest branch + trunk, and discard all other spurs.
        # First, update leaf_2d for EE leaves (project EE 3D → 2D so snap is correct).
        print("Phase 5b: Pruning skeleton to 5 MST segment paths...")
        t0 = time.time()
        
        leaf_2d_updated = leaf_2d.copy().astype(np.float64)
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                ee_2d = self._project_3d_to_2d(self.ee_poses_3d[0][ee_idx:ee_idx+1])  # (1, 2) -> [row, col]
                leaf_2d_updated[leaf_local_idx] = ee_2d[0]
                print(f"  Updated leaf_2d[{leaf_local_idx}] for EE {ee_idx}: ({leaf_2d[leaf_local_idx]}) -> ({ee_2d[0].astype(int)})")
        
        skeleton_mask_raw = skeleton_mask.copy()  # Keep original for viz
        skeleton_mask, n_before, n_after = self._prune_skeleton_to_node_paths(
            skeleton_mask, branch_2d, leaf_2d_updated
        )
        timing['skeleton_pruning'] = time.time() - t0
        print(f"  Skeleton pixels: {n_before} -> {n_after} ({n_before - n_after} pruned)")
        
        # Re-extract skeleton point cloud from pruned skeleton (used by repulsion)
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        
        # Phase 6: Compute segment structure from pruned skeleton
        # Use leaf_2d_updated so EE leaves snap to skeleton near their injected positions
        print("Phase 6: Computing segment structure...")
        t0 = time.time()
        segment_structure = self._compute_skeleton_segment_structure(
            branch_2d, leaf_2d_updated, skeleton_mask, depth
        )
        timing['segment_structure'] = time.time() - t0
        
        if segment_structure is None:
            return {
                'success': False,
                'reason': 'failed_to_compute_segment_structure',
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
            }
        
        # Phase 7: Order segments by EE mapping
        print("Phase 7: Ordering segments by EE mapping...")
        t0 = time.time()
        ordered_segments, b_ee0, b_ee1 = self._order_segments_by_ee(
            segment_structure, ee_to_leaf_kp, n_branch
        )
        timing['segment_ordering'] = time.time() - t0
        
        # Extract ordered skeleton segment lengths (for reference only)
        skeleton_segment_lengths = [seg['length'] for seg in ordered_segments]
        print(f"  Found {len(skeleton_segment_lengths)} segments")
        for i, seg in enumerate(ordered_segments):
            print(f"    Segment {i} ({seg['type']:10s}): skeleton_length={seg['length']:.1f} mm, kp {seg['start_kp']} -> {seg['end_kp']}")
        print(f"  Total skeleton wire length: {sum(skeleton_segment_lengths):.1f} mm")
        
        # Build MST skeleton mask for visualization
        skeleton_coords = segment_structure['skeleton_coords']
        mst_skeleton_mask = np.zeros_like(skeleton_mask, dtype=np.uint8)
        mst_indices_set = set()
        for seg in ordered_segments:
            mst_indices_set.update(seg['path'])
        for idx in mst_indices_set:
            r, c = skeleton_coords[idx]
            if 0 <= r < mst_skeleton_mask.shape[0] and 0 <= c < mst_skeleton_mask.shape[1]:
                mst_skeleton_mask[r, c] = 1
        
        # Phase 8: Place keypoints using global FPS on MST with anchor seeds
        print("Phase 8: Placing keypoints (global FPS on MST)...")
        t0 = time.time()
        
        # CRITICAL: Snap EE leaf positions to nearest skeleton point
        # The EE gripper position may be off the skeleton surface, causing
        # misalignment between anchors and FPS interior points.
        skeleton_3d = segment_structure['skeleton_3d']
        mst_indices_set = set()
        for seg in ordered_segments:
            mst_indices_set.update(seg['path'])
        mst_skeleton_3d = skeleton_3d[sorted(mst_indices_set)]
        
        leaf_3d_snapped = leaf_3d.copy()
        if self.ee_poses_3d is not None and self.ee_to_leaf_mapping:
            for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
                leaf_local_idx = kp_idx - n_branch
                ee_pos = leaf_3d[leaf_local_idx]
                
                # Find nearest MST skeleton point
                dists = np.linalg.norm(mst_skeleton_3d - ee_pos, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_skel_pos = mst_skeleton_3d[nearest_idx]
                snap_dist = dists[nearest_idx]
                
                print(f"  EE {ee_idx} (leaf {leaf_local_idx}): snap to skeleton, dist={snap_dist:.2f}mm")
                print(f"    Original: {ee_pos}")
                print(f"    Snapped:  {nearest_skel_pos}")
                
                leaf_3d_snapped[leaf_local_idx] = nearest_skel_pos
        
        keypoints, segment_edges_raw = self._place_keypoints_along_ordered_segments(
            ordered_segments,
            branch_3d, leaf_3d_snapped, segment_structure
        )
        timing['keypoint_placement'] = time.time() - t0
        print(f"  Placed {len(keypoints)} keypoints")
        
        # Flatten segment edges to get all edges
        edges = []
        for seg in segment_edges_raw:
            edges.extend(seg)
        print(f"  Built {len(edges)} edges across {len(segment_edges_raw)} segments")
        
        # Compute segment lengths from actual edge sums (post-FPS)
        segment_3d_lengths = []
        for seg in segment_edges_raw:
            seg_len = sum(np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg)
            segment_3d_lengths.append(seg_len)
        
        edges_per_segment = [len(seg) for seg in segment_edges_raw]
        print(f"  Edges per segment: {edges_per_segment}")
        print(f"  Segment lengths (edge-sum): {[f'{l:.1f}' for l in segment_3d_lengths]} mm")
        print(f"  Total wire length (edge-sum): {sum(segment_3d_lengths):.1f} mm")
        
        # Store reference edges/lengths
        self.reference_edges = edges
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        
        # Store segment edges (already ordered)
        self.segment_edges = segment_edges_raw
        self.anchor_set = set(range(n_branch + n_leaf))
        
        # Identify free leaf indices
        self.free_leaf_indices = []
        for seg in ordered_segments:
            if seg['type'] == 'free_leaf':
                self.free_leaf_indices.append(seg['start_kp'])
        
        # Phase 9: Keypoint refinement (FPS+repulsion or GMM)
        print(f"Phase 9: Keypoint refinement (method={self.placement_method})...")
        t0 = time.time()
        fixed_mask = np.zeros(len(keypoints), dtype=bool)
        fixed_mask[:n_branch + n_leaf] = True
        
        if self.placement_method == 'gmm':
            keypoints = self._gmm_keypoint_placement(
                keypoints, skeleton_pc, fixed_mask, edges,
                segment_edges_raw, ordered_segments, segment_structure,
            )
        else:
            keypoints = self._repulsion_relaxation_with_topology(
                keypoints, skeleton_pc, fixed_mask, edges, segment_edges_raw,
                segment_lengths=skeleton_segment_lengths,
            )
        timing['refinement'] = time.time() - t0
        print(f"  Refinement done ({self.placement_method})")
        
        # Update reference keypoints after repulsion
        self.reference_keypoints = keypoints.copy()
        
        # Recompute edge lengths after repulsion
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        
        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        timing['total'] = time.time() - total_start
        
        # Print segment summary
        self._print_segment_summary(keypoints, edges)
        
        print(f"\nInitialization complete in {timing['total']:.3f}s")
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'reference_lengths': self.reference_lengths,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'foreground_mask': foreground_mask,
            'skeleton_mask': skeleton_mask,
            'skeleton_mask_raw': skeleton_mask_raw,
            'mst_skeleton_mask': mst_skeleton_mask,
            'segment_edges': self.segment_edges,
            'anchor_set': self.anchor_set,
            'free_leaf_indices': self.free_leaf_indices,
            'ee_to_leaf_mapping': self.ee_to_leaf_mapping,
            'segment_3d_lengths': segment_3d_lengths,
            'edges_per_segment': edges_per_segment,
            'timing': timing,
        }
    
    def _reorder_segment_edges_by_ee(
        self,
        segment_edges_raw: List[List[Tuple[int, int]]],
        segment_structure: dict,
    ) -> None:
        """
        Reorder segment edges based on EE mapping to ensure consistent ordering.
        
        Final order:
            Segment 0: ee_leaf with EE 0 → its branch (b_ee0)
            Segment 1: ee_leaf with EE 1 → its branch (b_ee1)
            Segment 2: free_leaf connected to b_ee0
            Segment 3: free_leaf connected to b_ee1
            Segment 4: trunk (b_ee0 → b_ee1)
        """
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        
        segment_endpoints = segment_structure['segment_endpoints']
        leaf_to_branch = segment_structure['leaf_to_branch']
        
        # Identify leaf indices
        leaf_indices = list(range(n_branch, n_branch + n_leaf))
        
        # Build mapping from EE to leaf keypoint
        if self.ee_to_leaf_mapping is None:
            # No EE mapping, use raw order
            self.segment_edges = segment_edges_raw
            self.anchor_set = set(range(n_branch + n_leaf))
            self.free_leaf_indices = []
            return
        
        # Find which branch each EE's leaf connects to
        ee_to_branch = {}
        for ee_idx, leaf_kp in self.ee_to_leaf_mapping.items():
            # leaf_kp is the keypoint index, find which segment endpoint this is
            leaf_local_idx = leaf_kp - n_branch  # local index within leaves
            if leaf_local_idx in leaf_to_branch:
                ee_to_branch[ee_idx] = leaf_to_branch[leaf_local_idx]
        
        b_ee0 = ee_to_branch.get(0, 0)
        b_ee1 = ee_to_branch.get(1, 1 if len(segment_structure['branch_skel_indices']) > 1 else 0)
        
        # Categorize segments
        ee_leaf_segments = {}  # ee_idx → (seg_idx, edges)
        free_leaf_segments = {}  # branch_idx → (seg_idx, edges)
        trunk_segment = None
        
        for seg_idx, (endpoints, edges) in enumerate(zip(segment_endpoints, segment_edges_raw)):
            start_type, start_idx, end_type, end_idx = endpoints
            
            if start_type == 'leaf' and end_type == 'branch':
                leaf_kp = n_branch + start_idx
                branch_idx = end_idx
                
                # Check if this leaf is an EE leaf
                ee_idx = None
                for e_idx, l_kp in self.ee_to_leaf_mapping.items():
                    if l_kp == leaf_kp:
                        ee_idx = e_idx
                        break
                
                if ee_idx is not None:
                    ee_leaf_segments[ee_idx] = (seg_idx, edges, branch_idx)
                else:
                    free_leaf_segments[branch_idx] = (seg_idx, edges, start_idx)
            
            elif start_type == 'branch' and end_type == 'branch':
                trunk_segment = (seg_idx, edges)
        
        # Reorder segments
        ordered_segments = []
        anchor_set = set(range(n_branch + n_leaf))
        free_leaf_indices = []
        
        # Segment 0: ee_leaf with EE 0
        if 0 in ee_leaf_segments:
            _, edges, _ = ee_leaf_segments[0]
            ordered_segments.append(edges)
            print(f"  Segment 0 (ee_leaf): EE0, {len(edges)} edges")
        
        # Segment 1: ee_leaf with EE 1
        if 1 in ee_leaf_segments:
            _, edges, _ = ee_leaf_segments[1]
            ordered_segments.append(edges)
            print(f"  Segment 1 (ee_leaf): EE1, {len(edges)} edges")
        
        # Segment 2: free_leaf connected to b_ee0
        if b_ee0 in free_leaf_segments:
            _, edges, leaf_local_idx = free_leaf_segments[b_ee0]
            ordered_segments.append(edges)
            free_leaf_indices.append(n_branch + leaf_local_idx)
            print(f"  Segment 2 (free_leaf): branch {b_ee0}, {len(edges)} edges")
        
        # Segment 3: free_leaf connected to b_ee1
        if b_ee1 in free_leaf_segments and b_ee1 != b_ee0:
            _, edges, leaf_local_idx = free_leaf_segments[b_ee1]
            ordered_segments.append(edges)
            free_leaf_indices.append(n_branch + leaf_local_idx)
            print(f"  Segment 3 (free_leaf): branch {b_ee1}, {len(edges)} edges")
        
        # Segment 4: trunk
        if trunk_segment is not None:
            _, edges = trunk_segment
            ordered_segments.append(edges)
            print(f"  Segment 4 (trunk): {len(edges)} edges")
        
        self.segment_edges = ordered_segments
        self.anchor_set = anchor_set
        self.free_leaf_indices = free_leaf_indices
    
    def initialize_streamlined(
        self,
        depth: np.ndarray,
        arm_depth: np.ndarray = None,
        precomputed_arm_mask: np.ndarray = None,
        n_keypoints_per_segment: List[int] = None,
    ) -> dict:
        """
        Streamlined initialization pipeline with unified Build Topology phase.
        
        Pipeline:
            Phase 1: Segmentation (background subtraction + skeletonization)
            Phase 2: Node identification (branch/leaf from MST degree)
            Phase 3: Topology pruning (prune to target branch/leaf count)
            Phase 4: 2D → 3D conversion for branch/leaf
            Phase 5: EE mapping & injection
            Phase 6: Build Topology (unified phase):
                6.1: Find MST paths, order by EE
                6.2: Estimate segment lengths (downsampled)
                6.3: Allocate keypoints per segment
                6.4: FPS per segment with anchor seeds
                6.5: Connect keypoints (greedy NN)
            Phase 7: Keypoint refinement (repulsion)
        
        Args:
            depth: H × W current frame depth
            arm_depth: H × W arm-only depth (not needed if precomputed_arm_mask provided)
            precomputed_arm_mask: H × W binary mask where 1=arm, 0=keep
            n_keypoints_per_segment: Optional explicit keypoints per segment (length=5)
        
        Returns:
            dict with success, keypoints, edges, segment_edges, etc.
        """
        timing = {}
        total_start = time.time()
        
        # Phase 1: Segmentation
        print("Phase 1: Segmentation...")
        t0 = time.time()
        seg_result = self.segment(depth, arm_depth, precomputed_arm_mask)
        timing['segmentation'] = time.time() - t0
        
        foreground_mask = seg_result['foreground_mask']
        skeleton_mask = seg_result['skeleton_mask']
        skeleton_pc = seg_result['skeleton_pc']
        
        n_skeleton_pixels = np.sum(skeleton_mask > 0)
        print(f"  Skeleton pixels: {n_skeleton_pixels}")
        
        if n_skeleton_pixels < self.min_skeleton_pixels:
            return {
                'success': False,
                'reason': f'insufficient_skeleton_pixels ({n_skeleton_pixels} < {self.min_skeleton_pixels})',
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
            }
        
        # Phase 2: Node identification
        print("Phase 2: Node identification...")
        t0 = time.time()
        branch_2d, leaf_2d, adjacency, coords = self._node_identification(skeleton_mask)
        timing['node_detection'] = time.time() - t0
        print(f"  Detected: {len(branch_2d)} branch, {len(leaf_2d)} leaf nodes")
        
        # Phase 3: Topology pruning
        print("Phase 3: Topology pruning...")
        t0 = time.time()
        if adjacency is not None:
            pruned = self._prune_to_target_topology(adjacency, coords)
            branch_2d = pruned["branch_coords"]
            leaf_2d = pruned["leaf_coords"]
        timing['pruning'] = time.time() - t0
        print(f"  After pruning: {len(branch_2d)} branch, {len(leaf_2d)} leaf nodes")
        
        # Phase 4: 2D → 3D for branch/leaf
        print("Phase 4: 2D → 3D conversion...")
        t0 = time.time()
        branch_3d = self._pixel_to_3d(branch_2d, depth)
        leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        timing['2d_to_3d'] = time.time() - t0
        
        # Store reference counts
        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf
        
        # Phase 5: EE mapping & injection
        print("Phase 5: EE mapping and injection...")
        t0 = time.time()
        ee_to_leaf_kp = self._establish_ee_mapping_from_leaves(leaf_3d, n_branch)
        
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            self.ee_to_leaf_mapping = ee_to_leaf_kp
            ee_positions = self.ee_poses_3d[0]  # Frame 0: (2, 3)
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                old_pos = leaf_3d[leaf_local_idx].copy()
                leaf_3d[leaf_local_idx] = ee_positions[ee_idx]
                dist = np.linalg.norm(old_pos - ee_positions[ee_idx])
                print(f"  EE {ee_idx} -> Leaf keypoint {kp_idx} (shift={dist:.1f} mm)")
        else:
            self.ee_to_leaf_mapping = ee_to_leaf_kp if ee_to_leaf_kp else None
            if ee_to_leaf_kp:
                for ee_idx, leaf_kp in ee_to_leaf_kp.items():
                    print(f"  EE {ee_idx} -> Leaf keypoint {leaf_kp} (no pose injection)")
            else:
                print("  No EE poses provided, skipping")
        timing['ee_injection'] = time.time() - t0
        
        # Update leaf_2d for EE leaves (project EE 3D → 2D)
        leaf_2d_updated = leaf_2d.copy().astype(np.float64)
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                ee_2d = self._project_3d_to_2d(self.ee_poses_3d[0][ee_idx:ee_idx+1])
                leaf_2d_updated[leaf_local_idx] = ee_2d[0]
        
        # Keep raw skeleton for viz
        skeleton_mask_raw = skeleton_mask.copy()
        
        # Prune skeleton to only keep the 5 MST segment paths (like Phase 5b in original)
        print("  Pruning skeleton to MST segment paths...")
        skeleton_mask, n_before, n_after = self._prune_skeleton_to_node_paths(
            skeleton_mask, branch_2d, leaf_2d_updated
        )
        print(f"    Skeleton pixels: {n_before} -> {n_after} ({n_before - n_after} pruned)")
        
        # Re-extract skeleton point cloud from pruned skeleton (used by repulsion)
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        
        # Phase 6: Build Topology (unified)
        print("Phase 6: Build Topology (unified)...")
        t0 = time.time()
        try:
            keypoints, edges, segment_edges, ordered_segments = self._build_topology(
                branch_3d, leaf_3d, branch_2d, leaf_2d_updated,
                skeleton_mask, depth, ee_to_leaf_kp,
                n_keypoints_per_segment=n_keypoints_per_segment,
            )
        except ValueError as e:
            return {
                'success': False,
                'reason': str(e),
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
            }
        timing['build_topology'] = time.time() - t0
        
        # Store reference edges/lengths
        self.reference_edges = edges
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        
        # Store segment edges
        self.segment_edges = segment_edges
        self.anchor_set = set(range(n_branch + n_leaf))
        
        # Identify free leaf indices
        self.free_leaf_indices = []
        for seg in ordered_segments:
            if seg['type'] == 'free_leaf':
                self.free_leaf_indices.append(seg['start_kp'])
        
        # Compute segment lengths from edge sums
        segment_3d_lengths = []
        for seg in segment_edges:
            seg_len = sum(np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg)
            segment_3d_lengths.append(seg_len)
        
        edges_per_segment = [len(seg) for seg in segment_edges]
        print(f"  Segment lengths (edge-sum): {[f'{l:.1f}' for l in segment_3d_lengths]} mm")
        print(f"  Total wire length (edge-sum): {sum(segment_3d_lengths):.1f} mm")
        
        # Re-extract skeleton point cloud for repulsion
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        
        # Get skeleton-based segment lengths for repulsion targets
        skeleton_segment_lengths = [seg.get('estimated_length', seg.get('length', 0)) for seg in ordered_segments]
        
        # Phase 7: Keypoint refinement (repulsion)
        print(f"Phase 7: Keypoint refinement (repulsion)...")
        t0 = time.time()
        fixed_mask = np.zeros(len(keypoints), dtype=bool)
        fixed_mask[:n_branch + n_leaf] = True
        
        keypoints = self._repulsion_relaxation_with_topology(
            keypoints, skeleton_pc, fixed_mask, edges, segment_edges,
            segment_lengths=skeleton_segment_lengths,
        )
        timing['refinement'] = time.time() - t0
        print(f"  Refinement done")
        
        # Update reference keypoints after repulsion
        self.reference_keypoints = keypoints.copy()
        
        # Recompute edge lengths after repulsion
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        
        # Project to 2D
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        timing['total'] = time.time() - total_start
        
        # Print segment summary
        self._print_segment_summary(keypoints, edges)
        
        print(f"\nInitialization complete in {timing['total']:.3f}s")
        
        # Build MST skeleton mask for visualization
        mst_skeleton_mask = skeleton_mask.copy()  # After build_topology, skeleton is already pruned
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'reference_lengths': self.reference_lengths,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'foreground_mask': foreground_mask,
            'skeleton_mask': skeleton_mask,
            'skeleton_mask_raw': skeleton_mask_raw,
            'mst_skeleton_mask': mst_skeleton_mask,
            'segment_edges': self.segment_edges,
            'anchor_set': self.anchor_set,
            'free_leaf_indices': self.free_leaf_indices,
            'ee_to_leaf_mapping': self.ee_to_leaf_mapping,
            'segment_3d_lengths': segment_3d_lengths,
            'skeleton_segment_lengths': skeleton_segment_lengths,  # Target lengths for repulsion
            'edges_per_segment': edges_per_segment,
            'timing': timing,
        }
    
    def get_state(self) -> dict:
        """Get current initialization state for use in tracking."""
        return {
            'reference_keypoints': self.reference_keypoints,
            'reference_edges': self.reference_edges,
            'reference_lengths': self.reference_lengths,
            'reference_n_branch': self.reference_n_branch,
            'reference_n_leaf': self.reference_n_leaf,
            'segment_edges': self.segment_edges,
            'anchor_set': self.anchor_set,
            'free_leaf_indices': self.free_leaf_indices,
            'ee_to_leaf_mapping': self.ee_to_leaf_mapping,
        }
