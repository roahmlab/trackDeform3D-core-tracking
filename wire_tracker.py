"""
WireTracker: Unified wire tracking with CPD and geometry constraints.

Pipeline:
    Phase 1: Segmentation (every frame)
        - Background subtraction + depth threshold + top-K CC + skeletonization
    
    Phase 2: Initialization (Frame 0 only)
        - Node identification + pruning + FPS + repulsion + topology extraction
    
    Phase 3: CPD Tracking (Frame N > 0)
        - Full CPD → Hungarian replace anchors → Joint edge + projection constraints

Author: Auto-generated
Date: 2026-02-14
"""

import math
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

# Import WireInitializer for segment-aware keypoint allocation
from wire_initializer_combined import WireInitializer


class WireTracker:
    """
    Unified wire tracking with CPD and geometry constraints.
    
    Keypoint ordering: [branch_0, ..., branch_{N_b-1}, leaf_0, ..., leaf_{N_l-1}, intermediate...]
    
    Usage:
        tracker = WireTracker(intrinsics)
        for frame_idx, (depth, arm_depth, rgb) in enumerate(frames):
            result = tracker.process_frame(depth, arm_depth, rgb)
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
        # CPD parameters
        cpd_beta: float = 10.0,
        cpd_lambda: float = 2.0,
        cpd_w: float = 0.1,
        cpd_max_iter: int = 100,
        cpd_tol: float = 1e-3,
        cpd_downsample: int = 500,
        # Geometry constraint parameters
        n_outer_iterations: int = 5,
        n_edge_iterations: int = 20,
        edge_weight: float = 0.5,
        edge_tolerance: float = 0.15,
        # DLO balance tuning (BDLO-style projection ↔ edge loop)
        dlo_proj_blend_start: float = 0.45,
        dlo_proj_blend_end: float = 0.20,
        dlo_balance_stages: int = 4,
        # Repulsion parameters
        repulsion_iterations: int = 40,
        repulsion_lr: float = 5.0,
        repulsion_k_neighbors: int = 3,
        # Warm restart
        max_skips_before_restart: int = 3,
        min_skeleton_pixels: int = 100,
        # Ablation flags
        enable_cpd: bool = True,
        enable_node_matching: bool = True,
        enable_geometry_constraint: bool = True,
        enable_ee_injection: bool = True,
        # End-effector pose injection
        ee_poses_3d: np.ndarray = None,
        # BDLO segment keypoint allocation
        keypoints_per_segment: list = None,
    ):
        """
        Initialize WireTracker.
        
        Args:
            intrinsics: 3×3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
            n_keypoints: Total number of keypoints to track
            target_branch_nodes: Target number of branch nodes (Frame 0 pruning)
            target_leaf_nodes: Target number of leaf nodes (Frame 0 pruning)
            bg_threshold: Background subtraction distance threshold (mm)
            max_depth: Maximum valid depth (mm)
            top_k_components: Number of largest connected components to keep
            arm_dilation_pixels: Pixels to dilate arm mask
            cpd_beta: CPD motion coherence (larger = more rigid)
            cpd_lambda: CPD regularization weight
            cpd_w: CPD outlier weight [0, 1]
            cpd_max_iter: CPD maximum iterations
            cpd_tol: CPD convergence tolerance
            cpd_downsample: Maximum points for CPD target
            n_outer_iterations: Edge + projection cycles
            n_edge_iterations: Edge constraint iterations per cycle
            edge_weight: Edge constraint strength [0, 1]
            edge_tolerance: Allowed edge length deviation fraction
            dlo_proj_blend_start: Initial projection blend for DLO two-stage balancing
            dlo_proj_blend_end: Final projection blend for DLO two-stage balancing
            dlo_balance_stages: Number of projection→edge stages per tracking frame
            repulsion_iterations: Repulsion relaxation iterations
            repulsion_lr: Repulsion learning rate
            repulsion_k_neighbors: Neighbors for repulsion
            max_skips_before_restart: Consecutive skips before warm restart
            min_skeleton_pixels: Minimum skeleton pixels to process
            enable_node_matching: Enable Hungarian node matching (ablation flag)
            enable_geometry_constraint: Enable geometry constraint optimization (ablation flag)
            enable_ee_injection: Enable replacing mapped leaf nodes with EE poses
            min_skeleton_pixels: Minimum skeleton pixels to process
            ee_poses_3d: (n_frames, 2, 3) array of end-effector 3D positions. If provided,
                         two leaf nodes will be replaced with EE positions after node matching.
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
        self.dlo_proj_blend_start = dlo_proj_blend_start
        self.dlo_proj_blend_end = dlo_proj_blend_end
        self.dlo_balance_stages = dlo_balance_stages
        
        # Repulsion parameters
        self.repulsion_iterations = repulsion_iterations
        self.repulsion_lr = repulsion_lr
        self.repulsion_k_neighbors = repulsion_k_neighbors
        
        # Warm restart parameters
        self.max_skips_before_restart = max_skips_before_restart
        self.min_skeleton_pixels = min_skeleton_pixels
        
        # Ablation flags
        self.enable_cpd = enable_cpd
        self.enable_node_matching = enable_node_matching
        self.enable_geometry_constraint = enable_geometry_constraint
        self.enable_ee_injection = enable_ee_injection
        
        # End-effector pose injection
        self.ee_poses_3d = ee_poses_3d  # (n_frames, 2, 3) or None
        self.ee_to_leaf_mapping = None  # {0: kp_idx, 1: kp_idx} set during initialization
        
        # BDLO segment keypoint allocation
        self.keypoints_per_segment = keypoints_per_segment  # [ee0, ee1, free0, free1, trunk] or None
        
        # Auto-compute n_keypoints from keypoints_per_segment if provided
        if keypoints_per_segment is not None:
            # n_keypoints = branch_nodes + leaf_nodes + sum(intermediate_keypoints)
            computed_n = target_branch_nodes + target_leaf_nodes + sum(keypoints_per_segment)
            if n_keypoints != computed_n:
                print(f"[WireTracker] keypoints_per_segment={keypoints_per_segment} implies n_keypoints={computed_n}")
                print(f"              (overriding provided n_keypoints={n_keypoints})")
                self.n_keypoints = computed_n
        
        # State (set during initialization)
        self.reference_keypoints = None    # K × 3
        self.reference_edges = None        # List of (i, j)
        self.reference_lengths = None      # Array of edge lengths
        self.reference_n_branch = 0
        self.reference_n_leaf = 0
        self.prev_keypoints = None         # K × 3
        self.clean_path_mask = None        # H × W clean BFS path mask for DLO
        self.consecutive_skips = 0
        self.is_initialized = False
        self.frame_count = 0
        
        # Segment-ordered edges for geometry correction (set during initialization)
        self.segment_edges = None          # List of 5 lists of ordered (i, j) edges
        self.anchor_set = None             # Set of anchor keypoint indices
        self.free_leaf_indices = None      # List of free leaf indices
    
    # ================================================================
    # PHASE 1: SEGMENTATION
    # ================================================================
    
    def _depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.
        
        Args:
            depth: H × W depth image (mm)
        
        Returns:
            points_3d: H × W × 3 point cloud
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        z = depth.astype(np.float64)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.stack([x, y, z], axis=-1)
    
    def _expand_arm_depth(self, arm_depth: np.ndarray) -> np.ndarray:
        """
        Expand arm depth by dilating valid region and filling with nearest values.
        
        Args:
            arm_depth: H × W arm depth image
        
        Returns:
            expanded_depth: H × W expanded depth image
        """
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
        """
        Remove robot arm via 3D point cloud distance.
        
        Args:
            full_pc: H × W × 3 full scene point cloud
            arm_pc: H × W × 3 arm point cloud
        
        Returns:
            foreground_mask: H × W binary mask (1 = foreground)
        """
        diff = np.linalg.norm(full_pc - arm_pc, axis=-1)
        foreground_mask = (diff > self.bg_threshold).astype(np.uint8)
        return foreground_mask
    
    def _apply_depth_threshold(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """
        Filter mask by valid depth range.
        
        Args:
            mask: H × W binary mask
            depth: H × W depth image
        
        Returns:
            filtered_mask: H × W filtered mask
        """
        filtered = mask.copy()
        filtered[depth > self.max_depth] = 0
        filtered[depth <= 0] = 0
        filtered[np.isnan(depth)] = 0
        filtered[np.isinf(depth)] = 0
        return filtered
    
    def _get_top_k_components(self, mask: np.ndarray, k: int = None) -> np.ndarray:
        """
        Keep only the k largest connected components.
        
        Args:
            mask: H × W binary mask
            k: Number of components to keep (default: self.top_k_components)
        
        Returns:
            filtered_mask: H × W mask with top-k components
        """
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
        """
        Skeletonize binary mask.
        
        Args:
            mask: H × W binary mask
        
        Returns:
            skeleton: H × W skeleton mask
        """
        return skeletonize(mask > 0).astype(np.uint8)
    
    def segment(self, depth: np.ndarray, arm_depth: np.ndarray = None, n_components: int = None,
                 precomputed_arm_mask: np.ndarray = None, is_init: bool = False) -> dict:
        """
        Phase 1: Create foreground and skeleton masks.
        
        Args:
            depth: H × W current frame depth
            arm_depth: H × W arm-only depth (not needed if precomputed_arm_mask provided)
            n_components: Number of connected components to keep (default: self.top_k_components)
                          Use n_components=1 for frame 0 initialization
            precomputed_arm_mask: H × W binary mask where 1=arm (to be removed), 0=keep
                                  If provided, skips background subtraction and uses this directly.
        
        Returns:
            dict with 'foreground_mask', 'skeleton_mask', 'skeleton_pc', 'seg_time'
        """
        import time
        
        t_start = time.time()
        
        if precomputed_arm_mask is not None:
            # Use precomputed mask directly: arm pixels are 1, wire/foreground is 0
            # Foreground = NOT arm AND valid depth
            foreground_mask = ((precomputed_arm_mask == 0) & (depth > 0)).astype(np.uint8)
        else:
            # Original path: compute from arm_depth
            # Expand arm depth (not counted in timing - can be pre-computed)
            arm_depth_expanded = self._expand_arm_depth(arm_depth)
            
            # Convert to point clouds (not counted in timing - can be pre-computed)
            full_pc = self._depth_to_point_cloud(depth)
            arm_pc = self._depth_to_point_cloud(arm_depth_expanded)
            
            # Background subtraction
            foreground_mask = self._background_subtraction(full_pc, arm_pc)
        
        # Depth thresholding
        foreground_mask = self._apply_depth_threshold(foreground_mask, depth)
        
        # Top-K: k=1 for frame 0 (init), skip filtering for tracking frames
        if is_init:
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
        
        return np.array(coords_3d, dtype=np.float64) if coords_3d else np.empty((0, 3), dtype=np.float64)
    
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
    
    def _extract_point_cloud(
        self, mask: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
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
    
    # ================================================================
    # NODE IDENTIFICATION
    # ================================================================
    
    def _node_identification(self, skeleton_mask: np.ndarray) -> tuple:
        """
        Identify branch and leaf nodes via MST degree analysis.
        
        Args:
            skeleton_mask: H × W skeleton mask
        
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
        
        Args:
            adjacency: N × N adjacency matrix
            coords: N × 2 coordinates
        
        Returns:
            dict with 'adjacency', 'coords', 'branch_coords', 'leaf_coords'
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
        
        # Second pass: merge close branch nodes to reach target_branch_nodes
        # Find pairs of branches connected by degree-2 chains (trunks),
        # merge the shortest trunk by collapsing one branch into the other.
        for _ in range(max_iterations):
            degrees = np.zeros(n, dtype=np.int64)
            for i in range(n):
                if active[i]:
                    degrees[i] = np.sum((adjacency[i, :] > 0) & active)

            branch_indices = np.where((degrees >= 3) & active)[0]
            if len(branch_indices) <= self.target_branch_nodes:
                break

            # Find shortest trunk between any two branch nodes
            best_trunk = None
            best_length = np.inf

            for b_idx in branch_indices:
                for neighbor in np.where((adjacency[b_idx, :] > 0) & active)[0]:
                    path = [b_idx, neighbor]
                    path_length = adjacency[b_idx, neighbor]
                    current = neighbor
                    prev = b_idx

                    # Follow degree-2 chain
                    while degrees[current] == 2:
                        nexts = np.where((adjacency[current, :] > 0) & active)[0]
                        nexts = [nb for nb in nexts if nb != prev]
                        if len(nexts) == 0:
                            break
                        prev = current
                        current = nexts[0]
                        path_length += adjacency[prev, current]
                        path.append(current)

                    # Check if we reached another branch node
                    if degrees[current] >= 3 and current != b_idx:
                        if path_length < best_length:
                            best_length = path_length
                            best_trunk = path

            if best_trunk is None:
                break

            # Merge: keep first branch, remove trunk interior + second branch
            keep_branch = best_trunk[0]
            remove_branch = best_trunk[-1]
            trunk_interior = best_trunk[1:-1]

            # Deactivate trunk interior nodes
            for node in trunk_interior:
                active[node] = False
                adjacency[node, :] = 0
                adjacency[:, node] = 0

            # Redirect removed branch's connections to kept branch
            for neighbor in np.where((adjacency[remove_branch, :] > 0) & active)[0]:
                if neighbor != keep_branch:
                    dist = adjacency[remove_branch, neighbor]
                    adjacency[keep_branch, neighbor] = dist + best_length
                    adjacency[neighbor, keep_branch] = dist + best_length

            # Deactivate removed branch
            active[remove_branch] = False
            adjacency[remove_branch, :] = 0
            adjacency[:, remove_branch] = 0

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
    # SKELETON GRAPH UTILITIES (for edge validation)
    # ================================================================
    
    def _build_skeleton_graph(
        self, skeleton_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[List[Tuple[int, float]]]]:
        """
        Build a graph representation of the skeleton mask for path queries.
        
        Args:
            skeleton_mask: H × W skeleton mask
        
        Returns:
            coords: N × 2 skeleton pixel coordinates (row, col)
            coord_to_idx: dict mapping (row, col) to node index
            adjacency: list of list of (neighbor_idx, distance)
        """
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
    
    def _snap_to_skeleton(
        self, 
        pixel: Tuple[int, int], 
        skeleton_coords: np.ndarray,
        skeleton_tree: Optional[KDTree] = None
    ) -> int:
        """
        Find the nearest skeleton pixel index to a given pixel.
        
        Args:
            pixel: (row, col) tuple
            skeleton_coords: N × 2 skeleton coordinates
            skeleton_tree: optional KDTree for faster lookup
        
        Returns:
            index of nearest skeleton pixel, or -1 if empty
        """
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
        """
        Check if a path exists in skeleton graph from start to goal,
        without passing through blocked nodes.
        
        Args:
            adjacency: skeleton graph adjacency list
            start: start skeleton node index
            goal: goal skeleton node index
            blocked: set of skeleton node indices to avoid
        
        Returns:
            True if path exists
        """
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
    
    def _find_skeleton_path(
        self,
        adjacency: List[List[Tuple[int, float]]],
        start: int,
        goal: int,
    ) -> List[int]:
        """
        Find the actual path from start to goal in the skeleton graph.
        
        Args:
            adjacency: skeleton graph adjacency list
            start: start skeleton node index
            goal: goal skeleton node index
        
        Returns:
            List of node indices from start to goal, or empty list if no path
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
        
        while queue:
            node = queue.pop(0)
            
            for nbr, _ in adjacency[node]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                parent[nbr] = node
                
                if nbr == goal:
                    # Reconstruct path
                    path = [goal]
                    current = goal
                    while current != start:
                        current = parent[current]
                        path.append(current)
                    path.reverse()
                    return path
                
                queue.append(nbr)
        
        return []  # No path found
    
    def _get_keypoint_blocked_regions(
        self,
        keypoints: np.ndarray,
        skeleton_coords: np.ndarray,
        skeleton_tree: Optional[KDTree],
        block_radius: int = 3,
    ) -> Tuple[np.ndarray, List[Set[int]]]:
        """
        For each keypoint, find its skeleton index and the set of skeleton
        indices within block_radius that should be blocked.
        
        Args:
            keypoints: K × 3 keypoint positions
            skeleton_coords: N × 2 skeleton coordinates
            skeleton_tree: KDTree for skeleton
            block_radius: pixel radius to block around each keypoint
        
        Returns:
            keypoint_skel_indices: K array of skeleton indices
            blocked_regions: list of K sets, each containing blocked skeleton indices
        """
        K = keypoints.shape[0]
        keypoint_skel_indices = np.zeros(K, dtype=int)
        blocked_regions: List[Set[int]] = []
        
        H, W = 720, 1280  # Assume standard resolution; will be clipped anyway
        
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
    
    def _fps_with_anchors(
        self, points: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """
        Farthest Point Sampling with anchor points as initial seeds.
        
        Args:
            points: N × 3 point cloud
            anchors: A × 3 anchor points
        
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
    
    def _find_valid_neighbors(
        self,
        keypoints: np.ndarray,
        skeleton_mask: np.ndarray,
        k_neighbors: int = 6,
        block_radius: int = 3,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """
        Find valid neighbors for each keypoint using skeleton path validation.
        
        For each KNN pair, check if a path exists in the skeleton without
        passing through other keypoints.
        
        Args:
            keypoints: K × 3 keypoint positions
            skeleton_mask: H × W skeleton mask
            k_neighbors: number of KNN neighbors to consider
            block_radius: pixel radius to block around each keypoint
        
        Returns:
            valid_neighbors: list of list of valid neighbor indices for each keypoint
            valid_edges: list of (i, j) tuples for valid edges
        """
        K = keypoints.shape[0]
        
        if K < 2:
            return [[] for _ in range(K)], []
        
        # Build skeleton graph
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if skeleton_coords.shape[0] == 0:
            # Fallback to simple KNN if no skeleton
            return [[] for _ in range(K)], []
        
        # Build KDTree for skeleton
        skeleton_tree = KDTree(skeleton_coords) if skeleton_coords.shape[0] > 1 else None
        
        # Get skeleton indices and blocked regions for each keypoint
        keypoint_skel_indices, blocked_regions = self._get_keypoint_blocked_regions(
            keypoints, skeleton_coords, skeleton_tree, block_radius
        )
        
        # KNN in 3D space
        k = min(k_neighbors + 1, K)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(keypoints)
        _, knn_indices = nn.kneighbors(keypoints)
        
        # Union of all blocked regions
        all_blocked: Set[int] = set()
        for region in blocked_regions:
            all_blocked.update(region)
        
        # Collect candidate pairs
        candidate_pairs: Set[Tuple[int, int]] = set()
        for i in range(K):
            for j in knn_indices[i, 1:]:
                if j != i:
                    candidate_pairs.add((min(i, j), max(i, j)))
        
        # Validate each pair via skeleton path
        valid_edges: List[Tuple[int, int]] = []
        
        for i, j in candidate_pairs:
            skel_i = keypoint_skel_indices[i]
            skel_j = keypoint_skel_indices[j]
            
            if skel_i < 0 or skel_j < 0:
                continue
            
            # Blocked = all blocked EXCEPT i and j's regions
            blocked = all_blocked - blocked_regions[i] - blocked_regions[j]
            
            if self._skeleton_path_exists(adjacency, skel_i, skel_j, blocked):
                valid_edges.append((i, j))
        
        # Build neighbor lists
        valid_neighbors: List[List[int]] = [[] for _ in range(K)]
        for i, j in valid_edges:
            valid_neighbors[i].append(j)
            valid_neighbors[j].append(i)
        
        return valid_neighbors, valid_edges
    
    def _repulsion_relaxation(
        self, 
        keypoints: np.ndarray, 
        target_points: np.ndarray, 
        fixed_mask: np.ndarray,
        skeleton_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Spring-based relaxation with skeleton-aware neighbor validation.
        
        Uses spring forces between valid neighbors (validated via skeleton path)
        to achieve uniform spacing along the wire.
        
        Args:
            keypoints: K × 3 keypoints
            target_points: N × 3 points to project onto
            fixed_mask: K bool array (True = don't move)
            skeleton_mask: H × W skeleton mask for neighbor validation
        
        Returns:
            relaxed: K × 3 relaxed keypoints
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        epsilon = 1e-8
        
        if K <= 1 or len(target_points) == 0:
            return keypoints
        
        # Build NN index for projection
        cloud_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        cloud_nn.fit(target_points)
        
        # Find valid neighbors using skeleton path validation
        if skeleton_mask is not None:
            valid_neighbors, valid_edges = self._find_valid_neighbors(
                keypoints, skeleton_mask, 
                k_neighbors=self.repulsion_k_neighbors + 3,  # Consider more candidates
                block_radius=3
            )
        else:
            # Fallback to simple KNN
            nn = NearestNeighbors(n_neighbors=min(self.repulsion_k_neighbors + 1, K))
            nn.fit(keypoints)
            _, knn_indices = nn.kneighbors(keypoints)
            valid_neighbors = [list(knn_indices[i, 1:]) for i in range(K)]
            valid_edges = []
        
        # Compute target edge length from valid edges
        if valid_edges:
            edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in valid_edges]
            target_length = np.mean(edge_lengths)
        else:
            # Fallback: use pairwise distances
            all_dists = []
            for i in range(K):
                for j in valid_neighbors[i]:
                    all_dists.append(np.linalg.norm(keypoints[i] - keypoints[j]))
            target_length = np.mean(all_dists) if all_dists else 50.0
        
        # Relaxation iterations
        rebuild_every = 10
        
        for iteration in range(self.repulsion_iterations):
            # Optionally rebuild neighbor graph
            if iteration > 0 and skeleton_mask is not None and iteration % rebuild_every == 0:
                valid_neighbors, valid_edges = self._find_valid_neighbors(
                    keypoints, skeleton_mask,
                    k_neighbors=self.repulsion_k_neighbors + 3,
                    block_radius=3
                )
            
            # Compute spring forces
            forces = np.zeros_like(keypoints)
            
            for i in range(K):
                if fixed_mask[i]:
                    continue
                
                for j in valid_neighbors[i]:
                    v = keypoints[i] - keypoints[j]  # Vector from j to i
                    d = np.linalg.norm(v)
                    
                    if d < epsilon:
                        v = np.random.randn(3)
                        d = np.linalg.norm(v) + epsilon
                    
                    unit_v = v / d
                    
                    # Spring force: push if too close, pull if too far
                    force_mag = (target_length - d) / target_length
                    forces[i] += force_mag * unit_v
            
            # Normalize and apply
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            max_norm = np.max(force_norms)
            if max_norm > epsilon:
                forces = forces / max_norm * self.repulsion_lr
            
            # Update positions
            for i in range(K):
                if not fixed_mask[i]:
                    keypoints[i] += forces[i]
            
            # Project to target cloud
            _, target_indices = cloud_nn.kneighbors(keypoints)
            for i in range(K):
                if not fixed_mask[i]:
                    keypoints[i] = target_points[target_indices[i, 0]]
        
        return keypoints
    
    def _build_keypoint_topology(
        self, 
        keypoints: np.ndarray,
        skeleton_mask: np.ndarray = None,
    ) -> tuple:
        """
        Build MST edges on keypoints, with skeleton path validation and edge repair.
        
        When an MST edge is rejected (no valid skeleton path), we search for an
        alternative edge that:
        1. Has a valid skeleton path
        2. Reconnects the disconnected components
        
        Args:
            keypoints: K × 3 keypoints
            skeleton_mask: H × W skeleton mask for edge validation (optional)
        
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
        # Build adjacency for current validated edges
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
            # Check current connectivity
            components = get_connected_components(validated_edges, K)
            
            # Find which components rej_i and rej_j belong to
            comp_i = None
            comp_j = None
            for idx, comp in enumerate(components):
                if rej_i in comp:
                    comp_i = idx
                if rej_j in comp:
                    comp_j = idx
            
            # If they're already connected, no need for replacement
            if comp_i == comp_j:
                continue
            
            # Find best replacement edge connecting the two components
            # Sort candidate edges by distance
            candidates = []
            for node_i in components[comp_i]:
                for node_j in components[comp_j]:
                    if node_i != node_j:
                        candidates.append((node_i, node_j, dists[node_i, node_j]))
            
            # Sort by distance
            candidates.sort(key=lambda x: x[2])
            
            # Find first valid candidate
            replacement_found = False
            for cand_i, cand_j, _ in candidates:
                edge = (min(cand_i, cand_j), max(cand_i, cand_j))
                if edge not in validated_edges and is_valid_edge(cand_i, cand_j):
                    validated_edges.append(edge)
                    replacement_found = True
                    break
            
            if not replacement_found:
                # No valid replacement found - keep original edge as fallback
                # to maintain connectivity (better than disconnected graph)
                validated_edges.append((rej_i, rej_j))
        
        # Compute lengths for final edges
        edges = validated_edges
        lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges]
        
        return edges, np.array(lengths)
    
    def _extract_segment_ordered_edges(self) -> None:
        """
        Extract edges ordered by segment for sequential geometry correction.
        
        Simple algorithm for tree structure:
        1. For each leaf: trace edges from branch → leaf (corrections propagate outward)
        2. For trunk: trace edges from branch_0 → branch_1 (avoiding leaves)
        
        No BFS needed - just follow the chain since it's a tree (no cycles).
        
        Sets:
            self.segment_edges: List of segment edge lists, ordered branch→leaf or branch→branch
            self.anchor_set: Set of anchor keypoint indices (branches only)
            self.free_leaf_indices: List of free leaf indices
        """
        if self.reference_edges is None:
            print("Warning: reference_edges not set, cannot extract segments")
            return
        
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        K = self.n_keypoints
        
        # Identify node types
        branch_indices = set(range(n_branch))  # [0, 1] for 2 branches
        leaf_indices = set(range(n_branch, n_branch + n_leaf))  # [2, 3, 4, 5] for 4 leaves
        
        # EE-mapped leaves are anchors, others are free
        if self.ee_to_leaf_mapping is not None:
            ee_leaf_indices = set(self.ee_to_leaf_mapping.values())
        else:
            ee_leaf_indices = set()
        
        free_leaf_indices = leaf_indices - ee_leaf_indices
        
        # Only branch nodes are anchors; EE leaves can move during edge correction
        self.anchor_set = branch_indices
        self.free_leaf_indices = list(free_leaf_indices)
        
        print(f"  Anchor nodes (branches only): {sorted(self.anchor_set)}")
        print(f"  EE leaf nodes (free): {sorted(ee_leaf_indices)}")
        print(f"  Free leaf nodes: {sorted(self.free_leaf_indices)}")
        
        # Build adjacency list from reference edges
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(K)}
        edge_set = set()
        for i, j in self.reference_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
            edge_set.add((min(i, j), max(i, j)))
        
        segment_edges = []
        used_edges = set()
        
        # 1. For each leaf: trace from leaf to branch
        for leaf in sorted(leaf_indices):
            path = [leaf]
            current = leaf
            visited = {leaf}
            
            # Follow chain until we hit a branch
            while current not in branch_indices:
                # Find next node (should be exactly one unvisited neighbor in a tree)
                next_node = None
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        # Don't go through other leaves
                        if neighbor in leaf_indices and neighbor != leaf:
                            continue
                        next_node = neighbor
                        break
                
                if next_node is None:
                    print(f"  Warning: Dead end at node {current} tracing from leaf {leaf}")
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if current not in branch_indices:
                continue  # Failed to reach branch
            
            # Reverse ALL leaf segments to branch → leaf order
            # This ensures corrections propagate from fixed branch toward leaf endpoints
            # (applies to both EE and free leaf segments)
            path = path[::-1]  # Reverse: now branch → ... → leaf
            
            # Convert path to edges
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # Mark edges as used
            for e in path_edges:
                used_edges.add((min(e[0], e[1]), max(e[0], e[1])))
            
            segment_edges.append(path_edges)
            
            seg_type = "free_leaf" if leaf in free_leaf_indices else "ee_leaf"
            print(f"  Segment {len(segment_edges)-1} ({seg_type}): {path}, {len(path_edges)} edges")
        
        # 2. Trunk: trace from branch_0 to branch_1 (using remaining unused edges)
        if n_branch >= 2:
            branch_list = sorted(branch_indices)
            start_branch = branch_list[0]
            end_branch = branch_list[1]
            
            path = [start_branch]
            current = start_branch
            visited = {start_branch}
            
            while current != end_branch:
                next_node = None
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        # Check if edge is unused (not part of leaf segments)
                        edge_key = (min(current, neighbor), max(current, neighbor))
                        if edge_key in used_edges:
                            continue
                        # Don't go through leaves
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
            
            if current == end_branch:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                for e in path_edges:
                    used_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                segment_edges.append(path_edges)
                print(f"  Segment {len(segment_edges)-1} (trunk): {path}, {len(path_edges)} edges")
        
        # Check coverage
        uncovered = edge_set - used_edges
        if uncovered:
            print(f"  Warning: {len(uncovered)} edges not covered: {uncovered}")
            segment_edges.append(list(uncovered))
        
        self.segment_edges = segment_edges
        print(f"  Total segments: {len(self.segment_edges)}, total edges: {len(used_edges)}")
        
        # Print segment summary with node indices
        print(f"\n  === Segment Node Summary ===")
        for seg_idx, seg in enumerate(self.segment_edges):
            if len(seg) > 0:
                # Extract ordered nodes from edges
                nodes = [seg[0][0]] + [e[1] for e in seg]
                start_node = nodes[0]
                end_node = nodes[-1]
                
                # Determine segment type
                if start_node in leaf_indices or end_node in leaf_indices:
                    leaf_node = end_node if end_node in leaf_indices else start_node
                    if leaf_node in free_leaf_indices:
                        seg_type = "free_leaf"
                    else:
                        seg_type = "ee_leaf"
                elif start_node in branch_indices and end_node in branch_indices:
                    seg_type = "trunk"
                else:
                    seg_type = "other"
                
                print(f"    Segment {seg_idx} ({seg_type:10s}): nodes {nodes}")
        print(f"  ============================\n")
    
    # ================================================================
    # CPD REGISTRATION
    # ================================================================
    
    def _cpd_register(self, Y: np.ndarray, X: np.ndarray) -> tuple:
        """
        Non-rigid Coherent Point Drift registration.
        
        Args:
            Y: M × D template (previous keypoints)
            X: N × D target (current skeleton points)
        
        Returns:
            T_Y: M × D transformed template
            P: M × N correspondence matrix
        """
        Y = np.asarray(Y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        
        M, D = Y.shape
        N = X.shape[0]
        
        if M == 0 or N == 0:
            return Y.copy(), np.zeros((M, N))
        
        # Initialize
        T_Y = Y.copy()
        W = np.zeros((M, D))
        
        # Gaussian kernel for motion coherence
        # Use pairwise distances in Y to set appropriate scale
        diff_Y = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist_Y = np.sqrt(np.sum(diff_Y ** 2, axis=2))
        # Set beta based on median neighbor distance (more adaptive)
        np.fill_diagonal(dist_Y, np.inf)
        median_dist = np.median(np.min(dist_Y, axis=1))
        beta = max(self.cpd_beta * median_dist, 0.01)  # Scale beta by data
        
        G = np.exp(-np.sum(diff_Y ** 2, axis=2) / (2 * beta ** 2))
        
        # Initialize sigma^2 - use a smaller initial value based on data scale
        # This is critical: large sigma2 makes all correspondences uniform
        diff_init = X[np.newaxis, :, :] - Y[:, np.newaxis, :]
        dist2_init = np.sum(diff_init ** 2, axis=2)
        # Use median of minimum distances (not mean of all) for better initialization
        min_dists = np.min(dist2_init, axis=1)  # For each Y, find closest X
        sigma2 = np.median(min_dists) / 2.0  # Start smaller to get sharper correspondences
        sigma2 = max(sigma2, 1e-6)
        
        for iteration in range(self.cpd_max_iter):
            # ============================================================
            # E-step: Compute posterior probabilities
            # ============================================================
            diff = X[np.newaxis, :, :] - T_Y[:, np.newaxis, :]
            dist2 = np.sum(diff ** 2, axis=2)  # M x N
            
            # Numerically stable softmax-style normalization
            # Subtract max to prevent overflow
            log_p = -dist2 / (2 * sigma2)
            log_p_max = np.max(log_p, axis=0, keepdims=True)
            P_num = np.exp(log_p - log_p_max)
            
            c = (self.cpd_w / (1 - self.cpd_w + 1e-10)) * (M / N) * np.exp(-log_p_max)
            P_den = np.sum(P_num, axis=0, keepdims=True) + c
            P = P_num / (P_den + 1e-10)
            
            # ============================================================
            # M-step: Solve for W
            # ============================================================
            P1 = np.sum(P, axis=1)  # M
            Np = np.sum(P1)
            
            if Np < 1e-6:
                break
            
            P1_safe = np.maximum(P1, 1e-10)
            D_inv = np.diag(1.0 / P1_safe)
            
            # Standard CPD linear system: (G + λσ²D⁻¹)W = D⁻¹PX - Y
            A = G + self.cpd_lambda * sigma2 * D_inv
            B = D_inv @ P @ X - Y
            
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            
            T_Y_new = Y + G @ W
            
            # Update sigma^2
            diff_new = X[np.newaxis, :, :] - T_Y_new[:, np.newaxis, :]
            dist2_new = np.sum(diff_new ** 2, axis=2)
            sigma2_new = np.sum(P * dist2_new) / (Np * D + 1e-10)
            sigma2_new = max(sigma2_new, 1e-6)
            
            # Convergence check
            change = np.linalg.norm(T_Y_new - T_Y)
            if change < self.cpd_tol:
                T_Y = T_Y_new
                break
            
            T_Y = T_Y_new
            sigma2 = sigma2_new
        
        return T_Y, P
    
    # ================================================================
    # HUNGARIAN MATCHING
    # ================================================================
    
    def _hungarian_replace_anchors(
        self,
        cpd_keypoints: np.ndarray,
        detected_branch_3d: np.ndarray,
        detected_leaf_3d: np.ndarray,
        fallback_points: np.ndarray,
        ee_positions: np.ndarray = None,
        ee_to_leaf_mapping: dict = None,
    ) -> np.ndarray:
        """
        Replace anchor positions using Hungarian matching.

        Args:
            cpd_keypoints: K × 3 CPD-deformed keypoints
            detected_branch_3d: B × 3 detected branch candidates
            detected_leaf_3d: L × 3 detected leaf candidates
            fallback_points: N × 3 fallback point cloud
            ee_positions: 2 × 3 FK EE positions (optional)
            ee_to_leaf_mapping: {ee_idx: kp_idx} (optional)

        Returns:
            corrected: K × 3 with anchors replaced
        """
        corrected = cpd_keypoints.copy()
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf

        # Replace branch nodes
        if n_branch > 0 and len(detected_branch_3d) > 0:
            cpd_branch = cpd_keypoints[:n_branch]
            cost = cdist(cpd_branch, detected_branch_3d)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                corrected[r] = detected_branch_3d[c]
        elif n_branch > 0 and len(fallback_points) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(fallback_points)
            _, indices = nn.kneighbors(cpd_keypoints[:n_branch])
            for r, idx in enumerate(indices.flatten()):
                corrected[r] = fallback_points[idx]

        # Replace leaf nodes
        # EE-mapped leaves: closest detected leaf to FK EE position
        # Free leaves: Hungarian on remaining detected leaves
        if n_leaf > 0 and len(detected_leaf_3d) > 0:
            remaining_det = list(range(len(detected_leaf_3d)))
            assigned_locals = set()

            if ee_positions is not None and ee_to_leaf_mapping is not None:
                for ee_idx in sorted(ee_to_leaf_mapping.keys()):
                    kp_idx = ee_to_leaf_mapping[ee_idx]
                    local_idx = kp_idx - n_branch
                    if ee_idx < len(ee_positions) and len(remaining_det) > 0:
                        dists = np.linalg.norm(
                            detected_leaf_3d[remaining_det] - ee_positions[ee_idx], axis=1)
                        best = int(np.argmin(dists))
                        corrected[kp_idx] = detected_leaf_3d[remaining_det[best]]
                        assigned_locals.add(local_idx)
                        remaining_det.pop(best)

            free_locals = [i for i in range(n_leaf) if i not in assigned_locals]
            if len(free_locals) > 0 and len(remaining_det) > 0:
                cpd_free = cpd_keypoints[n_branch + np.array(free_locals)]
                det_free = detected_leaf_3d[remaining_det]
                cost = cdist(cpd_free, det_free)
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    corrected[n_branch + free_locals[r]] = det_free[c]
        elif n_leaf > 0 and len(fallback_points) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(fallback_points)
            _, indices = nn.kneighbors(cpd_keypoints[n_branch:n_branch + n_leaf])
            for r, idx in enumerate(indices.flatten()):
                corrected[n_branch + r] = fallback_points[idx]

        return corrected
    
    # ================================================================
    # GEOMETRY CONSTRAINTS
    # ================================================================
    
    def _joint_constraint_optimization(
        self,
        keypoints: np.ndarray,
        wire_points: np.ndarray,
    ) -> np.ndarray:
        """
        Joint edge length + wire projection optimization with segment-ordered processing.
        
        Uses DEFT-inspired sequential edge correction:
        - Edges are processed segment-by-segment
        - Within each segment, edges are ordered (e.g., branch → leaf)
        - Corrections propagate sequentially, allowing chain effects
        - Mass-weighted distribution: anchors get 0%, free nodes get proportional share
        
        Args:
            keypoints: K × 3 keypoints
            wire_points: N × 3 wire point cloud
        
        Returns:
            optimized: K × 3 constrained keypoints
        """
        keypoints = keypoints.copy().astype(np.float64)
        K = keypoints.shape[0]
        
        # Use pre-computed anchor set, or fall back to old behavior
        if self.anchor_set is not None:
            anchor_set = self.anchor_set
        else:
            # Fallback: all branch + leaf nodes are anchors
            anchor_set = set(range(self.reference_n_branch + self.reference_n_leaf))
        
        # For NoSnap BDLO: branch/leaf nodes are projected but then frozen for edge correction
        # Only intermediate nodes can be adjusted by edge constraints
        n_anchors = self.reference_n_branch + self.reference_n_leaf
        is_nosnap_bdlo = (not self.enable_node_matching) and (self.reference_n_branch > 0)
        if is_nosnap_bdlo:
            # Edge correction anchor set: all branch + leaf nodes (only intermediates are free)
            edge_anchor_set = set(range(n_anchors))
        else:
            edge_anchor_set = anchor_set
        
        # Build wire NN for projection
        wire_nn = None
        if len(wire_points) > 0:
            wire_nn = NearestNeighbors(n_neighbors=1).fit(wire_points)
        
        # Build edge index lookup for reference lengths
        edge_to_length = {}
        for edge_idx, (i, j) in enumerate(self.reference_edges):
            edge_to_length[(i, j)] = self.reference_lengths[edge_idx]
            edge_to_length[(j, i)] = self.reference_lengths[edge_idx]
        
        for outer_iter in range(self.n_outer_iterations):
            # Step 1: Wire projection FIRST (get keypoints close to wire surface)
            # For NoSnap BDLO: project branch/leaf nodes to skeleton (anchor_set only has EE leaves)
            if wire_nn is not None:
                _, indices = wire_nn.kneighbors(keypoints)
                for k in range(K):
                    if k not in anchor_set:
                        keypoints[k] = wire_points[indices[k, 0]]
            
            # Step 2: Edge length constraints AFTER projection
            # For NoSnap BDLO: branch/leaf nodes are now frozen (use edge_anchor_set)
            for edge_iter in range(self.n_edge_iterations):
                
                # Use segment-ordered edges if available, else fall back to reference_edges
                if self.segment_edges is not None:
                    # Process each segment sequentially
                    for segment in self.segment_edges:
                        for (i, j) in segment:
                            self._apply_edge_correction(
                                keypoints, i, j, edge_to_length, edge_anchor_set
                            )
                else:
                    # Fallback: batch processing (old behavior)
                    corrections = np.zeros_like(keypoints)
                    correction_counts = np.zeros(K)
                    
                    for edge_idx, (i, j) in enumerate(self.reference_edges):
                        if i >= K or j >= K:
                            continue
                        
                        edge_vec = keypoints[j] - keypoints[i]
                        current_length = np.linalg.norm(edge_vec)
                        
                        if current_length < 1e-6:
                            continue
                        
                        target_length = self.reference_lengths[edge_idx]
                        length_ratio = current_length / target_length
                        
                        if 1.0 - self.edge_tolerance <= length_ratio <= 1.0 + self.edge_tolerance:
                            continue
                        
                        length_diff = target_length - current_length
                        correction_mag = length_diff * self.edge_weight * 0.5
                        correction_dir = edge_vec / current_length
                        
                        if i not in edge_anchor_set:
                            corrections[i] -= correction_dir * correction_mag
                            correction_counts[i] += 1
                        if j not in edge_anchor_set:
                            corrections[j] += correction_dir * correction_mag
                            correction_counts[j] += 1
                    
                    for k in range(K):
                        if correction_counts[k] > 0 and k not in edge_anchor_set:
                            keypoints[k] += corrections[k] / correction_counts[k]
        
        return keypoints
    
    def _apply_edge_correction(
        self,
        keypoints: np.ndarray,
        i: int,
        j: int,
        edge_to_length: Dict[Tuple[int, int], float],
        anchor_set: Set[int],
    ) -> None:
        """
        Apply sequential edge length correction with mass-weighted distribution.
        
        Modifies keypoints in-place for immediate propagation to next edge.
        
        Args:
            keypoints: K × 3 keypoints (modified in-place)
            i, j: Edge endpoints
            edge_to_length: Dict mapping (i,j) to target length
            anchor_set: Set of anchor keypoint indices
        """
        K = keypoints.shape[0]
        if i >= K or j >= K:
            return
        
        edge_vec = keypoints[j] - keypoints[i]
        current_length = np.linalg.norm(edge_vec)
        
        if current_length < 1e-6:
            return
        
        # Get target length
        target_length = edge_to_length.get((i, j))
        if target_length is None:
            return
        
        length_ratio = current_length / target_length
        
        # Skip if within tolerance
        if 1.0 - self.edge_tolerance <= length_ratio <= 1.0 + self.edge_tolerance:
            return
        
        # Compute correction
        length_diff = target_length - current_length
        correction_dir = edge_vec / current_length
        
        # Determine weights based on anchor status
        i_free = i not in anchor_set
        j_free = j not in anchor_set
        
        if i_free and j_free:
            # Both free: split correction equally
            weight_i = 0.5
            weight_j = 0.5
        elif i_free:
            # Only i is free: i takes full correction
            weight_i = 1.0
            weight_j = 0.0
        elif j_free:
            # Only j is free: j takes full correction
            weight_i = 0.0
            weight_j = 1.0
        else:
            # Both anchored: no correction
            return
        
        # Apply correction immediately (for sequential propagation)
        correction_mag = length_diff * self.edge_weight
        keypoints[i] -= correction_dir * correction_mag * weight_i
        keypoints[j] += correction_dir * correction_mag * weight_j

    def _dlo_two_stage_balanced_optimization(
        self,
        keypoints: np.ndarray,
        wire_points: np.ndarray,
    ) -> np.ndarray:
        """
        DLO-specific BDLO-style balancing with repeated:
            projection blend -> sequential edge correction.

        This mirrors BDLO's alternating optimization style while exposing a
        scheduled projection blend to tune edge-vs-position balance.
        """
        keypoints = keypoints.copy().astype(np.float64)

        if self.anchor_set is not None:
            anchor_set = self.anchor_set
        else:
            anchor_set = {0, len(keypoints) - 1} if len(keypoints) > 1 else {0}

        # Build edge lookup
        edge_to_length = {}
        for edge_idx, (i, j) in enumerate(self.reference_edges):
            edge_to_length[(i, j)] = self.reference_lengths[edge_idx]
            edge_to_length[(j, i)] = self.reference_lengths[edge_idx]

        # Use segment-ordered edges if available; for DLO chain this should be one segment
        if self.segment_edges is not None:
            segments = self.segment_edges
        else:
            segments = [self.reference_edges]

        # Tunable stage schedule
        n_stages = max(1, int(self.dlo_balance_stages))
        blend_schedule = np.linspace(self.dlo_proj_blend_start, self.dlo_proj_blend_end, n_stages)

        # Total edge iterations are distributed across stages
        total_edge_iters = max(self.n_edge_iterations, n_stages)
        edge_iters_per_stage = max(1, int(np.ceil(total_edge_iters / n_stages)))

        wire_nn = None
        if len(wire_points) > 0:
            wire_nn = NearestNeighbors(n_neighbors=1).fit(wire_points)

        for proj_alpha in blend_schedule:
            # Stage A: projection blend for free nodes
            if wire_nn is not None and proj_alpha > 0:
                _, indices = wire_nn.kneighbors(keypoints)
                for k in range(len(keypoints)):
                    if k not in anchor_set:
                        nearest = wire_points[indices[k, 0]]
                        keypoints[k] = (1.0 - proj_alpha) * keypoints[k] + proj_alpha * nearest

            # Stage B: sequential edge correction (BDLO-style ordering)
            for _ in range(edge_iters_per_stage):
                for segment in segments:
                    for (i, j) in segment:
                        self._apply_edge_correction(keypoints, i, j, edge_to_length, anchor_set)

        return keypoints
    
    # ================================================================
    # END-EFFECTOR POSE INJECTION
    # ================================================================
    
    def _establish_ee_to_leaf_mapping(self, keypoints: np.ndarray, frame_idx: int) -> None:
        """
        Establish mapping from EE indices to leaf keypoint indices.
        
        Called during initialization (frame 0). For each of the 2 EE positions,
        find the closest leaf node among the target_leaf_nodes leaf keypoints.
        
        Args:
            keypoints: K × 3 current keypoints
            frame_idx: Current frame index (should be 0 for initialization)
        """
        if self.ee_poses_3d is None:
            return
        
        if frame_idx >= len(self.ee_poses_3d):
            print(f"Warning: frame_idx {frame_idx} >= ee_poses_3d length {len(self.ee_poses_3d)}")
            return
        
        ee_positions = self.ee_poses_3d[frame_idx]  # (2, 3)
        n_branch = self.reference_n_branch
        n_leaf = self.reference_n_leaf
        
        # Leaf keypoint indices: [n_branch, n_branch+1, ..., n_branch+n_leaf-1]
        leaf_indices = list(range(n_branch, n_branch + n_leaf))
        leaf_keypoints = keypoints[leaf_indices]  # (n_leaf, 3)
        
        # For each EE, find closest leaf node
        # Use Hungarian assignment to ensure unique mapping
        from scipy.spatial.distance import cdist
        cost_matrix = cdist(ee_positions, leaf_keypoints)  # (2, n_leaf)
        
        from scipy.optimize import linear_sum_assignment
        ee_idx_arr, leaf_local_idx_arr = linear_sum_assignment(cost_matrix)
        
        # Build mapping: EE index -> keypoint index
        self.ee_to_leaf_mapping = {}
        for ee_idx, leaf_local_idx in zip(ee_idx_arr, leaf_local_idx_arr):
            kp_idx = leaf_indices[leaf_local_idx]
            self.ee_to_leaf_mapping[ee_idx] = kp_idx
            dist = cost_matrix[ee_idx, leaf_local_idx]
            print(f"  EE {ee_idx} -> Leaf keypoint {kp_idx} (distance: {dist:.2f} mm)")
    
    def _replace_with_ee_poses(self, keypoints: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Replace mapped leaf keypoints with actual EE positions.
        
        Args:
            keypoints: K × 3 current keypoints
            frame_idx: Current frame index
        
        Returns:
            keypoints: K × 3 with EE positions injected
        """
        if (not self.enable_ee_injection) or self.ee_poses_3d is None or self.ee_to_leaf_mapping is None:
            return keypoints
        
        if frame_idx >= len(self.ee_poses_3d):
            print(f"Warning: frame_idx {frame_idx} >= ee_poses_3d length {len(self.ee_poses_3d)}")
            return keypoints
        
        ee_positions = self.ee_poses_3d[frame_idx]  # (2, 3)
        
        for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
            keypoints[kp_idx] = ee_positions[ee_idx]
        
        return keypoints
    
    def _create_path_mask_from_keypoints(self, keypoints_2d: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Create a path mask by drawing lines between consecutive keypoints.
        
        This creates a clean visualization showing only the tracked path,
        not the full noisy skeleton.
        
        Args:
            keypoints_2d: K × 2 keypoint coords (row, col)
            H: Image height
            W: Image width
        
        Returns:
            path_mask: H × W binary mask with lines between keypoints
        """
        import cv2
        path_mask = np.zeros((H, W), dtype=np.uint8)
        
        if keypoints_2d is None or len(keypoints_2d) < 2:
            return path_mask
        
        # Draw lines between consecutive keypoints (chain topology)
        for i in range(len(keypoints_2d) - 1):
            row1, col1 = int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1])
            row2, col2 = int(keypoints_2d[i + 1, 0]), int(keypoints_2d[i + 1, 1])
            
            # Clip to image bounds
            row1, col1 = max(0, min(H-1, row1)), max(0, min(W-1, col1))
            row2, col2 = max(0, min(H-1, row2)), max(0, min(W-1, col2))
            
            cv2.line(path_mask, (col1, row1), (col2, row2), 1, thickness=1)
        
        return path_mask

    def _uniform_sample_polyline(self, polyline: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Uniformly sample points along a 3D polyline by arc length.

        Args:
            polyline: N × 3 ordered path points
            n_samples: number of samples to return

        Returns:
            n_samples × 3 sampled points (includes endpoints)
        """
        if len(polyline) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if len(polyline) == 1 or n_samples <= 1:
            return np.repeat(polyline[:1], n_samples, axis=0)

        seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = arc[-1]

        if total_len < 1e-8:
            return np.repeat(polyline[:1], n_samples, axis=0)

        targets = np.linspace(0.0, total_len, n_samples)
        sampled = np.zeros((n_samples, 3), dtype=np.float64)

        j = 0
        for i, t in enumerate(targets):
            while j + 1 < len(arc) and arc[j + 1] < t:
                j += 1

            if j + 1 >= len(arc):
                sampled[i] = polyline[-1]
                continue

            a0, a1 = arc[j], arc[j + 1]
            if a1 - a0 < 1e-8:
                sampled[i] = polyline[j]
            else:
                alpha = (t - a0) / (a1 - a0)
                sampled[i] = (1.0 - alpha) * polyline[j] + alpha * polyline[j + 1]

        return sampled
    
    # ================================================================
    # SINGLE DLO INITIALIZATION (chain topology)
    # ================================================================
    
    def _project_ee_to_point_cloud(self, ee_pos: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
        """
        Project EE position to nearest point on the point cloud.
        
        Args:
            ee_pos: (3,) EE position
            point_cloud: (N, 3) point cloud
        
        Returns:
            (3,) nearest point on point cloud
        """
        distances = np.linalg.norm(point_cloud - ee_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return point_cloud[nearest_idx].copy()
    
    def _build_chain_from_ee(self, keypoints: np.ndarray, start_idx: int) -> list:
        """
        Build chain by iteratively finding nearest unvisited neighbor starting from start_idx.
        
        Args:
            keypoints: (N, 3) keypoints
            start_idx: index of the starting keypoint (EE position)
        
        Returns:
            list of keypoint indices in chain order
        """
        n = len(keypoints)
        visited = np.zeros(n, dtype=bool)
        chain = [start_idx]
        visited[start_idx] = True
        
        current = start_idx
        while len(chain) < n:
            # Find nearest unvisited neighbor
            distances = np.full(n, np.inf)
            for i in range(n):
                if not visited[i]:
                    distances[i] = np.linalg.norm(keypoints[i] - keypoints[current])
            
            nearest = np.argmin(distances)
            if distances[nearest] == np.inf:
                break  # No more unvisited points
            
            chain.append(nearest)
            visited[nearest] = True
            current = nearest
        
        return chain
    
    def _build_mst_and_find_path(self, keypoints: np.ndarray, start_idx: int, end_idx: int) -> list:
        """
        Build MST on keypoints and find path from start to end.
        
        Args:
            keypoints: (N, 3) keypoints
            start_idx: index of start node (EE0)
            end_idx: index of end node (EE1)
        
        Returns:
            list of keypoint indices forming path from start to end
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from collections import deque
        
        n = len(keypoints)
        
        # Build complete distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(keypoints[i] - keypoints[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        # Compute MST
        mst = minimum_spanning_tree(csr_matrix(dist_matrix))
        mst_array = mst.toarray()
        
        # Build adjacency list from MST (undirected)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                    if j not in adjacency[i]:
                        adjacency[i].append(j)
                    if i not in adjacency[j]:
                        adjacency[j].append(i)
        
        # BFS to find path from start to end
        visited = np.zeros(n, dtype=bool)
        parent = [-1] * n
        queue = deque([start_idx])
        visited[start_idx] = True
        
        while queue:
            current = queue.popleft()
            if current == end_idx:
                break
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # Reconstruct path
        if not visited[end_idx]:
            return []  # No path found
        
        path = []
        current = end_idx
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        return path
    
    def _initialize_with_segment_allocation(self, skeleton_mask: np.ndarray, depth: np.ndarray) -> dict:
        """
        Initialize BDLO using WireInitializer with segment-aware keypoint allocation.
        
        Uses WireTracker's own node identification and pruning, then delegates to
        WireInitializer's _build_topology for segment-aware FPS allocation.
        
        Args:
            skeleton_mask: H × W skeleton mask
            depth: H × W depth image
            
        Returns:
            dict with 'keypoints', 'keypoints_2d', 'edges', 'n_branch', 'n_leaf', 'timing'
        """
        import time
        timing = {}
        total_start = time.time()
        
        # Step 1: Node identification (same as default path)
        t0 = time.time()
        branch_2d, leaf_2d, adjacency, coords = self._node_identification(skeleton_mask)
        timing['node_detection'] = time.time() - t0
        
        # Step 2: Topology pruning (same as default path)
        t0 = time.time()
        if adjacency is not None:
            pruned = self._prune_to_target_topology(adjacency, coords)
            branch_2d = pruned["branch_coords"]
            leaf_2d = pruned["leaf_coords"]
        timing['pruning'] = time.time() - t0

        # Stash for debug visualization
        self._last_branch_2d = branch_2d.copy()
        self._last_leaf_2d = leaf_2d.copy()

        # Step 3: 2D → 3D conversion
        t0 = time.time()
        branch_3d = self._pixel_to_3d(branch_2d, depth)
        leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        timing['2d_to_3d'] = time.time() - t0
        
        # Store reference counts
        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf
        
        # Step 4: Create WireInitializer for segment-aware methods
        t0 = time.time()
        initializer = WireInitializer(
            intrinsics=self.intrinsics,
            n_keypoints=self.n_keypoints,
            target_branch_nodes=self.target_branch_nodes,
            target_leaf_nodes=self.target_leaf_nodes,
            max_depth=self.max_depth,
            repulsion_iterations=self.repulsion_iterations,
            repulsion_lr=self.repulsion_lr,
            ee_poses_3d=self.ee_poses_3d,
        )
        
        # Step 5: EE mapping (establish which leaf nodes are EE-attached)
        ee_to_leaf_kp = initializer._establish_ee_mapping_from_leaves(leaf_3d, n_branch)
        
        # Inject EE poses if available
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            initializer.ee_to_leaf_mapping = ee_to_leaf_kp
            self.ee_to_leaf_mapping = ee_to_leaf_kp
            ee_positions = self.ee_poses_3d[0]  # Frame 0
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                old_pos = leaf_3d[leaf_local_idx].copy()
                leaf_3d[leaf_local_idx] = ee_positions[ee_idx]
                dist = np.linalg.norm(old_pos - ee_positions[ee_idx])
                print(f"  EE {ee_idx} -> Leaf {kp_idx} (shift={dist:.1f} mm)")
        else:
            initializer.ee_to_leaf_mapping = ee_to_leaf_kp if ee_to_leaf_kp else None
            self.ee_to_leaf_mapping = ee_to_leaf_kp
        timing['ee_mapping'] = time.time() - t0
        
        # Update leaf_2d for EE leaves (project EE 3D → 2D)
        leaf_2d_updated = leaf_2d.copy().astype(np.float64)
        if self.ee_poses_3d is not None and ee_to_leaf_kp:
            for ee_idx, kp_idx in ee_to_leaf_kp.items():
                leaf_local_idx = kp_idx - n_branch
                ee_2d = self._project_3d_to_2d(self.ee_poses_3d[0][ee_idx:ee_idx+1])
                leaf_2d_updated[leaf_local_idx] = ee_2d[0]
        
        # Step 6: Build topology with segment-aware allocation (via WireInitializer)
        t0 = time.time()
        try:
            keypoints, edges, segment_edges, ordered_segments = initializer._build_topology(
                branch_3d, leaf_3d, branch_2d, leaf_2d_updated,
                skeleton_mask, depth, ee_to_leaf_kp,
                n_keypoints_per_segment=self.keypoints_per_segment,
            )
        except ValueError as e:
            return {'success': False, 'reason': str(e)}
        timing['build_topology'] = time.time() - t0
        
        # Store edges and segment info
        self.reference_edges = edges
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        self.segment_edges = segment_edges
        # Only branches are anchors; all leaves (EE and free) can be projected
        self.anchor_set = set(range(n_branch))
        
        # Identify free leaf indices
        self.free_leaf_indices = []
        for seg in ordered_segments:
            if seg['type'] == 'free_leaf':
                self.free_leaf_indices.append(seg['start_kp'])
        
        # Step 7: Repulsion refinement
        t0 = time.time()
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        skeleton_segment_lengths = [seg.get('estimated_length', 0) for seg in ordered_segments]
        
        fixed_mask = np.zeros(len(keypoints), dtype=bool)
        fixed_mask[:n_branch + n_leaf] = True
        
        keypoints = initializer._repulsion_relaxation_with_topology(
            keypoints, skeleton_pc, fixed_mask, edges, segment_edges,
            segment_lengths=skeleton_segment_lengths,
        )
        timing['repulsion'] = time.time() - t0
        
        # Store reference keypoints after repulsion
        self.reference_keypoints = keypoints.copy()
        
        # Recompute edge lengths after repulsion
        self.reference_lengths = np.array([
            np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges
        ])
        
        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True
        self.consecutive_skips = 0
        
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        timing['total'] = time.time() - total_start
        
        # Print segment summary
        seg_lengths = []
        for seg in segment_edges:
            seg_len = sum(np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in seg)
            seg_lengths.append(seg_len)
        print(f"  Segment lengths: {[f'{l:.1f}' for l in seg_lengths]} mm")
        print(f"  Total wire length: {sum(seg_lengths):.1f} mm")
        print(f"  keypoints_per_segment used: {self.keypoints_per_segment}")
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'mode': 'init_segment_aware',
            'timing': timing,
        }
    
    def _initialize_single_dlo(self, skeleton_mask: np.ndarray, depth: np.ndarray) -> dict:
        """
        Initialize keypoints for a single DLO (0 branch, 2 leaf) with chain topology.
        
        Algorithm:
            1. Build skeleton graph (2D pixel adjacency)
            2. Project EE poses to skeleton (find nearest skeleton pixel)
            3. Find shortest path on skeleton from EE0 to EE1 (clean 1D path)
            4. Convert path to 3D point cloud
            5. FPS with EE as anchors on the clean path
            6. Apply repulsion to spread keypoints evenly
            7. Build chain edges
        
        Args:
            skeleton_mask: H × W skeleton mask
            depth: H × W depth image
        
        Returns:
            dict with 'keypoints', 'keypoints_2d', 'edges', 'n_branch', 'n_leaf', 'timing'
        """
        import time
        from collections import deque
        timing = {}
        total_start = time.time()
        
        # Step 1: Build skeleton graph (2D)
        t0 = time.time()
        skeleton_coords, coord_to_idx, adjacency = self._build_skeleton_graph(skeleton_mask)
        
        if len(skeleton_coords) < self.n_keypoints:
            return {'success': False, 'reason': 'insufficient_skeleton_points'}
        
        print(f"  Skeleton graph: {len(skeleton_coords)} nodes")
        timing['build_graph'] = time.time() - t0
        
        # Step 2: Project EE poses to nearest skeleton pixel
        t0 = time.time()
        ee_positions = None
        ee_skel_indices = None  # indices in skeleton_coords
        
        if self.ee_poses_3d is not None:
            ee_positions = self.ee_poses_3d[0]  # (2, 3) for frame 0
            ee_skel_indices = []
            
            for i in range(2):
                # Project 3D EE to 2D pixel
                ee_3d = ee_positions[i]
                z = ee_3d[2]
                if z > 0:
                    col = int(ee_3d[0] * self.fx / z + self.cx)
                    row = int(ee_3d[1] * self.fy / z + self.cy)
                else:
                    row, col = 0, 0
                
                # Find nearest skeleton pixel
                dists = np.linalg.norm(skeleton_coords - np.array([row, col]), axis=1)
                nearest_idx = np.argmin(dists)
                ee_skel_indices.append(nearest_idx)
                
                nearest_pixel = skeleton_coords[nearest_idx]
                print(f"  EE{i}: 3D={ee_3d} -> pixel=({row},{col}) -> skeleton[{nearest_idx}]=({nearest_pixel[0]},{nearest_pixel[1]})")
        else:
            return {'success': False, 'reason': 'no_ee_poses'}
        
        timing['project_ee'] = time.time() - t0
        
        # Step 3: Find shortest path on skeleton graph from EE0 to EE1 (BFS)
        t0 = time.time()
        start_idx, end_idx = ee_skel_indices[0], ee_skel_indices[1]
        
        # BFS for shortest path
        visited = np.zeros(len(skeleton_coords), dtype=bool)
        parent = [-1] * len(skeleton_coords)
        queue = deque([start_idx])
        visited[start_idx] = True
        
        while queue:
            current = queue.popleft()
            if current == end_idx:
                break
            # adjacency[current] contains tuples of (neighbor_idx, distance)
            for neighbor_idx, _ in adjacency[current]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    parent[neighbor_idx] = current
                    queue.append(neighbor_idx)
        
        if not visited[end_idx]:
            return {'success': False, 'reason': 'no_path_between_ee'}
        
        # Reconstruct path
        path_indices = []
        current = end_idx
        while current != -1:
            path_indices.append(current)
            current = parent[current]
        path_indices.reverse()
        
        path_2d = skeleton_coords[path_indices]  # (path_len, 2) [row, col]
        print(f"  Shortest path: {len(path_indices)} pixels from EE0 to EE1")
        timing['find_path'] = time.time() - t0
        
        # Step 4: Convert path to 3D point cloud
        t0 = time.time()
        path_3d = self._pixel_to_3d(path_2d, depth)
        
        # Remove invalid points (z=0)
        valid_mask = path_3d[:, 2] > 0
        path_3d = path_3d[valid_mask]
        path_2d = path_2d[valid_mask]
        
        # Create clean path mask (only the BFS path pixels, not full skeleton)
        H, W = skeleton_mask.shape
        clean_path_mask = np.zeros((H, W), dtype=np.uint8)
        for row, col in path_2d:
            if 0 <= row < H and 0 <= col < W:
                clean_path_mask[int(row), int(col)] = 1
        
        print(f"  Valid 3D path: {len(path_3d)} points")
        timing['path_to_3d'] = time.time() - t0
        
        # Step 5: FPS with EE as anchors on the clean path
        t0 = time.time()
        
        # EE projected positions on path (first and last)
        ee0_3d = path_3d[0]
        ee1_3d = path_3d[-1]
        anchors = np.array([ee0_3d, ee1_3d])
        
        keypoints = self._fps_with_anchors(path_3d, anchors)
        print(f"  FPS keypoints: {len(keypoints)}")
        timing['fps'] = time.time() - t0
        
        # Step 6: Repulsion relaxation along the path (use existing working method)
        t0 = time.time()
        # Fixed mask: first 2 keypoints are anchors (EE0, EE1)
        fixed_mask = np.zeros(self.n_keypoints, dtype=bool)
        fixed_mask[0] = True
        fixed_mask[1] = True
        
        # Use existing _repulsion_relaxation with skeleton_mask for validation
        keypoints = self._repulsion_relaxation(keypoints, path_3d, fixed_mask, skeleton_mask)
        print(f"  After repulsion: keypoints spread along path")
        timing['repulsion'] = time.time() - t0

        # Step 6.5: Uniform arc-length resampling to enforce even spacing on clean path
        t0 = time.time()
        keypoints = self._uniform_sample_polyline(path_3d, self.n_keypoints)
        print(f"  Uniform resample: enforced {self.n_keypoints} evenly-spaced keypoints")
        timing['uniform_resample'] = time.time() - t0
        
        # Step 7: Build chain by ordering keypoints along the path
        t0 = time.time()
        
        # Project keypoints to path and get their position along the path
        # Use cumulative arc length
        arc_lengths = np.zeros(len(path_3d))
        for i in range(1, len(path_3d)):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(path_3d[i] - path_3d[i-1])
        total_length = arc_lengths[-1]
        
        # Find arc length position for each keypoint
        kp_arc_lengths = np.zeros(len(keypoints))
        for i, kp in enumerate(keypoints):
            # Find nearest point on path
            dists = np.linalg.norm(path_3d - kp, axis=1)
            nearest_idx = np.argmin(dists)
            kp_arc_lengths[i] = arc_lengths[nearest_idx]
        
        # Sort keypoints by arc length (EE0 at 0, EE1 at end)
        chain_order = np.argsort(kp_arc_lengths)
        keypoints_ordered = keypoints[chain_order]
        
        print(f"  Chain order by arc length: {chain_order.tolist()}")
        print(f"  Arc lengths: {[f'{kp_arc_lengths[i]:.1f}' for i in chain_order]}")
        print(f"  Total path length: {total_length:.1f}mm")
        
        timing['build_chain'] = time.time() - t0
        
        # Step 8: Set up state
        t0 = time.time()
        n_branch = 0
        n_leaf = 2
        
        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf
        self.ee_to_leaf_mapping = {0: 0, 1: len(keypoints_ordered) - 1}
        
        # Build chain edges
        n_chain = len(keypoints_ordered)
        edges = [(i, i + 1) for i in range(n_chain - 1)]
        lengths = np.array([np.linalg.norm(keypoints_ordered[i] - keypoints_ordered[i + 1]) 
                           for i in range(n_chain - 1)])
        
        print(f"  Edges: {edges}")
        print(f"  Edge lengths: {[f'{l:.1f}' for l in lengths]}")
        
        timing['setup'] = time.time() - t0
        
        # Store reference
        self.reference_keypoints = keypoints_ordered.copy()
        self.reference_edges = edges
        self.reference_lengths = lengths
        self.ordered_edge_sequence = edges.copy()
        self.clean_path_mask = clean_path_mask  # Store clean BFS path for visualization

        # For DLO chain: use segment-ordered geometry correction in tracking
        self.segment_edges = [edges.copy()]
        
        # Set anchor_set for DLO: endpoints are anchors (EE indices)
        self.anchor_set = {0, len(keypoints_ordered) - 1}
        
        self.prev_keypoints = keypoints_ordered.copy()
        self.is_initialized = True
        self.consecutive_skips = 0
        
        keypoints_2d_proj = self._project_3d_to_2d(keypoints_ordered)
        
        timing['total'] = time.time() - total_start
        
        return {
            'success': True,
            'keypoints': keypoints_ordered,
            'keypoints_2d': keypoints_2d_proj,
            'edges': edges,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'detected_branch': np.empty((0, 3)),
            'detected_leaf': keypoints_ordered[[0, -1]],
            'clean_path_mask': clean_path_mask,
            'timing': timing,
            'mode': 'init',
        }
    
    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    
    def initialize(self, skeleton_mask: np.ndarray, depth: np.ndarray) -> dict:
        """
        Phase 2: Frame 0 initialization.
        
        Args:
            skeleton_mask: H × W skeleton mask
            depth: H × W depth image
        
        Returns:
            dict with 'keypoints', 'keypoints_2d', 'edges', 'n_branch', 'n_leaf', 'timing'
        """
        # For single DLO (0 branch, 2 leaf), use chain topology initialization
        if self.target_branch_nodes == 0 and self.target_leaf_nodes == 2:
            print("  Using single DLO initialization (chain topology)")
            return self._initialize_single_dlo(skeleton_mask, depth)
        
        # BDLO initialization
        # If keypoints_per_segment is specified, use WireInitializer for segment-aware allocation
        if self.keypoints_per_segment is not None:
            print(f"  keypoints_per_segment specified: {self.keypoints_per_segment}")
            print(f"  Using WireInitializer (segment-aware FPS allocation)")
            return self._initialize_with_segment_allocation(skeleton_mask, depth)
        
        # Default: global FPS allocation
        print("  Using global FPS allocation (default)")
        import time
        timing = {}
        total_start = time.time()
        
        # Step 2.1: Node identification
        t0 = time.time()
        branch_2d, leaf_2d, adjacency, coords = self._node_identification(skeleton_mask)
        timing['node_detection'] = time.time() - t0
        
        # Step 2.2: Topology pruning
        t0 = time.time()
        if adjacency is not None:
            pruned = self._prune_to_target_topology(adjacency, coords)
            branch_2d = pruned["branch_coords"]
            leaf_2d = pruned["leaf_coords"]
        timing['pruning'] = time.time() - t0
        
        # Step 2.3: 2D → 3D
        t0 = time.time()
        branch_3d = self._pixel_to_3d(branch_2d, depth)
        leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        
        n_branch = len(branch_3d)
        n_leaf = len(leaf_3d)
        timing['2d_to_3d'] = time.time() - t0
        
        # Step 2.4: FPS with anchors
        t0 = time.time()
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)
        
        if len(skeleton_pc) < self.n_keypoints:
            return {'success': False, 'reason': 'insufficient_skeleton_points'}
        
        anchors = np.vstack([branch_3d, leaf_3d]) if n_branch + n_leaf > 0 else np.empty((0, 3))
        keypoints = self._fps_with_anchors(skeleton_pc, anchors)
        timing['fps'] = time.time() - t0
        
        # Store reference counts before EE injection
        self.reference_n_branch = n_branch
        self.reference_n_leaf = n_leaf
        
        # Step 2.5: EE pose injection BEFORE repulsion (if available)
        # This ensures repulsion distributes intermediate nodes based on true EE positions
        t0 = time.time()
        if self.ee_poses_3d is not None:
            # Establish mapping: which 2 leaf nodes correspond to which EE
            self._establish_ee_to_leaf_mapping(keypoints, frame_idx=0)
            # Keep tracked positions — do not hard-replace with EE poses
        timing['ee_injection'] = time.time() - t0
        
        # Step 2.6: Repulsion relaxation (with skeleton-aware neighbor validation)
        # EE-injected leaf nodes are still marked as fixed
        t0 = time.time()
        fixed_mask = np.zeros(self.n_keypoints, dtype=bool)
        fixed_mask[:n_branch + n_leaf] = True
        
        keypoints = self._repulsion_relaxation(keypoints, skeleton_pc, fixed_mask, skeleton_mask)
        timing['repulsion'] = time.time() - t0
        
        # Step 2.7: Topology connection (with skeleton path validation)
        t0 = time.time()
        edges, lengths = self._build_keypoint_topology(keypoints, skeleton_mask)
        timing['topology'] = time.time() - t0
        
        # Store reference
        self.reference_keypoints = keypoints.copy()
        self.reference_edges = edges
        self.reference_lengths = lengths
        
        # Step 2.8: Extract segment-ordered edges for geometry correction
        t0 = time.time()
        self._extract_segment_ordered_edges()
        timing['segment_extraction'] = time.time() - t0
        
        self.prev_keypoints = keypoints.copy()
        self.is_initialized = True
        self.consecutive_skips = 0
        
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        timing['total'] = time.time() - total_start
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': edges,
            'n_branch': n_branch,
            'n_leaf': n_leaf,
            'mode': 'init',
            'timing': timing,
        }
    
    def track(
        self,
        skeleton_mask: np.ndarray,
        skeleton_pc: np.ndarray,
        depth: np.ndarray,
    ) -> dict:
        """
        Phase 3: CPD tracking for Frame N > 0.
        
        Pipeline:
            1. Node detection (no pruning)
            2. CPD registration
            3. Hungarian node matching (if enabled)
            4. EE pose injection (if ee_poses_3d provided)
            5. Geometry constraint (if enabled)
            6. Re-apply EE poses (ensure exact positions)
        
        Args:
            skeleton_mask: H × W skeleton mask
            skeleton_pc: N × 3 skeleton point cloud
            depth: H × W depth image
        
        Returns:
            dict with 'keypoints', 'keypoints_2d', 'edges', 'edge_errors', 'timing'
        """
        import time
        timing = {}
        total_start = time.time()
        
        if not self.is_initialized:
            return {'success': False, 'reason': 'not_initialized'}
        
        # Step 3.1: Node detection (NO pruning)
        t0 = time.time()
        branch_2d, leaf_2d, _, _ = self._node_identification(skeleton_mask)
        detected_branch_3d = self._pixel_to_3d(branch_2d, depth)
        detected_leaf_3d = self._pixel_to_3d(leaf_2d, depth)
        timing['node_detection'] = time.time() - t0
        
        # Step 3.2: Full CPD on ALL keypoints (conditional on ablation flag)
        t0 = time.time()
        if self.enable_cpd:
            cpd_target = skeleton_pc
            if len(cpd_target) > self.cpd_downsample:
                indices = np.random.choice(len(cpd_target), self.cpd_downsample, replace=False)
                cpd_target = cpd_target[indices]
            
            cpd_keypoints, _ = self._cpd_register(self.prev_keypoints, cpd_target)
        else:
            # Skip CPD - use previous keypoints as starting point
            cpd_keypoints = self.prev_keypoints.copy()
        timing['cpd'] = time.time() - t0
        
        # Step 3.3: Hungarian replace anchors (conditional on ablation flag)
        # Use prev_keypoints (not cpd_keypoints) as reference for assignment
        # to maintain stable identity across frames
        # NOTE: For single DLO (0 branch, 2 leaf), skip Hungarian and use EE directly
        t0 = time.time()
        is_single_dlo = (self.reference_n_branch == 0 and self.reference_n_leaf == 2)
        
        if is_single_dlo:
            adjusted = cpd_keypoints.copy()
            # Match detected leaf nodes to prev leaf positions and update
            if (self.enable_ee_injection and self.ee_to_leaf_mapping is not None
                    and detected_leaf_3d is not None and len(detected_leaf_3d) >= 2):
                leaf_kp_indices = list(self.ee_to_leaf_mapping.values())
                prev_leaf_pos = self.prev_keypoints[leaf_kp_indices]  # (2, 3)
                from scipy.spatial.distance import cdist
                cost = cdist(prev_leaf_pos, detected_leaf_3d)  # (2, N_detected)
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    adjusted[leaf_kp_indices[r]] = detected_leaf_3d[c]
        elif self.enable_node_matching:
            # Get current EE positions for leaf matching reference
            frame_idx = self.frame_count - 1
            ee_pos = None
            if (self.enable_ee_injection and self.ee_poses_3d is not None
                    and frame_idx < len(self.ee_poses_3d)):
                ee_pos = self.ee_poses_3d[frame_idx]
            adjusted = self._hungarian_replace_anchors(
                self.prev_keypoints,  # Use prev as reference for matching
                detected_branch_3d,
                detected_leaf_3d,
                skeleton_pc,
                ee_positions=ee_pos,
                ee_to_leaf_mapping=self.ee_to_leaf_mapping,
            )
            # For non-anchor keypoints, use CPD output
            n_anchors = self.reference_n_branch + self.reference_n_leaf
            adjusted[n_anchors:] = cpd_keypoints[n_anchors:]
        else:
            # Skip node matching - use CPD output directly
            adjusted = cpd_keypoints
        timing['hungarian'] = time.time() - t0
        
        # Step 3.4: EE mapping preserved — leaf nodes keep tracked positions
        # (no hard-replace, but EE-mapped leaves are still fixed during optimization)
        timing['ee_injection'] = 0.0
        
        # Step 3.5: Joint geometry + projection (conditional on ablation flag)
        # Adjust anchor_set based on tracking mode
        t0 = time.time()
        if is_single_dlo:
            if self.enable_ee_injection and self.ee_to_leaf_mapping is not None:
                # Full mode: EE-mapped keypoints are anchors during tracking
                self.anchor_set = set(self.ee_to_leaf_mapping.values())
            else:
                # NoSnap mode: all points free during tracking (EE only used for initialization)
                self.anchor_set = set()
        else:
            # BDLO case
            if not self.enable_node_matching:
                # NoSnap for BDLO: only EE-attached leaves are anchors
                # Branch nodes and free leaves should be free to move with geometry constraint
                if self.enable_ee_injection and self.ee_to_leaf_mapping is not None:
                    self.anchor_set = set(self.ee_to_leaf_mapping.values())
                else:
                    self.anchor_set = set()
        
        if self.enable_geometry_constraint:
            if is_single_dlo:
                keypoints = self._dlo_two_stage_balanced_optimization(adjusted, skeleton_pc)
            else:
                keypoints = self._joint_constraint_optimization(adjusted, skeleton_pc)
        else:
            # NoGeometry ablation: if CPD is off, still project free nodes to wire cloud
            # but do not run edge optimization.
            keypoints = adjusted.copy()
            if (not self.enable_cpd) and len(skeleton_pc) > 0:
                projection_nn = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
                _, projection_idx = projection_nn.kneighbors(keypoints)
                for k in range(len(keypoints)):
                    if self.anchor_set is None or k not in self.anchor_set:
                        keypoints[k] = skeleton_pc[projection_idx[k, 0]]
        timing['geometry'] = time.time() - t0

        # Re-apply EE poses after geometry constraint to ensure exact anchor positions
        # (geometry constraint should already preserve anchor nodes, but this is explicit)
        # keypoints = self._replace_with_ee_poses(keypoints, current_frame_idx)
        
        # Compute edge errors
        edge_errors = self._compute_edge_errors(keypoints)
        edge_rmse_mm = self._compute_edge_rmse_mm(keypoints)
        
        # Update state
        self.prev_keypoints = keypoints.copy()
        self.consecutive_skips = 0
        
        keypoints_2d = self._project_3d_to_2d(keypoints)
        
        # Update clean_path_mask from keypoints (draw lines between consecutive keypoints)
        H, W = skeleton_mask.shape
        self.clean_path_mask = self._create_path_mask_from_keypoints(keypoints_2d, H, W)
        
        timing['total'] = time.time() - total_start
        
        return {
            'success': True,
            'keypoints': keypoints,
            'keypoints_2d': keypoints_2d,
            'edges': self.reference_edges,
            'edge_errors': edge_errors,
            'edge_rmse_mm': edge_rmse_mm,
            'detected_branch': detected_branch_3d,
            'detected_leaf': detected_leaf_3d,
            'clean_path_mask': self.clean_path_mask,
            'mode': 'track',
            'timing': timing,
        }
    
    def _compute_edge_errors(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute relative edge length errors."""
        errors = []
        for edge_idx, (i, j) in enumerate(self.reference_edges):
            current = np.linalg.norm(keypoints[i] - keypoints[j])
            ref = self.reference_lengths[edge_idx]
            errors.append(abs(current - ref) / (ref + 1e-6))
        return np.array(errors)

    def _compute_edge_rmse_mm(self, keypoints: np.ndarray) -> float:
        """Compute edge length RMSE in mm."""
        abs_errors = []
        for edge_idx, (i, j) in enumerate(self.reference_edges):
            current = np.linalg.norm(keypoints[i] - keypoints[j])
            ref = self.reference_lengths[edge_idx]
            abs_errors.append(abs(current - ref))

        if len(abs_errors) == 0:
            return 0.0

        return float(np.sqrt(np.mean(np.square(abs_errors))))
    
    def process_frame(
        self,
        depth: np.ndarray,
        arm_depth: np.ndarray = None,
        rgb: np.ndarray = None,
        precomputed_arm_mask: np.ndarray = None,
    ) -> dict:
        """
        Main entry point: auto-selects initialize vs track.
        
        Args:
            depth: H × W current frame depth
            arm_depth: H × W arm-only depth (not needed if precomputed_arm_mask provided)
            rgb: H × W × 3 RGB image (optional)
            precomputed_arm_mask: H × W binary mask where 1=arm (to be removed), 0=keep
                                  If provided, skips background subtraction and uses this directly.
        
        Returns:
            dict with tracking results
        """
        import time
        
        self.frame_count += 1
        
        # Phase 1: Segmentation
        # Use n_components=1 for frame 0 (initialization), top_k_components for tracking
        is_init_frame = not self.is_initialized
        n_components = 1 if is_init_frame else self.top_k_components
        
        seg_result = self.segment(depth, arm_depth, n_components=n_components,
                                   precomputed_arm_mask=precomputed_arm_mask,
                                   is_init=is_init_frame)
        
        foreground_mask = seg_result['foreground_mask']
        skeleton_mask = seg_result['skeleton_mask']
        skeleton_pc = seg_result['skeleton_pc']
        seg_time = seg_result.get('seg_time', 0.0)  # Core segmentation time (excludes arm expansion + pc conversion)
        
        # Check minimum skeleton
        if np.sum(skeleton_mask) < self.min_skeleton_pixels:
            self.consecutive_skips += 1
            return {
                'success': False,
                'reason': 'insufficient_skeleton',
                'foreground_mask': foreground_mask,
                'skeleton_mask': skeleton_mask,
                'skeleton_pc': skeleton_pc,
                'mode': 'skip',
            }
        
        # Route to appropriate phase
        if not self.is_initialized:
            result = self.initialize(skeleton_mask, depth)
        else:
            # Check warm restart
            if self.consecutive_skips > self.max_skips_before_restart:
                # Re-initialize with n_components=1
                self.is_initialized = False
                seg_result_restart = self.segment(depth, arm_depth, n_components=1,
                                                   precomputed_arm_mask=precomputed_arm_mask,
                                                   is_init=True)
                skeleton_mask = seg_result_restart['skeleton_mask']
                foreground_mask = seg_result_restart['foreground_mask']
                result = self.initialize(skeleton_mask, depth)
                result['mode'] = 'restart'
            else:
                result = self.track(skeleton_mask, skeleton_pc, depth)
        
        # Add segmentation time to timing (core seg time only)
        if 'timing' in result:
            result['timing']['segmentation'] = seg_time
        
        # Add segmentation results
        result['foreground_mask'] = foreground_mask
        result['skeleton_mask'] = skeleton_mask
        result['skeleton_pc'] = skeleton_pc
        result['frame_idx'] = self.frame_count - 1
        
        # Add clean path mask for DLO visualization (if available)
        if self.clean_path_mask is not None:
            result['clean_path_mask'] = self.clean_path_mask
        
        return result
    
    def skip_frame(self):
        """Manually mark frame as skipped."""
        self.consecutive_skips += 1
    
    def get_state(self) -> dict:
        """Get current tracker state."""
        return {
            'is_initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'consecutive_skips': self.consecutive_skips,
            'n_branch': self.reference_n_branch,
            'n_leaf': self.reference_n_leaf,
            'n_keypoints': self.n_keypoints,
            'reference_edges': self.reference_edges,
        }
