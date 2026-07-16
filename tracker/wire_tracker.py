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

from initialization.wire_init import WireInitMixin


class WireTracker(WireInitMixin):
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
        max_depth: float = 1000.0,
        top_k_components: int = 5,
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
            max_depth: Maximum valid depth (mm)
            top_k_components: Number of largest connected components to keep
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
        self.max_depth = max_depth
        self.top_k_components = top_k_components
        
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
        
        if precomputed_arm_mask is None:
            raise ValueError("segment() requires precomputed_arm_mask (1=arm/remove, 0=keep)")
        # Use precomputed mask directly: arm pixels are 1, wire/foreground is 0
        # Foreground = NOT arm AND valid depth
        foreground_mask = ((precomputed_arm_mask == 0) & (depth > 0)).astype(np.uint8)
        
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
    
    
    # ================================================================
    # FPS + REPULSION
    # ================================================================
    
    
    # ================================================================
    # CPD REGISTRATION
    # ================================================================
    
    
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

    
    # ================================================================
    # SINGLE DLO INITIALIZATION (chain topology)
    # ================================================================
    
    
    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    

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
        
        # Step 3.2: use previous keypoints as the starting point
        t0 = time.time()
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
            # NoGeometry ablation: still project free nodes to wire cloud
            # but do not run edge optimization.
            keypoints = adjusted.copy()
            if len(skeleton_pc) > 0:
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
    
    