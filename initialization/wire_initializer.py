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
    
    
    # ================================================================
    # NODE IDENTIFICATION
    # ================================================================
    
    
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

    
    # ================================================================
    # END-EFFECTOR POSE INJECTION
    # ================================================================
    
    
    # ================================================================
    # MAIN INITIALIZATION
    # ================================================================
    
    