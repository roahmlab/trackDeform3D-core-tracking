"""Frame-0 initialization for the wire (DLO/BDLO) tracker.

WireInitMixin is inherited by tracker.wire_tracker.WireTracker; the methods were
moved verbatim from the tracker class, so `self` is the tracker instance and
tracking-side helpers (_node_identification, _prune_to_target_topology,
_extract_point_cloud, _pixel_to_3d, ...) resolve through the class MRO.
"""
import math
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

from initialization.wire_initializer import WireInitializer


class WireInitMixin:
    """Initialization methods for WireTracker (moved verbatim from the tracker;
    the tracker class inherits this mixin, so `self` is the tracker instance)."""

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
        
        # Default: global FPS allocation is not supported (removed as dead code);
        # BDLO runs must specify keypoints_per_segment.
        raise ValueError(
            "initialize(): global FPS allocation was removed; pass keypoints_per_segment "
            "for BDLO or use target_branch_nodes=0/target_leaf_nodes=2 for single DLO")

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
