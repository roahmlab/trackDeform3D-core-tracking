"""Frame-0 initialization for the fabric tracker (mask corners -> grid + FPS).

FabricInitMixin is inherited by tracker.fabric_tracker.FabricTracker; the
methods were moved verbatim from the tracker class.  NOTE: _farthest_point_sampling
draws its first sample with an unseeded np.random.randint -- this is the one
potential nondeterminism source of the fabric pipeline (kept as-is by design).
"""
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


class FabricInitMixin:
    """Initialization methods for FabricTracker (moved verbatim from the tracker;
    the tracker class inherits this mixin, so `self` is the tracker instance)."""

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
