"""Frame-0 initialization for the cloth tracker (contour corners -> rect grid).

ClothInitMixin is inherited by tracker.cloth_tracker.ClothTracker; the methods
were moved verbatim from the tracker class.  Helpers used by BOTH init and
tracking (e.g. _detect_contour_corners, _extract_contour_3d, _snap_to_contour_3d)
stay in the tracker and resolve through the class MRO.
"""
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

from initialization.cloth_contour import (
    find_max_inscribed_rectangle_rotated,
    get_bounding_rect_same_orientation,
)


class ClothInitMixin:
    """Initialization methods for ClothTracker (moved verbatim from the tracker;
    the tracker class inherits this mixin, so `self` is the tracker instance)."""

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
