#!/usr/bin/env python3
"""
Cloth Initialization: Rectangle Inpaint + Real Contour Keypoints

Algorithm:
1. Find max inscribed rectangle inside contour
2. Find min bounding rectangle with SAME ORIENTATION as max inscribed
3. Use these two rectangles for grid interpolation
4. Detect corners on REAL contour (approxPolyDP)
5. Build edges: contour sequential, interior grid, interior↔contour

Author: Auto-generated
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from scipy import ndimage


def find_max_inscribed_rectangle(mask: np.ndarray) -> tuple:
    """
    Find the maximum area axis-aligned rectangle inscribed in the mask.
    Uses the largest rectangle in histogram approach.
    
    Returns:
        (center_x, center_y, width, height, angle=0)
    """
    # Find largest axis-aligned rectangle using dynamic programming
    H, W = mask.shape
    
    # Build height histogram for each row
    heights = np.zeros((H, W), dtype=int)
    heights[0] = mask[0].astype(int)
    for i in range(1, H):
        heights[i] = np.where(mask[i], heights[i-1] + 1, 0)
    
    max_area = 0
    best_rect = None  # (row, col, width, height)
    
    # For each row, find largest rectangle in histogram
    for row in range(H):
        hist = heights[row]
        
        # Use stack-based approach for largest rectangle in histogram
        stack = []  # (index, height)
        
        for i, h in enumerate(hist):
            start = i
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                width = i - idx
                area = width * height
                if area > max_area:
                    max_area = area
                    # Rectangle: bottom-left is (idx, row - height + 1), size is (width, height)
                    best_rect = (row - height + 1, idx, width, height)
                start = idx
            stack.append((start, h))
        
        # Process remaining in stack
        for idx, height in stack:
            width = W - idx
            area = width * height
            if area > max_area:
                max_area = area
                best_rect = (row - height + 1, idx, width, height)
    
    if best_rect is None:
        return None
    
    row_start, col_start, width, height = best_rect
    center_x = col_start + width / 2
    center_y = row_start + height / 2
    
    return (center_x, center_y, width, height, 0)  # angle = 0 for axis-aligned


def find_max_inscribed_rectangle_rotated(mask: np.ndarray, n_angles: int = 180) -> tuple:
    """
    Find the maximum area rectangle inscribed in the mask at any angle.
    Uses cv2.minAreaRect to get orientation, then finds inscribed rect at that angle.
    
    Returns:
        (center_x, center_y, width, height, angle_deg)
    """
    # Use minAreaRect to get the optimal orientation quickly
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx_rect, cy_rect), (w_rect, h_rect), angle = rect
    
    # minAreaRect returns angle in range [-90, 0) for OpenCV 4.5+
    # Normalize to [0, 180) for consistency
    if angle < 0:
        angle = angle + 90
    
    H, W = mask.shape
    center = (W // 2, H // 2)
    
    # Test the angle from minAreaRect and a few nearby angles for robustness
    angles_to_test = [angle, angle + 90]  # Also test perpendicular
    
    best_area = 0
    best_result = None
    
    for test_angle in angles_to_test:
        # Normalize angle to [0, 180)
        test_angle = test_angle % 180
        
        # Rotate mask
        M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
        rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (W, H), 
                                       flags=cv2.INTER_NEAREST)
        
        # Find max inscribed rectangle in rotated mask
        result = find_max_inscribed_rectangle(rotated_mask > 0)
        if result is None:
            continue
        
        cx_rot, cy_rot, w, h, _ = result
        area = w * h
        
        if area > best_area:
            best_area = area
            # Transform center back to original coordinates
            M_inv = cv2.getRotationMatrix2D(center, -test_angle, 1.0)
            pt = np.array([[cx_rot, cy_rot, 1]]).T
            pt_orig = M_inv @ pt
            cx_orig, cy_orig = pt_orig[0, 0], pt_orig[1, 0]
            
            best_result = (cx_orig, cy_orig, w, h, test_angle)
    
    return best_result


def get_bounding_rect_same_orientation(contour_points: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Get the minimum bounding rectangle of contour_points with a fixed orientation.
    
    Args:
        contour_points: (N, 2) array of (col, row) points
        angle_deg: Orientation angle in degrees
    
    Returns:
        4 corners of the bounding rectangle in original coordinates (4, 2)
    """
    # Rotate points to align with axis
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotation matrix (rotate by -angle to align)
    R = np.array([[cos_a, sin_a],
                  [-sin_a, cos_a]])
    
    # Rotate points
    rotated = contour_points @ R.T
    
    # Find axis-aligned bounding box in rotated space
    min_x, min_y = rotated.min(axis=0)
    max_x, max_y = rotated.max(axis=0)
    
    # 4 corners in rotated space
    corners_rot = np.array([
        [min_x, min_y],  # TL
        [max_x, min_y],  # TR
        [max_x, max_y],  # BR
        [min_x, max_y],  # BL
    ])
    
    # Rotate back to original space
    R_inv = np.array([[cos_a, -sin_a],
                      [sin_a, cos_a]])
    corners_orig = corners_rot @ R_inv.T
    
    return corners_orig


def farthest_point_sampling_anchored(points: np.ndarray, n_samples: int, 
                                      anchor_points: np.ndarray) -> np.ndarray:
    """
    FPS with anchor points (corners) that are already selected.
    Returns indices of selected points from `points` array.
    """
    if n_samples <= 0:
        return np.array([], dtype=int)
    
    if len(points) <= n_samples:
        return np.arange(len(points))
    
    n_points = len(points)
    
    # Initialize distances from anchor points
    min_distances = np.full(n_points, np.inf)
    for anchor in anchor_points:
        dists = np.linalg.norm(points - anchor, axis=1)
        min_distances = np.minimum(min_distances, dists)
    
    selected_indices = []
    
    for _ in range(n_samples):
        idx = np.argmax(min_distances)
        selected_indices.append(idx)
        new_dists = np.linalg.norm(points - points[idx], axis=1)
        min_distances = np.minimum(min_distances, new_dists)
    
    return np.array(selected_indices)


def relax_segment_points(corner_start: np.ndarray, corner_end: np.ndarray,
                         fps_points_3d: np.ndarray, segment_3d: np.ndarray,
                         target_length: float, iterations: int = 50,
                         lr: float = 0.1) -> np.ndarray:
    """
    Apply repulsion relaxation to FPS points along a contour segment.
    
    Uses 1D parametric movement along the segment (by index) to avoid
    issues with 3D forces being perpendicular to curved contours.
    
    Args:
        corner_start: 3D position of start corner (FIXED)
        corner_end: 3D position of end corner (FIXED)
        fps_points_3d: (N, 3) FPS points to relax
        segment_3d: (M, 3) dense contour segment for snapping
        target_length: target edge length
        iterations: number of relaxation iterations
        lr: learning rate
    
    Returns:
        relaxed_fps_points: (N, 3) relaxed positions
    """
    if len(fps_points_3d) == 0:
        return fps_points_3d.copy()
    
    n_fps = len(fps_points_3d)
    n_segment = len(segment_3d)
    
    # Find initial indices of FPS points on segment
    fps_indices = []
    for pt in fps_points_3d:
        dists = np.linalg.norm(segment_3d - pt, axis=1)
        fps_indices.append(np.argmin(dists))
    fps_indices = np.array(fps_indices, dtype=np.float64)
    
    # Chain indices: [0 (corner_start), fps_idx_0, fps_idx_1, ..., n_segment-1 (corner_end)]
    # We'll work in index space (0 to n_segment-1)
    # corner_start maps to index -1 (virtual), corner_end maps to index n_segment (virtual)
    
    # Compute cumulative arc length along segment for accurate distance computation
    arc_lengths = np.zeros(n_segment)
    for i in range(1, n_segment):
        arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(segment_3d[i] - segment_3d[i-1])
    
    # Distance from corner_start to segment[0]
    dist_start_to_seg0 = np.linalg.norm(segment_3d[0] - corner_start)
    # Distance from segment[-1] to corner_end
    dist_segN_to_end = np.linalg.norm(corner_end - segment_3d[-1])
    
    def get_distance(idx1, idx2):
        """Get 3D distance between two positions (can be -1 for start, n_segment for end)."""
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        if idx1 < 0:  # corner_start
            if idx2 < 0:
                return 0
            elif idx2 >= n_segment:
                return dist_start_to_seg0 + arc_lengths[-1] + dist_segN_to_end
            else:
                return dist_start_to_seg0 + arc_lengths[int(idx2)]
        elif idx2 >= n_segment:  # corner_end
            return (arc_lengths[-1] - arc_lengths[int(idx1)]) + dist_segN_to_end
        else:
            return arc_lengths[int(idx2)] - arc_lengths[int(idx1)]
    
    # Debug: before
    chain_indices = np.concatenate([[-1], fps_indices, [n_segment]])
    before_lengths = [get_distance(chain_indices[i], chain_indices[i+1]) for i in range(len(chain_indices)-1)]
    
    for _ in range(iterations):
        # Compute forces in index space
        forces = np.zeros(n_fps)
        
        chain_indices = np.concatenate([[-1], fps_indices, [n_segment]])
        
        for i in range(len(chain_indices) - 1):
            dist = get_distance(chain_indices[i], chain_indices[i+1])
            force = (dist - target_length) * lr
            
            # Apply force to movable points only
            if i > 0:  # fps point at position i-1 in fps_indices
                forces[i-1] += force  # push right (increase index)
            if i < n_fps:  # fps point at position i in fps_indices
                forces[i] -= force  # push left (decrease index)
        
        # Update indices
        fps_indices = fps_indices + forces
        
        # Clamp to valid range
        fps_indices = np.clip(fps_indices, 0, n_segment - 1)
        
        # Ensure ordering is maintained
        for i in range(1, n_fps):
            if fps_indices[i] < fps_indices[i-1]:
                fps_indices[i] = fps_indices[i-1]
    
    # Debug: after
    chain_indices = np.concatenate([[-1], fps_indices, [n_segment]])
    after_lengths = [get_distance(chain_indices[i], chain_indices[i+1]) for i in range(len(chain_indices)-1)]
    print(f"      Repulsion: before={[f'{l:.1f}' for l in before_lengths]} -> after={[f'{l:.1f}' for l in after_lengths]} (target={target_length:.1f})")
    
    # Convert indices back to 3D points
    result = np.array([segment_3d[int(round(idx))] for idx in fps_indices])
    return result


def initialize_rect_with_real_contour(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                                      fx, fy, cx, cy,
                                      n_corners: int = 10,
                                      grid_rows: int = 8, grid_cols: int = 8) -> tuple:
    """
    Rectangle inpaint + real contour keypoints:
    
    1. Fit rectangle, 8×8 bilinear → interior grid points + edge_length
    2. Detect corners on real contour
    3. FPS fill contour segments based on edge_length
    4. Build edges: contour sequential, interior grid, interior↔contour
    
    Returns:
        keypoints: (N, 3) array
        edges: List of (i, j)
        types: List of 'corner', 'contour', 'interior'
        info: Dict with metadata
    """
    H, W = mask.shape
    valid_mask = mask & (depth > 1020) & (depth < max_depth)
    valid_mask_uint8 = (valid_mask > 0).astype(np.uint8) * 255
    
    # Get contour
    contours, _ = cv2.findContours(valid_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_2d = largest_contour.squeeze()  # (M, 2) in (col, row)
    
    # ================================================================
    # STEP 1: Find orientation from max inscribed rect, then min bounding rect
    # ================================================================
    print("  Step 1: Find orientation + bounding rectangle")
    
    # Find max inscribed rectangle to get ORIENTATION only
    inscribed_result = find_max_inscribed_rectangle_rotated(valid_mask, n_angles=90)
    if inscribed_result is None:
        print("    ERROR: Could not find inscribed rectangle")
        return None, None, None, None
    
    insc_cx, insc_cy, insc_w, insc_h, insc_angle = inscribed_result
    print(f"    Inscribed rect angle: {insc_angle:.1f}° (used for orientation only)")
    
    # Find min bounding rectangle with SAME orientation as inscribed
    bounding_corners_2d = get_bounding_rect_same_orientation(contour_2d, insc_angle)
    
    # Order bounding corners: TL, TR, BR, BL (based on position)
    # Find corner with min (x + y) as TL
    min_sum_idx = np.argmin(bounding_corners_2d[:, 0] + bounding_corners_2d[:, 1])
    bounding_corners_2d = np.roll(bounding_corners_2d, -min_sum_idx, axis=0)
    
    # Check orientation: TR should be to the right of TL
    v1 = bounding_corners_2d[1] - bounding_corners_2d[0]
    v2 = bounding_corners_2d[3] - bounding_corners_2d[0]
    if v1[0] < v2[0]:
        bounding_corners_2d = bounding_corners_2d[[0, 3, 2, 1]]
    
    TL_2d, TR_2d, BR_2d, BL_2d = bounding_corners_2d
    
    # Calculate dimensions from BOUNDING rectangle
    rect_width = np.linalg.norm(TR_2d - TL_2d)
    rect_height = np.linalg.norm(BL_2d - TL_2d)
    edge_length_px = (rect_width / (grid_cols - 1) + rect_height / (grid_rows - 1)) / 2
    
    # Average depth for 3D conversion
    avg_depth = np.mean(depth[valid_mask])
    edge_length_3d = edge_length_px * avg_depth / fx
    
    print(f"    Bounding rect: {rect_width:.1f} x {rect_height:.1f} px")
    print(f"    Edge length: {edge_length_px:.1f} px, ~{edge_length_3d:.1f} mm")
    
    # Create FULL raw grid on BOUNDING rectangle (8×8 points for visualization)
    raw_grid_3d = np.zeros((grid_rows, grid_cols, 3))
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = c / (grid_cols - 1)
            v = r / (grid_rows - 1)
            
            top = (1 - u) * TL_2d + u * TR_2d
            bottom = (1 - u) * BL_2d + u * BR_2d
            pt_2d = (1 - v) * top + v * bottom
            
            col_px, row_px = pt_2d
            col_int, row_int = int(round(col_px)), int(round(row_px))
            
            # Get depth (use avg_depth if invalid)
            if 0 <= row_int < H and 0 <= col_int < W:
                z = depth[row_int, col_int]
                if z <= 0 or z >= max_depth:
                    z = avg_depth
            else:
                z = avg_depth
            
            x = (col_px - cx) * z / fx
            y = (row_px - cy) * z / fy
            raw_grid_3d[r, c] = [x, y, z]
    
    # Create interior AND outside grid points based on mask
    # Interior = grid point is INSIDE the real contour (valid_mask == True)
    # Outside = grid point is OUTSIDE the real contour (valid_mask == False)
    interior_grid_3d = []
    interior_grid_2d = []
    interior_grid_rc = []  # (row, col) position in grid
    
    outside_grid_3d = []
    outside_grid_2d = []
    outside_grid_rc = []  # (row, col) position in grid
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = c / (grid_cols - 1)
            v = r / (grid_rows - 1)
            
            # Bilinear interpolation on BOUNDING rectangle
            top = (1 - u) * TL_2d + u * TR_2d
            bottom = (1 - u) * BL_2d + u * BR_2d
            pt_2d = (1 - v) * top + v * bottom
            
            col_px, row_px = pt_2d
            col_int, row_int = int(round(col_px)), int(round(row_px))
            
            # Check if inside mask
            is_inside = (0 <= row_int < H and 0 <= col_int < W and valid_mask[row_int, col_int])
            
            # Get 3D position (use avg_depth for outside points or invalid depth)
            if 0 <= row_int < H and 0 <= col_int < W:
                z = depth[row_int, col_int]
                if z <= 0 or z >= max_depth:
                    z = avg_depth
            else:
                z = avg_depth
            
            x = (col_px - cx) * z / fx
            y = (row_px - cy) * z / fy
            pt_3d = [x, y, z]
            
            if is_inside:
                interior_grid_3d.append(pt_3d)
                interior_grid_2d.append([col_px, row_px])
                interior_grid_rc.append((r, c))
            else:
                outside_grid_3d.append(pt_3d)
                outside_grid_2d.append([col_px, row_px])
                outside_grid_rc.append((r, c))
    
    interior_grid_3d = np.array(interior_grid_3d) if interior_grid_3d else np.zeros((0, 3))
    outside_grid_3d = np.array(outside_grid_3d) if outside_grid_3d else np.zeros((0, 3))
    n_interior = len(interior_grid_3d)
    n_outside = len(outside_grid_3d)
    print(f"    Interior grid points (inside mask): {n_interior}")
    print(f"    Outside grid points (outside mask): {n_outside}")
    
    # Convert bounding rectangle corners to 3D (for visualization)
    rect_corners_3d = []
    for col, row in bounding_corners_2d:
        row_int, col_int = int(np.clip(row, 0, H-1)), int(np.clip(col, 0, W-1))
        z = depth[row_int, col_int]
        if z <= 0 or z >= max_depth:
            z = avg_depth
        x = (col - cx) * z / fx
        y = (row - cy) * z / fy
        rect_corners_3d.append([x, y, z])
    rect_corners_3d = np.array(rect_corners_3d)
    
    # ================================================================
    # STEP 2: Detect corners on REAL contour
    # ================================================================
    print("  Step 2: Detect corners on real contour")
    
    peri = cv2.arcLength(largest_contour, True)
    
    # Binary search for epsilon
    eps_low, eps_high = 0.001, 0.1
    best_corners = None
    
    for _ in range(20):
        eps_mid = (eps_low + eps_high) / 2
        approx = cv2.approxPolyDP(largest_contour, eps_mid * peri, True)
        n_found = len(approx)
        
        if n_found == n_corners:
            best_corners = approx.squeeze()
            break
        elif n_found > n_corners:
            eps_low = eps_mid
        else:
            eps_high = eps_mid
        
        if best_corners is None or abs(n_found - n_corners) < abs(len(best_corners) - n_corners):
            best_corners = approx.squeeze()
    
    corners_2d = best_corners
    print(f"    Detected {len(corners_2d)} corners")
    
    # Convert corners to 3D
    corners_3d = []
    corners_2d_valid = []
    for col, row in corners_2d:
        row_int, col_int = int(row), int(col)
        if 0 <= row_int < H and 0 <= col_int < W:
            z = depth[row_int, col_int]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                corners_3d.append([x, y, z])
                corners_2d_valid.append([col, row])
    
    corners_3d = np.array(corners_3d)
    corners_2d_valid = np.array(corners_2d_valid)
    n_corners_valid = len(corners_3d)
    print(f"    Valid corners: {n_corners_valid}")
    
    # ================================================================
    # STEP 3: Build dense contour and find corner indices
    # ================================================================
    print("  Step 3: FPS fill contour segments")
    
    # Build dense contour 3D
    contour_3d_dense = []
    contour_2d_dense = []
    for col, row in contour_2d:
        row_int, col_int = int(row), int(col)
        if 0 <= row_int < H and 0 <= col_int < W:
            z = depth[row_int, col_int]
            if 0 < z < max_depth:
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                contour_3d_dense.append([x, y, z])
                contour_2d_dense.append([col, row])
    
    contour_3d_dense = np.array(contour_3d_dense)
    contour_2d_dense = np.array(contour_2d_dense)
    
    # Find contour indices for each corner
    corner_contour_indices = []
    for c2d in corners_2d_valid:
        dists = np.linalg.norm(contour_2d_dense - c2d, axis=1)
        corner_contour_indices.append(np.argmin(dists))
    
    # Sort corners by contour order
    sorted_order = np.argsort(corner_contour_indices)
    corners_3d = corners_3d[sorted_order]
    corners_2d_valid = corners_2d_valid[sorted_order]
    corner_contour_indices = [corner_contour_indices[i] for i in sorted_order]
    
    # ================================================================
    # STEP 4: FPS on each contour segment + repulsion relaxation
    # ================================================================
    contour_keypoints_3d = []
    contour_keypoints_2d = []
    contour_types = []
    
    for i in range(n_corners_valid):
        # Add corner
        contour_keypoints_3d.append(corners_3d[i])
        contour_keypoints_2d.append(corners_2d_valid[i])
        contour_types.append('corner')
        
        # Get segment
        idx_start = corner_contour_indices[i]
        idx_end = corner_contour_indices[(i + 1) % n_corners_valid]
        
        if idx_end > idx_start:
            segment_3d = contour_3d_dense[idx_start+1:idx_end]
            segment_2d = contour_2d_dense[idx_start+1:idx_end]
        else:
            segment_3d = np.vstack([contour_3d_dense[idx_start+1:], contour_3d_dense[:idx_end]]) if idx_start+1 < len(contour_3d_dense) else contour_3d_dense[:idx_end]
            segment_2d = np.vstack([contour_2d_dense[idx_start+1:], contour_2d_dense[:idx_end]]) if idx_start+1 < len(contour_2d_dense) else contour_2d_dense[:idx_end]
        
        if len(segment_3d) < 2:
            continue
        
        # Calculate segment length
        segment_length = np.sum(np.linalg.norm(np.diff(segment_3d, axis=0), axis=1))
        
        # Number of FPS points (round to nearest, then subtract 1)
        # e.g., 4.2 -> round to 4 -> 3 FPS points
        # e.g., 4.6 -> round to 5 -> 4 FPS points
        n_intervals = int(round(segment_length / edge_length_3d))
        n_fps = max(0, n_intervals - 1)
        n_fps = min(n_fps, len(segment_3d) - 1, 6)  # Cap at 4
        
        print(f"    Segment {i}: length={segment_length:.1f}mm, ratio={segment_length/edge_length_3d:.1f}, n_fps={n_fps}")
        
        if n_fps > 0:
            # Anchored FPS
            anchor_pts = np.array([corners_3d[i], corners_3d[(i + 1) % n_corners_valid]])
            fps_indices = farthest_point_sampling_anchored(segment_3d, n_fps, anchor_pts)
            
            # IMPORTANT: Sort FPS indices by their position along the segment
            # so they maintain contour order (not FPS selection order)
            fps_indices_sorted = np.sort(fps_indices)
            
            # Get FPS points
            fps_points_3d = segment_3d[fps_indices_sorted]
            fps_points_2d = segment_2d[fps_indices_sorted]
            
            # Apply repulsion relaxation to achieve uniform spacing
            corner_start = corners_3d[i]
            corner_end = corners_3d[(i + 1) % n_corners_valid]
            
            relaxed_fps_3d = relax_segment_points(
                corner_start, corner_end,
                fps_points_3d, segment_3d,
                target_length=edge_length_3d,
                iterations=100, lr=0.15
            )
            
            # Update 2D positions by finding nearest on segment_2d
            # (since we snapped to segment_3d, find corresponding 2D)
            for j, pt_3d in enumerate(relaxed_fps_3d):
                # Find nearest point in segment_3d to get index
                dists = np.linalg.norm(segment_3d - pt_3d, axis=1)
                nearest_idx = np.argmin(dists)
                
                contour_keypoints_3d.append(pt_3d)
                contour_keypoints_2d.append(segment_2d[nearest_idx])
                contour_types.append('contour')
    
    contour_keypoints_3d = np.array(contour_keypoints_3d)
    contour_keypoints_2d = np.array(contour_keypoints_2d)
    n_contour = len(contour_keypoints_3d)
    
    n_corners_final = sum(1 for t in contour_types if t == 'corner')
    n_fps_final = sum(1 for t in contour_types if t == 'contour')
    print(f"    Contour keypoints: {n_contour} ({n_corners_final} corners, {n_fps_final} FPS)")
    
    # ================================================================
    # STEP 5: Build contour↔interior connections via boundary detection
    # ================================================================
    print("  Step 5: Build contour↔interior mapping")
    
    # Build lookups
    outside_rc_set = set(outside_grid_rc)
    interior_rc_set = set(interior_grid_rc)
    outside_rc_to_3d = {rc: outside_grid_3d[i] for i, rc in enumerate(outside_grid_rc)}
    interior_rc_to_3d = {rc: interior_grid_3d[i] for i, rc in enumerate(interior_grid_rc)}
    rc_to_interior_idx = {rc: i for i, rc in enumerate(interior_grid_rc)}
    
    # Find BOUNDARY interior points: interior points that have at least one outside neighbor
    boundary_interior_rc = set()
    for r, c in interior_rc_set:
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        for nr, nc in neighbors:
            if (nr, nc) in outside_rc_set:
                boundary_interior_rc.add((r, c))
                break
    
    print(f"    Boundary interior points: {len(boundary_interior_rc)}")
    
    # For each contour point, find the nearest BOUNDARY interior point directly
    # This is more robust than going through outside nodes
    contour_to_interior_connections = {}  # contour_idx -> set of interior_idx
    interior_to_contour_connections = {}  # interior_idx -> set of contour_idx
    
    if len(boundary_interior_rc) > 0:
        boundary_interior_list = list(boundary_interior_rc)
        boundary_interior_3d = np.array([interior_rc_to_3d[rc] for rc in boundary_interior_list])
        
        for contour_idx in range(n_contour):
            contour_pt_3d = contour_keypoints_3d[contour_idx]
            
            # Find nearest boundary interior point
            dists = np.linalg.norm(boundary_interior_3d - contour_pt_3d, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            
            # Only connect if within reasonable distance (2x edge length)
            if nearest_dist < 2.0 * edge_length_3d:
                nearest_rc = boundary_interior_list[nearest_idx]
                int_idx = rc_to_interior_idx[nearest_rc]
                
                if contour_idx not in contour_to_interior_connections:
                    contour_to_interior_connections[contour_idx] = set()
                contour_to_interior_connections[contour_idx].add(int_idx)
                
                if int_idx not in interior_to_contour_connections:
                    interior_to_contour_connections[int_idx] = set()
                interior_to_contour_connections[int_idx].add(contour_idx)
    
    # Also keep the outside_rc mapping for visualization
    contour_to_outside_rc = {}
    if len(outside_rc_set) > 0:
        outside_rc_list = list(outside_rc_set)
        outside_3d_arr = np.array([outside_rc_to_3d[rc] for rc in outside_rc_list])
        for contour_idx in range(n_contour):
            contour_pt_3d = contour_keypoints_3d[contour_idx]
            dists = np.linalg.norm(outside_3d_arr - contour_pt_3d, axis=1)
            nearest_idx = np.argmin(dists)
            contour_to_outside_rc[contour_idx] = outside_rc_list[nearest_idx]
    
    print(f"    {len(interior_to_contour_connections)} interior points have contour connections")
    print(f"    {len(contour_to_interior_connections)} contour points have interior connections")
    
    # ================================================================
    # STEP 6: Merge close pairs (interior→contour)
    # ================================================================
    print("  Step 6: Merge close interior-contour pairs")
    
    merge_threshold = 0.35 * edge_length_3d
    
    # Check ALL interior points against ALL contour points for closeness
    # Not just connected ones - this catches edge cases
    merged_interior_to_contour = {}  # interior_idx -> contour_idx (merge target)
    interior_to_keep = []
    
    for int_idx in range(len(interior_grid_rc)):
        int_pt = interior_grid_3d[int_idx]
        int_rc = interior_grid_rc[int_idx]
        
        # Find closest contour point (check all, not just connected)
        dists = np.linalg.norm(contour_keypoints_3d - int_pt, axis=1)
        min_dist = np.min(dists)
        closest_contour = np.argmin(dists)
        
        # Merge if close enough
        if min_dist < merge_threshold:
            merged_interior_to_contour[int_idx] = closest_contour
            print(f"    Merged interior {int_idx} ({int_rc}) -> contour {closest_contour} (dist={min_dist:.1f}mm)")
        else:
            interior_to_keep.append(int_idx)
    
    # Build merged_rc_to_contour for later use
    merged_rc_to_contour = {}
    for int_idx, contour_idx in merged_interior_to_contour.items():
        rc = interior_grid_rc[int_idx]
        merged_rc_to_contour[rc] = contour_idx
    
    # Update interior grid (remove merged ones)
    interior_grid_3d_new = interior_grid_3d[interior_to_keep] if interior_to_keep else np.zeros((0, 3))
    interior_grid_rc_new = [interior_grid_rc[i] for i in interior_to_keep]
    
    # Build mapping from old interior idx to new interior idx
    old_to_new_interior_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(interior_to_keep)}
    
    # ================================================================
    # STEP 6b: Merge close interior-interior pairs (with transitive closure)
    # ================================================================
    print("  Step 6b: Merge close interior-interior pairs")
    
    interior_merge_threshold = 0.4 * edge_length_3d
    
    # Use Union-Find for transitive merging
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # Path compression
            return self.parent[x]
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return
            # Union by rank - keep lower index as root for consistency
            if rx < ry:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry
    
    n_new_interior = len(interior_grid_rc_new)
    uf = UnionFind(n_new_interior)
    
    # Build rc to new interior idx mapping
    rc_to_new_interior_idx = {rc: i for i, rc in enumerate(interior_grid_rc_new)}
    
    # Find pairs to merge
    for i, (r, c) in enumerate(interior_grid_rc_new):
        pt_i = interior_grid_3d_new[i]
        
        # Check grid neighbors
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        for nr, nc in neighbors:
            if (nr, nc) in rc_to_new_interior_idx:
                j = rc_to_new_interior_idx[(nr, nc)]
                if j <= i:
                    continue
                
                pt_j = interior_grid_3d_new[j]
                dist = np.linalg.norm(pt_i - pt_j)
                
                if dist < interior_merge_threshold:
                    uf.union(i, j)
                    print(f"    Merged interior {interior_grid_rc_new[j]} -> interior {interior_grid_rc_new[i]} (dist={dist:.1f}mm)")
    
    # Find unique roots (points to keep)
    roots = set()
    for i in range(n_new_interior):
        roots.add(uf.find(i))
    
    final_interior_indices = sorted(roots)  # Keep sorted for consistency
    interior_grid_3d_final = interior_grid_3d_new[final_interior_indices] if final_interior_indices else np.zeros((0, 3))
    interior_grid_rc_final = [interior_grid_rc_new[i] for i in final_interior_indices]
    
    # Build mapping: new_interior_idx -> final_interior_idx
    new_to_final_interior_idx = {}
    for final_idx, new_idx in enumerate(final_interior_indices):
        new_to_final_interior_idx[new_idx] = final_idx
    
    n_interior = len(interior_grid_3d_final)
    n_merged_to_contour = len(merged_interior_to_contour)
    n_merged_interior = n_new_interior - n_interior
    print(f"    Kept {n_interior} interior points")
    print(f"    Merged {n_merged_to_contour} interior -> contour")
    print(f"    Merged {n_merged_interior} interior -> interior")
    
    # ================================================================
    # STEP 7: Build edges - CLEAN GRID with ~4 edges per interior
    # ================================================================
    print("  Step 7: Build edges")
    
    # All keypoints: contour first, then interior
    if n_interior > 0:
        all_keypoints = np.vstack([contour_keypoints_3d, interior_grid_3d_final])
    else:
        all_keypoints = contour_keypoints_3d
    
    all_types = contour_types + ['interior'] * n_interior
    
    edges = set()
    
    # 1. Contour edges: sequential (closed loop)
    for i in range(n_contour):
        j = (i + 1) % n_contour
        edge = (min(i, j), max(i, j))
        edges.add(edge)
    
    print(f"    Contour edges (sequential): {n_contour}")
    
    # Build lookup for final interior positions
    rc_to_final_interior_idx = {rc: i for i, rc in enumerate(interior_grid_rc_final)}
    
    # Helper: get final global index for a grid position (with redirect for merged points)
    # Returns (global_idx, is_contour) or (None, None)
    def get_final_idx(r, c):
        # Merged to contour?
        if (r, c) in merged_rc_to_contour:
            return merged_rc_to_contour[(r, c)], True
        
        # Check if it was an original interior point
        if (r, c) not in rc_to_interior_idx:
            return None, None
        
        old_int_idx = rc_to_interior_idx[(r, c)]
        
        # Was it merged to contour?
        if old_int_idx in merged_interior_to_contour:
            return merged_interior_to_contour[old_int_idx], True
        
        # Get new interior index
        if old_int_idx not in old_to_new_interior_idx:
            return None, None
        new_int_idx = old_to_new_interior_idx[old_int_idx]
        
        # Follow union-find to get root
        root_new_idx = uf.find(new_int_idx)
        
        # Get final index
        if root_new_idx not in new_to_final_interior_idx:
            return None, None
        final_int_idx = new_to_final_interior_idx[root_new_idx]
        
        return n_contour + final_int_idx, False
    
    # 2. Interior grid edges: each original interior connects to RIGHT and DOWN neighbors
    interior_edges = 0
    
    for r, c in interior_grid_rc:  # All ORIGINAL interior positions
        src_idx, src_is_contour = get_final_idx(r, c)
        if src_idx is None:
            continue
        
        # Only RIGHT and DOWN to avoid duplicates
        for nr, nc in [(r, c + 1), (r + 1, c)]:
            dst_idx, dst_is_contour = get_final_idx(nr, nc)
            if dst_idx is None or src_idx == dst_idx:
                continue
            # Do NOT add contour-to-contour (those are FIXED sequential)
            if src_is_contour and dst_is_contour:
                continue
            edge = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            edges.add(edge)
            interior_edges += 1
    
    print(f"    Interior grid edges: {interior_edges}")
    
    # 3. Contour ↔ Interior edges from pre-computed mapping
    cross_edges = 0
    for contour_idx, connected_interiors in contour_to_interior_connections.items():
        for old_int_idx in connected_interiors:
            old_rc = interior_grid_rc[old_int_idx]
            final_idx, is_contour = get_final_idx(*old_rc)
            
            if final_idx is None or final_idx == contour_idx:
                continue
            # No contour-to-contour
            if is_contour:
                continue
            
            edge = (min(contour_idx, final_idx), max(contour_idx, final_idx))
            edges.add(edge)
            cross_edges += 1
    
    print(f"    Contour↔Interior edges: {cross_edges}")
    
    edges = list(edges)
    print(f"    Total edges (before fixes): {len(edges)}")
    
    # ================================================================
    # STEP 8: Fix interior nodes with < 4 edges
    # ================================================================
    print("  Step 8: Fix interior nodes with < 4 edges")
    
    # Build adjacency list for current edges
    def build_adjacency(edge_list, n_nodes):
        adj = {i: set() for i in range(n_nodes)}
        for i, j in edge_list:
            adj[i].add(j)
            adj[j].add(i)
        return adj
    
    n_total = len(all_keypoints)
    adjacency = build_adjacency(edges, n_total)
    
    # For each interior node, check degree and add edges if needed
    # Interior nodes are at indices [n_contour, n_contour + n_interior)
    edges_added_step8 = 0
    
    # Build reverse lookup: final_interior_idx -> rc
    final_interior_idx_to_rc = {i: rc for i, rc in enumerate(interior_grid_rc_final)}
    
    for int_local_idx in range(n_interior):
        global_idx = n_contour + int_local_idx
        current_degree = len(adjacency[global_idx])
        
        if current_degree >= 4:
            continue
        
        # Get grid position of this interior node
        rc = final_interior_idx_to_rc[int_local_idx]
        r, c = rc
        int_pt_3d = all_keypoints[global_idx]
        
        # First pass: Check all 4 grid neighbors and add edges if missing
        grid_neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        
        for nr, nc in grid_neighbors:
            if current_degree >= 4:
                break
            
            # Get final index of neighbor
            neighbor_final_idx, neighbor_is_contour = get_final_idx(nr, nc)
            
            if neighbor_final_idx is None:
                continue
            
            # Check if edge already exists
            if neighbor_final_idx in adjacency[global_idx]:
                continue
            
            # Add edge
            edge = (min(global_idx, neighbor_final_idx), max(global_idx, neighbor_final_idx))
            if edge not in edges:
                edges.append(edge)
                adjacency[global_idx].add(neighbor_final_idx)
                adjacency[neighbor_final_idx].add(global_idx)
                edges_added_step8 += 1
                current_degree += 1
                print(f"    Added grid edge: interior {int_local_idx} ({rc}) <-> node {neighbor_final_idx}")
        
        # Second pass: If still < 4 edges, connect to nearest contour nodes
        if current_degree < 4:
            # Find contour nodes not yet connected, sorted by distance
            contour_distances = []
            for contour_idx in range(n_contour):
                if contour_idx in adjacency[global_idx]:
                    continue  # Already connected
                contour_pt_3d = all_keypoints[contour_idx]
                dist = np.linalg.norm(int_pt_3d - contour_pt_3d)
                contour_distances.append((dist, contour_idx))
            
            contour_distances.sort(key=lambda x: x[0])
            
            # Add edges to nearest contour nodes until degree >= 4
            for dist, contour_idx in contour_distances:
                if current_degree >= 4:
                    break
                
                # Only connect if within reasonable distance (1.5x edge length)
                if dist > 1.5 * edge_length_3d:
                    print(f"    Warning: interior {int_local_idx} ({rc}) has only {current_degree} edges, nearest contour too far ({dist:.1f}mm)")
                    break
                
                edge = (min(global_idx, contour_idx), max(global_idx, contour_idx))
                if edge not in edges:
                    edges.append(edge)
                    adjacency[global_idx].add(contour_idx)
                    adjacency[contour_idx].add(global_idx)
                    edges_added_step8 += 1
                    current_degree += 1
                    print(f"    Added contour edge: interior {int_local_idx} ({rc}) <-> contour {contour_idx} (dist={dist:.1f}mm)")
        
        if current_degree < 4:
            print(f"    WARNING: interior {int_local_idx} ({rc}) still has only {current_degree} edges!")
    
    print(f"    Added {edges_added_step8} edges to fix interior degrees")
    
    # ================================================================
    # STEP 9: Remove contour FPS nodes with 0 interior connections
    # ================================================================
    print("  Step 9: Remove contour FPS nodes with 0 interior connections")
    
    # Rebuild adjacency after step 8
    adjacency = build_adjacency(edges, n_total)
    
    # Find contour FPS nodes (non-corner) with 0 interior connections
    fps_to_remove = []  # List of global indices to remove
    
    for contour_idx in range(n_contour):
        if contour_types[contour_idx] == 'corner':
            continue  # Skip corners
        
        # Count interior connections
        interior_connections = 0
        for neighbor_idx in adjacency[contour_idx]:
            if neighbor_idx >= n_contour:  # It's an interior node
                interior_connections += 1
        
        if interior_connections == 0:
            fps_to_remove.append(contour_idx)
            print(f"    Marking FPS node {contour_idx} for removal (0 interior connections)")
    
    if len(fps_to_remove) == 0:
        print("    No FPS nodes to remove")
    else:
        print(f"    Removing {len(fps_to_remove)} FPS nodes")
        
        # Build mapping from old index to new index
        # Nodes not in fps_to_remove get new sequential indices
        fps_to_remove_set = set(fps_to_remove)
        old_to_new_idx = {}
        new_idx = 0
        for old_idx in range(n_total):
            if old_idx not in fps_to_remove_set:
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        # Update keypoints
        keep_mask = [i not in fps_to_remove_set for i in range(n_total)]
        all_keypoints = all_keypoints[keep_mask]
        
        # Update types
        all_types = [t for i, t in enumerate(all_types) if i not in fps_to_remove_set]
        
        # Update edges: remap indices and handle contour reconnection
        new_edges = set()
        
        # First, find contour neighbors for removed nodes (for reconnection)
        # Contour is a sequential loop: node i connects to (i-1) % n_contour and (i+1) % n_contour
        for old_idx in fps_to_remove:
            prev_old = (old_idx - 1) % n_contour
            next_old = (old_idx + 1) % n_contour
            
            # Find valid prev (skip if also removed)
            while prev_old in fps_to_remove_set and prev_old != old_idx:
                prev_old = (prev_old - 1) % n_contour
            
            # Find valid next (skip if also removed)
            while next_old in fps_to_remove_set and next_old != old_idx:
                next_old = (next_old + 1) % n_contour
            
            # Add reconnection edge (prev -> next) if both valid and different
            if prev_old not in fps_to_remove_set and next_old not in fps_to_remove_set and prev_old != next_old:
                new_prev = old_to_new_idx[prev_old]
                new_next = old_to_new_idx[next_old]
                edge = (min(new_prev, new_next), max(new_prev, new_next))
                new_edges.add(edge)
        
        # Remap all existing edges (excluding those involving removed nodes)
        for i, j in edges:
            if i in fps_to_remove_set or j in fps_to_remove_set:
                continue
            new_i = old_to_new_idx[i]
            new_j = old_to_new_idx[j]
            edge = (min(new_i, new_j), max(new_i, new_j))
            new_edges.add(edge)
        
        edges = list(new_edges)
        
        # Update counts
        n_contour = sum(1 for t in all_types if t in ['corner', 'contour'])
        n_interior = sum(1 for t in all_types if t == 'interior')
        
        print(f"    After removal: {n_contour} contour nodes, {n_interior} interior nodes")
    
    print(f"    Final total edges: {len(edges)}")
    
    # Info dict
    info = {
        'bounding_corners_2d': bounding_corners_2d,
        'bounding_corners_3d': rect_corners_3d,
        'orientation_angle': insc_angle,
        'contour_3d_dense': contour_3d_dense,
        'edge_length_3d': edge_length_3d,
        'n_contour': n_contour,
        'n_interior': n_interior,
        'interior_grid_rc': interior_grid_rc_final,
        'raw_grid_3d': raw_grid_3d,  # Full grid on BOUNDING rect
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'contour_to_outside_rc': contour_to_outside_rc,  # For debug visualization
        'outside_rc_to_3d': outside_rc_to_3d,  # For debug visualization
        'outside_grid_rc': outside_grid_rc,  # List of (r,c) for outside nodes
    }
    
    return all_keypoints, edges, all_types, info


def visualize_3d(keypoints: np.ndarray, edges: list, types: list,
                 contour_3d: np.ndarray, rect_corners_3d: np.ndarray,
                 raw_grid_3d: np.ndarray = None,
                 point_cloud: np.ndarray = None,
                 contour_to_outside_rc: dict = None,
                 outside_rc_to_3d: dict = None,
                 save_path: str = None):
    """
    3D visualization with:
    - Point cloud (gray, sparse)
    - Contour line (cyan)
    - Bounding rectangle (magenta dashed)
    - Raw grid on bounding (yellow)
    - Keypoints colored by GRADIENT (to show chain order)
    - Edges (blue)
    - Chain mapping lines (contour -> outside grid point)
    """
    import plotly.express as px
    
    fig = go.Figure()
    
    # 1. Point cloud (sparse)
    if point_cloud is not None:
        pc_sparse = point_cloud[::20]
        fig.add_trace(go.Scatter3d(
            x=pc_sparse[:, 0], y=pc_sparse[:, 1], z=pc_sparse[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.3),
            name='Point Cloud'
        ))
    
    # 2. Contour line (cyan, thin)
    fig.add_trace(go.Scatter3d(
        x=contour_3d[:, 0], y=contour_3d[:, 1], z=contour_3d[:, 2],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Dense Contour'
    ))
    
    # 3. Bounding rectangle outline (magenta)
    rect_closed = np.vstack([rect_corners_3d, rect_corners_3d[0]])
    fig.add_trace(go.Scatter3d(
        x=rect_closed[:, 0], y=rect_closed[:, 1], z=rect_closed[:, 2],
        mode='lines',
        line=dict(color='magenta', width=4, dash='dash'),
        name='Bounding Rect'
    ))
    
    # 3.5 Raw grid on BOUNDING rect (light gray, thin)
    if raw_grid_3d is not None:
        grid_rows, grid_cols = raw_grid_3d.shape[:2]
        
        # Raw grid edges (light gray, thin)
        for r in range(grid_rows):
            for c in range(grid_cols):
                # Horizontal edge
                if c < grid_cols - 1:
                    p1, p2 = raw_grid_3d[r, c], raw_grid_3d[r, c+1]
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        showlegend=False
                    ))
                # Vertical edge
                if r < grid_rows - 1:
                    p1, p2 = raw_grid_3d[r, c], raw_grid_3d[r+1, c]
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        showlegend=False
                    ))
    
    # 4. Edges (blue)
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[keypoints[i, 0], keypoints[j, 0]],
            y=[keypoints[i, 1], keypoints[j, 1]],
            z=[keypoints[i, 2], keypoints[j, 2]],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    
    # 5. Keypoints by type with gradient coloring for contour
    # Contour points with gradient (to show order)
    contour_indices = [i for i, t in enumerate(types) if t in ['corner', 'contour']]
    n_contour = len(contour_indices)
    
    if n_contour > 0:
        contour_pts = keypoints[contour_indices]
        colors = np.linspace(0, 1, n_contour)
        
        # FPS points (green)
        fps_local = [i for i, idx in enumerate(contour_indices) if types[idx] == 'contour']
        if fps_local:
            fps_pts = contour_pts[fps_local]
            fig.add_trace(go.Scatter3d(
                x=fps_pts[:, 0], y=fps_pts[:, 1], z=fps_pts[:, 2],
                mode='markers',
                marker=dict(size=7, color='green', line=dict(width=1, color='black')),
                name=f'FPS ({len(fps_local)})'
            ))
        
        # Corner points (red, same size as FPS)
        corner_local = [i for i, idx in enumerate(contour_indices) if types[idx] == 'corner']
        if corner_local:
            corner_pts = contour_pts[corner_local]
            fig.add_trace(go.Scatter3d(
                x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
                mode='markers',
                marker=dict(size=7, color='red', symbol='circle', line=dict(width=1, color='black')),
                name=f'Corners ({len(corner_local)})'
            ))
    
    # 6. Interior points (orange)
    interior_indices = [i for i, t in enumerate(types) if t == 'interior']
    if interior_indices:
        interior_pts = keypoints[interior_indices]
        fig.add_trace(go.Scatter3d(
            x=interior_pts[:, 0], y=interior_pts[:, 1], z=interior_pts[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='orange',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name=f'Interior ({len(interior_indices)})'
        ))
    
    # Layout
    fig.update_layout(
        title='Cloth Initialization: Interior Grid + Real Contour',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        width=1200,
        height=800,
        legend=dict(x=0.02, y=0.98)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved 3D visualization to: {save_path}")
    
    return fig


def extract_point_cloud(mask: np.ndarray, depth: np.ndarray, max_depth: float,
                        fx, fy, cx, cy) -> np.ndarray:
    """Extract 3D point cloud from mask and depth."""
    valid = mask & (depth > 1020) & (depth < max_depth)
    rows, cols = np.where(valid)
    
    if len(rows) == 0:
        return np.array([]).reshape(0, 3)
    
    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    
    return np.stack([x, y, z], axis=1)


def visualize_3d_matplotlib(keypoints: np.ndarray, edges: list, types: list,
                            contour_3d: np.ndarray, point_cloud: np.ndarray = None,
                            save_path: str = None, title: str = None):
    """
    Save 3D visualization as PNG using matplotlib.
    Shows: point cloud (colored by depth), contour line, keypoints, edges.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Point cloud (colored by depth/z)
    if point_cloud is not None and len(point_cloud) > 0:
        pc_sparse = point_cloud[::10]  # Subsample for speed
        sc = ax.scatter(pc_sparse[:, 0], pc_sparse[:, 1], pc_sparse[:, 2],
                       c=pc_sparse[:, 2], cmap='viridis', s=1, alpha=0.3, label='Point Cloud')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Depth (mm)')
    
    # 2. Contour line (cyan)
    if contour_3d is not None and len(contour_3d) > 0:
        ax.plot(contour_3d[:, 0], contour_3d[:, 1], contour_3d[:, 2],
               'c-', linewidth=1, alpha=0.7, label='Dense Contour')
    
    # 3. Edges (blue)
    for i, j in edges:
        ax.plot([keypoints[i, 0], keypoints[j, 0]],
               [keypoints[i, 1], keypoints[j, 1]],
               [keypoints[i, 2], keypoints[j, 2]],
               'b-', linewidth=1, alpha=0.6)
    
    # 4. Keypoints by type
    # FPS points (green)
    fps_indices = [i for i, t in enumerate(types) if t == 'contour']
    if fps_indices:
        fps_pts = keypoints[fps_indices]
        ax.scatter(fps_pts[:, 0], fps_pts[:, 1], fps_pts[:, 2],
                  c='green', s=30, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'FPS ({len(fps_indices)})')
    
    # Corner points (red)
    corner_indices = [i for i, t in enumerate(types) if t == 'corner']
    if corner_indices:
        corner_pts = keypoints[corner_indices]
        ax.scatter(corner_pts[:, 0], corner_pts[:, 1], corner_pts[:, 2],
                  c='red', s=30, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'Corners ({len(corner_indices)})')
    
    # Interior points (orange)
    interior_indices = [i for i, t in enumerate(types) if t == 'interior']
    if interior_indices:
        interior_pts = keypoints[interior_indices]
        ax.scatter(interior_pts[:, 0], interior_pts[:, 1], interior_pts[:, 2],
                  c='orange', s=25, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'Interior ({len(interior_indices)})')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    if title:
        ax.set_title(title)
    
    ax.legend(loc='upper left', fontsize=8)
    
    # Set equal aspect ratio
    if len(keypoints) > 0:
        max_range = np.array([keypoints[:, 0].max() - keypoints[:, 0].min(),
                             keypoints[:, 1].max() - keypoints[:, 1].min(),
                             keypoints[:, 2].max() - keypoints[:, 2].min()]).max() / 2.0
        mid_x = (keypoints[:, 0].max() + keypoints[:, 0].min()) * 0.5
        mid_y = (keypoints[:, 1].max() + keypoints[:, 1].min()) * 0.5
        mid_z = (keypoints[:, 2].max() + keypoints[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set view to XY plane (looking down Z axis)
    ax.view_init(elev=-90, azim=-90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG visualization to: {save_path}")
    
    plt.close(fig)
    return fig


def visualize_3d_matplotlib_to_array(keypoints: np.ndarray, edges: list, types: list,
                                      contour_3d: np.ndarray, point_cloud: np.ndarray = None,
                                      title: str = None):
    """
    Render 3D visualization to numpy array for video creation.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Point cloud (colored by depth/z)
    if point_cloud is not None and len(point_cloud) > 0:
        pc_sparse = point_cloud[::10]
        sc = ax.scatter(pc_sparse[:, 0], pc_sparse[:, 1], pc_sparse[:, 2],
                       c=pc_sparse[:, 2], cmap='viridis', s=1, alpha=0.3, label='Point Cloud')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Depth (mm)')
    
    # 2. Contour line (cyan)
    if contour_3d is not None and len(contour_3d) > 0:
        ax.plot(contour_3d[:, 0], contour_3d[:, 1], contour_3d[:, 2],
               'c-', linewidth=1, alpha=0.7, label='Dense Contour')
    
    # 3. Edges (blue)
    for i, j in edges:
        ax.plot([keypoints[i, 0], keypoints[j, 0]],
               [keypoints[i, 1], keypoints[j, 1]],
               [keypoints[i, 2], keypoints[j, 2]],
               'b-', linewidth=1, alpha=0.6)
    
    # 4. Keypoints by type
    fps_indices = [i for i, t in enumerate(types) if t == 'contour']
    if fps_indices:
        fps_pts = keypoints[fps_indices]
        ax.scatter(fps_pts[:, 0], fps_pts[:, 1], fps_pts[:, 2],
                  c='green', s=30, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'FPS ({len(fps_indices)})')
    
    corner_indices = [i for i, t in enumerate(types) if t == 'corner']
    if corner_indices:
        corner_pts = keypoints[corner_indices]
        ax.scatter(corner_pts[:, 0], corner_pts[:, 1], corner_pts[:, 2],
                  c='red', s=30, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'Corners ({len(corner_indices)})')
    
    interior_indices = [i for i, t in enumerate(types) if t == 'interior']
    if interior_indices:
        interior_pts = keypoints[interior_indices]
        ax.scatter(interior_pts[:, 0], interior_pts[:, 1], interior_pts[:, 2],
                  c='orange', s=25, marker='o', edgecolors='black', linewidths=0.5,
                  label=f'Interior ({len(interior_indices)})')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    if title:
        ax.set_title(title)
    
    ax.legend(loc='upper left', fontsize=8)
    
    if len(keypoints) > 0:
        max_range = np.array([keypoints[:, 0].max() - keypoints[:, 0].min(),
                             keypoints[:, 1].max() - keypoints[:, 1].min(),
                             keypoints[:, 2].max() - keypoints[:, 2].min()]).max() / 2.0
        mid_x = (keypoints[:, 0].max() + keypoints[:, 0].min()) * 0.5
        mid_y = (keypoints[:, 1].max() + keypoints[:, 1].min()) * 0.5
        mid_z = (keypoints[:, 2].max() + keypoints[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=-90, azim=-90)
    
    plt.tight_layout()
    
    # Render to numpy array
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGBA to RGB
    
    plt.close(fig)
    return img


def main():
    import cv2
    
    # Paths
    data_dir = Path("/home/yehengz/deformable_seg/data/arm_traj5_cloth")
    rgbd_path = data_dir / "rgbd.npz"
    masks_dir = data_dir / "masks"
    output_dir = data_dir / "init_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera intrinsics
    fx, fy, cx, cy = 606.6875, 606.24609375, 641.7900390625, 366.8428955078125
    max_depth = 1200.0
    
    # Load depth data
    print("Loading RGBD data...")
    rgbd_data = np.load(str(rgbd_path))
    all_depth = rgbd_data['depth']  # Shape: (N_frames, H, W)
    n_frames = len(all_depth)
    print(f"Total frames: {n_frames}")
    
    # Find all mask files
    mask_files = sorted(masks_dir.glob("mask_frame_*.npy"))
    print(f"Found {len(mask_files)} mask files")
    
    # Statistics storage
    stats = []
    
    # Video writer (will be initialized on first frame)
    video_writer = None
    video_path = output_dir / "init_video.mp4"
    
    for mask_file in mask_files:
        # Extract frame number from filename
        frame_str = mask_file.stem.split('_')[-1]  # e.g., "0000"
        frame_idx = int(frame_str)
        
        print(f"\n{'='*60}")
        print(f"Processing Frame {frame_idx}")
        print(f"{'='*60}")
        
        # Load mask and depth
        mask = np.load(str(mask_file))
        
        if frame_idx >= n_frames:
            print(f"  Warning: frame {frame_idx} exceeds depth array size {n_frames}, skipping")
            continue
        
        depth = all_depth[frame_idx]
        
        print(f"  Mask pixels: {np.sum(mask)}")
        
        # Skip if mask is too small
        if np.sum(mask) < 100:
            print(f"  Skipping: mask too small")
            stats.append({
                'frame': frame_idx,
                'n_nodes': 0,
                'n_edges': 0,
                'n_corners': 0,
                'n_fps': 0,
                'n_interior': 0,
                'status': 'SKIPPED'
            })
            continue
        
        # Extract point cloud for visualization
        point_cloud = extract_point_cloud(mask, depth, max_depth, fx, fy, cx, cy)
        print(f"  Point cloud: {len(point_cloud)} points")
        
        # Run initialization
        keypoints, edges, types, info = initialize_rect_with_real_contour(
            mask, depth, max_depth, fx, fy, cx, cy,
            n_corners=10, grid_rows=10, grid_cols=10
        )
        
        if keypoints is None:
            print(f"  Failed to initialize!")
            stats.append({
                'frame': frame_idx,
                'n_nodes': 0,
                'n_edges': 0,
                'n_corners': 0,
                'n_fps': 0,
                'n_interior': 0,
                'status': 'FAILED'
            })
            continue
        
        # Count node types
        n_corners = sum(1 for t in types if t == 'corner')
        n_fps = sum(1 for t in types if t == 'contour')
        n_interior = sum(1 for t in types if t == 'interior')
        n_nodes = len(keypoints)
        n_edges = len(edges)
        
        # Save statistics
        stats.append({
            'frame': frame_idx,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_corners': n_corners,
            'n_fps': n_fps,
            'n_interior': n_interior,
            'status': 'OK'
        })
        
        print(f"  Result: {n_nodes} nodes ({n_corners} corners, {n_fps} FPS, {n_interior} interior), {n_edges} edges")
        
        # Render frame to array
        title = f"Frame {frame_idx}: {n_nodes} nodes, {n_edges} edges"
        frame_img = visualize_3d_matplotlib_to_array(
            keypoints, edges, types,
            contour_3d=info['contour_3d_dense'],
            point_cloud=point_cloud,
            title=title
        )
        
        # Initialize video writer on first frame
        if video_writer is None:
            h, w = frame_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
            print(f"  Video initialized: {w}x{h}")
        
        # Write frame (convert RGB to BGR for OpenCV)
        video_writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {video_path}")
    
    # Save statistics to single file
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    
    stats_path = output_dir / "init_stats.csv"
    with open(stats_path, 'w') as f:
        f.write("frame,n_nodes,n_edges,n_corners,n_fps,n_interior,status\n")
        for s in stats:
            f.write(f"{s['frame']},{s['n_nodes']},{s['n_edges']},{s['n_corners']},{s['n_fps']},{s['n_interior']},{s['status']}\n")
            print(f"  Frame {s['frame']:4d}: {s['n_nodes']:3d} nodes, {s['n_edges']:3d} edges - {s['status']}")
    
    print(f"\nStatistics saved to: {stats_path}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
