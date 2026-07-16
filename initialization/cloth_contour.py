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
