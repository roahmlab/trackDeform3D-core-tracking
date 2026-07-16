"""Frame-0 initialization HTML visualizations (plotly) for cloth and fabric.

The two variants share the same skeleton but are NOT identical: the cloth one
adds detected-corner / corner-segment / face traces and uses different styling.
Kept as two explicit functions (verbatim moves from the drivers).
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

def save_init_visualization_3d_fabric(
    keypoints: np.ndarray,
    edges: list,
    point_cloud: np.ndarray,
    save_path: Path,
    corner_indices: list = None,
    border_indices: list = None,
    downsample_pc: int = 2000,
    contour_3d: np.ndarray = None,
    contour_3d_raw: np.ndarray = None,
    ee_poses: np.ndarray = None,
    segment_lengths: dict = None,
):
    """Save interactive 3D visualization of initialization using Plotly."""
    if keypoints is None or len(keypoints) == 0:
        print("  [Init Vis] No keypoints to visualize")
        return

    traces = []

    # Downsample point cloud if needed
    if point_cloud is not None and len(point_cloud) > 0:
        pc = point_cloud.copy()
        if len(pc) > downsample_pc:
            indices = np.random.choice(len(pc), downsample_pc, replace=False)
            pc = pc[indices]

        traces.append(go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='lightgrey', opacity=0.5),
            name='Point Cloud',
            hoverinfo='skip',
        ))

    # Raw/noisy contour trace (red dashed line)
    if contour_3d_raw is not None and len(contour_3d_raw) > 0:
        contour_raw_vis = contour_3d_raw[::5] if len(contour_3d_raw) > 200 else contour_3d_raw
        contour_raw_vis = np.vstack([contour_raw_vis, contour_raw_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_raw_vis[:, 0], y=contour_raw_vis[:, 1], z=contour_raw_vis[:, 2],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Raw Contour',
            hoverinfo='skip',
        ))

    # Denoised contour trace (blue solid line)
    if contour_3d is not None and len(contour_3d) > 0:
        contour_vis = contour_3d[::5] if len(contour_3d) > 200 else contour_3d
        contour_vis = np.vstack([contour_vis, contour_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_vis[:, 0], y=contour_vis[:, 1], z=contour_vis[:, 2],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Denoised Contour',
            hoverinfo='skip',
        ))

    # EE poses (purple)
    if ee_poses is not None and len(ee_poses) > 0:
        valid_ee = ~np.any(np.isnan(ee_poses), axis=1)
        ee_valid = ee_poses[valid_ee]
        ee_idx = np.where(valid_ee)[0]
        if len(ee_valid) > 0:
            traces.append(go.Scatter3d(
                x=ee_valid[:, 0], y=ee_valid[:, 1], z=ee_valid[:, 2],
                mode='markers',
                marker=dict(size=12, color='purple', symbol='diamond'),
                name='EE Poses',
                text=[f'EE{i}' for i in ee_idx],
                hoverinfo='text',
            ))

    # Edge traces (blue lines)
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            edge_x.extend([keypoints[i, 0], keypoints[j, 0], None])
            edge_y.extend([keypoints[i, 1], keypoints[j, 1], None])
            edge_z.extend([keypoints[i, 2], keypoints[j, 2], None])

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Edges',
        hoverinfo='skip',
    ))

    # Keypoint traces - color by type
    corner_indices = corner_indices or []
    border_indices = border_indices or []

    # Interior nodes (green)
    interior_mask = [i not in corner_indices and i not in border_indices for i in range(len(keypoints))]
    interior_pts = keypoints[interior_mask]
    interior_idx = [i for i in range(len(keypoints)) if interior_mask[i]]
    if len(interior_pts) > 0:
        traces.append(go.Scatter3d(
            x=interior_pts[:, 0], y=interior_pts[:, 1], z=interior_pts[:, 2],
            mode='markers',
            marker=dict(size=6, color='green'),
            name='Interior',
            text=[f'Node {i}' for i in interior_idx],
            hoverinfo='text',
        ))

    # Border nodes (orange)
    border_pts = keypoints[[i for i in border_indices if i < len(keypoints)]]
    border_idx = [i for i in border_indices if i < len(keypoints)]
    if len(border_pts) > 0:
        traces.append(go.Scatter3d(
            x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
            mode='markers',
            marker=dict(size=8, color='orange'),
            name='Border',
            text=[f'Node {i}' for i in border_idx],
            hoverinfo='text',
        ))

    # Corner nodes (red, larger)
    corner_pts = keypoints[[i for i in corner_indices if i < len(keypoints)]]
    corner_idx = [i for i in corner_indices if i < len(keypoints)]
    if len(corner_pts) > 0:
        traces.append(go.Scatter3d(
            x=corner_pts[:, 0], y=corner_pts[:, 1], z=corner_pts[:, 2],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Corners',
            text=[f'Corner {i}' for i in corner_idx],
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)

    # Compute edge length stats for title
    edge_lengths = [np.linalg.norm(keypoints[i] - keypoints[j]) for i, j in edges if i < len(keypoints) and j < len(keypoints)]
    if edge_lengths:
        avg_len = np.mean(edge_lengths)
        std_len = np.std(edge_lengths)
        title = f'Init: {len(keypoints)} nodes, {len(edges)} edges | Avg edge: {avg_len:.1f}mm, Std: {std_len:.1f}mm ({std_len/avg_len*100:.1f}%)'
    else:
        title = f'Init: {len(keypoints)} nodes, {len(edges)} edges'

    if segment_lengths:
        seg_str = ' | Seg: ' + ', '.join([f'{k}:{v:.0f}mm' for k, v in segment_lengths.items()])
        title += seg_str

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [Init Vis] Saved 3D visualization to {save_path}")


def save_init_visualization_3d_cloth(
    keypoints: np.ndarray,
    edges: list,
    point_cloud: np.ndarray,
    save_path: Path,
    corner_indices: list = None,
    border_indices: list = None,
    downsample_pc: int = 2000,
    contour_3d: np.ndarray = None,
    contour_3d_raw: np.ndarray = None,
    ee_poses: np.ndarray = None,
    segment_lengths: dict = None,
    rect_corners_3d: np.ndarray = None,
    detected_corners_3d: np.ndarray = None,
    all_grid_edges: list = None,
    border_grid_indices: list = None,
    valid_faces: list = None,
):
    """Save interactive 3D visualization of initialization using Plotly."""
    if keypoints is None or len(keypoints) == 0:
        print("  [Init Vis] No keypoints to visualize")
        return

    traces = []

    # Downsample point cloud if needed
    if point_cloud is not None and len(point_cloud) > 0:
        pc = point_cloud.copy()
        if len(pc) > downsample_pc:
            indices = np.random.choice(len(pc), downsample_pc, replace=False)
            pc = pc[indices]

        traces.append(go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=pc[:, 2],
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='Depth (mm)', x=1.02, len=0.5),
            ),
            name='Point Cloud',
            hoverinfo='skip',
        ))

    # Raw/noisy contour trace (red dashed line)
    if contour_3d_raw is not None and len(contour_3d_raw) > 0:
        contour_raw_vis = contour_3d_raw[::5] if len(contour_3d_raw) > 200 else contour_3d_raw
        contour_raw_vis = np.vstack([contour_raw_vis, contour_raw_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_raw_vis[:, 0], y=contour_raw_vis[:, 1], z=contour_raw_vis[:, 2],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Raw Contour',
            hoverinfo='skip',
        ))

    # Denoised contour trace (green solid line)
    if contour_3d is not None and len(contour_3d) > 0:
        contour_vis = contour_3d[::5] if len(contour_3d) > 200 else contour_3d
        contour_vis = np.vstack([contour_vis, contour_vis[0:1]])
        traces.append(go.Scatter3d(
            x=contour_vis[:, 0], y=contour_vis[:, 1], z=contour_vis[:, 2],
            mode='lines',
            line=dict(color='lime', width=5),
            name='Contour',
            hoverinfo='skip',
        ))

    # Detected corners on contour
    if detected_corners_3d is not None and len(detected_corners_3d) > 0:
        valid_corners = ~np.any(np.isnan(detected_corners_3d), axis=1)
        corners_valid = detected_corners_3d[valid_corners]
        corner_labels = [f'C{i}' for i in range(len(detected_corners_3d))]
        valid_labels = [corner_labels[i] for i in range(len(detected_corners_3d)) if valid_corners[i]]
        if len(corners_valid) > 0:
            traces.append(go.Scatter3d(
                x=corners_valid[:, 0], y=corners_valid[:, 1], z=corners_valid[:, 2],
                mode='markers',
                marker=dict(size=14, color='yellow', symbol='diamond',
                           line=dict(color='black', width=2)),
                name='Detected Corners',
                text=valid_labels,
                hoverinfo='text',
            ))

    # Straight lines between consecutive corner pairs
    if detected_corners_3d is not None and len(detected_corners_3d) > 1:
        seg_colors = ['red', 'blue', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink']
        n_c = len(detected_corners_3d)
        for seg_idx in range(n_c):
            c_s = detected_corners_3d[seg_idx]
            c_e = detected_corners_3d[(seg_idx + 1) % n_c]
            if np.any(np.isnan(c_s)) or np.any(np.isnan(c_e)):
                continue
            color = seg_colors[seg_idx % len(seg_colors)]
            traces.append(go.Scatter3d(
                x=[c_s[0], c_e[0]], y=[c_s[1], c_e[1]], z=[c_s[2], c_e[2]],
                mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                name=f'Line C{seg_idx}→C{(seg_idx+1)%n_c}',
                showlegend=False,
            ))

    # EE poses (purple)
    if ee_poses is not None and len(ee_poses) > 0:
        valid_ee = ~np.any(np.isnan(ee_poses), axis=1)
        ee_valid = ee_poses[valid_ee]
        ee_idx = np.where(valid_ee)[0]
        if len(ee_valid) > 0:
            traces.append(go.Scatter3d(
                x=ee_valid[:, 0], y=ee_valid[:, 1], z=ee_valid[:, 2],
                mode='markers',
                marker=dict(size=12, color='purple', symbol='diamond'),
                name='EE Poses',
                text=[f'EE{i}' for i in ee_idx],
                hoverinfo='text',
            ))

    # Bounding rectangle (cyan dashed line)
    if rect_corners_3d is not None and len(rect_corners_3d) == 4:
        rect_closed = np.vstack([rect_corners_3d, rect_corners_3d[0:1]])
        traces.append(go.Scatter3d(
            x=rect_closed[:, 0], y=rect_closed[:, 1], z=rect_closed[:, 2],
            mode='lines+markers',
            line=dict(color='cyan', width=6, dash='dash'),
            marker=dict(size=10, color='cyan', symbol='square'),
            name='Bounding Rect',
            text=['TL', 'TR', 'BR', 'BL', 'TL'],
            hoverinfo='text',
        ))

    # Full rectangular grid edges (gray)
    grid_positions = None
    if all_grid_edges is not None and len(all_grid_edges) > 0:
        full_grid_x, full_grid_y, full_grid_z = [], [], []
        n_grid = int(np.sqrt(len(keypoints)))
        if rect_corners_3d is not None and len(rect_corners_3d) == 4:
            TL, TR, BR, BL = rect_corners_3d
            grid_positions = np.zeros((len(keypoints), 3))
            for idx in range(len(keypoints)):
                row, col = idx // n_grid, idx % n_grid
                u = col / (n_grid - 1)
                v = row / (n_grid - 1)
                top = (1 - u) * TL + u * TR
                bottom = (1 - u) * BL + u * BR
                grid_positions[idx] = (1 - v) * top + v * bottom

            for i, j in all_grid_edges:
                if i < len(grid_positions) and j < len(grid_positions):
                    full_grid_x.extend([grid_positions[i, 0], grid_positions[j, 0], None])
                    full_grid_y.extend([grid_positions[i, 1], grid_positions[j, 1], None])
                    full_grid_z.extend([grid_positions[i, 2], grid_positions[j, 2], None])

            traces.append(go.Scatter3d(
                x=full_grid_x, y=full_grid_y, z=full_grid_z,
                mode='lines',
                line=dict(color='lightgray', width=1),
                name='Bilinear Grid (reference)',
                hoverinfo='skip',
                opacity=0.4,
            ))

    # Quad faces (semi-transparent mesh)
    if valid_faces is not None and len(valid_faces) > 0:
        vert_set = set()
        for tl, tr, br, bl in valid_faces:
            vert_set.update([tl, tr, br, bl])
        vert_list = sorted(vert_set)
        vert_remap = {v: i for i, v in enumerate(vert_list)}
        mesh_verts = keypoints[vert_list]

        tri_i, tri_j, tri_k = [], [], []
        for tl, tr, br, bl in valid_faces:
            if any(np.any(np.isnan(keypoints[idx])) for idx in [tl, tr, br, bl]):
                continue
            a, b, c, d = vert_remap[tl], vert_remap[tr], vert_remap[br], vert_remap[bl]
            tri_i.extend([a, a])
            tri_j.extend([b, c])
            tri_k.extend([c, d])

        if tri_i:
            traces.append(go.Mesh3d(
                x=mesh_verts[:, 0], y=mesh_verts[:, 1], z=mesh_verts[:, 2],
                i=tri_i, j=tri_j, k=tri_k,
                color='lightskyblue',
                opacity=0.3,
                name=f'Faces ({len(valid_faces)} quads)',
                hoverinfo='skip',
            ))

    # T-Line edges (blue)
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            if np.any(np.isnan(keypoints[i])) or np.any(np.isnan(keypoints[j])):
                continue
            edge_x.extend([keypoints[i, 0], keypoints[j, 0], None])
            edge_y.extend([keypoints[i, 1], keypoints[j, 1], None])
            edge_z.extend([keypoints[i, 2], keypoints[j, 2], None])

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='T-Line Edges',
        hoverinfo='skip',
    ))

    # ACTUAL keypoint positions with indices — split by corner / border / interior
    valid_idx = [i for i in range(len(keypoints)) if not np.any(np.isnan(keypoints[i]))]
    n_grid = int(np.sqrt(len(keypoints)))
    corner_set = set(corner_indices) if corner_indices else set()
    border_set = set(border_indices) if border_indices else set()

    corner_idx = [i for i in valid_idx if i in corner_set]
    border_idx = [i for i in valid_idx if i in border_set and i not in corner_set]
    interior_idx = [i for i in valid_idx if i not in corner_set and i not in border_set]

    print(f"  [Init Vis] Corners: {corner_idx}")
    print(f"  [Init Vis] Border (non-corner): {border_idx}")
    print(f"  [Init Vis] Interior: {interior_idx}")

    if corner_idx:
        pts = keypoints[corner_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='purple', symbol='diamond', opacity=1.0),
            text=[str(i) for i in corner_idx],
            textposition='top center',
            textfont=dict(size=11, color='purple'),
            name=f'Corners ({len(corner_idx)})',
            hovertext=[f'Corner idx {i} = [{i//n_grid},{i%n_grid}]' for i in corner_idx],
            hoverinfo='text',
        ))
    if border_idx:
        pts = keypoints[border_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=12, color='gold', opacity=0.9),
            text=[str(i) for i in border_idx],
            textposition='top center',
            textfont=dict(size=9, color='goldenrod'),
            name=f'Border ({len(border_idx)})',
            hovertext=[f'Border idx {i} = [{i//n_grid},{i%n_grid}]' for i in border_idx],
            hoverinfo='text',
        ))
    if interior_idx:
        pts = keypoints[interior_idx]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='green', opacity=0.9),
            text=[str(i) for i in interior_idx],
            textposition='top center',
            textfont=dict(size=8, color='green'),
            name=f'Interior ({len(interior_idx)})',
            hovertext=[f'Interior idx {i} = [{i//n_grid},{i%n_grid}]' for i in interior_idx],
            hoverinfo='text',
        ))

    # Border nodes (orange circles)
    if border_grid_indices is not None and len(border_grid_indices) > 0:
        border_valid_idx = []
        for idx, grid_idx in enumerate(border_grid_indices):
            if grid_idx < len(keypoints) and not np.any(np.isnan(keypoints[grid_idx])):
                border_valid_idx.append(idx)

        if len(border_valid_idx) > 0:
            border_grid_nodes = [border_grid_indices[i] for i in border_valid_idx]
            border_pts = keypoints[border_grid_nodes]
            traces.append(go.Scatter3d(
                x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
                mode='markers',
                marker=dict(size=12, color='orange', symbol='circle',
                           line=dict(color='black', width=2)),
                name=f'Border ({len(border_grid_nodes)} nodes)',
                hovertext=[f'Border[{i}] = Grid {border_grid_indices[i]}' for i in border_valid_idx],
                hoverinfo='text',
            ))

            border_edge_x, border_edge_y, border_edge_z = [], [], []
            for i in range(len(border_grid_nodes)):
                j = (i + 1) % len(border_grid_nodes)
                p1 = keypoints[border_grid_nodes[i]]
                p2 = keypoints[border_grid_nodes[j]]
                border_edge_x.extend([p1[0], p2[0], None])
                border_edge_y.extend([p1[1], p2[1], None])
                border_edge_z.extend([p1[2], p2[2], None])

            traces.append(go.Scatter3d(
                x=border_edge_x, y=border_edge_y, z=border_edge_z,
                mode='lines',
                line=dict(color='orange', width=3, dash='dash'),
                name='Border Chain',
                hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)

    # Compute edge length stats for title (filter NaN)
    edge_lengths = []
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            if not np.any(np.isnan(keypoints[i])) and not np.any(np.isnan(keypoints[j])):
                edge_lengths.append(np.linalg.norm(keypoints[i] - keypoints[j]))

    n_valid = np.sum(~np.any(np.isnan(keypoints), axis=1))
    n_valid_edges = len(edge_lengths)

    if edge_lengths:
        avg_len = np.mean(edge_lengths)
        std_len = np.std(edge_lengths)
        n_faces = len(valid_faces) if valid_faces else 0
        title = f'Init: {n_valid} nodes, {n_valid_edges} edges, {n_faces} faces | Avg edge: {avg_len:.1f}mm, Std: {std_len:.1f}mm ({std_len/avg_len*100:.1f}%)'
    else:
        n_faces = len(valid_faces) if valid_faces else 0
        title = f'Init: {n_valid} nodes, {n_valid_edges} edges, {n_faces} faces'

    if segment_lengths:
        seg_str = ' | Seg: ' + ', '.join([f'{k}:{v:.0f}mm' for k, v in segment_lengths.items()])
        title += seg_str

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [Init Vis] Saved 3D visualization to {save_path}")


def save_border_init_visualization(
    keypoints: np.ndarray,
    border_grid_indices: list,
    detected_corners_3d: np.ndarray,
    contour_3d: np.ndarray,
    save_path: Path,
    segment_interior_nodes: list = None,
):
    """
    Simple visualization to check border initialization.

    Shows:
    - Green contour line
    - Yellow diamonds: detected corners (C0-C7)
    - Colored circles: border grid nodes (each segment different color)
    - Blue text: grid index at each border node
    """
    if keypoints is None or border_grid_indices is None:
        print("  [Border Viz] Missing data")
        return

    # Segment colors (8 distinct colors for 8 segments)
    segment_colors = [
        'red', 'blue', 'magenta', 'cyan',
        'orange', 'purple', 'brown', 'pink'
    ]

    traces = []

    # Contour (green line)
    if contour_3d is not None and len(contour_3d) > 0:
        traces.append(go.Scatter3d(
            x=contour_3d[:, 0], y=contour_3d[:, 1], z=contour_3d[:, 2],
            mode='lines',
            line=dict(color='green', width=3),
            name='Contour',
        ))

    # Detected corners (yellow diamonds with C0-C7 labels)
    if detected_corners_3d is not None and len(detected_corners_3d) > 0:
        traces.append(go.Scatter3d(
            x=detected_corners_3d[:, 0], y=detected_corners_3d[:, 1], z=detected_corners_3d[:, 2],
            mode='markers+text',
            marker=dict(size=14, color='yellow', symbol='diamond',
                       line=dict(color='black', width=2)),
            text=[f'C{i}' for i in range(len(detected_corners_3d))],
            textposition='top center',
            textfont=dict(size=14, color='black'),
            name=f'Corners ({len(detected_corners_3d)})',
        ))

        # Straight lines between each consecutive corner pair (dashed, per-segment color)
        n_c = len(detected_corners_3d)
        for seg_idx in range(n_c):
            c_start = detected_corners_3d[seg_idx]
            c_end = detected_corners_3d[(seg_idx + 1) % n_c]
            if np.any(np.isnan(c_start)) or np.any(np.isnan(c_end)):
                continue
            color = segment_colors[seg_idx % len(segment_colors)]
            traces.append(go.Scatter3d(
                x=[c_start[0], c_end[0]], y=[c_start[1], c_end[1]], z=[c_start[2], c_end[2]],
                mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                name=f'Line C{seg_idx}→C{(seg_idx+1)%n_c}',
                showlegend=False,
            ))

    # Border grid nodes - color by segment
    border_pts = []
    border_labels = []
    valid_indices = []
    n_nan_border = 0
    for i, grid_idx in enumerate(border_grid_indices):
        if grid_idx < len(keypoints) and not np.any(np.isnan(keypoints[grid_idx])):
            border_pts.append(keypoints[grid_idx])
            border_labels.append(f'{grid_idx}')
            valid_indices.append(i)
        else:
            n_nan_border += 1
    print(f"  [Border Viz] border_grid_indices has {len(border_grid_indices)} entries, "
          f"{len(border_pts)} valid, {n_nan_border} NaN")

    if len(border_pts) > 0:
        border_pts = np.array(border_pts)

        # Determine segment for each border node
        if segment_interior_nodes is not None:
            segment_starts = [0]
            for n_int in segment_interior_nodes:
                segment_starts.append(segment_starts[-1] + 1 + n_int)

            node_colors = []
            for orig_idx in valid_indices:
                seg_idx = 0
                for s in range(len(segment_starts) - 1):
                    if segment_starts[s] <= orig_idx < segment_starts[s + 1]:
                        seg_idx = s
                        break
                node_colors.append(segment_colors[seg_idx % len(segment_colors)])
        else:
            node_colors = ['orange'] * len(border_pts)

        traces.append(go.Scatter3d(
            x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
            mode='markers+text',
            marker=dict(size=12, color=node_colors, symbol='circle',
                       line=dict(color='black', width=2)),
            text=border_labels,
            textposition='bottom center',
            textfont=dict(size=10, color='blue'),
            name=f'Border Nodes ({len(border_pts)})',
            hovertext=[f'Border[{valid_indices[i]}] → Grid {border_grid_indices[valid_indices[i]]}' for i in range(len(border_pts))],
            hoverinfo='text',
        ))

        # Draw edges between consecutive border nodes, colored by segment
        if segment_interior_nodes is not None:
            segment_starts = [0]
            for n_int in segment_interior_nodes:
                segment_starts.append(segment_starts[-1] + 1 + n_int)

            for seg_idx in range(len(segment_interior_nodes)):
                seg_start = segment_starts[seg_idx]
                seg_end = segment_starts[seg_idx + 1]
                color = segment_colors[seg_idx % len(segment_colors)]

                edge_x, edge_y, edge_z = [], [], []
                for orig_i in range(seg_start, seg_end):
                    orig_j = orig_i + 1
                    if orig_j >= len(border_grid_indices):
                        orig_j = 0  # Wrap around

                    if orig_i in valid_indices and orig_j in valid_indices:
                        vi = valid_indices.index(orig_i)
                        vj = valid_indices.index(orig_j)
                        edge_x.extend([border_pts[vi, 0], border_pts[vj, 0], None])
                        edge_y.extend([border_pts[vi, 1], border_pts[vj, 1], None])
                        edge_z.extend([border_pts[vi, 2], border_pts[vj, 2], None])

                if edge_x:
                    traces.append(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color=color, width=4),
                        name=f'Seg {seg_idx}',
                        showlegend=True,
                    ))
        else:
            edge_x, edge_y, edge_z = [], [], []
            for i in range(len(border_pts)):
                j = (i + 1) % len(border_pts)
                edge_x.extend([border_pts[i, 0], border_pts[j, 0], None])
                edge_y.extend([border_pts[i, 1], border_pts[j, 1], None])
                edge_z.extend([border_pts[i, 2], border_pts[j, 2], None])

            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='orange', width=4, dash='dash'),
                name='Border Chain',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Border Init: {len(border_grid_indices)} samples → {len(border_pts)} grid nodes',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"  [Border Viz] Saved: {save_path}")
