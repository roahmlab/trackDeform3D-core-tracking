"""Single-panel tracking video renderer, shared by all four drivers.

Style follows deform_with_hands/render_raw_tracking.py:
  - one panel, no text overlays
  - background dimmed to a shadow; the object mask foreground kept bright
  - keypoints coloured on a pink -> blue gradient (per-node parameter t),
    connected by their skeleton/grid edges (edge colour = mean of endpoints)
  - anchor nodes drawn +2 px bigger
  - per-keypoint trajectory tails blending toward white with age, capped at
    TRAIL_WHITE (70% white + 30% node colour at the oldest end -- reads well on
    the shadowed background without washing out to pure white)
"""
from collections import defaultdict, deque

import cv2
import numpy as np

PINK = np.array([235.0, 120.0, 190.0])  # RGB
BLUE = np.array([70.0, 130.0, 235.0])   # RGB
KP_R = 5          # px; keypoint radius
ANCHOR_R = KP_R + 2
EDGE_W = 3        # px
TRAIL_W = 2       # px
TRAIL_WHITE = 0.7  # oldest tail segment = this much white blended into the node colour
SHADOW = 0.35     # background dim factor


def path_gradient_t(kp0, edges, start, end):
    """Gradient parameter t in [0,1] along the main path start -> end (arc length
    on frame-0 keypoints); side branches inherit their attachment's t."""
    adj = defaultdict(list)
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for n in adj[cur]:
            if n not in prev:
                prev[n] = cur
                q.append(n)
    path = [end]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path = path[::-1]
    seglen = [np.linalg.norm(kp0[path[k + 1]] - kp0[path[k]]) for k in range(len(path) - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    t = np.zeros(len(kp0))
    for k, node in enumerate(path):
        t[node] = cum[k] / max(cum[-1], 1e-9)
    q = deque(path)
    seen = set(path)
    while q:
        cur = q.popleft()
        for n in adj[cur]:
            if n not in seen:
                t[n] = t[cur]
                seen.add(n)
                q.append(n)
    return t


def grid_gradient_t(rows, cols):
    """Gradient parameter t by grid column (node index = row*cols + col)."""
    t = np.zeros(rows * cols)
    for r in range(rows):
        for c in range(cols):
            t[r * cols + c] = c / max(cols - 1, 1)
    return t


def render_tracking_video(out_path, colors_bgr, kp3d_seq, intrinsics, edges, node_t,
                          anchors=(), masks=None, fps=30, trail=15, shadow=SHADOW):
    """Render the clip video from 3D keypoints (mm, camera frame).

    Args:
        out_path: target mp4 path
        colors_bgr: (T,H,W,3) uint8 BGR frames (clip range, aligned with kp3d_seq)
        kp3d_seq: (T,K,3) keypoints; NaN rows (failed frames) are skipped
        intrinsics: 3x3 K
        edges: (E,2) int array
        node_t: (K,) gradient parameter in [0,1] (pink at 0 -> blue at 1)
        anchors: node indices drawn larger
        masks: (T,H,W) {0,1} foreground masks kept bright (optional)
    """
    kp3d_seq = np.asarray(kp3d_seq, dtype=np.float64)
    T, K_n = kp3d_seq.shape[:2]
    H, W = colors_bgr[0].shape[:2]
    edges = np.asarray(edges, dtype=int)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    kp_rgb = (1 - node_t[:, None]) * PINK + node_t[:, None] * BLUE
    kp_bgr = kp_rgb[:, ::-1].astype(int)
    edge_bgr = (0.5 * (kp_rgb[edges[:, 0]] + kp_rgb[edges[:, 1]]))[:, ::-1].astype(int)
    anchors = set(int(a) for a in anchors)

    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.stack([fx * kp3d_seq[..., 0] / kp3d_seq[..., 2] + cx,
                       fy * kp3d_seq[..., 1] / kp3d_seq[..., 2] + cy], -1)  # (T,K,2)

    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i in range(T):
        img = colors_bgr[i].astype(np.float32) * shadow
        if masks is not None:
            fg = masks[i] > 0
            img[fg] = colors_bgr[i][fg]
        img = np.ascontiguousarray(img.astype(np.uint8))

        j0 = max(0, i - trail)
        for k in range(K_n):
            for j in range(j0, i):
                p1, p2 = uv[j, k], uv[j + 1, k]
                if np.isfinite(p1).all() and np.isfinite(p2).all():
                    w = TRAIL_WHITE * (i - 1 - j) / max(trail, 1)  # newest -> oldest
                    col_rgb = (1 - w) * kp_rgb[k] + w * 255.0
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                             tuple(int(c) for c in col_rgb[::-1]), TRAIL_W, cv2.LINE_AA)

        for e_idx, (a_, b_) in enumerate(edges):
            p1, p2 = uv[i, a_], uv[i, b_]
            if np.isfinite(p1).all() and np.isfinite(p2).all():
                cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                         tuple(int(c) for c in edge_bgr[e_idx]), EDGE_W, cv2.LINE_AA)

        for k in range(K_n):
            p = uv[i, k]
            if np.isfinite(p).all():
                r = ANCHOR_R if k in anchors else KP_R
                cv2.circle(img, tuple(p.astype(int)), r,
                           tuple(int(c) for c in kp_bgr[k]), -1, cv2.LINE_AA)

        vw.write(img)
    vw.release()
