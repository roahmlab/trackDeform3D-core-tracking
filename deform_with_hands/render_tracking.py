"""Render the tracked BDLO over RGB -- single panel (grid-4 style), publication look.

  - no text overlays
  - MANO hand meshes rendered on top (HaMeR renderer: blue=left, pink=right)
  - keypoints coloured pink -> blue ALONG THE ROPE (right hand -> left hand,
    danglers inherit their branch colour), connected by their skeleton edges
  - anchor nodes (the two EE leaves + the two branch nodes) drawn +2 px bigger
  - 15-frame trajectory tail per keypoint, blending toward white with age
    (capped at 70% white so the node colour stays visible)
  - background rendered as shadow (dimmed); rope-mask foreground kept as-is

Needs the HaMeR renderer -> run with the hamer env:
  /home/yehengz/miniconda3/envs/hamer/bin/python render_raw_tracking.py \
      [path/to/smoothed_3d_keypoints.npz]
Default npz: output/tracking/clip_0/smoothed_3d_keypoints.npz
Writes tracking_deform_with_hands.mp4 next to the npz.
"""
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import HAMER_ROOT
sys.path.insert(0, str(HAMER_ROOT))
os.chdir(str(HAMER_ROOT))  # hamer resolves ./_DATA relative to cwd

import cv2
import numpy as np

from hamer.configs import get_config
from hamer.models import DEFAULT_CHECKPOINT
from hamer.utils.renderer import Renderer

from paths import UNDIST_NPZ as _U
UNDIST_NPZ = str(_U)
from paths import HANDS_NPZ as _H
HANDS_NPZ = str(_H)
from paths import ROPE_MASKS_NPZ as _RM
ROPE_NPZ = str(_RM)
DEFAULT_TRACKING = f'{DIR}/output/tracking/clip_0/smoothed_3d_keypoints.npz'

HAND_COLOR = [(0.44, 0.61, 0.86), (0.90, 0.55, 0.72)]  # RGB 0-1: blue=left, pink=right
PINK = np.array([235.0, 120.0, 190.0])  # RGB
BLUE = np.array([70.0, 130.0, 235.0])
TRAIL = 15        # trajectory tail (frames)
KP_R = 5          # px; keypoint radius
ANCHOR_R = KP_R + 2  # px; EE leaves + branch nodes (the anchors)
EDGE_W = 3        # px
TRAIL_W = 2       # px
TRAIL_WHITE = 0.7  # oldest tail segment = this much white blended into the node colour
SHADOW = 0.35     # background dim factor
FPS = 29.98698


def rope_gradient_t(kp0, edges, start, end):
    """Same gradient as viser_rope.py (kept local: this script runs in the hamer
    env where viser is not installed): 0 at the right grasp, 1 at the left
    grasp, arc-length along the main path; danglers inherit their branch t."""
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


def main():
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_TRACKING)
    tz = np.load(npz_path)
    kp = tz['full']  # mm, camera frame (projection is unit-free)
    edges = tz['edge_connection'].astype(int)
    T, K_n, _ = kp.shape

    d = np.load(UNDIST_NPZ)
    color, K = d['color'], d['K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = color.shape[1], color.shape[2]
    masks = np.load(ROPE_NPZ)['masks']
    h = np.load(HANDS_NPZ)
    verts, valid, ee = h['verts_cam'], h['valid'], h['ee']

    # pink -> blue along the rope, right hand (slot 1) -> left hand
    d_r = [np.linalg.norm(kp[0, k] - ee[0, 1] * 1000.0) for k in (2, 3)]
    start = 2 if d_r[0] < d_r[1] else 3
    end = 5 - start
    t = rope_gradient_t(kp[0], edges, start, end)
    kp_rgb = (1 - t[:, None]) * PINK + t[:, None] * BLUE
    KP_BGR = kp_rgb[:, ::-1].astype(int)            # cv2 colours
    EDGE_BGR = (0.5 * (kp_rgb[edges[:, 0]] + kp_rgb[edges[:, 1]]))[:, ::-1].astype(int)
    anchors = {0, 1, 2, 3}  # B0, B1 + the two EE leaves

    uv = np.stack([fx * kp[..., 0] / kp[..., 2] + cx,
                   fy * kp[..., 1] / kp[..., 2] + cy], -1)  # (T,K,2) float

    cfg = get_config(str(Path(DEFAULT_CHECKPOINT).parent.parent / 'model_config.yaml'),
                     update_cachedir=True)
    renderer = Renderer(cfg, faces=h['faces'])

    out_path = npz_path.parent / 'tracking_deform_with_hands.mp4'
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for i in range(T):
        # background as shadow, rope-mask foreground untouched
        img = color[i].astype(np.float32) * SHADOW
        fg = masks[i] > 0
        img[fg] = color[i][fg]

        # MANO hands on top (rendered in RGB [0,1], composited, back to BGR)
        rgb = img[:, :, ::-1] / 255.0
        for slot in range(2):
            if not valid[i, slot]:
                continue
            rgba = renderer.render_rgba(
                verts[i, slot], cam_t=np.zeros(3), mesh_base_color=HAND_COLOR[slot],
                render_res=[W, H], focal_length=fx, is_right=slot)
            a = rgba[:, :, 3:4]
            rgb = rgb * (1 - a) + rgba[:, :, :3] * a
        img = np.ascontiguousarray((rgb[:, :, ::-1] * 255).astype(np.uint8))

        # 15-frame trajectory tails, fading toward white with age
        j0 = max(0, i - TRAIL)
        for k in range(K_n):
            for j in range(j0, i):
                p1, p2 = uv[j, k], uv[j + 1, k]
                if np.isfinite(p1).all() and np.isfinite(p2).all():
                    w = TRAIL_WHITE * (i - 1 - j) / max(TRAIL, 1)  # newest -> oldest
                    col_rgb = (1 - w) * kp_rgb[k] + w * 255.0
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                             tuple(int(c) for c in col_rgb[::-1]), TRAIL_W, cv2.LINE_AA)

        # skeleton edges (gradient: mean of endpoint colours)
        for e_idx, (a_, b_) in enumerate(edges):
            p1, p2 = uv[i, a_], uv[i, b_]
            if np.isfinite(p1).all() and np.isfinite(p2).all():
                cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)),
                         tuple(int(c) for c in EDGE_BGR[e_idx]), EDGE_W, cv2.LINE_AA)

        # keypoints (anchors +2 px)
        for k in range(K_n):
            p = uv[i, k]
            if np.isfinite(p).all():
                r = ANCHOR_R if k in anchors else KP_R
                cv2.circle(img, tuple(p.astype(int)), r,
                           tuple(int(c) for c in KP_BGR[k]), -1, cv2.LINE_AA)

        vw.write(img)
        if i % 100 == 0:
            print(f'  {i}/{T}', flush=True)
    vw.release()
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
