"""Animated viser viz for the tracked BDLO (deform_with_hands).

Auto-plays through the clip:
  - point cloud split into TWO toggleable scene nodes: /fg (rope-mask points,
    full resolution) and /bg (rest of the scene, voxel-downsampled) -- hide or
    show either from the viser scene tree
  - MANO hand meshes (blue=left, pink=right)
  - TRACKED KEYPOINTS from output/tracking/clip_0/smoothed_3d_keypoints.npz,
    coloured pink -> blue ALONG THE ROPE from the right hand to the left hand
    (pink at the right-hand grasp, blue at the left-hand grasp; danglers inherit
    their branch colour), connected by their skeleton edges, each keypoint
    trailing its past 15-frame trajectory (dimmed polyline)

  python viser_rope.py     # serves http://localhost:8081
"""
import os
import threading
import time
from collections import defaultdict, deque

import numpy as np
import viser

DIR = os.path.dirname(os.path.abspath(__file__))
from paths import UNDIST_NPZ as _U
UNDIST_NPZ = str(_U)
from paths import HANDS_NPZ as _H
HANDS_NPZ = str(_H)
from paths import ROPE_MASKS_NPZ as _RM
ROPE_NPZ = str(_RM)
TRACKING_NPZ = f'{DIR}/output/tracking/clip_0/smoothed_3d_keypoints.npz'

MESH_COLOR = [(70, 130, 235), (235, 120, 190)]  # RGB: blue=left, pink=right
PINK = np.array([235.0, 120.0, 190.0])
BLUE = np.array([70.0, 130.0, 235.0])
TRAIL = 15           # keypoint trajectory tail (frames)
VOXEL = 0.03         # m; background point-cloud voxel size
Z_MAX = 2.6          # m; drop the far background
HIDDEN_SEG = np.zeros((1, 2, 3), np.float32)  # degenerate segment = invisible


def voxel_downsample(pts, rgb, v):
    key = np.floor(pts / v).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx], rgb[idx]


def rope_gradient_t(kp0, edges, start, end):
    """Gradient parameter per keypoint: 0 at `start` (right grasp), 1 at `end`
    (left grasp), arc-length along the main right->left path; dangler keypoints
    inherit the t of the main-path node they attach to (graph-nearest)."""
    adj = defaultdict(list)
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    # BFS path start -> end
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
    path = path[::-1]  # start ... end
    # arc-length t along the path
    seglen = [np.linalg.norm(kp0[path[k + 1]] - kp0[path[k]]) for k in range(len(path) - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    t = np.zeros(len(kp0))
    on_path = {}
    for k, node in enumerate(path):
        t[node] = cum[k] / max(cum[-1], 1e-9)
        on_path[node] = t[node]
    # danglers: BFS from the path outward, inherit the attachment t
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
    d = np.load(UNDIST_NPZ)
    color, depth, K = d['color'], d['depth'], d['K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h = np.load(HANDS_NPZ)
    verts, valid, ee = h['verts_cam'], h['valid'], h['ee']
    FACES = [h['faces_left'], h['faces']]
    masks = np.load(ROPE_NPZ)['masks']
    tz = np.load(TRACKING_NPZ)
    kp = tz['full'] / 1000.0  # tracker works in mm -> metres
    edges = tz['edge_connection'].astype(int)
    T, K_n, _ = kp.shape

    # pink -> blue along the rope, right hand -> left hand (slot 1 = right)
    d_r = [np.linalg.norm(kp[0, k] - ee[0, 1]) for k in (2, 3)]
    start = 2 if d_r[0] < d_r[1] else 3   # right-grasp leaf
    end = 5 - start                        # left-grasp leaf
    t = rope_gradient_t(kp[0], edges, start, end)
    KP_COLOR = ((1 - t[:, None]) * PINK + t[:, None] * BLUE).astype(int)
    EDGE_COLOR = KP_COLOR[edges]  # (E, 2, 3): per-endpoint colours -> gradient edges

    print('preparing per-frame fg/bg point clouds...')
    uu, vv = np.meshgrid(np.arange(depth.shape[2], dtype=np.float32),
                         np.arange(depth.shape[1], dtype=np.float32))
    fg_clouds, bg_clouds = [], []
    for i in range(T):
        z = depth[i].astype(np.float32) / 1000.0
        fg = (masks[i] > 0) & (z > 0)
        bg = (~(masks[i] > 0)) & (z > 0) & (z < Z_MAX)
        pf = np.stack([(uu[fg] - cx) * z[fg] / fx, (vv[fg] - cy) * z[fg] / fy, z[fg]], 1)
        fg_clouds.append((pf, color[i][fg][:, ::-1]))
        pb = np.stack([(uu[bg] - cx) * z[bg] / fx, (vv[bg] - cy) * z[bg] / fy, z[bg]], 1)
        bg_clouds.append(voxel_downsample(pb, color[i][bg][:, ::-1], VOXEL))
    print(f'  fg ~{np.mean([len(c[0]) for c in fg_clouds]):.0f} pts/frame, '
          f'bg ~{np.mean([len(c[0]) for c in bg_clouds]):.0f} pts/frame')

    server = viser.ViserServer(port=8081)
    server.scene.set_up_direction('-y')  # OpenCV camera: +y is down
    # parent nodes created once: toggle these in the scene tree to show/hide
    server.scene.add_frame('/fg', show_axes=False)
    server.scene.add_frame('/bg', show_axes=False)

    play = server.gui.add_checkbox('play', True)
    fps = server.gui.add_slider('fps', 1, 60, 1, 20)
    sld = server.gui.add_slider('frame', 0, T - 1, 1, 0)
    state = {'i': 0}

    def render(i):
        pf, cf = fg_clouds[i]
        server.scene.add_point_cloud('/fg/cloud', pf, cf, point_size=0.004,
                                     point_shape='circle')
        pb, cb = bg_clouds[i]
        server.scene.add_point_cloud('/bg/cloud', pb, cb, point_size=0.006,
                                     point_shape='circle')
        for slot in range(2):
            tag = ['L', 'R'][slot]
            if valid[i, slot]:
                server.scene.add_mesh_simple(f'/hand_{tag}', verts[i, slot], FACES[slot],
                                             color=MESH_COLOR[slot], flat_shading=False)
            else:
                server.scene.add_icosphere(f'/hand_{tag}', radius=0.0, visible=False)
        # keypoints connected by their skeleton edges (gradient-coloured)
        server.scene.add_line_segments('/edges', kp[i][edges], colors=EDGE_COLOR,
                                       line_width=4.0)
        for k in range(K_n):
            server.scene.add_icosphere(f'/kp/{k:02d}', radius=0.011,
                                       position=kp[i, k], color=tuple(KP_COLOR[k]))
            # past-15-frame trajectory tail, dimmed keypoint colour
            j0 = max(0, i - TRAIL)
            if i - j0 >= 1:
                traj = kp[j0:i + 1, k]
                segs = np.stack([traj[:-1], traj[1:]], axis=1)
                n = len(segs)
                w = 0.7 * (np.arange(n)[::-1] / max(TRAIL, 1))[:, None]  # oldest -> 70% white
                cols = ((1 - w) * KP_COLOR[k] + w * 255.0).astype(int)
            else:
                segs = HIDDEN_SEG
                cols = np.array([[255, 255, 255]])
            server.scene.add_line_segments(f'/trail/{k:02d}', segs,
                                           colors=cols[:, None, :].repeat(2, 1),
                                           line_width=3.0)

    @sld.on_update
    def _(_):
        if not play.value:
            state['i'] = sld.value
            render(sld.value)

    def loop():
        while True:
            if play.value:
                state['i'] = (state['i'] + 1) % T
                sld.value = state['i']
                render(state['i'])
            time.sleep(1.0 / max(fps.value, 1))

    render(0)
    threading.Thread(target=loop, daemon=True).start()
    print(f'serving {T} frames at http://localhost:8081  (Ctrl-C to stop)')
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
