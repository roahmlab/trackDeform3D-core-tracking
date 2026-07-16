"""DBSCAN depth clustering of {branched rope + hands} vs {body + background}.

Per frame:
  1. 1D k-means (k=3) on depth locates the BODY cluster centre; everything nearer than
     (body_centre - 0.10 m) is candidate foreground.  This gate is deliberately generous
     -- the deepest rope point measured over the clip is 0.14 m in front of the body
     centre -- so the far end of the rope is never clipped.  A gate is required because
     DBSCAN cannot separate touching surfaces: hand -> forearm -> torso is one connected
     surface in the cloud.
  2. DBSCAN (eps = 3 cm, 3D) clusters the candidate points; clusters within 5 cm of
     either goal-1 end-effector are kept.  The rope touches the hands, so rope + both
     hands come out as the seeded clusters; chair/cart flying-pixel junk does not.

The mask is rope + hands + whatever forearm falls inside the gate ("noise on hand").

Outputs under output/hand_rope_masks/:
  masks.npz               key 'masks', (T,720,1280) uint8 {0,1} + frames + gate_z
  mask_png/XXXX.png       per-frame binary (0/255)
  overlay_png/XXXX.png    per-frame RGB overlay (mask tinted red, EEs circled)
  mask.mp4, overlay.mp4
"""
import os

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

UNDIST_NPZ = '/home/yehengz/hamer/deform_with_hands/data/rgbd_undist.npz'
HANDS_NPZ = '/home/yehengz/hamer/deform_with_hands/output/hands.npz'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'hand_rope_masks')

Z_MIN, Z_MAX = 0.35, 2.5   # m; scene of interest
GATE_MARGIN = 0.10         # m; gate = body cluster centre - margin
DB_EPS = 0.03              # m; DBSCAN neighbourhood
DB_MIN = 8                 # DBSCAN min_samples
SEED_DIST = 0.05           # m; cluster kept if within this 3D distance of an EE
FPS = 29.98698


def kmeans_1d(z, k=3, iters=20):
    """Deterministic 1D Lloyd's k-means (percentile init).  Returns sorted centres."""
    c = np.percentile(z, np.linspace(15, 85, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(z[:, None] - c[None, :]), axis=1)
        for j in range(k):
            if (lab == j).any():
                c[j] = z[lab == j].mean()
    return np.sort(c)


def main():
    d = np.load(UNDIST_NPZ)
    color, depth, K, frames = d['color'], d['depth'], d['K'], d['frames']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ee = np.load(HANDS_NPZ)['ee']  # (T,2,3) m, camera frame
    T, H, W = depth.shape

    stride = np.zeros((H, W), bool)
    stride[::2, ::2] = True  # DBSCAN on a 2x2-strided subset, mask recovered by dilation

    for sub in ['mask_png', 'overlay_png']:
        os.makedirs(f'{OUT}/{sub}', exist_ok=True)
    vw_m = cv2.VideoWriter(f'{OUT}/mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H), False)
    vw_o = cv2.VideoWriter(f'{OUT}/overlay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    masks = np.zeros((T, H, W), np.uint8)
    px_count, gate_z = np.zeros(T, int), np.zeros(T)
    for i in range(T):
        z = depth[i].astype(np.float32) / 1000.0
        zv = z[(z > Z_MIN) & (z < Z_MAX)][::4]
        gate = float(kmeans_1d(zv)[1] - GATE_MARGIN)
        gate_z[i] = gate

        sel = (z > Z_MIN) & (z < gate)
        ys, xs = np.nonzero(sel & stride)
        if len(ys) < DB_MIN:
            continue
        zc = z[ys, xs]
        P = np.stack([(xs - cx) * zc / fx, (ys - cy) * zc / fy, zc], 1)
        lab = DBSCAN(eps=DB_EPS, min_samples=DB_MIN).fit(P).labels_

        dmin = {c: min(np.linalg.norm(P[lab == c] - ee[i, 0], axis=1).min(),
                       np.linalg.norm(P[lab == c] - ee[i, 1], axis=1).min())
                for c in np.unique(lab) if c >= 0}
        kept = [c for c, dd in dmin.items() if dd < SEED_DIST]
        if not kept and dmin:  # EE off by more than SEED_DIST (rare): take the nearest cluster
            kept = [min(dmin, key=dmin.get)]

        m = np.zeros((H, W), np.uint8)
        on = np.isin(lab, kept)
        m[ys[on], xs[on]] = 1
        m = cv2.dilate(m, np.ones((3, 3), np.uint8)) & sel.astype(np.uint8)  # fill stride gaps
        masks[i] = m
        px_count[i] = int(m.sum())

        ov = np.ascontiguousarray(color[i])
        ov[m > 0] = (0.35 * ov[m > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        for s in range(2):
            u = int(round(fx * ee[i, s, 0] / ee[i, s, 2] + cx))
            v = int(round(fy * ee[i, s, 1] / ee[i, s, 2] + cy))
            cv2.circle(ov, (u, v), 8, (0, 255, 255), 2)
        cv2.putText(ov, f'frame {int(frames[i])}  gate={gate:.2f}m  mask={px_count[i]}px',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(f'{OUT}/mask_png/{int(frames[i]):04d}.png', m * 255)
        cv2.imwrite(f'{OUT}/overlay_png/{int(frames[i]):04d}.png', ov)
        vw_m.write(m * 255)
        vw_o.write(ov)
        if i % 100 == 0:
            print(f'[{i:3d}/{T}] gate={gate:.2f}m mask={px_count[i]}px', flush=True)

    vw_m.release()
    vw_o.release()
    np.savez_compressed(f'{OUT}/masks.npz', masks=masks, frames=frames, gate_z=gate_z)
    print(f'\nwrote {OUT}/masks.npz + {T} PNGs x2 + mask.mp4 + overlay.mp4')
    print(f'gate depth: median={np.median(gate_z):.2f}m range=[{gate_z.min():.2f}, {gate_z.max():.2f}]')
    print(f'mask px: median={np.median(px_count):.0f} min={px_count.min()} max={px_count.max()}')
    print(f'empty frames: {(px_count == 0).sum()}')


if __name__ == '__main__':
    main()
