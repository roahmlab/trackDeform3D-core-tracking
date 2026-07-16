"""Stage 3 -- ROPE MASK of the branched deformable object (env `trackdeform3d`).

Step-by-step mask extraction, two phases in one stage:

  PHASE 1 -- segment_hand_rope(): {rope + hands + arms} from depth only.
     1D k-means locates the body -> generous near-gate (body - 10 cm) ->
     DBSCAN (eps 3 cm) on the 3D points -> keep clusters within 5 cm of either
     EE.  -> output/hand_rope_masks/ (masks + per-frame png + overlay videos).

  PHASE 2 -- extract_rope_mask(): remove hands and arms (user-designed).
     HANDS: z-buffer render (keep rope in front) + EE-corner bbox + finger
     shell.  ARMS: per wrist, points behind the wrist depth, connected to the
     wrist, in the upper half of the mask bbox.  No despeckle.
     -> output/rope_masks/ (masks + diagnostics + rope_mask_overlay_with_ee.mp4).
"""
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from paths import UNDIST_NPZ as _U
from paths import HANDS_NPZ as _H
from scipy.spatial import cKDTree



UNDIST_NPZ = str(_U)
HANDS_NPZ = str(_H)

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


def segment_hand_rope():
    """PHASE 1: {rope+hands+arms} masks, returned IN MEMORY (nothing saved)."""
    d = np.load(UNDIST_NPZ)
    color, depth, K, frames = d['color'], d['depth'], d['K'], d['frames']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ee = np.load(HANDS_NPZ)['ee']  # (T,2,3) m, camera frame
    T, H, W = depth.shape

    stride = np.zeros((H, W), bool)
    stride[::2, ::2] = True  # DBSCAN on a 2x2-strided subset, mask recovered by dilation

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

        if i % 100 == 0:
            print(f'[{i:3d}/{T}] gate={gate:.2f}m mask={px_count[i]}px', flush=True)

    print(f'\nphase 1 done (in memory): gate median={np.median(gate_z):.2f}m, '
          f'mask px median={np.median(px_count):.0f}, empty frames={(px_count == 0).sum()}')
    return masks




from paths import ROPE_MASKS_NPZ as _RM
ROPE_MASKS_NPZ = str(_RM)

KEEP_FRONT = 0.005  # m; observed depth this much nearer than the hand surface = in
                    # front (rope); tight, so grasped rope is not classified as hand
ZBUF_PAD = 2        # px; z-buffer crop padding
FINGER_SHELL = 0.015  # m; points this close to the mesh are finger skin
BBOX_MARGIN = 0.01  # m; lateral/top margin of the EE-cropped hand bbox
WRIST_SEED = 0.05   # m; a behind-wrist component is arm if it comes this close
                    # to the wrist joint (the arm attaches at the wrist)


def hand_zbuffer(V, faces, fx, fy, cx, cy, H, W):
    """Front-surface depth of the hand mesh per pixel (inf where no hand), full frame.

    Painter's algorithm on a padded bbox crop, as in run_hamer.py:visible_verts.
    """
    uv = np.stack([fx * V[:, 0] / V[:, 2] + cx, fy * V[:, 1] / V[:, 2] + cy], 1)
    u0 = max(int(np.floor(uv[:, 0].min())) - ZBUF_PAD, 0)
    v0 = max(int(np.floor(uv[:, 1].min())) - ZBUF_PAD, 0)
    u1 = min(int(np.ceil(uv[:, 0].max())) + ZBUF_PAD + 1, W)
    v1 = min(int(np.ceil(uv[:, 1].max())) + ZBUF_PAD + 1, H)
    zbuf = np.full((H, W), np.inf, np.float32)
    if u1 <= u0 or v1 <= v0:
        return zbuf
    local = np.full((v1 - v0, u1 - u0), np.inf, np.float32)
    polys = np.round(uv[faces] - [u0, v0]).astype(np.int32)
    zf = V[faces, 2].mean(1)
    for f in np.argsort(-zf):  # far faces first
        cv2.fillPoly(local, [polys[f]], float(zf[f]))
    zbuf[v0:v1, u0:u1] = local
    return zbuf


def ee_corner_bbox(V, e):
    """Camera-axis-aligned hand bbox cropped so the EE is one CORNER vertex.

    Per axis, the bound on the EE's side is clamped to the EE coordinate (the
    hand centroid picks the side).  Assumption (user): the EE is the lowest 2D
    point of the hand, so the hand sits above/behind the EE and the cropped box
    never reaches below or in front of the EE where the rope is.
    Returns (lo_full, hi_full, lo, hi).
    """
    lo_full = V.min(0) - BBOX_MARGIN
    hi_full = V.max(0) + BBOX_MARGIN
    lo, hi = lo_full.copy(), hi_full.copy()
    cen = V.mean(0)
    for k in range(3):
        if cen[k] >= e[k]:
            lo[k] = e[k]
        else:
            hi[k] = e[k]
    return lo_full, hi_full, lo, hi


def extract_rope_mask(hand_rope_masks):
    """PHASE 2: hands+arms removed; saves ONLY output/rope_masks.npz."""
    d = np.load(UNDIST_NPZ)
    color, depth, K, frames = d['color'], d['depth'], d['K'], d['frames']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hr = hand_rope_masks
    h = np.load(HANDS_NPZ)
    verts, valid, ee = h['verts_cam'], h['valid'], h['ee']
    wrist = h['joints_cam'][:, :, 0]   # OpenPose hand joint 0 = wrist
    faces = [h['faces_left'], h['faces']]
    T, H, W = hr.shape

    masks = np.zeros((T, H, W), np.uint8)
    miss, px_count = [[], []], np.zeros(T, int)
    for i in range(T):
        z = depth[i].astype(np.float32) / 1000.0
        # 1a. hand mask remove: rendered silhouette, depth-aware
        removal = np.zeros((H, W), np.uint8)
        for s in range(2):
            if not valid[i, s]:
                miss[s].append(int(frames[i]))
                continue
            zb = hand_zbuffer(verts[i, s], faces[s], fx, fy, cx, cy, H, W)
            removal |= (np.isfinite(zb) & (z >= zb - KEEP_FRONT)).astype(np.uint8)
        removal = cv2.dilate(removal, np.ones((3, 3), np.uint8)) & hr[i]
        m = (hr[i] & (1 - removal)).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(ys):
            zc = z[ys, xs]
            P = np.stack([(xs - cx) * zc / fx, (ys - cy) * zc / fy, zc], 1)
            # arms only live in the UPPER HALF of the stage-1 mask bbox (user):
            # the arm cut must never touch lower-half points
            rows_hr = np.nonzero(hr[i].any(axis=1))[0]
            upper = ys < (rows_hr[0] + rows_hr[-1]) / 2.0
            cut = np.zeros(len(P), bool)
            for s in range(2):
                if not valid[i, s]:
                    continue
                # 1b. hand: remove everything inside the EE-corner bbox
                lo_full, hi_full, lo, hi = ee_corner_bbox(verts[i, s], ee[i, s])
                cut |= ((P > lo) & (P < hi)).all(1)
                # 1c. hand: remove points near the fingers (mesh shell)
                cut |= cKDTree(verts[i, s]).query(P, k=1,
                                                  distance_upper_bound=FINGER_SHELL)[0] < FINGER_SHELL
                # 2. arm (user spec): UPPER-HALF points BEHIND this wrist's depth
                #    whose connected component reaches the wrist -- the arm
                #    attaches at the wrist, never at the fingers, so free-hanging
                #    rope that is merely deeper than the wrist is never falsely cut
                behind = (zc > wrist[i, s, 2]) & upper
                if behind.any():
                    B = np.zeros((H, W), np.uint8)
                    B[ys[behind], xs[behind]] = 1
                    n_cc, lab_cc = cv2.connectedComponents(B, 8)
                    lab_pts = lab_cc[ys, xs]
                    for c_ in range(1, n_cc):
                        sel = behind & (lab_pts == c_)
                        if np.linalg.norm(P[sel] - wrist[i, s], axis=1).min() < WRIST_SEED:
                            cut |= sel
            m = np.zeros_like(m)
            m[ys[~cut], xs[~cut]] = 1

        masks[i] = m
        px_count[i] = int(m.sum())

        if i % 100 == 0:
            print(f'[{i:3d}/{T}] rope={px_count[i]}px', flush=True)

    np.savez_compressed(ROPE_MASKS_NPZ, masks=masks, frames=frames,
                        missing_left=np.array(miss[0], int),
                        missing_right=np.array(miss[1], int))
    print(f'\nwrote {ROPE_MASKS_NPZ}')
    print(f'rope px: median={np.median(px_count):.0f} '
          f'min={px_count.min()} max={px_count.max()}')
    print(f'missing LEFT hand  ({len(miss[0])} frames, absolute idx): {miss[0]}')
    print(f'missing RIGHT hand ({len(miss[1])} frames, absolute idx): {miss[1]}')


def main():
    hand_rope_masks = segment_hand_rope()
    extract_rope_mask(hand_rope_masks)


if __name__ == '__main__':
    main()
