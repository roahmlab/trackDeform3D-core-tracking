"""Rope-only mask from the DBSCAN {hand+rope+arm} masks.  Pipeline (user-designed):

  1. HANDS (user spec: mask remove + bounding box remove + remove near fingers):
     a. MASK REMOVE: the MANO mesh is rasterised into a z-buffer (painter's
        algorithm); pixels inside the rendered silhouette whose observed depth is
        at/behind the hand front surface (>= z_hand - KEEP_FRONT) are removed.
        Finite z-buffer only inside the silhouette; rope IN FRONT survives.
     b. BBOX REMOVE: everything inside the EE-corner 3D bounding box -- the
        camera-axis-aligned bbox of the vertices, cropped so THE EE IS ONE CORNER
        VERTEX (per axis, the bound on the EE's side is clamped to the EE; the
        hand centroid picks the side).  Assumption: the EE is the lowest 2D point
        of the hand, so the box sits above/behind the EE, away from the rope.
     c. NEAR FINGERS: points within FINGER_SHELL of the mesh vertices -- the
        finger parts protruding below/lateral of the EE corner, which a and b
        cannot reach.  Grasped rope inside that shell is removed with it.
  2. ARMS (user spec, simple): the rendered/aligned hand gives the WRIST depth.
     Per hand, independently: points BEHIND that wrist depth (z > z_wrist) that
     are spatially CONNECTED TO THE WRIST are the arm -- the arm attaches at the
     wrist, never at the fingers.  Connectivity is what prevents false cuts on
     the wire: rope hanging deeper than a wrist is connected only through the
     grasps, not the wrist, so it survives.  ADDITIONALLY (user constraint): the
     arms only ever occupy the UPPER HALF of the stage-1 mask's 2D bounding box,
     so the arm cut applies to upper-half points ONLY -- lower-half points (the
     rope sag) are never touched.  Hands are inpainted, so both wrists exist on
     every frame.
  NO despeckle / small-component removal (user spec): every stage-1 pixel not cut
  by the hand/arm rules above stays in the rope mask.

A hand with no valid HaMeR detection is skipped (stays in the mask) and reported
per hand in `missing_left` / `missing_right` (absolute 0-899 frame indices).

Outputs under output/rope_masks/:
  masks.npz                       'masks' (T,720,1280) uint8 {0,1}, 'frames',
                                  'missing_left', 'missing_right'
  mask_png/XXXX.png               per-frame binary (0/255)
  overlay_png/XXXX.png            diagnostic overlay: rope red, removed green,
                                  hand bbox full=white / EE-cropped=orange, EEs yellow
  mask.mp4, overlay.mp4
  overlay_ee_png/XXXX.png         CLEAN overlay: final mask red on the original
                                  image + EE dots (left=blue, right=pink)
  rope_mask_overlay_with_ee.mp4
"""
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

HANDROPE_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'output', 'hand_rope_masks', 'masks.npz')
UNDIST_NPZ = '/home/yehengz/hamer/deform_with_hands/data/rgbd_undist.npz'
HANDS_NPZ = '/home/yehengz/hamer/deform_with_hands/output/hands.npz'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'rope_masks')

KEEP_FRONT = 0.005  # m; observed depth this much nearer than the hand surface = in
                    # front (rope); tight, so grasped rope is not classified as hand
ZBUF_PAD = 2        # px; z-buffer crop padding
FINGER_SHELL = 0.015  # m; points this close to the mesh are finger skin
BBOX_MARGIN = 0.01  # m; lateral/top margin of the EE-cropped hand bbox
WRIST_SEED = 0.05   # m; a behind-wrist component is arm if it comes this close
                    # to the wrist joint (the arm attaches at the wrist)
EE_BGR = [(235, 130, 70), (190, 120, 235)]  # left=blue, right=pink (BGR)
FPS = 29.98698


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


def main():
    d = np.load(UNDIST_NPZ)
    color, depth, K, frames = d['color'], d['depth'], d['K'], d['frames']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hr = np.load(HANDROPE_NPZ)['masks']
    h = np.load(HANDS_NPZ)
    verts, valid, ee = h['verts_cam'], h['valid'], h['ee']
    wrist = h['joints_cam'][:, :, 0]   # OpenPose hand joint 0 = wrist
    faces = [h['faces_left'], h['faces']]
    T, H, W = hr.shape

    for sub in ['mask_png', 'overlay_png', 'overlay_ee_png']:
        os.makedirs(f'{OUT}/{sub}', exist_ok=True)
    vw_m = cv2.VideoWriter(f'{OUT}/mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H), False)
    vw_o = cv2.VideoWriter(f'{OUT}/overlay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    vw_e = cv2.VideoWriter(f'{OUT}/rope_mask_overlay_with_ee.mp4',
                           cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

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
        arm = removal.copy()  # everything cut from the hand+rope mask, for the overlay
        bboxes = []
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
                bboxes.append((lo_full, hi_full, lo, hi))
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
            arm[ys[cut], xs[cut]] = 1
            m = np.zeros_like(m)
            m[ys[~cut], xs[~cut]] = 1

        masks[i] = m
        px_count[i] = int(m.sum())

        # diagnostic overlay (copy: color[i] is a view into the loaded array)
        ov = color[i].copy()
        ov[arm > 0] = (0.35 * ov[arm > 0] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        ov[m > 0] = (0.35 * ov[m > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        for lo_full, hi_full, lo, hi in bboxes:  # full bbox white, EE-corner orange
            for lo_, hi_, col, w_ in ((lo_full, hi_full, (255, 255, 255), 1),
                                      (lo, hi, (0, 165, 255), 2)):
                crn = np.array([[x_, y_, z_] for x_ in (lo_[0], hi_[0])
                                for y_ in (lo_[1], hi_[1]) for z_ in (lo_[2], hi_[2])])
                px = np.stack([fx * crn[:, 0] / crn[:, 2] + cx,
                               fy * crn[:, 1] / crn[:, 2] + cy], 1).round().astype(int)
                for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7),
                             (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    cv2.line(ov, tuple(px[a]), tuple(px[b]), col, w_)
        eepx = []
        for s in range(2):
            u = int(round(fx * ee[i, s, 0] / ee[i, s, 2] + cx))
            v = int(round(fy * ee[i, s, 1] / ee[i, s, 2] + cy))
            eepx.append((u, v))
            cv2.circle(ov, (u, v), 8, (0, 255, 255), 2)
        tag = ''.join('LR'[s] for s in range(2) if not valid[i, s])
        label = f'frame {int(frames[i])}  rope={px_count[i]}px  removed={int(arm.sum())}px' + \
                (f'  missing {tag} (not subtracted)' if tag else '')
        cv2.putText(ov, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # clean overlay: final mask + EE dots (left=blue, right=pink)
        ov_ee = color[i].copy()
        ov_ee[m > 0] = (0.35 * ov_ee[m > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        for s in range(2):
            cv2.circle(ov_ee, eepx[s], 9, EE_BGR[s], -1)
            cv2.circle(ov_ee, eepx[s], 9, (255, 255, 255), 1)
        cv2.putText(ov_ee, f'frame {int(frames[i])}  blue=left EE  pink=right EE',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imwrite(f'{OUT}/mask_png/{int(frames[i]):04d}.png', m * 255)
        cv2.imwrite(f'{OUT}/overlay_png/{int(frames[i]):04d}.png', ov)
        cv2.imwrite(f'{OUT}/overlay_ee_png/{int(frames[i]):04d}.png', ov_ee)
        vw_m.write(m * 255)
        vw_o.write(ov)
        vw_e.write(ov_ee)
        if i % 100 == 0:
            print(f'[{i:3d}/{T}] {label}', flush=True)

    vw_m.release()
    vw_o.release()
    vw_e.release()
    np.savez_compressed(f'{OUT}/masks.npz', masks=masks, frames=frames,
                        missing_left=np.array(miss[0], int),
                        missing_right=np.array(miss[1], int))
    print(f'\nwrote {OUT}/masks.npz + {T} PNGs x3 + mask.mp4 + overlay.mp4 + '
          f'rope_mask_overlay_with_ee.mp4')
    print(f'rope px: median={np.median(px_count):.0f} '
          f'min={px_count.min()} max={px_count.max()}')
    print(f'missing LEFT hand  ({len(miss[0])} frames, absolute idx): {miss[0]}')
    print(f'missing RIGHT hand ({len(miss[1])} frames, absolute idx): {miss[1]}')


if __name__ == '__main__':
    main()
