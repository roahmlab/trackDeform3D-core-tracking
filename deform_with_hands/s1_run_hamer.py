"""Stage 1 -- RUN HAMER (env `hamer`).

Preprocess + detect in one stage:
  a. undistort the raw capture and crop the t=2-18s window (cached at
     data/rgbd_undist.npz -- delete that file to force a redo);
  b. HaMeR on every frame: ViTDet person -> ViTPose hand boxes -> HaMeR ->
     real-focal metric lift + ray correction -> depth alignment (visible-vertex,
     translation-only) -> EE = midpoint of thumb & middle fingertips.

Outputs data/rgbd_undist.npz + output/hands.npz (metres, OpenCV camera frame).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import HAMER_ROOT
sys.path.insert(0, str(HAMER_ROOT))
os.chdir(str(HAMER_ROOT))  # hamer resolves ./_DATA and ./third-party relative to cwd

import cv2
import numpy as np
import torch

from hamer.configs import get_config
from hamer.models import HAMER, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from vitpose_model import ViTPoseModel

import kinect

from paths import OUTPUT_DIR
OUT = str(OUTPUT_DIR)

THUMB_TIP, MIDDLE_TIP = 4, 12  # OpenPose hand order (= MANO verts 744 / 443)
N_HANDS = 2                    # slot 0 = left (is_right=0), slot 1 = right (is_right=1)

# Coarse-to-fine depth-acceptance band.  HaMeR's raw metric depth is 0.1-0.3m off from the
# Kinect, so a tight band on iteration 0 rejects every correspondence and drops the frame; we
# start loose and tighten as the mesh settles onto the observed surface.
ALIGN_BANDS = [0.30, 0.18, 0.12, 0.10, 0.10]
ALIGN_TRIM = 0.7    # keep the best 70% of correspondences (robust to the grasped rope)
ZBUF_TOL = 0.008    # m; a vertex is visible if nothing is nearer than this at its pixel



def undistort_raw():
    """Old stage 0: undistort colour + registered depth once for the window."""
    K, dist = kinect.load_calib()
    mapx, mapy = kinect.undistort_maps(K, dist)
    frames, t, fps = kinect.frame_window()
    print(f'undistorting: fps={fps:.5f}  frames {frames[0]}..{frames[-1]}  ({len(frames)} frames)')

    color_raw = kinect.memmap_npz(f'{kinect.RAW_DIR}/rgbd.npz', 'color')
    depth_raw = kinect.memmap_npz(f'{kinect.RAW_DIR}/rgbd.npz', 'depth')

    color = np.empty((len(frames), kinect.H, kinect.W, 3), np.uint8)
    depth = np.empty((len(frames), kinect.H, kinect.W), np.uint16)
    for i, f in enumerate(frames):
        color[i] = cv2.remap(np.asarray(color_raw[f]), mapx, mapy, cv2.INTER_LINEAR)
        depth[i] = cv2.remap(np.asarray(depth_raw[f]), mapx, mapy, cv2.INTER_NEAREST)
        if i % 100 == 0:
            print(f'  {i}/{len(frames)}')

    os.makedirs(os.path.dirname(kinect.UNDIST_NPZ), exist_ok=True)
    np.savez(kinect.UNDIST_NPZ, color=color, depth=depth, frames=frames, t=t,
             K=K, dist=dist, fps=fps)
    print(f'wrote {kinect.UNDIST_NPZ}  ({os.path.getsize(kinect.UNDIST_NPZ) / 1e9:.2f} GB)')


def load_model():
    cfg_path = str(Path(DEFAULT_CHECKPOINT).parent.parent / 'model_config.yaml')
    model_cfg = get_config(cfg_path, update_cachedir=True)
    model_cfg.defrost()
    model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
    model_cfg.freeze()
    # init_renderer=False: we never use pyrender (it cannot take a real principal point)
    model = HAMER.load_from_checkpoint(DEFAULT_CHECKPOINT, strict=False, cfg=model_cfg,
                                       init_renderer=False)
    return model, model_cfg


def load_detector():
    from detectron2.config import LazyConfig
    import hamer
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    cfg = LazyConfig.load(str(Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'))
    cfg.train.init_checkpoint = ('https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
                                 'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl')
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    return DefaultPredictor_Lazy(cfg)


def cam_crop_to_full_real(pred_cam, box_center, box_size, K):
    """HaMeR's crop->full camera, with the REAL intrinsics instead of the render-only f=25000.

    Generalises hamer/utils/renderer.py:cam_crop_to_full to fx != fy and a true
    principal point.  Yields metric (metres) camera-frame translation.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    bs = box_size * s + 1e-9
    tz = 2.0 * fx / bs
    return np.stack([tx + (box_center[:, 0] - cx) * tz / fx,
                     ty + (box_center[:, 1] - cy) * tz / fy,
                     tz], axis=-1)


def vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    vn = np.zeros_like(V)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def ray_rotation(box_center, K):
    """Rotation taking the optical axis to the ray through the crop centre.

    HaMeR's cam_crop_to_full is a WEAK-perspective mapping: it ignores that an
    off-centre crop looks down a tilted ray.  With the fake focal (25000 px) that is
    harmless, but with the true focal (606 px) it leaves the mesh mis-oriented by the
    ray angle -- ~20 deg for a hand at u=460.  Rotating the mesh by that ray cuts the
    reprojection error against ViTPose's 2D keypoints from 16.9 px to 10.9 px.
    This is a deterministic camera-model correction, not a fitted rotation.
    """
    d = np.linalg.inv(K) @ np.array([box_center[0], box_center[1], 1.0])
    d /= np.linalg.norm(d)
    v = np.cross([0.0, 0.0, 1.0], d)
    c = d[2]
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def visible_verts(V, faces, K, tol=ZBUF_TOL):
    """Vertices that are the nearest mesh surface at their own pixel (z-buffer test).

    Back-face culling alone is not enough -- a front-facing thumb vertex hidden behind
    the index finger would otherwise pick up the finger's depth as its correspondence.
    """
    uv = kinect.project(V, K)
    u0, v0 = np.floor(uv.min(0)).astype(int) - 2
    u1, v1 = np.ceil(uv.max(0)).astype(int) + 2
    h, w = v1 - v0, u1 - u0
    if h <= 0 or w <= 0 or h > 2000 or w > 2000:
        return np.zeros(len(V), bool)

    uvl = uv - [u0, v0]
    zf = V[faces, 2].mean(1)
    zbuf = np.full((h, w), np.inf, np.float32)
    for f in np.argsort(-zf):  # painter's algorithm: far faces first
        cv2.fillPoly(zbuf, [np.round(uvl[faces[f]]).astype(np.int32)], float(zf[f]))

    px = np.round(uvl).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    vis = np.zeros(len(V), bool)
    vis[ok] = V[ok, 2] <= zbuf[px[ok, 1], px[ok, 0]] + tol
    return vis & (vertex_normals(V, faces)[:, 2] < 0)


def align_to_cloud(V, faces, cloud, dvalid, K, k_fix=None):
    """Align the hand to the depth cloud by SCALE + TRANSLATION only (no rotation).

    Correspondences are exactly the requested ones: take the visible vertices, project
    them to 2D, read the depth at that pixel.  Rotation is deliberately not fitted --
    the orientation comes from HaMeR (+ the ray correction above); letting a projective
    fit touch it just lets it absorb model error (a free rigid ICP rotated the hand 26 deg
    and shifted the reprojection by 20+ px).

    k_fix=None solves scale and translation; otherwise scale is held and only the
    translation is solved.  Per-frame scale is ill-conditioned -- the visible hand
    surface spans only ~6 cm in depth, giving k a +-5-9% uncertainty -- so the caller
    fits k once per hand over the whole window and then holds it here.
    """
    z_img, k_tot, t_tot = cloud[..., 2], 1.0, np.zeros(3)
    rms, n_in = np.nan, 0
    for band in ALIGN_BANDS:
        vis = visible_verts(V, faces, K)
        if vis.sum() < 50:
            break
        Vv = V[vis]
        px = np.round(kinect.project(Vv, K)).astype(int)
        ok = ((px[:, 0] >= 0) & (px[:, 0] < z_img.shape[1]) &
              (px[:, 1] >= 0) & (px[:, 1] < z_img.shape[0]))
        ok &= dvalid[px[:, 1].clip(0, z_img.shape[0] - 1),
                     px[:, 0].clip(0, z_img.shape[1] - 1)] > 0
        Vv, px = Vv[ok], px[ok]
        if len(Vv) < 50:
            break
        Q = cloud[px[:, 1], px[:, 0]]
        d = np.abs(Q[:, 2] - Vv[:, 2])
        keep = d < band
        if keep.sum() < 50:
            break
        # trim the worst residuals: the grasped rope occludes the fingers and would
        # otherwise drag the fit onto the rope's surface
        keep &= d <= np.quantile(d[keep], ALIGN_TRIM)
        n_in = int(keep.sum())

        P, Q = Vv[keep], Q[keep]
        if k_fix is None:
            pc, qc = P.mean(0), Q.mean(0)
            dp = P - pc
            k = float((dp * (Q - qc)).sum() / max((dp * dp).sum(), 1e-12))
            t = qc - k * pc
        else:
            k = k_fix / k_tot                 # reach k_fix in total, given what we applied already
            t = (Q - k * P).mean(0)
        V = k * V + t
        k_tot *= k
        t_tot = k * t_tot + t
        rms = float(np.sqrt(((k * P + t - Q) ** 2).sum(1).mean()))
    return V, k_tot, t_tot, rms, n_in


def main():
    if not os.path.exists(kinect.UNDIST_NPZ):
        undistort_raw()
    else:
        print(f'undistort cache found: {kinect.UNDIST_NPZ}')
    os.makedirs(OUT, exist_ok=True)
    device = torch.device('cuda')
    model, model_cfg = load_model()
    model = model.to(device).eval()
    detector = load_detector()
    cpm = ViTPoseModel(device)

    faces_r = model.mano.faces.astype(np.int64)
    faces_l = faces_r[:, [0, 2, 1]]  # left = right model x-mirrored => winding inverts
    FACES = [faces_l, faces_r]

    color, depth, K, frames, t = kinect.load_undistorted()
    T = int(os.environ.get('LIMIT', 0)) or len(frames)
    frames, t = frames[:T], t[:T]

    z = np.full((T, N_HANDS), np.nan, np.float32)
    out = dict(
        betas=np.full((T, N_HANDS, 10), np.nan, np.float32),
        global_orient=np.full((T, N_HANDS, 1, 3, 3), np.nan, np.float32),
        hand_pose=np.full((T, N_HANDS, 15, 3, 3), np.nan, np.float32),
        pred_cam=np.full((T, N_HANDS, 3), np.nan, np.float32),
        box_center=np.full((T, N_HANDS, 2), np.nan, np.float32),
        box_size=np.full((T, N_HANDS), np.nan, np.float32),
        cam_t=np.full((T, N_HANDS, 3), np.nan, np.float32),
        R_ray=np.full((T, N_HANDS, 3, 3), np.nan, np.float32),
        align_k=z.copy(),
        align_t=np.full((T, N_HANDS, 3), np.nan, np.float32),
        verts_cam=np.full((T, N_HANDS, 778, 3), np.nan, np.float32),
        joints_cam=np.full((T, N_HANDS, 21, 3), np.nan, np.float32),
        ee=np.full((T, N_HANDS, 3), np.nan, np.float32),
        valid=np.zeros((T, N_HANDS), bool),
        align_rms=z.copy(), align_n=np.zeros((T, N_HANDS), np.int32), reproj_px=z.copy(),
    )
    # pre-alignment meshes (HaMeR + ray correction), kept for the second, fixed-scale pass
    V0_all = np.full((T, N_HANDS, 778, 3), np.nan, np.float32)
    J0_all = np.full((T, N_HANDS, 21, 3), np.nan, np.float32)
    k_free = np.full((T, N_HANDS), np.nan, np.float32)

    for i in range(T):
        bgr = np.asarray(color[i])
        cloud = kinect.unproject(depth[i], K)
        dvalid = (depth[i] > 0).astype(np.uint8)

        det = detector(bgr)['instances']
        keep = (det.pred_classes == 0) & (det.scores > 0.5)
        if keep.sum() == 0:
            continue
        boxes_p = det.pred_boxes.tensor[keep].cpu().numpy()
        scores_p = det.scores[keep].cpu().numpy()

        poses = cpm.predict_pose(bgr[:, :, ::-1],  # predict_pose flips internally -> give it RGB
                                 [np.concatenate([boxes_p, scores_p[:, None]], 1)])

        # COCO-WholeBody: kpts 91..111 = LEFT hand, 112..132 = RIGHT hand.  ViTPose emits the
        # two hands as separate keypoint sets, so there is no chirality to distrust.
        bboxes, sides = [], []
        for p in poses:
            for slot, kp in ((0, p['keypoints'][-42:-21]), (1, p['keypoints'][-21:])):
                v = kp[:, 2] > 0.5
                if v.sum() > 3:
                    bboxes.append([kp[v, 0].min(), kp[v, 1].min(), kp[v, 0].max(), kp[v, 1].max()])
                    sides.append(slot)
        if not bboxes:
            continue

        ds = ViTDetDataset(model_cfg, bgr, np.stack(bboxes), np.array(sides), rescale_factor=2.0)
        batch = recursive_to(next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds)))), device)
        with torch.no_grad():
            pred = model(batch)

        right = batch['right'].cpu().numpy()
        pred_cam = pred['pred_cam'].cpu().numpy()
        pred_cam[:, 1] *= (2 * right - 1)  # un-mirror tx BEFORE crop->full (demo.py:136-138)
        bc = batch['box_center'].cpu().numpy()
        bsz = batch['box_size'].cpu().numpy()
        cam_t = cam_crop_to_full_real(pred_cam, bc, bsz, K)

        for n, slot in enumerate(sides):
            sgn = 2 * right[n] - 1
            verts = pred['pred_vertices'][n].cpu().numpy().copy()
            kp3d = pred['pred_keypoints_3d'][n].cpu().numpy().copy()
            verts[:, 0] *= sgn
            kp3d[:, 0] *= sgn  # demo.py forgets this one; without it the EE joints are mirrored

            Rr = ray_rotation(bc[n], K)
            V0 = verts @ Rr.T + cam_t[n]
            J0 = kp3d @ Rr.T + cam_t[n]

            _, k, _, _, n_in = align_to_cloud(V0, FACES[slot], cloud, dvalid, K)
            if n_in == 0:
                continue

            V0_all[i, slot], J0_all[i, slot], k_free[i, slot] = V0, J0, k
            out['betas'][i, slot] = pred['pred_mano_params']['betas'][n].cpu().numpy()
            out['global_orient'][i, slot] = pred['pred_mano_params']['global_orient'][n].cpu().numpy()
            out['hand_pose'][i, slot] = pred['pred_mano_params']['hand_pose'][n].cpu().numpy()
            out['pred_cam'][i, slot] = pred_cam[n]
            out['box_center'][i, slot] = bc[n]
            out['box_size'][i, slot] = bsz[n]
            out['cam_t'][i, slot] = cam_t[n]
            out['R_ray'][i, slot] = Rr
            out['valid'][i, slot] = True

        if i % 50 == 0:
            print(f'[{i:3d}/{T}] valid={out["valid"][i].sum()} '
                  f'k_free={np.nanmean(k_free[i]):.3f}', flush=True)

    # One scale per hand for the whole window: a hand does not change size, and per-frame
    # k has a +-5-9% uncertainty (the visible surface spans only ~6cm in depth).  The median
    # over ~480 frames is ~20x tighter and immune to the occasional collapsed fit.
    # Keep MANO's hand size (k=1): depth-fitted scale improves the fit by <1.5mm but is
    # biased low by Kinect's depth smoothing and yields a sub-anatomical 8cm palm.  The
    # hand is placed by translation only; its size stays anatomical (~9cm palm).
    k_global = np.ones(N_HANDS)
    print(f'\nglobal scale held at k=1 (translation-only placement)')

    for i in range(T):
        cloud = kinect.unproject(depth[i], K)
        dvalid = (depth[i] > 0).astype(np.uint8)
        for slot in range(N_HANDS):
            if not out['valid'][i, slot]:
                continue
            V0, J0 = V0_all[i, slot], J0_all[i, slot]
            uv0 = kinect.project(V0, K)
            V, k, t, rms, n_in = align_to_cloud(V0, FACES[slot], cloud, dvalid, K,
                                                k_fix=float(k_global[slot]))
            if n_in == 0:
                out['valid'][i, slot] = False
                continue
            J = k * J0 + t
            out['align_k'][i, slot] = k
            out['align_t'][i, slot] = t
            out['verts_cam'][i, slot] = V
            out['joints_cam'][i, slot] = J
            out['ee'][i, slot] = 0.5 * (J[THUMB_TIP] + J[MIDDLE_TIP])
            out['align_rms'][i, slot] = rms
            out['align_n'][i, slot] = n_in
            out['reproj_px'][i, slot] = float(np.linalg.norm(kinect.project(V, K) - uv0, axis=1).mean())

    # short EE gaps -> linear interpolation (fingers occluded by the rope now and then)
    for slot in range(N_HANDS):
        v = out['valid'][:, slot]
        gaps = np.where(~v)[0]
        if len(gaps):
            print(f'slot {slot} ({"left" if slot == 0 else "right"}): '
                  f'{len(gaps)} invalid frames -> {gaps.tolist()}')
        if v.sum() >= 2:
            for c in range(3):
                out['ee'][~v, slot, c] = np.interp(np.where(~v)[0], np.where(v)[0],
                                                   out['ee'][v, slot, c])

    np.savez(f'{OUT}/hands.npz', frames=frames, t=t, K=K, units='m', k_global=k_global,
             is_right=np.array([0, 1]), faces=faces_r, faces_left=faces_l, **out)
    print(f'\nwrote {OUT}/hands.npz')
    J = out['joints_cam']
    for slot, name in enumerate(['left', 'right']):
        v = out['valid'][:, slot]
        palm = np.linalg.norm(J[v, slot, 5] - J[v, slot, 0], axis=1).mean()  # wrist->index MCP
        print(f'{name:5s}: valid {v.sum()}/{T}  k {k_global[slot]:.3f}  '
              f'palm {palm * 100:.1f}cm  '
              f'depth_rms {np.nanmedian(out["align_rms"][v, slot]) * 1000:.1f}mm  '
              f'n {int(np.median(out["align_n"][v, slot]))}  '
              f'reproj_shift {np.nanmedian(out["reproj_px"][v, slot]):.1f}px  '
              f'ee_z {np.nanmedian(out["ee"][v, slot, 2]):.3f}m  '
              f'box_u {np.nanmean(out["box_center"][v, slot, 0]):.0f}px')


if __name__ == '__main__':
    main()
