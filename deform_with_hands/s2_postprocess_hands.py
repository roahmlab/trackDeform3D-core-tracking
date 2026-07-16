"""Stage 2 -- POST-PROCESS HAMER (env `trackdeform3d`).

Two in-place updates of output/hands.npz, in canonical order:
  a. INPAINT missing hand detections (user method): nearest valid neighbour's
     pose translated so its EE lands on the linearly interpolated EE; flags
     valid_raw / inpainted / inpaint_src; original kept once as
     hands_noinpaint.npz.  Idempotent from valid_raw.
  b. SMOOTH the EE trajectory: Gaussian sigma=2 per coordinate along time
     (utils.smoothing.smooth_trajectories); ee_raw kept; idempotent from ee_raw.

NOTE: a fresh s1 -> s2 run is the canonical order.  Re-running (a) after (b)
re-anchors the inpainted meshes to the SMOOTHED EE (mm-scale vert shifts at the
inpainted frames) -- harmless, but not byte-stable.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smoothing import smooth_trajectories  # numpy+scipy only

from paths import HANDS_NPZ as _H, OUTPUT_DIR
HANDS_NPZ = str(_H)
OUT = str(OUTPUT_DIR)
SIGMA = 2.0


def inpaint():
    src = f'{OUT}/hands.npz'
    bak = f'{OUT}/hands_noinpaint.npz'
    if not os.path.exists(bak):
        shutil.copy(src, bak)
        print(f'backup -> {bak}')
    d = dict(np.load(src))
    if 'valid_raw' in d:  # idempotent: always inpaint from the raw flags
        d['valid'] = d['valid_raw'].copy()

    valid, ee, frames = d['valid'], d['ee'], d['frames']
    T = len(frames)
    inpainted = np.zeros((T, 2), bool)
    inpaint_src = np.full((T, 2), -1, np.int64)

    for s in range(2):
        good = np.flatnonzero(valid[:, s])
        for i in np.flatnonzero(~valid[:, s]):
            j = good[np.argmin(np.abs(good - i))]  # nearest valid neighbour
            delta = ee[i, s] - ee[j, s]            # interpolated EE - neighbour EE
            d['verts_cam'][i, s] = d['verts_cam'][j, s] + delta
            d['joints_cam'][i, s] = d['joints_cam'][j, s] + delta
            inpainted[i, s] = True
            inpaint_src[i, s] = frames[j]
            print(f'frame {int(frames[i])} {"LR"[s]}: pose from frame {int(frames[j])} '
                  f'(shift {np.linalg.norm(delta) * 100:.1f} cm)')

    d['valid_raw'] = valid.copy()
    d['valid'] = valid | inpainted
    d['inpainted'] = inpainted
    d['inpaint_src'] = inpaint_src
    np.savez(src, **d)
    print(f'\nwrote {src}: valid {int(d["valid"].sum())}/{2 * T} '
          f'({int(inpainted.sum())} inpainted), all-valid = {bool(d["valid"].all())}')


def smooth_ee():
    d = dict(np.load(HANDS_NPZ))
    raw = d.get('ee_raw', d['ee'])
    d['ee_raw'] = raw
    d['ee'] = smooth_trajectories(raw, sigma=SIGMA).astype(raw.dtype)
    d['ee_smooth_sigma'] = np.float64(SIGMA)
    np.savez(HANDS_NPZ, **d)

    disp = np.linalg.norm(d['ee'] - raw, axis=-1) * 1000  # mm
    speed_raw = np.linalg.norm(np.diff(raw, axis=0), axis=-1) * 1000
    speed_smo = np.linalg.norm(np.diff(d['ee'], axis=0), axis=-1) * 1000
    jitter_raw = np.abs(np.diff(speed_raw, axis=0)).mean()
    jitter_smo = np.abs(np.diff(speed_smo, axis=0)).mean()
    print(f'wrote {HANDS_NPZ} (sigma={SIGMA})')
    print(f'displacement: median {np.median(disp):.1f} mm, p95 {np.percentile(disp, 95):.1f} mm, '
          f'max {disp.max():.1f} mm')
    print(f'frame-to-frame accel (jitter proxy): {jitter_raw:.2f} -> {jitter_smo:.2f} mm/frame^2')


def main():
    inpaint()
    smooth_ee()


if __name__ == '__main__':
    main()
