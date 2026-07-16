"""Smooth the EE trajectories in hands.npz with the repo's Gaussian smoother.

Reuses smooth_trajectories() from bdlo_tracking.py (gaussian_filter1d per
coordinate along time, sigma=2.0, mode='nearest') on the (T,2,3) EE array.

Updates hands.npz IN PLACE, keeping provenance (idempotent -- always smooths
from ee_raw):
  ee      -> smoothed trajectory (what every downstream consumer uses)
  ee_raw  -> the pre-smoothing (inpainted) trajectory
  ee_smooth_sigma -> the sigma used
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdlo_tracking import smooth_trajectories

HANDS_NPZ = '/home/yehengz/hamer/deform_with_hands/output/hands.npz'
SIGMA = 2.0


def main():
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


if __name__ == '__main__':
    main()
