"""2D EE-trajectory video: each frame shows the current EEs projected with K plus a
30-frame fading tail.  blue = left hand, pink = right hand; frame index on top.

Uses the smoothed EE from hands.npz.  Writes output/ee_traj_2d.mp4.
"""
import os

import cv2
import numpy as np

UNDIST_NPZ = '/home/yehengz/hamer/deform_with_hands/data/rgbd_undist.npz'
HANDS_NPZ = '/home/yehengz/hamer/deform_with_hands/output/hands.npz'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
EE_BGR = [(235, 130, 70), (190, 120, 235)]  # left=blue, right=pink
TAIL = 30
FPS = 29.98698


def main():
    d = np.load(UNDIST_NPZ)
    color, K, frames = d['color'], d['K'], d['frames']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ee = np.load(HANDS_NPZ)['ee']  # (T,2,3) m, smoothed + inpainted
    T, H, W = len(ee), color.shape[1], color.shape[2]
    px = np.stack([fx * ee[..., 0] / ee[..., 2] + cx,
                   fy * ee[..., 1] / ee[..., 2] + cy], -1).round().astype(int)  # (T,2,2)

    os.makedirs(OUT, exist_ok=True)
    vw = cv2.VideoWriter(f'{OUT}/ee_traj_2d.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                         FPS, (W, H))
    for i in range(T):
        img = color[i].copy()
        for s in range(2):
            for j in range(max(0, i - TAIL), i):  # tail: fading, older = dimmer
                a = 1.0 - (i - j) / TAIL
                col = tuple(int(c * (0.15 + 0.85 * a)) for c in EE_BGR[s])
                cv2.line(img, tuple(px[j, s]), tuple(px[j + 1, s]), col, 3)
            cv2.circle(img, tuple(px[i, s]), 9, EE_BGR[s], -1)
            cv2.circle(img, tuple(px[i, s]), 9, (255, 255, 255), 1)
        cv2.putText(img, f'frame {int(frames[i])}  blue=left EE  pink=right EE',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        vw.write(img)
        if i % 100 == 0:
            print(f'  {i}/{T}', flush=True)
    vw.release()
    print(f'wrote {OUT}/ee_traj_2d.mp4')


if __name__ == '__main__':
    main()
