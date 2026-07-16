# deform_with_hands — hand-held branched-rope tracking from a raw RGBD capture

End-to-end pipeline: from the raw Azure Kinect capture to tracked BDLO keypoints,
with the two hands detected by HaMeR acting as trusted end-effectors.

```
input : ../input_data/deform_with_hands/{rgbd.npz, calibration.json}
output: ./output/{rgbd_undist.npz, hands.npz, rope_masks.npz, tracking/}   (one artifact per stage, nothing else)
run   : ./run_pipeline.sh        (orchestrates both conda envs)
```

HaMeR is vendored at `./thirdparty/hamer` (env `hamer`); everything else runs in
env `trackdeform3d`. All paths live in **`paths.py`** — change locations there,
nowhere else.

## Stages

Everything is inside this repo — HaMeR lives at `./thirdparty/hamer` (17 GB, git-ignored).

| stage | script | env | what it does → output |
|---|---|---|---|
| 1 | `s1_run_hamer.py` | hamer | **run HaMeR**: undistort + crop t=2–18 s (cached at `output/rgbd_undist.npz`); ViTDet→ViTPose→HaMeR per hand; real-focal metric lift + ray correction; depth alignment (visible-vertex, translation-only); EE = midpoint of thumb tip (4) & middle tip (12) → `output/hands.npz` (metres) |
| 2 | `s2_postprocess_hands.py` | trackdeform3d | **post-process HaMeR**: inpaint missing detections (nearest pose onto interpolated EE; `hands_noinpaint.npz` kept), then smooth the EE (Gaussian σ=2, `ee_raw` kept). Canonical order is a fresh s1→s2 |
| 3 | `s3_get_mask.py` | trackdeform3d | **rope mask, step by step**: phase 1 = {rope+hands+arms} (body-depth gate → DBSCAN eps 3 cm → EE-seeded), kept IN MEMORY; phase 2 = remove hands (z-buffer + EE-corner bbox + finger shell) + arms (behind-wrist depth, wrist-connected, upper-half) → `output/rope_masks.npz` (the stage's only artifact) |
| 4 | `s4_tracking.py` | trackdeform3d | `DeformWithHandsTracker` (EE hard-replace first; branch re-identification + length-prior recal; free leaves = lowest nearby tip; projection+edge optimization; final projection) → `output/tracking/clip_0/` = {3d_keypoints.npz, smoothed_3d_keypoints.npz, summary.txt} + tracking_deform_with_hands.mp4 (rendered by render_tracking.py as the pipeline's final step) |

Stage 4 reuses `bdlo_tracking.process_clip` for reporting — do not move
`process_clip` out of `bdlo_tracking.py` (s4 monkeypatches
`bdlo_tracking.WireTracker`).

## Visualizations (optional, run any time after the stage that feeds them)

| script | env | shows |
|---|---|---|
| `render_tracking.py` | hamer | publication-style render: shadowed bg, MANO hands, gradient keypoints (after s4) |
| `viser_rope.py` | trackdeform3d | interactive 3D: fg/bg clouds + hands + tracked keypoints (port 8081) — the one viewer for everything |

## Conventions

- `hands.npz` is in **metres**, camera frame; the tracker (s4) works in **mm** (×1000 at the boundary).
- Left hand = slot 0 = blue (appears image-RIGHT; video not mirrored); right = slot 1 = pink.
- Depth is registered to color; everything is lifted with the color intrinsics `K`.
- Masks are `(T,720,1280) uint8 {0,1}`; frame indices in npz files are absolute (0–899) of the 30 s video.
- Known sensor limit: the thin rope in front of the bright whiteboard has invalid/fused depth on
  some frames — those stretches are absent from the masks and bridged by the tracker's edges.
