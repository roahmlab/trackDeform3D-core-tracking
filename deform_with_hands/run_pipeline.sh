#!/bin/bash
# deform_with_hands: full pipeline from the raw capture to tracked keypoints.
#   input : ../input_data/deform_with_hands/{rgbd.npz, calibration.json}
#   output: ./output/{rgbd_undist.npz, hands.npz, rope_masks.npz, tracking/}
# Everything lives inside this repo (HaMeR = ./thirdparty/hamer).
# Two conda envs: `hamer` for stage 1, `trackdeform3d` for stages 2-4.
set -e
cd "$(dirname "$0")"
H=/home/yehengz/miniconda3/envs/hamer/bin/python
T=/home/yehengz/miniconda3/envs/trackdeform3d/bin/python

$H s1_run_hamer.py          # undistort (cached) + HaMeR -> aligned hands + EE
$T s2_postprocess_hands.py  # inpaint missing hands + smooth the EE trajectory
$T s3_get_mask.py      # {rope+hands+arms} DBSCAN -> remove hands -> remove arms
$T s4_tracking.py           # EE-anchored BDLO tracking -> output/tracking/
$H render_tracking.py       # -> output/tracking/clip_0/tracking_deform_with_hands.mp4

echo "Pipeline done. Optional visualisations:"
echo "  $T viser_rope.py           # interactive 3D viewer (port 8081)"
