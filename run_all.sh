#!/bin/bash
# Run all four trackers on the example chunks shipped in input_data/.
# Activate the conda env first:   conda activate trackdeform3d
set -e

# DLO — open chain (blue cable, chunk 1)
python dlo_tracking.py --chunk 1 --clip_seconds 10 --n_keypoints 15

# BDLO — Y-shape branched DLO (chunk 7)
# Note: --keypoints_per_segment order MUST be: [ee0, ee1, free0, free1, trunk]
# n_keypoints = 2 (branches) + 4 (leaves) + sum(keypoints_per_segment) = 25 here
python bdlo_tracking.py --chunk 7 --clip_seconds 10 --n_keypoints 25 \
    --keypoints_per_segment 4 4 3 3 5

# Fabric — rectangular cloth (chunk 14)
python fabric_tracking.py --chunk 14 --clip_seconds 10

# Cloth — T-shirt (chunk 0)
# Note: --segment_interior_nodes is REQUIRED; 8 counts in corner-traversal order C0→C1→…→C7→C0
python cloth_tracking.py --chunk 0 --clip_seconds 10 \
    --segment_interior_nodes "1,1,5,3,5,1,1,7"
