"""Single source of truth for every deform_with_hands pipeline path.

The pipeline starts from the raw capture in the repo's input_data and keeps all
derived data + results inside this folder.  HaMeR is treated as a third-party
dependency living at HAMER_ROOT (run stages s0-s2 and the hand visualisations
with the `hamer` conda env; everything else with `trackdeform3d`).

Import style (stdlib-only, works in both envs):
    from paths import UNDIST_NPZ, HANDS_NPZ, OUTPUT_DIR
"""
from pathlib import Path

DIR = Path(__file__).resolve().parent               # .../deform_with_hands
REPO = DIR.parent                                    # trackDeform3D-core-tracking

# third-party HaMeR checkout (env `hamer`); its scripts chdir here for ./_DATA
HAMER_ROOT = DIR / 'thirdparty' / 'hamer'

# raw capture (input): Azure Kinect rgbd.npz + calibration.json
RAW_DIR = REPO / 'input_data' / 'deform_with_hands'

# results: every stage saves into OUTPUT_DIR, nothing else is kept
OUTPUT_DIR = DIR / 'output'

UNDIST_NPZ = OUTPUT_DIR / 'rgbd_undist.npz'          # s1a (undistort cache)
HANDS_NPZ = OUTPUT_DIR / 'hands.npz'                 # s1 (+s2 post-process)
ROPE_MASKS_NPZ = OUTPUT_DIR / 'rope_masks.npz'       # s3 (phase 1 stays in memory)
TRACKING_DIR = OUTPUT_DIR / 'tracking'               # s4
