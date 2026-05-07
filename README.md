# TrackDeform3D — Core Tracking

The core tracking algorithm of **TrackDeform3D**.

> Given the foreground point cloud of a deformable object (DLO, branched DLO,
> rectangular fabric, or T-shirt/cloth), recover its **keypoints** and the
> **topology connection** (graph edges) on the first frame, then **track them
> consistently** across the recording.

Four trackers share the same loader / metric / smoothing / video-rendering
scaffolding and differ only in the per-class topology and the per-frame
optimization rules:

| Script                                    | Object               | Default topology              |
| ----------------------------------------- | -------------------- | ----------------------------- |
| [`dlo_tracking.py`](dlo_tracking.py)      | open chain (DLO)     | 1×N chain, 2 leaves           |
| [`bdlo_tracking.py`](bdlo_tracking.py)    | branched DLO (BDLO)  | Tree-shape, 2 branch + 4 leaves  |
| [`fabric_tracking.py`](fabric_tracking.py) | rectangular fabric  | N×N grid, 4 corners           |
| [`cloth_tracking.py`](cloth_tracking.py)  | T-shirt              | grid + 8-corner contour       |

Each script:

1. Tracks every frame with the full pipeline (snap → geometry → EE-anchor).
2. Computes evaluation metrics (edge length, position RMSE, Chamfer / F-scores)
   on the **raw** keypoints.
3. Post-processes the 3D keypoint sequence with a Gaussian filter
   (`--sigma`, default 3.0) and **renders the tracking video using the smoothed
   trajectory** — metrics stay on raw, video uses smooth.
4. Saves both `3d_keypoints.npz` (raw) and `smoothed_3d_keypoints.npz` (smoothed)
   per clip, plus chunk-summary aggregates.

EE positions are used purely as a **labelling/disambiguation signal** (which
leaf or corner each robot arm grasps) — they are **never** written into the
keypoints as an optimization target. The actual anchor positions always come
from the camera observation.

**Why:** the current hand-eye calibration is not accurate enough — the residual
error is dominated by **camera–robot latency**, which makes the EE FK position
disagree with where the gripper actually is in the depth/RGB frame at the same
timestamp. Trusting EE FK as a hard target would drag keypoints away from the
true observation. The latency itself has since been fixed at the capture stack,
but that fix has **not yet been integrated with this tracking pipeline**; once
it is, EE FK can be re-enabled as an optimization anchor (e.g. by uncommenting
`_replace_with_ee_poses` in `wire_tracker.py` and adding the equivalent step
in cloth/fabric). See the per-tracker docstrings for details.

---

## 1. Environment setup (conda + pip)

Tested with Python 3.11 on Linux. The four trackers depend on:
`numpy`, `opencv-python`, `scipy`, `scikit-learn`, `scikit-image`, `matplotlib`,
`plotly`, `tqdm`.

```bash
conda create -n trackdeform3d python=3.11 -y
conda activate trackdeform3d
pip install -r requirements.txt
```

If you don't want to keep a separate `requirements.txt`, this single line
installs everything the trackers need:

```bash
pip install numpy opencv-python scipy scikit-learn scikit-image matplotlib plotly tqdm
```

---

## 2. Dataset format

Sample data for all four trackers can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1emeePJieKG0wwYZt4PY1E7GW2_mIJVrk?usp=sharing).
Extract the contents so that the `input_data/` folder sits next to the tracker scripts.

Each tracker reads from `input_data/<tracker>/chunk_<N>/`. Calibration
(camera intrinsics + the two `T_base→cam` transforms) lives next to it under
`input_data/<tracker>/calibration/`. 

```
input_data/
├── dlo/
│   ├── calibration/
│   │   └── transform_ee_cam_world.npz       # T_left_base2cam, T_right_base2cam, K
│   └── chunk_<N>/
│       ├── rgbd.npz                          # color (T,H,W,3) uint8 BGR; depth (T,H,W) uint16 mm
│       ├── masks/masks.npz                   # key 'masks' — DLO foreground mask
│       ├── left_arm_poses.npz                # arr_0..arr_{T-1}, each [x,y,z,qw,qx,qy,qz] in left-base frame (m)
│       └── right_arm_poses.npz               # same, right arm
│
├── bdlo/
│   ├── calibration/transform_ee_cam_world.npz
│   └── chunk_<N>/
│       ├── rgbd.npz
│       ├── masks/masks.npz                   # key 'masks' — BDLO foreground mask
│       ├── left_arm_poses.npz
│       └── right_arm_poses.npz
│
├── fabric/
│   ├── calibration/transform_ee_cam_world.npz
│   └── chunk_<N>/
│       ├── rgbd.npz
│       ├── fg_mask.npz                       # key 'fg_mask' — rectangular cloth foreground
│       ├── left_arm_poses.npz
│       └── right_arm_poses.npz
│
└── cloth/
    ├── calibration/transform_ee_cam_world.npz
    └── chunk_<N>/
        ├── rgbd.npz
        ├── fg_masks/masks.npz                # key 'masks' — T-shirt foreground (note 'fg_masks/' subfolder)
        ├── left_arm_poses.npz
        └── right_arm_poses.npz
```

**Notes**

- `transform_ee_cam_world.npz` has three arrays: `T_left_base2cam` (4×4),
  `T_right_base2cam` (4×4), `K` (3×3 intrinsics). Cloth and fabric share one
  rig calibration; DLO and BDLO each have their own (different days).
- `rgbd.npz` keys are `color` and `depth`. Color is BGR uint8; depth is
  uint16 millimetres. Loaders take all frames except for `bdlo_tracking.py`,
  which keeps only the **last 600** frames per chunk.
- Mask file location and key vary by tracker (third column above). All four
  are positive (`==1`) on the deformable object foreground and zero elsewhere.
- Pose `.npz`s store one frame per array (`arr_0`, `arr_1`, …), each is a 7-vec
  `[x, y, z, qw, qx, qy, qz]` in the corresponding robot base frame, with
  position in **meters** (loaders multiply by 1000 to get camera-frame mm).
- The example data already provided is **one chunk per tracker**:
  `dlo/chunk_1`, `bdlo/chunk_7`, `fabric/chunk_14`, `cloth/chunk_0`. To add
  more chunks just drop them into the appropriate `chunk_<N>/` folder.

Outputs land in `output/<tracker>/chunk_<N>/clip_<i>/` next to the script.

---

## 3. Run tracking

The four example commands are bundled in [`run_all.sh`](run_all.sh):

```
bash run_all.sh
```

Or run them individually (these are the recipes for the example chunks shipped
in `input_data/`):

```bash
# Open-chain DLO (blue cable, chunk 1)
python dlo_tracking.py --chunk 1 --clip_seconds 10 --n_keypoints 15

# Branched DLO (Y-shape, chunk 7) — keypoints_per_segment order is fixed:
#   [ee0, ee1, free0, free1, trunk]
# Total n_keypoints must equal 2 (branches) + 4 (leaves) + sum(segments).
python bdlo_tracking.py --chunk 7 --clip_seconds 10 --n_keypoints 25 \
    --keypoints_per_segment 4 4 3 3 5

# Rectangular fabric (chunk 14) — simplest, no extra topology hints
python fabric_tracking.py --chunk 14 --clip_seconds 10

# T-shirt cloth (chunk 0) — segment_interior_nodes is REQUIRED and the
# 8 counts must follow the corner traversal C0→C1→…→C7→C0
python cloth_tracking.py --chunk 0 --clip_seconds 10 \
    --segment_interior_nodes "1,1,5,3,5,1,1,7"
```

Common optional flags shared by all four:

| Flag             | Default | Meaning                                                                 |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `--chunk`        | required | Chunk index under `input_data/<tracker>/chunk_<N>/`                    |
| `--clip_seconds` | 10 | Seconds per clip; trackers are reinitialized at each clip boundary |
| `--fps`          | 30      | Source frame rate                                                       |
| `--sigma`        | 3.0     | Gaussian-σ used to smooth the 3D trajectory for the **video only** (metrics stay raw). Pass `--sigma 0` to disable. |
