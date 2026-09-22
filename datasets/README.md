# Deformable-Object Tracking Dataset (`datasets`)

3-D keypoint-graph tracking results for six deformable objects. Each clip is a window of **up to**
150 frames (5 s @ 30 fps) and contains `3d_keypoints.npz` (raw), `smoothed_3d_keypoints.npz`, and
`summary.txt` (metrics). Keypoint arrays are `full` of shape `(frames, N_keypoints, 3)` in
camera-frame millimetres, with a fixed edge list and per-edge `reference_lengths`. The smoothed files
also carry **`full_upright`**, the same trajectory in a gravity-aligned frame — see
[Coordinate frames](#coordinate-frames).

> **Use `smoothed_3d_keypoints.npz` unless you specifically want the unfiltered signal.** The raw
> `3d_keypoints.npz` is noticeably jittery: on `rope/chunk_0/clip_0` the mean frame-to-frame
> acceleration is 1.278 mm/f² raw versus 0.061 smoothed, a 21x reduction.

> **Not every clip is 150 frames.** `branched_rope`, `fabric` and `t-shirt` are uniformly 150, but
> `rope` ranges 131–150, `wire` 100–150, and `branched_wire` 26–150. Always read the frame count from
> `full.shape[0]` rather than assuming 150.

> **Videos are not in this repository.** Per-clip tracking `.mp4`s, init `.html` visualizations and
> `.png`s add ~2.5 GB and are kept alongside the source captures instead.

---

## 1. Object types & topology

Each object is tracked as a fixed graph of keypoints (nodes) joined by edges. **Leaf nodes** have
degree 1 (free ends), **branch/junction nodes** have degree ≥ 3, everything else is an interior chain
node. Sheets are tracked as a 2-D **grid primitive** (mesh) instead of a chain.

| object | type | keypoints | edges | topology |
|---|---|---|---|---|
| **rope** | open DLO (single rope) | 14 | 13 | linear chain; **2 leaf** ends (nodes 0, 13); 0 branches |
| **wire** | open DLO (single wire) | 15 | 14 | linear chain; **2 leaf** ends (nodes 0, 14); 0 branches |
| **branched_rope** | BDLO (branched rope) | 25 | 24 | **2 branch** + **4 leaf** (see below) |
| **branched_wire** | BDLO (branched "yellow" wire) | 24 | 23 | **2 branch** + **5 leaf** (see below) |
| **fabric** | 2-D sheet | 36 | 60 | **6 × 6 grid** primitive; 4 corners (2 grasped by EEs) |
| **t-shirt** | T-shaped 2-D sheet | 81 | 96 | **9 × 9 grid** primitive, T-shape; 8 contour corners (2 grasped) |

### Branched-object detail

**branched_rope** — an "H"/double-Y tree: two junctions linked by a trunk, each junction sprouting two
arms that end in leaves.
- Branch (junction) nodes: **0** and **1** (both degree 3).
  - node **0** neighbours → `{13, 19, 24}`
  - node **1** neighbours → `{9, 16, 20}`
- Leaf nodes (degree 1): **2, 3, 4, 5** (2 leaves hang off each junction).

**branched_wire** (yellow BDLO) — trunk `ee0 – b0 – b1 – ee1` with extra "free" leaf branches.
- Branch (junction) nodes: **0** (degree 3) and **1** (degree 4).
  - node **0** neighbours → `{7, 15, 19}`  (trunk + one grasped end + 1 free leaf)
  - node **1** neighbours → `{12, 17, 18, 23}`  (trunk + one grasped end + 2 free leaves)
- Leaf nodes (degree 1): **2, 3, 4, 5, 6** → 2 grasped ends (ee0, ee1) + 3 free leaves.

*(Node ordering convention for BDLOs: `[branch_0, branch_1, leaf_0 … leaf_k, interior …]`.)*

---

## 2. Data volume

| object | clips | frames | duration (mm:ss) | frames/clip |
|---|---|---|---|---|
| rope | 227 | 33,872 | 18:49 | 131–150 |
| wire | 142 | 21,107 | 11:43 | 100–150 |
| branched_rope | 189 | 28,350 | 15:45 | 150 |
| branched_wire | 108 | 14,291 | 07:56 | 26–150 |
| fabric | 156 | 23,400 | 13:00 | 150 |
| t-shirt | 208 | 31,200 | 17:20 | 150 |
| **TOTAL** | **1,030** | **152,220** | **84:34** |  |

### File format

Every `3d_keypoints.npz` / `smoothed_3d_keypoints.npz` holds:

| key | shape | notes |
|---|---|---|
| `full` | `(frames, N_keypoints, 3)` | camera-frame millimetres |
| `full_upright` | `(frames, N_keypoints, 3)` | gravity-aligned; **smoothed files only** |
| `edge_connection` *or* `edge_connections` | `(N_edges, 2)` | **name differs by object — see below** |
| `reference_lengths` | `(N_edges,)` | frozen initial edge lengths |
| `full_world` | `(frames, N_keypoints, 3)` | **only** in `rope` and `wire` |

Two inconsistencies to code around:

- **Edge-list key name.** The chain/branched objects (`rope`, `wire`, `branched_rope`,
  `branched_wire`) use the **singular** `edge_connection`; the sheet objects (`fabric`, `t-shirt`)
  use the **plural** `edge_connections`.
- **`full_world`** (world-frame coordinates) is present only for `rope` and `wire`. The other four
  objects provide camera-frame `full` only.
- **`t-shirt` contains NaNs by design.** The garment is a T-shape stored inside a full 9×9 grid, so
  the 24 off-shape cells (the two "armpit" blocks) are `NaN` in every frame of all 208 clips — 57 of
  the 81 nodes are real. This is structural padding, not tracking failure, and the NaN node set is
  identical everywhere:

  ```
  node indices that are always NaN:
  27 28 34 35 36 37 43 44 45 46 52 53 54 55 61 62 63 64 70 71 72 73 79 80

  9x9 occupancy   (X = tracked, . = NaN)
      X X X X X X X X X
      X X X X X X X X X
      X X X X X X X X X
      . . X X X X X . .
      . . X X X X X . .
      . . X X X X X . .
      . . X X X X X . .
      . . X X X X X . .
      . . X X X X X . .
  ```

  Mask them before computing anything: `live = np.isfinite(xyz).all(axis=(0, 2))`. The other five
  objects are NaN-free.

```python
import numpy as np
d = np.load("datasets/rope/chunk_0/clip_0/3d_keypoints.npz")
xyz   = d["full"]                     # (frames, N, 3) in mm
edges = d["edge_connection" if "edge_connection" in d.files else "edge_connections"]
```

---

### Coordinate frames

`full` is in the **camera frame, OpenCV convention**:

| axis | direction | typical range here |
|---|---|---|
| `+x` | right | ±400 mm |
| `+y` | **down** (toward the floor) | ±400 mm |
| `+z` | **forward — depth from the camera** | ~800–1600 mm |

**`+z` is depth, not height.** Plotting `z` on a vertical axis makes the scene look like it is
standing on end; a hanging t-shirt appears to lie flat. Gravity points along **`+y`**.

Every `smoothed_3d_keypoints.npz` ships this already as **`full_upright`**, in `(right, depth, up)`
order — use it directly:

```python
d = np.load("datasets/rope/chunk_0/clip_0/smoothed_3d_keypoints.npz")
xyz = d["full_upright"]      # gravity-aligned: +z is up
```

For the raw `3d_keypoints.npz` files, which do not carry it, reorder yourself:

```python
upright = np.stack([xyz[..., 0], xyz[..., 2], -xyz[..., 1]], axis=-1)
```

This is exact — a pure axis permutation, no calibration needed. It is valid for **all six objects**:
the direction that maps to world up was computed from each rig's `T_cam2right` extrinsics, and every
camera is mounted level to within 1.25°.

| object | world-up in camera coords | tilt off `-y` |
|---|---|---|
| rope | `( 0.0017, -1.0000, -0.0010)` | 0.11° |
| wire | `(-0.0069, -1.0000,  0.0060)` | 0.52° |
| branched_rope | `(-0.0038, -1.0000, -0.0006)` | 0.22° |
| branched_wire | `(-0.0005, -1.0000, -0.0036)` | 0.21° |
| fabric | `( 0.0108, -0.9999, -0.0015)` | 0.62° |
| t-shirt | `(-0.0082, -1.0000,  0.0011)` | 0.47° |

#### World frame (`full_world`, rope and wire only)

`full_world` is the **right robot base frame** (`+z` up). It is reproduced exactly from `full` by the
rig extrinsics, translation in metres scaled to millimetres:

```python
T = np.load("transform_ee_cam_world.npz")["T_cam2right"]
world = full @ T[:3, :3].T + T[:3, 3] * 1000.0     # verified: 0.000 mm residual
```

Calibration is per capture session, not per object — `wire/4sec/chunk_{0,1,2}` use a second rig pose
(`..._poseB_chunks0-2.npz`) and are off by ~31 mm under the default one. The other 52 wire chunks and
all 19 rope chunks match the default calibration to 0.000 mm.

---

## 3. Metrics

Three metrics per object: **edge-length % error** (mean over all edges/frames of `|current −
reference| / reference`), **position RMSE (mm)**, and **Chamfer distance (mm)**.

- **Edge-length %** is **reference-free** — computed purely from the tracked keypoints against their
  own frozen initial edge lengths — so it is reliable for every clip and is reported over **all** clips.
- **Position RMSE** and **Chamfer distance** are scored against a reference point cloud derived from
  the segmentation mask. For some clips that mask is **not clean** — over-inclusive / covering more
  than the tracked primitive (e.g. the reference extends well beyond the fabric grid or a t-shirt
  sleeve) — which inflates pos/CD without reflecting any real tracking error. Those pos/CD values are
  **not representative**, so for these two metrics we **drop the unreasonable clips** (per-object IQR
  upper-fence, `> Q3 + 1.5·IQR`) before reporting. Edge-length is left untouched. The `[−n]` column is
  how many unreasonable evaluation clips were excluded from that metric.

| object | edge % (all) med / mean | pos mm med / mean `[−n]` | CD mm med / mean `[−n]` |
|---|---|---|---|
| rope | 1.43 / 1.43 | 0.73 / 0.73 `[−6]` | 2.67 / 2.71 `[−5]` |
| wire | 1.75 / 1.84 | 1.52 / 1.53 `[−7]` | 3.57 / 3.64 `[−2]` |
| branched_rope | 2.17 / 2.19 | 3.19 / 3.18 `[−0]` | 3.14 / 3.16 `[−8]` |
| branched_wire | 2.45 / 2.44 | 5.65 / 5.68 `[−6]` | 4.26 / 4.33 `[−6]` |
| fabric | 2.81 / 2.80 | 6.46 / 6.64 `[−15]` | 7.06 / 7.18 `[−16]` |
| t-shirt | 2.29 / 2.31 | 4.23 / 4.24 `[−3]` | 5.30 / 5.45 `[−12]` |

Curation filter used per object (pos/cd thresholds here only *gated* curation; they are not the
reported quality numbers above):

| object | curation filter |
|---|---|
| rope | best clips (edge ≲ 1.6%) |
| wire | edge-based |
| branched_rope | edge<3% ∧ pos<5 ∧ cd<6, no node-jump |
| branched_wire | edge<4% ∧ pos<10, no node-jump |
| fabric | edge<4% |
| t-shirt | edge<3% |

---

## 4. Viewing a clip (viser)

Each object has a viser viewer under `viser/`. Run from the repo root
(`trackDeform3D-core-tracking/`) with the tracking env, then open the printed
`http://localhost:<port>` in a browser. All clips here are 5 s, so pass **`--clip-seconds 5`**.

```
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/<obj>_viser.py \
    --root <datasets subdir> --data-root <source captures> \
    --chunk <N> --clip <M> --clip-seconds 5 --port <P>
```

`--root` points at the cleaned clips; `--data-root` at the original captures (only needed for the
RGB-D point-cloud overlay — the tracked mesh renders without it). Match `--data-root` to the same
dataset as `--root`. Each command below is **self-contained** (copy-paste one whole block); the viewer
opens `chunk`/`clip` dropdowns, so you don't need to pass `--chunk/--clip` (add them only to open on a
specific clip). Then open the printed `http://localhost:<port>` in a browser.

```bash
# ROPE  (root has chunk_*/clip_* directly)            port 8081
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/rope_viser.py \
  --root datasets/rope \
  --data-root /media/roahmlab/data/dlo1_first400 \
  --clip-seconds 5 --port 8081

# WIRE  (subfolders: 2sec | 4sec)                     port 8082
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/wire_viser.py \
  --root datasets/wire/4sec \
  --data-root /media/roahmlab/data/captured_data_double_arm/dlo_blue_4sec \
  --clip-seconds 5 --port 8082

# BRANCHED_ROPE  (subfolders: 2sec | 4sec | bdlo_contact)   port 8083
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/branch_rope_viser.py \
  --root datasets/branched_rope/4sec \
  --data-root /media/roahmlab/data/captured_data_double_arm/bdlo_no_contact_4sec \
  --clip-seconds 5 --port 8083

# BRANCHED_WIRE (yellow)  (subfolders: 2sec | 4sec)   port 8086
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/branched_wire_viser.py \
  --root datasets/branched_wire/4sec \
  --data-root /media/roahmlab/data/captured_data_double_arm/bdlo_yellow_4sec \
  --clip-seconds 5 --port 8086

# FABRIC  (subfolders = the 7 cloth datasets)         port 8084
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/fabric_viser.py \
  --root datasets/fabric/cloth_no_occlusion_front_4sec \
  --data-root /media/roahmlab/data/captured_data_double_arm/cloth_no_occlusion_front_4sec \
  --clip-seconds 5 --port 8084

# T-SHIRT  (subfolders: test_0302_tshirt_25 | test_0302_tshirt_4sec | t_shirt_3sec | t_shirt_4sec)   port 8085
cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/trackDeform3D-core-tracking && \
env -u PYTHONPATH -u PYTHONHOME ~/.venvs/trackdeform/bin/python viser/t_shirt_viser.py \
  --root datasets/t-shirt/t_shirt_4sec \
  --data-root /media/roahmlab/data/captured_data_double_arm/t_shirt_4sec \
  --clip-seconds 5 --port 8085
```

**Notes**
- Run the `~/.venvs/trackdeform/bin/python` binary **directly** (as above) — do NOT stuff it into a
  shell variable like `PY="… ~/.venvs/…"`, because `~` does not expand inside a quoted variable and
  the command fails. That was the bug in the earlier version of this section.
- To open on a specific clip, append `--chunk <N> --clip <M>` (e.g. `--chunk 6 --clip 3`); otherwise
  use the in-browser dropdowns.
- For a **2sec** wire/branched dataset, point both `--root …/2sec` and `--data-root …_2sec`
  (e.g. `dlo_blue_2sec`, `bdlo_no_contact_2sec`, `bdlo_yellow_2sec`).
- For any **fabric**/**t-shirt** dataset, set `--root datasets/<obj>/<dataset>` and
  `--data-root /media/roahmlab/data/captured_data_double_arm/<dataset>` (same dataset name).
- `branch_rope_viser` and `branched_wire_viser` both default to port **8083**, so branched_wire uses
  **8086** above; change any `--port` if it's already in use.
