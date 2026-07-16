"""BDLO tracking on the deform_with_hands capture -- DEDICATED pipeline.

DeformWithHandsTracker implements the frame pipeline specified for this capture.
It is built on WireTracker's primitives (skeletonisation, segment-aware
initialisation, the projection+edge optimizer) but does NOT use
WireTracker.process_frame / track():

  1. FIXED INPUTS.  The segmented rope mask (output/rope_masks) and the smoothed,
     interpolated hand EE (hands.npz 'ee', metres -> mm) are used AS-IS.  No
     background threshold, no top-K component filtering, no depth thresholding,
     no dilation -- the mask IS the foreground; the skeleton/centreline and its
     point cloud are derived from it, nothing is removed from it.
  2. INITIALISATION: EE HARD-REPLACE FIRST, THEN OPTIMIZE.  The segment-aware
     initialiser replaces the two EE-mapped leaf endpoints with the frame-0 EE
     positions BEFORE the topology is built, and holds branch+leaf nodes fixed
     while the repulsion optimisation places the intermediate keypoints
     (wire_tracker.py:_initialize_with_segment_allocation steps 5-7).
     Reference edge lengths therefore come from the EE-anchored geometry.
  3. TRACKING, EVERY FRAME (user spec): keypoints = previous frame's keypoints;
     a. EE HARD-REPLACE FIRST: the two EE-mapped leaves are set to this frame's
        hand EEs (never matched to detections -- the EE is trusted).
     b. RE-IDENTIFY the BRANCH nodes (largest component -- junctions are always
        on the main body): Hungarian vs previous positions, accepted matches are
        snapped AND ANCHORED; matches beyond MATCH_GATE (10 cm) rejected.  GATED
        RE-CALIBRATION (user design): if a matched branch jumps more than
        RECAL_GATE (20 mm), the raw candidate set is suspect (spur-junction
        chains) and the frame is re-matched against the PRUNED junctions.
        FREE LEAVES (user design, "search nearby + assign lowest"): leaf tips
        are detected on the FULL skeleton -- all components, connected or not,
        so broken-fragment tips count (skeleton endpoints, 8-neighbour count
        == 1).  For each free leaf, candidates within LEAF_SEARCH of its
        previous position are collected and the LOWEST one (max camera-Y, i.e.
        lowest in the image -- the hanging tip is the lowest point, spur tips
        sit higher up the dangler) becomes the leaf's INITIALIZATION.  The leaf
        is then FREE during optimization (never anchored), refined
        branch -> leaf, and ends on the final projection.
     b2. The free-leaf segments are re-ordered at init so edge corrections
        propagate FROM the branch node TO the free leaf (user spec; the
        initializer emits them leaf-first).
     c. THEN OPTIMIZE with the EE leaves + all snapped nodes as immovable
        anchors (projection skips them, edge corrections give them zero weight).
  4. NO CPD.  The optimisation is _joint_constraint_optimization: alternating
     (a) projection of every non-anchor keypoint onto the skeleton point cloud
     and (b) segment-ordered edge-length (geometry distance) refinement,
     n_outer_iterations x n_edge_iterations, same parameters as the repo config.

     NOTE on enable_node_matching=True: no matching code runs (we never call
     track()); the flag is required only so _joint_constraint_optimization uses
     edge_anchor_set == anchor_set == {EE leaves} -- i.e. branches, free leaves
     and intermediates are ALL optimised, only the EE leaves are frozen.

Reporting (metrics, per-frame CSV, summary, sigma-smoothed visualisation video)
is reused unchanged from bdlo_tracking.process_clip, which instantiates this
tracker via the module patch below.

Usage:
    python deform_with_hands_tracking.py
    # optional: --clip_seconds 20 --keypoints_per_segment 4 4 3 3 5 --sigma 3.0
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bdlo_tracking
from bdlo_tracking import process_clip  # reporting harness (metrics/video/CSV)
from tracker.wire_tracker import WireTracker


class DeformWithHandsTracker(WireTracker):
    """Dedicated deform_with_hands tracker: fixed mask, EE-first, no CPD."""

    def process_frame(self, depth, arm_depth=None, rgb=None, precomputed_arm_mask=None):
        self.frame_count += 1
        frame_idx = self.frame_count - 1

        # --- 1. fixed inputs: the rope mask is the foreground, used AS-IS ---
        foreground_mask = (precomputed_arm_mask == 0).astype(np.uint8)
        skeleton_mask = self._skeletonize(foreground_mask)
        skeleton_pc = self._extract_point_cloud(skeleton_mask, depth)

        if np.sum(skeleton_mask) < self.min_skeleton_pixels:
            self.consecutive_skips += 1
            return {'success': False, 'reason': 'insufficient_skeleton', 'mode': 'skip',
                    'foreground_mask': foreground_mask, 'skeleton_mask': skeleton_mask,
                    'skeleton_pc': skeleton_pc, 'frame_idx': frame_idx}

        if not self.is_initialized:
            # --- 2. init: EE hard-replace first (inside, before topology build),
            #        branch+leaf fixed during the repulsion optimisation.
            #        Topology discovery needs ONE connected skeleton, so the init
            #        frame uses the largest mask component (identical to the stock
            #        pipeline's is_init top-1) -- the stored mask is untouched and
            #        every TRACKING frame uses the full, unfiltered mask. ---
            init_skeleton = self._skeletonize(self._get_top_k_components(foreground_mask, k=1))
            result = self._initialize_with_segment_allocation(init_skeleton, depth)
            if result.get('success') and self.ee_to_leaf_mapping:
                # user spec 2 -- the topology builder snaps the leaf keypoints back
                # onto the skeleton (undoing the EE replacement done before it), so
                # enforce the order here: HARD-REPLACE the EE leaves with the
                # frame-0 EEs FIRST, THEN optimize the INTERMEDIATES only.  All six
                # anchor nodes are FIXED during this optimization (user spec): the
                # 2 branch nodes and the 4 leaves (2 of them = the EEs) stay at
                # their identified positions -- only intermediate keypoints are
                # projected/edge-refined.  The EE-anchored result is the reference
                # geometry for tracking.
                kp = result['keypoints'].copy()
                for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
                    kp[kp_idx] = self.ee_poses_3d[0][ee_idx]
                self.anchor_set = set(range(self.reference_n_branch + self.reference_n_leaf))
                kp = self._joint_constraint_optimization(
                    kp, self._extract_point_cloud(init_skeleton, depth))
                self.reference_lengths = np.array([
                    np.linalg.norm(kp[i] - kp[j]) for i, j in self.reference_edges])
                self.reference_keypoints = kp.copy()
                self.prev_keypoints = kp.copy()
                result['keypoints'] = kp
                result['keypoints_2d'] = self._project_3d_to_2d(kp)
                # tracking anchors: the EE-mapped leaves
                self.anchor_set = set(self.ee_to_leaf_mapping.values())
                # user spec: free-leaf segments must propagate corrections FROM the
                # branch node TO the free leaf (the initializer emits them
                # leaf-first); reorder those segments branch->leaf
                nb = self.reference_n_branch
                free = set(range(nb, nb + self.reference_n_leaf)) - self.anchor_set
                self.segment_edges = [
                    self._orient_branch_to_leaf(seg, free, nb) for seg in self.segment_edges]
        else:
            result = self._track_ee_anchored(foreground_mask, skeleton_mask, skeleton_pc,
                                             depth, frame_idx)

        result['foreground_mask'] = foreground_mask
        result['skeleton_mask'] = skeleton_mask
        result['skeleton_pc'] = skeleton_pc
        result['frame_idx'] = frame_idx
        return result

    @staticmethod
    def _orient_branch_to_leaf(seg, free_leaves, n_branch):
        """Return the segment's edges ordered/oriented to walk BRANCH -> FREE LEAF.

        Segments not ending in a free leaf are returned unchanged.  Edge tuples
        are re-oriented along the walk; _joint_constraint_optimization looks up
        reference lengths under both (i,j) and (j,i), so orientation is safe.
        """
        from collections import defaultdict
        adj = defaultdict(list)
        for i, j in seg:
            adj[i].append(j)
            adj[j].append(i)
        ends = [n for n in adj if len(adj[n]) == 1]
        leaf_ends = [n for n in ends if n in free_leaves]
        branch_ends = [n for n in ends if n < n_branch]
        if not leaf_ends or not branch_ends:
            return seg
        order, prev, cur = [], None, branch_ends[0]
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            order.append((cur, nxt[0]))
            prev, cur = cur, nxt[0]
        return order if len(order) == len(seg) else seg

    MATCH_GATE = 100.0     # mm; a matched BRANCH farther than this from the previous
                           # position is a mis-identification -> node stays free
    RECAL_GATE = 12.0      # mm; a raw branch match jumping more than this triggers
                           # the pruned re-calibration for the frame (user design)
    LEAF_SEARCH = 50.0     # mm; free-leaf tip candidates are searched within this
                           # radius of the leaf's previous position (user design)

    def _match_anchors(self, keypoints, det_b, ee_kp, gate):
        """Match B0/B1 <- det_b (anchored when accepted; beyond `gate` rejected).

        FREE LEAVES ARE NEVER MATCHED (user policy: 'project to point cloud and
        free to optimize') -- the mask is not guaranteed connected, so leaf tips
        may sit on broken fragments; L2/L3 simply carry their tracked position
        into the optimizer, which projects them onto the cloud and refines them
        branch -> leaf, ending with the final projection.  Branch junctions are
        always on the main body, so largest-component candidates are valid.
        Returns (matched keypoints, anchored indices, max accepted BRANCH jump --
        the re-calibration trigger is branch-only per user spec)."""
        kp = keypoints.copy()
        anchors = set(ee_kp)
        max_branch_jump = 0.0
        nb = self.reference_n_branch
        if len(det_b) > 0 and nb > 0:
            r_, c_ = linear_sum_assignment(cdist(kp[:nb], det_b))
            for rr, cc in zip(r_, c_):
                jump = np.linalg.norm(kp[rr] - det_b[cc])
                if jump < gate:
                    kp[rr] = det_b[cc]
                    anchors.add(rr)
                    max_branch_jump = max(max_branch_jump, jump)
        return kp, anchors, max_branch_jump

    def _track_ee_anchored(self, foreground_mask, skeleton_mask, skeleton_pc, depth, frame_idx):
        # --- 3a. EE hard-replace FIRST ---
        keypoints = self.prev_keypoints.copy()
        ee_kp = set(self.ee_to_leaf_mapping.values()) if self.ee_to_leaf_mapping else set()
        if self.ee_to_leaf_mapping is not None and frame_idx < len(self.ee_poses_3d):
            ee = self.ee_poses_3d[frame_idx]
            for ee_idx, kp_idx in self.ee_to_leaf_mapping.items():
                keypoints[kp_idx] = ee[ee_idx]

        # --- 3b. fast path: RAW branch candidates + Hungarian.  Identification
        #         runs on the LARGEST component -- valid for the BRANCH nodes
        #         (user: junctions are always on the main body; leaf tips may sit
        #         on broken fragments, but free leaves are never matched) ---
        skel_big = self._skeletonize(self._get_top_k_components(foreground_mask, k=1))
        b2d, _, adjacency, coords = self._node_identification(skel_big)
        det_b = self._pixel_to_3d(b2d, depth)
        kp_raw, anchors, max_jump = self._match_anchors(
            keypoints, det_b, ee_kp, self.MATCH_GATE)

        if max_jump > self.RECAL_GATE:
            # --- gated re-calibration (user design): a big branch jump means the
            #     raw candidates are suspect (spur chains) -> re-match against the
            #     PRUNED junctions (same prune as init, REUSING the adjacency) ---
            det_b = np.empty((0, 3))
            if adjacency is not None:
                pruned = self._prune_to_target_topology(adjacency, coords)
                det_b = self._pixel_to_3d(pruned['branch_coords'], depth)
            keypoints, anchors, _ = self._match_anchors(
                keypoints, det_b, ee_kp, self.MATCH_GATE)
            self.recal_frames = getattr(self, 'recal_frames', []) + [frame_idx]
        else:
            keypoints = kp_raw

        # --- 3b2. free leaves (user design): tips on the FULL skeleton (all
        #          fragments count), search within LEAF_SEARCH of the previous
        #          position, ASSIGN THE LOWEST candidate as initialization; the
        #          leaf stays FREE for the optimization ---
        simg = (skeleton_mask > 0).astype(np.uint8)
        nbrs = cv2.filter2D(simg, -1, np.ones((3, 3), np.uint8)) - simg
        tr_, tc_ = np.nonzero((simg > 0) & (nbrs == 1))  # skeleton endpoints
        tips = self._pixel_to_3d(np.stack([tr_, tc_], 1), depth) if len(tr_) else np.empty((0, 3))
        nb = self.reference_n_branch
        for k in range(nb, nb + self.reference_n_leaf):
            if k in ee_kp:
                continue
            if len(tips):
                near = tips[np.linalg.norm(tips - keypoints[k], axis=1) < self.LEAF_SEARCH]
                if len(near):
                    keypoints[k] = near[np.argmax(near[:, 1])]  # lowest: camera +y is down

        # --- 3c. optimize with EE leaves + all snapped nodes ANCHORED:
        #         projection to point cloud + segment-ordered edge refinement ---
        self.anchor_set = anchors
        keypoints = self._joint_constraint_optimization(keypoints, skeleton_pc)

        # --- 3d. FINAL PROJECTION back to the point cloud (user spec, mirrors
        #         the initializer's 'Final projection' contract): the optimizer
        #         ends on an edge pass, so free keypoints can be left off-cloud;
        #         snap every non-anchor keypoint to its nearest cloud point ---
        if len(skeleton_pc) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(skeleton_pc)
            _, idx = nn.kneighbors(keypoints)
            for k in range(len(keypoints)):
                if k not in self.anchor_set:
                    keypoints[k] = skeleton_pc[idx[k, 0]]

        self.prev_keypoints = keypoints.copy()
        self.consecutive_skips = 0
        return {'success': True, 'mode': 'track',
                'keypoints': keypoints,
                'keypoints_2d': self._project_3d_to_2d(keypoints),
                'edges': self.reference_edges,
                'edge_errors': self._compute_edge_errors(keypoints),
                'edge_rmse_mm': self._compute_edge_rmse_mm(keypoints)}


bdlo_tracking.WireTracker = DeformWithHandsTracker  # process_clip instantiates this

DIR = Path(__file__).resolve().parent
from paths import UNDIST_NPZ as _U
UNDIST_NPZ = str(_U)
from paths import HANDS_NPZ as _H
HANDS_NPZ = str(_H)
from paths import ROPE_MASKS_NPZ as _RM
ROPE_NPZ = str(_RM)


def main():
    parser = argparse.ArgumentParser(description='Dedicated BDLO tracking for deform_with_hands')
    parser.add_argument('--clip_seconds', type=int, default=20, help='Clip duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    parser.add_argument('--keypoints_per_segment', type=int, nargs=5, default=[4, 4, 3, 3, 5],
                        help='Intermediate keypoints per segment: [ee0, ee1, free0, free1, trunk]')
    parser.add_argument('--skip_clips', type=int, nargs='*', default=[],
                        help='Clip indices to skip')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Gaussian smoothing sigma for the visualization trajectories '
                             '(metrics are still on raw)')
    args = parser.parse_args()

    # BDLO: 2 branch + 4 leaf + sum(intermediate)
    n_keypoints = 2 + 4 + sum(args.keypoints_per_segment)

    output_dir = DIR / 'output' / 'tracking'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('DEFORM_WITH_HANDS BDLO TRACKING (dedicated pipeline, EE-anchored leaves)')
    print('=' * 80)

    d = np.load(UNDIST_NPZ)
    masks = np.load(ROPE_NPZ)['masks']
    h = np.load(HANDS_NPZ)
    assert len(masks) == len(d['depth']) == len(h['ee']), 'frame counts disagree'

    data = {
        'color': d['color'],          # BGR, as bdlo_tracking expects
        'depth': d['depth'],          # uint16 mm; process_clip casts to float32
        'masks': masks,               # rope-only {0,1}, used AS-IS
        'n_frames': len(masks),
    }
    transforms = {'K': d['K']}
    ee_poses_3d = h['ee'].astype(np.float64) * 1000.0  # metres -> mm (tracker units)

    print(f"  Color: {data['color'].shape}")
    print(f"  Depth: {data['depth'].shape}")
    print(f"  Rope masks: {masks.shape}")
    print(f"  EE poses: {ee_poses_3d.shape} (mm, camera frame; smoothed + inpainted)")
    print(f"  n_keypoints: {n_keypoints}  keypoints_per_segment: {args.keypoints_per_segment}")

    frames_per_clip = args.clip_seconds * args.fps
    n_clips = (data['n_frames'] + frames_per_clip - 1) // frames_per_clip
    print(f"\nClip configuration: {n_clips} clip(s) x {frames_per_clip} frames "
          f"(last: {data['n_frames'] - (n_clips - 1) * frames_per_clip})")

    all_clip_results = []
    for clip_idx in range(n_clips):
        if clip_idx in set(args.skip_clips):
            print(f'\n  Skipping clip {clip_idx}')
            continue
        start = clip_idx * frames_per_clip
        end = min(start + frames_per_clip, data['n_frames'])
        all_clip_results.append(process_clip(
            data=data,
            transforms=transforms,
            ee_poses_3d=ee_poses_3d,
            clip_idx=clip_idx,
            start_frame=start,
            end_frame=end,
            output_dir=output_dir,
            n_keypoints=n_keypoints,
            tail_length=30,  # user: 30 past frames, the default 60 is too long
            fps=args.fps,
            keypoints_per_segment=args.keypoints_per_segment,
            sigma=args.sigma,
        ))

    # keep: 3d_keypoints.npz, smoothed_3d_keypoints.npz, summary.txt (user spec).
    # The video is NOT the harness one -- render_tracking.py (env hamer) creates
    # tracking_deform_with_hands.mp4 as the pipeline's final step.
    for r in all_clip_results:
        clip_dir = output_dir / f"clip_{r['clip_idx']}"
        full = clip_dir / 'tracking_full.mp4'
        if full.exists():
            full.unlink()
    print(f'\nPer-clip outputs (3d_keypoints.npz, smoothed_3d_keypoints.npz, '
          f'summary.txt): {output_dir}/clip_*/  -- video: run render_tracking.py')


if __name__ == '__main__':
    main()
