---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "TrackDeform3D"
date:   2026-06-19 00:00:00 -0400
description: >- # Supports markdown
  Markerless and Autonomous **3D Keypoint Tracking** and **Dataset Collection** for Deformable Objects
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: https://roahmlab.github.io/trackDeform3D-core-tracking/web_elements/trackdeform3d_title.png
  height: 100
  width: 256
  alt: TrackDeform3D title figure

authors:
  - name: Yeheng Zong
    email: yehengz@umich.edu
    footnotes: "*"
  - name: Yizhou Chen
    email: yizhouch@umich.edu
    footnotes: "*"
  - name: Alexander Bowler
    email: albowler@umich.edu
  - name: Chia-Tung Yang
    email: yangct@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

author-footnotes: >-
  <sup>*</sup>&nbsp;Equal contribution.
  All authors are affiliated with the Department of Robotics and the Department of Mechanical Engineering of the University of Michigan, Ann Arbor.

links:
  - icon: arxiv
    icon-library: simpleicons
    text: Arxiv
    url: https://arxiv.org/abs/2603.17068
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/trackDeform3D-core-tracking

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---

# Abstract

Structured 3D representations such as keypoints and meshes offer compact, expressive descriptions of deformable objects, jointly capturing geometric and topological information useful for downstream tasks such as dynamics modeling and motion planning.
However, robustly extracting such representations remains challenging, as current perception methods struggle to handle complex deformations.
Moreover, large-scale 3D data collection remains a bottleneck: existing approaches either require prohibitive data collection efforts, such as labor-intensive annotation or expensive motion capture setups, or rely on simplifying assumptions that break down in unstructured environments.
As a result, large-scale 3D datasets and benchmarks for deformable objects remain scarce.
To address these challenges, this paper presents an affordable and autonomous framework for collecting 3D datasets of deformable objects using only RGB-D cameras.
The proposed method identifies 3D keypoints and robustly tracks their trajectories, incorporating motion consistency constraints to produce temporally smooth and geometrically coherent data.
TrackDeform3D is evaluated against several state-of-the-art tracking methods across diverse object categories and demonstrates consistent improvements in both geometric and tracking accuracy.
Using this framework, this paper presents a high-quality, large-scale dataset consisting of 6 deformable objects, totaling 110 minutes of trajectory data.
<p align="center">
  <img src="{{ '/web_elements/trackdeform3d_title.png' | relative_url }}" class="img-responsive" alt="TrackDeform3D title figure" style="width: 100%; height: auto;">
</p>
Given only the foreground point cloud of a deformable object from an RGB-D camera, TrackDeform3D recovers its keypoints and topology (graph edges) on the first frame, then tracks them consistently across the recording — with no markers and no motion-capture rig.

---

# Method
<div markdown="1" class="content-block grey justify no-pre">
Given an RGB-D video, TrackDeform3D first lifts depth images to point clouds and segments the deformable object via point cloud differencing (§III-C). From the first segmented frame, the object type is classified and 3D keypoints are initialized by detecting anchor points, generating warm-start positions, inferring object topology, and solving a constrained geometric optimization (§III-B, §III-D). During tracking, anchor points are re-detected at each frame and the same optimization is solved recursively, warm-started from the previous solution. A temporal moving-average filter is applied to suppress high-frequency jitter, producing smooth and temporally consistent 3D keypoint trajectories (§III-E).

<p align="center">
  <img src="{{ '/web_elements/trackdeform3d_method.png' | relative_url }}" class="img-responsive" alt="TrackDeform3D method overview" style="width: 100%; height: auto;">
</p>
</div>

---

# Dataset
<div markdown="1" class="content-block grey justify no-pre">
Using this framework, we collect a high-quality, large-scale dataset of **6 deformable objects** totaling **110 minutes** of trajectory data, captured with only RGB-D cameras — no markers and no motion-capture rig.

- **Raw data:** available on [Google Drive](https://drive.google.com/drive/folders/1emeePJieKG0wwYZt4PY1E7GW2_mIJVrk?usp=sharing) (RGB-D frames, foreground masks, and per-frame arm poses).
- **Processed dataset:** *coming soon* — link will be posted here.

For data format and usage, see the [code repository](https://github.com/roahmlab/trackDeform3D-core-tracking).
</div>

---

# Demo Video
<div class="fullwidth">
<video controls="" width="100%">
    <source src="{{ '/web_elements/supplementary_video.mp4' | relative_url }}" type="video/mp4">
</video>
</div>

<div markdown="1" class="content-block grey justify">

# [Citation](#citation)

This project was developed in the [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at the University of Michigan - Ann Arbor.

```bibtex
@misc{zong2026trackdeform3d,
      title={TrackDeform3D: Markerless and Autonomous 3D Keypoint Tracking and Dataset Collection for Deformable Objects},
      author={Yeheng Zong and Yizhou Chen and Alexander Bowler and Chia-Tung Yang and Ram Vasudevan},
      year={2026},
      eprint={2603.17068},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
</div>

---
