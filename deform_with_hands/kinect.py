"""Shared Kinect I/O for the deform_with_hands pipeline.

The capture is an Azure Kinect recording whose depth is already registered into the
colour frame (both 1280x720), so everything is lifted with the *colour* intrinsics.
"""
import json
import struct
import zipfile

import cv2
import numpy as np

from paths import RAW_DIR as _RAW, UNDIST_NPZ as _UNDIST
RAW_DIR = str(_RAW)
UNDIST_NPZ = str(_UNDIST)

# t = 2s .. 18s at capture_fps = 29.98698
T_START, T_END = 2.0, 18.0

W, H = 1280, 720


def load_calib(path=f'{RAW_DIR}/calibration.json'):
    """Colour K (3x3) and OpenCV 8-param rational distCoeffs.

    calibration.json lists the coefficients in Azure Kinect SDK order
    (k1..k6, codx, cody, p2, p1) -- note p2 BEFORE p1 -- so the distCoeffs
    vector is built by name, never by key order.
    """
    with open(path) as f:
        c = json.load(f)['color']
    i = c['intrinsics']
    K = np.array(c['K'], dtype=np.float64)
    dist = np.array([i['k1'], i['k2'], i['p1'], i['p2'],
                     i['k3'], i['k4'], i['k5'], i['k6']], dtype=np.float64)
    return K, dist


def undistort_maps(K, dist):
    return cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)


def memmap_npz(path, name):
    """Memory-map one member of an uncompressed .npz (avoids a 4.1 GB read)."""
    with open(path, 'rb') as fh:
        info = zipfile.ZipFile(fh).getinfo(name + '.npy')
        fh.seek(info.header_offset)
        n, m = struct.unpack('<HH', fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(fh)
        shape, _, dtype = np.lib.format._read_array_header(fh, version)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def frame_window():
    """Frame indices and timestamps for t in [T_START, T_END]."""
    fps = float(np.load(f'{RAW_DIR}/rgbd.npz')['capture_fps'])
    frames = np.arange(int(round(T_START * fps)), int(round(T_END * fps)) + 1)
    return frames, frames / fps, fps


def load_undistorted():
    """Undistorted colour (BGR) + depth (uint16 mm) for the window, plus K."""
    d = np.load(UNDIST_NPZ)
    return d['color'], d['depth'], d['K'], d['frames'], d['t']


def unproject(depth_mm, K):
    """Organised point cloud (H,W,3) in metres, camera frame. Invalid depth -> 0."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth_mm.astype(np.float32) / 1000.0
    u, v = np.meshgrid(np.arange(depth_mm.shape[1], dtype=np.float32),
                       np.arange(depth_mm.shape[0], dtype=np.float32))
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def project(pts, K):
    """(N,3) camera-frame metres -> (N,2) pixels."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.maximum(pts[:, 2], 1e-6)
    return np.stack([fx * pts[:, 0] / z + cx, fy * pts[:, 1] / z + cy], axis=1)
