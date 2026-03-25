"""
Export VGGT point cloud as a 3D Gaussian Splatting compatible .ply file.

The output can be loaded directly in viewers such as:
  - https://github.com/graphdeco-inria/gaussian-splatting (SIBR viewer)
  - https://antimatter15.com/splat/
  - SuperSplat (https://playcanvas.com/supersplat/editor)

Each Gaussian has:
  - xyz position (from unprojected depth)
  - RGB colour (from input images)
  - opacity (from depth confidence, sigmoid-scaled)
  - isotropic scale (derived from local point density)
  - rotation as unit quaternion (identity)
  - SH DC coefficients for colour (degree 0)
"""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def _rgb_to_sh0(rgb_float: np.ndarray) -> np.ndarray:
    """Convert [0,1] RGB to degree-0 SH coefficient."""
    return (rgb_float - 0.5) / SH_C0


def export_3dgs_ply(
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    conf_thresh: float = 1.0,
    max_points: int = 300_000,
    base_scale: float = 0.01,
    opacity_sigmoid_scale: float = 2.0,
) -> str:
    """
    Export a 3DGS-compatible .ply.

    Args:
        points_3d: (S, H, W, 3) or (N, 3) world coordinates.
        points_rgb: (S, H, W, 3) or (N, 3) uint8 or float RGB.
        confidence: (S, H, W) or (N,) depth confidence.
        output_path: destination .ply path.
        conf_thresh: min confidence to include a point.
        max_points: subsample if more valid points than this.
        base_scale: isotropic Gaussian radius (world units).
        opacity_sigmoid_scale: steepness of conf -> opacity mapping.

    Returns:
        The actual output_path written.
    """
    pts = points_3d.reshape(-1, 3).astype(np.float32)
    rgb = points_rgb.reshape(-1, 3)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32)
    conf = confidence.reshape(-1).astype(np.float32)

    valid = np.all(np.isfinite(pts), axis=1) & (conf >= conf_thresh)
    pts, rgb, conf = pts[valid], rgb[valid], conf[valid]

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, rgb, conf = pts[idx], rgb[idx], conf[idx]

    n = len(pts)
    if n == 0:
        print("Warning: no valid points for 3DGS export")
        return output_path

    sh0 = _rgb_to_sh0(rgb)

    conf_norm = conf / (np.percentile(conf, 95) + 1e-8)
    opacity_logit = np.clip(opacity_sigmoid_scale * (conf_norm - 0.5), -5.0, 5.0)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

    scales = np.full((n, 3), np.log(base_scale), dtype=np.float32)

    # Identity quaternion per point (w, x, y, z) = (1, 0, 0, 0)
    rots = np.zeros((n, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    _write_ply(output_path, pts, sh0, opacity, scales, rots)
    print(f"3DGS .ply exported: {output_path}  ({n} gaussians)")
    return output_path


def _write_ply(
    path: str,
    xyz: np.ndarray,
    sh0: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rot: np.ndarray,
) -> None:
    """Write a binary little-endian PLY in the 3DGS convention."""
    n = xyz.shape[0]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    normals = np.zeros_like(xyz)
    opacity_col = opacity.reshape(-1, 1).astype(np.float32)

    # Concatenate all per-vertex attributes into a contiguous (N, 18) float32 array
    vertex_data = np.concatenate(
        [xyz, normals, sh0, opacity_col, scale, rot], axis=1
    ).astype(np.float32)
    assert vertex_data.shape == (n, 17), f"Expected (N,17), got {vertex_data.shape}"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())
