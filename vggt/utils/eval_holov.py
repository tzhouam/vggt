"""
Evaluation utilities for VGGT outputs.

Implements the same metrics used in the VGGT CVPR 2025 paper:

  Camera Pose:
    - RRA (Relative Rotation Accuracy) — angular error between all image pairs
    - RTA (Relative Translation Accuracy) — angular error of translation direction
    - AUC@θ of min(RRA, RTA) — area under accuracy-threshold curve
    (Paper reference: Sec 4.1, Table 1)

  Dense Depth / Point Map (DTU / ETH3D style):
    - Accuracy  ↓  mean distance from prediction to nearest GT point
    - Completeness  ↓  mean distance from GT to nearest predicted point
    - Overall (Chamfer)  ↓  average of Accuracy and Completeness
    (Paper reference: Sec 4.2–4.3, Tables 2–3)

When ground-truth is not available, we fall back to comparing a pruned run
against the unpruned baseline using the same metrics (treating baseline as GT).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


# =========================================================================== #
#  Camera Pose — RRA / RTA / AUC  (VGGT Sec 4.1)
# =========================================================================== #

def _rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle (degrees) of a 3×3 rotation matrix."""
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def relative_rotation_accuracy(ext_a: np.ndarray, ext_b: np.ndarray) -> np.ndarray:
    """
    RRA: for every image pair (i,j), compute the angular error (degrees)
    between the relative rotations from ext_a and ext_b.

    ext_a, ext_b: (S, 3, 4) extrinsic matrices.
    Returns: (S*(S-1)/2,) array of angular errors in degrees.
    """
    S = ext_a.shape[0]
    errors = []
    for i in range(S):
        for j in range(i + 1, S):
            R_rel_a = ext_a[i, :3, :3] @ ext_a[j, :3, :3].T
            R_rel_b = ext_b[i, :3, :3] @ ext_b[j, :3, :3].T
            R_diff = R_rel_a @ R_rel_b.T
            errors.append(_rotation_angle_deg(R_diff))
    return np.array(errors, dtype=np.float64)


def relative_translation_accuracy(ext_a: np.ndarray, ext_b: np.ndarray) -> np.ndarray:
    """
    RTA: for every image pair (i,j), compute the angular error (degrees)
    between the relative translation directions from ext_a and ext_b.

    ext_a, ext_b: (S, 3, 4) extrinsic matrices.
    Returns: (S*(S-1)/2,) array of angular errors in degrees.
    """
    S = ext_a.shape[0]
    errors = []
    for i in range(S):
        for j in range(i + 1, S):
            t_rel_a = ext_a[j, :3, 3] - ext_a[j, :3, :3] @ ext_a[i, :3, :3].T @ ext_a[i, :3, 3]
            t_rel_b = ext_b[j, :3, 3] - ext_b[j, :3, :3] @ ext_b[i, :3, :3].T @ ext_b[i, :3, 3]
            n_a = np.linalg.norm(t_rel_a)
            n_b = np.linalg.norm(t_rel_b)
            if n_a < 1e-8 or n_b < 1e-8:
                errors.append(0.0)
                continue
            cos = np.dot(t_rel_a, t_rel_b) / (n_a * n_b)
            cos = np.clip(cos, -1.0, 1.0)
            errors.append(float(np.degrees(np.arccos(cos))))
    return np.array(errors, dtype=np.float64)


def auc_at_threshold(errors: np.ndarray, max_thresh: float = 30.0, num_bins: int = 1000) -> float:
    """
    Area under the accuracy-threshold curve up to `max_thresh` degrees,
    normalised to [0, 1].
    """
    thresholds = np.linspace(0, max_thresh, num_bins + 1)
    accuracies = [(errors <= t).mean() for t in thresholds]
    return float(np.trapz(accuracies, thresholds) / max_thresh)


def camera_pose_metrics(ext_ref: np.ndarray, ext_test: np.ndarray) -> dict:
    """
    Evaluate camera pose quality using VGGT's official protocol:
      AUC@30 of min(RRA, RTA) for all image pairs.
    """
    rra = relative_rotation_accuracy(ext_ref, ext_test)
    rta = relative_translation_accuracy(ext_ref, ext_test)
    combined = np.minimum(rra, rta)

    return {
        "RRA_mean_deg": float(rra.mean()),
        "RRA_median_deg": float(np.median(rra)),
        "RTA_mean_deg": float(rta.mean()),
        "RTA_median_deg": float(np.median(rta)),
        "AUC@5": float(auc_at_threshold(combined, 5.0)),
        "AUC@10": float(auc_at_threshold(combined, 10.0)),
        "AUC@30": float(auc_at_threshold(combined, 30.0)),
        "n_pairs": int(len(rra)),
    }


# =========================================================================== #
#  Dense Geometry — Accuracy / Completeness / Overall  (DTU / ETH3D style)
# =========================================================================== #

def _nn_distance(pc_a: np.ndarray, pc_b: np.ndarray, n_sample: int = 100_000) -> float:
    """Mean nearest-neighbour L2 distance from pc_a to pc_b (subsampled)."""
    if len(pc_a) == 0 or len(pc_b) == 0:
        return float("inf")
    if len(pc_a) > n_sample:
        pc_a = pc_a[np.random.choice(len(pc_a), n_sample, replace=False)]
    if len(pc_b) > n_sample:
        pc_b = pc_b[np.random.choice(len(pc_b), n_sample, replace=False)]

    from scipy.spatial import cKDTree
    tree_b = cKDTree(pc_b)
    d, _ = tree_b.query(pc_a, k=1)
    return float(d.mean())


def point_cloud_metrics(
    pts_ref: np.ndarray,
    pts_test: np.ndarray,
    conf_ref: Optional[np.ndarray] = None,
    conf_test: Optional[np.ndarray] = None,
    conf_thresh: float = 1.0,
) -> dict:
    """
    DTU / ETH3D-style dense geometry metrics:
      Accuracy   = mean NN distance from test → ref  (↓ better)
      Completeness = mean NN distance from ref → test  (↓ better)
      Overall    = (Accuracy + Completeness) / 2   (= Chamfer, ↓ better)
    """
    p_r = pts_ref.reshape(-1, 3).astype(np.float64)
    p_t = pts_test.reshape(-1, 3).astype(np.float64)

    ok_r = np.all(np.isfinite(p_r), axis=1)
    ok_t = np.all(np.isfinite(p_t), axis=1)
    if conf_ref is not None:
        ok_r &= conf_ref.reshape(-1) >= conf_thresh
    if conf_test is not None:
        ok_t &= conf_test.reshape(-1) >= conf_thresh

    p_r, p_t = p_r[ok_r], p_t[ok_t]
    accuracy = _nn_distance(p_t, p_r)
    completeness = _nn_distance(p_r, p_t)
    overall = (accuracy + completeness) / 2.0

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "overall_chamfer": overall,
        "n_pts_ref": int(ok_r.sum()),
        "n_pts_test": int(ok_t.sum()),
    }


# =========================================================================== #
#  Depth map metrics (per-pixel, when GT available or comparing to baseline)
# =========================================================================== #

def depth_metrics(depth_ref: np.ndarray, depth_test: np.ndarray) -> dict:
    """
    Standard monocular / MVS depth metrics:
      AbsRel, SqRel, RMSE, RMSE_log, δ<1.25, δ<1.25², δ<1.25³
    """
    d_r = depth_ref.squeeze().astype(np.float64)
    d_t = depth_test.squeeze().astype(np.float64)
    ok = np.isfinite(d_r) & np.isfinite(d_t) & (d_r > 0) & (d_t > 0)
    if ok.sum() == 0:
        return {"valid_pix": 0}

    dr, dt = d_r[ok], d_t[ok]
    abs_err = np.abs(dr - dt)
    abs_rel = (abs_err / np.clip(dr, 1e-8, None)).mean()
    sq_rel = ((abs_err ** 2) / np.clip(dr, 1e-8, None)).mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    rmse_log = np.sqrt(((np.log(np.clip(dt, 1e-8, None)) - np.log(np.clip(dr, 1e-8, None))) ** 2).mean())

    ratio = np.maximum(dt / np.clip(dr, 1e-8, None), dr / np.clip(dt, 1e-8, None))
    d1 = (ratio < 1.25).mean()
    d2 = (ratio < 1.25 ** 2).mean()
    d3 = (ratio < 1.25 ** 3).mean()

    return {
        "valid_pix": int(ok.sum()),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "delta_1.25": float(d1),
        "delta_1.25^2": float(d2),
        "delta_1.25^3": float(d3),
        "mae": float(abs_err.mean()),
    }


# =========================================================================== #
#  Aggregate report
# =========================================================================== #

def full_comparison_report(
    ext_ref, ext_test,
    depth_ref, depth_test,
    conf_ref, conf_test,
    pts_ref, pts_test,
    conf_thresh: float = 1.0,
) -> dict:
    report = {}
    report["camera_pose (RRA/RTA/AUC)"] = camera_pose_metrics(ext_ref, ext_test)
    report["depth (per-pixel)"] = depth_metrics(depth_ref, depth_test)
    try:
        report["point_cloud (Acc/Comp/Chamfer)"] = point_cloud_metrics(
            pts_ref, pts_test, conf_ref, conf_test, conf_thresh=conf_thresh
        )
    except ImportError:
        report["point_cloud (Acc/Comp/Chamfer)"] = {
            "error": "scipy not installed — Chamfer distance skipped"
        }
    return report


def save_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Quality report saved to {path}")


def print_report(report: dict) -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print("  VGGT Baseline vs HoloV-Pruned — Quality Report")
    print(f"  (Camera: RRA/RTA/AUC per VGGT Sec 4.1)")
    print(f"  (Geometry: Accuracy/Completeness/Overall per DTU/ETH3D)")
    print(bar)
    for section, metrics in report.items():
        print(f"\n  [{section}]")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k:30s}  {v:.6f}")
                else:
                    print(f"    {k:30s}  {v}")
    print(f"\n{bar}\n")
