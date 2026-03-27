# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap

try:
    import cv2
except ImportError:
    cv2 = None


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=-1,
        help="Confidence threshold for depth filtering (wo BA). "
        "depth_conf uses expp1 activation (values >= 1.0, typically max ~3). "
        "Default: -1 (auto — pick the top-N most confident points where N = max_points_for_colmap)"
    )
    ######### HoloV scatter (experimental, no retraining) #########
    parser.add_argument(
        "--holov_scatter",
        action="store_true",
        help="Apply HoloV-style sparse selection + dense scatter on patch tokens before DPT heads",
    )
    parser.add_argument(
        "--holov_keep_ratio",
        type=float,
        default=0.5,
        help="Target fraction of patch tokens to keep when --holov_scatter is set",
    )
    parser.add_argument(
        "--holov_num_groups",
        type=int,
        default=16,
        help="Number of groups along flattened patch sequence (HoloV num_patches)",
    )
    parser.add_argument(
        "--holov_prune_layer",
        type=int,
        default=4,
        help="Prune tokens INSIDE the aggregator after this AA block (0-indexed). "
        "Blocks 0..N run with full tokens; blocks N+1..23 run with reduced tokens. "
        "Lower = more speedup but coarser importance estimates. Default: 4",
    )
    ######### Token merge pipeline (improved unmerge) #########
    parser.add_argument(
        "--token_merge",
        action="store_true",
        help="Use improved merge-unmerge pipeline inside global attention layers "
        "(training-free acceleration with residual-compensated unmerge)",
    )
    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.75,
        help="Fraction of non-anchor patch tokens that become src (merged). "
        "Higher = more compression / speedup. Default: 0.75",
    )
    parser.add_argument(
        "--merge_salient_ratio",
        type=float,
        default=0.1,
        help="Fraction of patch tokens protected as salient anchors. Default: 0.1",
    )
    parser.add_argument(
        "--merge_residual_weight",
        type=float,
        default=0.3,
        help="Residual compensation weight during unmerge (0=pure copy, 1=full residual). Default: 0.3",
    )
    parser.add_argument(
        "--merge_start_block",
        type=int,
        default=0,
        help="First global-attention block to apply merging (0-indexed). Default: 0",
    )
    ######### AVGGT-style fast mode (quality-preserving) #########
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="AVGGT-style quality-preserving acceleration: convert early global layers "
        "to frame attention + KV subsampling in later layers. Preferred over --token_merge.",
    )
    parser.add_argument(
        "--fast_early_frame_layers",
        type=int,
        default=8,
        help="Number of early global layers to run as frame attention (no cross-view). Default: 8",
    )
    parser.add_argument(
        "--fast_kv_ratio",
        type=float,
        default=0.25,
        help="Fraction of K/V tokens to keep in global attention (later layers). Default: 0.25",
    )
    parser.add_argument(
        "--fast_no_mean_fill",
        action="store_true",
        help="Disable mean-fill token in KV subsampling (not recommended).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to local VGGT-1B model.pt. If the file does not exist, it is downloaded once from Hugging Face "
        "and saved here. Default: ~/.cache/vggt/VGGT-1B_model.pt (overridable with env VGGT_WEIGHTS)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save RGB / depth / confidence PNGs under scene_dir/visuals/ (requires opencv-python)",
    )
    parser.add_argument(
        "--visualize_max_frames",
        type=int,
        default=0,
        help="Max frames to export for --visualize (0 = all frames)",
    )
    ######### Quality evaluation #########
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the unpruned baseline alongside the accelerated method and print quality metrics "
        "(depth, pose, Chamfer). Works with --holov_scatter, --token_merge, or --fast_mode.",
    )
    parser.add_argument(
        "--dup_factors",
        type=str,
        default=None,
        help="Comma-separated list of duplication factors for scaling benchmark. "
        "E.g. '1,2,4,8' with 25 images → runs at 25, 50, 100, 200 frames. "
        "Requires --eval. Prints a summary table at the end.",
    )
    ######### 3DGS export #########
    parser.add_argument(
        "--export_3dgs",
        action="store_true",
        help="Export a 3D Gaussian Splatting compatible .ply under scene_dir/sparse/",
    )
    parser.add_argument(
        "--gs_max_points",
        type=int,
        default=300000,
        help="Maximum number of Gaussians in the 3DGS .ply export",
    )
    return parser.parse_args()


_VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _resolve_vggt_weights_path(args) -> Path:
    if getattr(args, "weights", None):
        return Path(args.weights).expanduser().resolve()
    env = os.environ.get("VGGT_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "vggt" / "VGGT-1B_model.pt").resolve()


def _torch_load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_vggt_visuals(
    scene_dir: str,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    images_tensor: torch.Tensor,
    max_frames: int = 0,
) -> None:
    """
    Export per-frame RGB (resized to depth resolution), depth colormap, and confidence maps to scene_dir/visuals/.
    """
    if cv2 is None:
        print("Skipping --visualize: install opencv-python (see requirements_demo.txt)")
        return

    visuals_dir = os.path.join(scene_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    d = np.asarray(depth_map)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    c = np.asarray(depth_conf)
    if c.ndim == 4 and c.shape[-1] == 1:
        c = c[..., 0]

    num_frames = d.shape[0]
    h, w = int(d.shape[1]), int(d.shape[2])
    n_out = num_frames if max_frames <= 0 else min(num_frames, max_frames)

    images_cpu = images_tensor.detach().cpu()
    conf_cmap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_INFERNO)

    readme = [
        "VGGT visualization outputs (same resolution as VGGT inference, e.g. 518).",
        "- frame_XXXX_triptych.png: RGB | depth (turbo) | confidence",
        "- frame_XXXX_depth.png, frame_XXXX_conf.png: single channels as color maps",
        "Open sparse/points.ply in MeshLab or CloudCompare for the point cloud.",
    ]
    with open(os.path.join(visuals_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme) + "\n")

    for fi in range(n_out):
        depth_hw = d[fi]
        conf_hw = c[fi]

        valid = np.isfinite(depth_hw) & (depth_hw > 0)
        if valid.any():
            lo, hi = np.nanpercentile(depth_hw[valid], [2.0, 98.0])
            dn = np.clip(depth_hw, lo, hi)
            dn = (dn - lo) / (hi - lo + 1e-8)
        else:
            dn = np.zeros_like(depth_hw, dtype=np.float64)
        dn_u8 = (np.clip(dn, 0, 1) * 255).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(dn_u8, cv2.COLORMAP_TURBO)

        valid_c = np.isfinite(conf_hw)
        if valid_c.any():
            lo_c, hi_c = np.nanpercentile(conf_hw[valid_c], [2.0, 98.0])
            cn = np.clip(conf_hw, lo_c, hi_c)
            cn = (cn - lo_c) / (hi_c - lo_c + 1e-8)
        else:
            cn = np.zeros_like(conf_hw, dtype=np.float64)
        cn_u8 = (np.clip(cn, 0, 1) * 255).astype(np.uint8)
        conf_bgr = cv2.applyColorMap(cn_u8, conf_cmap)

        img = images_cpu[fi : fi + 1]
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        rgb = img.squeeze(0).numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        triptych = np.concatenate([rgb_bgr, depth_bgr, conf_bgr], axis=1)
        stem = f"frame_{fi:04d}"
        cv2.imwrite(os.path.join(visuals_dir, f"{stem}_triptych.png"), triptych)
        cv2.imwrite(os.path.join(visuals_dir, f"{stem}_depth.png"), depth_bgr)
        cv2.imwrite(os.path.join(visuals_dir, f"{stem}_conf.png"), conf_bgr)

    print(f"Saved visualization PNGs to {visuals_dir} ({n_out} frames)")


def load_vggt_state_dict(weights_path: Path) -> dict:
    """
    Load VGGT-1B weights from a local file if present; otherwise download, save, then load.
    """
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.is_file():
        print(f"Loading VGGT weights from local file: {weights_path}")
        return _torch_load_state_dict(weights_path)
    print(f"Downloading VGGT weights from {_VGGT_WEIGHTS_URL}")
    print(f"Saving to {weights_path} (reused on next run)")
    state_dict = torch.hub.load_state_dict_from_url(_VGGT_WEIGHTS_URL, map_location="cpu")
    torch.save(state_dict, weights_path)
    return state_dict


def _print_timing(label: str, timing: dict) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  Timing: {label}")
    print(bar)
    e2e = timing.get("end_to_end", 0)
    for k, v in timing.items():
        pct = f"({v / e2e * 100:5.1f}%)" if e2e > 0 and k != "end_to_end" else ""
        print(f"    {k:28s}  {v:8.4f}s  {pct}")
    print(bar + "\n")


def _timing_speedup(t_bl: dict, t_accel: dict) -> dict:
    result = {}
    for k in t_bl:
        bl_v = t_bl.get(k, 0)
        ac_v = t_accel.get(k, 0)
        if bl_v > 0 and ac_v > 0:
            result[k] = f"{bl_v / ac_v:.2f}x"
        elif bl_v > 0:
            result[k] = "inf (0 in accel)"
        else:
            result[k] = "n/a"
    for k in t_accel:
        if k not in result:
            result[k] = "n/a (not in baseline)"
    return result


def _print_timing_comparison(t_bl: dict, t_accel: dict, label: str = "Accel") -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  Timing Comparison: Baseline vs {label}")
    print(bar)
    col = label[:10]
    print(f"  {'Stage':28s}  {'Baseline':>10s}  {col:>10s}  {'Speedup':>10s}")
    print(f"  {'─' * 28}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    all_keys = list(dict.fromkeys(list(t_bl.keys()) + list(t_accel.keys())))
    for k in all_keys:
        bl_v = t_bl.get(k, 0)
        ac_v = t_accel.get(k, 0)
        sp = f"{bl_v / ac_v:.2f}x" if bl_v > 0 and ac_v > 0 else "—"
        print(f"  {k:28s}  {bl_v:9.4f}s  {ac_v:9.4f}s  {sp:>10s}")
    print(bar + "\n")


def _print_dup_summary(rows, label: str = "Accel"):
    """
    rows: list of dicts with keys:
      dup, n_frames, bl_e2e, ac_e2e, speedup,
      bl_frame, ac_frame, bl_global, ac_global,
      sp_frame, sp_global
    """
    col = label[:6]
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"  Scaling Benchmark Summary — Baseline vs {label} at different frame counts")
    print(bar)
    hdr = (f"  {'dup':>4s} {'frames':>6s}  "
           f"{'BL e2e':>9s} {col + ' e2e':>9s} {'e2e↑':>7s}  "
           f"{'BL frame':>9s} {col + ' frm':>9s} {'frm↑':>7s}  "
           f"{'BL global':>9s} {col + ' glb':>9s} {'glb↑':>7s}")
    print(hdr)
    print(f"  {'─' * 4} {'─' * 6}  "
          f"{'─' * 9} {'─' * 9} {'─' * 7}  "
          f"{'─' * 9} {'─' * 9} {'─' * 7}  "
          f"{'─' * 9} {'─' * 9} {'─' * 7}")
    for r in rows:
        print(f"  {r['dup']:>4s} {r['n_frames']:>6d}  "
              f"{r['bl_e2e']:>8.3f}s {r['ac_e2e']:>8.3f}s {r['speedup']:>6.2f}x  "
              f"{r['bl_frame']:>8.3f}s {r['ac_frame']:>8.3f}s {r['sp_frame']:>6.2f}x  "
              f"{r['bl_global']:>8.3f}s {r['ac_global']:>8.3f}s {r['sp_global']:>6.2f}x")
    print(bar + "\n")


def _cuda_sync_time():
    """Return wall-clock seconds after a CUDA synchronise (safe on CPU too)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _scatter_intermediate_back(reduced_inter, kept_patch_idx, nearest_for_hole,
                               hole_idx, patch_start_idx, P_full, B, S):
    """
    Expand a pruned intermediate [B, S, patch_start_idx+K, D] back to [B, S, P_full, D].
    Holes are filled by copying the nearest kept token (precomputed).
    """
    D = reduced_inter.shape[-1]
    device = reduced_inter.device
    full = torch.zeros(B, S, P_full, D, device=device, dtype=reduced_inter.dtype)
    full[:, :, :patch_start_idx, :] = reduced_inter[:, :, :patch_start_idx, :]
    kept_part = reduced_inter[:, :, patch_start_idx:, :]
    full[:, :, patch_start_idx + kept_patch_idx, :] = kept_part
    if len(hole_idx) > 0:
        full[:, :, patch_start_idx + hole_idx, :] = kept_part[:, :, nearest_for_hole, :]
    return full


def _run_aggregator_timed(aggregator, images, prune_config=None):
    """
    Run Aggregator with per-stage timing.

    prune_config: None (no pruning) or dict with:
        'after_block': int   — prune after this AA block (0-indexed, default 4)
        'keep_ratio': float  — fraction of patch tokens to keep
        'num_groups': int    — HoloV segment count

    When pruning is active, token count is reduced at the prune point so
    all subsequent frame / global attention layers run with fewer tokens
    (quadratic speedup in self-attention).

    Returns (aggregated_tokens_list, patch_start_idx, timing_dict).
    """
    B, S, C_in, H, W = images.shape
    t = {}
    psi = aggregator.patch_start_idx
    patch_h = H // aggregator.patch_size
    patch_w = W // aggregator.patch_size
    N_patch = patch_h * patch_w

    # --- 1. DINO / patch embed ---
    t0 = _cuda_sync_time()
    norm_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
    flat_images = norm_images.view(B * S, C_in, H, W)
    patch_tokens = aggregator.patch_embed(flat_images)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    t["dino_patch_embed"] = _cuda_sync_time() - t0

    from vggt.models.aggregator import slice_expand_and_flatten
    camera_token = slice_expand_and_flatten(aggregator.camera_token, B, S)
    register_token = slice_expand_and_flatten(aggregator.register_token, B, S)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

    pos = None
    if aggregator.rope is not None:
        pos = aggregator.position_getter(B * S, patch_h, patch_w, device=images.device)
    if psi > 0:
        pos = pos + 1
        pos_special = torch.zeros(B * S, psi, 2, device=images.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    _, P, C = tokens.shape
    P_full = P

    # Pruning bookkeeping
    pruned = False
    kept_patch_idx = None
    hole_idx = None
    nearest_for_hole = None

    frame_idx = 0
    global_idx = 0
    output_list = []
    t["frame_attn"] = 0.0
    t["global_attn"] = 0.0
    t["holov_prune"] = 0.0

    for block_num in range(aggregator.aa_block_num):
        for attn_type in aggregator.aa_order:
            if attn_type == "frame":
                ts = _cuda_sync_time()
                tokens, frame_idx, frame_inter = aggregator._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos
                )
                t["frame_attn"] += _cuda_sync_time() - ts
            elif attn_type == "global":
                ts = _cuda_sync_time()
                tokens, global_idx, global_inter = aggregator._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos
                )
                t["global_attn"] += _cuda_sync_time() - ts

        if pruned:
            for i in range(len(frame_inter)):
                fi = _scatter_intermediate_back(
                    frame_inter[i], kept_patch_idx, nearest_for_hole, hole_idx, psi, P_full, B, S)
                gi = _scatter_intermediate_back(
                    global_inter[i], kept_patch_idx, nearest_for_hole, hole_idx, psi, P_full, B, S)
                output_list.append(torch.cat([fi, gi], dim=-1))
        else:
            for i in range(len(frame_inter)):
                output_list.append(torch.cat([frame_inter[i], global_inter[i]], dim=-1))

        # --- Prune after specified block ---
        if prune_config is not None and not pruned and block_num == prune_config["after_block"]:
            ts = _cuda_sync_time()
            kr = max(min(float(prune_config["keep_ratio"]), 1.0), 1e-6)
            keep_num = max(1, int(N_patch * kr))
            num_groups = prune_config["num_groups"]

            if tokens.shape[0] != B * S:
                tokens = tokens.view(B, S, P, C).reshape(B * S, P, C)
            if pos is not None and pos.shape[0] != B * S:
                pos = pos.view(B, S, P, 2).reshape(B * S, P, 2)

            cur_patches = tokens[:, psi:, :].detach()
            attn_proxy = cur_patches.norm(dim=-1).mean(dim=0)
            avg_patch = cur_patches.mean(dim=0)

            from vggt.utils.holov_scatter import _holov_single
            mask = _holov_single(avg_patch, attn_proxy, num_groups, keep_num)
            kept_patch_idx = torch.where(mask)[0]
            hole_idx = torch.where(~mask)[0]
            K = len(kept_patch_idx)

            if len(hole_idx) > 0:
                device = tokens.device
                rows = torch.arange(N_patch, device=device).float()
                rc = torch.stack([rows // patch_w, rows % patch_w], dim=1)
                dist = torch.cdist(rc[hole_idx], rc[kept_patch_idx], p=2.0)
                nearest_for_hole = dist.argmin(dim=1)
            else:
                nearest_for_hole = torch.zeros(0, dtype=torch.long, device=tokens.device)

            tokens = torch.cat([tokens[:, :psi, :], tokens[:, psi:][:, kept_patch_idx, :]], dim=1)
            if pos is not None:
                pos = torch.cat([pos[:, :psi, :], pos[:, psi:][:, kept_patch_idx, :]], dim=1)

            P = psi + K
            C = tokens.shape[-1]
            pruned = True

            print(f"  Pruned after block {block_num}: {N_patch} → {K} patch tokens "
                  f"({K / N_patch * 100:.0f}%), P {P_full} → {P}")
            t["holov_prune"] = _cuda_sync_time() - ts

    if t["holov_prune"] == 0.0:
        del t["holov_prune"]

    return output_list, aggregator.patch_start_idx, t


def _run_aggregator_fast_timed(aggregator, images,
                               early_frame_layers=4, kv_ratio=0.25, use_mean_fill=True):
    """Run Aggregator.forward_fast with per-stage timing breakdown."""
    from vggt.models.aggregator import slice_expand_and_flatten
    from vggt.utils.token_merge import select_kv_indices, run_block_kv_subsample

    B, S, C_in, H, W = images.shape
    t = {}
    psi = aggregator.patch_start_idx
    patch_h = H // aggregator.patch_size
    patch_w = W // aggregator.patch_size

    # --- 1. DINO / patch embed ---
    t0 = _cuda_sync_time()
    norm_images = (images - aggregator._resnet_mean) / aggregator._resnet_std
    flat_images = norm_images.view(B * S, C_in, H, W)
    patch_tokens = aggregator.patch_embed(flat_images)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    t["dino_patch_embed"] = _cuda_sync_time() - t0

    camera_token = slice_expand_and_flatten(aggregator.camera_token, B, S)
    register_token = slice_expand_and_flatten(aggregator.register_token, B, S)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

    pos = None
    if aggregator.rope is not None:
        pos = aggregator.position_getter(B * S, patch_h, patch_w, device=images.device)
    if psi > 0 and pos is not None:
        pos = pos + 1
        pos_special = torch.zeros(B * S, psi, 2, device=images.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    _, P, C = tokens.shape

    frame_idx = 0
    global_idx = 0
    output_list = []
    t["frame_attn"] = 0.0
    t["global_attn"] = 0.0
    kv_idx_cache = None

    for _ in range(aggregator.aa_block_num):
        for attn_type in aggregator.aa_order:
            if attn_type == "frame":
                ts = _cuda_sync_time()
                tokens, frame_idx, frame_inter = aggregator._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos
                )
                t["frame_attn"] += _cuda_sync_time() - ts

            elif attn_type == "global":
                ts = _cuda_sync_time()

                if global_idx < early_frame_layers:
                    # Early layers: run as frame attention
                    if tokens.shape != (B * S, P, C):
                        tokens = tokens.view(B, S, P, C).view(B * S, P, C)
                    if pos is not None and pos.shape != (B * S, P, 2):
                        pos = pos.view(B, S, P, 2).view(B * S, P, 2)
                    tokens = aggregator.global_blocks[global_idx](tokens, pos=pos)
                    global_inter = [tokens.view(B, S, P, C)]
                else:
                    # Later layers: KV subsampling
                    if tokens.shape != (B, S * P, C):
                        tokens = tokens.view(B, S, P, C).view(B, S * P, C)
                    if pos is not None and pos.shape != (B, S * P, 2):
                        pos = pos.view(B, S, P, 2).view(B, S * P, 2)
                    if kv_idx_cache is None:
                        kv_idx_cache = select_kv_indices(
                            S, P, psi, tokens.device,
                            kv_ratio=kv_ratio, protect_first_frame=True,
                        )
                    tokens = run_block_kv_subsample(
                        aggregator.global_blocks[global_idx],
                        tokens, pos, kv_idx_cache, use_mean_fill=use_mean_fill,
                    )
                    global_inter = [tokens.view(B, S, P, C)]

                global_idx += 1
                t["global_attn"] += _cuda_sync_time() - ts

        for i in range(len(frame_inter)):
            output_list.append(torch.cat([frame_inter[i], global_inter[i]], dim=-1))

    return output_list, aggregator.patch_start_idx, t


def run_VGGT(
    model,
    images,
    dtype,
    resolution=518,
    holov_scatter=False,
    holov_keep_ratio=0.5,
    holov_num_groups=16,
    holov_prune_layer=4,
    token_merge=False,
    merge_ratio=0.75,
    merge_salient_ratio=0.1,
    merge_residual_weight=0.3,
    merge_start_block=0,
    fast_mode=False,
    fast_early_frame_layers=4,
    fast_kv_ratio=0.25,
    fast_mean_fill=True,
):
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    timing = {}
    t_e2e_start = _cuda_sync_time()

    use_token_merge = token_merge and not holov_scatter and not fast_mode

    prune_config = None
    if holov_scatter and not fast_mode:
        kr = max(min(float(holov_keep_ratio), 1.0), 1e-6)
        if kr < 1.0:
            prune_config = {
                "after_block": holov_prune_layer,
                "keep_ratio": kr,
                "num_groups": holov_num_groups,
            }

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension

            if fast_mode:
                aggregated_tokens_list, ps_idx, agg_t = _run_aggregator_fast_timed(
                    model.aggregator, images,
                    early_frame_layers=fast_early_frame_layers,
                    kv_ratio=fast_kv_ratio,
                    use_mean_fill=fast_mean_fill,
                )
                timing.update(agg_t)
            elif use_token_merge:
                ts = _cuda_sync_time()
                aggregated_tokens_list, ps_idx = model.aggregator.forward_merged(
                    images,
                    merge_ratio=merge_ratio,
                    salient_ratio=merge_salient_ratio,
                    residual_weight=merge_residual_weight,
                    merge_start_block=merge_start_block,
                )
                timing["aggregator_merged"] = _cuda_sync_time() - ts
            else:
                aggregated_tokens_list, ps_idx, agg_t = _run_aggregator_timed(
                    model.aggregator, images, prune_config=prune_config
                )
                timing.update(agg_t)

        # --- Camera head ---
        ts = _cuda_sync_time()
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        timing["camera_head"] = _cuda_sync_time() - ts

        # --- Depth head ---
        ts = _cuda_sync_time()
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        timing["depth_head"] = _cuda_sync_time() - ts

    timing["end_to_end"] = _cuda_sync_time() - t_e2e_start

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf, timing


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    weights_path = _resolve_vggt_weights_path(args)
    model.load_state_dict(load_vggt_state_dict(weights_path))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # ---- Parse dup_factors ----
    dup_factors = None
    has_accel = args.holov_scatter or args.token_merge or args.fast_mode

    if getattr(args, "dup_factors", None):
        dup_factors = [int(x.strip()) for x in args.dup_factors.split(",") if x.strip()]
        if not args.eval or not has_accel:
            print("Warning: --dup_factors requires --eval plus an acceleration flag. Ignoring.")
            dup_factors = None

    def _build_accel_kwargs(args):
        """Build the keyword dict that enables whichever acceleration the user chose."""
        kw = {}
        if args.fast_mode:
            kw["fast_mode"] = True
            kw["fast_early_frame_layers"] = args.fast_early_frame_layers
            kw["fast_kv_ratio"] = args.fast_kv_ratio
            kw["fast_mean_fill"] = not args.fast_no_mean_fill
        elif args.token_merge:
            kw["token_merge"] = True
            kw["merge_ratio"] = args.merge_ratio
            kw["merge_salient_ratio"] = args.merge_salient_ratio
            kw["merge_residual_weight"] = args.merge_residual_weight
            kw["merge_start_block"] = args.merge_start_block
        elif args.holov_scatter:
            kw["holov_scatter"] = True
            kw["holov_keep_ratio"] = args.holov_keep_ratio
            kw["holov_num_groups"] = args.holov_num_groups
            kw["holov_prune_layer"] = args.holov_prune_layer
        return kw

    accel_label = "[FastMode]" if args.fast_mode else "[TokenMerge]" if args.token_merge else "[HoloV]" if args.holov_scatter else ""

    # ---- When --eval: run baseline FIRST, then accelerated, both after warmup ----
    if args.eval and has_accel:
        from vggt.utils.eval_holov import full_comparison_report, save_report, print_report

        accel_kw = _build_accel_kwargs(args)
        factors = dup_factors if dup_factors else [1]
        bench_rows = []

        for dup in factors:
            if dup <= 1:
                imgs_run = images
                dup_tag = "1x"
            else:
                imgs_run = images.repeat(dup, 1, 1, 1)
                dup_tag = f"{dup}x"

            n_frames = imgs_run.shape[0]
            print(f"\n{'#' * 60}")
            print(f"  dup={dup_tag}  →  {n_frames} frames")
            print(f"{'#' * 60}")

            # 0) warmup at this frame count (different sizes need separate kernel compilation)
            print(f"Warming up CUDA kernels for {n_frames} frames …")
            _ = run_VGGT(model, imgs_run, dtype, vggt_fixed_resolution)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("Warmup done.")

            # 1) timed baseline
            print(f"Running baseline ({n_frames} frames) …")
            ext_bl, int_bl, dep_bl, conf_bl, timing_bl = run_VGGT(
                model, imgs_run, dtype, vggt_fixed_resolution
            )
            _print_timing(f"run_VGGT [baseline {dup_tag}]", timing_bl)

            # 2) timed accelerated method
            print(f"Running {accel_label} ({n_frames} frames) …")
            ext_ac, int_ac, dep_ac, conf_ac, timing_ac = run_VGGT(
                model, imgs_run, dtype, vggt_fixed_resolution,
                **accel_kw,
            )
            _print_timing(f"run_VGGT {accel_label} {dup_tag}", timing_ac)
            _print_timing_comparison(timing_bl, timing_ac, label=accel_label)

            _safe = lambda d, k: d.get(k, 1e-9)
            bench_rows.append({
                "dup": dup_tag,
                "n_frames": n_frames,
                "bl_e2e": _safe(timing_bl, "end_to_end"),
                "ac_e2e": _safe(timing_ac, "end_to_end"),
                "speedup": _safe(timing_bl, "end_to_end") / max(_safe(timing_ac, "end_to_end"), 1e-9),
                "bl_frame": _safe(timing_bl, "frame_attn"),
                "ac_frame": _safe(timing_ac, "frame_attn"),
                "sp_frame": _safe(timing_bl, "frame_attn") / max(_safe(timing_ac, "frame_attn"), 1e-9),
                "bl_global": _safe(timing_bl, "global_attn"),
                "ac_global": _safe(timing_ac, "global_attn"),
                "sp_global": _safe(timing_bl, "global_attn") / max(_safe(timing_ac, "global_attn"), 1e-9),
            })

        # Use the last (or only) run for downstream outputs
        extrinsic, intrinsic, depth_map, depth_conf, timing = ext_ac, int_ac, dep_ac, conf_ac, timing_ac

        if len(bench_rows) > 1:
            _print_dup_summary(bench_rows, label=accel_label)

        # Quality report on 1x images (no dup)
        if 1 in (factors or [1]):
            ext_bl_1x, int_bl_1x, dep_bl_1x, conf_bl_1x, _ = run_VGGT(
                model, images, dtype, vggt_fixed_resolution
            )
            ext_ac_1x, int_ac_1x, dep_ac_1x, conf_ac_1x, timing_1x = run_VGGT(
                model, images, dtype, vggt_fixed_resolution,
                **accel_kw,
            )
            points_3d = unproject_depth_map_to_point_map(dep_ac_1x, ext_ac_1x, int_ac_1x)
            pts_bl = unproject_depth_map_to_point_map(dep_bl_1x, ext_bl_1x, int_bl_1x)
            report = full_comparison_report(
                ext_bl_1x, ext_ac_1x,
                dep_bl_1x, dep_ac_1x,
                conf_bl_1x, conf_ac_1x,
                pts_bl, points_3d,
                conf_thresh=args.conf_thres_value,
            )
            report[f"timing_{accel_label}"] = {k: f"{v:.4f}s" for k, v in timing_1x.items()}
            print_report(report)
            extrinsic, intrinsic, depth_map, depth_conf = ext_ac_1x, int_ac_1x, dep_ac_1x, conf_ac_1x
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        else:
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        report_path = os.path.join(args.scene_dir, "visuals", "eval_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        if 'report' in dir():
            save_report(report, report_path)
    else:
        # ---- Normal (single) timed run ----
        print("Warming up CUDA kernels …")
        _ = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup done.\n")

        extrinsic, intrinsic, depth_map, depth_conf, timing = run_VGGT(
            model,
            images,
            dtype,
            vggt_fixed_resolution,
            holov_scatter=args.holov_scatter,
            holov_keep_ratio=args.holov_keep_ratio,
            holov_num_groups=args.holov_num_groups,
            holov_prune_layer=args.holov_prune_layer,
            token_merge=args.token_merge,
            merge_ratio=args.merge_ratio,
            merge_salient_ratio=args.merge_salient_ratio,
            merge_residual_weight=args.merge_residual_weight,
            merge_start_block=args.merge_start_block,
            fast_mode=args.fast_mode,
            fast_early_frame_layers=args.fast_early_frame_layers,
            fast_kv_ratio=args.fast_kv_ratio,
            fast_mean_fill=not args.fast_no_mean_fill,
        )
        label = "run_VGGT"
        if args.fast_mode:
            label += " [FastMode]"
        elif args.holov_scatter:
            label += " [HoloV]"
        elif args.token_merge:
            label += " [TokenMerge]"
        _print_timing(label, timing)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.visualize:
        save_vggt_visuals(
            args.scene_dir,
            depth_map,
            depth_conf,
            images,
            max_frames=args.visualize_max_frames,
        )

    # ---------- 3DGS export ----------
    if args.export_3dgs:
        from vggt.utils.export_3dgs import export_3dgs_ply

        pts_for_gs = points_3d
        rgb_for_gs = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        rgb_for_gs = (rgb_for_gs.detach().cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        gs_dir = f"sparse_fast" if args.fast_mode else "sparse"
        gs_suffix = f"_fast_{args.fast_early_frame_layers}_{args.fast_kv_ratio}" if args.fast_mode else ""
        gs_path = os.path.join(args.scene_dir, gs_dir, f"gaussians{gs_suffix}.ply")
        export_3dgs_ply(
            pts_for_gs,
            rgb_for_gs,
            depth_conf,
            gs_path,
            conf_thresh=args.conf_thres_value,
            max_points=args.gs_max_points,
        )

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        if conf_thres_value > 0:
            conf_mask = depth_conf >= conf_thres_value
            conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
        else:
            # Auto: keep the top-N most confident pixels
            flat_conf = depth_conf.ravel()
            n_keep = min(max_points_for_colmap, flat_conf.size)
            # Partial sort to find the n_keep-th largest value
            kth = flat_conf.size - n_keep
            auto_thres = np.partition(flat_conf, kth)[kth]
            conf_mask = depth_conf >= auto_thres
            conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
            print(f"  Auto conf threshold: {auto_thres:.4f} "
                  f"(top {n_keep:,} of {flat_conf.size:,} pixels, "
                  f"conf range [{flat_conf.min():.2f}, {flat_conf.max():.2f}])")

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    if args.fast_mode:
        suffix = f"fast_{args.fast_early_frame_layers}_{args.fast_kv_ratio}"
        sparse_dir_name = f"sparse_fast"
        ply_name = f"points_{suffix}.ply"
    else:
        sparse_dir_name = "sparse"
        ply_name = "points.ply"

    sparse_reconstruction_dir = os.path.join(args.scene_dir, sparse_dir_name)
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    ply_path = os.path.join(sparse_reconstruction_dir, ply_name)
    trimesh.PointCloud(points_3d, colors=points_rgb).export(ply_path)
    print(f"Point cloud saved to {ply_path}")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # PNG previews if run with --visualize

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
