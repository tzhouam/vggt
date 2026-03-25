# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# HoloV-style sparse selection + scatter back to a dense patch grid for VGGT inference.
# Adapted from the HoloV reference (PKU-YuanGroup) pruning logic; no new trainable layers.

from __future__ import annotations

from typing import List, Literal

import torch

FillMode = Literal["zero", "mean", "nearest", "idw", "interpolate"]


def attention_proxy_l2_norm(patch_tokens: torch.Tensor) -> torch.Tensor:
    """Per-patch saliency proxy: L2 norm of each token (no extra forward, no LM)."""
    return patch_tokens.norm(dim=-1)


def _holov_single(
    image_token: torch.Tensor,
    image_attention: torch.Tensor,
    num_groups: int,
    keep_num: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [N] where True means keep original token at that index.
    """
    device = image_token.device
    dtype = image_token.dtype
    N, D = image_token.shape
    keep_num = max(1, min(keep_num, N))

    if keep_num >= N:
        return torch.ones(N, dtype=torch.bool, device=device)

    patch_size = N // num_groups
    remainder = N % num_groups

    image_tokens_patches: List[torch.Tensor] = []
    attention_patches: List[torch.Tensor] = []
    patch_start_indices: List[int] = []
    start_idx = 0
    for p in range(num_groups):
        current_patch_size = patch_size + (1 if p < remainder else 0)
        end_idx = start_idx + current_patch_size
        if current_patch_size > 0:
            image_tokens_patches.append(image_token[start_idx:end_idx])
            attention_patches.append(image_attention[start_idx:end_idx])
            patch_start_indices.append(start_idx)
        start_idx = end_idx

    patch_scores: List[torch.Tensor] = []
    all_patches: List[torch.Tensor] = []
    all_patch_indices: List[torch.Tensor] = []

    for p in range(len(image_tokens_patches)):
        patch_tokens = image_tokens_patches[p]
        patch_attn = attention_patches[p]
        patch_start = patch_start_indices[p]
        current_patch_size = len(patch_tokens)

        patch_indices = torch.arange(patch_start, patch_start + current_patch_size, device=device)

        if current_patch_size <= 1:
            patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
            all_patches.append(patch_tokens)
            all_patch_indices.append(patch_indices)
            continue

        with torch.no_grad():
            f_norm = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + eps)
            s = torch.mm(f_norm, f_norm.transpose(0, 1))
            eye_mask = 1 - torch.eye(current_patch_size, device=device, dtype=s.dtype)
            s_masked = s * eye_mask
            valid_entries = current_patch_size - 1
            mean_sim = s_masked.sum(dim=1) / valid_entries
            var_sim = ((s_masked - mean_sim.unsqueeze(1)) ** 2).sum(dim=1) / valid_entries
            patch_attn_scaled = patch_attn.to(dtype) * 1000.0
            var_scaling = torch.mean(torch.abs(patch_attn_scaled)) / (torch.mean(torch.abs(var_sim)) + eps)
            var_sim_scaled = var_sim * var_scaling
            alpha, beta = 1.0, 0.09
            token_scores = alpha * patch_attn_scaled + beta * var_sim_scaled
            patch_score = token_scores.mean()
            patch_scores.append(patch_score)
            all_patches.append(patch_tokens)
            all_patch_indices.append(patch_indices)

    patch_scores_t = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)
    mask = torch.zeros(N, dtype=torch.bool, device=device)

    if len(patch_scores_t) == 0:
        return mask

    power = 1
    weights = (patch_scores_t**power) / ((patch_scores_t**power).sum() + eps)
    allocated = (weights * keep_num).floor().long()
    remaining = int((keep_num - int(allocated.sum().item())))
    if remaining > 0 and len(weights) > 0:
        k_top = min(remaining, len(weights))
        _, extra_idx = torch.topk(weights, k=k_top)
        for j in range(k_top):
            allocated[extra_idx[j]] += 1

    for i, (patch, alloc, patch_indices) in enumerate(zip(all_patches, allocated, all_patch_indices)):
        p_sz = len(patch)
        if alloc <= 0:
            continue
        if alloc >= p_sz:
            mask[patch_indices] = True
        else:
            patch_attn = attention_patches[i]
            _, top_indices = torch.topk(patch_attn, k=min(int(alloc.item()), p_sz))
            kept_idx = patch_indices[top_indices]
            mask[kept_idx] = True

    # If rounding left us with too many or too few (rare), trim or add by global attention proxy
    k_cur = int(mask.sum().item())
    if k_cur > keep_num:
        kept_flat = torch.where(mask)[0]
        drop_scores = image_attention[kept_flat]
        drop_order = torch.argsort(drop_scores, descending=False)
        to_drop = kept_flat[drop_order[: k_cur - keep_num]]
        mask[to_drop] = False
    elif k_cur < keep_num:
        unkept = torch.where(~mask)[0]
        if len(unkept) > 0:
            add_scores = image_attention[unkept]
            add_order = torch.argsort(add_scores, descending=True)
            need = min(keep_num - k_cur, len(unkept))
            mask[unkept[add_order[:need]]] = True

    return mask


def holov_keep_mask(
    patch_tokens: torch.Tensor,
    attention: torch.Tensor,
    num_groups: int,
    keep_ratio: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        patch_tokens: [BS, N, D]
        attention: [BS, N] non-negative saliency (e.g. L2 norm proxy)
        num_groups: number of contiguous segments along the flattened patch sequence (HoloV "num_patches")
        keep_ratio: target fraction of tokens to keep

    Returns:
        mask: [BS, N] bool — True = keep original feature at that position
    """
    bs, n_tok, _ = patch_tokens.shape
    keep_num = max(1, int(n_tok * keep_ratio))
    masks = []
    for b in range(bs):
        m = _holov_single(patch_tokens[b], attention[b], num_groups, keep_num, eps=eps)
        masks.append(m)
    return torch.stack(masks, dim=0)


def _fill_idw(
    feats: torch.Tensor,
    mask: torch.Tensor,
    patch_h: int,
    patch_w: int,
    power: float = 2.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Fill pruned positions by inverse-distance weighting (Shepard) in 2D patch grid coordinates.
    Each hole gets a convex combination of *all* kept tokens, weighted by 1 / dist^power.

    feats: [N, D], mask [N] True = keep original feature at that linear index.
    """
    device = feats.device
    dtype = feats.dtype
    n, d = feats.shape
    assert n == patch_h * patch_w
    if mask.all():
        return feats
    if not mask.any():
        return feats.mean(dim=0, keepdim=True).expand_as(feats)

    rows = (torch.arange(n, device=device) // patch_w).to(dtype=dtype)
    cols = (torch.arange(n, device=device) % patch_w).to(dtype=dtype)
    rc = torch.stack([rows, cols], dim=1)

    kept_idx = torch.where(mask)[0]
    hole_idx = torch.where(~mask)[0]
    rc_k = rc[kept_idx]
    rc_h = rc[hole_idx]
    f_k = feats[kept_idx]

    dist = torch.cdist(rc_h, rc_k, p=2.0)
    w = 1.0 / (dist.pow(power) + eps)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w_norm = w / denom
    filled = w_norm @ f_k

    out = feats.clone()
    out[hole_idx] = filled.to(dtype)
    return out


def _fill_nearest(
    feats: torch.Tensor,
    mask: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    """feats [N, D], mask [N] True=keep source for copy; holes filled from nearest kept in 2D grid."""
    device = feats.device
    n = feats.shape[0]
    assert n == patch_h * patch_w
    if mask.all():
        return feats
    if not mask.any():
        return feats.mean(dim=0, keepdim=True).expand_as(feats)

    rows = torch.arange(n, device=device) // patch_w
    cols = torch.arange(n, device=device) % patch_w
    filled_idx = torch.where(mask)[0]
    hole_idx = torch.where(~mask)[0]
    rc_f = torch.stack([rows[filled_idx], cols[filled_idx]], dim=1).float()
    rc_h = torch.stack([rows[hole_idx], cols[hole_idx]], dim=1).float()
    dist = torch.cdist(rc_h, rc_f, p=1)
    nearest = dist.argmin(dim=1)
    out = feats.clone()
    out[hole_idx] = feats[filled_idx[nearest]]
    return out


def scatter_patch_tokens_dense(
    patch_tokens: torch.Tensor,
    mask: torch.Tensor,
    patch_h: int,
    patch_w: int,
    fill_mode: FillMode = "zero",
) -> torch.Tensor:
    """
    Apply keep mask; holes filled deterministically. patch_tokens [BS, N, D], mask [BS, N].
    """
    bs, n, d = patch_tokens.shape
    assert n == patch_h * patch_w
    out = patch_tokens.clone()
    m = mask.unsqueeze(-1)

    if fill_mode == "zero":
        out = torch.where(m, patch_tokens, torch.zeros_like(patch_tokens))
    elif fill_mode == "mean":
        mean = patch_tokens.mean(dim=1, keepdim=True)
        out = torch.where(m, patch_tokens, mean)
    elif fill_mode == "nearest":
        for b in range(bs):
            row = out[b].clone()
            kept = patch_tokens[b].clone()
            filled = _fill_nearest(kept, mask[b], patch_h, patch_w)
            out[b] = torch.where(m[b], patch_tokens[b], filled)
    elif fill_mode in ("idw", "interpolate"):
        for b in range(bs):
            out[b] = _fill_idw(patch_tokens[b], mask[b], patch_h, patch_w)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")
    return out


def apply_holov_scatter_to_aggregated_list(
    aggregated_tokens_list: List[torch.Tensor],
    patch_start_idx: int,
    images: torch.Tensor,
    patch_size: int,
    num_groups: int,
    keep_ratio: float,
    fill_mode: FillMode,
    layer_indices: List[int],
) -> List[torch.Tensor]:
    """
    For selected transformer layers, replace patch token segment with HoloV-scattered dense grid features.
    Special tokens (camera, register) are unchanged.

    Args:
        aggregated_tokens_list: outputs from Aggregator (length = depth)
        patch_start_idx: index where patch tokens start
        images: [B, S, 3, H, W]
        patch_size: ViT patch size (e.g. 14)
        num_groups: HoloV segment count along flattened patches
        keep_ratio: fraction of patch tokens to keep
        fill_mode: "zero" | "mean" | "nearest" | "idw" | "interpolate" (IDW on 2D patch grid)
        layer_indices: which layers to rewrite (e.g. DPT intermediate indices)
    """
    b, s, _, h, w = images.shape
    patch_h, patch_w = h // patch_size, w // patch_size
    n_patch = patch_h * patch_w

    out_list = list(aggregated_tokens_list)

    for li in layer_indices:
        if li < 0 or li >= len(aggregated_tokens_list):
            continue
        t = aggregated_tokens_list[li]
        p_len = t.shape[2] - patch_start_idx
        if p_len != n_patch:
            raise ValueError(
                f"Layer {li}: expected {n_patch} patch tokens (patch_h*patch_w), got {p_len}. "
                f"Check image size vs patch_size."
            )

        patch = t[:, :, patch_start_idx:, :].reshape(b * s, n_patch, -1)
        attn = attention_proxy_l2_norm(patch.detach())
        mask = holov_keep_mask(patch, attn, num_groups=num_groups, keep_ratio=keep_ratio)
        patch_new = scatter_patch_tokens_dense(
            patch, mask, patch_h, patch_w, fill_mode=fill_mode
        )
        patch_new = patch_new.reshape(b, s, n_patch, -1)

        new_t = t.clone()
        new_t[:, :, patch_start_idx:, :] = patch_new
        out_list[li] = new_t

    return out_list
