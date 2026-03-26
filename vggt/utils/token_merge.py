# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Training-free token merge/unmerge pipeline for VGGT's Global Attention layers.
# Combines FastVGGT-style three-part partitioning, block-wise similarity matching,
# and residual-compensated geometry-aware unmerge.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MergeState:
    """Bookkeeping produced by merge(), consumed by unmerge()."""

    anchor_global_idx: torch.Tensor  # [N_anchor] long
    dst_global_idx: torch.Tensor  # [N_dst] long
    src_global_idx: torch.Tensor  # [N_src] long
    src_to_dst_local: torch.Tensor  # [N_src] long — maps each src to its matched dst (local index)
    merge_residual: torch.Tensor  # [B, N_src, C] — pre-merge (src - matched_dst)
    N_orig: int
    N_anchor: int
    N_dst: int


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


def partition_tokens(
    S: int,
    P: int,
    patch_start_idx: int,
    device: torch.device,
    salient_scores: Optional[torch.Tensor] = None,
    protect_first_frame: bool = True,
    salient_ratio: float = 0.1,
    merge_ratio: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three-way partition of the flattened S*P token sequence.

    Returns (anchor_mask, dst_mask, src_mask) each of shape [S*P] bool.

    * anchor — never merged (special tokens, first-frame patches, salient tokens)
    * dst    — destination tokens for merge averaging
    * src    — source tokens that will be folded into the nearest dst
    """
    N = S * P

    anchor_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # 1) Special tokens (camera + register) of every frame → anchor
    frame_offsets = torch.arange(S, device=device) * P  # [S]
    for off in range(patch_start_idx):
        anchor_mask[frame_offsets + off] = True

    # 2) First frame entirely → anchor
    if protect_first_frame:
        anchor_mask[:P] = True

    # 3) Salient patch tokens among the remaining candidates
    candidate_mask = ~anchor_mask
    candidate_idx = candidate_mask.nonzero(as_tuple=False).squeeze(1)

    if candidate_idx.numel() > 0 and salient_ratio > 0:
        n_salient = max(1, int(candidate_idx.numel() * salient_ratio))
        if salient_scores is not None:
            cand_scores = salient_scores[candidate_idx]
            k = min(n_salient, cand_scores.numel())
            _, top_local = torch.topk(cand_scores, k=k)
            anchor_mask[candidate_idx[top_local]] = True
        else:
            stride = max(1, candidate_idx.numel() // n_salient)
            salient_local = torch.arange(0, candidate_idx.numel(), stride, device=device)[:n_salient]
            anchor_mask[candidate_idx[salient_local]] = True

    # 4) Split remaining into dst / src via stride sampling
    remaining_idx = (~anchor_mask).nonzero(as_tuple=False).squeeze(1)

    dst_mask = torch.zeros(N, dtype=torch.bool, device=device)
    src_mask = torch.zeros(N, dtype=torch.bool, device=device)

    if remaining_idx.numel() > 0:
        n_dst = max(1, int(remaining_idx.numel() * (1.0 - merge_ratio)))
        stride = max(1, remaining_idx.numel() // n_dst)
        dst_local = torch.arange(0, remaining_idx.numel(), stride, device=device)[:n_dst]
        is_dst = torch.zeros(remaining_idx.numel(), dtype=torch.bool, device=device)
        is_dst[dst_local] = True
        dst_mask[remaining_idx[is_dst]] = True
        src_mask[remaining_idx[~is_dst]] = True

    return anchor_mask, dst_mask, src_mask


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_tokens(
    tokens: torch.Tensor,
    pos: Optional[torch.Tensor],
    anchor_mask: torch.Tensor,
    dst_mask: torch.Tensor,
    src_mask: torch.Tensor,
    S: int,
    P: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], MergeState]:
    """Merge src tokens into their most similar dst token (average pooling).

    Args:
        tokens: [B, N, C]  where N = S*P
        pos:    [B, N, 2]  or None (RoPE positions)
        anchor_mask, dst_mask, src_mask: [N] bool
        S, P: sequence length and tokens-per-frame

    Returns:
        merged_tokens: [B, M, C]  M = N_anchor + N_dst
        merged_pos:    [B, M, 2]  or None
        state:         MergeState
    """
    B, N, C = tokens.shape
    device = tokens.device

    anchor_idx = anchor_mask.nonzero(as_tuple=False).squeeze(1)
    dst_idx = dst_mask.nonzero(as_tuple=False).squeeze(1)
    src_idx = src_mask.nonzero(as_tuple=False).squeeze(1)

    N_anchor = anchor_idx.numel()
    N_dst = dst_idx.numel()
    N_src = src_idx.numel()

    if N_src == 0:
        merged = torch.cat([tokens[:, anchor_idx], tokens[:, dst_idx]], dim=1)
        m_pos = torch.cat([pos[:, anchor_idx], pos[:, dst_idx]], dim=1) if pos is not None else None
        state = MergeState(
            anchor_global_idx=anchor_idx,
            dst_global_idx=dst_idx,
            src_global_idx=src_idx,
            src_to_dst_local=torch.zeros(0, dtype=torch.long, device=device),
            merge_residual=torch.zeros(B, 0, C, device=device, dtype=tokens.dtype),
            N_orig=N, N_anchor=N_anchor, N_dst=N_dst,
        )
        return merged, m_pos, state

    # --- similarity matching (batch-mean, per-frame for scalability) ---
    src_tokens = tokens[:, src_idx]  # [B, N_src, C]
    dst_tokens = tokens[:, dst_idx]  # [B, N_dst, C]

    src_to_dst_local = _match_src_to_dst(
        src_tokens.mean(dim=0), dst_tokens.mean(dim=0),
        src_idx, dst_idx, S, P,
    )  # [N_src]

    # --- residual for unmerge ---
    matched_dst_pre = dst_tokens[:, src_to_dst_local]  # [B, N_src, C]
    merge_residual = src_tokens - matched_dst_pre  # [B, N_src, C]

    # --- average merge: fold src into dst ---
    merged_dst = dst_tokens.clone()
    counts = torch.ones(B, N_dst, 1, device=device, dtype=tokens.dtype)

    expand_map = src_to_dst_local.unsqueeze(0).unsqueeze(-1).expand(B, N_src, C)
    merged_dst.scatter_add_(1, expand_map, src_tokens)

    count_inc = torch.ones(B, N_src, 1, device=device, dtype=tokens.dtype)
    counts.scatter_add_(1, src_to_dst_local.unsqueeze(0).unsqueeze(-1).expand(B, N_src, 1), count_inc)
    merged_dst = merged_dst / counts.clamp(min=1)

    merged_tokens = torch.cat([tokens[:, anchor_idx], merged_dst], dim=1)

    merged_pos = None
    if pos is not None:
        merged_pos = torch.cat([pos[:, anchor_idx], pos[:, dst_idx]], dim=1)

    state = MergeState(
        anchor_global_idx=anchor_idx,
        dst_global_idx=dst_idx,
        src_global_idx=src_idx,
        src_to_dst_local=src_to_dst_local,
        merge_residual=merge_residual,
        N_orig=N, N_anchor=N_anchor, N_dst=N_dst,
    )
    return merged_tokens, merged_pos, state


def _match_src_to_dst(
    src_avg: torch.Tensor,
    dst_avg: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    S: int,
    P: int,
) -> torch.Tensor:
    """Per-frame cosine-similarity matching of src → dst.

    For each src, only considers dst tokens within the same frame (± 1 frame
    for border tokens). Falls back to all-dst if a frame has no dst.

    Returns: [N_src] long tensor of dst-local indices.
    """
    device = src_idx.device
    N_src = src_idx.numel()
    N_dst = dst_idx.numel()

    src_frame = src_idx // P  # [N_src]
    dst_frame = dst_idx // P  # [N_dst]

    src_norm = F.normalize(src_avg, dim=-1)  # [N_src, C]
    dst_norm = F.normalize(dst_avg, dim=-1)  # [N_dst, C]

    matches = torch.zeros(N_src, dtype=torch.long, device=device)

    # Small input: full similarity matrix
    if N_src * N_dst < 5_000_000:
        sim = src_norm @ dst_norm.T
        return sim.argmax(dim=1)

    # Large input: per-frame matching
    for f in range(S):
        s_sel = (src_frame == f).nonzero(as_tuple=False).squeeze(1)
        if s_sel.numel() == 0:
            continue

        # dst in frames [f-1, f, f+1]
        d_sel = ((dst_frame >= max(f - 1, 0)) & (dst_frame <= min(f + 1, S - 1))).nonzero(as_tuple=False).squeeze(1)
        if d_sel.numel() == 0:
            d_sel = torch.arange(N_dst, device=device)

        sim = src_norm[s_sel] @ dst_norm[d_sel].T  # [n_s, n_d]
        local_best = sim.argmax(dim=1)
        matches[s_sel] = d_sel[local_best]

    return matches


# ---------------------------------------------------------------------------
# Unmerge
# ---------------------------------------------------------------------------


def unmerge_tokens(
    merged_output: torch.Tensor,
    state: MergeState,
    residual_weight: float = 0.3,
) -> torch.Tensor:
    """Geometry-aware unmerge with residual compensation.

    merged_output: [B, M, C]  where M = N_anchor + N_dst
    Returns: [B, N_orig, C]
    """
    B = merged_output.shape[0]
    C = merged_output.shape[-1]
    N = state.N_orig
    device = merged_output.device
    dtype = merged_output.dtype

    full = torch.zeros(B, N, C, device=device, dtype=dtype)

    # anchor tokens — direct placement
    if state.N_anchor > 0:
        full[:, state.anchor_global_idx] = merged_output[:, : state.N_anchor]

    # dst tokens — direct placement
    if state.N_dst > 0:
        dst_out = merged_output[:, state.N_anchor : state.N_anchor + state.N_dst]
        full[:, state.dst_global_idx] = dst_out
    else:
        dst_out = merged_output[:, :0]  # empty

    # src tokens — residual-compensated recovery
    if state.src_global_idx.numel() > 0:
        matched_dst_out = dst_out[:, state.src_to_dst_local]  # [B, N_src, C]
        src_recovered = matched_dst_out + residual_weight * state.merge_residual.to(dtype)
        full[:, state.src_global_idx] = src_recovered

    return full


def smooth_unmerged_features(
    feat_2d: torch.Tensor,
    src_mask_2d: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Optional Gaussian smoothing at recovered (src) positions only.

    feat_2d:     [BS, C, H, W]
    src_mask_2d: [BS, 1, H, W] bool
    """
    k = kernel_size
    device = feat_2d.device
    dtype = feat_2d.dtype
    x = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    gauss_1d = torch.exp(-0.5 * (x ** 2) / (sigma ** 2))
    kernel_2d = (gauss_1d[:, None] * gauss_1d[None, :]).to(dtype)
    kernel_2d = kernel_2d / kernel_2d.sum()

    C = feat_2d.shape[1]
    weight = kernel_2d.view(1, 1, k, k).expand(C, 1, k, k)
    smoothed = F.conv2d(feat_2d, weight, padding=k // 2, groups=C)
    return torch.where(src_mask_2d, smoothed, feat_2d)


# ---------------------------------------------------------------------------
# AVGGT-style KV subsampling  (quality-preserving acceleration)
# ---------------------------------------------------------------------------
# Key insight from AVGGT (arxiv 2512.02541):
#   - Keep ALL Q tokens → every position gets its own updated representation
#   - Subsample K/V → reduces attention from O(N²d) to O(NMd)
#   - No unmerge needed → zero information loss at output positions
#   - Add mean-fill token for global context preservation


def select_kv_indices(
    S: int,
    P: int,
    patch_start_idx: int,
    device: torch.device,
    kv_ratio: float = 0.25,
    protect_first_frame: bool = True,
) -> torch.Tensor:
    """Select which token indices to keep for K/V subsampling.

    Strategy: keep all special tokens, first-frame tokens, and stride-sample
    remaining patch tokens for uniform spatial coverage.

    Returns: [M] long tensor of selected indices within [0, S*P).
    """
    N = S * P
    keep = torch.zeros(N, dtype=torch.bool, device=device)

    # Always keep special tokens (camera + register) of every frame
    frame_offsets = torch.arange(S, device=device) * P
    for off in range(patch_start_idx):
        keep[frame_offsets + off] = True

    # Keep entire first frame
    if protect_first_frame:
        keep[:P] = True

    # Stride-sample remaining patch tokens
    remaining = (~keep).nonzero(as_tuple=False).squeeze(1)
    if remaining.numel() > 0:
        target = max(1, int(N * kv_ratio) - keep.sum().item())
        if target > 0 and target < remaining.numel():
            stride = max(1, remaining.numel() // target)
            selected = torch.arange(0, remaining.numel(), stride, device=device)[:target]
            keep[remaining[selected]] = True
        else:
            keep[remaining] = True

    return keep.nonzero(as_tuple=False).squeeze(1)


def run_block_kv_subsample(
    block,
    x: torch.Tensor,
    pos: Optional[torch.Tensor],
    kv_idx: torch.Tensor,
    use_mean_fill: bool = True,
) -> torch.Tensor:
    """Run a Block with K/V subsampled — all Q positions are updated.

    This is the core of the AVGGT-style acceleration:
    - QKV projection on ALL N tokens
    - RoPE applied to full Q, full K (before subsampling)
    - K, V subsampled to M tokens (+ optional mean token)
    - Attention: Q[N] × K[M]ᵀ → output [N, C]  (full resolution)
    - MLP on ALL N tokens (full resolution)

    Args:
        block: a Block module (norm1 → attn → ls1 → norm2 → mlp → ls2)
        x:     [B, N, C]
        pos:   [B, N, 2] or None
        kv_idx: [M] long — which token positions to keep for K/V
        use_mean_fill: prepend a mean-K/mean-V token for global context

    Returns: [B, N, C]  — same shape as input, no information loss
    """
    attn_mod = block.attn

    # --- Attention with KV subsampling + residual ---
    x_normed = block.norm1(x)
    B, N, C = x_normed.shape

    qkv = attn_mod.qkv(x_normed).reshape(
        B, N, 3, attn_mod.num_heads, attn_mod.head_dim,
    ).permute(2, 0, 3, 1, 4)  # [3, B, h, N, d]
    q, k, v = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

    if attn_mod.rope is not None and pos is not None:
        q = attn_mod.rope(q, pos)
        k = attn_mod.rope(k, pos)

    # Subsample K/V AFTER RoPE so positional encoding is correct
    k_sub = k[:, :, kv_idx]  # [B, h, M, d]
    v_sub = v[:, :, kv_idx]  # [B, h, M, d]

    if use_mean_fill:
        k_mean = k.mean(dim=2, keepdim=True)  # [B, h, 1, d]
        v_mean = v.mean(dim=2, keepdim=True)
        k_sub = torch.cat([k_sub, k_mean], dim=2)  # [B, h, M+1, d]
        v_sub = torch.cat([v_sub, v_mean], dim=2)

    # Attention: Q[N] @ K[M]ᵀ → [N,M] → softmax → @ V[M] → [N, d]
    attn_out = F.scaled_dot_product_attention(
        q, k_sub, v_sub,
        dropout_p=attn_mod.attn_drop.p if attn_mod.training else 0.0,
    )
    attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
    attn_out = attn_mod.proj(attn_out)
    attn_out = attn_mod.proj_drop(attn_out)

    x = x + block.ls1(attn_out)

    # --- MLP (full resolution, no subsampling) ---
    x = x + block.ls2(block.mlp(block.norm2(x)))

    return x


# ---------------------------------------------------------------------------
# Merge-index cache
# ---------------------------------------------------------------------------


class MergeCache:
    """Caches token partition across layers to avoid redundant computation."""

    def __init__(self, refresh_layers: Optional[List[int]] = None):
        self.refresh_layers = set(refresh_layers or [0, 4, 8, 11, 14, 17, 20, 23])
        self._partition: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def should_refresh(self, layer_idx: int) -> bool:
        return self._partition is None or layer_idx in self.refresh_layers

    def get_partition(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self._partition

    def set_partition(self, partition: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        self._partition = partition

    def reset(self) -> None:
        self._partition = None
