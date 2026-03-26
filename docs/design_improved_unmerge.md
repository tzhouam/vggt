# VGGT Unmerge 算法改进设计文档

## 1. 背景与动机

### 1.1 VGGT 架构概述

VGGT（Visual Geometry Grounded Transformer）是 Meta 提出的 CVPR 2025 最佳论文，基于 1.2B 参数的 Transformer，通过一次前向推理联合预测相机参数、深度图、点图和 2D track。其核心架构为 **Aggregator**——24 层交替的 Frame Attention 和 Global Attention：

- **Frame Attention**: tokens 以 `[B*S, P, C]` 形状参与帧内 self-attention
- **Global Attention**: tokens 以 `[B, S*P, C]` 形状参与全局跨帧 self-attention

其中 `P = 1(camera) + 4(register) + 1369(patch)` = 1374 tokens/帧。对于 S 帧输入，Global Attention 的时间复杂度为 O((S*P)²d)，是长序列推理的主要瓶颈。

Aggregator 输出层 `[4, 11, 17, 23]` 的中间特征（frame + global 拼接后为 2048-dim）供 DPT Head 消费，用于密集预测（深度、点云等）。

### 1.2 当前 HoloV 实现分析

当前代码库已集成 HoloV 风格的 token 剪枝/散射方案，分为两条路径：

#### 路径 A: 后处理 Scatter（`holov_scatter.py` + `vggt.py`）

在 Aggregator 完整运行后，对输出 `aggregated_tokens_list` 的指定层做后处理：
1. 用 L2-norm 作为 saliency proxy 计算每个 patch token 的重要性
2. 按分组（`num_groups`）分配保留配额（score-weighted allocation）
3. 每组内按 attention proxy top-k 保留 token
4. 被剪枝的位置用 fill mode（zero / mean / nearest / IDW）填充

**问题**: 这条路径不减少 Aggregator 的计算量，只是在 DPT Head 前做近似重建。

#### 路径 B: Aggregator 内部剪枝（`demo_colmap.py :: _run_aggregator_timed`）

在指定 block 后真正减少 token 数，后续层用更少的 tokens 运行：
1. 在 block N 后，平均所有 batch 的 patch tokens 计算 saliency
2. 用 `_holov_single` 生成 keep mask
3. 物理移除被剪枝的 tokens，后续层 P 减小
4. DPT Head 所需的中间特征用 `_scatter_intermediate_back` 扩回全长序列

### 1.3 当前 Unmerge/Scatter 的缺陷

| 缺陷 | 描述 | 影响 |
|------|------|------|
| **信息丢失严重** | 被剪枝 token 直接丢弃，非 merge（合并），pruned 位置只能靠近似填充 | 空间细节退化、深度/点云边界模糊 |
| **填充质量低** | `_scatter_intermediate_back` 用最近邻复制，`scatter_patch_tokens_dense` 的 IDW 仅在 2D 网格上做距离加权 | 几何不连续、高频信息丢失 |
| **无跨层复用** | 路径 A 每层独立计算 mask，路径 B 仅一次剪枝 | 前者重复计算，后者 early layers 信息利用不足 |
| **无 head-wise 感知** | 所有 attention head 共享同一剪枝决策 | 压缩后 head 间多样性丧失（HTTM 论文实证） |
| **Batch 维度串行** | `holov_keep_mask` 和 fill 函数用 Python for 循环遍历 batch | GPU 利用率低 |
| **不区分初始帧** | 所有帧 token 同等对待 | 世界坐标系锚点帧的 token 被削弱，pose 漂移 |

---

## 2. 相关工作关键技术总结

### 2.1 FastVGGT — 三分区 Token Merging

**核心思路**: Merge-Self-Attn-Unmerge 流水线，仅作用于 Global Attention。

**Token 三分区策略**:
1. **初始帧 tokens**: 全部保留为 dst tokens（世界坐标系锚点）
2. **Salient tokens**: 每帧按 stride 保留 ~10% 高显著性 tokens，不参与 merge
3. **Region-based random**: 对剩余 tokens 在 2D patch 网格内区域随机抽样为 dst/src

**Merge**: src 基于 cosine similarity 匹配到最相似的 dst，均值合并 `x_d' = (x_d + x_s) / 2`

**Unmerge**: 简单复制——merged token 的输出复制回所有 constituent tokens

**性能**: 1000 帧 4x 加速，Chamfer Distance 不降反升（长序列缓解了误差累积）

### 2.2 HTTM — 逐 Head 时序 Token Merging

**核心创新**:
- **Head-wise merging**: 每个 attention head 独立决定 merge 策略，保持 head 间特征多样性
- **Temporal reordering**: 将 tokens 按空间-时序块重排，使相似 tokens 聚合在固定大小的 block 内
- **Block-wise matching**: 固定大小的 merge block 将计算从 O(N²) 降为 O(N)
- **Adaptive outlier filtering**: 跨 head 全局预算分配 outlier 保护

**Unmerge**: 按 query 的 merge mapping 复制还原——被 merge 的 src token 的输出直接取其 dst 的输出

**性能**: 7x 加速，negligible performance drop

### 2.3 LiteVGGT — 几何感知缓存 Merging

**核心洞察**:
1. 局部区域 tokens 有内在几何相关性 → 高 similarity → 可 merge
2. Token similarity 跨相邻层稳定 → merge indices 可缓存复用

**技术**:
- Geometry-aware feature map (Ψ_GA): 结合 token variance + gradient 评估几何重要性
- Anchor token 优化选择
- **Cached merge indices**: 每 K 层重新计算一次 merge map，其余层复用

**性能**: 10x 加速，支持 1000 帧场景

---

## 3. 改进方案设计

### 3.1 设计目标

1. **将 prune-scatter 升级为 merge-unmerge**: 真正融合冗余 tokens 的信息而非丢弃
2. **几何感知的 unmerge**: 利用 2D 空间和跨帧时序关系做高质量恢复
3. **跨层 merge index 缓存**: 减少 merge 开销
4. **Head-wise 多样性保持**: 避免 uniform merge 导致的特征坍缩
5. **Training-free**: 不引入新可训练参数，即插即用

### 3.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Aggregator Block Loop                     │
│                                                             │
│  For block_idx in [0..23]:                                  │
│    ┌──────────────┐                                         │
│    │ Frame Attn   │  ← 不做 merge（帧内 P 较小，非瓶颈）     │
│    └──────┬───────┘                                         │
│           │                                                 │
│    ┌──────▼───────────────────────────────────────────────┐  │
│    │          Improved Global Attention Layer              │  │
│    │                                                      │  │
│    │  1. Token Partition (三分区 + geometry-aware)          │  │
│    │  2. Head-wise Merge (block-wise + temporal reorder)   │  │
│    │  3. Compressed Self-Attention                         │  │
│    │  4. Head-wise Geometry-Aware Unmerge ← 核心改进       │  │
│    │  5. Cache merge indices for next K layers             │  │
│    └──────┬───────────────────────────────────────────────┘  │
│           │                                                 │
│    If block_idx in [4,11,17,23]:                            │
│      → Save full-resolution intermediates for DPT Head     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 模块详细设计

#### 模块 1: 改进的 Token 分区策略（`ImprovedTokenPartition`）

结合 FastVGGT 的三分区 + LiteVGGT 的几何感知：

```python
class ImprovedTokenPartition:
    """
    将 S*P tokens 分为三类:
    - anchor: 初始帧全部 tokens + 高几何重要性 tokens (不参与 merge)
    - dst:    每帧 region-based 均匀采样的代表 tokens
    - src:    冗余 tokens，将被 merge 到最相似的 dst
    """

    def partition(self, tokens, frame_lengths, patch_h, patch_w, keep_ratio):
        # Step 1: 初始帧全部标记为 anchor
        anchor_mask[0:frame_len] = True

        # Step 2: 几何感知 saliency 评估
        # 使用 token variance（跨帧方差）+ L2 norm 的混合指标
        # 比纯 L2-norm proxy 更准确
        geo_importance = self._geometry_aware_score(tokens, patch_h, patch_w)

        # Step 3: 每帧 top-k% 标记为 anchor (salient tokens)
        # k 由 keep_ratio 和目标加速比确定
        per_frame_salient = select_topk_per_frame(geo_importance, k_ratio=0.1)
        anchor_mask |= per_frame_salient

        # Step 4: 剩余 tokens 按 region-based random sampling 分为 dst/src
        dst_mask, src_mask = region_random_split(
            ~anchor_mask, patch_h, patch_w, dst_ratio=0.25
        )

        return anchor_mask, dst_mask, src_mask

    def _geometry_aware_score(self, tokens, patch_h, patch_w):
        """LiteVGGT-inspired: token variance + spatial gradient"""
        # 跨帧 variance: 高 variance → token 在不同视角下表现不同 → 重要
        var_score = tokens.var(dim=0)  # [P, C] → [P]

        # 2D 空间 gradient: 与邻居差异大 → 边缘/纹理 → 重要
        token_grid = tokens.reshape(patch_h, patch_w, -1)
        grad_h = (token_grid[1:] - token_grid[:-1]).norm(dim=-1)
        grad_w = (token_grid[:, 1:] - token_grid[:, :-1]).norm(dim=-1)
        grad_score = pad_and_combine(grad_h, grad_w)

        return alpha * var_score.norm(dim=-1) + beta * grad_score
```

#### 模块 2: Head-wise Merge with Temporal Reordering（`HeadwiseTemporalMerge`）

结合 HTTM 的 head-wise + temporal block 思路：

```python
class HeadwiseTemporalMerge:
    """
    在 Global Attention 的 QKV 投影之后、Attention 计算之前:
    1. 时序重排: 将跨帧同一空间位置的 tokens 聚合到 merge blocks
    2. Head-wise merge: 每个 head 独立在 block 内做 cosine-similarity matching
    3. 压缩后的 QKV 用于 attention 计算
    """

    def __init__(self, block_size=1024, spatial_block_size=64, num_temporal_frames=16):
        self.block_size = block_size
        self.n_s = spatial_block_size    # 空间块大小
        self.n_t = num_temporal_frames   # 时序堆叠帧数

    def merge(self, Q, K, V, partition):
        """
        Q, K, V: [h, N, d_head]  (已投影)
        partition: (anchor_mask, dst_mask, src_mask)
        Returns: merged Q', K', V', merge_info (for unmerge)
        """
        h, N, d = Q.shape
        anchor_mask, dst_mask, src_mask = partition

        # 时序重排: 按 (spatial_idx % n_s, frame_idx // n_t) 排序
        reorder_idx = self._temporal_reorder(N, self.n_s, self.n_t)

        merge_info = []
        Q_merged, K_merged, V_merged = [], [], []

        for head_i in range(h):
            q_h = Q[head_i][reorder_idx]
            k_h = K[head_i][reorder_idx]
            v_h = V[head_i][reorder_idx]

            # Block-wise merge for Q
            q_merged, q_map = self._block_merge(
                q_h, dst_mask[reorder_idx], src_mask[reorder_idx]
            )
            # Block-wise merge for K (independent mapping)
            k_merged, k_map = self._block_merge(
                k_h, dst_mask[reorder_idx], src_mask[reorder_idx]
            )
            # V follows K's mapping
            v_merged = self._apply_merge_map(v_h, k_map)

            Q_merged.append(q_merged)
            K_merged.append(k_merged)
            V_merged.append(v_merged)
            merge_info.append({
                'q_map': q_map,
                'k_map': k_map,
                'reorder_idx': reorder_idx,
            })

        return (torch.stack(Q_merged), torch.stack(K_merged),
                torch.stack(V_merged), merge_info)

    def _block_merge(self, tokens, dst_mask, src_mask):
        """在固定大小的 blocks 内做 cosine-similarity matching + 均值合并"""
        N = tokens.shape[0]
        merge_map = {}  # src_idx -> dst_idx

        for block_start in range(0, N, self.block_size):
            block_end = min(block_start + self.block_size, N)
            block_dst = tokens[block_start:block_end][dst_mask[block_start:block_end]]
            block_src = tokens[block_start:block_end][src_mask[block_start:block_end]]

            if len(block_src) == 0 or len(block_dst) == 0:
                continue

            # Cosine similarity
            sim = F.normalize(block_src, dim=-1) @ F.normalize(block_dst, dim=-1).T
            best_match = sim.argmax(dim=1)

            # 均值合并到 dst
            for s_idx, d_idx in enumerate(best_match):
                # 记录 mapping 用于 unmerge
                ...

        return merged_tokens, merge_map
```

#### 模块 3: Geometry-Aware Unmerge（核心改进）— `GeometryAwareUnmerge`

**这是本设计的核心创新点**。现有方法（FastVGGT、HTTM、ToMeSD）的 unmerge 都采用简单复制策略：被 merge 的 src token 直接复制其所属 dst token 的输出。这导致：
- 空间上相邻但不同的位置获得完全相同的特征
- 密集预测任务（深度/点云）的分辨率实质降低
- DPT Head 的多尺度融合得到的是"阶梯状"特征而非平滑过渡

**我们提出 Geometry-Aware Weighted Unmerge**:

```python
class GeometryAwareUnmerge:
    """
    核心改进: 不使用简单复制, 而是基于几何关系的加权恢复。

    思路:
    1. 每个被 merge 的 src token 保存其 merge 前的残差
    2. Unmerge 时, 用加权组合替代简单复制
    3. 权重基于 2D 空间距离 + 特征相似度的联合度量
    """

    def __init__(self, patch_h, patch_w, residual_weight=0.3):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.residual_weight = residual_weight

    def unmerge(self, merged_output, merge_info, original_tokens,
                head_idx, anchor_output):
        """
        Args:
            merged_output: [M, d_head]  attention 后的 merged tokens 输出
            merge_info: dict with q_map, reorder_idx etc.
            original_tokens: [N, d_head]  merge 前的原始 token (用于残差)
            head_idx: 当前 head 索引
            anchor_output: [N_anchor, d_head]  anchor tokens 的 attention 输出

        Returns:
            full_output: [N, d_head]  恢复全长的输出
        """
        N = len(merge_info['reorder_idx'])
        q_map = merge_info['q_map']
        inv_reorder = merge_info['reorder_idx'].argsort()
        full_output = torch.zeros(N, merged_output.shape[-1],
                                  device=merged_output.device,
                                  dtype=merged_output.dtype)

        # Step 1: Anchor tokens 直接放回
        full_output[anchor_indices] = anchor_output

        # Step 2: Dst tokens 直接放回
        full_output[dst_indices] = merged_output[dst_positions]

        # Step 3: Src tokens — Geometry-Aware Weighted Recovery
        for src_idx, dst_idx in q_map.items():
            # 3a. 基础: merged dst 的输出
            base_feature = merged_output[dst_idx]

            # 3b. 残差补偿: merge 前 src 与 dst 的差异
            pre_merge_residual = (original_tokens[src_idx]
                                  - original_tokens[dst_idx])

            # 3c. 邻域加权: 从 2D 网格上的邻近 kept tokens 借用信息
            neighbor_features = self._gather_spatial_neighbors(
                full_output, src_idx, self.patch_h, self.patch_w,
                radius=2
            )
            if neighbor_features is not None:
                spatial_weight = self._compute_spatial_weights(
                    src_idx, neighbor_indices, self.patch_h, self.patch_w
                )
                neighbor_blend = (spatial_weight * neighbor_features).sum(dim=0)
                # 混合: dst 输出 + 残差补偿 + 邻域修正
                full_output[src_idx] = (
                    (1 - self.residual_weight) * base_feature
                    + self.residual_weight * (base_feature + pre_merge_residual)
                    + 0.1 * (neighbor_blend - base_feature)
                )
            else:
                full_output[src_idx] = base_feature + self.residual_weight * pre_merge_residual

        # Step 4: 逆时序重排恢复原始顺序
        full_output = full_output[inv_reorder]
        return full_output
```

**关键创新点解读**:

##### 3.3.1 残差补偿（Residual Compensation）

merge 前，src token `x_s` 和 dst token `x_d` 的差异 `Δ = x_s - x_d` 包含了 src 独有的空间信息。merge 后 attention 输出 `o_d'` 是二者的混合表示。unmerge 时：

```
o_s_recovered = o_d' + α · Δ_projected
```

其中 `Δ_projected` 是残差在 attention 输出空间的投影，α 为衰减系数。这比纯复制 `o_s = o_d'` 能保留更多位置特异性信息。

直觉：两个被 merge 的 tokens 通过 attention 获得了全局上下文，但它们在 merge 前的差异反映了它们各自的局部独特性。通过残差补偿，我们把这种独特性叠加回去。

##### 3.3.2 邻域插值（Spatial Neighbor Interpolation）

对于 DPT Head 需要的中间特征（层 4, 11, 17, 23），空间连续性至关重要。我们利用 2D patch 网格的拓扑：

```
对于 src token 在位置 (r, c):
  找到半径 R 内所有已恢复的 tokens (anchor + dst)
  用 IDW 或 bilinear 权重混合它们的特征
  与残差补偿结果加权融合
```

这保证了 DPT Head 的 ConvTranspose 上采样得到平滑的特征图而非块状伪影。

##### 3.3.3 Head-wise 差异化 Unmerge

HTTM 论文证明了 head-wise merge 保持特征多样性的重要性。我们进一步扩展到 unmerge：

- 每个 head 的 residual_weight α 可以不同
- 基于每个 head 的 attention entropy 自适应调节：entropy 高（注意力分散）→ α 小（更信任 merged 结果）；entropy 低（注意力集中）→ α 大（更依赖残差恢复局部细节）

```python
def adaptive_residual_weight(attn_entropy, base_alpha=0.3):
    """
    attn_entropy: [h]  每个 head 的 attention entropy
    高 entropy → 全局信息充分 → 降低残差权重
    低 entropy → 局部信息更重要 → 提高残差权重
    """
    normalized = (attn_entropy - attn_entropy.min()) / (attn_entropy.max() - attn_entropy.min() + 1e-8)
    return base_alpha * (1.5 - normalized)  # range: [0.5*α, 1.5*α]
```

#### 模块 4: Merge Index 缓存（`MergeIndexCache`）

借鉴 LiteVGGT 的跨层缓存策略：

```python
class MergeIndexCache:
    """
    观察: Token similarity 跨相邻层变化缓慢。
    策略: 每 K 层重新计算 merge indices，中间层复用。

    对于 24 层的 VGGT:
    - 层 0: 完整计算 partition + merge indices
    - 层 1-3: 复用层 0 的 indices
    - 层 4: 重新计算 (DPT intermediate, 需要高质量)
    - 层 5-10: 复用层 4 的 indices
    - 层 11: 重新计算 (DPT intermediate)
    - ...以此类推
    """

    def __init__(self, refresh_layers=None):
        # DPT 中间层 + 额外检查点
        self.refresh_layers = refresh_layers or [0, 4, 8, 11, 14, 17, 20, 23]
        self.cached_partition = None
        self.cached_merge_info = None

    def should_refresh(self, layer_idx):
        return layer_idx in self.refresh_layers

    def get_or_compute(self, layer_idx, tokens, partitioner, merger):
        if self.should_refresh(layer_idx):
            self.cached_partition = partitioner.partition(tokens, ...)
            # merge_info 在 actual merge 时产生, 但 partition 可以缓存
        return self.cached_partition
```

### 3.4 向量化实现优化

当前实现的一个重要问题是 batch 维度的 Python 循环。改进方案应全面向量化：

```python
# 当前: Python 循环 (慢)
for b in range(bs):
    m = _holov_single(patch_tokens[b], attention[b], ...)
    masks.append(m)

# 改进: 全向量化 (快)
def batch_merge(Q, K, V, partition_masks):
    """
    Q: [B, h, N, d_head]
    完全用 einsum / bmm / scatter 操作, 无 Python 循环
    """
    # 批量 cosine similarity
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)

    # 基于 partition mask 的 scatter-based merge
    # 使用 torch.scatter_reduce 实现批量均值合并
    merged = torch.zeros_like(dst_tokens)
    counts = torch.zeros(n_dst, device=Q.device)
    merged.scatter_reduce_(0, match_indices, src_tokens, reduce='mean')

    return merged
```

### 3.5 DPT Head 友好的中间层 Unmerge

DPT Head 从层 [4, 11, 17, 23] 提取特征并 reshape 为 `[B*S, C, patch_h, patch_w]` 的 2D 特征图。因此这些层的 unmerge 质量直接决定最终输出质量。

**特殊处理**:

```python
def dpt_aware_unmerge(merged_inter, merge_info, layer_idx, patch_h, patch_w):
    """
    对 DPT intermediate layers 使用增强 unmerge:
    1. 标准 geometry-aware unmerge 恢复到 [B, S, P, C]
    2. 额外的 2D 空间平滑: 将恢复后的 patch tokens reshape 到
       [B*S, C, patch_h, patch_w]，用轻量 3x3 depthwise conv 平滑
    3. 仅对 src 位置施加平滑 (dst 和 anchor 保持原样)
    """
    # 标准 unmerge
    full_tokens = geometry_aware_unmerge(merged_inter, merge_info)

    # Reshape 到 2D
    feat_2d = full_tokens[:, :, patch_start_idx:, :].reshape(
        B * S, patch_h, patch_w, C
    ).permute(0, 3, 1, 2)

    # 轻量平滑 (仅对 pruned 位置)
    smoothed = depthwise_conv_3x3(feat_2d)
    src_mask_2d = src_mask.reshape(B * S, 1, patch_h, patch_w)
    feat_2d = torch.where(src_mask_2d, smoothed, feat_2d)

    return feat_2d
```

注意：这里的 3x3 depthwise conv 可以用固定的 Gaussian kernel（不需要训练），保持 training-free 属性。

---

## 4. 对比分析

| 维度 | 当前 HoloV | FastVGGT | HTTM | LiteVGGT | **本方案** |
|------|-----------|----------|------|----------|-----------|
| Merge 方式 | 纯剪枝（丢弃） | 均值合并 | 均值合并 | 均值合并 | 均值合并 |
| Unmerge 方式 | 最近邻/IDW 填充 | 简单复制 | 简单复制 | 简单复制 | **残差补偿 + 邻域插值** |
| Head 感知 | 无 | 无 | 有 | 无 | **有** |
| 初始帧保护 | 无 | 有 | 无 | 无 | **有** |
| 几何感知 | L2-norm proxy | Norm-based | Cosine sim | Variance+Gradient | **Variance+Gradient+Spatial** |
| 跨层缓存 | 无 | 无 | 无 | 有 | **有** |
| Merge 开销 | O(N) | O(N²) per layer | O(N) block-wise | O(N/K) cached | **O(N) block-wise + cached** |
| DPT 适配 | IDW 后处理 | 无特殊处理 | 无特殊处理 | 无特殊处理 | **2D 空间平滑** |
| 向量化 | 部分 | 部分 | 是 | 是 | **是** |
| Training-free | 是 | 是 | 是 | 需 fine-tune | **是** |

---

## 5. 实现计划

### Phase 1: 基础 Merge-Unmerge 框架（替换 prune-scatter）

**文件变更**:

| 文件 | 变更 |
|------|------|
| `vggt/utils/token_merge.py` | 新增：ImprovedTokenPartition, HeadwiseTemporalMerge |
| `vggt/utils/token_unmerge.py` | 新增：GeometryAwareUnmerge, MergeIndexCache |
| `vggt/models/aggregator.py` | 修改：在 `_process_global_attention` 中集成 merge-attn-unmerge |
| `vggt/models/vggt.py` | 修改：替换 holov_scatter 调用为新的 merge pipeline |
| `vggt/utils/holov_scatter.py` | 保留：作为 fallback / baseline 对比 |

### Phase 2: 优化与向量化

- 消除所有 batch 维度的 Python 循环
- 使用 `torch.scatter_reduce` 实现批量 merge
- Triton kernel 实现 block-wise similarity 计算（可选）

### Phase 3: 质量验证与调参

- 在 ScanNet-50、7Scenes、NRGBD 上测试
- 消融实验：残差权重 α、邻域半径 R、缓存刷新频率
- 与 FastVGGT、HTTM、LiteVGGT 数值对比

---

## 6. 预期效果

| 指标 | 当前 HoloV 50% keep | 预期改进 |
|------|---------------------|---------|
| 加速比 (100 帧) | ~1.5x | ~3-4x |
| 加速比 (1000 帧) | ~2x | ~6-8x |
| 深度 MAE 退化 | ~15-20% | <5% |
| Chamfer Distance 退化 | ~10-15% | <3% |
| Pose ATE 退化 | ~8-12% | <2% |
| 显存峰值 (100 帧) | ~28 GB | ~18 GB |

**加速来源分解**:
- Global Attention token 减少 → 二次复杂度收益（主要）
- Merge index 缓存 → 减少 merge 计算开销
- 向量化 → 消除 Python 循环开销
- Block-wise matching → 避免全局 O(N²) similarity 计算

---

## 7. 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| 残差补偿引入噪声 | α 自适应 + clamp 残差幅度 |
| 邻域插值计算开销 | 仅对 DPT intermediate layers 启用，限制 radius=2 |
| Head-wise merge 增加 bookkeeping | Block-wise 限制 scope + merge info 用紧凑格式存储 |
| 缓存的 merge indices 跨层不稳定 | 在 DPT layers 强制刷新 + 监控 similarity drift |
| 极端场景（无重叠视角） | Fallback: 当 mean similarity < threshold 时禁用 merge |

---

## 8. 后续扩展

1. **可学习残差投影矩阵**: 如果允许轻量 fine-tuning，可以学习一个将 merge 前残差投影到 attention 输出空间的线性层
2. **Multi-scale Merge**: 浅层高 merge ratio，深层低 merge ratio（浅层特征更冗余）
3. **Track Head 适配**: 当前设计聚焦 DPT Head（深度/点云），后续需验证对 Track Head 的影响
4. **与 FlashAttention-3 集成**: 将 block-wise merge 嵌入到 FlashAttention 的 tile 循环中，进一步减少内存往返
5. **FP8 量化兼容**: 确保 merge/unmerge 操作在 FP8 精度下数值稳定
