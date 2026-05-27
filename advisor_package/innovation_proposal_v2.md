# 创新点方案 v2：Mamba 结合小波变换用于水下图像增强

> **课题**：WaterMamba — 基于 State Space Model 的水下图像增强  
> **现有基础**：SS2D 四向选择性扫描 + DWT/IDWT 无损下采样 U-Net  
> **优化版本**：经评审后重新梳理创新点，将"小波门控"替换为更深入的"跨频带交互建模"

---

## 一、研究动机（精简版）

### 1.1 核心矛盾

Mamba 的 SS2D 在**统一特征空间**中做扫描——低频的色偏和光照退化、高频的模糊和噪声退化，被同一个状态空间模型统一处理。

小波变换天然将图像分解为 LL（低频颜色/光照）、LH/HL（中频边缘）、HH（高频噪声/纹理）。**不同频带对应不同物理退化机制**，但当前 WaterMamba 的 DWT 仅在编码器-解码器间做无损下采样，**未利用小波的频率解耦能力**。

### 1.2 创新逻辑

```
DWT 分解 → 4 个独立频带 → 每个频带用独立的 SS2D 建模
                                    ↓
                       频带间通过 Cross-Attention 交换信息
                                    ↓
                       IDWT 重建 → 恢复为统一特征
```

两个核心创新均围绕"频带感知 + SS2D"：

| 创新 | 本质 | 解决什么问题 |
|------|------|-------------|
| **频带自适应 SS2D** | 4 个频带 → 4 个独立 SS2D（权重不共享） | "不同退化用不同模型" |
| **跨频带交互建模** | 频带间通过轻量 Cross-Gating 通信 | "频带间互相指导增强方向" |

---

## 二、核心创新 1：频带自适应 SS2D（Wavelet-Adaptive SS2D）

### 2.1 动机

LL 频带的色偏/光照退化和 HH 频带的噪声/纹理退化，物理机制完全不同。用一个 SS2D 统一处理所有频率，相当于强迫模型在低维状态空间中"同时解决色偏和去噪"——状态空间负担过重。

### 2.2 方法

```
原始特征 x [B, C, H, W]
    ↓ DWT 分解
    ├─ LL: [B, C, H/2, W/2]  →  SS2D_LL (d_state=32, 低频专用)
    ├─ LH: [B, C, H/2, W/2]  →  SS2D_LH (d_state=16, 水平边缘)
    ├─ HL: [B, C, H/2, W/2]  →  SS2D_HL (d_state=16, 垂直边缘)
    └─ HH: [B, C, H/2, W/2]  →  SS2D_HH (d_state=8,  高频噪声/纹理)
    ↓ 4 个输出拼接 & 可学习加权融合
    ↓ IDWT 重建
增强后的特征 y [B, C, H, W]
```

**关键设计**：
- LL 分配两倍 state dimension（d_state=32），因为色偏/光照建模需要更大的状态容量
- HH 分配一半 state dimension（d_state=8），高频分量稀疏，小状态足以捕捉
- 融合权重 `α = softmax([w_LL, w_LH, w_HL, w_HH])` 可学习，让网络自适应决定不同层对不同频带的侧重

### 2.3 创新点

| 方面 | 分析 |
|------|------|
| **物理可解释性** | 频带分离=退化分离，低频处理色偏、高频处理纹理/噪声 |
| **计算效率** | DWT后分辨率减半，SS2D FLOPs 不变（4×C×HW/4 = C×HW） |
| **消融实验** | 共享SS2D(baseline) vs 独立SS2D vs 独立SS2D+d_state差异化 |
| **可扩展** | Haar可替换为Daubechies小波，LL可进一步分解（多级DWT） |

### 2.4 消融实验链

| 配置 | 描述 | 预计提升 |
|------|------|---------|
| Baseline | 原 WaterMamba（DWT 仅做下采样） | — |
| S-1 | 4频带独立SS2D（d_state 全部=16） | +0.2~0.4 dB |
| S-2 | S-1 + d_state差异化（32/16/16/8） | +0.3~0.5 dB |
| S-3 | S-2 + 可学习频带融合权重 | +0.4~0.6 dB |
| **Full** | S-3 + Cross-Band Interaction（见下） | +0.6~1.0 dB |

---

## 三、核心创新 2：跨频带交互建模（Cross-Band Interaction）

> **设计决策**：交叉门控而非简单空间门控。空间门控（方向一的原始方案）仅告诉模型"注意边缘区域"，跨频带交互告诉模型"LL 的信息如何指导 LH，HH 的信息如何约束 LL"，技术深度更足。

### 3.1 动机

频带独立建模（创新1）忽略了频带间的物理约束：
- **LL → LH/HL**：色偏严重的区域，边缘增强应更保守（避免放大偏色）
- **HH → LL**：噪声密集区域，光照增强应避免过度提升（防止噪声放大）
- **LH/HL → HH**：强边缘位置的 HH 分量是真实纹理，弱边缘位置的 HH 是噪声

### 3.2 方法：频带交叉门控（Cross-Band Gating）

每个频带在 SS2D 扫描前，先接收其他频带的**上下文摘要**作为门控信号：

```
频带 LL 的处理流程：
  LH_summary = GlobalAvgPool(LH) → Linear(C→C) → 调制系数 γ_LH→LL
  HL_summary = GlobalAvgPool(HL) → Linear(C→C) → 调制系数 γ_HL→LL
  HH_summary = GlobalAvgPool(HH) → Linear(C→C) → 调制系数 γ_HH→LL
  
  LL_gated = LL ⊙ σ(γ_LH→LL + γ_HL→LL + γ_HH→LL)
  LL_out  = SS2D_LL(LL_gated)
```

**设计要点**：
- 用 `GlobalAvgPool` 将其他频带压缩为 **channel-wise 向量**，不引入额外空间计算
- 交叉门控发生在 SS2D **之前**（调制扫描输入），而非 SS2D 之后（调制输出）
- 门控向量通过线性投影+相加，参数量级仅为 O(C²)，相比原模型增加 < 1%

### 3.3 与"空间门控"的对比

| 方案 | 机制 | 输入 | 效果 |
|------|------|------|------|
| 原方向二（小波门控） | 高频幅度 → Sigmoid → 空间 mask | 仅高频分量自己 | "注意边缘" |
| **本文方案（Cross-Band Gating）** | LL→LH, HH→LL 等交叉信息 | 所有其他频带 | 频带间互相指导增强方向 |

**Cross-Band Gating 的优势**：LL 的门控来自 LH/HL/HH，意味着 LL 的状态扫描可以"感知"到高频纹理的存在，从而在平滑区域和纹理区域调整增强策略。这不是简单的空间注意力，而是**跨频带的物理约束嵌入**。

### 3.4 联合设计：创新1 + 创新2 = Wavelet-Mamba Block

将两个创新整合为一个模块：

```python
class WaveletMambaBlock(nn.Module):
    """整合频带自适应SS2D + 跨频带交互"""
    def __init__(self, dim, d_state=16):
        # 4个独立SS2D
        self.ss2d_ll = SS2D(dim, d_state*2)
        self.ss2d_lh = SS2D(dim, d_state)
        self.ss2d_hl = SS2D(dim, d_state)
        self.ss2d_hh = SS2D(dim, d_state//2)
        # 跨频带门控投影（12个方向：4频带 × 3其他频带）
        self.cross_gate = CrossBandGating(dim)
        # 频带自适应融合
        self.band_fusion = AdaptiveBandFusion(dim)

    def forward(self, x):
        # DWT分解
        ll, lh, hl, hh = self.dwt(x)
        # 跨频带门控（注入其他频带信息）
        ll_g, lh_g, hl_g, hh_g = self.cross_gate(ll, lh, hl, hh)
        # 独立SS2D扫描
        ll_out = self.ss2d_ll(ll_g)
        lh_out = self.ss2d_lh(lh_g)
        hl_out = self.ss2d_hl(hl_g)
        hh_out = self.ss2d_hh(hh_g)
        # 自适应融合 + IDWT重建
        return self.idwt(self.band_fusion(ll_out, lh_out, hl_out, hh_out))
```

---

## 四、可学习小波基（辅助实验，不进核心创新）

### 4.1 动机

固定 Haar 小波的分解基是"通用"的——等权重的低通/高通滤波。水下图图像中，色偏（偏蓝绿）和非均匀光照可能更适合"任务特定"的分解基。

### 4.2 方案（简化版）

```python
# 不是完全参数化DWT，而是用 Depthwise Conv 学习频带投影
class LearnablePreDWT(nn.Module):
    """DWT前添加可学习的频带预投影"""
    def __init__(self, dim):
        super().__init__()
        self.pre_proj = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.post_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.pre_proj(x)     # 可学习的频带预投影
        ll, lh, hl, hh = dwt(x)  # 标准 Haar DWT
        return self.post_proj(ll), lh, hl, hh
```

此方案规避了"直接学习小波基→重构失败"的风险，同时赋予模型一定灵活度。

### 4.3 定位

- 作为 **消融实验的一个子项**，而非独立创新点
- 对比：固定 Haar vs 可学习 Pre-DWT vs Daubechies-4
- 论文中占 1 页实验分析即可

---

## 五、整体架构：Wavelet-Mamba U-Net

```
输入 (256×256×3)
  ↓ OverlapPatchEmbed
  ┌──────────────────────────────────────────────┐
  │  Encoder L1: WaveletMambaBlock × N1           │  ← 创新1+2
  │    ↓ DWT 无损下采样 (2×)                      │
  │  Encoder L2: WaveletMambaBlock × N2           │  ← 创新1+2
  │    ↓ DWT 无损下采样 (4×)                      │
  │  Encoder L3: WaveletMambaBlock × N3           │  ← 创新1+2
  │    ↓ DWT 无损下采样 (8×)                      │
  │  Latent:     WaveletMambaBlock × N4           │  ← 创新1+2
  │    ↑ IDWT 上采样 + Skip Connection             │
  │  Decoder L3 → Decoder L2 → Decoder L1         │
  └──────────────────────────────────────────────┘
  ↓ Conv 1×1 + 残差连接
输出增强图像
```

**相比于原 WaterMamba 的改动**：
- 原：`SCOSSBlock(SS2D_unified)` → 新：`WaveletMambaBlock(SS2D_perband + CrossGating)`
- 原 DWT 仅做下采样，新设计在每个 WaveletMambaBlock 内部也做 DWT→频带处理→IDWT
- 架构改动行数 < 100 行，属于"即插即用"的模块替换

---

## 六、实验计划

### 6.1 消融实验矩阵

| 实验 | 频带自适应 SS2D | Cross-Band Gating | 可学习 Pre-DWT | 预期 PSNR |
|------|:---:|:---:|:---:|---|
| Baseline (WaterMamba) | | | | 21.37 |
| E1 | ✅ | | | 21.57 |
| E2 | | ✅ | | 21.47 |
| E3 | ✅ | ✅ | | 21.77 |
| E4 | ✅ | ✅ | ✅ | 21.97 |
| E5 (E3 + 更多iter) | ✅ | ✅ | | 22.1+ |

### 6.2 可视化分析（为论文提供有力的 Figure）

| 可视化内容 | 展示什么 |
|-----------|---------|
| 频带激活热力图 | 证明 LL/LH/HL/HH 确实学到了不同退化 |
| 交叉门控权重 | 展示哪些频带间的信息交换最活跃 |
| 模块替换前后对比 | SS2D统一扫描 vs 频带解耦扫描的效果差异 |
| 失败案例分析 | 哪些类型的水下退化仍难以处理 → 为未来工作铺垫 |

### 6.3 对比方法

| 类别 | 方法 |
|------|------|
| Mamba | 原WaterMamba、VMamba |
| Transformer | Uformer、Restormer |
| CNN | UIEC^2-Net、Water-Net |
| 传统 | UDCP、Fusion-based |

---

## 七、时间规划

| 阶段 | 任务 | 时间 |
|------|------|------|
| **Sprint 1** | 实现 WaveletMambaBlock（创新1+2），跑通单卡训练 | 3-4 周 |
| **Sprint 2** | 完整消融实验（E1-E4）、可学习Pre-DWT实验 | 2-3 周 |
| **Sprint 3** | 多数据集泛化验证（EUVP/UIEB）、Comparison实验 | 2-3 周 |
| **Sprint 4** | 论文撰写（Intro/Method/Experiment/Figure） | 4 周 |
| **Buffer** | 返修、补充实验 | 2 周 |

---

## 八、核心文献

1. Gu A, Dao T. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. arXiv:2312.00752, 2023.
2. Zhu L, et al. **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model**. ICML 2024.
3. Liu Y, et al. **VMamba: Visual State Space Model**. NeurIPS 2024.
4. Mallat S. **A Wavelet Tour of Signal Processing**. Academic Press, 1999.
5. Liu P, et al. **Multi-level Wavelet-CNN for Image Restoration**. CVPRW 2018.
6. Guo M-H, et al. **Attention Mechanisms in Computer Vision: A Survey**. Computational Visual Media, 2022.（交叉门控 vs 注意力背景）
