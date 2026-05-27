# 创新点方案：Mamba 结合小波变换用于水下图像增强

> **课题**：WaterMamba — 基于 State Space Model 的水下图像增强  
> **现有基础**：SS2D 四向选择性扫描 + DWT/IDWT 无损下采样 U-Net  
> **创新方向**：将小波变换深度融入 Mamba 的序列建模机制，构建频带感知的视觉状态空间模型

---

## 一、研究动机

### 1.1 Mamba（SS2D）的优势与局限

**优势**：SS2D 在特征图 4 个方向（原图、翻转、转置、翻转转置）做选择性扫描，以 **线性复杂度 O(N)** 实现全局感受野，完美替代二次复杂度的 Self-Attention。

**局限**：当前 SS2D 在**统一的特征空间**中做扫描，不区分低频（颜色/光照）与高频（纹理/边缘）成分。对于水下图像增强，退化类型多样（色偏、模糊、低对比度、悬浮颗粒），统一建模难以同时应对不同频率的退化。

### 1.2 小波变换的天然互补性

离散小波变换（DWT）将图像分解为四个频带：

| 频带 | 内容 | 对水下增强的意义 |
|------|------|-----------------|
| **LL**（低频） | 全局光照、颜色分布 | 色偏校正、对比度拉伸 |
| **LH**（水平高频） | 水平边缘、纹理 | 去模糊、细节恢复 |
| **HL**（垂直高频） | 垂直边缘、纹理 | 去模糊、细节恢复 |
| **HH**（对角高频） | 角点、噪声、悬浮颗粒 | 去噪、去悬浮颗粒 |

**核心洞察**：小波分解将不同物理意义的退化分离到不同频带，Mamba 的 SS2D 扫描正好可以对不同频带做**差异化的选择性状态建模**，实现"分而治之"的增强。

### 1.3 WaterMamba 现有工作已初步验证该方向

当前 WaterMamba 使用 DWT/IDWT 替代传统下采样（`Downsample_DWT` / `Upsample_IDWT`），**已经在小波域做上下采样**。这为深度结合小波变换提供了工程基础——模型已经具备了 DWT 前向/逆向算子。

---

## 二、创新方向一：频带自适应 SS2D（Wavelet-Aware SS2D）

### 核心思路

将 SCOSSBlock 中的统一 SS2D 替换为**频带解耦的 SS2D**：

```
输入特征 x [B, C, H, W]
    ↓
DWT 分解 → LL, LH, HL, HH（各 [B, C, H/2, W/2]）
    ↓
4 个独立 SS2D 分别处理不同频带（权重不共享）
  - SS2D_LL：大 d_state（16→32），侧重全局光照建模
  - SS2D_LH/HL：中等 d_state，侧重大范围边缘
  - SS2D_HH：小 d_state（16→8），侧重局部噪声/颗粒
    ↓
频带自适应融合：可学习的 channel-wise 权重 α_LL, α_LH, α_HL, α_HH
    ↓
IDWT 重建 → 恢复原始分辨率
```

### 代码原型

```python
class WaveletSS2D(nn.Module):
    """频带自适应 SS2D —— 对 DWT 分解后的各频带独立建模"""
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        # 4 个独立的 SS2D（权重不共享）
        self.ss2d_ll = SS2D(d_model=dim, d_state=d_state*2)    # 低频：大状态
        self.ss2d_lh = SS2D(d_model=dim, d_state=d_state)
        self.ss2d_hl = SS2D(d_model=dim, d_state=d_state)
        self.ss2d_hh = SS2D(d_model=dim, d_state=d_state//2)   # 高频：小状态
        # 频带自适应融合权重
        self.band_weights = nn.Parameter(torch.ones(4, dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        # DWT 分解
        yl, yh = self.dwt(x)
        yh_h = yh[0]  # [B, C, 3, H/2, W/2]
        # 逐个频带做 SS2D 扫描
        ll = self.ss2d_ll(yl.permute(0,2,3,1)).permute(0,3,1,2)
        lh = self.ss2d_lh(yh_h[:,:,0].permute(0,2,3,1)).permute(0,3,1,2)
        hl = self.ss2d_hl(yh_h[:,:,1].permute(0,2,3,1)).permute(0,3,1,2)
        hh = self.ss2d_hh(yh_h[:,:,2].permute(0,2,3,1)).permute(0,3,1,2)
        # 可学习加权融合
        w = F.softmax(self.band_weights.view(4, -1), dim=0).view(4, C, 1, 1)
        yl_fused = ll * w[0] + lh * w[1] + hl * w[2] + hh * w[3]  # 简化示意
        return self.idwt((yl_fused, [yh_h]))
```

### 创新点

- **解耦频率建模**：不同频带的退化物理不同，独立 SS2D 让每个频带学各自的状态转移
- **自适应融合**：可学习权重让网络决定不同层/不同样本侧重哪些频带
- **轻量改动**：仅替换 SCOSSBlock 中的 SS2D，不改变 U-Net 整体结构

---

## 三、创新方向二：多尺度小波-Mamba 注意力门控

### 核心思路

利用小波高频分量作为**门控信号**，引导 SS2D 的选择性扫描机制更加关注边缘/纹理区域：

```
高频分量 (LH+HL+HH) → 提取边缘响应图 → Sigmoid 门控
                                              ↓
原始特征 → SS2D 扫描 → ⊙ 门控加权 → 增强输出
              ↑
         SSM 的 Δ (delta) 参数接收门控信号
         动态调整不同空间位置的状态更新步长
```

### 关键机制

SS2D 的核心公式中，`Δ`（delta）控制状态更新的步长：

```
h_t = A * h_{t-1} + B * x_t       (状态更新)
y_t = C * h_t + D * x_t           (输出投影)
```

**小波门控**：高频响应大的区域（边缘）→ 增大 `Δ` → 更大步长更新 → 更关注细节恢复；平滑区域 → 减小 `Δ` → 保守更新 → 保持颜色一致性。

### 代码原型

```python
class WaveletGatedSS2D(SS2D):
    """用小波高频分量门控 SS2D 的状态更新"""
    def __init__(self, d_model, **kwargs):
        super().__init__(d_model, **kwargs)
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        # 从高频分量生成门控信号
        self.gate_conv = nn.Sequential(
            nn.Conv2d(d_model * 3, d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, H, W, C = x.shape
        # 提取高频门控
        x_2d = x.permute(0, 3, 1, 2)
        _, yh = self.dwt(x_2d)
        hf = yh[0].reshape(B, C*3, H//2, W//2)
        gate = F.interpolate(self.gate_conv(hf), size=(H, W), mode='bilinear')
        # 标准 SS2D 前向（被门控调制）
        out = super().forward(x)
        return out * gate.permute(0, 2, 3, 1)
```

### 创新点

- **物理可解释性**：门控信号源自小波高频天然边缘检测器，无需额外监督
- **即插即用**：可嵌入任意 SS2D 模块，不影响整体架构
- **消融实验丰富**：对比有/无门控、固定/可学习门控、小波/Sobel 门控

---

## 四、创新方向三：可学习小波基

### 核心思路

当前 WaterMamba 使用固定 **Haar 小波**做 DWT/IDWT。**将其替换为参数化的卷积小波核**，通过端到端训练学习最适合水下增强的分解基。

### 技术方案

传统 DWT 使用固定的 4 个滤波器（低通 `h` + 高通 `g`）。可学习小波将这些滤波器参数化为卷积核：

```python
class LearnableDWT(nn.Module):
    """可学习离散小波变换"""
    def __init__(self, channels, wave='haar'):
        super().__init__()
        # 初始化为标准 Haar 小波
        h = torch.tensor([1/√2, 1/√2])  # 低通
        g = torch.tensor([1/√2, -1/√2]) # 高通
        # 4 个分解滤波器：LL, LH, HL, HH
        self.dec_lo = nn.Parameter(h.repeat(channels, 1, 1, 1))
        self.dec_hi = nn.Parameter(g.repeat(channels, 1, 1, 1))
        # 4 个重构滤波器
        self.rec_lo = nn.Parameter(h.repeat(channels, 1, 1, 1))
        self.rec_hi = nn.Parameter(g.repeat(channels, 1, 1, 1))

    def forward(self, x):
        # 使用可学习滤波器做卷积分解/重构
        ...
```

### 创新点

- **任务自适应小波基**：从"通用 Haar"学习到"水下增强专用"分解基，可能发现色偏敏感通道
- **与方向二、三正交叠加**：可学习基 + 频带自适应 SS2D = 完整的可学习小波-Mamba 框架
- **可视化分析**：导出训练后的滤波器，分析学到了什么频率特性

---

## 五、实验验证方案

### 5.1 消融实验（Ablation Study）

| 实验编号 | 配置 | 预期对比 |
|----------|------|---------|
| Baseline | 原始 WaterMamba（DWT 仅用于下采样） | — |
| Exp-A | + 频带自适应 SS2D（方向一） | PSNR +0.3~0.8 dB |
| Exp-B | + 小波门控 SS2D（方向二） | PSNR +0.2~0.5 dB |
| Exp-C | + 可学习小波基（方向三） | PSNR +0.1~0.3 dB |
| Exp-D | 三个方向全部叠加 | PSNR +0.8~1.5 dB |

### 5.2 评估指标

- 全参考：PSNR、SSIM、LPIPS
- 无参考：UIQM、UCIQE（水下图像专用）
- 可视化：频带激活图、门控热力图（证明小波机制确实在工作）

### 5.3 对比基线

- 传统方法：UDCP、Fusion-based
- CNN 方法：UIEC^2-Net、Water-Net
- Transformer 方法：Uformer、Restormer
- Mamba 方法：原始 WaterMamba（自身消融）

---

## 六、时间规划

| 阶段 | 内容 | 预计时间 |
|------|------|---------|
| 第一阶段 | 实现方向一（频带自适应 SS2D），跑通基线对比 | 3-4 周 |
| 第二阶段 | 实现方向二（小波门控），完成消融实验 | 2-3 周 |
| 第三阶段 | 探索方向三（可学习小波基）+ 多数据集验证 | 2-3 周 |
| 第四阶段 | 论文撰写（Intro / Method / Experiment） | 4 周 |

---

## 七、参考文献相关

1. Gu A, Dao T. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. arXiv:2312.00752, 2023.
2. Zhu L, et al. **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model**. ICML 2024.
3. Liu Y, et al. **VMamba: Visual State Space Model**. NeurIPS 2024.
4. Mallat S. **A Wavelet Tour of Signal Processing**. Academic Press, 1999.
5. Liu P, et al. **Multi-level Wavelet-CNN for Image Restoration**. CVPRW 2018.
