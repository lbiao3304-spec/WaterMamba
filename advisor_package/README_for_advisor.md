# WaterMamba：基于 Mamba 的水下图像增强

> **学生**：[你的姓名]  
> **课题方向**：Mamba 状态空间模型 + 小波变换在水下图像增强中的应用  
> **代码框架**：PyTorch + CUDA 11.8, RTX 4070 Laptop (8GB)

---

## 一、课题概述

WaterMamba 是一个基于 **Mamba（State Space Model，状态空间模型）** 的端到端水下图像增强网络。核心创新是用 **SS2D（四向选择性扫描）** 替代传统的 Self-Attention，在线性复杂度 O(N) 下实现全局感受野建模，显著降低计算量。同时采用 **离散小波变换（DWT）做无损下采样**，完整保留边缘/纹理等高频信息。

模型在 200 对水下图像上训练 30 万次迭代，在 90 对未见过的验证集上达到：
- **PSNR: 21.37 dB**
- **SSIM: 0.814**

---

## 二、模型架构

### 整体结构：U-Net 编码器-解码器（4 级）

```
输入 (256×256×3)
  ↓ OverlapPatchEmbed (3×3 Conv)
Encoder L1: SCOSSBlock × N1
  ↓ Downsample_DWT (小波下采样)
Encoder L2: SCOSSBlock × N2 (dim×2, 128×128)
  ↓ Downsample_DWT
Encoder L3: SCOSSBlock × N3 (dim×4, 64×64)
  ↓ Downsample_DWT
Latent: SCOSSBlock × N4 (dim×8, 32×32)  ← 最深特征
  ↓ Upsample_IDWT (小波上采样) + Skip Connection
Decoder L3 → Decoder L2 → Decoder L1
  ↓ Conv 1×1 + 残差连接
输出增强图像
```

### 核心模块：SCOSSBlock

```
LayerNorm → SS2D (4 方向选择性扫描) → DropPath → 残差连接
```

**SCOSSBlock** 是 WaterMamba 的基本构建块，等价于 Transformer 中的 Attention Block，但将 Self-Attention 替换为 **SS2D**。

### 核心算子：SS2D（4-Directional Selective Scan）

SS2D 在特征图的 **4 个方向**上做选择性状态空间扫描：

1. **原图方向** → 从左到右、从上到下建模全局上下文  
2. **转置方向** → 交换 H/W 轴，捕捉转置视角下的依赖  
3. **翻转方向** → 反向扫描，获取逆向序列信息  
4. **翻转转置** → 组合翻转和转置，覆盖 4 个方向的全局感受野

4 个方向的输出相加融合，**以 O(N) 线性复杂度实现类似 Self-Attention 的全局建模能力**。

### 关键设计：DWT/IDWT 无损下采样

传统 U-Net 用 Strided Conv / PixelUnshuffle 做下采样，会丢失高频细节。本模型采用 **Haar 小波变换（DWT）**：

- **DWT 分解**：将特征图分解为 LL（低频）+ LH（水平高频）+ HL（垂直高频）+ HH（对角高频）四个频带
- **1×1 卷积融合**：将 4 个频带融合为 2 倍通道数，实现 2× 下采样，同时保留所有频率信息
- **IDWT 逆变换**：上采样时从融合特征中重建 4 个频带再做小波逆变换，完整恢复空间信息

---

## 三、实验结果

### 定量结果（ValSet，90 对未见图像）

| 指标 | 数值 |
|------|------|
| **PSNR ↑** | 21.37 dB |
| **SSIM ↑** | 0.814 |

> 注：TrainSet 上 PSNR 约 30.18 dB，仅反映训练集重建质量；学术评估以 ValSet 指标为准，因其反映真实泛化能力。

### 可视化对比

详见 `results/visualizations/` 目录（8 对 LQ → 增强 → GT 对比图）。

每张对比图包含三个部分：
- **左侧**：原始水下图像（偏蓝绿色、模糊、低对比度）
- **中间**：WaterMamba 增强结果（色彩校正、去模糊、对比度提升）  
- **右侧**：Ground Truth 参考图像

### 训练曲线

TensorBoard 日志位于 `logs/tensorboard/`，可运行以下命令查看：

```bash
tensorboard --logdir logs/tensorboard/WaterMamba/
```

完整的训练日志位于 `logs/train_WaterMamba_20260509_113701.log`。

---

## 四、运行方式

### 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.0.1 + CUDA 11.8 |
| GPU | RTX 4070 Laptop (8GB VRAM) |
| Conda 环境 | `vmamba` |

### 关键依赖

```
pytorch_wavelets  (小波变换)
mamba-ssm         (SS2D selective scan CUDA kernel)
timm              (DropPath / LayerNorm)
einops            (张量重排)
```

### 测试命令

```bash
source /home/right_limit/miniconda3/etc/profile.d/conda.sh
conda activate vmamba
cd /path/to/WaterMamba
PYTHONPATH=/path/to/WaterMamba python basicsr/test.py -opt WaterMamba.yml
```

### 训练命令

```bash
PYTHONPATH=/path/to/WaterMamba python basicsr/train.py -opt WaterMamba.yml
```

**重要**：本项目**不能** `pip install basicsr`，必须通过 `PYTHONPATH` 导入本地 `basicsr/` 源码。

---

## 五、后续创新方向

详见 **`innovation_proposal_v2.md`**（推荐）或 `innovation_proposal.md`（初版）。

v2 版本的核心思路是**将小波变换深度融入 Mamba 的序列建模机制**，包含以下创新点：

### 核心创新：频带自适应 SS2D（Wavelet-Adaptive SS2D）
对 DWT 分解后的 LL/LH/HL/HH 四个频带分别使用**独立权重的 SS2D**，低频侧重色偏校正（大 d_state）、高频侧重纹理恢复（小 d_state），实现"分而治之"的退化建模。

### 核心创新：跨频带交互建模（Cross-Band Interaction）
在频带独立建模基础上引入**轻量交叉门控**：LL 感知高频纹理后调整增强策略，HH 感知低频色偏后约束噪声放大。与简单空间门控不同，这是频带间的物理约束嵌入。

### 辅助增强：可学习小波基
DWT 前加 Depthwise Conv 学习频带预投影，为 Haar 分解增加任务自适应性。

**创新1+2 联合设计为 WaveletMambaBlock**，预期 PSNR 提升 0.6~1.0 dB。

---

## 六、文件清单

```
advisor_package/
├── README_for_advisor.md               ← 本文件（导师必读）
├── innovation_proposal_v2.md           ← 创新点方案 v2（推荐，交叉门控设计）
├── innovation_proposal.md              ← 创新点方案 v1（初版，含小波门控）
│
├── code/
│   ├── WaterMamba_arch.py              ← 模型完整架构代码（核心）
│   ├── WaterMamba.yml                  ← 训练/测试超参数配置
│   ├── train.sh                        ← 训练启动脚本
│   └── .clinerules                     ← 环境配置规则
│
├── results/
│   └── visualizations/                 ← 8 对 LQ→增强→GT 对比图
│
├── logs/
│   ├── train_WaterMamba_20260509_113701.log  ← 完整训练日志
│   └── tensorboard/                    ← TensorBoard 训练曲线
│
└── weights/
    └── net_g_300000.pth                ← 30万迭代最终模型权重 (~350 MB)
```

---

## 七、已有工作基础总结

| 项目 | 完成度 |
|------|--------|
| 模型架构设计与实现 | ✅ 完成 |
| SS2D 四向选择性扫描 CUDA 封装 | ✅ 完成 |
| DWT/IDWT 无损下采样集成 | ✅ 完成 |
| 训练框架搭建（basicsr 修改版） | ✅ 完成 |
| 200 对训练集 + 90 对验证集准备 | ✅ 完成 |
| 30 万迭代完整训练 | ✅ 完成 |
| ValSet 测试与定量评估 | ✅ 完成 |
| 可视化对比图生成 | ✅ 完成 |
| **下一阶段：创新点实现与论文撰写** | 🔲 待开展 |
