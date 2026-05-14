# WaterMamba 项目踩坑记录 & 历史决策

> 记录本项目的关键踩坑经验、架构理解和历史修改决策，供后续维护参考。
> 部分重要条目可考虑合并到 `.clinerules` 中。

---

## 1. 环境配置

### 1.1 PYTHONPATH 必须手动设置

**问题：** 本项目**不能**使用 `pip install basicsr` 或 `pip install -e .`，否则会与系统 torch 版本冲突或触发 native 编译失败。

**正确用法：**
```bash
PYTHONPATH=/home/right_limit/WaterMamba python basicsr/test.py -opt WaterMamba.yml
```

**原理：** `basicsr/` 是项目的本地修改版，需要直接通过 PYTHONPATH 导入源码，而非作为 pip 包安装。

**状态：** ✅ 已写入 `.clinerules`

### 1.2 Conda 环境

- 环境名：`vmamba`
- Python 3.10
- PyTorch 2.0.1 + CUDA 11.8
- GPU：RTX 4070 Laptop (8GB VRAM)

### 1.3 CUDA / PyTorch 版本对应

云端训练环境可能与本地不同。当前本地环境 PyTorch 2.0.1+cu118 加载云端训练的 pth 文件兼容，未出现版本不匹配问题。

---

## 2. 数据集结构

### 2.1 TrainSet (训练集)

| 目录 | 内容 | 数量 |
|------|------|------|
| `dataset/raw-800/` | 水下原始图像 (LQ) | ~200 对 |
| `dataset/reference-800/` | 对应清晰 GT 图像 | ~200 对 |

- 图像文件名一一对应（如 `10_img_.png` ↔ `10_img_.png`）
- 每张均为**独立完整高清大图**（尺寸各异：800×450、640×360、550×363 等），非拼接图
- 训练时通过 `Dataset_PairedImage` 随机裁剪 256×256 patch
- **这是模型训练使用的数据**

### 2.1.1 输出文件命名

test 结果的 TrainSet 目录下每个输入产生两个文件：

| 文件名模式 | 示例 | 内容 |
|-----------|------|------|
| `{img_name}.png` | `362_img_.png` | 模型增强输出 |
| `{img_name}_gt.png` | `362_img__gt.png` | Ground Truth 参考图（清晰原图）|

命名逻辑（来自 `image_restoration_model.py` 的 `nondist_validation`）：
```python
img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
# "362_img_." → img_name = "362_img_"
save_img_path    = f'{img_name}.png'     # 模型输出
save_gt_img_path = f'{img_name}_gt.png'   # GT（注意会有双下划线 __gt）
```

### 2.2 ValSet (验证集)

| 目录 | 内容 | 数量 |
|------|------|------|
| `dataset/raw_val/` | 水下原始验证图像 | 90 对 |
| `dataset/ref_val/` | 对应清晰 GT | 90 对 |

- **模型训练过程中从未见过这些图像**
- 文件名与 TrainSet 不重合（编号不同，如 `50_img_.png`、`3650.png`）
- ValSet 的指标反映模型的**真实泛化能力**

### 2.3 ⚠️ 重要：TrainSet vs ValSet 的指标差异

| 数据集 | PSNR | SSIM | 含义 |
|--------|------|------|------|
| TrainSet (200对) | **30.18** | **0.967** | 训练重建质量（模型见过） |
| ValSet (90对) | **21.37** | **0.814** | 真实泛化能力（模型未见过） |

**结论：**
- PSNR 差距约 9dB，说明存在**明显过拟合**。
- **学术报告时应只使用 ValSet 的指标**，在论文中把 TrainSet 结果当作测试集结果报告是严重错误。
- 训练集上做 test 只能用于检查模型是否成功学习了训练数据，不能反映泛化性能。

---

## 3. 推理（Test）机制

### 3.1 tile_test 滑动窗口分块推理

`basicsr/models/image_restoration_model.py` 中的 `tile_test()` 方法：
- 将大图按 256×256 分块（重叠 32px）
- 每个 tile 独立推理后用**高斯权重融合**缝合
- 好处：8GB 显存也能处理任意尺寸大图

### 3.2 DWT 对齐要求

模型使用了 DWT（离散小波变换），要求输入尺寸能被 8 整除（配置中 `window_size: 8`）。

- `pad_test()` 会在推理前用 `reflect` 模式自动补齐到 8 的倍数
- 不需要手动调整输入图像尺寸

### 3.3 推理流程

```
全图输入 → pad_test 补齐 → tile_test 分块推理 → 高斯融合缝合 →
pad_test 裁掉补齐部分 → 输出与输入同尺寸的结果 → 与 GT 计算 PSNR/SSIM
```

---

## 4. 路径配置问题

### 4.1 云端 vs 本地绝对路径

`WaterMamba.yml` 中的路径使用**绝对路径**，云端和本地不同：

| 配置项 | 云端 | 本地 |
|--------|------|------|
| 根目录 | `/root/WaterMamba/` | `/home/right_limit/WaterMamba/` |
| 数据集 | `/root/WaterMamba/dataset/` | `/home/right_limit/WaterMamba/dataset/` |
| 模型权重 | `/root/WaterMamba/experiments/...` | `/home/right_limit/WaterMamba/experiments/...` |

**每次从云端同步配置后必须修改路径。**

### 4.2 实验结果压缩包

云端训练结果打包为 `experiment_results.tar.gz`，解压后包含：
- `experiments/WaterMamba/models/` — 模型权重 (.pth)
- `experiments/WaterMamba/training_states/` — 训练状态（断点续训用）
- `experiments/WaterMamba/visualization/` — 训练过程中的可视化

解压命令：
```bash
tar -xzf experiment_results.tar.gz
```

---

## 5. 历史修改决策

### 5.1 tile_test batch>1 的 shape bug（已修复）

**问题：** 原始 `tile_test()` 在 batch_size > 1 时，所有 batch 图像的输出被叠加到同一个 `output` tensor 上，导致输出 shape 与输入不匹配。

**修复：** 添加了 batch 维度循环，每个 batch 元素独立处理后再 `torch.cat` 合并。

**修改位置：** `basicsr/models/image_restoration_model.py` → `tile_test()` 方法

### 5.2 旧配置路径修复

旧 `WaterMamba.yml` 指向 `/root/WaterMamba/` 云端路径，已批量改为本地 `/home/right_limit/WaterMamba/`。

**涉及字段：**
- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `datasets.val.dataroot_gt`
- `datasets.val.dataroot_lq`
- `path.pretrain_network_g`

### 5.3 旧结果 R90 清理

项目中发现旧的 `results/R90/` 目录（来自不同配置的测试），已清除。当前唯一有效结果在 `results/WaterMamba/`。

### 5.4 TrainSet test 输出尺寸异常（待修复）

**现象：** 跑完 test 后，TrainSet 输出图片全部变成 518×518 正方形，而原始输入尺寸各异（800×450、640×360 等）。这可能导致了之前误认为是"4 图拼接"数据集。

| 输入文件 | 原始尺寸 | 输出尺寸 |
|----------|---------|----------|
| `100_img_.png` | 800×450 | **518×518** |
| `106_img_.png` | 440×300 | **518×518** |
| `10909.png` | 640×360 | **518×518** |

ValSet 输出正常（与原始尺寸一致）。怀疑 `tile_test()` 分块推理或 `pad_test` 对齐逻辑存在 bug。
---

## 6. 已知问题 & 待改进

### 6.1 过拟合严重

- TrainSet PSNR 30 vs ValSet PSNR 21，差距 9dB
- 可能原因：训练数据量偏小（200对）、数据增强不足（`geometric_augs: false`）
- 建议：增加训练数据、开启数据增强、使用正则化

### 6.2 模型权重路径

当前使用 `net_g_latest.pth`（30万 iter）。另有 `net_g_300000.pth` 和 `net_g_76000.pth` 可作为中间检查点。

### 6.3 结果目录归档

每次运行 test 会自动归档旧结果到 `results/WaterMamba_archived_xxx/`，可能导致磁盘占用增长。建议手动清理不需要的 archived 目录。

---

## 7. 常用命令速查

```bash
# 激活环境 + 测试
source /home/right_limit/miniconda3/etc/profile.d/conda.sh && \
  conda activate vmamba && \
  cd /home/right_limit/WaterMamba && \
  PYTHONPATH=/home/right_limit/WaterMamba python basicsr/test.py -opt WaterMamba.yml

# 解压云端训练结果
tar -xzf experiment_results.tar.gz

# 查看 git 状态
git status
git remote -v
# origin:  https://github.com/lbiao3304-spec/WaterMamba.git
# upstream: https://github.com/Guan-MS/WaterMamba.git
