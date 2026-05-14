"""对比 tile_test vs nonpad_test 的验证效果"""
import os, sys
sys.path.insert(0, '/root/WaterMamba')

import torch
import numpy as np
from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils.options import parse
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# 加载配置
opt = parse('WaterMamba.yml', is_train=False)
opt['is_train'] = True
opt['dist'] = False
opt['val']['save_img'] = True

# 创建模型并加载 checkpoint
model = ImageCleanModel(opt)
model.net_g.eval()

# 找最新 checkpoint
ckpt_dir = 'experiments/WaterMamba/models'
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
print(f"Checkpoints: {ckpts[-3:]}")
latest = ckpts[-1]
ckpt_path = os.path.join(ckpt_dir, latest)
print(f"Loading: {ckpt_path}")

state = torch.load(ckpt_path, map_location='cpu')
# 适配不同 key 格式
if 'params' in state:
    model.net_g.load_state_dict(state['params'], strict=False)
elif 'params_ema' in state:
    model.net_g.load_state_dict(state['params_ema'], strict=False)
else:
    model.net_g.load_state_dict(state, strict=False)

model.net_g.eval()
model.net_g.to(model.device)

# 构建 dataloader
from basicsr.data.paired_image_dataset import Dataset_PairedImage
from torch.utils.data import DataLoader

val_cfg = dict(opt['datasets']['val'])
val_cfg['phase'] = 'val'
val_cfg['scale'] = opt['scale']
val_cfg['name'] = 'ValSet'

dataset = Dataset_PairedImage(val_cfg)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# 取第一张图测试
for data in dataloader:
    lq = data['lq'].to(model.device)
    gt = data['gt'].to(model.device)
    img_name = os.path.basename(data['lq_path'][0])
    print(f"\n=== Testing: {img_name}  shape: {lq.shape} ===")
    
    # 方法1: nonead_test (整图推理)
    model.lq = lq
    model.nonpad_test()
    out_full = model.output.detach().cpu()
    print(f"  整图推理完成, output shape: {out_full.shape}")
    
    # 方法2: tile_test (分块推理)
    model.tile_test(lq)
    out_tile = model.output.detach().cpu()
    print(f"  分块推理完成, output shape: {out_tile.shape}")
    
    # 转为 [0,1] float 计算 PSNR/SSIM
    gt_cpu = gt.detach().cpu()
    
    psnr_full = calculate_psnr(out_full, gt_cpu, crop_border=0, test_y_channel=False)
    ssim_full = calculate_ssim(out_full, gt_cpu, crop_border=0, test_y_channel=False)
    
    psnr_tile = calculate_psnr(out_tile, gt_cpu, crop_border=0, test_y_channel=False)
    ssim_tile = calculate_ssim(out_tile, gt_cpu, crop_border=0, test_y_channel=False)
    
    print(f"  --- 整图推理 ---  PSNR: {psnr_full:.4f}  SSIM: {ssim_full:.4f}")
    print(f"  --- 分块推理 ---  PSNR: {psnr_tile:.4f}  SSIM: {ssim_tile:.4f}")
    print(f"  tile - full diff:  ΔPSNR={psnr_tile-psnr_full:.4f}  ΔSSIM={ssim_tile-ssim_full:.4f}")
    
    # 同时验证原验证流程的实际输出 (tensor2img 之后)
    from basicsr.utils.img_util import tensor2img
    sr_full_np = tensor2img([out_full], rgb2bgr=True)
    sr_tile_np = tensor2img([out_tile], rgb2bgr=True)
    gt_np = tensor2img([gt_cpu], rgb2bgr=True)
    
    psnr_full_img = calculate_psnr(sr_full_np, gt_np, crop_border=0, test_y_channel=False)
    ssim_full_img = calculate_ssim(sr_full_np, gt_np, crop_border=0, test_y_channel=False)
    psnr_tile_img = calculate_psnr(sr_tile_np, gt_np, crop_border=0, test_y_channel=False)
    ssim_tile_img = calculate_ssim(sr_tile_np, gt_np, crop_border=0, test_y_channel=False)
    
    print(f"\n  --- 经过 tensor2img 后 ---")
    print(f"  --- 整图推理 ---  PSNR: {psnr_full_img:.4f}  SSIM: {ssim_full_img:.4f}")
    print(f"  --- 分块推理 ---  PSNR: {psnr_tile_img:.4f}  SSIM: {ssim_tile_img:.4f}")
    
    break

print("\n✅ 测试完成")
