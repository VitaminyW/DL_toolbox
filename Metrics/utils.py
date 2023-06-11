from torchmetrics.functional import structural_similarity_index_measure as torch_ssim
from torchmetrics.functional import peak_signal_noise_ratio as torch_psnr
import torch

def cal_psnr(gts:list,yps:list):
    gts = gts[0] # abs value B Frame H W
    yps = yps[0] # complex value B Frame H W
    B,Frame,H,W = gts.shape
    # gts = torch.abs(gts)
    yps = torch.abs(yps)
    gts = gts.reshape(B*Frame,H,W)
    yps = yps.reshape(B*Frame,H,W)
    # 归一化
    gts /= (gts.max(dim=-1)[0].max(dim=-1)[0]).unsqueeze(-1).unsqueeze(-1)
    yps /= (yps.max(dim=-1)[0].max(dim=-1)[0]).unsqueeze(-1).unsqueeze(-1)
    gts = gts.unsqueeze(1)
    yps = yps.unsqueeze(1)
    mean_psnr = torch_psnr(yps,gts,data_range=1)
    return mean_psnr

def cal_ssim(gts:list,yps:list):
    gts = gts[0] # abs value B Frame H W
    yps = yps[0] # complex value B Frame H W
    B,Frame,H,W = gts.shape
    # gts = torch.abs(gts)
    yps = torch.abs(yps)
    gts = gts.reshape(B*Frame,H,W)
    yps = yps.reshape(B*Frame,H,W)
    # 归一化
    gts /= (gts.max(dim=-1)[0].max(dim=-1)[0]).unsqueeze(-1).unsqueeze(-1)
    yps /= (yps.max(dim=-1)[0].max(dim=-1)[0]).unsqueeze(-1).unsqueeze(-1)
    gts = gts.unsqueeze(1)
    yps = yps.unsqueeze(1)
    mean_ssim = torch_ssim(yps, gts,data_range = 1)
    return mean_ssim