import torch
from torchmetrics.functional import structural_similarity_index_measure as torch_ssim

def inv_ssim_loss(gts:list,yps:list):
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
    ssims = torch_ssim(yps, gts,data_range = 1,reduction=None)
    return (1-ssims).mean()
    
    
    