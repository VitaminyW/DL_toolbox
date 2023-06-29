import torch
from typing import Union
from torchmetrics.functional import structural_similarity_index_measure as torch_ssim
from .utils import Loss_Function


def inv_ssim_loss(input:torch.Tensor,target:torch.Tensor,**argv):
    """计算1-SSIM损失

    Args:
        input (torch.Tensor): MRI重建得到的复数图
        target (torch.Tensor): MRI的实数真解图

    Returns:
        _type_: _description_
    """
    gts = target # abs value B Frame H W
    yps = input # complex value B Frame H W
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

class INV_SSIM_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(inv_ssim_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

    
    
    