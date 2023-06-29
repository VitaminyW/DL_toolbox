"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""

import  matplotlib.pyplot  as plt
from matplotlib.colors import Normalize
from .sos import sos
import torch

def imshow(image,cmap='gray',coil_dim=-1,norm=None,colorbar=False):
    """
    用于显示图像
    :param image: 待显示的图像,支持MRI多通道
    :param cmap: 颜色模式 default:gray
    :param coil_dim: 通道所在维度 default:-1
    :param norm: 是否设置显示范围 default:None
    :param colorbar: 是否显示颜色范围条 default:False
    """
    if len(image.shape) > 3:
        raise ValueError('该函数不支持维数大于3的数据')
    elif len(image.shape) <= 3:
        if isinstance(image,torch.Tensor):
            image = image.detach().cpu().numpy()
        image_sos = sos(image,coil_dim,False)
        if colorbar:
            plt.colorbar()
        plt.imshow(image_sos,cmap=cmap,norm=norm) # type: ignore
        plt.show()
        
def imwrite(image,save_path,cmap='gray',coil_dim=-1,norm=None,colorbar=False):
    """
    用于保存图像
    :param image: 待显示的图像,支持MRI多通道
    :param save_path: 图像保存路径
    :param cmap: 颜色模式 default:gray
    :param coil_dim: 通道所在维度 default:-1
    :param norm: 是否设置显示范围 default:None
    :param colorbar: 是否显示颜色范围条 default:False
    """
    if len(image.shape) > 3:
        raise ValueError('该函数不支持维数大于3的数据')
    elif len(image.shape) <= 3:
        image_sos = sos(image,coil_dim,False) if len(image.shape) == 3 else image
        if isinstance(image_sos,torch.Tensor):
            image_sos = image_sos.detach().cpu()
        if colorbar:
            plt.colorbar()
        plt.imshow(image_sos,cmap=cmap,norm=norm) # type: ignore
        plt.savefig(save_path)
        plt.clf()