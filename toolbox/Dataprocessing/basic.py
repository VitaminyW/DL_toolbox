"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""

import torch
import numpy as np
from ..MRI.fft import fft2,ifft2
import torch.nn.functional as F

def crop_or_pad(data:torch.Tensor or np.ndarray, crop_size = [256,256], axes = [-2,-1]):
    """
    用于将图像统一大小
    :param data: k空间数据 任意维度(dim>=2)
    :param crop_size: 待裁剪的大小 [crop_x,crop_y]
    :param axes: 裁剪图的维度位置
    
    return 图像域进行裁剪的k空间数据
    """
    if isinstance(data,np.ndarray):
        data = torch.tensor(data)
    data = ifft2(data,axes=axes) # type: ignore
    for i in range(2):
        if axes[i] < 0:
            axes[i] = len(data.shape) + axes[i]
    pad_size = []
    crop_list = []
    for size_index, size in enumerate(data.shape):
        if size_index in axes:
            ii = axes.index(size_index)
            center = size // 2
            if size < crop_size[ii]:
                rp =  (crop_size[ii] - crop_size[ii]//2) - (size - center)
                lp = crop_size[ii]//2 - center
                pad_size.extend([lp,rp])
                crop_list.append(slice(0,crop_size[ii]))
            else:
                pad_size.extend([0,0])
                crop_list.append(slice(center - crop_size[ii]//2,center + crop_size[ii]//2))
        else:
            pad_size.extend([0,0])
            crop_list.append(slice(0,size))
    data = F.pad(data,pad_size[::-1],mode='constant')
    data = data[crop_list]
    data = fft2(data,axes=axes) # type: ignore
    return data