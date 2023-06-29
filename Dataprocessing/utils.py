import torch
import numpy as np
from ..MRI.fft import fft2,ifft2
import torch.nn.functional as F
from typing import List,Union

def crop_or_pad2(data:Union[torch.Tensor,np.ndarray], crop_size:List[int] = [256,256],
                axes:List[int]= [-2,-1], do_ifft:bool = True):
    """用于将图像统一大小,仅支持在两个维度裁剪

    Args:
        data (Union[torch.Tensor,np.ndarray]): k空间数据或图像数据 任意维度(dim>=2)
        crop_size (List[int], optional): 待裁剪的大小. Defaults to [256,256].
        axes (List[int], optional): 裁剪图的维度位置. Defaults to [-2,-1].
        do_ifft (bool, optional): 是否对输入数据进行逆傅里叶变换. Defaults to True.

    Returns:
        _type_: 图像域进行裁剪的k空间数据
    """
    if isinstance(data,np.ndarray):
        data = torch.tensor(data)
    if do_ifft:
        data = ifft2(data,axes=axes)
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
    data = F.pad(data,pad_size[::-1],mode='constant') # type: ignore
    data = data[crop_list]
    if do_ifft:
        data = fft2(data,axes=axes) # type: ignore
    return data