"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""
import torch
import numpy as np

def _sos_np(x,coil_dim=-1,keepdims=True):
    dim = len(x.shape)
    if dim <= 2:
        return np.abs(x)
    else:
        return np.sum(np.abs(x ** 2), axis=coil_dim, keepdims=keepdims) ** 0.5


def _sos_torch(x, coil_dim=-1, keepdims=True):
    """
    :param x: [B,2*C,H,W]
    :return: rss [B, 1, H, W]
     (sum(abs(x.^2),dim)).^(1/2)
     abs((a+bj)**2) = a**2 + b**2
     so complex value can divide to real num and imag num , then calculate square of them, finally do sum operate
    """
    dim = len(x.shape)
    if dim <= 2:
        return torch.abs(x)
    else:
        return torch.sum(torch.abs(x ** 2), axis=coil_dim, keepdims=keepdims) ** 0.5 # type: ignore

def sos(x, coil_dim=-1, keepdims=True):
    """
    :param x: 任意维度(大于2)的图像数据,eg: [slice,frame,coil,Kx,Ky]
    :param coil_dim: 通道维度
    """
    if  isinstance(x,np.ndarray):
        return _sos_np(x,coil_dim,keepdims)
    elif isinstance(x,torch.Tensor):
        return _sos_torch(x,coil_dim,keepdims)
    else:
        ValueError('Except x as type np.ndarray or torch.Tensor')