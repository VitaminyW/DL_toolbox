"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""
import torch
import numpy as np

def _ifft2_n(kdata,axes = (-2,-1)):
    k_shift = np.fft.ifftshift(kdata, axes =axes)  # 进行K空间移位，使其符合FFT变换规则
    image = np.fft.ifft2(k_shift, axes = axes)  # IFFT变换获得图像
    image_shift = np.fft.fftshift(image, axes = axes)  # 再移位回来
    return image_shift


def _fft2_n(image, axes = (-2,-1)):
    image_shift = np.fft.ifftshift(image, axes = axes)
    kdata = np.fft.fft2(image_shift, axes = axes)
    k_data = np.fft.fftshift(kdata, axes = axes)
    return k_data


def _fft2_t(image, axes = (-2,-1)):
    image_shift = torch.fft.ifftshift(image, dim = axes)
    kdata = torch.fft.fft2(image_shift, norm='ortho', dim = axes)
    kdata = torch.fft.fftshift(kdata, dim = axes)
    return kdata


def _ifft2_t(kdata, axes = (-2,-1)):
    k_shift = torch.fft.ifftshift(kdata, dim=axes)
    image = torch.fft.ifft2(k_shift, norm='ortho', dim=axes)
    image_shift = torch.fft.fftshift(image, dim=axes)
    return image_shift

def fft2(image, axes=(-2,-1)):
    """
    :param image: 任意维度的图像数据,eg: [slice,frame,coil,x,y]
    :param axes: 需要做二维傅里叶变换的维度
    """
    if  isinstance(image,np.ndarray):
        return _fft2_n(image,axes)
    elif isinstance(image,torch.Tensor):
        return _fft2_t(image,axes)
    else:
        raise ValueError('Except image as type np.ndarray or torch.Tensor')
    

def ifft2(kdata, axes=(-2,-1)):
    """
    :param kdata: 任意维度的二维频域数据,eg: [slice,frame,coil,Kx,Ky]
    :param axes: 需要做二维逆傅里叶变换的维度
    """
    if  isinstance(kdata,np.ndarray):
        return _ifft2_n(kdata,axes)
    elif isinstance(kdata,torch.Tensor):
        return _ifft2_t(kdata,axes)
    else:
        raise ValueError('Except kdata as type np.ndarray or torch.Tensor')

