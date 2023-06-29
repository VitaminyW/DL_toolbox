import torch
from typing import Union
from pathlib import Path

def read_mat(path:str, complex_type = torch.complex64):
    """
    :param path: mat文件的加载地址
    """
    import scipy.io as sio
    
    mat = sio.loadmat(path)
    data = mat[list(mat.keys())[-1]]
    data = torch.tensor(data).to(complex_type)
    return data

def read_image(path:Union[str,Path],color_mode :int = 2):
    """用于读取png格式的图片

    Args:
        path (Union[str,Path]): 图片文件地址
        color_mode (int, optional): cv2的读取模式 {1:读取一副彩色图片，图片的透明度会被忽略,
                                                   2:以灰度模式读取一张图片,
                                                   3:加载一副彩色图像，透明度不会被忽略} 
                                                   Defaults to 2.
    """
    from cv2 import imdecode
    import numpy as np
    import torchvision.transforms as transforms
    img = imdecode(np.fromfile(path, dtype=np.uint8), color_mode)
    return transforms.ToTensor()(img) # torch(C,H,W)
    
load_methods = {'mat':read_mat,'png':read_image,'jpg':read_image}

def get_csv_sample_func(file_format):
    """
    获取csv文件中一行中记录的文件地址的数据
    :param file_format: 文件格式
    :param postprocessing_func: 后处理函数
    """
    if file_format not in load_methods:
        raise ValueError(f'{file_format}文件类型的读取方式未定义')
    load_method = load_methods[file_format]
    def sample_func(example):
        data_example = []
        for item in example:
            data_example.append(load_method(item))
        return data_example
    return sample_func
        