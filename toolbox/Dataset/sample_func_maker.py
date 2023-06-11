"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""
import torch

def read_mat(path:str, complex_type = torch.complex64):
    """
    :param path: mat文件的加载地址
    """
    import scipy.io as sio
    
    mat = sio.loadmat(path)
    data = mat[list(mat.keys())[-1]]
    data = torch.tensor(data).to(complex_type)
    return data

load_methods = {'mat':read_mat}

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
        