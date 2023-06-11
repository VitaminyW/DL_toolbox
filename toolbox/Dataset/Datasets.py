"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""
from sample_func_maker import get_csv_sample_func
from typing import List, Any
from types import FunctionType
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import numpy as np
from pathlib import PosixPath
import pandas as pd


class Basic_Dataset(Dataset):
    """
    该类为基本类,用于定义加载数据的基本行为
    """
    def __init__(self, examples:List[Any], get_sample_fuc:FunctionType):
        """
        :param examples: 包含所有样本列表
        :param get_sample_fuc: 用于加载样本的函数
        example: dataset = Basic_Dataset(examples,get_sample_fuc=get_sample_fuc)
        """
        self.examples = examples
        self.get_sample_fuc = get_sample_fuc
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.get_sample_fuc(self.examples[index])

class Dataframe_Dataset(Basic_Dataset):
    def __init__(self, csv_path:PosixPath, file_format = 'mat'):
        """
        :param csv_path: csv文件的地址
        :param file_format: csv文件中包含路径的文件格式
        example: dataset = Dataframe_Dataset(csv_path)
        """
        self.file_format = file_format
        self.dataframe = pd.read_csv(csv_path,header=None) # type: ignore
        super(Dataframe_Dataset, self).__init__(self.dataframe.values.tolist(),get_csv_sample_func(self.file_format)) # type: ignore
    
    def __getitem__(self, index):
        return self.get_sample_fuc(self.examples[index]) # mat -> tensor

"""
下面代码待调整
"""
class RandomCycleIter:
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j] # type: ignore
        else:
            yield temp_tuple[j] # type: ignore
        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=1, ):
        num_classes = len(np.unique(data_source.labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

