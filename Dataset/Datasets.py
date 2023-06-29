from typing import Union
import torch

from torch.utils.benchmark import examples
from utils import get_csv_sample_func,load_methods
from typing import Callable, List, Any
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import pandas as pd
from pathlib import Path


class BasicDataset(Dataset):
    """该类为基本类,用于定义加载数据的基本行为
    """
    def __init__(self, examples:List[Any], get_sample_fuc:Callable):
        """

        Args:
            examples (List[Any]): 包含所有样本列表
            get_sample_fuc (Callable): 用于加载样本的函数
        Example: 
            dataset = Basic_Dataset(examples,get_sample_fuc=get_sample_fuc)
        """
        self.examples = examples
        self.get_sample_fuc = get_sample_fuc
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.get_sample_fuc(self.examples[index])
"""
    定义各种任务类型所需要的Dataset
    目前支持：
        a. 以csv文件保存数据路径的数据加载 CsvDataset
        b. 以文件夹为类别，保存不同类别图片的数据加载 FolderBaseImageClassfierDataset
"""
class CsvDataset(BasicDataset):
    """用于加载以csv文件的一行作为一个example的任务
    csv文件如:
        noise_image1.mat, clear_image1.mat
    """
    def __init__(self, csv_path:Union[Path,str], file_format:str = 'mat'):
        """
        Args:
            csv_path (Path,str): csv文件的地址
            file_format (str, optional): csv文件中包含路径的文件格式. Defaults to 'mat'.
        Example:
            dataset = Dataframe_Dataset(csv_path)
        """
        self.file_format = file_format
        self.dataframe = pd.read_csv(csv_path,header=None) # type: ignore
        super(Dataframe_Dataset, self).__init__(self.dataframe.values.tolist(),get_csv_sample_func(self.file_format)) # type: ignore

class FolderBaseImageClassfierDataset(BasicDataset):
    """用于加载以不同文件夹保存不同类别的图片形式的数据集
    文件夹的统领格式为:
    /home/classfier/cat [1.png,2.pnd,3.png]
    /home/classfier/dog [1.png,2.pnd,3.png]
    """
    def __init__(self, root_path:Union[Path,str], file_format = 'png'):
        """
        Args:
            root_path (Path,str): 类别文件夹的根目录
            file_format (str, optional): 图片的文件类型. Defaults to 'png'.
        Example:
            dataset = Dataframe_Dataset(csv_path)
        """
        self.file_format = file_format
        if not isinstance(root_path, Path):
            root_path = Path(root_path)
        self.root_path = root_path
        class_folders = [item for item in self.root_path.glob('*') if item.is_dir()] # 遍历所有的类别文件夹
        # 为了后续训练，将标签编码
        # 形如 {'dog':0,'cat':1}
        classes_mapping = {item.stem:i for i, item in enumerate(class_folders)}
        # 提取example
        self.examples = []
        for folder in class_folders:
            for image_path in folder.glob(f'*.{file_format}'):
                self.examples.append([image_path,classes_mapping[folder.stem]]) # [path,num_label]
        def get_image_classfier_sample(sample):
            return load_methods[self.file_format](sample[0]), torch.tensor(sample[1])
        super(FolderBaseImageClassfierDataset, self).__init__(self.examples, get_image_classfier_sample)

"""
下面代码待调整,用于不均匀样本分类任务
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

