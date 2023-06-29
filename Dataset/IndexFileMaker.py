"""
Author: Chen Yewei
From: Xiamen University
Github: https://github.com/VitaminyW
If you want to use this code please cite this repository.
Email: 1779723554@qq.com
"""

from pathlib import Path, PosixPath
import pandas as pd
import numpy as np
from typing import List, Any, Union, Callable


class IndexMaker:
    def __init__(self, examples:List[Any], val_size: int or float):
        """
        :param examples: 包含所有样本列表
        :param val_size: 验证集的规模
        example: marker = IndexMaker(examples,val_size=0.3)
                 val_files_list = marker['val']
                 train_files_list = marker['train']
        """
        self.examples = examples
        if isinstance(val_size,float):
            val_size = int(val_size * len(self.examples))
        val_indexs = np.random.choice(range(len(self.examples)),val_size,replace=False).tolist()
        train_indexs = [item for item in range(len(self.examples)) if item not in val_indexs]
        self.indexs = {'val':val_indexs,'train':train_indexs}
    
    def __getitem__(self, key):
        if key not in ['val','train']:
            raise ValueError('key的取值只能在val和train中')
        return np.concatenate([self.examples[index_] for index_ in self.indexs[key]]).tolist()
        

class IndexFileMaker(IndexMaker):
    def __init__(self, files:List[Path], val_size: int or float, split_example_func:Union[None,Callable]):
        """
        :param files: 包含所有样本文件路径的列表
        :param val_size: 验证集的规模
        :param split_example_func: 用于分解样本的函数
        
        example: marker = IndexFileMaker(list(Path('./data').glob('*')),val_size=0.3,split_example_func=lambda item:item)
                 val_files_list = marker['val']
                 train_files_list = marker['train']
                 dataframe_dict = marker.get_dataframe(other_parents=None)
                 val_dataframe = dataframe_dict['val']
                 train_dataframe = dataframe_dict['train']
                 marker.save_dataframe(other_parents=None,Path('.'))
        """
        if split_example_func is None:
            split_example_func = lambda item:item # type: ignore
        examples = {}
        for file in files:
            example_name = split_example_func(file) # type: ignore
            if example_name not in examples:
                examples[example_name] = [file]
            else:
                examples[example_name].append(file)
        self.examples = list(examples.items()) # List[(example_name,List[PosixPath])]
        if isinstance(val_size,float):
            val_size = int(val_size * len(self.examples))
        val_indexs = np.random.choice(range(len(self.examples)),val_size,replace=False).tolist()
        train_indexs = [item for item in range(len(self.examples)) if item not in val_indexs]
        self.indexs = {'val':val_indexs,'train':train_indexs}
    
    def __getitem__(self, key):
        if key not in ['val','train']:
            raise ValueError('key的取值只能在val和train中')
        if len(self.indexs[key]) == 0:
            return []
        return np.concatenate([self.examples[index_][1] for index_ in self.indexs[key]]).tolist()
    
    def get_dataframe(self, other_parents:None or List[PosixPath]):
        """
        :param other_parents:使用同一个文件名在不同路径下作为不同的输入，如['csm/A.mat','xu/A.mat','gt/A.mat']
        """
        if other_parents is None:
            result = {}
            for key in self.indexs:
                if len(self.indexs[key]) == 0:
                    result[key] = []
                else:
                    result[key] = pd.DataFrame([[item] for item in self[key]])
            return result
        else:
            df_dict = {}
            for key in self.indexs:
                filenames = [item.name for item in self[key]]
                temp_dfl = []
                for filename in filenames:
                    temp_dfl.append([parent / filename for parent in other_parents])
                df_dict[key] = pd.DataFrame(temp_dfl)
            return df_dict
    
    def save_dataframe(self,other_parents:None or List[PosixPath],save_path:PosixPath):
        df_dict = self.get_dataframe(other_parents)
        for key in df_dict:
            df_dict[key].to_csv(save_path / (key+'.csv'),header=None,index=False) # type: ignore
            
            
if __name__ == '__main__':
    """
    完成测试    
    """
    test_index_file_marker = IndexFileMaker(list(Path('/data/disk1/yewei/fastmri_multi_brain/50spoke/pFISTA_Net/train/nc_kdata').glob('*')), # type: ignore
                                            0.3,None) # type: ignore
    print(test_index_file_marker.get_dataframe([Path('csm'),Path('gt'),Path('xu')])['train']) # type: ignore
    print(len(test_index_file_marker.indexs['val']))
    