from typing import Any, Union,List,Callable
import torch.nn.functional as F
import torch

class Loss_Function:
    """损失函数的基本使用
    """
    def __init__(self,func:Callable) -> None:
        self._func = func
        self.params = self._func.__annotations__
        self.__doc__ = self._func.__doc__
        
    def __call__(self, preds:dict, gts:dict, other_param:Union[None,dict]) -> torch.Tensor: 
        raise NotImplementedError("在子类中实现调用函数")

"""自定义损失函数模板
class My_Loss_Function:
    def __init__(self,func:Callable) -> None:
        def my_loss(input1:Tensor,input2:Tensor,target:Tensor,weight:Tensor,**avgv):
            return torch.sum((input1+input2)/target * weight)
        super().__init__(my_loss)
    
    def __call__(self, preds:dict, gts:dict, other_param:Union[None,dict]) -> torch.Tensor: 
        input1 = preds['input1']
        input2 = preds['input2']
        target = gts['target']
        weight = other_param['other_param']
        return self._func(input1=input1,input2=input2,target=target,weight=weight)
"""