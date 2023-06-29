from typing import Callable,Union
import torch
from utils import Loss_Function
from torch.nn.functional import (mse_loss,l1_loss,cross_entropy,nll_loss,
                                 poisson_nll_loss,kl_div,binary_cross_entropy,
                                 binary_cross_entropy_with_logits,
                                 margin_ranking_loss,
                                 hinge_embedding_loss,
                                 multilabel_margin_loss,
                                 multilabel_soft_margin_loss,
                                 cosine_embedding_loss,
                                 multi_margin_loss,
                                 triplet_margin_with_distance_loss,
                                 triplet_margin_loss,
                                 ctc_loss)
"""以下兼容了大多数torch自带的损失函数
1.MSE_Loss
2.L1_Loss
3.Cross_Entropy_Loss
4.NLL_Loss
5.Poisson_Nll_loss
6.KL_Loss
7.Binary_Cross_Entropy_Loss
8.Binary_Cross_Entropy_With_Logits_Loss
9.Margin_Ranking_Loss
10.Hinge_Embedding_Loss
11.Multilabel_Margin_Loss
12.Multilabel_Soft_Margin_Loss
13.Cosine_Embedding_loss
14.Multi_Margin_loss

PS: 通过Multi_Margin_loss.params可以得到对应类需要输入的参数名称以及参数类型
"""

class MSE_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(mse_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class L1_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(l1_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Cross_Entropy_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(cross_entropy)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class NLL_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(nll_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Poisson_Nll_loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(poisson_nll_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class KL_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(kl_div)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Binary_Cross_Entropy_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(binary_cross_entropy)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Binary_Cross_Entropy_With_Logits_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(binary_cross_entropy_with_logits)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Margin_Ranking_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(margin_ranking_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input1 = preds['input1']
        input2 = preds['input2']
        target = gts['target']
        return self._func(input1=input1,input2=input2,target=target,**other_param)

class Hinge_Embedding_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(hinge_embedding_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Multilabel_Margin_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(multilabel_margin_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Multilabel_Soft_Margin_Loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(multilabel_soft_margin_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)

class Cosine_Embedding_loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(cosine_embedding_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input1 = preds['input1']
        input2 = preds['input2']
        target = gts['target']
        return self._func(input1=input1,input2=input2,target=target,**other_param)

class Multi_Margin_loss(Loss_Function):
    def __init__(self) -> None:
        super().__init__(multi_margin_loss)
    
    def __call__(self, preds: dict, gts: dict,other_param:Union[None,dict]) -> torch.Tensor:
        input_ = preds['input']
        target = gts['target']
        return self._func(input=input_,target=target,**other_param)
    
    