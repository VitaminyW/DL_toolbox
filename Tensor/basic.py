"""
数学张量运算
包含以维度k展开: tenmat
向量转hankel矩阵: vector2hankel
hankel矩阵转向量: hankel2vector
kronecker积: kronecker
多矩阵进行以模式mode展开的张量积: khatri_rao_structure
将秩一向量转为张量:get_r1_tensors
将多个矩阵合并成张量,其中各个矩阵的第i列被抽取用于组合成秩1张量:combine_tensor
"""
import torch
from typing import List

def tenmat(tensor:torch.Tensor,mode:int):
    """
    将张量按第mode维展开,eg: t.shape = [1,30,40,50], mode = 0; t0 = tenmat(t,mode) -> t0.shape = [1,30,2000]
    ps: 目前只测试三维数据 [Batch,X,Y,Z]
    :param tensor: 待展开的张量
    :param mode: 展开模式
    """
    mode += 1 # 为了支持Batch
    tensor_shape = list(tensor.shape)
    dims_list= list(range(len(tensor_shape)))
    dims_list.pop(mode)
    dims_list.insert(1,mode)
    dims_list[2:] = dims_list[2:][::-1]
    tensor = tensor.permute(*dims_list)
    tensor = tensor.reshape([tensor_shape[0], tensor_shape[mode],-1])
    return tensor

def vector2hankel(vector:torch.Tensor):
    """
    向量转hankel矩阵函数
    :param vector: 待转向量  [Batch,npt,1]
    """
    npts= vector.shape[1]
    num_hankel = (npts + 1) // 2
    list_hankel = []
    for i_hankel in range(npts + 1  - num_hankel):
        list_hankel.append(vector[:, i_hankel:i_hankel + num_hankel,:].squeeze(-1))
    hankel = torch.stack(list_hankel, dim=2).to(vector.device)
    return hankel

def hankel2vector(hankel:torch.Tensor):
    """
    hankel矩阵转向量函数
    :param hankel: hankel矩阵  [Batch,Nx,Ny]
    """
    hankel=hankel
    batch, row_num, col_num= hankel.shape
    len_vec = col_num + row_num - 1
    vec = torch.zeros((batch, len_vec), dtype=torch.complex64).to(hankel.device)
    for i in range(col_num):
        vec[:, i:i + row_num] = vec[:, i:i + row_num] + hankel[:, :, i]
    return torch.unsqueeze(vec,dim=2).to(hankel.device)

def kronecker(A:torch.Tensor, B:torch.Tensor):
    """
    Kronecker Products 张量积  A @ B
    :param A: 张量积 左矩阵  [Batch,Nx,Ny]
    :param A: 张量积 右矩阵  [Batch,Nx,Ny]
    """
    AB = torch.einsum("ab,cb->acb", A, B)
    AB = AB.reshape(A.size(0)*B.size(0), A.size(1))
    return AB

def khatri_rao_structure(mode:int,UT:List[torch.Tensor]):
    """
    多矩阵进行以模式mode展开的张量积
    :param mode: 展开模式
    :param UT: 矩阵列表
    """
    batch = UT[-1].size(0)
    c = UT[-1].size(1)
    mode += 1 # mode以0为起始，配准到以1为起始
    
    result = []
    n = len(UT)
    d = list(range(n-1,-1,-1))
    d.pop(n-mode)
    for b in range(batch):
        Z = torch.ones(1,c).to(UT[-1].device)
        for i in d:
            Z = kronecker(Z,UT[i][b].transpose(-2,-1))
        result.append(Z)
    return torch.stack(result,dim=0).to(UT[0].device)

def get_r1_tensors(ias:torch.Tensor, ibs:torch.Tensor, ics:torch.Tensor):
    """
    将秩一向量转为张量
    :param ias: a向量 [Batch, npts_a, 1]
    :param ibs: b向量 [Batch, npts_b, 1]
    :param ics: c向量 [Batch, npts_c, 1]
    """
    # ias, ibs, ics = x.split(self.dims, dim=1)
    t1 = (ias @ ibs.permute(0, 2, 1)).reshape(ias.shape[0], ias.shape[1] * ibs.shape[1], 1)
    tensors = (t1 @ ics.permute(0, 2, 1)).reshape(ias.shape[0], ias.shape[1], ibs.shape[1], ics.shape[1])
    return tensors

def combine_tensor(est_R:int,UT:List[torch.Tensor]):
    """
    将多个矩阵合并成张量,其中各个矩阵的第i列被抽取用于组合成秩1张量
    :param est_R: 秩的大小
    :param UT: 矩阵列表
    """
    result = torch.zeros(UT[0].shape[0],UT[0].shape[2],UT[1].shape[2],UT[2].shape[2]).to(torch.complex64).to(UT[0].device)
    for r in range(est_R):
        result += get_r1_tensors(UT[0][:,r,:].unsqueeze(-1),UT[1][:,r,:].unsqueeze(-1),UT[2][:,r,:].unsqueeze(-1))
    return result

if __name__ == '__main__':
    A = torch.randn(1,20)
    B = torch.randn(50,20)
    AB = kronecker(A,B)
    print(None)