import torch 
from toolbox.MRI.fft import fft2, ifft2
from toolbox.MRI.sos import sos

def cine_normalize(data_items:list):
    """
    inputs [undersampling_kdata, csm, gt]
    undersampling_kdata: Batch,Frame, Coil, Nx, Ny
    csm: Batch,Coil,Nx,Ny
    gt: Batch,Frame, Coil, Nx, Ny
    """
    bias = 1e-9
    undersampling_image = ifft2(data_items[0])
    fully_image = ifft2(data_items[2])
    csm = data_items[1]
    # 获取背景mask
    back_mask = (csm != 0).to(torch.int).to(csm.device) # Batch,Coil,Nx,Ny
    inv_back_mask = (csm == 0).to(torch.int).to(csm.device) # Batch,Coil,Nx,Ny
    # CSM归一化
    csm = inv_back_mask*bias + csm # 给值为0的地方加一个bias
    csm = csm / sos(csm,coil_dim=1,keepdims=True) # Batch,Coil,Nx,Ny 归一化CSM
    csm = csm*back_mask # 去除之前加的背景bias
    # Frame共享CSM
    csm = csm.unsqueeze(1) #B, 1, Coil,Nx,Ny
    # 合并欠采图像
    combined_undersampling_image = torch.sum(undersampling_image * csm.conj(),dim=2) #B,F,X,Y
    # 获得归一化因子 整个时间维共享一个归一化因子
    factor = torch.abs(combined_undersampling_image).max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #B 1 1 1
    # 归一化欠采图像
    combined_undersampling_image = combined_undersampling_image / factor
    # 合并全采样图像
    zero_bias = (fully_image == 0).to(torch.int).to(fully_image.device) # 避免除0
    nonzero_ = (fully_image != 0).to(torch.int).to(fully_image.device) # 去除bias影响
    fully_image_bias = fully_image + zero_bias*bias
    fully_csm = (fully_image_bias) / sos(fully_image_bias,coil_dim=2,keepdims=True)
    # 去除bias影响
    fully_csm = fully_csm*nonzero_ # B F C X Y
    # 全采样图像去背景
    fully_csm = fully_csm*back_mask.unsqueeze(1) # B F C X Y
    # 合并全采样图像
    combined_fully_image = torch.sum(fully_image * fully_csm.conj(),dim=2) #B,F,X,Y
    # 归一化全采样图像
    combined_fully_image = combined_fully_image / factor
    combined_fully_image = torch.abs(combined_fully_image)
    
    return [combined_fully_image], [combined_undersampling_image]
    
def cine_normalize_with_DC(data_items:list):
    """
    inputs [undersampling_kdata, csm, gt]
    undersampling_kdata: Batch,Frame, Coil, Nx, Ny
    csm: Batch,Coil,Nx,Ny
    gt: Batch,Frame, Coil, Nx, Ny
    """
    sampling_mask = (torch.abs(data_items[0]) != 0).to(torch.int).to(data_items[0].device) # Batch,Coil,Nx,Ny
    bias = 1e-9
    undersampling_image = ifft2(data_items[0])
    fully_image = ifft2(data_items[2])
    csm = data_items[1]
    # 获取背景mask
    back_mask = (csm != 0).to(torch.int).to(csm.device) # Batch,Coil,Nx,Ny
    inv_back_mask = (csm == 0).to(torch.int).to(csm.device) # Batch,Coil,Nx,Ny
    # CSM归一化
    csm = inv_back_mask*bias + csm # 给值为0的地方加一个bias
    csm = csm / sos(csm,coil_dim=1,keepdims=True) # Batch,Coil,Nx,Ny 归一化CSM
    csm = csm*back_mask # 去除之前加的背景bias
    # Frame共享CSM
    csm = csm.unsqueeze(1) #B, 1, Coil,Nx,Ny
    # 合并欠采图像
    combined_undersampling_image = torch.sum(undersampling_image * csm.conj(),dim=2) #B,F,X,Y
    # 获得归一化因子 整个时间维共享一个归一化因子
    factor = torch.abs(combined_undersampling_image).max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #B 1 1 1
    # 归一化欠采图像
    combined_undersampling_image = combined_undersampling_image / factor
    # 合并全采样图像
    zero_bias = (fully_image == 0).to(torch.int).to(fully_image.device) # 避免除0
    nonzero_ = (fully_image != 0).to(torch.int).to(fully_image.device) # 去除bias影响
    fully_image_bias = fully_image + zero_bias*bias
    fully_csm = (fully_image_bias) / sos(fully_image_bias,coil_dim=2,keepdims=True)
    # 去除bias影响
    fully_csm = fully_csm*nonzero_ # B F C X Y
    # 全采样图像去背景
    fully_csm = fully_csm*back_mask.unsqueeze(1) # B F C X Y
    # 合并全采样图像
    combined_fully_image = torch.sum(fully_image * fully_csm.conj(),dim=2) #B,F,X,Y
    # 归一化全采样图像
    combined_fully_image = combined_fully_image / factor
    combined_fully_image = torch.abs(combined_fully_image)
    
    undersampling_image = combined_undersampling_image.unsqueeze(2) * csm
    undersampling_kdata = fft2(undersampling_image)
    return [combined_fully_image], [combined_undersampling_image,csm,undersampling_kdata,sampling_mask]