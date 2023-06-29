import numpy as np
from scipy import signal
import torch
import sys
# sys.path.append('/data/disk1/yewei/MICCA_cine/Code/toolbox/MRI') # TODO
import utils
from utils import *
import scipy.io as scio


def crop_center(img,cropy,cropx):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_center3d(img,cropy,cropx):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx,:]

def crop_center4d(img,cropy,cropx):
    batch,z,y,x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[..., starty:starty + cropy, startx:startx + cropx]


def crop_center4d_spirit(img,cropy,cropx):
    y,x,z,batch = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx,...]


def zp(data,row,col):
    x,y = data.shape
    tx = row-x
    ty = col-y
    tx1 = int(tx / 2)
    tx2 = tx - tx1

    ty1 = int(ty / 2)
    ty2 = ty - ty1

    z = np.pad(data, ((tx1, tx2), (ty1, ty2)))
    return z

def zp3D(data,row,col):
    x,y, chanel = data.shape
    tx = row - x
    ty = col - y
    tx1 = int(tx / 2)
    tx2 = tx - tx1

    ty1 = int(ty / 2)
    ty2 = ty - ty1

    z = np.pad(data,((tx1, tx2), (ty1, ty2),(0,0)))
    return z

def zp4D(data,row,col):
    x,y, _,__ = data.shape
    tx = row - x
    ty = col - y
    tx1 = int(tx / 2)
    tx2 = tx - tx1

    ty1 = int(ty / 2)
    ty2 = ty - ty1

    z = np.pad(data,((tx1, tx2), (ty1, ty2),(0,0),(0,0)))
    return z


def getCalibSize(mask):
    sx = 2
    sy = 2

    xflag = 0
    yflag = 0

    while(1):
        if xflag == 0:
            tmp = crop_center(mask,sx+1,sy)
            if np.sum(tmp) == np.size(tmp):
                sx = sx + 1
            else:
                xflag = 1

        if yflag == 0:
            tmp = crop_center(mask,sx,sy+1)
            if np.sum(tmp) == np.size(tmp):
                sy = sy + 1
            else:
                yflag = 1

        if sx == mask.shape[0]:
            xflag = 1
        if sy == mask.shape[1]:
            yflag = 1

        if xflag == 1 and yflag == 1:
            break

    calibSize = np.array([sx,sy])
    x,y = np.meshgrid(np.linspace(-1,1,mask.shape[1]),np.linspace(-1,1,mask.shape[0]))
    r = np.sqrt(np.power(x,2)+np.power(y,2))
    circMask = r<=1
    calibMask = zp(np.ones(calibSize),mask.shape[0],mask.shape[1])
    circMask = circMask - calibMask
    R = np.sum(circMask.flatten()*mask.flatten()) / np.sum(circMask.flatten())
    densComp = 1 / R * (1-calibMask) + calibMask

    return calibSize, densComp


def esimate_maps(center_data,Nx,Ny):
    Nx_center, Ny_center, NCoil = center_data.shape
    ksp_coil = np.zeros([Nx,Ny,NCoil])
    ksp_coil = ksp_coil.astype(np.complex64)

    tukey_window = np.zeros([Nx,Ny])
    DL1 = (Nx-Nx_center)/2+1
    DH1 = (Nx - Nx_center) / 2 + Nx_center
    DL2 = (Ny - Ny_center) / 2 + 1
    DH2 = (Ny - Ny_center) / 2 + Ny_center
    kernel_size = min(Nx_center,Ny_center)
    tukey_window[int((Nx-kernel_size)/2):int((Nx-kernel_size)/2)+kernel_size, int((Ny-kernel_size)/2):int((Ny-kernel_size)/2)+kernel_size]\
        =signal.windows.tukey(kernel_size,0.5)[:, np.newaxis] @ signal.windows.tukey(kernel_size,0.5)[np.newaxis,:]

    for ncoil in range(NCoil):
        ksp_coil[:,:,ncoil] = np.zeros([Nx,Ny])
        ksp_coil[int(DL1-1):int(DH1),int(DL2-1):int(DH2),ncoil] = center_data[:,:,ncoil]
        ksp_coil[:,:, ncoil] = ksp_coil[:,:, ncoil] * tukey_window

    img_coil = ifft2c_np(ksp_coil)


    img_sos = sos_np(img_coil)
    csm = img_coil / np.repeat(img_sos[:,:,np.newaxis],NCoil,axis=2)
    return csm


def im2row(im, winSize):
    row,col,channel = im.shape
    res = np.zeros([(row-winSize[0]+1)*(col-winSize[1]+1),winSize[0]*winSize[1],channel]).astype(np.complex64)
    count = 0
    for y in range(0,winSize[1]):
        for x in range(0,winSize[0]):
            res[:,count,:] = np.reshape(im[x:row-winSize[0]+x+1,y:col-winSize[1]+y+1,:],[(row-winSize[0]+1)*(col-winSize[1]+1),channel],order='F')
            count += 1

    return res

def im2row_gpu(im,winSize):
    batch,channel,row,col = im.shape
    im = torch.permute(im,[0,1,3,2])
    unfold = torch.nn.Unfold(kernel_size=(col-winSize[0]+1,row-winSize[1]+1),dilation=1,padding=0,stride=1)
    res = unfold(im)
    res = torch.reshape(res,[batch,channel,-1,winSize[0]*winSize[1]])
    res = torch.permute(res,[2,3,1,0])
    return res


def dat2Kernel(data, ksize):
    row,col,channel = data.shape
    tmp = im2row(data,ksize)
    tsx, tsy, tsz = tmp.shape
    A = np.reshape(tmp,[tsx,-1],order='F')
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = np.conj(Vh).T
    kernel = np.reshape(V, [ksize[0], ksize[1], channel, -1], order='F')
    return kernel, S


def dat2Kernel_gpu(data,ksize):
    batch,channel,row,col = data.shape
    tmp = im2row_gpu(data,ksize)
    tsx, tsy, tsz, batch = tmp.shape
    tmp = torch.permute(tmp,[3,0,2,1])
    A = torch.reshape(tmp,[batch,tsx,tsy*tsz])
    U, S, Vh =torch.linalg.svd(A,full_matrices=False)
    V = torch.conj(Vh)
    kernel = torch.reshape(V, [batch,-1, channel, ksize[1], ksize[0]])
    kernel = torch.permute(kernel,[0,4,3,2,1])
    return kernel, S


def kernelEig(kernel, imSize):
    k1,k2,nc,nv = kernel.shape
    k = np.transpose(kernel,[0,1,3,2])
    k = np.reshape(k,[-1,nc],order='F')

    if k1 > k2:
        U,S,Vh = np.linalg.svd(k)
    else:
        U, S, Vh = np.linalg.svd(k,full_matrices=False)
    V = np.conj(Vh).T
    k = k @ V
    kernel = np.reshape(k,[k1,k2,nv,nc],order='F')
    kernel = np.transpose(kernel,[0,1,3,2])
    KERNEL = fft2c_np(zp4D(kernel[::-1, ::-1, :, :].conj() * np.sqrt(imSize[0]*imSize[1]),imSize[0],imSize[1]))
    KERNEL = KERNEL / np.sqrt(k1*k2)


    C,D,Vh = np.linalg.svd(KERNEL,full_matrices=False)
    ph = np.exp(-1j * np.angle(C[:,:,0,:]))[:,:,:,None]
    ph = np.transpose(ph,[0,1,3,2])
    C = V @ (C*ph)
    D = np.real(D)
    EigenVals = D[:,:,::-1]
    EigenVecs = C[:,:,:,::-1]
    return EigenVecs, EigenVals

def kernelEig_gpu(kernel, imsize):
    batch,k1,k2,nc,nv = kernel.shape
    k = torch.reshape()


def cal_csm(DATA, mask):
    row,col,channel,slice = DATA.shape
    CalibSize, _ = getCalibSize(mask)
    ksize = [6,6]
    eigThresh_1 = 0.02
    eigThresh_2 = 0.9  # （0.9）去掉背景的阈值，需要自己调
    csm = np.zeros(DATA.shape).astype(np.complex64)
    csm_crop_background = np.zeros(DATA.shape).astype(np.complex64)
    for i in range(slice):
        calib = crop_center3d(DATA[:,:,:,i], CalibSize[0],CalibSize[1])
        k, S = dat2Kernel(calib,ksize)
        idx = np.max(np.where(S>=S[0]*eigThresh_1))
        M, W = kernelEig(k[:,:,:, 0: idx+1], [row, col])

        #backup csm = M(:,:,:, end).*repmat(W(:,:, end) > eigThresh_2, [1, 1, Nc]) #去背景
        temp_W = W[:,:,-1] > eigThresh_2
        seg_mask = np.tile(np.expand_dims(temp_W,-1), (1,1,channel))
        csm_crop_background[:,:,:,i] = M[:,:,:,-1] * seg_mask
        csm[:,:,:,i] = M[:,:,:,-1]
    return csm, csm_crop_background, seg_mask
