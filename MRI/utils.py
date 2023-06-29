import numpy as np
import torch
import torch.nn.functional as F


# def fft2c(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = 1 / np.sqrt(fctr) * torch.fft.fft2(x, dim=(0, 1))
#     return result
#
#
# def ifft2c(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = np.sqrt(fctr) * torch.fft.ifft2(x, dim=(0, 1))
#     return result
#
#
# def fft2c_np(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = 1 / np.sqrt(fctr) * np.fft.fft2(x, axes=(0, 1))
#     return result
#
#
# def ifft2c_np(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = np.sqrt(fctr) * np.fft.ifft2(x, axes=(0, 1))
#     return result


# def ifft2c_np_shift(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x,axes=(0,1)),axes=(0,1)),axes=(0,1))
#     return result

def sos(x):
    if x.ndim <= 2:
        return torch.abs(x)
    else:
        return torch.sqrt(torch.sum(torch.abs(torch.pow(x, 2)), dim = 2))

def sos_np(x):
    if x.ndim <= 2:
        return np.abs(x)
    else:
        return np.sqrt(np.sum(np.abs(np.power(x, 2)), axis = 2))

def ifftc(x,n):
    fctr = x.shape[n]
    result = np.sqrt(fctr) * torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x,dim=n), dim=n),dim=n)
    return result

def ifftc_np(x,n):
    fctr = x.shape[n]
    result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x,axes=n), axis=n),axes=n)
    return result

def fftc_np(x,n):
    fctr = x.shape[n]
    result = 1 / np.sqrt(fctr) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x,axes=n),axis=n),axes=n)
    return result

# normal
def fft2c(x):
    fctr = x.shape[0] * x.shape[1]
    result = 1 / np.sqrt(fctr) * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x),dim=(0,1)))
    return result

def fft2c_np(x):
    fctr = x.shape[0] * x.shape[1]
    result = 1 / np.sqrt(fctr) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), axes=(0, 1)))
    return result

def ifft2c(x):
    fctr = x.shape[0] * x.shape[1]
    result = np.sqrt(fctr) * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), dim=(0,1)))
    return result

def ifft2c_np(x):
    fctr = x.shape[0] * x.shape[1]
    result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x,axes=(0,1)),axes=(0,1)),axes=(0,1))
    return result

# for T1WI
# def fft2c(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = 1 / np.sqrt(fctr) * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x,dim=1),dim=(0,1)),dim=(0,1))
#     return result
#
# def fft2c_np(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = 1 / np.sqrt(fctr) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x,axes=1), axes=(0,1)),axes=(0,1))
#     return result
#
# def ifft2c(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = np.sqrt(fctr) * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x,dim=(0,1)), dim=(0,1)),dim=1)
#     return result
#
# def ifft2c_np(x):
#     fctr = x.shape[0] * x.shape[1]
#     result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x,axes=(0,1)),axes=(0,1)),axes=1)
#     return result


def fft3c(x):
    fctr = x.shape[2]
    x = fft2c_np(x)
    result = 1/np.sqrt(fctr) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x,axes=2),axis=2),axes=2)
    return result

def ifft3c_np(x):
    fctr = x.shape[2]
    x = ifft2c_np(x)
    result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x,axes=2),axis=2),axes=2)
    return result

def FU_sense(mask, c, b, type):
    if type == 1:
        k_coil = mask * b
        im_coil = ifft2c(k_coil)
        return torch.sum(torch.conj(c)*im_coil,dim = 2)
    else:
        im_coil = c * torch.unsqueeze(b, 2)
        k_coil = fft2c(im_coil)
        return mask * k_coil

def rlne(im_ori, im_rec):
    imError = im_ori - im_rec
    return torch.norm(imError) / torch.norm(im_ori)

def soft_thresholding(data,value):
    mag = torch.abs(data)
    mag = 1 - (value/mag)
    mag = torch.clip(mag,max=None,min=0)
    return mag * data

def soft_thresholding_v2(data,value):
    tmp = torch.abs(data) - value
    tmp = (tmp+torch.abs(tmp)) / 2
    return torch.sgn(data) * tmp


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic' or mode == 'periodization':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def spirit_core(kernel,x, type):
    if type==1:
        res = kernel.conj() * np.reshape(x,[kernel.shape[0],kernel.shape[1],1,kernel.shape[2],kernel.shape[4]])
        res = np.sum(res,axis=2)
        res = res - x
    else:
        res = kernel * np.reshape(x, [kernel.shape[0], kernel.shape[1], kernel.shape[2], 1, kernel.shape[4]])
        res = np.sum(res, axis=2)
        res = res - x
    return res


def FU_spirit(mask, b, kernel, lamba_1, type):
    if type == 1:
        row, _, _, slice = b.shape
        tmp_a = ifft2c_np(mask[:,:,np.newaxis,np.newaxis] * b[:int(row/2), :, :, :])
        tmp_b = -np.sqrt(lamba_1) * spirit_core(kernel,b[int(row/2):,:,:,:],1)
        return tmp_a + tmp_b
    else:
        tmp_a = mask[:,:,np.newaxis,np.newaxis] * fft2c_np(b)
        tmp_b = -np.sqrt(lamba_1) * spirit_core(kernel,b,2)
        return np.concatenate([tmp_a,tmp_b],axis=0)


def spirit_core_torch(kernel,x, type):
    if type==1:
        res = kernel.conj() * torch.reshape(x,[kernel.shape[0],kernel.shape[1],1,kernel.shape[2],kernel.shape[4]])
        res = torch.sum(res,dim=3)
        res = res - x
    else:
        res = kernel * torch.reshape(x, [kernel.shape[0], kernel.shape[1], kernel.shape[2], 1, kernel.shape[4]])
        res = torch.sum(res, dim=2)
        res = res - x
    return res


def FU_spirit_torch(mask, b, kernel, lamba_1, type):
    if type == 1:
        row, _, _, slice = b.shape
        tmp_a = ifft2c(mask[:,:,None,None] * b[:int(row/2), :, :, :])
        tmp_b = -np.sqrt(lamba_1) * spirit_core_torch(kernel,b[int(row/2):,:,:,:],1)
        return tmp_a + tmp_b
    else:

        tmp_a = mask[:,:,None,None] * fft2c(b)
        tmp_b = -np.sqrt(lamba_1) * spirit_core_torch(kernel,b,2)

        # import scipy.io as scio
        # _a = scio.loadmat('./ab.mat')['tmp_a']
        # _b = scio.loadmat('./ab.mat')['tmp_b']
        # test1 = np.sum(torch.squeeze(tmp_a).cpu().numpy() - _a)
        # test2 = np.sum(torch.squeeze(tmp_b).cpu().numpy() - _b)


        return torch.cat([tmp_a,tmp_b],dim=0)