import torch

def complex2real(data, dim=1,stack=True):
    """
    
    """
    if stack:
        return torch.stack([torch.real(data), torch.imag(data)], dim=dim).to(data.device)
    else:
        return torch.concat([torch.real(data), torch.imag(data)], dim=dim).to(data.device)


def real2complex(data, dim=1,stack=True, complex_type = torch.complex64):
    data = torch.split(data, [data.shape[dim]//2]*2, dim)
    return torch.complex(data[0].squeeze(dim), data[1].squeeze(dim)).to(data[0].device) # .to()