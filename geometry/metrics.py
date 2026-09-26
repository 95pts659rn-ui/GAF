import torch
import torch.nn.functional as F

def backreaction(original, recovered, mode='vector'):
    difference = original - recovered
    if mode == 'vector':
        return difference
    elif mode == 'norm':
        return torch.norm(difference, dim=-1)
    elif mode == 'energy':
        return torch.sum(difference*difference, dim=-1)
    elif mode == 'divergence':
        batch_size = original.shape[0]
        if batch_size < 2:
            return torch.zeros(batch_size, device=original.device, dtype=original.dtype)
        order = torch.argsort(torch.norm(original, dim=-1))
        x = original[order]
        f = difference[order]
        _x = torch.roll(x, shifts=1, dims=0)
        x_ = torch.roll(x, shifts=-1, dims=0)
        _f = torch.roll(f, shifts=1, dims=0)
        f_ = torch.roll(f, shifts=-1, dims=0)
        j = torch.zeros(batch_size, 3, 3, device=original.device, dtype=original.dtype)
        for axis in range(3):
            d = x_[:, axis:axis + 1] - _x[:, axis:axis + 1]
            d = torch.where(torch.abs(d) < 1e-8, torch.ones_like(d) * 1e-8, d)
            j[:, :, axis] = (f_ - _f) / d
        order_ = torch.empty_like(order)
        order_[order] = torch.arange(batch_size, device=order.device)
        divergence = torch.diagonal(j, dim1=1, dim2=2).sum(dim=-1)
        return divergence[order_]
    else:
        raise ValueError(f"Unknown mode specified: {mode}")
