import torch

def batch_entropy(D):
    assert len(D.shape) == 2
    dlogd = D * torch.log(D)
    dlogd[D == 0] = 0
    return -dlogd.sum(dim=1)
