import torch
import ot

class RBF_Kernel():

    def __init__(self, n_kernels=10, mul_factor=2.0):
        self.n_kernels = n_kernels
        self.bandwidth_multipliers = mul_factor ** (torch.arange(-2, -2 + n_kernels))
        
    def get_kernel_value(self, x, y):
        distances = torch.cdist(x, y) ** 2
        bandwidths = self.bandwidth_multipliers[:, None, None].to(x.device)
        kernel_vals = torch.exp(-0.5 * distances[None, ...] / bandwidths)
        return kernel_vals.mean(dim=(1, 2)).sum()



class MMDLoss():
    def __init__(self, kernel=RBF_Kernel()):
        self.kernel = kernel

    def get_mmd(self, x, y):
        xx = self.kernel.get_kernel_value(x,x)
        xy = self.kernel.get_kernel_value(x,y)
        yy = self.kernel.get_kernel_value(y,y)
        return (xx - 2 * xy + yy) ** .5


def get_mmd(samples1, samples2, device='cuda:0'):
    if not isinstance(samples1, torch.Tensor):
        samples1 = torch.tensor(samples1, dtype=torch.float32)
    if not isinstance(samples2, torch.Tensor):
        samples2 = torch.tensor(samples2, dtype=torch.float32)
    samples1 = samples1.to(device); samples2 = samples2.to(device)
    return MMDLoss().get_mmd(samples1, samples2)



def get_w2(samples1, samples2, device='cuda:0'):
    if not isinstance(samples1, torch.Tensor):
        samples1 = torch.tensor(samples1, dtype=torch.float32)
    if not isinstance(samples2, torch.Tensor):
        samples2 = torch.tensor(samples2, dtype=torch.float32)
    samples1 = samples1.to(device); samples2 = samples2.to(device)
    n, m = samples1.shape[0], samples2.shape[0]
    M = ot.dist(samples1, samples2)
    a, b = torch.ones((n,), device=M.device) / n, torch.ones((m,), device=M.device) / m
    return ot.emd2(a, b, M)**.5