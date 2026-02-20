import abc
import torch
from torch.distributions import Normal
import yaml
from math import pi, log

class Distribution(abc.ABC):
    def __init__(self):
        super().__init__()
        self.potential_minimizer = None
        self.potential_min = None
        self.keep_minimizer = False
    
    def log_prob(self, x):
        log_dens = self._log_prob(x)
        if self.keep_minimizer:
            xp = x.view((-1,self.dim))
            log_dens_vals = log_dens.view((-1,1))
            argmin = torch.argmin(-log_dens_vals)
            minimum = -log_dens_vals[argmin] 
            
            if self.potential_min is None or minimum < self.potential_min:
                self.potential_min = minimum
                self.potential_minimizer = xp[argmin]  
        return log_dens
    
    def _grad_log_prob(self,x):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            torch.autograd.set_detect_anomaly(True)
            pot = self.log_prob(x)
            return torch.autograd.grad(pot.sum(),x)[0].detach()
    
    def grad_log_prob(self,x):
        return self._grad_log_prob(x)
    
    def gradient(self, x):
        return torch.exp(self.log_prob(x)) * self.grad_log_prob(x)    

class ModifiedMueller(Distribution):
    def __init__(self, device):
        super().__init__()
        self.dim = 2
        self.n = 4
        self.A = torch.tensor([-200., -100, -170, 15], device=device, dtype=torch.float32)
        self.a = torch.tensor([-1, -1, -6.5, 0.7], device=device, dtype=torch.float32)
        self.b = torch.tensor([0, 0, 11, 0.6], device=device, dtype=torch.float32)
        self.c = torch.tensor([-10, -10, -6.5, 0.7], device=device, dtype=torch.float32)
        self.XX = torch.tensor([1, 0, -0.5, -1], device=device, dtype=torch.float32)
        self.YY = torch.tensor([0, 0.5, 1.5, 1], device=device, dtype=torch.float32)
        self.x_c = -0.033923
        self.y_c = 0.465694      
        self.beta = .1
        self.translation_x = 3.5
        self.translation_y = -6.5
        self.dilatation = 1/5
        self.Z = 22340.9983
        
    def transformation(self, xx):
        x = self.dilatation * (xx[:,0] - self.translation_x)
        y = self.dilatation * (xx[:,1] - self.translation_y)
        return x,y
    
    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

        xi = x.unsqueeze(1) - self.XX
        yi = y.unsqueeze(1) - self.YY
        V_m = torch.sum(self.A * torch.exp(
            self.a * xi**2 + self.b * xi * yi + self.c * yi**2), dim=1)
        V_q = 35.0136 * (x-self.x_c)**2 + 59.8399 * (y-self.y_c)**2
        
        return -self.beta * (V_q + V_m).view(new_shape)
    
    def _grad_log_prob(self, xx):
        curr_shape = list(xx.shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

        xi = x.unsqueeze(1) - self.XX
        yi = y.unsqueeze(1) - self.YY
        ee = self.A * torch.exp(self.a * xi**2 + self.b * xi * yi + self.c * yi**2)
        grad_x = torch.sum(ee * (2 * self.a * xi + self.b * yi), dim=1)
        grad_y = torch.sum(ee * (self.b * xi + 2 * self.c * yi), dim=1)
        
        grad_x += 2 * 35.0136 * (x-self.x_c)
        grad_y += 2 * 59.8399 * (y-self.y_c)
        grad_x = grad_x.unsqueeze(-1)
        grad_y = grad_y.unsqueeze(-1)
        return -self.beta * torch.cat((grad_x,grad_y),dim=-1).view(curr_shape) * self.dilatation
       

class MultivariateGaussian(Distribution):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.Q = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(self.inv_cov)
        self.log_det = torch.log(torch.linalg.det(self.cov))
        self.dist = torch.distributions.MultivariateNormal(self.mean,self.cov)
        self.dim = mean.shape[0]
        self.Z = 1
    
    def _log_prob(self,x):
        new_shape = list(x.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        x = x.view((-1,self.dim))
        shift_cov = (self.L.T @ (x-self.mean).T).T
        log_prob = -.5 * ( self.dim * log(2 * pi) +  self.log_det + torch.sum(shift_cov**2,dim=1)) 
        log_prob = log_prob.view(new_shape)
        return log_prob

    def _grad_log_prob(self, x):
        curr_shape = x.shape
        x = x.view((-1,self.dim))
        grad = - (self.inv_cov @ (x - self.mean).T).T
        grad = grad.view(curr_shape)
        return grad
  

class MixtureDistribution(Distribution):
    def __init__(self,c,distributions):
        super().__init__()
        self.n = len(c)
        self.c = c
        self.cats = torch.distributions.Categorical(c)
        self.distributions = distributions
        self.accum = [0.]
        self.dim = self.distributions[0].dim
        for i in range(self.n):
            self.accum.append(self.accum[i] + self.c[i].detach().item())
        self.accum = self.accum[1:]
        self.Z = 1

    def _log_prob(self, x):
        log_probs = []
        for i in range(self.n):
            log_probs.append( log(self.c[i]) + self.distributions[i].log_prob(x) )
        log_probs = torch.cat(log_probs,dim=-1)
        log_dens = torch.logsumexp(log_probs,dim=-1,keepdim=True)
        return log_dens
    
    def _grad_log_prob(self, x):
        log_p = self.log_prob(x)
        grad = 0
        for i in range(self.n):
            log_pi = self.distributions[i].log_prob(x)
            grad+= self.c[i] * torch.exp(log_pi) * self.distributions[i].grad_log_prob(x)
        return grad/(torch.exp(log_p) + 1e-8)
    
    def sample(self, num_samples):
        one_sample = self.distributions[0].sample()
        samples = torch.zeros(num_samples,self.dim,
                              dtype=one_sample.dtype,
                              device=one_sample.device)
        for i in range(num_samples):
            idx = self.cats.sample()
            samples[i] = self.distributions[idx].sample()
        return samples

            
def get_distribution(name, device='cuda:0'):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.float32)

    if 'gmm' in name:
        params = yaml.safe_load(open(f'config/density_parameters/{name}.yaml'))
        weights = to_tensor_type(params['coeffs'])
        means = to_tensor_type(params['means'])
        covs = to_tensor_type(params['variances'])
        n = len(weights)
        return MixtureDistribution(
            c=weights, 
            distributions=[MultivariateGaussian(means[i], covs[i]) for i in range(n)])
    elif name == 'mueller':
        return ModifiedMueller(device=device)
    else:
        raise NotImplementedError(f"Distribution {name} not implemented")
 