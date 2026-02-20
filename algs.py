import numpy as np
import scipy as sp
import torch
import abc
from utils.densities import Distribution
from utils import optimizers
from samplers.rejection_sampler import get_samples as get_rgo_samples
from samplers.ula import get_ula_samples


class ScoreEstimator(abc.ABC):
    def __init__(self, dist: Distribution, device, num_samples=1024):
        self.dist = dist
        self.device = device
        self.num_samples = num_samples
        self.dim = self.dist.dim
        self.count = 0

    def __call__(self, x, tt, *args, **kwargs):
        return self.score_estimator(x, tt, *args, **kwargs)

    @abc.abstractmethod
    def score_estimator(self, x, tt, *args, **kwargs):
        pass


class ZODMC(ScoreEstimator):
    def __init__(self, dist: Distribution, device, num_samples=1024, max_iters_opt=50):
        super().__init__(dist, device, num_samples)
        self.dist.keep_minimizer = True
        minimizer = optimizers.newton_conjugate_gradient(
            torch.randn(self.dist.dim, device=device),
            lambda x: -self.dist.log_prob(x), max_iters_opt)
        self.dist.log_prob(minimizer)
    
    def score_estimator(self, x, tt, *args, **kwargs):
        samples_from_p0t, acc_idx = get_rgo_samples(
            x * torch.exp(tt), torch.expm1(2 * tt), self.dist, self.num_samples)
        self.count += self.num_samples
        num_good_samples = torch.sum(acc_idx, dim=(1, 2)).unsqueeze(-1).to(torch.double) / self.dim
        mean_estimate = torch.sum(samples_from_p0t * acc_idx,dim=1)
        num_good_samples[num_good_samples == 0] += 1
        mean_estimate /= num_good_samples
        return (x - torch.exp(-tt) * mean_estimate) / torch.expm1(-2 * tt)


class SNDMC(ScoreEstimator):
    def __init__(self, dist: Distribution, device, num_samples=1024):
        super().__init__(dist, device, num_samples)

    def score_estimator(self, x, tt, *args, **kwargs):
        big_x = x.repeat_interleave(self.num_samples, dim=0)
        noise = (-torch.expm1(-2 * tt)) ** .5 * torch.randn_like(big_x)
        log_probs = self.dist.log_prob(torch.exp(tt) * (big_x - noise)).view(-1, self.num_samples, 1)
        self.count += self.num_samples
        log_probs[log_probs < -1e10] = -1e10
        weights = torch.softmax(log_probs, dim=1)
        noise = noise.view(-1, self.num_samples, self.dim)
        return 1 / torch.expm1(-2 * tt) * torch.sum(noise * weights, dim=1)


class RDMC(ScoreEstimator):
    def __init__(self, dist: Distribution, device, num_samples=1024,
                 ula_step_size=0.01, ula_steps=10, init='importance'):
        super().__init__(dist, device, num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.init = init
        self.num_importance_samples = 64
        
    def score_estimator(self, x, tt, ula_steps=None, *args, **kwargs):
        ula_steps = ula_steps if ula_steps is not None else self.ula_steps
        big_x = x.repeat_interleave(self.num_samples, dim=0)
        def score_prob_0t(x0):
            return self.dist.grad_log_prob(x0) - (x0 - torch.exp(tt) * big_x) / torch.expm1(2 * tt)
        
        if self.init == 'normal':
            x0 = torch.exp(tt) * big_x + torch.expm1(2 * tt) **.5 * torch.randn_like(big_x)
        elif self.init == 'importance':
            x0 = self.posterior_importance_sampling(tt, big_x)
        else: 
            x0 = big_x

        samples_from_p0t = get_ula_samples(x0, score_prob_0t, self.ula_step_size, 
            ula_steps).view(-1, self.num_samples, self.dim) 
        samples_from_p0t.clamp_(min=-100, max=100) 
        self.count += self.num_samples * ula_steps
        mean_estimate = torch.mean(samples_from_p0t, dim = 1) 
        return (x - torch.exp(-tt) * mean_estimate) / torch.expm1(-2 * tt)
    
    def posterior_importance_sampling(self, tt, x):
        z = (torch.exp(tt) * x).unsqueeze(0) + torch.expm1(2 * tt) ** .5 * torch.randn(
            (self.num_importance_samples, *x.shape), device=x.device) 
        log_weight = self.dist.log_prob(z)
        log_weight[log_weight < -1e10] = -1e10
        self.count += self.num_importance_samples
        weights = torch.nn.functional.softmax(
            log_weight.view(self.num_importance_samples, -1), dim=0)
        idx = torch.multinomial(weights.T, 1).T.squeeze() 
        return z[idx, torch.arange(x.shape[0], device=x.device), :]


class RSDMC(ScoreEstimator):
    def __init__(self, dist: Distribution, device, num_samples=1024,
                 ula_step_size=0.01, ula_steps=10, num_recursive_steps=2, T=5,
                 init='importance'):
        super().__init__(dist, device, num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.num_recursive_steps = num_recursive_steps 
        self.T = T
        self.S = torch.tensor(T / num_recursive_steps, device=device)
        self.init = init
        self.num_importance_samples = 64

    def _recursive_langevin(self, x, k, tt):
        if k == -1: return self.dist.grad_log_prob(x)
        big_x = x.repeat_interleave(self.num_samples, dim=0).detach().clone()

        if self.init == 'normal':
            x0 = torch.exp(k * self.S + tt) * big_x + torch.expm1(2 * (k * self.S + tt)) **.5 * torch.randn_like(big_x)
        elif self.init == 'importance':
            x0 = self.posterior_importance_sampling(k * self.S + tt, big_x)
        else: 
            x0 = big_x
        x_kS = torch.exp(-k * self.S) * x0 + (-torch.expm1(-2 * k * self.S)) ** .5 * torch.randn_like(x0)
        x_kS.clamp_(min=-20, max=20) 

        def curr_score(x_kS):
            return self._recursive_langevin(x_kS, k-1, self.S) + (torch.exp(tt) * big_x - x_kS) / torch.expm1(2 * tt)
        x_kS = get_ula_samples(x_kS, curr_score, self.ula_step_size, self.ula_steps)
        x_kS.clamp_(min=-20, max=20) 
        self.count += self.num_samples * self.ula_steps
        mean_estimate = x_kS.view(-1, self.num_samples, self.dim).mean(dim=1)
        return (x - torch.exp(-tt) * mean_estimate) / torch.expm1(-2 * tt)
    
    def score_estimator(self, x, tt, *args, **kwargs):
        if tt > self.T - 0.1:
            return x / torch.expm1(-2 * tt)
        k = int(tt // self.S.item()); tt_ = tt % self.S
        if tt_ < 0.01: k -= 1; tt_ = self.S 
        return self._recursive_langevin(x, k, tt_)

    def posterior_importance_sampling(self, tt, x):
        z = (torch.exp(tt) * x).unsqueeze(0) + torch.expm1(2 * tt) ** .5 * torch.randn(
            (self.num_importance_samples, *x.shape), device=x.device) 
        z.clamp_(min=-20, max=20) 
        log_weight = self.dist.log_prob(z)
        log_weight[log_weight < -1e10] = -1e10
        self.count += self.num_importance_samples
        weights = torch.nn.functional.softmax(
            log_weight.view(self.num_importance_samples, -1), dim=0)
        idx = torch.multinomial(weights.T, 1).T.squeeze() 
        return z[idx, torch.arange(x.shape[0], device=x.device), :]


def get_score_function(method, dist, device, num_estimator_samples=1024, 
                        init='importance', rsdmc_num_recursive_steps=2, T=5,
                        ula_steps=100, ula_step_size=0.01):
    if method == 'zodmc':
        return ZODMC(dist, device, num_samples=num_estimator_samples)
    elif method == 'rdmc':
        return RDMC(dist, device, num_samples=num_estimator_samples,
                                   ula_step_size=ula_step_size, ula_steps=ula_steps,
                                   init=init)
    elif method == 'rsdmc':
        return RSDMC(dist, device, num_samples=num_estimator_samples,
                                    ula_step_size=ula_step_size, ula_steps=ula_steps,
                                    num_recursive_steps=rsdmc_num_recursive_steps, 
                                    T=T, init=init)
    elif method == 'sndmc':
        return SNDMC(dist, device, num_samples=num_estimator_samples)
    else:
        raise ValueError(f"Unknown score method: {method}")


def rds(method, dist:Distribution, batch_size=1024, num_steps=100,
           T=5, delta=5e-3, device='cuda:0', 
           ula_step_size=0.01, ula_steps=100, *args, **kwargs):
    model = get_score_function(method, dist, device=device,
                               ula_step_size=ula_step_size, ula_steps=ula_steps, T=T,
                               *args, **kwargs)
    time_pts = torch.linspace(0, T - delta, num_steps, device=device)
    init_sigma_sq = 1 - np.exp(-2 * T)
    x = init_sigma_sq ** .5 * torch.randn(batch_size, dist.dim, dtype=torch.float32, device=device)
    w = - (x ** 2).sum(dim=1) / (2 * init_sigma_sq) - dist.dim / 2 * np.log(2 * np.pi * init_sigma_sq)

    for i in range(num_steps - 1):
        n01_x = torch.randn_like(x)
        t = time_pts[i]; dt = time_pts[i+1] - time_pts[i]
        score = model(x, T - t)
        x = torch.exp(dt) * x + 2 * torch.expm1(dt) * score + torch.expm1(2 * dt) ** .5 * n01_x
        rho = 2 ** .5 * torch.expm1(dt) / (torch.expm1(2 * dt) * dt) ** .5
        n01_w = rho * n01_x + (1 - rho ** 2) ** .5 * torch.randn_like(n01_x)
        w += dt * (score ** 2).sum(dim=1) + (2 * dt) ** .5 * (score * n01_w).sum(dim=1)
    w += -dist.log_prob(x).squeeze() - (T - delta) * dist.dim
    return x, torch.exp(-w), model.count


def ais(dist:Distribution, lamda0=100, batch_size=1024, steps=2**16,
		step_size=0.01):
	x = np.zeros((batch_size, dist.dim)) / lamda0 ** .5
	log_z0 = dist.log_prob(x).squeeze() + dist.dim / 2 * np.log(2 * np.pi / lamda0)
	w = np.zeros((batch_size, ))
	lambdas = lamda0 * np.linspace(1, 0, steps) ** 2
	for i in range(steps - 1):
		x += step_size * np.nan_to_num(dist.grad_log_prob(x) - lambdas[i+1] * x) + (2 * step_size) ** .5 * np.random.randn(*x.shape)
		w -= .5 * (lambdas[i] - lambdas[i+1]) * (x ** 2).sum(axis=1)
	return x, np.exp(log_z0 - w)



def ti(dist, beta=50, eps=1, batch_size=1024, num_samples=32, 
       ula_step_size=0.01, ula_steps=50):
    def lmc(x, grad_log_prob, h, num_iters, disable_pbar=True):
        for _ in range(num_iters):
            x += h * np.nan_to_num(grad_log_prob(x)) + (2 * h) ** .5 * np.random.randn(*x.shape)
        return x

    lamdas = [dist.dim * beta / eps]
    while lamdas[-1] > 1 / 2 / dist.dim ** 0.5:
        lamdas.append(lamdas[-1] / (1 + 1 / dist.dim ** 0.5) * 1.45)
    lamdas.append(0)

    log_Z = np.full((batch_size,), 
                    dist.log_prob(np.zeros((1, dist.dim))).item() + dist.dim / 2 * np.log(2 * np.pi / lamdas[0]))
    x = np.zeros((batch_size, num_samples, dist.dim))
    for k in range(len(lamdas) - 1):
        x = lmc(x, lambda x: dist.grad_log_prob(x) - lamdas[k] * x, 
                ula_step_size, ula_steps)
        x_norm_sq = np.sum(x**2, axis=-1)
        log_Z += sp.special.logsumexp((lamdas[k] - lamdas[k+1]) / 2 * x_norm_sq, 
                                      axis=1) - np.log(num_samples)
    return x[:, 0, :], np.exp(log_Z)