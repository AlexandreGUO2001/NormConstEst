import numpy as np
import scipy as sp
import abc
import yaml


class Distribution(abc.ABC):
    def __init__(self):
        super().__init__()
        self.dim = None
        self.potential_minimizer = None
        self.potential_min = None
        self.keep_minimizer = False

    def log_prob(self, x):
        log_dens = self._log_prob(x)
        if self.keep_minimizer:
            x_ = x.reshape(-1, self.dim)
            log_dens_ = log_dens.reshape(-1, 1)
            argmin = np.argmin(-log_dens_)
            minimum = -log_dens_[argmin] 
            if self.potential_min is None or minimum < self.potential_min:
                self.potential_min = minimum
                self.potential_minimizer = x_[argmin]  
        return log_dens
    
    @abc.abstractmethod
    def _log_prob(self, x):
        raise NotImplementedError
    
    @abc.abstractmethod
    def grad_log_prob(self, x):
        raise NotImplementedError

    def grad_prob(self, x):
        return np.exp(self.log_prob(x)) * self.grad_log_prob(x)    


class MultivariateGaussian(Distribution):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.cov_inv = np.linalg.inv(self.cov)
        self.dim = self.mean.shape[0]
        self.log_norm_const = -.5 * self.dim * np.log(2 * np.pi) - .5 * np.log(np.linalg.det(self.cov))

    def _log_prob(self, x):
        shape = x.shape
        x_ = x.reshape(-1, self.dim)
        diff = x_ - self.mean
        exponent = -0.5 * np.sum(diff @ self.cov_inv * diff, axis=1, keepdims=True)
        log_prob = self.log_norm_const + exponent
        return log_prob.reshape(shape[:-1] + (1,))

    def grad_log_prob(self, x):
        shape = x.shape
        x_ = x.reshape(-1, self.dim)
        diff = x_ - self.mean
        grad = -diff @ self.cov_inv
        return grad.reshape(shape[:-1] + (self.dim,))
    
    def sample(self, n_samples):
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)



class MixtureDistribution(Distribution):
    def __init__(self, weights, distributions):
        super().__init__()
        self.n = len(weights)
        self.weights = np.asarray(weights)
        self.weights /= np.sum(self.weights)
        self.distributions = distributions
        self.dim = self.distributions[0].dim
        self.Z = 1

    def _log_prob(self, x):
        log_probs = np.stack([
            np.log(self.weights[i]) + self.distributions[i].log_prob(x) 
            for i in range(self.n)
            ], axis=-1)
        return sp.special.logsumexp(log_probs, axis=-1)

    def grad_log_prob(self, x):
        log_probs = np.stack(
            [np.log(self.weights[i]) + self.distributions[i].log_prob(x) 
            for i in range(self.n)
            ], axis=-1)
        weights = sp.special.softmax(log_probs, axis=-1)
        grad_log_probs = np.stack(
            [self.distributions[i].grad_log_prob(x) for i in range(self.n)],
            axis=-1
        )
        return np.sum(weights * grad_log_probs, axis=-1)

    def sample(self, num_samples):
        idxs = np.random.choice(self.n, size=num_samples, p=self.weights)
        samples = np.concatenate(
            [self.distributions[i].sample(np.sum(idxs == i))
             for i in range(self.n)], axis=0)
        np.random.shuffle(samples)
        return samples


class ModifiedMueller(Distribution):
    def __init__(self):
        super().__init__()
        self.dim = 2
        self.n = 4
        self.A = np.array([-200., -100, -170, 15])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.XX = np.array([1, 0, -0.5, -1])
        self.YY = np.array([0, 0.5, 1.5, 1])
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
        return x, y
    
    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.reshape(-1, self.dim)
        x, y = self.transformation(xx)

        xi = x[:, None] - self.XX[None, :]
        yi = y[:, None] - self.YY[None, :]
        exp_terms = self.A[None, :] * np.exp(
            self.a[None, :] * xi**2 + self.b[None, :] * xi * yi + self.c[None, :] * yi**2
        )
        V_m = np.sum(exp_terms, axis=1)
        V_q = 35.0136 * (x - self.x_c)**2 + 59.8399 * (y - self.y_c)**2

        return -self.beta * (V_q + V_m).reshape(new_shape)
    
    def grad_log_prob(self, xx):
        curr_shape = list(xx.shape)
        xx = xx.reshape(-1, self.dim)
        x, y = self.transformation(xx)

        xi = x[:, None] - self.XX[None, :]
        yi = y[:, None] - self.YY[None, :]
        exp_terms = self.A[None, :] * np.exp(
            self.a[None, :] * xi**2 + self.b[None, :] * xi * yi + self.c[None, :] * yi**2
        )

        grad_x = np.sum(exp_terms * (2 * self.a[None, :] * xi + self.b[None, :] * yi), axis=1)
        grad_y = np.sum(exp_terms * (self.b[None, :] * xi + 2 * self.c[None, :] * yi), axis=1)

        grad_x += 2 * 35.0136 * (x - self.x_c)
        grad_y += 2 * 59.8399 * (y - self.y_c)
        grad = np.stack((grad_x, grad_y), axis=-1)
        grad = grad.reshape(curr_shape)
        return -self.beta * grad * self.dilatation


def get_distribution(name):
    if 'gmm' in name:
        params = yaml.safe_load(open(f'config/density_parameters/{name}.yaml'))
        weights = params['coeffs']
        means = params['means']
        covs = params['variances']
        n = len(weights)
        return MixtureDistribution(
            weights=weights,
            distributions=[MultivariateGaussian(mean=means[i], cov=covs[i])
                for i in range(n)
            ]
        )
    elif name == 'mueller':
        return ModifiedMueller()
    else:
        raise NotImplementedError(f"Distribution {name} not implemented")