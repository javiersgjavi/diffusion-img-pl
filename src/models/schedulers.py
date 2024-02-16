import torch
import numpy as np

class Scheduler:

    def _get_property(self, data, step, reshaped):
        res = data[step.to(data.device)]
        res = res.reshape(-1, 1, 1, 1) if reshaped else res
        return res.to(step.device)

    def get_beta(self, step, reshaped=True):
        return self._get_property(self.beta, step, reshaped)
    
    def get_alpha(self, step, reshaped=True):
        return self._get_property(self.alpha, step, reshaped)
    
    def get_alpha_hat(self, step, reshaped=True):
        return self._get_property(self.alpha_hat, step, reshaped)
    
    def forward(self, x, step):
        alpha_hat_t = self.get_alpha_hat(step)

        sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
        sqrt_alpha_minus_one = torch.sqrt(1-alpha_hat_t)

        noise = torch.randn_like(x)

        x_t = sqrt_alpha_hat*x + sqrt_alpha_minus_one*noise

        return x_t, noise
    
    def backwards(self, x_t, predicted_noise, step):
        alpha = self.get_alpha(step)
        alpha_hat = self.get_alpha_hat(step)
        beta = self.get_beta(step)

        if step[0].item() > 1:
            added_noise = torch.randn_like(x_t)
        else:
            added_noise = torch.zeros_like(x_t)

        first_term = 1/torch.sqrt(alpha)
        second_term = x_t - (beta/torch.sqrt(1-alpha_hat)) * predicted_noise
        mean = first_term*second_term

        variance = torch.sqrt(beta)*added_noise

        res = mean + variance

        return res


class LinearScheduler(Scheduler):
    def __init__(self, steps, beta_start=1e-4, beta_end=0.02):
        '''self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps'''

        self.beta = torch.linspace(beta_start, beta_end, steps)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

class CosineScheduler(Scheduler):
    def __init__(self, s=0.008, max_beta=0.999, num_steps=1000):
        self.max_beta = max_beta
        self.s = s

        self.beta = self.create_beta(num_steps).float()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).float()

    def create_beta(self, num_steps):
        betas = np.zeros(num_steps)
        for i in range(num_steps):
            t1 = i / num_steps
            t2 = (i+1) / num_steps
            res = 1 - (self.alpha_bar_fn(t2) / self.alpha_bar_fn(t1))
            betas[i] = res

        betas = torch.from_numpy(betas)
        return torch.clamp(betas, max=self.max_beta)

    def alpha_bar_fn(self, t):
        return np.cos((t + self.s) / (1+self.s) * np.pi / 2) ** 2
