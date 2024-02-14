import torch

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
    def __init__(self, steps, beta_start, beta_end):
        '''self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps'''

        self.beta = torch.linspace(beta_start, beta_end, steps)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    