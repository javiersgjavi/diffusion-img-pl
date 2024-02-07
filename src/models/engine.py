import os
import torch
from tqdm import tqdm
from tsl.engines.predictor import Predictor
from tsl.metrics import torch as torch_metrics

from src.models.diffusion_model import DiffusionModel
from src.models.unet import UNet

from src.models.noise_schedulers import LinearNoiseScheduler

class NoiseApplicator:
    def __init__(self, noise_steps=1000):
        self.noise_steps = noise_steps
        self.scheduler = LinearNoiseScheduler(self.noise_steps)

    def apply_noise(self, x, t):
        # Ya me cuadra la fórmula, pero no me cuadra como se coge los alphas
        α_hat = self.scheduler.get_alpha_hat(t, like=x).to(x.device)
        α = self.scheduler.get_alpha(t, like=x).to(x.device)
        ε = torch.randn_like(x)

        x_t = torch.sqrt(α_hat) * x + torch.sqrt(1-α) * ε

        return x_t, ε

class PositionalEncoder:
    def __init__(self, emb_size, device, nodes, t_steps, batch_size):
        self.emb_size = emb_size
        self.device = device
        self.nodes = nodes
        self.t_steps = t_steps
        self.batch_size = batch_size

    def encode(self, t):
        batch_size = t.shape[0]
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t = t.expand(-1, self.t_steps, self.nodes, -1)

        t_emb = torch.zeros(batch_size, self.t_steps, self.nodes, self.time_emb_size)

        div_term = 1 / (10000 ** (torch.arange(0, self.time_emb_size, 2).float() / self.time_emb_size))

        t_emb[:, :, :, 0::2] = torch.sin(t * div_term)
        t_emb[:, :, :, 1::2] = torch.cos(t * div_term)

        return t_emb.to(self.devices)

class TimeStepSampler:
    def __init__(self, noise_steps=1000, size=1e5):
        self.noise_steps = noise_steps
        self.size = int(size)
        self.timesteps = torch.randint(low=1, high=self.noise_steps, size=(self.size,))
        self.i = 0

    def __check_t(self, n):
        if self.i + n >= len(self.timesteps):
            self.timesteps = torch.randint(low=0, high=self.noise_steps, size=(self.size,))
            self.i = 0

    def sample(self, like):
        self.__check_t(like.shape[0])
        value = self.timesteps[self.i:self.i+like.shape[0]]
        self.i += like.shape[0]
        return value
    
class DiffusionEngine(Predictor):
    def __init__(self, scaler=None, batch_size=None, *args, **kwargs):
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            'rmse': torch_metrics.MaskedRMSE(),
            'mse': torch_metrics.MaskedMSE(),
            #'mre': torch_metrics.MaskedMRE(),
        }
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.t_sampler = TimeStepSampler(noise_steps=1000)
        self.loss_fn = torch.nn.MSELoss()
        #self.model = DiffusionModel(device=self.device, time_emb_size=4)
        self.model = UNet(c_in=6, c_out=6)
        self.scaler = scaler
        #self.pos_encoder = PositionalEncoder()
        self.noise_app = NoiseApplicator()

    def training_step(self, batch, batch_idx): 
        x = batch.get('x')

        t = self.t_sampler.sample(like=x)#.to(self.device)
        #t_emb = self.pos_encoder.encode(t)

        x_t, ε = self.noise_app.apply_noise(x, t)

        # Compute predictions and compute loss
        ε_pred = self.model(x_t, t.to(self.device))

        loss = self.loss_fn(ε, ε_pred)

        self.log_loss('train', loss, batch_size=batch.batch_size)
        self.log('mse', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch.get('x')
        
        t = self.t_sampler.sample(like=x)#.to(self.device)

        x_t, ε = self.noise_app.apply_noise(x, t)

        # Compute predictions and compute loss
        ε_pred = self.model(x_t, t.to(self.device))

        # Compute loss
        loss = self.loss_fn(ε, ε_pred)

        # Logging
        self.val_metrics.update(ε_pred, ε, torch.ones_like(ε))
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)
        return loss
    
    def sample(self, shape):
        x_t = torch.randn(shape).to(self.device)
        for i in tqdm(range(self.noise_app.noise_steps-1, -1, -1)):
            t = torch.ones(shape[0], dtype=int) * i
            α = self.noise_app.scheduler.get_alpha(t, like=x_t).to(x_t.device)
            α_hat = self.noise_app.scheduler.get_alpha_hat(t, like=x_t).to(x_t.device)
            β = self.noise_app.scheduler.get_beta(t, like=x_t).to(x_t.device)

            z = torch.randn_like(x_t) if i != 0 else torch.zeros_like(x_t)

            ε_pred = self.model(x_t, t.to(self.device))

            # revisar esta fórmula
            x_t = 1/torch.sqrt(α) * (x_t - ((1 - α)/(torch.sqrt(1 - α_hat))) * ε_pred) + torch.sqrt(β) * z

        return x_t

    
    def test_step(self, batch, batch_idx):

        if batch_idx > 1:
            return torch.zeros(1)
        
        shape = batch.get('x').shape
        new_samples = self.sample(shape)

        # Compute loss
        loss = torch.zeros(1)

        os.makedirs('samples', exist_ok=True)
        torch.save(new_samples, f'samples/sample_{batch_idx}.pt')

        # Logging
        #print(torch.cat([x_imputated_denorm[eval_mask].int(), y[eval_mask].int()], dim=-1))
        #self.test_metrics.update(torch.ones(10), torch.ones(10), torch.ones(10))
        #self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', loss, batch_size=batch.batch_size)
        return loss
    
