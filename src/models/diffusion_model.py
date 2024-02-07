import torch
import torch.nn as nn
from tsl.nn.models.base_model import BaseModel
from src.models.noise_schedulers import LinearNoiseScheduler
from src.models.bidirectional_models import BiModel


class DiffusionModel(BaseModel):
    def __init__(self, noise_steps=1000, device=None, time_emb_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.noise_steps = noise_steps
        self.device = device
        self.time_emb_size = time_emb_size
        self.noise_scheduler = LinearNoiseScheduler(noise_steps, device=self.device)
        self.model = BiModel(time_emb_size=self.time_emb_size)

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)


    def apply_noise(self, x, t, mask):
        α_hat = self.noise_scheduler.get_alpha_hat(t, like=x).to(x.device) 
        ε = torch.randn_like(x)
        ε = torch.where(~mask.bool(), ε, torch.zeros_like(ε))

        x_t = torch.sqrt(α_hat) * x + torch.sqrt(1-α_hat) * ε
        x_t = torch.where(~mask.bool(), x_t, torch.zeros_like(x_t))

        return x_t, ε
    
    def predict_noise(self, batch, t, t_emb):
        x = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask
        
        x_t, ε = self.apply_noise(x, t, mask)
        x_co = torch.where(mask.bool(), x, torch.zeros_like(x))

        ε_pred = self.model.forward(x_t, x_co, mask, edge_index, edge_weight, t_emb, u)
        ε_pred = torch.where(~mask.bool(), ε_pred, torch.zeros_like(ε_pred))

        
        return ε, ε_pred
    
    def sample(self, batch):
        x_co = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask

        x_t = torch.randn_like(x_co)
        x_t = torch.where(~mask.bool(), x_t, torch.zeros_like(x_t))

        x_co = torch.where(mask.bool(), x_co, torch.zeros_like(x_co))

        for t in reversed(range(1, self.noise_steps)):


            t_tensor = torch.ones(x_t.shape[0], dtype=torch.int64) * t

            α_t = self.noise_scheduler.get_alpha(t_tensor, like=x_t).to(x_t.device)
            α_hat_t = self.noise_scheduler.get_alpha_hat(t_tensor, like=x_t).to(x_t.device)
            β = self.noise_scheduler.get_beta(t_tensor, like=x_t).to(x_t.device)

            
            t_emb = self.encode_t(t_tensor, like=x_t)

            
            ε_pred = self.model.forward(x_t, x_co, mask, edge_index, edge_weight, t_emb, u)


            if t > 1:
                ε = torch.randn_like(x_t)
            else:
                ε = torch.zeros_like(x_t)

            x_t = 1/torch.sqrt(α_t) * (x_t - ((1 - α_t)/(torch.sqrt(1 - α_hat_t))) * ε_pred) + torch.sqrt(β)*ε

        return x_t

