import copy
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from diffusers import DDPMScheduler

from src.models.unet import UNet_conditional, EMA
from src.utils import save_samples

class Scheduler:
    def __init__(self, **kwargs):
        self.scheduler = DDPMScheduler(**kwargs)

    def forward(self, x, step):
        noise = torch.randn_like(x)
        return self.scheduler.add_noise(x, noise, step), noise
    
    def backwards(self, x_t, predicted_noise, step):
        step = step[0]
        return self.scheduler.step(predicted_noise, step, x_t).prev_sample
    
class RandomStack:
    def __init__(self, n):
        self.n = int(n)
        self.stack = np.random.uniform(0, 1, size=self.n)
        self.idx = 0

    def get(self):
        if self.idx == self.n:
            self.stack = np.random.uniform(0, 1, self.n)
            self.idx = 0
        res = self.stack[self.idx]
        self.idx += 1
        return res

class CondDiffusionModel(pl.LightningModule):
    def __init__(self, noise_steps=1000, img_size=64, channels=3, num_classes=10):
        super().__init__()
        
        self.img_size=img_size
        self.noise_steps = noise_steps
        self.channels = channels
        self.num_classes = num_classes
        self.scheduler = Scheduler(beta_schedule='squaredcos_cap_v2')
        self.ema = EMA(0.995)
        self.random = RandomStack(1e5)
        self.prob_uncond = 0.1

        self.unet = UNet_conditional(num_classes=self.num_classes)
        self.ema_model = copy.deepcopy(self.unet)#.eval().requires_grad_(False) # mirar esto, no se bien que es
        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters()


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def get_loss(self, batch):
        x, y = batch
        y = None if self.random.get() < self.prob_uncond else y

        t = self.sample_timesteps(x.shape[0]).to(self.device)

        x_t, noise = self.scheduler.forward(x, t)
        
        predicted_noise = self.unet(x_t, t, y)

        return self.loss_fn(predicted_noise, noise)


    def sample(self, n, cfg_scale=3):
        y = torch.arange(self.num_classes).to(self.device)
        tensor_shape = (n, self.channels, self.img_size, self.img_size)

        x = torch.randn(tensor_shape).to(self.device)

        pbar = tqdm(range(1, self.noise_steps), desc=f'[INFO] Generating {n} new samples...')
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = self.ema_model(x, t, y)
            if cfg_scale > 0:
                uncond_predicted_noise = self.ema_model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

            x = self.scheduler.backwards(x, predicted_noise, t)#.clamp(-1, 1)
            pbar.update(1)

        return x
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.unet.parameters(), lr=1e-4, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]
    
    def training_step(self, batch, batch_id):
        loss = self.get_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.step_ema(self.ema_model, self.unet)
    

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        samples = self.sample(self.num_classes)
        save_samples(samples, self.logger.version)
            

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            samples = self.sample(self.num_classes)
            save_samples(samples, self.logger.version)

        loss = torch.tensor(0.0)

        return loss
    
    



    