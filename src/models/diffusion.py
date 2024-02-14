import torch
import pytorch_lightning as pl
from tqdm import tqdm

from src.models.schedulers import LinearScheduler
from src.models.unet import UNet
from src.utils import save_samples

class DiffusionModel(pl.LightningModule):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, channels=3):
        super().__init__()
        
        self.img_size=img_size
        self.noise_steps = noise_steps
        self.channels = channels

        self.unet = UNet()
        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters()
        self.scheduler = LinearScheduler(noise_steps, beta_start, beta_end)


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, n):
        tensor_shape = (n, self.channels, self.img_size, self.img_size)

        x = torch.randn(tensor_shape).to(self.device)

        pbar = tqdm(range(1, self.noise_steps), desc=f'[INFO] Generating {n} new samples...')
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = self.unet(x, t)

            x = self.scheduler.backwards(x, predicted_noise, t)
            pbar.update(1)

        return x
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=1e-4, weight_decay=0.1)
    
    def training_step(self, batch, batch_id):
        x, _ = batch

        t = self.sample_timesteps(x.shape[0]).to(self.device)

        x_t, noise = self.scheduler.forward(x, t)

        predicted_noise = self.unet(x_t, t)

        loss = self.loss_fn(predicted_noise, noise)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def on_validation_epoch_end(self):
        samples = self.sample(16)
        save_samples(samples, self.logger.version)

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        t = self.sample_timesteps(x.shape[0]).to(self.device)
        
        x_t, noise = self.scheduler.forward(x, t)

        predicted_noise = self.unet(x_t, t)

        loss = self.loss_fn(predicted_noise, noise)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
            

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            x, _ = batch

            samples = self.sample(x)
            
            torch.save(samples.cpu(), './samples/samples_final.pt')

        loss = torch.tensor(0.0)

        return loss
    
    



    