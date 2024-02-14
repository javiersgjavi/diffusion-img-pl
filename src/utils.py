import os
import torch
import torchvision
from PIL import Image

def save_samples(tensors, v_num):
    path_tensors = f'./samples/exp_{v_num}/tensors'
    path_images = f'./samples/exp_{v_num}/images'
    os.makedirs(path_tensors, exist_ok=True)
    os.makedirs(path_images, exist_ok=True)

    content_len = len(os.listdir(path_tensors))

    
    torch.save(tensors.cpu(), f'{path_tensors}/tensor_{content_len}.pt')

    tensors = (tensors.clamp(-1, 1) + 1) / 2
    tensors = (tensors * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(tensors, nrow=4)
    ndarr = grid.permute(1,2,0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(f'{path_images}/image_{content_len}.png')


