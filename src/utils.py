from dataclasses import dataclass
import torch

@dataclass
class Config:
    VAE_INIT_CHANNELS:int = 128     # initial channels size for VAE
    UNET_INIT_CHANNELS:int = 320    # intiial channels size for UNet
    TIME_DIM:int = 1280             # time emebddings dimension
    TIME_POS_DIM:int = 320          # postional time embedding dimension
    PROMPT_DIM:int = 768            # prompt embeddings dimension (i.e. BERT embedding dimensions)
    IMG_SIZE:tuple = (256, 256)
    BATCH_SIZE:int = 32
    DROPOUT:float = 0.2
    LATENT_H:int = 64               # latent height
    LATENT_W:int = 64               # latent weight

# function to convert [0, 255] to [-1, 1] and vice versa
def rescale(x, old, new, clamp=False, to_image=False):
    old_min, old_max = old
    new_min, new_max = new
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    if to_image:
        x = x.to(torch.uint8)
    return x


def get_time_embedding(step):
    freqs = torch.pow(10000, torch.arange(0, 160, dtype=torch.float32) / 160)
    x = torch.tensor([step], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
