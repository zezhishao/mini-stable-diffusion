import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np

@dataclass
class Constants:
    height: int = 512
    width: int = 512
    latent_height: int = height // 8
    latent_width: int = width // 8

c = Constants()

def generate(prompt:str, 
             u_prompt = None,
             input_img=None, 
             strength=0.8, 
             cfg=True, 
             cfg_weight = 7.5, 
             n_inference_step = 50, 
             models = {}, 
             seed = None, 
             device = None,
             idle_device = None, 
             tokenizer = None):
    
    with torch.no_grad():
    
        assert 0 < strength <= 1, 'Strength should be in between 0 and 1'

        if idle_device:
            to_idle_device= lambda x: x.to(idle_device)
        else:
            to_idle_device= lambda x: x
        
        generator = torch.Generator(device=device)
        if seed:
            generator.manual_seed(seed)
        else:
            generator.seed()

        bert = models['bert']
        bert.to(device)

        tokens = bert.tokenize([prompt])
        cprompt_logits = bert(tokens)
        if cfg:
            utokens = bert.tokenize([u_prompt])
            uprompt_logits = bert(utokens)
            prompt_logits = torch.cat([cprompt_logits, uprompt_logits])
        
        else:
            prompt_logits = cprompt_logits

        to_idle_device(bert)

        # sampler = DDPMSampler(generator)
        # sample.set_inference_steps(n_inference_steps)

        latent_shape = (1, 4, c.latent_height, c.latent_width)

        if input_img:
            encoder = models['encoder']
            encoder.to(device)

            input_img_tensor = input_img.resize((c.height, c.width))
            input_img_tensor = np.array(input_img_tensor)
            input_img_tensor = torch.tensor(input_img_tensor, dtype = torch.int32, device=device)

            # input_img_tensor = rescale(input_img_tensor, (0, 255), (-1, 1))

            # Batch Size = 1 if image is provided
            input_img_tensor = input_img_tensor.unsqueeze(0)

            # (1, H, W, C) -> (1, C, H, W)  
            input_img_tensor = input_img_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            latents = encoder(input_img, encoder_noise)

            # sampler.set_strength(strength=strength)
            # latents = sampler.add_noise(latents, sample.timestamps[0])
            to_idle_device(encoder)

        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        # timesteps = tqdm(sampler.timesteps)
        timesteps = [0] * 1000

        for i, step in timesteps:
            time_embedding = 0 #get_time_embedding(step).to(device)

            x = latents

            if cfg:
                x = x.repeat(2, 1, 1, 1)

            y = diffusion(x, prompt_logits, time_embedding)

            if cfg:
                y_cond, y_uncond = y.chunk(2)
                y = (y_cond - y_uncond) * cfg_weight + y_uncond

            # latents = sampler.step(step, latents, y)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle_device(decoder)

        # images = rescale(images, (-1, 1), (0, 255))

        # (B, C, H, W) -> (B, H, W, C)  
        images = images.permute(0, 2, 3, 1)
        images = images.detach().numpy()

        return images[0]


def rescale(x, old, new, clamp=False):
    old_min, old_max = old
    new_min, new_max = new
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(step):
    freqs = torch.pow(10000, torch.arange(0, 160, dtype=torch.float32) / 160) 
    x = torch.tensor([step], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
