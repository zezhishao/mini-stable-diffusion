import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .unet import UNet
from .vae import VAE_Encoder, VAE_Decoder
from .bert import BERT
from .utils import Config, rescale
import numpy as np
from typing import Union


config = Config()


class DiffusionModel(nn.Module):
    def __init__(self, n_embd:int=config.TIME_POS_DIM) -> None:
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.SiLU(),
            nn.Linear(n_embd * 4, n_embd * 4),
            nn.SiLU()
        )
        self.bert = BERT()
        self.unet = UNet()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        # self.scheduler = Scheduler()

    def forward(self, x, noise, prompt, time):
        time = self.time_emb(time)


        x = self.encoder(x, noise)
        x = self.unet(x, prompt, time)
        out = self.decoder(x)

        return out

    def generate(self,
                 prompt:str,
                 u_prompt:Union[str, None] = None,
                 input_img:Union[Image.Image, None]=None,
                 strength:float=0.8,
                 cfg:bool=True,
                 cfg_weight:float=7.5,
                 n_inference_step:int=50,
                 seed = None,
                ):

        with torch.no_grad():

            assert 0 < strength <= 1, 'Strength should be in between 0 and 1'

            generator = torch.Generator(device=self.device)
            if seed:
                generator.manual_seed(seed)
            else:
                generator.seed()

            tokens = self.bert.tokenize([prompt])
            cprompt_logits = self.bert(tokens)
            if cfg:
                utokens = self.bert.tokenize([u_prompt])
                uprompt_logits = self.bert(utokens)
                prompt_logits = torch.cat([cprompt_logits, uprompt_logits])

            else:
                prompt_logits = cprompt_logits



            latent_shape = (1, 4, config.LATENT_H, config.LATENT_H)

            if input_img:
                input_img_tensor = input_img.resize((config.LATENT_H, config.LATENT_W))
                input_img_tensor = np.array(input_img_tensor)
                input_img_tensor = torch.tensor(input_img_tensor, dtype = torch.int32, device=device)

                input_img_tensor = rescale(input_img_tensor, (0, 255), (-1, 1))

                # Batch Size = 1 if image is provided
                input_img_tensor = input_img_tensor.unsqueeze(0)

                # (1, H, W, C) -> (1, C, H, W)
                input_img_tensor = input_img_tensor.permute(0, 3, 1, 2)

                encoder_noise = torch.randn(latent_shape, generator=generator, device=self.device)
                latents = self.encoder(input_img, encoder_noise)

                # self.sampler.set_strength(strength=strength)
                # latents = sampler.add_noise(latents, sample.timestamps[0])

            else:
                latents = torch.randn(latent_shape, generator=generator, device=self.device)

            # self.schedular.set_inference_steps(n_inference_steps)
            # timesteps = tqdm(sampler.timesteps)
            timesteps = torch.arange(1000, device=self.device)

            for step in timesteps:
                time_embedding = 0 #get_time_embedding(step).to(device)

                x = latents

                if cfg:
                    x = x.repeat(2, 1, 1, 1)
                y = self(x, prompt_logits, time_embedding)

                if cfg:
                    y_cond, y_uncond = y.chunk(2)
                    y = (y_cond - y_uncond) * cfg_weight + y_uncond

                # latents = sampler.step(step, latents, y)

            images = self.decoder(latents)

            images = rescale(images, (-1, 1), (0, 255))

            # (B, C, H, W) -> (B, H, W, C)
            images = images.permute(0, 2, 3, 1)
            images = images.detach().numpy()

            return images[0]
