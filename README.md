# Mini Stable Diffusion
![Stable Diffusion](https://cdm.link/app/uploads/2022/08/stablediffusion.jpg)
Welcome to Mini Stable Diffusion! This is a simple minimal implementation of one of the most (if not the most) popular architecture in the domain of image generation using text-to-image and image-to-image. This was build using `PyTorch` only and uses `CLIP` and `BERT` for text embeddings to generate image. I used BERT for research puposes and investigate how does it contribute to generate images.  
PS: Above images are not the model output

# Stable Diffusion Architecture
![Architecture](https://jalammar.github.io/images/stable-diffusion/article-Figure3-1-1536x762.png)

Here is a simple explanation of every block:

1. **Pixel Space**: These consists of 2 min blocks of a VAE Encoder and VAE Decoder. The Encoder is used to represent the input image (x) in a [latent space](https://samanemami.medium.com/a-comprehensive-guide-to-latent-space-9ae7f72bdb2f). The decoder is used to produce a tensor of Image from that latent space.

2. **Latent Space**: After obtaining the latent space, a [UNet](https://arxiv.org/pdf/1505.04597) is used as an AutoEncoder to first decrease the image size while increasing the channels and followed by a small bottleneck and then increasing image size while decreasing channels.  

3. **Conditioning**: This is probably the most important block which consists of the Scheduler, Text and Time embeddings. Scheduler is used to generate noisy image(x_t) to the input image(x) given a timestamp. Text and Time Embeddings are used in the UNet to obtain more information to generate better images.

# Sample Training Code
Here is training code for the project. You can also refer to `main.py`.
```python 
import torch
from stable_diffusion import DiffusionModel, CocoImageLoader
from stable_diffusion.utils import Config


# Optional: For faster training if Float16 available in GPU
config = Config()
if 'cuda' in config.DEVICE:
    torch.set_float32_matmul_precision('high') 

# Initialise Dataset object
imgFolder = 'test2017'
captionsFile = f'annotations/captions_{imgFolder}.json'
data_loader = CocoImageLoader(imgFolder, captionsFile)

# Initialize model
m = DiffusionModel()
m.to(config.DEVICE)
m = torch.compile(m)

# Train Model
# NOTE: 'autocast=True' for faster training using bfloat16 if available
state_dict = m.train(data_loader, return_state_dict=True, autocast=True)
```

# Why Mini Stable Diffusion?
This architecture uses lesser DownConv and UpConv layers in UNet which means this is a smaller achitecture as compared to original stable diffusion. This is done due to lack of resource to train a very deep neural network model with millions of parameters. This project is concerned with reproducing and understanding how the model works.


# Contribution
Contributions are welcome. You can fork this repository, clone it in your local and use following commands to setup with Gitbash on Windows, Linux and MacOS:
```bash
cd path/to/your/local/repository/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Original Papers:

This project is inspired from following papers:

1. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer

2. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) by Jonathan Ho, Ajay Jain, Pieter Abbeel