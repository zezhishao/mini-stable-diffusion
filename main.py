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