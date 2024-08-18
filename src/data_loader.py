import os
import torch
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from .utils import Config


config = Config()


class CocoImageLoader:
    def __init__(self, img_folder: str, captions_file: str, reshape_size: tuple = config.IMG_SIZE, batch_size: int = config.BATCH_SIZE):
        self.img_folder = img_folder
        self.coco_cap = COCO(captions_file)
        self.transforms = transforms.Compose([
            transforms.Resize(reshape_size),
            transforms.ToTensor()
        ])
        self.img_names = [name for name in os.listdir(img_folder) if name.lower().endswith(('.jpg', '.png'))]
        self.img_ids = [int(name.replace('.jpg', '').replace('.png', '')) for name in self.img_names]
        self.n_images = len(self.img_names)
        self.batch_size = batch_size

    def __len__(self):
        return self.n_images // self.batch_size

    def get_batch(self):
        indices = torch.randint(0, self.n_images, (self.batch_size,))
        img_paths = [os.path.join(self.img_folder, self.img_names[idx]) for idx in indices]
        img_ids = [self.img_ids[idx] for idx in indices]

        image_tensors = []
        captions = []
        
        for img_path, img_id in zip(img_paths, img_ids):
            img = Image.open(img_path).convert('RGB')  
            img_tensor = self.transforms(img)
            
            ann_ids = self.coco_cap.getAnnIds(imgIds=img_id)
            anns = self.coco_cap.loadAnns(ann_ids)
            if anns:
                idx = torch.randint(0, len(anns), (1,)).item()
                caption = anns[idx]['caption']
            else:
                caption = ''  
            
            image_tensors.append(img_tensor)
            captions.append(caption)

        return torch.stack(image_tensors), captions
