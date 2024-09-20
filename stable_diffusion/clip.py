from transformers import AutoTokenizer, CLIPTextModel
import torch
import torch.nn as nn
from typing import Union, List


class CLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def tokenize(self, texts: Union[List[str], str], max_length=77):
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

    def forward(self, texts: Union[List[str], str], max_length=77) -> torch.Tensor:
        tokens = self.tokenize(texts, max_length=max_length)
        with torch.no_grad():
            return self.model(**tokens).last_hidden_state