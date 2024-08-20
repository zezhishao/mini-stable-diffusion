import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import Union, List

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
        self.m = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            out = self.m(**x)
            return out.last_hidden_state

    def tokenize(self, texts:Union[List[str], str], max_length:int = 64) -> torch.Tensor:
        return self.tokenizer(texts, max_length = max_length, padding='max_length', truncation=True, return_tensors='pt')
    